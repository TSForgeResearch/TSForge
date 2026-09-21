import torch
import torch.nn as nn


class Ensembler(nn.Module):
    """Ensemble the overlapping forecasts that target the same date.

    Each target date is forecast once per forecast creation date (FCD) whose horizon
    reaches it. Combining those repeated forecasts reduces the variance of the estimate
    and so reduces forecast volatility.

    The combination is causal: the forecast issued at FCD t pools only its own
    prediction and those from earlier FCDs, never a later one. For a shared target date
    a larger horizon step holds an older forecast, so the causal pool for horizon step h
    is h..H-1, which is implemented by reversing the horizon axis around the kernels.

    Expects predictions laid out as [B, C, T, H, Q], the layout produced at the output
    layer, and returns the same shape.
    """

    def __init__(self, config):
        super().__init__()
        self.stride = config.stride
        methods = {
            'mean':     self._cumulative_mean,
            'median':   self._cumulative_median,
            'ewm':      self._cumulative_ewm,
            'identity': self._identity,
        }
        if config.ensemble_method not in methods:
            raise ValueError(f"ensemble_method must be one of {list(methods.keys())}")
        self.ensemble_method = config.ensemble_method
        self.ensembler = methods[config.ensemble_method]
        self.ensemble_window_size = getattr(config, 'ensemble_window_size', config.h)
        self.alpha = getattr(config, 'alpha', 0.9)

    def forward(self, preds, mask=None):
        """preds: [B, C, T, H, Q] -> [B, C, T, H, Q]."""
        if preds.dim() != 5:
            raise NotImplementedError(
                f"Ensembler expects [B, C, T, H, Q]; got a {preds.dim()}-D tensor. "
                "Ensembling at ensemble_level='embedding' is not supported: the "
                "encoder output is [B*C, patch_num, d] and the forking-sequences "
                "unfold into (T, H) windows happens later in the decoder, so there is "
                "no target-date structure to ensemble over at that point."
            )
        # the ensembling logic is written for [B, T, H, C, Q]
        out = self.ensemble(preds.permute(0, 2, 3, 1, 4), mask=mask)
        return out.permute(0, 3, 1, 2, 4)

    def ensemble(self, preds, mask=None):
        """
        preds: (B, T, H, C, Q) torch.Tensor
        mask:  (B, T, H, C)    torch.Tensor, int/bool — 1=valid, 0=mask out
        returns: (B, T, H, C, Q)
        """
        B, T, H, C, Q = preds.shape
        stride = self.stride

        if self.ensemble_method == 'identity':
            return preds

        if stride == H:
            raise ValueError(
                f"stride={stride} equals H={H}: windows are non-overlapping so each "
                "target date has exactly one forecast. Ensembling has no effect — "
                "use ensemble_method='identity' instead."
            )
        elif stride > H:
            raise ValueError(
                f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
                "Please review the selected stride parameter used in your experiment."
            )

        # fold Q into C → (B, T, H, C*Q) so all internal logic is unchanged
        preds_flat = preds.reshape(B, T, H, C * Q)
        mask_flat = (
            mask.repeat_interleave(Q, dim=-1) if mask is not None else None
        )

        by_date = self._reshape_windows_by_date(preds_flat, mask=mask_flat)  # (B,S,H,C*Q)
        B_, S_, H_, CQ_ = by_date.shape
        flat = by_date.reshape(B_ * S_, H_, CQ_)

        # larger h = older forecast, so reversing makes "everything up to here" mean
        # "this FCD and the ones before it", which is the causal pool
        flat = torch.flip(flat, dims=[1])
        ensembled = self.ensembler(flat)
        ensembled = torch.flip(ensembled, dims=[1])

        ensembled = ensembled.reshape(B_, S_, H_, CQ_)
        out = self._gather_windows_from_dates(ensembled, n_windows=T)
        return out.reshape(B, T, H, C, Q)

    def _reshape_windows_by_date(self, x, mask=None):
        """(B, T, H, C) -> (B, (T-1)*stride + H, H, C), grouped by target date."""
        B, T, H, C = x.shape
        S = (T - 1) * self.stride + H

        if mask is not None:
            x = torch.where(mask == 0, torch.full_like(x, float('nan')), x)

        t_grid, h_grid = torch.meshgrid(
            torch.arange(T, device=x.device), torch.arange(H, device=x.device),
            indexing='ij',
        )
        d_grid = t_grid * self.stride + h_grid
        out = torch.full((B, S, H, C), float('nan'), dtype=x.dtype, device=x.device)
        out[:, d_grid, h_grid, :] = x
        return out

    def _gather_windows_from_dates(self, x, n_windows):
        """(B, S, H, C) -> (B, T, H, C), the inverse of the scatter above."""
        H = x.shape[2]
        t_grid, h_grid = torch.meshgrid(
            torch.arange(n_windows, device=x.device),
            torch.arange(H, device=x.device),
            indexing='ij',
        )
        d_grid = t_grid * self.stride + h_grid
        return x[:, d_grid, h_grid, :]

    # --- ensembling strategies (torch versions) ---

    def _cumulative_mean(self, preds):
        valid = ~torch.isnan(preds)
        W = self.ensemble_window_size
        if W is None or W >= preds.shape[1]:
            total = torch.nan_to_num(preds).cumsum(dim=1)
            counts = valid.to(preds.dtype).cumsum(dim=1)
        else:
            total = torch.stack([
                torch.nan_to_num(preds[:, max(0, i - W + 1):i + 1, :]).sum(dim=1)
                for i in range(preds.shape[1])
            ], dim=1)
            counts = torch.stack([
                valid[:, max(0, i - W + 1):i + 1, :].to(preds.dtype).sum(dim=1)
                for i in range(preds.shape[1])
            ], dim=1)
        ensemble = total / counts
        ensemble[counts == 0] = float('nan')
        return ensemble

    def _cumulative_median(self, preds):
        H = preds.shape[1]
        W = self.ensemble_window_size
        start = (lambda i: max(0, i - W + 1)) if W else (lambda i: 0)
        # nanquantile(0.5) averages the two middle values for an even-sized pool, which
        # is what np.nanmedian does; torch.nanmedian would return the lower one instead.
        return torch.stack(
            [torch.nanquantile(preds[:, start(i):i + 1, :], 0.5, dim=1) for i in range(H)],
            dim=1,
        )

    def _cumulative_ewm(self, preds):
        _, H, _ = preds.shape
        alpha = self.alpha
        dtype, device = preds.dtype, preds.device
        beta = torch.log(torch.tensor(alpha / (1 - alpha), dtype=dtype, device=device))
        W = self.ensemble_window_size if self.ensemble_window_size is not None else H

        h_idx = torch.arange(H, device=device)
        i_idx = torch.arange(H, device=device)
        starts = torch.clamp(h_idx - W + 1, min=0)
        in_window = (i_idx[None, :] >= starts[:, None]) & (i_idx[None, :] <= h_idx[:, None])

        weight_matrix = torch.where(
            in_window,
            torch.exp(beta * (i_idx[None, :] - starts[:, None]).to(dtype)),
            torch.zeros((), dtype=dtype, device=device),
        )
        valid = ~torch.isnan(preds)
        preds_clean = torch.nan_to_num(preds)
        weighted_sum = torch.einsum('hi,bic->bhc', weight_matrix, valid.to(dtype) * preds_clean)
        w_sum = torch.einsum('hi,bic->bhc', weight_matrix, valid.to(dtype))
        return torch.where(
            w_sum > 0, weighted_sum / w_sum,
            torch.tensor(float('nan'), dtype=dtype, device=device),
        )

    def _identity(self, preds):
        return preds
