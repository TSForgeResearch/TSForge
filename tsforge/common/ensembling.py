import numpy as np

class Ensembler:
    def __init__(self, ensemble_method='mean', stride=1, **kwargs):
        self.kwargs = kwargs
        self.stride = stride
        methods = {
            'mean':     self._cumulative_mean,
            'median':   self._cumulative_median,
            'ewm':      self._cumulative_ewm,
            'identity': self._identity,
        }
        if ensemble_method not in methods:
            raise ValueError(f"ensemble_method must be one of {list(methods.keys())}")
        self.ensembler = methods[ensemble_method]

    def ensemble(self, preds, mask=None):
        """
        preds: (B, T, H, C, Q)
        mask:  (B, T, H, C)    int — 1=valid, 0=mask out
        returns: (B, T, H, C, Q)
        """
        B, T, H, C, Q = preds.shape

        stride = self.stride
        
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

        # expand mask to cover C*Q channels
        mask_flat = (
            np.repeat(mask, Q, axis=-1)   # (B, T, H, C*Q)
            if mask is not None else None
        )

        masked_preds = self._reshape_windows_by_date(
            x=preds_flat, 
            mask=mask_flat
        )  # (B, S, H, C*Q)

        B_, S_, H_, CQ_ = masked_preds.shape
        flat      = masked_preds.reshape(B_ * S_, H_, CQ_)                       # (B*S, H, C*Q)

        # For a shared target date a larger h holds an OLDER forecast, so the causal
        # pool for horizon h is h..H-1 (this FCD and earlier), not 0..h. The kernels
        # below all accumulate over 0..i, so reverse the horizon axis around them.
        flat      = flat[:, ::-1, :]
        ensembled = self.ensembler(flat, **self.kwargs)                           # (B*S, H, C*Q)
        ensembled = ensembled[:, ::-1, :]

        ensembled = ensembled.reshape(B_, S_, H_, CQ_)                           # (B, S, H, C*Q)

        out = self.ensembled_preds_reshape_for_windows(ensembled)                # (B, T, H, C*Q)

        return out.reshape(B, T, H, C, Q)

    def _reshape_windows_by_date(self, x, mask=None):
        """
        Rearranges overlapping forecast windows from [B, T, H, C] into
        [B, (T-1)*stride + H, H, C], grouping predictions by their target date.

        In a rolling forecast setup, multiple windows overlap on the same target date — e.g.
        the 1-step-ahead prediction from window t and the 2-step-ahead from window t-1 both
        target date t. This function collects those predictions into a single row, enabling
        direct comparison of forecasts that share the same target.

        The output has (T-1)*stride + H rows (one per unique target date) and H columns
        (one per forecast horizon that could predict that date). Edge dates are partially
        observed and will contain NaNs for horizons that don't reach that date.

        Parameters
        ----------
        x : np.ndarray [B, T, H, C]
        mask : np.ndarray [B, T, H, C] or None
            If provided, masked positions (mask == 0) are set to NaN before rearranging.

        Returns
        -------
        np.ndarray [B, (T-1)*stride + H, H, C]
        """
        B, T, H, C = x.shape
        S = (T - 1) * self.stride + H
        
        if mask is not None:
            x = np.where(mask == 0, np.nan, x.copy())

        t_grid, h_grid = np.meshgrid(np.arange(T), np.arange(H), indexing='ij')
        d_grid = t_grid * self.stride + h_grid  # (T, H): target date for each (t, h) 
        out = np.full((B, S, H, C), np.nan)
        out[:, d_grid, h_grid, :] = x       # scatter (B, T, H, C) → (B, S, H, C)
        return out

    def ensembled_preds_reshape_for_windows(self, ensembled_preds):
        """unchanged — operates on (B, S, H, C*Q)"""
        B, S, H, C = ensembled_preds.shape
        T = (S - H) // self.stride + 1

        t_grid, h_grid = np.meshgrid(np.arange(T), np.arange(H), indexing='ij')
        d_grid = t_grid * self.stride + h_grid  # same index grid as scatter 
        return ensembled_preds[:, d_grid, h_grid, :]  # gather → (B, T, H, C)

    def _cumulative_mean(self, preds, window_size=None, **kwargs):
        _, H, _ = preds.shape
        if window_size is None:
            cumsum = np.nancumsum(preds, axis=1)
            counts = np.cumsum(~np.isnan(preds), axis=1)
        else:
            cumsum = np.stack([np.nansum(preds[:, max(0, h-window_size+1):h+1, :], axis=1) for h in range(H)], axis=1)
            counts = np.stack([np.sum(~np.isnan(preds[:, max(0, h-window_size+1):h+1, :]), axis=1) for h in range(H)], axis=1)
        ensemble = cumsum / counts
        ensemble[counts == 0] = np.nan
        return ensemble

    def _cumulative_median(self, preds, window_size=None, **kwargs):
        _, H, _ = preds.shape
        return np.stack([
            np.nanmedian(preds[:, max(0, h-window_size+1) if window_size else 0:h+1, :], axis=1)
            for h in range(H)
        ], axis=1)

    def _cumulative_ewm(self, preds, alpha=0.5, window_size=None, **kwargs):
        B, H, C = preds.shape
        beta   = np.log(alpha / (1 - alpha))
        W      = window_size if window_size is not None else H
        h_idx  = np.arange(H)
        i_idx  = np.arange(H)
        starts = np.maximum(0, h_idx - W + 1)
        mask   = (i_idx[None, :] >= starts[:, None]) & (i_idx[None, :] <= h_idx[:, None])
        weight_matrix = np.where(mask, np.exp(beta * (i_idx[None, :] - starts[:, None])), 0.0)
        valid         = ~np.isnan(preds)
        preds_clean   = np.nan_to_num(preds)
        weighted_sum  = np.einsum('hi,bic->bhc', weight_matrix, valid * preds_clean)
        w_sum         = np.einsum('hi,bic->bhc', weight_matrix, valid.astype(float))
        return np.where(w_sum > 0, weighted_sum / w_sum, np.nan)

    def _identity(self, preds, **kwargs):
        return preds
