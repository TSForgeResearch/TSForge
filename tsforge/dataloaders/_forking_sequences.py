from typing import Dict, Tuple, Optional
import torch
from torch import Tensor

from ..common._utils import mix_seed


def _gather_block(
    src:          Tensor,   # [B, S, C, *extra]
    window_start: Tensor,   # [B]
    block_len:    int,
    T:            int,
) -> Tensor:
    """Gather a contiguous block of `block_len` steps forward from window_start."""
    B     = src.shape[0]
    extra = src.shape[2:]
    offsets = torch.arange(block_len, device=src.device)
    grid    = (window_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, T - 1)
    return src.gather(
        1,
        grid.unsqueeze(-1).unsqueeze(-1).expand(B, block_len, *extra)
    )


def _gather_mask(
    mask:         Tensor,   # [B, S, C]
    window_start: Tensor,   # [B]
    block_len:    int,
    T:            int,
) -> Tensor:
    B, _, C = mask.shape
    grid = (
        window_start.unsqueeze(1)
        + torch.arange(block_len, device=mask.device).unsqueeze(0)
    ).clamp(0, T - 1)                                     # [B, block_len]
    grid = grid.unsqueeze(-1).expand(B, block_len, C)     # [B, block_len, C]
    return mask.gather(1, grid)                           # [B, block_len, C]


def _unfold_windows(src: Tensor, size: int, step: int) -> Tensor:
    """
    Unfold time dim into sliding windows of `size` spaced `step` apart.
    torch.unfold only creates COMPLETE windows (does not exceed available data).

    Number of windows produced:
        n_fcds = floor((T - size) / step) + 1

    [B, T, C, *extra]  ->  [B, n_fcds, size, C, *extra]
    [B, T]             ->  [B, n_fcds, size]
    """
    unfolded = src.unfold(dimension=1, size=size, step=step)
    if unfolded.ndim == 3:                         # mask: [B, n_fcds, size]
        return unfolded.contiguous()
    ndim  = unfolded.ndim
    order = [0, 1, ndim - 1] + list(range(2, ndim - 1))
    return unfolded.permute(*order).contiguous()   # [B, n_fcds, size, C, *extra]


def _pad_right(src: Tensor, n_pad: int) -> Tensor:
    """
    Append `n_pad` zero timesteps to the time dim of [B, T, ...].

    Used by the blocked eval strategy so the trailing block — which runs past
    the end of the series whenever n_fcds isn't a multiple of the block size —
    reads real (zero, masked-out) entries instead of an out-of-range index.
    """
    return torch.cat(
        [src, src.new_zeros(src.shape[0], n_pad, *src.shape[2:])], dim=1
    )


#: Minimum real observations required in a window's normalization context for
#: its statistics to be usable. Below 2 the variance is not estimable: with
#: exactly one observation mean==x and mean_sq==x**2, so the variance is 0 and
#: stdev collapses to sqrt(eps)~0.003, which inflates any real target ~300x.
#: Windows under this threshold are dropped from the loss (see __call__).
MIN_NORM_OBS = 2


def n_valid_fcds(T: int, context_len: int, horizon: int, stride: int) -> int:
    """
    How many complete FCD windows fit in a series of length T.

    A window needs (context_len + horizon) consecutive timesteps.
    torch.unfold enforces completeness automatically; this function makes
    the arithmetic explicit for planning and assertions.

    Example
    -------
    T=10, L=3, H=2, step=2
        window_size = 5
        n_fcds = floor((10 - 5) / 2) + 1 = 3
        Windows cover t=[0..4], [2..6], [4..8]
        t=[6..10] would overflow -> dropped.
    """
    window_size = context_len + horizon
    if T < window_size:
        return 0
    return (T - window_size) // stride + 1


class ForkingSequences:
    """
    Reformat a full-series batch into forking-sequence model inputs.

    Parameters
    ----------
    context_len : int
    stride      : int
    fcd_sampler    : str    'heterogeneous' (default) | 'homogeneous'

    fcd_samples != -1  (training)
        Sampler picks window_start such that the block fits within [0, S-1].
        The train series is extended by H-1 masked val rows, so the last H-1
        windows within train are now reachable. Predictions landing in the
        masked extension rows have outsample_mask=0 and contribute nothing
        to the loss.

    fcd_samples == -1  (val / test)
        The full series is consumed (ctx_rows + eval rows + H-1 extension).
        _unfold_windows produces floor((S - L - H) / step) + 1 windows.
        ctx_rows and extension rows have available_mask=0, so predictions
        landing there are excluded from the loss. Any incomplete trailing
        window is dropped — no overflow.

    Inputs  (left-padded, from collate)
    ------------------------------------
    x_enc : [B, S, C, 1+Vh]
    available_mask : [B, S, C]  0=pad/missing, 1=real

    Outputs
    -------
    insample_y     : [B, enc_size, C, 1+Vh]    enc_size = block_len - H
    outsample_y    : [B, n_fcds,   H,  C]
    outsample_mask : [B, n_fcds,   H,  C]      0 where loss should be ignored
    available_mask : [B, enc_size, C]
    """

    SAMPLERS = {"heterogeneous", "homogeneous"}

    def __init__(
        self,
        context_len: int,
        fcd_samples: int = -1,
        patch_len: int = 1,
        stride: int = 1,
        fcd_sampler: str = "heterogeneous",
        norm_window_size: int = -1,
        blocked: bool = False,
        fcd_seed: int = 0,
    ):
        if fcd_sampler not in self.SAMPLERS:
            raise ValueError(
                f"fcd_sampler must be one of {self.SAMPLERS}, got '{fcd_sampler}'"
            )
        # Window sampling draws from a dedicated generator keyed on
        # (fcd_seed, step) rather than the global torch stream — see
        # _generator(). Only the sampling strategies use it; the blocked and
        # all-FCD strategies enumerate windows and never sample.
        self.fcd_seed = fcd_seed
        self._step = 0
        self._gens = {}
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride
        self.norm_window_size = norm_window_size
        self.causal_stats = None
        self.blocked = blocked
        self.fcd_sampler = (
            self._heterogeneous_sampler
            if fcd_sampler == "heterogeneous"
            else self._homogeneous_sampler
        )

        if blocked:
            if context_len == -1 or fcd_samples == -1:
                raise ValueError(
                    "blocked=True requires a fixed context_len and a positive "
                    f"fcd_samples, got context_len={context_len}, "
                    f"fcd_samples={fcd_samples}."
                )
            self._strategy = self._blocked_fcds_fixed_context
        elif context_len != -1 and fcd_samples != -1:
            self._strategy = self._sampled_fcds_fixed_context
        elif context_len != -1 and fcd_samples == -1:
            self._strategy = self._all_fcds_fixed_context
        elif context_len == -1 and fcd_samples != -1:
            self._strategy = self._sampled_fcds_full_context
        else:
            self._strategy = self._all_fcds_full_context

    def _generator(self, device: torch.device) -> torch.Generator:
        """
        Generator keyed on (fcd_seed, step), re-seeded on every call.

        Re-seeding rather than letting it advance is deliberate: it makes the
        draw at step N a pure function of N, independent of how many times the
        sampler ran before it. A run resumed from a checkpoint therefore sees
        the same windows an uninterrupted run would.

        The generator is bound to a device because multinomial on a CUDA
        tensor requires a CUDA generator; cached per device so DDP ranks and
        CPU tests both work without branching.
        """
        g = self._gens.get(device)
        if g is None:
            g = self._gens[device] = torch.Generator(device=device)
        g.manual_seed(mix_seed(self.fcd_seed, self._step))
        return g

    def _homogeneous_sampler(
        self,
        available_mask: Tensor,
        channel_mask: Tensor,
        fcd_samples:    int,
        horizon:        int,
    ) -> Tuple[Tensor, int]:
        
        B, S, _ = available_mask.shape
        H = horizon
        L = self.context_len if self.context_len != -1 else self.patch_len
        block_len = L + (fcd_samples - 1) * self.stride + H
        max_start = S - block_len
        if max_start < 0:
            raise ValueError(
                f"Series length {S} is too short for block_len {block_len}. "
                f"Reduce fcd_samples, context_len, or horizon."
            )

        # aggregate availability across batch and channels — only sample positions
        # where real data exists for ALL series and ALL channels
        if channel_mask is not None:
            # only consider real channels — set padded channels to 1 so min ignores them
            padded = (channel_mask == 0).unsqueeze(1).expand_as(available_mask)
            masked_avail = available_mask.clone()
            masked_avail[padded] = 1
            time_mask = masked_avail.min(dim=2).values   # [B, S]
        else:
            time_mask = available_mask.min(dim=2).values

        batch_mask = time_mask.min(dim=0).values       # [S]    min over batch

        sample_weights = batch_mask.float().clone()
        sample_weights[max_start + 1:] = 0.0           # enforce block fits

        if sample_weights.sum() == 0:
            # fallback: no position valid for all series — just respect max_start
            # (device must match: a CPU index here would crash the CUDA gather
            # in _gather_block — the heterogeneous path already gets this right)
            sample_weights = torch.ones(S, device=sample_weights.device)
            sample_weights[max_start + 1:] = 0.0

        window_start = torch.multinomial(
            sample_weights, num_samples=1,
            generator=self._generator(sample_weights.device),
        )  # [1]
        window_start = window_start.repeat(B)                            # [B]
        return window_start, block_len

    def _heterogeneous_sampler(
        self,
        available_mask: Tensor,
        channel_mask: Tensor,
        fcd_samples: int,
        horizon: int,
    ) -> Tuple[Tensor, int]:
        """
        Sample one window_start per series, proportional to availability.

        Constraints on window_start[b]
        --------------------------------
        Valid positions are those where available_mask == 1 after collapsing
        channels with min. A timestep is valid only if ALL [non-padded] channels 
        have  real data there.

        window_start + block_len - 1 <= S - 1
        block_len already includes horizon, so no further subtraction needed.

        The train dataset is extended by H-1 masked rows from the val set, so
        windows whose horizons land in those rows are geometrically valid but
        contribute zero loss (available_mask=0 → outsample_mask=0).

        T >= L+H is enforced by the dataset so at least one valid position
        always exists per series.

        Returns
        -------
        window_start : [B]   per-series index sampled from [first_real[b], max_start]
        block_len    : int   total span of one fcd_samples block
        """
        B, S, C = available_mask.shape
        L, H    = self.context_len, horizon

        block_len = L + (fcd_samples - 1) * self.stride + H
        max_start = S - block_len

        if max_start < 0:
            raise ValueError(
                f"Series length {S} is too short for context_len={L} + "
                f"fcd_samples={fcd_samples} * stride={self.stride} + horizon={H} "
                f"= block_len={block_len}. Reduce context_len or fcd_samples."
            )
        
        if channel_mask is not None:
            # only consider real channels — set padded channels to 1 so min ignores them
            padded = (channel_mask == 0).unsqueeze(1).expand_as(available_mask)
            masked_avail = available_mask.clone()
            masked_avail[padded] = 1
            time_mask = masked_avail.min(dim=2).values   # [B, S]
        else:
            time_mask = available_mask.min(dim=2).values

        sample_weights = time_mask.float().clone()
        sample_weights[:, max_start + 1:] = 0.0

        # A series with no real timestep anywhere in [0, max_start] (e.g. a
        # short, left-padded series where max_start is small and every
        # candidate start position is itself padding) leaves that row all
        # zero, which torch.multinomial can't sample from. Fall back to
        # uniform over [0, max_start] for just that row — same fallback
        # _homogeneous_sampler already applies, just per-series instead of
        # batch-wide since each series samples its own window_start here.
        zero_rows = sample_weights.sum(dim=1) == 0
        if zero_rows.any():
            fallback = torch.ones(S, device=sample_weights.device)
            fallback[max_start + 1:] = 0.0
            # torch.where, not `sample_weights[zero_rows] = fallback`: the
            # deterministic index_put_ CUDA kernel mis-broadcasts a 1-D value
            # into a boolean-mask selection and trips an internal assert
            # (fires under torch.use_deterministic_algorithms(True), which
            # set_determinism enables). where() is equivalent and unaffected.
            sample_weights = torch.where(
                zero_rows.unsqueeze(1), fallback.unsqueeze(0), sample_weights
            )

        window_start = torch.multinomial(
            sample_weights, num_samples=1,
            generator=self._generator(sample_weights.device),
        ).squeeze(1)
        return window_start, block_len
    
    def _sampled_fcds_fixed_context(
        self, 
        batch: Dict[str, Tensor],
        horizon: int,
        fcd_samples: int,
    ) -> Dict[str, Tensor]:
        """
        Fixed context length, sampled windows.

        Gathers a contiguous block of fcd_samples windows from a randomly sampled
        anchor per series (heterogeneous) or shared anchor (homogeneous). Window
        size is always context_len + horizon — consistent across the batch because
        context_len is fixed.
        """

        x_enc_full = batch["x_enc"]
        available_mask = batch["available_mask"]
        channel_mask = batch["channel_mask"]
        loss_mask = batch["loss_mask"]
        #hist_mask = batch.get("hist_mask")

        B, T, C, _ = x_enc_full.shape

        window_start, block_len = self.fcd_sampler(
            available_mask=available_mask,
            channel_mask=channel_mask,
            fcd_samples=fcd_samples,
            horizon=horizon,
        )
        enc_block  = _gather_block(
            src=x_enc_full, 
            window_start=window_start, 
            block_len=block_len, 
            T=T,
        )
        mask_block = _gather_mask(
            mask=available_mask, 
            window_start=window_start, 
            block_len=block_len, 
            T=T,
        )
        loss_mask_block = _gather_mask(
            mask=loss_mask, 
            window_start=window_start, 
            block_len=block_len, 
            T=T,
        )
        window_size = self.context_len + horizon
        
        return enc_block, mask_block, loss_mask_block, window_size, fcd_samples, window_start

    def _blocked_fcds_fixed_context(
        self,
        batch: Dict[str, Tensor],
        horizon: int,
        fcd_samples: int,
    ) -> Dict[str, Tensor]:
        """
        Fixed context length, all valid windows — but chunked into blocks
        shaped exactly like a training block and folded into the batch
        dimension.

        `_all_fcds_fixed_context` hands the encoder the whole series in one
        shared pass, so a window late in the series can draw on far more
        history than any training block ever contained (bounded only by the
        series length). This covers the same FCDs, but each block spans only
        block_len = L + (fcd_samples-1)*stride + H — the identical geometry
        `_sampled_fcds_fixed_context` produces at train time — so the
        encoder never sees a longer sequence at eval than it did at train.

        Compute is still shared *within* a block (one pass covers
        fcd_samples origins); it is not shared across block boundaries.

        Blocks tile the FCD axis: block j covers FCDs
        [j*fcd_samples, (j+1)*fcd_samples). The trailing block is right-padded
        with masked-out zeros; those padded FCDs are additionally zeroed out of
        outsample_mask in __call__, since an FCD straddling the boundary has
        real context but an incomplete horizon and so isn't a valid window.
        """
        x_enc_full     = batch["x_enc"]           # [B, T, C, 1+Vh]
        available_mask = batch["available_mask"]  # [B, T, C]
        loss_mask      = batch["loss_mask"]

        B, T, C, X1 = x_enc_full.shape
        L, H = self.context_len, horizon
        device = x_enc_full.device

        n_fcds = (T - L - H) // self.stride + 1
        if n_fcds < 1:
            raise ValueError(
                f"Series length {T} is too short for context_len={L} + "
                f"horizon={H}: no valid FCD windows."
            )

        n_blocks  = (n_fcds + fcd_samples - 1) // fcd_samples   # ceil
        block_len = L + (fcd_samples - 1) * self.stride + H

        starts  = torch.arange(n_blocks, device=device) * fcd_samples * self.stride
        offsets = torch.arange(block_len, device=device)

        # n_fcds rarely divides evenly by fcd_samples, so the trailing block
        # runs past the end of the series. Right-pad rather than clamping the
        # gather index: padded rows are real entries carrying available_mask=0
        # and loss_mask=0, so attention masking and _compute_norm_stats exclude
        # them on their own — where a clamp would silently duplicate the last
        # real timestep's value instead.
        n_pad = max(0, int(starts[-1].item()) + block_len - T)
        if n_pad:
            x_enc_full     = _pad_right(x_enc_full, n_pad)
            available_mask = _pad_right(available_mask, n_pad)
            loss_mask      = _pad_right(loss_mask, n_pad)

        # One gather builds every block: the index is [n_blocks, block_len],
        # so the full series is never materialized once per block.
        idx  = starts.unsqueeze(1) + offsets.unsqueeze(0)   # [n_blocks, block_len]
        flat = idx.reshape(-1).unsqueeze(0).expand(B, -1)   # [B, n_blocks*block_len]

        enc_block = x_enc_full.gather(
            1, flat.unsqueeze(-1).unsqueeze(-1).expand(B, flat.shape[1], C, X1)
        ).reshape(B * n_blocks, block_len, C, X1)

        mask_flat = flat.unsqueeze(-1).expand(B, flat.shape[1], C)
        mask_block = available_mask.gather(1, mask_flat).reshape(B * n_blocks, block_len, C)
        loss_mask_block = loss_mask.gather(1, mask_flat).reshape(B * n_blocks, block_len, C)

        # b-major, block-minor — matches the reshapes above and the
        # channel_mask repeat_interleave in __call__
        window_start = starts.unsqueeze(0).expand(B, n_blocks).reshape(-1)   # [B*n_blocks]

        return enc_block, mask_block, loss_mask_block, L + H, fcd_samples, window_start

    def _all_fcds_fixed_context(
        self,
        batch: Dict[str, Tensor],
        horizon: int,
        **_,
    ) -> Dict[str, Tensor]:
        """
        Fixed context length, all valid windows.

        Passes the full series to _unfold_windows with window_size = context_len +
        horizon. valid_fcds is derived from T — no sampling.
        """
        
        x_enc_full = batch["x_enc"]
        available_mask = batch["available_mask"]
        loss_mask = batch["loss_mask"]
        #hist_mask = batch.get("hist_mask")

        B, T, C, _ = x_enc_full.shape

        enc_block   = x_enc_full
        mask_block  = available_mask
        loss_mask_block = loss_mask
        window_size = self.context_len + horizon
        valid_fcds = (T - self.context_len - horizon) // self.stride + 1

        return enc_block, mask_block, loss_mask_block, window_size, valid_fcds, None

    def _sampled_fcds_full_context(
        self, 
        batch: Dict[str, Tensor],
        horizon: int,
        fcd_samples: int,
    ) -> Dict[str, Tensor]:
        """
        Full context (context_len=-1), sampled windows.

        Must use the homogeneous sampler — without a fixed context_len, a per-series
        anchor would produce different window sizes across the batch, breaking the
        tensor shape assumption. The homogeneous sampler enforces a single shared
        anchor so window_size = block_end - (fcd_samples-1)*stride is consistent
        across all series in the batch.
        """
        
        x_enc_full = batch["x_enc"]
        available_mask = batch["available_mask"]
        channel_mask = batch["channel_mask"]
        loss_mask = batch["loss_mask"]
        #hist_mask = batch.get("hist_mask")

        B, T, C, _ = x_enc_full.shape

        block_len = self.patch_len + (fcd_samples - 1) * self.stride + horizon
        window_start, _ = self._homogeneous_sampler(
            available_mask=available_mask, 
            channel_mask=channel_mask,
            fcd_samples=fcd_samples, 
            horizon=horizon,
        )
        block_end = window_start[0].item() + block_len  # same for all series in batch

        # align window_start to patch boundary
        remainder = window_start[0].item() % self.stride
        if remainder != 0:
            up = window_start[0].item() + (self.stride - remainder)
            down = window_start[0].item() - remainder
            # prefer rounding up unless it pushes block_end past T
            aligned = up if up + block_len <= T else down
            window_start = torch.tensor([aligned]).repeat(B)
            block_end = aligned + block_len

        enc_block = x_enc_full[:, :block_end]
        mask_block = available_mask[:, :block_end]
        loss_mask_block = loss_mask[:, :block_end]
        window_size = block_end - (fcd_samples - 1) * self.stride

        # window_start is INTERNAL here — it only picks block_end. enc_block is
        # the 0-based prefix x_enc_full[:, :block_end], not a block gathered at
        # window_start (that's _sampled_fcds_fixed_context), so handing it back
        # would make __call__ offset its stats lookup by it and read past the
        # end of the series. None is how the other 0-based strategies say
        # "my block starts at 0".
        return enc_block, mask_block, loss_mask_block, window_size, fcd_samples, None

    def _all_fcds_full_context(
        self, 
        batch: Dict[str, Tensor],
        horizon: int,
        **_,
    ) -> Dict[str, Tensor]:
        """
        Full context (context_len=-1), all valid windows.

        Passes the full series to _unfold_windows with window_size = 1 + horizon.
        Every timestep is a valid anchor — autoregressive interpretation. valid_fcds
        derived from T, no sampling.
        """
            
        x_enc_full = batch["x_enc"]
        available_mask = batch["available_mask"]
        loss_mask = batch["loss_mask"]
        #hist_mask = batch.get("hist_mask")

        B, T, C, _ = x_enc_full.shape

        enc_block   = x_enc_full
        mask_block  = available_mask
        loss_mask_block = loss_mask
        window_size = self.patch_len + horizon
        valid_fcds = (T - self.patch_len - horizon) // self.stride + 1

        return enc_block, mask_block, loss_mask_block, window_size, valid_fcds, None

    def __call__(
        self,
        batch: Dict[str, Tensor],
        horizon: int,
        fcd_samples: int = -1,
        step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        # Window draws key off (fcd_seed, step) so they never depend on how
        # much of the global RNG stream anything else consumed — dropout,
        # DataLoader worker seeding, a different loss function. Two runs
        # differing only in their loss therefore see identical FCD windows on
        # identical series at identical steps.
        #
        # Eval passes no step: its strategies enumerate windows, never sample.
        if step is not None:
            self._step = int(step)

        enc_block, mask_block, loss_mask_block, window_size, valid_fcds, window_start = self._strategy(
            batch=batch,
            horizon=horizon,
            fcd_samples=fcd_samples,
        )

        enc_windows = _unfold_windows(src=enc_block, size=window_size, step=self.stride)
        loss_mask_windows = _unfold_windows(src=loss_mask_block, size=window_size, step=self.stride)

        eff_L = window_size - horizon
        outsample_mask = loss_mask_windows[:, :, eff_L:, :]
        enc_size = enc_block.shape[1] - horizon

        # ── Normalization stats ──
        # Computed over the FULL series before any windowing — the blocked
        # strategy folds blocks into the batch dim, but stats must still be
        # cumulative over all real history the way training computes them,
        # not restart at each block boundary.
        x_full = batch["x_enc"]
        avail_full = batch["available_mask"]
        B, S = x_full.shape[:2]

        # 1 for every non-blocked strategy; B*n_blocks rows otherwise
        n_blocks = enc_block.shape[0] // B

        if window_start is not None:
            ws = window_start.reshape(B, n_blocks)          # [B, n_blocks]
        else:
            ws = torch.zeros(B, n_blocks, dtype=torch.long, device=x_full.device)

        # The blocked strategy right-pads its trailing block; mirror that here
        # so the stats source spans the same range. Padded rows carry
        # available_mask=0, so _compute_norm_stats leaves the cumulative
        # counts untouched and real positions are unaffected.
        need = int(ws.max().item()) + enc_size
        if need > S:
            x_full = _pad_right(x_full, need - S)
            avail_full = _pad_right(avail_full, need - S)

        mask_full = avail_full.unsqueeze(-1).expand_as(x_full)
        stats = self._compute_norm_stats(x_full, mask_full)
        mean, stdev = stats['mean'], stats['stdev']

        def _gather_one(src: Tensor, offsets: Tensor, n_out: int):
            """
            Gather [B*n_blocks, n_out, C, X+1] from [B, S, C, X+1] without
            materializing a per-block copy of the full series: index in the
            [B, n_blocks*n_out] layout, then reshape.
            """
            idx = ws.unsqueeze(-1) + offsets.view(1, 1, -1)
            idx = idx.reshape(B, n_blocks * n_out)
            idx = idx.unsqueeze(-1).unsqueeze(-1).expand(B, n_blocks * n_out, *src.shape[2:])
            return src.gather(1, idx).reshape(B * n_blocks, n_out, *src.shape[2:])

        def _gather_stats(offsets: Tensor, n_out: int):
            return (_gather_one(mean, offsets, n_out),
                    _gather_one(stdev, offsets, n_out))

        # Per-timestep stats for norm
        ts_offsets = torch.arange(enc_size, device=x_full.device)
        ts_mean, ts_stdev = _gather_stats(ts_offsets, enc_size)

        # Per-FCD stats for denorm/norm_targets
        fcd_offsets = torch.arange(valid_fcds, device=x_full.device) * self.stride + eff_L - 1
        fcd_mean, fcd_stdev = _gather_stats(fcd_offsets, valid_fcds)

        # A window whose normalization context contains no real observations
        # has no usable statistics: _compute_norm_stats falls back to mean 0
        # and stdev sqrt(eps)~0.003, which divides a real target by ~0.003 and
        # inflates it ~300x. Such a window is not a legitimate training
        # example, so drop it from the loss exactly as geometrically-invalid
        # windows are dropped. On heavily left-padded short series this is a
        # large fraction of windows (~40% on M-competition yearly), and those
        # windows otherwise dominate the reported loss.
        fcd_count   = _gather_one(stats['count'], fcd_offsets, valid_fcds)
        has_history = (fcd_count[..., 0] >= MIN_NORM_OBS).unsqueeze(2)  # [B*nb, n_fcds, 1, C]
        outsample_mask = outsample_mask * has_history.to(outsample_mask.dtype)

        channel_mask = batch['channel_mask']
        if n_blocks > 1:
            # b-major, block-minor — matches enc_block's reshape ordering
            channel_mask = channel_mask.repeat_interleave(n_blocks, dim=0)

            # The trailing block's padded slots aren't fully covered by the
            # zero-padded loss_mask: an FCD straddling the boundary has real
            # context and a partly-real horizon, so its mask comes out partly
            # live even though it isn't a valid window. Zero those FCDs
            # explicitly so they contribute neither loss nor forecasts.
            n_fcds = (S - self.context_len - horizon) // self.stride + 1
            fcd_ids = torch.arange(
                n_blocks * valid_fcds, device=x_full.device
            ).reshape(1, n_blocks, valid_fcds)
            fcd_ok = (fcd_ids < n_fcds).expand(B, n_blocks, valid_fcds)
            fcd_ok = fcd_ok.reshape(B * n_blocks, valid_fcds)
            outsample_mask = outsample_mask * fcd_ok[:, :, None, None].to(outsample_mask.dtype)
        else:
            n_fcds = valid_fcds

        out = dict(
            insample_y=enc_block[:, :enc_size],
            outsample_y=enc_windows[:, :, eff_L:, :, 0],
            outsample_mask=outsample_mask,
            available_mask=mask_block[:, :enc_size],
            channel_mask=channel_mask,
            fcd_samples=valid_fcds,
            horizon=horizon,
            n_blocks=n_blocks,
            n_fcds=n_fcds,
        )
        out["norm_stats"] = {'mean': ts_mean, 'stdev': ts_stdev}
        out["norm_fcd_stats"] = {'mean': fcd_mean, 'stdev': fcd_stdev}

        return out


    def _compute_norm_stats(
        self,
        x_full: Tensor,          # [B, S, C, X+1]
        mask_full: Tensor,       # [B, S, C, X+1]
        eps: float = 1e-5,
    ) -> dict:
        """
        Compute per-timestep normalization statistics over the full series.

        Parameters
        ----------
        x_full : [B, S, C, X+1]
        mask_full : [B, S, C, X+1]  1=valid, 0=masked
        norm_window_size : int
            -1  → causal (cumulative from 0..t)
            >0  → rolling window of this size ending at t

        Returns
        -------
        dict with 'mean', 'stdev': each [B, S, C, X+1]
        """
        B, S, C, X1 = x_full.shape

        if self.norm_window_size == -1:
            # Causal: cumulative stats from 0..t
            raw_counts = torch.cumsum(mask_full, dim=1)
            counts  = raw_counts.clamp(min=1)
            mean    = torch.cumsum(x_full * mask_full, dim=1) / counts
            mean_sq = torch.cumsum((x_full ** 2) * mask_full, dim=1) / counts
        else:
            # Windowed: rolling stats over last W steps ending at t
            W = min(self.norm_window_size, S)
            x_windows    = x_full.unfold(1, W, 1)        # [B, S-W+1, C, X+1, W]
            mask_windows = mask_full.unfold(1, W, 1)      # [B, S-W+1, C, X+1, W]
            raw_counts = mask_windows.sum(dim=-1)
            counts  = raw_counts.clamp(min=1)
            mean    = (x_windows * mask_windows).sum(dim=-1) / counts
            mean_sq = ((x_windows ** 2) * mask_windows).sum(dim=-1) / counts

            # Pad early positions (0..W-2) with causal stats as fallback
            if W > 1:
                early_mask    = mask_full[:, :W-1]
                early_raw     = torch.cumsum(early_mask, dim=1)
                early_counts  = early_raw.clamp(min=1)
                early_mean    = torch.cumsum(x_full[:, :W-1] * early_mask, dim=1) / early_counts
                early_mean_sq = torch.cumsum((x_full[:, :W-1] ** 2) * early_mask, dim=1) / early_counts
                mean    = torch.cat([early_mean, mean], dim=1)
                mean_sq = torch.cat([early_mean_sq, mean_sq], dim=1)
                raw_counts = torch.cat([early_raw, raw_counts], dim=1)

        stdev = torch.sqrt((mean_sq - mean ** 2).clamp(min=0) + eps)
        # 'count' is the UNCLAMPED number of real observations behind each
        # statistic. count==0 means the normalization window held no real data,
        # so mean/stdev are the degenerate fallback (0, sqrt(eps)) rather than
        # anything measured — callers use it to drop those windows.
        return {'mean': mean, 'stdev': stdev, 'count': raw_counts}