import numpy as np
from metrics.eval_losses import quantile_loss
     

def _reshape_windows_by_date(x, stride, mask=None):
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
    stride : int
        Step size between consecutive forecast windows.
    mask : np.ndarray [B, T, H, C] or None
        If provided, masked positions (mask == 0) are set to NaN before rearranging.
 
    Returns
    -------
    np.ndarray [B, (T-1)*stride + H, H, C]
    """

    B, T, H, C = x.shape
    S = (T - 1) * stride + H
        
    if mask is not None:
        x = np.where(mask == 0, np.nan, x.copy())

    t_grid, h_grid = np.meshgrid(np.arange(T), np.arange(H), indexing='ij')
    d_grid = t_grid * stride + h_grid  # (T, H): target date for each (t, h) 
    out = np.full((B, S, H, C), np.nan)
    out[:, d_grid, h_grid, :] = x       # scatter (B, T, H, C) → (B, S, H, C)
    return out

def excess_volatility(targets, preds, quantiles, stride=1, scaling=True, mask=None):
    """
    Excess Volatility (EV) — measures harmful forecast instability by comparing
    the cost of a revision against the accuracy improvement it produced.
 
        EV = QL(ŷ_update, ŷ_before)               # revision cost
           - (QL(y, ŷ_before) - QL(y, ŷ_update))  # accuracy improvement

    For each overlapping window pair (ŷ_before, ŷ_update) predicting the same
    target date:
    - revision_cost   = QL(ŷ_update_median, ŷ_before)  how much ŷ_before mispredicts ŷ_update
    - accuracy_before = QL(y, ŷ_before)                 error of older forecast vs truth
    - accuracy_update = QL(y, ŷ_update)                 error of newer forecast vs truth

    Parameters
    ----------
    targets : np.ndarray [B, T, H, C]
        Ground truth targets.
    preds : np.ndarray [B, T, H, C, Q]
        Quantile predictions across T forecast windows.
    quantiles : list[float]
        Quantile levels, e.g. [0.1, 0.5, 0.9].
    stride : int
        Step size between consecutive forecast windows. Must be < H, otherwise
        target dates have at most one forecast and stability is undefined.
    scaling : bool
        If True, normalises EV by sum(|y|) over the masked pairs to make it
        scale-independent. Legacy defaults this to False — match whichever the
        run you are comparing against used.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing. Independently of this, the
        structurally unreachable horizons at edge dates are always excluded.
 
    Returns
    -------
    float
    """

    B, T, H, C, Q = preds.shape
    S = (T - 1) * stride + H
 
    if stride == H:
        raise ValueError(
            f"stride={stride} equals H={H}: windows are non-overlapping so each "
            "target date has exactly one forecast. Cannot measure forecast stability in this case."
        )
    elif stride > H:
        raise ValueError(
            f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
            "Please review the selected stride parameter used in your experiment."
        )
 
    reshaped_preds = _reshape_windows_by_date(
        x=preds.reshape(B, T, H, C * Q),
        mask=np.repeat(mask, Q, axis=-1) if mask is not None else None,
        stride=stride,
    )
    reshaped_y = _reshape_windows_by_date(x=targets, stride=stride)

    K = H - stride                                     # pairs per target date
    y_hat_before = reshaped_preds[:, :, stride:,  :]   # [B, T, K, C*Q]  larger h = OLDER
    y_hat_update = reshaped_preds[:, :, :-stride, :]   # [B, T, K, C*Q]  smaller h = NEWER
    reshaped_y   = reshaped_y[:, :, stride:, :]        # [B, T, K, C]    aligned with `before`

    reshaped_mask = _reshape_windows_by_date(
        x=mask if mask is not None else np.ones_like(targets, dtype=float),
        stride=stride,
    )
    pair_mask = np.logical_and(
        np.nan_to_num(reshaped_mask[:, :, stride:,  :], nan=0.0),   # before
        np.nan_to_num(reshaped_mask[:, :, :-stride, :], nan=0.0),   # update
    ).astype(float)
 
    y_hat_before = np.nan_to_num(y_hat_before, nan=0.0).reshape(B, S, K, C, Q)
    y_hat_update = np.nan_to_num(y_hat_update, nan=0.0).reshape(B, S, K, C, Q)
    reshaped_y   = np.nan_to_num(reshaped_y, nan=0.0)

    # aggregate=None: return per-element losses so EV is computed before any aggregation
    revision_cost = quantile_loss(
        preds=y_hat_before,     # pred  = OLDER forecast
        targets=y_hat_update,   # truth = NEWER forecast, full quantile vector
        quantiles=quantiles,
        mask=pair_mask,
        aggregate=None,
    ) / Q  # scale by quantiles
    accuracy_before = quantile_loss(
        preds=y_hat_before,
        targets=reshaped_y,
        quantiles=quantiles,
        mask=pair_mask,
        aggregate=None,
    ) / Q  # scale by quantiles
    accuracy_update = quantile_loss(
        preds=y_hat_update,
        targets=reshaped_y,
        quantiles=quantiles,
        mask=pair_mask,
        aggregate=None,
    ) / Q  # scale by quantiles
 
    EV = (revision_cost - (accuracy_before - accuracy_update)).sum()
    denom = np.sum(np.abs(reshaped_y) * pair_mask) + 1e-8
 
    return EV / denom if scaling else EV



def forecast_percentage_change(preds, stride=1, symmetric=True, mask=None, eps=1e-6):
    """
    Forecast Percentage Change (FPC) — measures the relative magnitude of
    forecast revisions across consecutive Forecast Creation Dates (FCDs).
 
    For each overlapping window pair on the same target date, where ŷ_before is
    the OLDER forecast and ŷ_update the NEWER one:
 
        symmetric=True  (sFPC, sMAPE-style):
            200 * mean( |ŷ_update - ŷ_before| / (|ŷ_update| + |ŷ_before| + ε) )
 
        symmetric=False (FPC):
            mean( |ŷ_update - ŷ_before| / (|ŷ_before| + ε) )

    These are two different quantities, not two scalings of one: sFPC is symmetric in
    the two forecasts and bounded in [0, 200], while FPC divides by the older forecast
    alone, is unbounded, and carries no factor of 200. Higher values indicate greater
    forecast volatility in both cases.

    Parameters
    ----------
    preds : np.ndarray [B, T, H, C]
        Point predictions across T forecast windows.
    stride : int
        Step size between consecutive forecast windows. Must be < H, otherwise
        target dates have at most one forecast and stability is undefined.
    symmetric : bool
        If True, uses the symmetric denominator and scales by 200 (sFPC).
        If False, uses the one-sided denominator |ŷ_before| and does not
        scale (FPC). These are different quantities, not two scalings of one.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing. Masked positions are
        excluded from the mean via nan propagation.
    eps : float
        Denominator floor, guarding division by zero. Legacy uses 1e-4; smaller
        values amplify revisions on series that approach zero, so match legacy
        if you are comparing against archived runs.
 
    Returns
    -------
    float
    """
    _, _, H, _ = preds.shape

    if stride == H:
        raise ValueError(
            f"stride={stride} equals H={H}: windows are non-overlapping so each "
            "target date has exactly one forecast. Cannot measure forecast stability in this case."
        )
    elif stride > H:
        raise ValueError(
            f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
             "Please review the selected stride parameter used in your experiment."
        )
    
    reshaped_preds = _reshape_windows_by_date(
        x=preds, 
        mask=mask, 
        stride=stride
    )

    y_hat_before = reshaped_preds[:, :, stride:,  :]   # [B, S, H-stride, C]  larger h = OLDER
    y_hat_update = reshaped_preds[:, :, :-stride, :]   # [B, S, H-stride, C]  smaller h = NEWER
 
    num = np.abs(y_hat_update - y_hat_before)
    if symmetric:
        den = np.abs(y_hat_update) + np.abs(y_hat_before) + eps
        return 200 * np.nanmean(num / den)
    else:
        den = np.abs(y_hat_before) + eps               # one-sided, older forecast
        return np.nanmean(num / den)                   # no 200x
