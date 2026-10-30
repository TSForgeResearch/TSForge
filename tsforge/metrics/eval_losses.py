"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""

import numpy as np


def mae(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None, # [B, T, H, C]
) -> float:
    err = np.abs(preds - targets)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def mse(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None,
) -> float:
    err = (preds - targets) ** 2
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def rmse(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    np.ndarray | None = None,
) -> float:
    return np.sqrt(mse(preds, targets, mask))


def mape(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = np.abs((targets - preds) / (np.abs(targets) + eps))
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def smape(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = 2 * np.abs(targets - preds) / (np.abs(targets) + np.abs(preds) + eps)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def quantile_loss(
    preds, 
    targets, 
    quantiles, 
    mask=None,
    aggregate='mean',
) -> float:
    """
    quantile_loss() returns the *mean pinball loss* across quantiles:
    QL = (1 / Q) * Σ_q L_q(y, ŷ_q)

    where L_q is the pinball loss:

        L_q(y, ŷ) = (y - ŷ) * q         if y >= ŷ
                  = (ŷ - y) * (1 - q)    if y <  ŷ
    """

    if targets.ndim < preds.ndim:
        errors = targets[..., np.newaxis] - preds # point target
    else:
        errors = targets - preds # per-quantile pairing
 
    q = np.array(quantiles, dtype=preds.dtype)
    loss = np.maximum(q * errors, (q - 1) * errors)
 
    if mask is not None:
        mask = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        loss = loss * mask
    if aggregate == 'mean':
        return loss.sum() / max(mask.sum(), 1) if mask is not None else loss.mean()
    return loss  # aggregate=None, return per-element


def coverage(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    lo_idx:    int = 0,
    hi_idx:    int = -1,
    mask:      np.ndarray | None = None,
) -> float:
    lo   = preds[..., lo_idx]                                          # [B, T, H, C]
    hi   = preds[..., hi_idx]                                          # [B, T, H, C]
    hits = ((targets >= lo) & (targets <= hi)).astype(np.float32)

    if mask is not None:
        return (hits * mask).sum() / max(mask.sum(), 1)
    return hits.mean()


def crps(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,
    scale:     bool = False,
    eps:       float = 1e-8,
) -> float:
    errors = targets[..., np.newaxis] - preds              # [B, T, H, C, Q]
    q      = np.array(quantiles, dtype=preds.dtype)
    loss   = 2 * np.maximum(q * errors, (q - 1) * errors) # [B, T, H, C, Q]

    if mask is not None:
        mask_q = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        score  = (loss * mask_q).sum() / max(mask_q.sum(), 1)
    else:
        score = loss.mean()

    if scale:
        denom = ((np.abs(targets) * mask).sum() / max(mask.sum(), 1)
                 if mask is not None else np.abs(targets).mean())
        score = score / (denom + eps)

    return score


def mase(
    preds:    np.ndarray,               # [B, T, H, C]
    targets:  np.ndarray,               # [B, T, H, C]
    insample: np.ndarray,               # [B, L, C] or [B, T, L, C]
    seasonality: int = 1,
    mask:     np.ndarray | None = None, # [B, T, H, C]
    eps:      float = 1e-8,
) -> float:
    """
    MASE = (1 / H) * Σ_h |y_{t+h} - ŷ_{t+h}| / ((1 / (T-m)) * Σ_j |y_j - y_{j-m}|)

    Scale is the in-sample seasonal naive MAE per series/channel (and per
    forecast creation date if insample is [B, T, L, C]). NaNs in insample are ignored.
    """
    scale = np.nanmean(np.abs(insample[..., seasonality:, :] - insample[..., :-seasonality, :]),
                       axis=-2)                                # [B, C] or [B, T, C]
    if scale.ndim == 2:
        scale = scale[:, np.newaxis, :]                        # [B, 1, C]
    scale = scale[..., np.newaxis, :]                          # [B, T|1, 1, C]

    err = np.abs(preds - targets) / (scale + eps)              # [B, T, H, C]
    if mask is None:
        return err.mean()

    # Per-series MASE over valid horizon steps, then equal-weight mean (M4 / NF style)
    n     = mask.sum(axis=2)                                   # [B, T, C]
    valid = n > 0
    score = (err * mask).sum(axis=2)[valid] / n[valid]
    return score.mean() if valid.any() else 0.0


def wql(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,
    eps:       float = 1e-8,
) -> float:
    """
    WQL = (2 / |Q|) * Σ_q Σ_h QL_q(y_{t+h}, ŷ^(q)_{t+h}) / Σ_h |y_{t+h}|

    Computed per series/window/channel (sum over H), then averaged.
    For the dataset-level version (sums over all series), use crps(..., scale=True).
    """
    errors = targets[..., np.newaxis] - preds                  # [B, T, H, C, Q]
    q      = np.array(quantiles, dtype=preds.dtype)
    loss   = np.maximum(q * errors, (q - 1) * errors).mean(-1) # [B, T, H, C]
    absy   = np.abs(targets)

    if mask is not None:
        loss, absy = loss * mask, absy * mask
        valid = mask.sum(axis=2) > 0                           # [B, T, C]
    else:
        valid = np.ones(loss.shape[:2] + loss.shape[3:], dtype=bool)

    score = 2 * loss.sum(axis=2) / (absy.sum(axis=2) + eps)   # [B, T, C]
    return score[valid].mean() if valid.any() else 0.0
