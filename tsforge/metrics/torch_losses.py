"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""
import torch

class LossFunction:
    def __init__(self, fn, quantiles=None):
        self.fn = fn
        self.quantiles = quantiles
        self.outputsize_multiplier = len(quantiles) if quantiles is not None else 1

    def __call__(self, *args, **kwargs):
        if self.quantiles is not None:
            kwargs.setdefault("quantiles", self.quantiles)
        return self.fn(*args, **kwargs)


def _mae_loss(
    preds:   torch.Tensor,             # [B, N, H, C]
    targets: torch.Tensor,             # [B, N, H, C]
    mask:    torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    err = torch.abs(preds - targets)
    if reduction == "none":
        return err
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()


def _mse_loss(
    preds:   torch.Tensor,             # [B, N, H, C]
    targets: torch.Tensor,             # [B, N, H, C]
    mask:    torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    err = (preds - targets) ** 2
    if reduction == "none":
        return err
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()


def _huber_loss(
    preds:   torch.Tensor,             # [B, N, H, C]
    targets: torch.Tensor,             # [B, N, H, C]
    mask:    torch.Tensor | None = None,
    delta:   float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    err  = torch.abs(preds - targets)
    loss = torch.where(err < delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    if reduction == "none":
        return loss
    if mask is not None:
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss.mean()


def _quantile_loss(
    preds:     torch.Tensor,           # [B, N, H, C, Q]
    targets:   torch.Tensor,           # [B, N, H, C]
    quantiles: list[float],
    mask:      torch.Tensor | None = None,  # [B, N, H, C]
    reduction: str = "mean",
) -> torch.Tensor:
    errors = targets.unsqueeze(-1) - preds                         # [B, N, H, C, Q]
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)
    loss = torch.max(q * errors, (q - 1) * errors)              # [B, N, H, C, Q]

    if reduction == "none":
        # Average over quantiles only -> [B, N, H, C], matching `mask`. The
        # masked path below divides by a denominator expanded Q times, so its
        # reduction over Q is also a mean; the two agree.
        return loss.mean(dim=-1)
    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss.mean()


def get_loss(name: str) -> LossFunction:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES.keys())}")
    return LOSSES[name]


LOSSES = {
    "mae":      LossFunction(_mae_loss),
    "mse":      LossFunction(_mse_loss),
    "huber":    LossFunction(_huber_loss),
    "quantile": LossFunction(_quantile_loss, quantiles=[0.1, 0.5, 0.9]),
}
