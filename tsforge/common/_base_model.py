import logging
import random
import time
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

from ._utils import CheckpointManager, EarlyStopper, build_optimizer, mix_seed
from ..dataloaders._forking_sequences import ForkingSequences
from ..metrics.torch_losses import get_loss
from ..scalers.torch_scalers import Scaler

logger = logging.getLogger(__name__)


class _InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter  = iter(loader)
        self._epoch = 0

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            sampler = (
                getattr(self.loader, "batch_sampler", None)
                or getattr(self.loader, "sampler", None)
            )
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            return next(self._iter)
        

class _BaseLoss(nn.Module):
    def __init__(self, loss_fn, scaler, reduction: str = "per_series"):
        super().__init__()
        self.loss_fn   = loss_fn
        self.scaler    = scaler
        self.reduction = reduction

    def _compute(self, batch):
        y, preds = batch["outsample_y"], batch["preds"]
        B, T, H, C = y.shape
        y = y.reshape(B * T, H, C)
        preds = preds.reshape(B * T, H, C, -1)
        mask = batch["outsample_mask"].reshape(B * T, H, C).float()

        if self.reduction == "flat":
            return self.loss_fn(preds=preds, targets=y, mask=mask)

        # ── per_series ───────────────────────────────────────────────────────
        # The flat mean pools every window of every series into one bag, so a
        # series influences the gradient in proportion to how many live
        # elements it contributed. Measured on M-competition yearly, that
        # spread is 6x at fcd_samples=1 and 279x at fcd_samples=96 — i.e. the
        # flat mean silently weights series by length.
        #
        # Averaging within each series first, then across series, makes every
        # series count once regardless of length. That also makes a per-series
        # weight applied downstream mean what it says: under the flat mean the
        # effective weight is w_i * n_i, here it is w_i.
        per_elem = self.loss_fn(preds=preds, targets=y, mask=mask, reduction="none")

        # Blocked eval folds blocks into the batch dim (b-major, block-minor),
        # so regroup them into their original series before reducing.
        n_blocks = int(batch.get("n_blocks", 1))
        B_true   = B // n_blocks

        num   = (per_elem * mask).reshape(B_true, -1).sum(dim=1)
        den   = mask.reshape(B_true, -1).sum(dim=1)
        ell   = num / den.clamp(min=1)                      # per-series mean
        valid = (den > 0).to(ell.dtype)

        # A series with no live window in this batch has no defined loss, so it
        # is dropped from the average rather than counted as zero.
        return (ell * valid).sum() / valid.sum().clamp(min=1)

class DenormSpaceLoss(_BaseLoss):
    """Denorms preds before computing loss."""
    def forward(self, batch):
        batch = self.scaler(batch, norm_type='denorm')
        return batch, self._compute(batch)

class NormSpaceLoss(_BaseLoss):
    """Keeps preds in norm space, norms targets to match."""
    def forward(self, batch):
        batch = self.scaler(batch, norm_type='norm_targets')
        return batch, self._compute(batch)


class BaseModel(nn.Module):
    """
    nn.Module base that also owns the step-based training loop.

    Subclass responsibilities
    ─────────────────────────
    __init__(self, ...)
        Call super().__init__(). Build architecture only — no training args.

    forward(self, batch) -> Tensor
        Implement the forward pass. batch is the output of forking-sequences.

    compute_loss(self, pred, batch) -> Tensor   [optional]
        Default: MSE against outsample_y. Override for custom losses.
    """

    #: Whether this architecture reads a fixed context window. Windowed models
    #: default norm_window_size to context_len, so normalization spans the same
    #: history the model attends to. Models that carry unbounded state
    #: (recurrent, conv) default to -1 (causal cumulative) instead. Subclasses
    #: override; see Transformer.
    WINDOWED_CONTEXT = False

    def __init__(self, config):
        super().__init__()
        self._training_ready = False
        self._rank = 0
        self._world_size = 1

        self.scaler = Scaler(
            scaler_type=config.scaler_type,
            stride=config.stride,
            eps=1e-5
        )
        
        # Normalization window. Unspecified → follow the architecture: a
        # windowed model normalizes over the same span it attends to
        # (context_len); a model with unbounded state uses causal cumulative
        # stats (-1). Note context_len is always set by
        # DataLoaderFactory._resolve_context_len, including for recurrent
        # models, so WINDOWED_CONTEXT — not "is context_len == -1" — is what
        # distinguishes them.
        norm_window_size = getattr(config, "norm_window_size", None)
        if norm_window_size is None:
            ctx = getattr(config, "context_len", -1)
            norm_window_size = ctx if (self.WINDOWED_CONTEXT and ctx != -1) else -1
        self.norm_window_size = norm_window_size

        self.fcd_samples = config.fcd_samples
        self._fork_sequences_train = ForkingSequences(
            context_len = config.context_len,
            fcd_samples = config.fcd_samples,
            patch_len = config.patch_len,
            stride = config.stride,
            fcd_sampler = config.fcd_sampler,
            norm_window_size = norm_window_size,
            fcd_seed = getattr(config, "seed", 0),
        )
        # Eval block size — exactly two sanctioned modes:
        #   None (default) → mirror fcd_samples, so every eval block has the
        #                    same geometry the model saw at train time.
        #   -1             → the whole series in one shared encoder pass.
        #                    Fastest, but a window late in the series draws on
        #                    more history than any training block held.
        # Any other value is rejected: it would neither match training nor be
        # maximally efficient, just wrong by an unquantified amount.
        fcd_samples_eval = getattr(config, "fcd_samples_eval", None)
        if fcd_samples_eval is None:
            fcd_samples_eval = config.fcd_samples

        if fcd_samples_eval != -1:
            if not isinstance(fcd_samples_eval, int) or fcd_samples_eval < 1:
                raise ValueError(
                    f"fcd_samples_eval must be null, -1, or a positive int equal "
                    f"to fcd_samples ({config.fcd_samples}); got {fcd_samples_eval!r}."
                )
            if fcd_samples_eval != config.fcd_samples:
                raise ValueError(
                    f"fcd_samples_eval={fcd_samples_eval} does not match "
                    f"fcd_samples={config.fcd_samples}. Eval blocks must have the "
                    "same geometry as the training blocks the model was fit on. "
                    "Use null (mirror training) or -1 (whole series in one pass, "
                    "faster but sees more history than training did)."
                )

        self.fcd_samples_eval = fcd_samples_eval

        # Only a mismatch is worth warning about: when training itself used
        # fcd_samples=-1 (all FCDs, whole series in one pass), evaluating with
        # -1 IS the exact match, not a deviation from it.
        if fcd_samples_eval == -1 and config.fcd_samples != -1:
            logger.warning(
                f"fcd_samples_eval=-1 with fcd_samples={config.fcd_samples}: "
                "evaluating with the whole series in one shared encoder pass. "
                "Windows late in a series draw on more history than any "
                "training block contained, so forecasts will not exactly "
                "reproduce the model's training-time behavior. Use null for an "
                "exact match."
            )

        self._fork_sequences_eval = ForkingSequences(
            context_len = config.context_len,
            fcd_samples = fcd_samples_eval,
            patch_len = config.patch_len,
            stride = config.stride,
            norm_window_size = norm_window_size,
            blocked = fcd_samples_eval != -1,
        )

        loss_fn = get_loss(config.loss)
        if hasattr(config, "quantiles") and config.quantiles:
            loss_fn.quantiles = list(config.quantiles)
            loss_fn.outputsize_multiplier = len(config.quantiles)

        self.loss_fn = loss_fn

        reduction = getattr(config, "loss_reduction", "per_series")
        if reduction not in ("flat", "per_series"):
            raise ValueError(
                f"loss_reduction must be 'flat' or 'per_series', got {reduction!r}."
            )
        self.loss_reduction = reduction

        if config.loss_space == 'norm':
            self.compute_loss = NormSpaceLoss(
                loss_fn=loss_fn, scaler=self.scaler, reduction=reduction)
        elif config.loss_space == 'denorm':
            self.compute_loss = DenormSpaceLoss(
                loss_fn=loss_fn, scaler=self.scaler, reduction=reduction)
        else:
            raise Exception('Loss space not recognized.')

        self.train_losses = []
        self.val_losses = []

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        ...

    def setup_training(
        self,
        mcfg,
        train_loader:   DataLoader,
        val_loaders:    Dict[str, DataLoader],
        optimizer:      Optional[torch.optim.Optimizer] = None,
        scheduler       = None,
        loss_fn:        Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        device:         Optional[torch.device] = None,
        seed:           int = 42,
    ) -> "BaseModel":
        self.mcfg         = mcfg
        self.train_loader = train_loader
        self.val_loaders  = val_loaders
        self.scheduler    = scheduler
        self.seed         = seed
        self.global_step  = 0

        # Single source of truth for FCD window sampling: it always derives
        # from the training seed, so one seed defines an experiment. There is
        # deliberately no separate fcd_seed config key — it would be silently
        # overwritten here, and two seeds is one more thing to get out of sync
        # between conditions.
        #
        # `seed` already carries the +rank offset applied by
        # _distributed_worker, so ranks draw different windows — they hold
        # different data slices. What matters for a controlled comparison is
        # that rank k sees the same windows in every condition, which it does.
        self._fork_sequences_train.fcd_seed = seed

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)

        # A custom loss passed to train()/setup_training() used to be accepted
        # and then silently dropped — the loss stayed whatever config.loss built
        # in __init__, so `train(loss_fn=...)` was a no-op. Wire it up.
        if loss_fn is not None:
            self.loss_fn = loss_fn
            space = getattr(mcfg, "loss_space", "denorm")
            reduction = getattr(mcfg, "loss_reduction", self.loss_reduction)
            if space == "norm":
                self.compute_loss = NormSpaceLoss(
                    loss_fn=loss_fn, scaler=self.scaler, reduction=reduction)
            elif space == "denorm":
                self.compute_loss = DenormSpaceLoss(
                    loss_fn=loss_fn, scaler=self.scaler, reduction=reduction)
            else:
                raise ValueError(f"Loss space {space!r} not recognized.")
            self.compute_loss.to(self.device)

        self.optimizer = optimizer or build_optimizer(self.parameters(), mcfg)

        self.early_stopper = EarlyStopper(
            patience = mcfg.early_stopping_patience,
            mode = "min",
        )
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir  = mcfg.checkpoint_dir,
            checkpoint_step = getattr(mcfg, "checkpoint_step", 1000),
        )
        self._training_ready = True
        return self

    def _assert_training_ready(self):
        if not self._training_ready:
            raise RuntimeError(
                "Call model.setup_training(mcfg, train_loader, val_loaders) "
                "before fit() / train_step() / validate()."
            )

    # ── batch preparation ────────────────────────────────────────────────────

    def _to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }


    def _prepare_batch(self, raw_batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raw_batch = self._to_device(raw_batch)
        horizon_override = getattr(self.mcfg, "horizon_override", None)
        horizon = int(horizon_override) if horizon_override else int(raw_batch["horizon"][0].item())
        if self.training:
            fcd_samples = self._get_fcd_samples()
            return self._fork_sequences_train(
                raw_batch, horizon, fcd_samples=fcd_samples, step=self.global_step,
            )
        return self._fork_sequences_eval(raw_batch, horizon, fcd_samples=self.fcd_samples_eval)


    def _get_fcd_samples(self):
        self._assert_training_ready()
        return self.fcd_samples # @ WP TODO: curriculum learning


    def train_step(self, raw_batch: Dict[str, Tensor]) -> float:
        self._assert_training_ready()
        self.train()

        # Re-key the global RNG stream to this step. Dropout draws from it, so
        # without this its masks depend on how much RNG everything else
        # consumed — a loss function that draws a single random number shifts
        # every subsequent dropout mask. Keying on (seed, global_step) makes
        # dropout a pure function of the step, so two conditions differing
        # only in their loss see identical masks.
        #
        # FCD window sampling does not rely on this (it has its own generator,
        # keyed the same way), but the two now behave consistently: everything
        # stochastic in a step is determined by the step number alone.
        torch.manual_seed(mix_seed(self.seed, self.global_step))

        batch = self._prepare_batch(raw_batch)

        self.optimizer.zero_grad(set_to_none=True)
        fwd  = self.__dict__.get('_ddp_model', self)

        batch = self.scaler(batch, norm_type='norm')
        batch["preds"] = fwd(batch)
        batch, loss = self.compute_loss(batch=batch)  # handles denorm internally
        loss.backward()
    
        nn.utils.clip_grad_norm_(self.parameters(), self.mcfg.gradient_clip_val)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def val_step(self, raw_batch: Dict[str, Tensor]) -> float:
        self._assert_training_ready()
        self.eval()
        batch = self._prepare_batch(raw_batch)
        batch = self.scaler(batch, norm_type='norm')
        batch["preds"] = self(batch)
        _, loss = self.compute_loss(batch)  # handles denorm internally
        return loss.item()

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict[str, float]]:
        self._assert_training_ready()
        self.eval()
        results: Dict[str, Dict[str, float]] = {}
        for name, loader in self.val_loaders.items():
            total, n = 0.0, 0
            for raw_batch in loader:
                total += self.val_step(raw_batch)
                n += 1
            results[name] = {"loss": total / n if n > 0 else float("nan")}
        return results

    @torch.no_grad()
    def predict_step(self, raw_batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Single batch inference. Returns pred, targets, outsample_mask for that batch."""
        self.eval()
        raw_batch = self._to_device(raw_batch)
        batch     = self._prepare_batch(raw_batch)

        batch = self.scaler(batch, norm_type='norm')
        preds = self(batch)
        batch["preds"] = preds
        batch = self.scaler(batch, norm_type='denorm')

        preds = batch["preds"].cpu().float()       # [B, n_fcds, H, C_out]
        targets = batch["outsample_y"].cpu()      # [B, n_fcds, H, C]
        outsample_mask = batch.get("outsample_mask")
        if outsample_mask is not None:
            outsample_mask = outsample_mask.cpu()

        # Blocked eval folds blocks into the batch dim; restore the original
        # [B, n_fcds, ...] layout (dropping the trailing block's padded FCDs)
        # so predict() sees exactly the same shapes either way.
        n_blocks = batch.get("n_blocks", 1)
        if n_blocks > 1:
            n_fcds = batch["n_fcds"]

            def _unfold_blocks(t):
                if t is None:
                    return None
                B = t.shape[0] // n_blocks
                return t.reshape(B, n_blocks * t.shape[1], *t.shape[2:])[:, :n_fcds]

            preds = _unfold_blocks(preds)
            targets = _unfold_blocks(targets)
            outsample_mask = _unfold_blocks(outsample_mask)

        return dict(preds=preds, targets=targets, outsample_mask=outsample_mask)

    @torch.no_grad()
    def predict(self, loader, device=None):
        self.eval()
        if device is not None:
            self.to(device)

        results = {}
        for raw_batch in tqdm(loader, desc="Predicting"):
            step            = self.predict_step(raw_batch)
            pred            = step["preds"]
            targets         = step["targets"]
            outsample_mask  = step["outsample_mask"]
            dataset_names   = raw_batch.get("dataset_name",   ["unknown"] * pred.shape[0])
            channel_ids     = raw_batch.get("channel_ids",    [None]      * pred.shape[0])
            is_multivariate = raw_batch.get("is_multivariate", False)

            for b in range(pred.shape[0]):
                name   = dataset_names[b]
                ids    = channel_ids[b]
                n_real = len(ids) if ids is not None else pred.shape[3]

                p = pred[b, :, :, :n_real, :]
                t = targets[b, :, :, :n_real]
                m = outsample_mask[b, :, :, :n_real] if outsample_mask is not None else None

                if is_multivariate:
                    # one dataset = one tensor set spanning all its channels (C axis).
                    bucket = results.setdefault(
                        name, {"channel_ids": ids, "preds": [], "targets": [], "outsample_mask": []}
                    )
                    bucket["preds"].append(p)
                    bucket["targets"].append(t)
                    if m is not None:
                        bucket["outsample_mask"].append(m)
                else:
                    # per-series: exactly one channel per sample — squeeze it out
                    # and nest by unique_id instead of flattening series together.
                    uid   = ids[0] if ids else name
                    entry = results.setdefault(name, {}).setdefault(
                        uid, {"preds": [], "targets": [], "outsample_mask": []}
                    )
                    entry["preds"].append(p[:, :, 0, :])   # [windows, horizon, quantiles]
                    entry["targets"].append(t[:, :, 0])    # [windows, horizon]
                    if m is not None:
                        entry["outsample_mask"].append(m[:, :, 0])

        for name, d in results.items():
            if "preds" in d:
                d["preds"]          = torch.cat(d["preds"],          dim=0)
                d["targets"]        = torch.cat(d["targets"],        dim=0)
                d["outsample_mask"] = torch.cat(d["outsample_mask"], dim=0) if d["outsample_mask"] else None
            else:
                for entry in d.values():
                    entry["preds"]          = torch.cat(entry["preds"],          dim=0)
                    entry["targets"]        = torch.cat(entry["targets"],        dim=0)
                    entry["outsample_mask"] = torch.cat(entry["outsample_mask"], dim=0) if entry["outsample_mask"] else None

        return results

    def _log_val_metrics(self, val_metrics: Dict[str, Dict[str, float]]):
        for name, metrics in val_metrics.items():
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info("  [val/%s] step %d  %s", name, self.global_step, parts)

    def fit(self) -> Dict[str, Dict[str, float]]:
        self._assert_training_ready()

        # Deliberate second reseed, not redundant with set_determinism().
        # Model construction consumes a variable amount of the global stream
        # depending on architecture; this resets it to a known position so
        # step 0 looks identical across conditions. Do not remove.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger.info(
            "Training — max_steps=%d  val_every=%d  patience=%d  device=%s",
            self.mcfg.max_steps,
            self.mcfg.val_check_interval,
            self.mcfg.early_stopping_patience,
            self.device,
        )

        primary    = next(iter(self.val_loaders)) if self.val_loaders else None
        train_iter = _InfiniteLoader(self.train_loader)
        best_val   = float("inf")
        final_metrics: Dict[str, Dict[str, float]] = {}
        t0 = time.time()

        pbar = tqdm(total=self.mcfg.max_steps, initial=self.global_step, desc="Training")
        while self.global_step < self.mcfg.max_steps:
            train_loss = self.train_step(next(train_iter))
            self.train_losses.append((self.global_step, train_loss))  # (step, loss)
            self.global_step += 1

            if self._world_size > 1:
                t = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                train_loss = (t / self._world_size).item()

            self.ckpt_manager.step(
                self.global_step, self,
                optimizer = self.optimizer.state_dict(),
                loss      = train_loss,
            )

            if self.global_step % self.mcfg.val_check_interval == 0:
                val_metrics = self.validate()
                monitor_val = val_metrics[primary].get("loss", float("nan"))
                self.val_losses.append((self.global_step, monitor_val))  # (step, loss)
                final_metrics = val_metrics
                self._log_val_metrics(val_metrics)

                # Keep the best-val weights alongside the final ones. Early
                # stopping returns the model `patience` checks PAST its best,
                # so final.pt is systematically post-peak. Saving both lets the
                # choice be made after the run rather than being forced here.
                # Note eval_test() runs on the in-memory (final) model — load
                # best.pt explicitly if that is the one you want to evaluate.
                if monitor_val < best_val:          # NaN never compares True
                    best_val = monitor_val
                    self.save_state(self.ckpt_manager.checkpoint_dir / "best.pt")

                if primary and primary in val_metrics:
                    monitor_val = val_metrics[primary].get("loss", float("nan"))
                    pbar.set_postfix({
                        "train": f"{train_loss:.4f}",
                        "val":   f"{monitor_val:.4f}",
                    })
                    if self.early_stopper.step(monitor_val):
                        logger.info(
                            "Early stopping at step %d (best=%.4f)",
                            self.global_step, self.early_stopper.best,
                        )
                        break
            else:
                pbar.set_postfix({"train": f"{train_loss:.4f}"})

            pbar.update(1)

        pbar.close()

        final_path = self.ckpt_manager.checkpoint_dir / "final.pt"
        self.save_state(final_path)
        logger.info("Saved → %s", final_path)

        return final_metrics

    def save_state(self, path: str | Path):
        self._assert_training_ready()
        torch.save({
            "global_step":   self.global_step,
            "model":         self.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "valid_losses":   self.val_losses,
            "early_stopper": (
                self.early_stopper.state_dict()
                if hasattr(self.early_stopper, "state_dict") else
                vars(self.early_stopper)   # fallback: save __dict__ directly
            ),
            "scheduler":     self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        logger.info("Trainer state saved → %s", path)

    def load_train(self, path: str | Path):
        self._assert_training_ready()
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("early_stopper"):
            if hasattr(self.early_stopper, "load_state_dict"):
                self.early_stopper.load_state_dict(ckpt["early_stopper"])
            else:
                self.early_stopper.__dict__.update(ckpt["early_stopper"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info("Trainer state loaded ← %s  (step=%d)", path, self.global_step)

    @staticmethod
    def load_weights(
        path: str | Path,
        model: nn.Module,
        map_location: str = "cpu",
    ) -> nn.Module:
        """Load only model weights — no training state needed."""
        ckpt  = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        logger.info("Model weights loaded ← %s", path)
        return model

    def setup_inference(
        self,
        mcfg,
        device: Optional[torch.device] = None,
    ) -> "BaseModel":
        self.mcfg   = mcfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)
        return self
