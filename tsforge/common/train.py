import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from ..dataloaders.factory import DataLoaderFactory
from ._utils import build_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



def train(
    model,
    mcfg:        SimpleNamespace,
    train_loader,
    val_loaders: dict,
    device:      torch.device | None = None,
    seed:        int = 42,
    resume:      str | None = None,
    loss_fn:     Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Single-process training loop.

    Parameters
    ----------
    model        Any BaseModel subclass (not yet set up for training).
    mcfg         Model/training config namespace.
    train_loader DataLoader from factory.train_dataloader().
    val_loaders  Dict of DataLoaders from factory.val_dataloaders().
    device       Defaults to CUDA if available, else CPU.
    seed         Random seed.
    resume       Path to checkpoint to resume from (optional).

    Returns
    -------
    Final validation metrics dict.
    """
    optimizer = build_optimizer(model.parameters(), mcfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=mcfg.max_steps, eta_min=mcfg.learning_rate / 10,
    )
    model.setup_training(
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
        seed         = seed,
        loss_fn      = loss_fn,
    )
    model._rank       = 0
    model._world_size = 1

    if resume:
        model.load_train(resume)
        logger.info("Resumed from step %d", model.global_step)

    metrics = model.fit()
    logger.info("Training complete.")

    save_path = Path(mcfg.checkpoint_dir) / "final.pt"
    model.save_state(save_path)
    logger.info("Saved → %s", save_path)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Distributed training
# ─────────────────────────────────────────────────────────────────────────────

def _distributed_worker(
    rank:        int,
    world_size:  int,
    model,
    mcfg:        SimpleNamespace,
    factory:     DataLoaderFactory,
    backend:     str,
    seed:        int,
    resume:      str | None,
    master_addr: str,
    master_port: str,
) -> None:
    """
    Per-rank worker.

    Data loading
    ─────────────
    Training:   Each rank gets a non-overlapping contiguous slice of the
                pool via HorizonBatchSampler(rank=rank, world_size=world_size).
                Padding ensures identical batch counts across ranks (DDP barrier
                requirement).

    Validation: Rank 0 only runs validation; all other ranks call dist.barrier()
                and wait. This keeps metrics exact (no padding/all-reduce) and
                val DataLoaders are never constructed on non-zero ranks.

    Checkpoint: Rank 0 only saves the final checkpoint.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    logger.info("Rank %d/%d initialised on %s", rank, world_size, device)

    # ── Per-rank train DataLoader ─────────────────────────────────────────────
    # Rebuild sharded datasets with correct rank/world_size so each rank
    # loads only its assigned shard files. Non-sharded datasets are unaffected.
    factory.rebuild_for_rank(rank, world_size)
    train_loader = factory.train_dataloader(
        distributed=True, rank=rank, world_size=world_size,
    )

    # ── Val DataLoader — full val set, every rank ─────────────────────────────
    # No DistributedSampler — each rank evaluates the complete val set.
    # Losses are averaged across ranks via all_reduce in _make_distributed_validate.
    val_loaders = factory.val_dataloaders()

    # ── DDP wrap ─────────────────────────────────────────────────────────────
    model = model.to(device)
    ddp_model   = DDP(model, device_ids=[rank], find_unused_parameters=False)
    inner_model = ddp_model.module

    optimizer = build_optimizer(ddp_model.parameters(), mcfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=mcfg.max_steps, eta_min=mcfg.learning_rate / 10,
    )
    inner_model.setup_training(
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
        seed         = seed + rank,      # different seed per rank
    )
    # Expose DDP wrapper so train_step gradients are all-reduced correctly
    #inner_model._ddp_model = ddp_model
    object.__setattr__(inner_model, '_ddp_model', ddp_model)
    # Tell the model its rank/world_size so fit() can all-reduce train loss
    inner_model._rank       = rank
    inner_model._world_size = world_size

    # Replace validate() with distributed all-reduce version
    _make_distributed_validate(inner_model, rank, world_size)

    if resume:
        inner_model.load_train(resume)
        if rank == 0:
            logger.info("Resumed from step %d", inner_model.global_step)

    inner_model.fit()

    if rank == 0:
        save_path = Path(mcfg.checkpoint_dir) / "final.pt"
        inner_model.save_state(save_path)
        logger.info("Training complete. Saved → %s", save_path)

    dist.destroy_process_group()


def _make_distributed_validate(model, rank: int, world_size: int):
    """
    Replace model.validate() with a distributed version.

    Every rank evaluates the FULL val set independently with fcd_samples=-1
    (all valid FCD windows). Each rank has a different model state because
    it trained on a different time partition, so we want the global metric
    to reflect all model states.

    Since every rank sees the same val windows, a simple mean of per-rank
    losses is exact — no padding correction needed (no DistributedSampler).

        global_loss = all_reduce(rank_loss, SUM) / world_size
    """
    original_validate = model.validate

    def _distributed_validate():
        # Each rank runs full val independently — same data, different weights
        results = original_validate()

        # All-reduce: average rank losses into a single global metric
        for name, metrics in results.items():
            for key, val in metrics.items():
                t = torch.tensor(val, device=model.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                metrics[key] = (t / world_size).item()

        return results

    model.validate = _distributed_validate


def train_distributed(
    model,
    mcfg:        SimpleNamespace,
    factory:     DataLoaderFactory,
    backend:     str = "nccl",
    seed:        int = 42,
    resume:      str | None = None,
    use_spawn:   bool = False,
    world_size:  int | None = None,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """
    Launch distributed DDP training.

    use_spawn=False (torchrun, default)
    ────────────────────────────────────
    torchrun sets LOCAL_RANK / WORLD_SIZE automatically.
    Call this once per process — torchrun handles spawning.

        torchrun --nproc_per_node=4 train_script.py

    use_spawn=True (mp.spawn, programmatic)
    ────────────────────────────────────────
    This function spawns world_size child processes itself.
    Useful for single-machine experiments without torchrun.

        train_distributed(model, mcfg, factory, use_spawn=True, world_size=4)

    After train_distributed returns, call eval_test() normally —
    it always runs on a single GPU and needs no distributed context.
    """
    if use_spawn:
        if world_size is None:
            world_size = torch.cuda.device_count()
            if world_size == 0:
                raise RuntimeError(
                    "No CUDA devices found. Set backend='gloo' and world_size "
                    "explicitly for CPU-only distributed training."
                )
        logger.info("Spawning %d processes (mp.spawn) …", world_size)
        mp.spawn(
            _distributed_worker,
            args=(
                world_size, model, mcfg, factory,
                backend, seed, resume, master_addr, master_port,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        rank       = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        logger.info("torchrun mode — rank %d / %d", rank, world_size)
        _distributed_worker(
            rank        = rank,
            world_size  = world_size,
            model       = model,
            mcfg        = mcfg,
            factory     = factory,
            backend     = backend,
            seed        = seed,
            resume      = resume,
            master_addr = os.environ.get("MASTER_ADDR", master_addr),
            master_port = os.environ.get("MASTER_PORT", master_port),
        )



def eval_test(
    model,
    factory:  DataLoaderFactory,
    device:   Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """
    Run inference on the full test set on a single GPU.

    Always called outside of any distributed context — either after
    train() or after train_distributed() has returned and
    destroy_process_group() has been called.

    Parameters
    ----------
    model    Trained BaseModel (DDP wrapper is unwrapped automatically).
    factory  DataLoaderFactory to obtain test loaders.
    device   Defaults to CUDA:0 if available, else CPU.

    Returns
    -------
    {dataset_name: predictions_tensor}
    """
    if dist.is_available() and dist.is_initialized():
        raise RuntimeError(
            "eval_test() must be called after dist.destroy_process_group(). "
            "Do not call it inside a distributed worker."
        )

    inner = getattr(model, "module", model)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loaders = factory.test_dataloaders()
    
    results = {}
    for loader in test_loaders.values():
        loader_results = inner.predict(loader, device=device)
        results.update(loader_results)
    return results
