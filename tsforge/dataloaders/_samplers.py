import itertools

import numpy as np
from torch.utils.data import Sampler


class HorizonBatchSampler(Sampler):
    """
    Samples batches grouped by (horizon, is_multivariate) with weighted dataset mixing.
    Batches never mix univariate and multivariate datasets.

    Distributed behaviour
    ─────────────────────
    When world_size > 1 each rank receives a non-overlapping CONTIGUOUS slice
    of every group's pool. Contiguous (not strided) slices are used for better
    cache locality.

    Padding ensures all ranks see the same number of batches — required by
    DDP's barrier synchronisation (unequal batch counts cause deadlock).
    Padding draws from the front of the pool so no rank sees another's data.
    """

    def __init__(
        self,
        group_datasets,
        group_weights,
        global_offsets,
        batch_size,
        batch_mixing_strategy="concat",
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
        weighted_batches=True,
    ):
        self.group_datasets  = group_datasets
        self.group_weights   = group_weights
        self.global_offsets  = global_offsets
        self.batch_size      = batch_size
        self.batch_mixing_strategy = batch_mixing_strategy
        self.shuffle         = shuffle
        self.drop_last       = drop_last
        self.weighted_batches = weighted_batches
        self.seed            = seed
        self.rank            = rank
        self.world_size      = world_size
        self._epoch          = 0
        self.groups          = sorted(group_datasets.keys())

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _group_batches(self, group_key, rng):
        datasets = self.group_datasets[group_key]
        weights  = self.group_weights[group_key]
        offsets  = self.global_offsets[group_key]

        per_ds = []
        for ds, offset in zip(datasets, offsets):
            idxs = np.arange(len(ds)) + offset
            if self.shuffle:
                rng.shuffle(idxs)
            per_ds.append(idxs)

        if not self.shuffle and len(set(weights)) == 1:
            pool    = [idx for idxs in per_ds for idx in idxs.tolist()]
            bs      = self.batch_size
            batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
            if not self.drop_last and len(pool) % bs:
                batches.append(pool[-(len(pool) % bs):])
            return batches

        total     = sum(len(a) for a in per_ds)
        # weighted_batches=False ignores the per-dataset weights when building
        # batches, so batch composition is identical no matter what a given
        # condition does with `weight` — which is what lets a loss-weighting
        # run be compared against a sampling-weighting one. The weights stay
        # readable in the config for loss-side use.
        #
        # Note "uniform" here means equal slots per dataset, not slots
        # proportional to dataset size. With every source at weight 1.0 this
        # reproduces exactly the batches the weighted path already produced.
        w_arr     = (np.ones(len(per_ds), dtype=np.float64)
                     if not self.weighted_batches
                     else np.array(weights, dtype=np.float64))
        w_arr     = w_arr / w_arr.sum()
        slots_per = (w_arr * total).round().astype(int)
        slots_per[np.argmax(slots_per)] += total - slots_per.sum()

        pool     = []
        ds_iters = [itertools.cycle(a.tolist()) for a in per_ds]
        for slots, it in zip(slots_per, ds_iters):
            pool.extend(itertools.islice(it, int(slots)))
        if self.shuffle:
            rng.shuffle(pool)

        if self.world_size > 1:
            total_slots = self.batch_size * self.world_size
            pad  = (-len(pool)) % total_slots
            pool = pool + pool[:pad]
            rank_size = len(pool) // self.world_size
            pool = pool[self.rank * rank_size : (self.rank + 1) * rank_size]

        if self.drop_last:
            pool = pool[: (len(pool) // self.batch_size) * self.batch_size]

        bs      = self.batch_size
        batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
        if not self.drop_last and len(pool) % bs:
            batches.append(pool[-(len(pool) % bs):])
        return batches

    def __iter__(self):
        rng               = np.random.default_rng(self.seed + self._epoch)
        group_batch_lists = {g: self._group_batches(g, rng) for g in self.groups}

        if self.batch_mixing_strategy == "round_robin":
            iters  = {g: iter(b) for g, b in group_batch_lists.items()}
            active = list(self.groups)
            while active:
                exhausted = []
                for g in active:
                    b = next(iters[g], None)
                    if b is None:
                        exhausted.append(g)
                    else:
                        yield b
                for g in exhausted:
                    active.remove(g)
        else:
            all_batches = [b for bl in group_batch_lists.values() for b in bl]
            if self.shuffle:
                order       = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
            yield from all_batches

    def __len__(self):
        total = 0
        for g, datasets in self.group_datasets.items():
            n = sum(len(ds) for ds in datasets)
            if self.drop_last:
                n = (n // self.batch_size) * self.batch_size
            total += max(1, n // self.batch_size // max(1, self.world_size))
        return total
