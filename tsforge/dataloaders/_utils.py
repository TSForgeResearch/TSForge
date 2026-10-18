import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from omegaconf import OmegaConf, DictConfig


def _to_cfg(entry):
    if isinstance(entry, DictConfig):
        return entry
    return OmegaConf.create(entry)


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format '{p.suffix}'. Use .parquet or .csv.")

    if "available_mask" not in df.columns:
        df["available_mask"] = 1.0
        
    return df


def _split_df(
    df: pd.DataFrame,
    val_size: int,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])
    return _split_per_series(df, val_size, test_size)


def _split_per_series(df, val_size, test_size):
    """
    Per series: take the last test_size rows as test, whatever's left (up to
    val_size) as val, and anything remaining as train. Series too short even
    for test_size are dropped outright — train/val naturally end up empty
    (rather than dropped) when there's leftover for val but none for train,
    or leftover for neither; downstream padding (see PerSeriesDataset) treats
    empty/short train and val the same as any other short series.
    """
    lengths = df.groupby("unique_id")["ds"].transform("count")
    insufficient = df.loc[lengths < test_size, "unique_id"].unique()
    if len(insufficient) > 0:
        warnings.warn(
            f"Dropping {len(insufficient)} series with fewer than "
            f"test_size ({test_size}) timesteps: {sorted(insufficient)}"
        )
    df = df[lengths >= test_size]

    def label(g):
        n = len(g)
        leftover = n - test_size
        val_n = min(val_size, leftover)
        train_n = leftover - val_n
        return g.assign(_split=(
            ["train"] * train_n
            + ["val"]  * val_n
            + ["test"] * test_size
        ))
    df = df.groupby("unique_id", group_keys=False).apply(label)
    return (
        df[df["_split"] == "train"].drop(columns="_split"),
        df[df["_split"] == "val"].drop(columns="_split"),
        df[df["_split"] == "test"].drop(columns="_split"),
    )


def _pivot_col(df: pd.DataFrame, col: str, channel_ids: List[str]) -> np.ndarray:
    return (
        df.pivot(index="ds", columns="unique_id", values=col)
        .loc[:, channel_ids]
        .values
        .astype(np.float32)
    )


def _pivot_to_arrays(
    df: pd.DataFrame,
    hist_exog_cols: List[str],
    futr_exog_cols: List[str],
    stat_exog_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    channel_ids = sorted(df["unique_id"].unique().tolist())

    lengths = df.groupby("unique_id")["ds"].count()
    if lengths.nunique() > 1:
        warnings.warn(f"Unequal series lengths {lengths.to_dict()} — forward-filling to align.")
        mi = pd.MultiIndex.from_product(
            [channel_ids, df["ds"].drop_duplicates().sort_values()],
            names=["unique_id", "ds"],
        )
        df = (
            df.set_index(["unique_id", "ds"])
            .reindex(mi)
            .groupby("unique_id")
            .ffill()
            .reset_index()
        )

    T, C           = df["ds"].nunique(), len(channel_ids)
    y              = _pivot_col(df, "y", channel_ids)
    hist           = np.stack([_pivot_col(df, c, channel_ids) for c in hist_exog_cols], axis=-1) if hist_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    futr           = np.stack([_pivot_col(df, c, channel_ids) for c in futr_exog_cols], axis=-1) if futr_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    stat           = df.groupby("unique_id", sort=False)[stat_exog_cols].first().loc[channel_ids].values.astype(np.float32) if stat_exog_cols else np.zeros((C, 0), dtype=np.float32)
    available_mask = _pivot_col(df, "available_mask", channel_ids)
    loss_mask      = _pivot_col(df, "loss_mask", channel_ids)

    return y, hist, futr, stat, channel_ids, available_mask, loss_mask


def _series_arrays(
    df: pd.DataFrame,
    hist_exog_cols: List[str],
    futr_exog_cols: List[str],
    stat_exog_cols: List[str],
) -> List[dict]:
    """
    Split a long-format frame into one independent array-set per unique_id.

    Unlike _pivot_to_arrays, this never reindexes series onto a shared time
    axis — these are independent univariate examples with unrelated native
    timelines, not channels of one multivariate signal, so there's nothing
    to align. Each series keeps its own native length; batching later pads
    to a common length per-batch (see DataLoaderFactory._full_series_collate_fn).
    """
    df = df.sort_values(["unique_id", "ds"])
    out = []
    for uid, g in df.groupby("unique_id", sort=True):
        out.append({
            "channel_id":     uid,
            "y":              g["y"].to_numpy(dtype=np.float32),
            "hist":           g[hist_exog_cols].to_numpy(dtype=np.float32) if hist_exog_cols else np.zeros((len(g), 0), dtype=np.float32),
            "futr":           g[futr_exog_cols].to_numpy(dtype=np.float32) if futr_exog_cols else np.zeros((len(g), 0), dtype=np.float32),
            "stat":           g[stat_exog_cols].iloc[0].to_numpy(dtype=np.float32) if stat_exog_cols else np.zeros((0,), dtype=np.float32),
            "available_mask": g["available_mask"].to_numpy(dtype=np.float32),
            "loss_mask":      g["loss_mask"].to_numpy(dtype=np.float32),
        })
    return out


def _extend_with_next_split(df: pd.DataFrame, next_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Append the first H-1 rows of next_df per series to df, with available_mask=0.
    These act as a buffer so windows near the split boundary can form without
    their horizons overflowing, while contributing zero loss.
    Only applied when horizon > 1 and next_df is non-empty.
    """
    if next_df is None or len(next_df) == 0 or horizon <= 1:
        return df
    extension = (
        next_df.groupby("unique_id", sort=False)
        # H-1, not H: an H-th row only adds one FCD whose whole horizon is
        # masked, i.e. a window that costs compute but contributes zero loss.
        .head(horizon - 1)
        .copy()
    )
    extension["available_mask"] = 0.0
    extension["loss_mask"] = 0.0
    return (
        pd.concat([df, extension])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def _pad_left(t: Tensor, target_len: int) -> Tensor:
    """Left-pad tensor along dim 0 (time) with zeros."""
    pad = target_len - t.shape[0]
    if pad == 0:
        return t
    return F.pad(t, [0] * (2 * (t.ndim - 1)) + [pad, 0])
