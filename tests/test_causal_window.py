"""
Regression tests for capping causal attention to context_len worth of
patches, regardless of fcd_samples or how long the shared encoder pass is.

Before this fix, `_make_causal_token_mask` used an unrestricted causal
mask, so a forking-sequences block (fcd_samples > 1 at train) or the whole
test series (eval, which always unfolds every valid window from one shared
pass) let later windows attend arbitrarily further back than context_len —
a train/inference mismatch. `causal_len_token_num` bands the mask so every
window sees exactly context_len worth of history, independent of position.

No pytest dependency — run directly:
    python tests/test_causal_window.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsforge.common._modules import _make_causal_token_mask
from tsforge.dataloaders._forking_sequences import ForkingSequences
from tsforge.encoders.tst_encoder import TSTEncoder


def _allowed_keys(mask, i):
    return set(j for j in range(mask.shape[-1]) if mask[0, 0, 0, i, j] == 1)


def test_banded_mask_matches_sliding_window():
    """Query i may only attend to keys [i - causal_len_token_num + 1, i]."""
    B, C, L, W = 1, 1, 10, 4
    key_padding_mask = torch.ones(B, C, L)

    banded = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"), causal_len_token_num=W)

    for i in range(L):
        expected = set(range(max(0, i - W + 1), i + 1))
        assert _allowed_keys(banded, i) == expected, f"row {i}: {_allowed_keys(banded, i)} != {expected}"


def test_none_reproduces_unrestricted_causal():
    """causal_len_token_num=None must be identical to the pre-fix behavior."""
    B, C, L = 1, 1, 10
    key_padding_mask = torch.ones(B, C, L)

    unrestricted = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"))
    explicit_none = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"), causal_len_token_num=None)
    assert torch.equal(unrestricted, explicit_none)

    for i in range(L):
        assert _allowed_keys(unrestricted, i) == set(range(0, i + 1))


def test_window_wider_than_sequence_is_a_noop():
    """A band wider than the sequence itself must not change anything."""
    B, C, L = 1, 1, 5
    key_padding_mask = torch.ones(B, C, L)

    unrestricted = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"))
    wide_band = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"), causal_len_token_num=L + 10)
    assert torch.equal(unrestricted, wide_band)


def test_padding_still_respected_within_the_band():
    """A padded key inside the band must still be blocked."""
    B, C, L, W = 1, 1, 6, 3
    key_padding_mask = torch.ones(B, C, L)
    key_padding_mask[0, 0, 2] = 0  # patch 2 is padding/missing

    banded = _make_causal_token_mask(key_padding_mask, device=torch.device("cpu"), causal_len_token_num=W)
    for i in range(L):
        expected = set(range(max(0, i - W + 1), i + 1)) - {2}
        assert _allowed_keys(banded, i) == expected


def _make_encoder_config(context_len, patch_len=2, stride=2, hidden_size=8, n_heads=2):
    return SimpleNamespace(
        mica_mixer_type=  "none",
        layerwise_beta=   False,
        n_layers=         2,
        res_attention=    False,
        dropout=          0.0,
        norm=             "layer",
        hidden_size=      hidden_size,
        linear_hidden_size=16,
        activation=       "gelu",
        pre_norm=         True,
        store_attn=       False,
        n_heads=          n_heads,
        d_k=              None,
        d_v=              None,
        qkv_bias=         False,
        proj_dropout=     0.0,
        attn_dropout=     0.0,
        context_len=      context_len,
        patch_len=        patch_len,
        stride=           stride,
    )


def test_causal_len_token_num_derivation():
    cfg = _make_encoder_config(context_len=8, patch_len=2, stride=2)
    enc = TSTEncoder(cfg)
    assert enc.causal_len_token_num == 4  # (8 - 2) / 2 + 1

    cfg_full = _make_encoder_config(context_len=-1, patch_len=2, stride=2)
    enc_full = TSTEncoder(cfg_full)
    assert enc_full.causal_len_token_num is None


@torch.no_grad()
def _effective_receptive_field(enc, hidden_size, long_len=20, probe=10):
    """Smallest independent window ending at `probe` that reproduces the shared pass."""
    torch.manual_seed(1)
    x_full = torch.randn(1, long_len, hidden_size)
    cm = torch.ones(1, 1)
    shared = enc(x=x_full, n_channels=1, key_padding_mask=torch.ones(1, 1, long_len), channel_mask=cm)

    for start in range(probe, -1, -1):
        span = probe - start + 1
        window = enc(
            x=x_full[:, start:probe + 1],
            n_channels=1,
            key_padding_mask=torch.ones(1, 1, span),
            channel_mask=cm,
        )
        if torch.allclose(shared[:, probe], window[:, -1], atol=1e-5):
            return span
    return None


@torch.no_grad()
def test_single_layer_matches_independent_window_exactly():
    """
    With one attention layer the band is exact: a patch's embedding in a long
    shared pass equals what an independent context_len-only window produces.
    """
    cfg = _make_encoder_config(context_len=8, patch_len=2, stride=2)
    cfg.n_layers = 1
    enc = TSTEncoder(cfg).eval()

    assert _effective_receptive_field(enc, cfg.hidden_size) == enc.causal_len_token_num


@torch.no_grad()
def test_multi_layer_receptive_field_grows_but_stays_bounded():
    """
    KNOWN LIMITATION (documented in tst_encoder.py): the band caps ONE layer's
    lookback, not the stack's. Each extra layer extends the true receptive
    field by up to (W-1) more patches, like a deeper CNN. Still bounded —
    unlike the pre-fix behavior, where it grew with the whole series length.
    """
    for n_layers in (2, 3):
        cfg = _make_encoder_config(context_len=8, patch_len=2, stride=2)
        cfg.n_layers = n_layers
        enc = TSTEncoder(cfg).eval()
        W = enc.causal_len_token_num

        rf = _effective_receptive_field(enc, cfg.hidden_size)
        assert rf == 1 + n_layers * (W - 1), f"n_layers={n_layers}: rf={rf}"
        assert rf > W, "sanity: multi-layer reach should exceed a single layer's band"


def test_blocked_eval_covers_same_fcds_as_legacy():
    """
    Blocked eval must cover exactly the FCDs the legacy whole-series pass
    covers — same targets, same masks — just computed in training-shaped
    blocks folded into the batch dim.
    """
    B, T, C, X1 = 2, 60, 1, 1
    L, H, stride = 8, 4, 2
    torch.manual_seed(0)
    batch = dict(
        x_enc=torch.randn(B, T, C, X1),
        available_mask=torch.ones(B, T, C),
        loss_mask=torch.ones(B, T, C),
        channel_mask=torch.ones(B, C),
    )

    legacy = ForkingSequences(context_len=L, fcd_samples=-1, patch_len=2, stride=stride)
    out_l = legacy(batch, horizon=H)

    for fcd_eval in (1, 3, 4, 7):
        blk = ForkingSequences(context_len=L, fcd_samples=fcd_eval, patch_len=2, stride=stride, blocked=True)
        out_b = blk(batch, horizon=H, fcd_samples=fcd_eval)
        n_blocks, n_fcds = out_b["n_blocks"], out_b["n_fcds"]

        def unfold(t):
            return t.reshape(t.shape[0] // n_blocks, n_blocks * t.shape[1], *t.shape[2:])[:, :n_fcds]

        assert torch.allclose(unfold(out_b["outsample_y"]), out_l["outsample_y"])
        assert torch.allclose(unfold(out_b["outsample_mask"]), out_l["outsample_mask"])
        # each block's encoder input is exactly a training block's insample length
        assert out_b["insample_y"].shape[1] == L + (fcd_eval - 1) * stride


def test_blocked_eval_masks_padded_trailing_fcds():
    """
    The trailing block is padded by clamping the gather index at T-1, which
    duplicates the last real timestep (loss_mask=1 there for test rows).
    Those slots must be zeroed or they'd add duplicate rows to val loss.
    """
    B, T, C, X1 = 2, 60, 1, 1
    L, H, stride, fcd_eval = 8, 4, 2, 4
    torch.manual_seed(0)
    batch = dict(
        x_enc=torch.randn(B, T, C, X1),
        available_mask=torch.ones(B, T, C),
        loss_mask=torch.ones(B, T, C),
        channel_mask=torch.ones(B, C),
    )

    blk = ForkingSequences(context_len=L, fcd_samples=fcd_eval, patch_len=2, stride=stride, blocked=True)
    out = blk(batch, horizon=H, fcd_samples=fcd_eval)
    n_blocks, n_fcds = out["n_blocks"], out["n_fcds"]

    mask = out["outsample_mask"].reshape(B, n_blocks * fcd_eval, H, C)
    assert n_blocks * fcd_eval > n_fcds, "test needs a partial trailing block"
    assert (mask[:, n_fcds:] == 0).all(), "padded FCDs must not contribute"
    assert (mask[:, :n_fcds] == 1).all(), "real FCDs must be untouched"


def test_full_context_sampled_stats_align_with_insample():
    """
    Regression: _sampled_fcds_full_context builds enc_block as the 0-based
    prefix x_enc_full[:, :block_end], so it must report window_start=None.
    Returning the sampled window_start made __call__ offset its stats lookup
    by it — pairing insample_y[0] with statistics from a much later timestep,
    and reading past the end of the series.
    """
    B, T, C, X1 = 2, 40, 1, 1
    H, stride, patch_len, fcd = 3, 2, 2, 3
    torch.manual_seed(0)
    x = torch.randn(B, T, C, X1)
    avail = torch.ones(B, T, C)
    avail[:, :20] = 0          # force a late window_start, which exposed the bug
    batch = dict(
        x_enc=x, available_mask=avail,
        loss_mask=torch.ones(B, T, C), channel_mask=torch.ones(B, C),
    )

    fs = ForkingSequences(context_len=-1, fcd_samples=fcd, patch_len=patch_len, stride=stride)
    torch.manual_seed(3)
    out = fs(batch, horizon=H, fcd_samples=fcd)

    enc_size = out["insample_y"].shape[1]
    assert torch.allclose(out["insample_y"], x[:, :enc_size]), "insample must be the 0-based prefix"

    m = avail.unsqueeze(-1).expand_as(x)
    expected = torch.cumsum(x * m, dim=1) / torch.cumsum(m, dim=1).clamp(min=1)
    assert torch.allclose(out["norm_stats"]["mean"], expected[:, :enc_size], atol=1e-6), (
        "norm stats are offset from the data they normalize"
    )


def test_blocked_eval_norm_stats_stay_full_series():
    """
    Stats must remain cumulative over the whole series (as training computes
    them), not restart at each block boundary.
    """
    B, T, C, X1 = 2, 60, 1, 1
    L, H, stride, fcd_eval = 8, 4, 2, 4
    torch.manual_seed(0)
    batch = dict(
        x_enc=torch.randn(B, T, C, X1),
        available_mask=torch.ones(B, T, C),
        loss_mask=torch.ones(B, T, C),
        channel_mask=torch.ones(B, C),
    )

    legacy = ForkingSequences(context_len=L, fcd_samples=-1, patch_len=2, stride=stride)
    blk = ForkingSequences(context_len=L, fcd_samples=fcd_eval, patch_len=2, stride=stride, blocked=True)
    out_l = legacy(batch, horizon=H)
    out_b = blk(batch, horizon=H, fcd_samples=fcd_eval)
    n_blocks, n_fcds = out_b["n_blocks"], out_b["n_fcds"]

    got = out_b["norm_fcd_stats"]["mean"].reshape(B, n_blocks * fcd_eval, C, X1)[:, :n_fcds]
    assert torch.allclose(out_l["norm_fcd_stats"]["mean"], got)

    per_block = out_b["norm_fcd_stats"]["mean"].reshape(B, n_blocks, fcd_eval, C, X1)
    assert not torch.allclose(per_block[:, 0, 0], per_block[:, 1, 0]), (
        "first FCD of each block has identical stats — stats restarted per block"
    )


_NS_B, _NS_T, _NS_C, _NS_X1 = 2, 60, 1, 1
_NS_L, _NS_H, _NS_STRIDE, _NS_PATCH = 8, 4, 2, 2


def _norm_stats_batch(x, avail):
    return dict(
        x_enc=x.clone(), available_mask=avail.clone(),
        loss_mask=torch.ones(_NS_B, _NS_T, _NS_C), channel_mask=torch.ones(_NS_B, _NS_C),
    )


def _norm_stats_strategies():
    L, P, S = _NS_L, _NS_PATCH, _NS_STRIDE
    return [
        ("train fixed-ctx sampled", ForkingSequences(context_len=L, fcd_samples=4, patch_len=P, stride=S), dict(fcd_samples=4)),
        ("eval legacy all-fcds",    ForkingSequences(context_len=L, fcd_samples=-1, patch_len=P, stride=S), dict()),
        ("eval blocked fcd=4",      ForkingSequences(context_len=L, fcd_samples=4, patch_len=P, stride=S, blocked=True), dict(fcd_samples=4)),
        ("eval blocked fcd=10",     ForkingSequences(context_len=L, fcd_samples=10, patch_len=P, stride=S, blocked=True), dict(fcd_samples=10)),
        ("full-ctx all-fcds",       ForkingSequences(context_len=-1, fcd_samples=-1, patch_len=P, stride=S), dict()),
        ("full-ctx sampled",        ForkingSequences(context_len=-1, fcd_samples=3, patch_len=P, stride=S), dict(fcd_samples=3)),
    ]


def _block_start(ws, r, n_blocks):
    if ws is None:
        return 0
    b, blk = divmod(r, n_blocks)
    return int(ws.reshape(_NS_B, n_blocks)[b, blk])


def test_norm_stats_align_with_absolute_timesteps():
    """
    norm_stats[row, t] must be the causal cumulative statistic at the ABSOLUTE
    timestep that insample_y[row, t] came from — for every strategy, including
    the ones whose blocks start at a nonzero offset. Positions past the end of
    the series belong to the blocked path's right-padding and must be masked.
    """
    B, T, C, X1 = _NS_B, _NS_T, _NS_C, _NS_X1
    torch.manual_seed(0)
    x = torch.randn(B, T, C, X1)
    avail = torch.ones(B, T, C)
    avail[:, :5] = 0                      # real left-padding, must be excluded from stats

    m = avail.unsqueeze(-1).expand_as(x)
    true_mean = torch.cumsum(x * m, 1) / torch.cumsum(m, 1).clamp(min=1)

    for label, fs, ck in _norm_stats_strategies():
        torch.manual_seed(7)
        *_, ws = fs._strategy(batch=_norm_stats_batch(x, avail), horizon=_NS_H, **ck)
        torch.manual_seed(7)
        out = fs(_norm_stats_batch(x, avail), horizon=_NS_H, **ck)

        ins, got, avm = out["insample_y"], out["norm_stats"]["mean"], out["available_mask"]
        n_blocks, enc = out["n_blocks"], ins.shape[1]

        for r in range(ins.shape[0]):
            start = _block_start(ws, r, n_blocks)
            for t in range(enc):
                p = start + t
                if p >= T:
                    assert not avm[r, t].any(), f"{label}: padded cell not masked at row {r}, t {t}"
                    continue
                assert torch.allclose(ins[r, t], x[r // n_blocks, p]), f"{label}: data misaligned"
                assert torch.allclose(got[r, t], true_mean[r // n_blocks, p], atol=1e-6), (
                    f"{label}: stats at row {r}, t {t} don't match timestep {p}"
                )


def test_norm_stats_are_causal():
    """
    Perturbing timestep K must leave every statistic at timesteps < K untouched,
    for both cumulative (norm_window_size=-1) and rolling-window stats.
    """
    B, T, C, X1 = _NS_B, _NS_T, _NS_C, _NS_X1
    K = 30
    torch.manual_seed(0)
    x0 = torch.randn(B, T, C, X1)
    x1 = x0.clone()
    x1[:, K] += 1000.0
    avail = torch.ones(B, T, C)

    cases = _norm_stats_strategies() + [
        (f"blocked norm_window={W}",
         ForkingSequences(context_len=_NS_L, fcd_samples=4, patch_len=_NS_PATCH,
                          stride=_NS_STRIDE, blocked=True, norm_window_size=W),
         dict(fcd_samples=4))
        for W in (5, 12)
    ]

    for label, fs, ck in cases:
        torch.manual_seed(7)
        *_, ws = fs._strategy(batch=_norm_stats_batch(x0, avail), horizon=_NS_H, **ck)
        torch.manual_seed(7); a = fs(_norm_stats_batch(x0, avail), horizon=_NS_H, **ck)
        torch.manual_seed(7); b = fs(_norm_stats_batch(x1, avail), horizon=_NS_H, **ck)

        n_blocks, enc = a["n_blocks"], a["insample_y"].shape[1]
        ma, mb = a["norm_stats"]["mean"], b["norm_stats"]["mean"]

        for r in range(ma.shape[0]):
            start = _block_start(ws, r, n_blocks)
            for t in range(enc):
                if start + t < K:
                    assert torch.allclose(ma[r, t], mb[r, t], atol=1e-9), (
                        f"{label}: stat at timestep {start+t} moved when t={K} changed — not causal"
                    )


def test_norm_fcd_stats_sit_on_forecast_origin():
    """
    The per-FCD stats that drive denorm must sit on each window's forecast
    origin (its last insample step) and must not reflect the horizon they are
    used to denormalize.
    """
    B, T, C, X1 = _NS_B, _NS_T, _NS_C, _NS_X1
    K, stride = 30, _NS_STRIDE
    torch.manual_seed(0)
    x0 = torch.randn(B, T, C, X1)
    x1 = x0.clone()
    x1[:, K] += 1000.0
    avail = torch.ones(B, T, C)

    m = avail.unsqueeze(-1).expand_as(x0)
    true_mean = torch.cumsum(x0 * m, 1) / torch.cumsum(m, 1).clamp(min=1)

    for label, fs, ck in _norm_stats_strategies():
        torch.manual_seed(7)
        *_, ws = fs._strategy(batch=_norm_stats_batch(x0, avail), horizon=_NS_H, **ck)
        torch.manual_seed(7); a = fs(_norm_stats_batch(x0, avail), horizon=_NS_H, **ck)
        torch.manual_seed(7); b = fs(_norm_stats_batch(x1, avail), horizon=_NS_H, **ck)

        n_blocks, n_fcd = a["n_blocks"], a["fcd_samples"]
        enc = a["insample_y"].shape[1]
        eff_L = enc - (n_fcd - 1) * stride
        fa, fb = a["norm_fcd_stats"]["mean"], b["norm_fcd_stats"]["mean"]

        for r in range(fa.shape[0]):
            start = _block_start(ws, r, n_blocks)
            for j in range(n_fcd):
                origin = start + j * stride + eff_L - 1
                if origin >= T:
                    continue                       # padded FCD, masked out elsewhere
                assert torch.allclose(fa[r, j], true_mean[r // n_blocks, origin], atol=1e-6), (
                    f"{label}: FCD {j} stats not at its origin ({origin})"
                )
                if origin < K:
                    assert torch.allclose(fa[r, j], fb[r, j], atol=1e-9), (
                        f"{label}: FCD {j} stats moved when a later timestep changed"
                    )


def _model_config(fcd_samples=10, fcd_samples_eval="__unset__"):
    cfg = SimpleNamespace(
        scaler_type="standard", stride=2, context_len=8, patch_len=2,
        fcd_samples=fcd_samples, fcd_sampler="heterogeneous", norm_window_size=-1,
        loss="mae", loss_space="denorm",
    )
    if fcd_samples_eval != "__unset__":
        cfg.fcd_samples_eval = fcd_samples_eval
    return cfg


def test_fcd_samples_eval_accepts_only_two_modes():
    """
    null (or absent) mirrors training; -1 is the legacy whole-series pass.
    Anything else is rejected — it would neither match training nor be
    maximally efficient.
    """
    from tsforge.common._base_model import BaseModel

    class _Dummy(BaseModel):
        def forward(self, batch):
            return None

    # accepted
    assert _Dummy(_model_config(10, "__unset__")).fcd_samples_eval == 10
    assert _Dummy(_model_config(10, None)).fcd_samples_eval == 10
    assert _Dummy(_model_config(10, 10)).fcd_samples_eval == 10
    assert _Dummy(_model_config(10, -1)).fcd_samples_eval == -1
    # fcd_samples=1 (window-sampling) may of course eval with blocks of 1
    assert _Dummy(_model_config(1, 1)).fcd_samples_eval == 1

    # -1 routes to the legacy non-blocked path, everything else to blocked
    assert _Dummy(_model_config(10, -1))._fork_sequences_eval.blocked is False
    assert _Dummy(_model_config(10, None))._fork_sequences_eval.blocked is True

    # rejected
    for bad in (1, 37, 0, -5, 2.5):
        try:
            _Dummy(_model_config(10, bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"fcd_samples_eval={bad!r} should have been rejected")


if __name__ == "__main__":
    tests = [
        test_banded_mask_matches_sliding_window,
        test_none_reproduces_unrestricted_causal,
        test_window_wider_than_sequence_is_a_noop,
        test_padding_still_respected_within_the_band,
        test_causal_len_token_num_derivation,
        test_single_layer_matches_independent_window_exactly,
        test_multi_layer_receptive_field_grows_but_stays_bounded,
        test_blocked_eval_covers_same_fcds_as_legacy,
        test_blocked_eval_masks_padded_trailing_fcds,
        test_full_context_sampled_stats_align_with_insample,
        test_blocked_eval_norm_stats_stay_full_series,
        test_norm_stats_align_with_absolute_timesteps,
        test_norm_stats_are_causal,
        test_norm_fcd_stats_sit_on_forecast_origin,
        test_fcd_samples_eval_accepts_only_two_modes,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


def test_val_fcd_sampling_fixed_across_checks():
    """
    fcd_samples_val draws starts with the training sampler but keyed on the
    val batch index, so repeated val checks score identical windows no matter
    what was sampled in between (e.g. training steps on the same forker seed).
    """
    B, T, C, X1 = 4, 60, 1, 1
    L, H, stride, N = 8, 4, 2, 3
    torch.manual_seed(0)
    batch = dict(
        x_enc=torch.randn(B, T, C, X1),
        available_mask=torch.ones(B, T, C),
        loss_mask=torch.ones(B, T, C),
        channel_mask=torch.ones(B, C),
    )
    fs = ForkingSequences(context_len=L, fcd_samples=N, patch_len=2, stride=stride, fcd_seed=1)

    first = [fs(batch, horizon=H, fcd_samples=N, step=i)["outsample_y"] for i in range(3)]
    fs(batch, horizon=H, fcd_samples=N, step=123)  # unrelated draw in between
    second = [fs(batch, horizon=H, fcd_samples=N, step=i)["outsample_y"] for i in range(3)]

    for a, b in zip(first, second):
        assert torch.equal(a, b)
    # distinct batch indices give distinct (not one shared) draws
    assert not torch.equal(first[0], first[1])
    assert first[0].shape[1] == N
