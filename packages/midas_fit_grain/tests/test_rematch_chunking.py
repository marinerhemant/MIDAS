"""Chunking the observed<->predicted association must be EXACT.

The association materialises several (B, S, K, M) tensors at once. At
full-layer FF scale that is 22327 x 244 x 2 x 168 = 13.3 GiB *each* in float64,
which OOMs a 47 GiB A6000 and made ``--refine-backend python --device cuda``
unusable on a whole layer.

Grains are independent in this step, so slicing B changes only peak memory.
These tests hold that line: chunked and unchunked must agree bit-for-bit, or
the "fix" has quietly become a different algorithm.
"""

import os

import pytest
import torch

# The package exports a FUNCTION named refine_block, which shadows the module
# of the same name. Neither `from midas_fit_grain import refine_block` nor
# `import midas_fit_grain.refine_block as rb` reaches the module: both bind
# getattr(package, "refine_block") AFTER import, i.e. the function.
# importlib returns the module object itself.
import importlib

rb = importlib.import_module("midas_fit_grain.refine_block")


def _fake_inputs(B=17, S=9, M=13, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = type("O", (), {})()
    obs.n_grains = B
    obs.s_max = S
    obs.omega = (torch.rand(B, S, generator=g) * 2 - 1) * 3.14159
    obs.eta = (torch.rand(B, S, generator=g) * 2 - 1) * 3.14159
    obs.valid = torch.rand(B, S, generator=g) > 0.2
    obs.spot_id = torch.arange(B * S).reshape(B, S)
    obs.ring_nr = torch.randint(0, 3, (B, S), generator=g)
    obs.y_lab = torch.zeros(B, S)
    obs.z_lab = torch.zeros(B, S)
    obs.two_theta = torch.zeros(B, S)
    obs.n_spots = torch.full((B,), S)

    def slice_grains(lo, hi, _o=obs):
        sub = type("O", (), {})()
        sub.n_grains = hi - lo
        sub.s_max = _o.s_max
        for f in ("omega", "eta", "valid", "spot_id", "ring_nr",
                  "y_lab", "z_lab", "two_theta", "n_spots"):
            setattr(sub, f, getattr(_o, f)[lo:hi])
        sub.slice_grains = lambda a, b, _s=sub: slice_grains(a, b, _s)
        return sub

    obs.slice_grains = slice_grains
    obs_ring_slot = torch.randint(0, 3, (B, S), generator=g)
    pred_ring_slot = torch.randint(0, 3, (M,), generator=g)

    class _Model:
        def __call__(self, euler, pos, lattice_params=None):
            b = euler.shape[0]
            gg = torch.Generator().manual_seed(1234)
            out = type("S", (), {})()
            # deterministic in b: derive from a fixed full-size draw, sliced.
            full_om = (torch.rand(200, K, M, generator=gg) * 2 - 1) * 3.14159
            gg2 = torch.Generator().manual_seed(5678)
            full_et = (torch.rand(200, K, M, generator=gg2) * 2 - 1) * 3.14159
            gg3 = torch.Generator().manual_seed(99)
            full_va = torch.rand(200, K, M, generator=gg3) > 0.15
            idx = _Model.offset
            out.omega = full_om[idx:idx + b]
            out.eta = full_et[idx:idx + b]
            out.valid = full_va[idx:idx + b]
            return out

    _Model.offset = 0
    return obs, obs_ring_slot, pred_ring_slot, _Model()


def _run(monkeypatch, chunk_gib, B=17):
    """Run the association with a given memory budget, tracking model offsets."""
    obs, ors, prs, model = _fake_inputs(B=B)
    pos = torch.zeros(B, 3, dtype=torch.float64)
    eul = torch.zeros(B, 3, dtype=torch.float64)
    lat = torch.ones(B, 6, dtype=torch.float64)

    # the fake model must return the SAME predictions for a grain regardless of
    # how the batch was sliced, so track where each sub-batch starts
    orig = type(model).__call__
    state = {"lo": 0}

    def tracked(self, euler, position, lattice_params=None):
        type(self).offset = state["lo"]
        out = orig(self, euler, position, lattice_params=lattice_params)
        state["lo"] += euler.shape[0]
        return out

    monkeypatch.setattr(type(model), "__call__", tracked)
    monkeypatch.setenv("MIDAS_FIT_GRAIN_MATCH_GIB", str(chunk_gib))
    return rb._rematch_batch(
        model=model, pos=pos, euler=eul, lattice=lat, obs=obs,
        obs_ring_slot=ors, pred_ring_slot=prs,
        omega_tolerance=1.0, eta_tolerance=1.0,
    )


def test_chunked_equals_unchunked_bitwise(monkeypatch):
    big = _run(monkeypatch, 1e6)          # one chunk: no slicing at all
    small = _run(monkeypatch, 1e-7)       # force many tiny chunks
    assert torch.equal(big.k_idx, small.k_idx)
    assert torch.equal(big.m_idx, small.m_idx)
    assert torch.equal(big.mask, small.mask)


@pytest.mark.parametrize("gib", [1e-7, 1e-6, 1e-5, 1e6])
def test_result_is_invariant_to_the_budget(monkeypatch, gib):
    ref = _run(monkeypatch, 1e6)
    got = _run(monkeypatch, gib)
    assert torch.equal(ref.k_idx, got.k_idx)
    assert torch.equal(ref.mask, got.mask)


def test_chunk_size_shrinks_as_the_budget_shrinks():
    obs = type("O", (), {"s_max": 244})()
    prs = torch.zeros(168)
    big = rb._match_chunk_size(obs, prs, torch.float64)
    os.environ["MIDAS_FIT_GRAIN_MATCH_GIB"] = "0.001"
    try:
        small = rb._match_chunk_size(obs, prs, torch.float64)
    finally:
        os.environ.pop("MIDAS_FIT_GRAIN_MATCH_GIB", None)
    assert small < big
    assert small >= 1, "must never round down to zero grains"


def test_full_layer_scale_is_actually_chunked():
    """The configuration that OOMed must now split into many chunks."""
    obs = type("O", (), {"s_max": 244})()
    prs = torch.zeros(168)
    os.environ["MIDAS_FIT_GRAIN_MATCH_GIB"] = "2.0"
    try:
        chunk = rb._match_chunk_size(obs, prs, torch.float64)
    finally:
        os.environ.pop("MIDAS_FIT_GRAIN_MATCH_GIB", None)
    assert chunk < 22327, "22327 seeds at 13.3 GiB/tensor must not be one block"
    # and the resulting per-chunk tensor must be within the budget
    assert chunk * 244 * 2 * 168 * 8 <= 2.0 * 1024 ** 3
