"""calc_radius must not use floating-point atomics.

Found while chasing why the FF pipeline produced a different Grains.csv on
every run (1-ID GE5 FF scan, Au3_cubes_ff_000008, 2026-07-30). After the
midas_peakfit RegionPool/TF32 fix made the peak list bit-identical, one
divergence remained: ``Result_StartNr_*.csv`` (merge output) matched between
runs but ``Radius_StartNr_*.csv`` did not, and the only field that survived
to Grains.csv was GrainRadius — 20.775146 vs 20.775148 µm.

Cause: the per-ring powder-intensity reduction used
``powder_int.scatter_add_(0, spot_match, ...)``. On CUDA that lowers to
floating-point atomicAdd, whose summation order is arbitrary per launch;
``powder_int`` then divides into GrainVolume, so the jitter came out in the
radius column.

The reduction is over the CONFIGURED RINGS (a handful), so a deterministic
per-ring ``sum`` costs nothing. These tests pin that: the source must not
reintroduce the atomic, and the kernel must be exactly repeatable and still
numerically correct against an independent reference.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from midas_transforms.radius import core as radius_core


def _synthetic_result_arr(n: int = 500, seed: int = 0) -> torch.Tensor:
    """A Result_*.csv-shaped table (18 cols) with radii clustered on rings."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((n, 18), dtype=np.float64)
    arr[:, 0] = np.arange(1, n + 1)                     # SpotID
    arr[:, 1] = rng.random(n) * 1e4 + 1.0               # IntegratedIntensity
    arr[:, 2] = rng.random(n) * 360.0 - 180.0           # Omega
    arr[:, 6] = arr[:, 2] - 0.125                       # MinOme
    arr[:, 7] = arr[:, 2] + 0.125                       # MaxOme
    # Radius in PIXELS, scattered tightly around three ring radii.
    centres_px = np.array([462.3, 534.0, 756.4])
    which = rng.integers(0, centres_px.size, n)
    arr[:, 12] = centres_px[which] + rng.normal(0, 0.4, n)
    arr[:, 13] = rng.random(n) * 360.0 - 180.0          # Eta
    return torch.from_numpy(arr)


_KW = dict(
    width_px=500.0, px_um=200.0, Lsd_um=1_666_219.585298,
    OmegaStep=-0.25, Hbeam=2000.0, Rsample=2000.0, Vsample=0.0,
    DiscModel=0, DiscArea=0.0, n_frames=1440, top_layer=False,
)


def _run(dtype=torch.float64):
    arr = _synthetic_result_arr().to(dtype)
    ring_numbers = torch.tensor([1, 2, 3], dtype=torch.int64)
    ring_radii_um = torch.tensor([462.3, 534.0, 756.4], dtype=dtype) * 200.0
    return radius_core._filter_and_compute_radius(
        arr, ring_numbers, ring_radii_um, **_KW
    )


def test_source_does_not_use_scatter_add():
    """Guard against the atomic coming back. ``scatter_add`` on a float
    tensor is nondeterministic on CUDA — PyTorch documents it as such."""
    src = inspect.getsource(radius_core._filter_and_compute_radius)
    # Comments are allowed to name it (the fix explains itself); code is not.
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "scatter_add" not in code, (
        "calc_radius reintroduced a floating-point scatter_add; that makes "
        "GrainRadius non-reproducible run to run"
    )


def test_powder_intensity_is_bit_repeatable():
    _, p1, _ = _run()
    _, p2, _ = _run()
    assert torch.equal(p1, p2)


def test_radius_output_is_bit_repeatable():
    out1, _, _ = _run()
    out2, _, _ = _run()
    assert out1.shape == out2.shape
    assert torch.equal(out1, out2), "calc_radius output is not repeatable"


def test_powder_intensity_matches_an_independent_reference():
    """Determinism is worthless if the value is wrong. Recompute the per-ring
    sum in numpy from the same pair list and require exact agreement in the
    float64 path."""
    arr = _synthetic_result_arr()
    ring_numbers = torch.tensor([1, 2, 3], dtype=torch.int64)
    ring_radii_um = torch.tensor([462.3, 534.0, 756.4], dtype=torch.float64) * 200.0
    _, powder, _ = radius_core._filter_and_compute_radius(
        arr, ring_numbers, ring_radii_um, **_KW
    )

    # Reproduce the (spot, ring) pair list the kernel builds.
    r_obs_um = arr[:, 12].numpy() * _KW["px_um"]
    diff = np.abs(r_obs_um[:, None] - ring_radii_um.numpy()[None, :])
    pairs = np.argwhere(diff < _KW["width_px"])
    ref = np.zeros(3, dtype=np.float64)
    for spot_i, ring_i in pairs:
        ref[ring_i] += arr[spot_i, 1].item()
    ref /= _KW["n_frames"]

    np.testing.assert_allclose(powder.numpy(), ref, rtol=1e-12, atol=0.0)


def test_empty_ring_gets_zero_not_nan():
    """A configured ring with no spots must contribute 0, the same as the
    scatter_add version did — otherwise GrainVolume divides by NaN."""
    arr = _synthetic_result_arr()
    ring_numbers = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    # Ring 4 is placed far from every spot, and the window is tightened so
    # nothing reaches it.
    ring_radii_um = torch.tensor(
        [462.3, 534.0, 756.4, 1900.0], dtype=torch.float64) * 200.0
    kw = dict(_KW, width_px=100.0)
    _, powder, _ = radius_core._filter_and_compute_radius(
        arr, ring_numbers, ring_radii_um, **kw
    )
    assert powder.shape == (4,)
    assert powder[3].item() == 0.0
    assert torch.isfinite(powder).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_repeats():
    arr = _synthetic_result_arr()
    ring_numbers = torch.tensor([1, 2, 3], dtype=torch.int64)
    rad = torch.tensor([462.3, 534.0, 756.4], dtype=torch.float64) * 200.0
    _, p_cpu, _ = radius_core._filter_and_compute_radius(
        arr, ring_numbers, rad, **_KW)
    _, p_gpu_a, _ = radius_core._filter_and_compute_radius(
        arr.cuda(), ring_numbers.cuda(), rad.cuda(), **_KW)
    _, p_gpu_b, _ = radius_core._filter_and_compute_radius(
        arr.cuda(), ring_numbers.cuda(), rad.cuda(), **_KW)
    assert torch.equal(p_gpu_a, p_gpu_b), "CUDA calc_radius still jitters"
    np.testing.assert_allclose(
        p_gpu_a.cpu().numpy(), p_cpu.numpy(), rtol=1e-12, atol=0.0)
