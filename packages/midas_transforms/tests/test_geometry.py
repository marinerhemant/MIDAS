"""Tests for ``midas_transforms.geometry`` — SampleGrid + beam profiles.

Covers:
- SampleGrid factory + .to(device) + voxels_in_grain / voxels_in_sample.
- TopHat fraction_over_voxel: fully-inside, fully-outside, partial, no-overlap.
- Gaussian fraction_over_voxel: peak ≈ 1 for narrow voxel, ≈ 0 far away.
- Differentiability: gradcheck on TopHat + Gaussian w.r.t. (scan_pos,
  voxel_center, voxel_size).  TopHat: w.r.t. scan_pos only inside the
  overlap region; gradient is zero a.e. outside.
- Device portability: CPU + MPS forward; CPU for fp64 gradcheck.
- TopHat.refine + Gaussian.refine_fwhm: torch.optim.parameters() picks up the
  refinable parameter (proves the ``nn.Module`` integration).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import pytest

torch = pytest.importorskip("torch")

from midas_transforms.geometry import (
    BeamProfile, Gaussian, SampleGrid, TopHat,
)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ----------------------------------------------------------- SampleGrid


def test_sample_grid_from_arrays_defaults():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        voxel_size_um=5.0,
    )
    assert sg.n_voxels == 2
    assert sg.dtype == torch.float64
    # No mask provided -> all True
    assert bool(sg.sample_mask.all().item())
    # No grain_map -> all -1
    assert sg.grain_map.tolist() == [-1, -1]


def test_sample_grid_grain_queries():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[i * 10.0, 0.0, 0.0] for i in range(5)],
        voxel_size_um=5.0,
        grain_map=[0, 1, 0, -1, 1],
        sample_mask=[True, True, True, False, True],
    )
    # grain 0: voxels 0, 2 (voxel 3 masked out anyway)
    assert sg.voxels_in_grain(0).tolist() == [0, 2]
    # grain 1: voxels 1, 4
    assert sg.voxels_in_grain(1).tolist() == [1, 4]
    # sample mask: 0, 1, 2, 4
    assert sg.voxels_in_sample().tolist() == [0, 1, 2, 4]
    # unique grain ids in sample
    assert sg.grain_ids().tolist() == [0, 1]


def test_sample_grid_shape_validation():
    with pytest.raises(ValueError, match="voxel_positions"):
        SampleGrid.from_arrays(
            voxel_positions=[1.0, 2.0, 3.0], voxel_size_um=5.0,
        )
    with pytest.raises(ValueError, match="grain_map"):
        SampleGrid.from_arrays(
            voxel_positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            voxel_size_um=5.0, grain_map=[0],
        )


@pytest.mark.parametrize("device", _devices())
def test_sample_grid_to_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        voxel_size_um=5.0,
        grain_map=[0, 1],
    )
    sg2 = sg.to(device=device, dtype=dtype)
    assert sg2.voxel_positions.device.type == torch.device(device).type
    assert sg2.voxel_positions.dtype == dtype
    assert sg2.grain_map.device.type == torch.device(device).type


# ----------------------------------------------------------- TopHat


def _t(x, **kw):  # noqa: ANN
    kw.setdefault("dtype", torch.float64)
    return torch.tensor(x, **kw)


def test_tophat_voxel_fully_inside_beam():
    """Voxel size 5, beam width 20 -> fraction = 1."""
    beam = TopHat(20.0)
    f = beam.fraction_over_voxel(_t(0.0), _t(0.0), _t(5.0))
    assert math.isclose(float(f), 1.0)


def test_tophat_beam_fully_inside_voxel():
    """Beam width 5, voxel size 10, both centered -> fraction = 5/10."""
    beam = TopHat(5.0)
    f = beam.fraction_over_voxel(_t(0.0), _t(0.0), _t(10.0))
    assert math.isclose(float(f), 0.5)


def test_tophat_partial_overlap():
    """Beam at -3, width 6 ([-6, 0]); voxel at 0, size 10 ([-5, 5]).
    Overlap = [-5, 0] = 5 units -> fraction = 5/10 = 0.5."""
    beam = TopHat(6.0)
    f = beam.fraction_over_voxel(_t(-3.0), _t(0.0), _t(10.0))
    assert math.isclose(float(f), 0.5)


def test_tophat_no_overlap():
    beam = TopHat(5.0)
    f = beam.fraction_over_voxel(_t(0.0), _t(100.0), _t(10.0))
    assert float(f) == 0.0


def test_tophat_broadcasts():
    """(3 scans, 3 voxels) -> (3, 3) fractions."""
    beam = TopHat(10.0)  # beam half-width = 5
    scan = torch.tensor([0.0, 5.0, 100.0], dtype=torch.float64).unsqueeze(1)   # (3, 1)
    vox  = torch.tensor([0.0, 5.0, 20.0], dtype=torch.float64).unsqueeze(0)    # (1, 3)
    f = beam.fraction_over_voxel(scan, vox, _t(10.0))
    assert f.shape == (3, 3)
    # scan=0 ([-5,5]): voxel 0 ([-5,5])=1, voxel 1 ([0,10]) overlap [0,5]=5/10=0.5,
    # voxel 2 ([15,25]) = 0
    assert torch.allclose(f[0], torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64))
    # scan=100 ([95,105]): no overlap with any voxel
    assert torch.allclose(f[2], torch.zeros(3, dtype=torch.float64))


def test_tophat_refine_registers_parameter():
    """``refine=True`` -> width_um is an nn.Parameter; .parameters() includes it."""
    beam = TopHat(5.0, refine=True)
    params = list(beam.parameters())
    assert len(params) == 1
    assert params[0] is beam.width_um


def test_tophat_no_refine_no_parameters():
    beam = TopHat(5.0, refine=False)
    assert list(beam.parameters()) == []
    # width_um is a buffer
    assert "width_um" in dict(beam.named_buffers())


def test_tophat_gradcheck_wrt_scan_pos():
    """Smooth inside the overlap region -> gradcheck passes."""
    beam = TopHat(10.0)  # width
    voxel_size = _t(5.0)

    def f(scan_pos):
        return beam.fraction_over_voxel(scan_pos, _t(0.0), voxel_size).sum()

    # Pick scan_pos = 1.5 → inside the partial-overlap region for a 10-wide
    # beam over a 5-wide voxel at 0 (always 100% covered actually; pick
    # 5.0 for partial overlap).
    scan = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (scan,), eps=1e-6, atol=1e-7)


# ----------------------------------------------------------- Gaussian


def test_gaussian_peak_amplitude_at_origin():
    """Voxel size << FWHM, voxel at peak -> fraction ≈ 1 (peak amplitude)."""
    beam = Gaussian(10.0, refine_fwhm=False)  # FWHM 10
    f = beam.fraction_over_voxel(_t(0.0), _t(0.0), _t(0.05))
    # Should be essentially exp(0) = 1
    assert abs(float(f) - 1.0) < 1e-4


def test_gaussian_far_from_peak():
    """Voxel many FWHM away -> ~0."""
    beam = Gaussian(2.0, refine_fwhm=False)
    f = beam.fraction_over_voxel(_t(0.0), _t(100.0), _t(1.0))
    assert float(f) < 1e-30


def test_gaussian_half_max_at_fwhm_over_two():
    """Voxel small at scan_pos = +FWHM/2 (the half-max point) -> fraction ≈ 0.5."""
    beam = Gaussian(10.0, refine_fwhm=False)
    f = beam.fraction_over_voxel(_t(0.0), _t(5.0), _t(0.05))
    assert abs(float(f) - 0.5) < 1e-3


def test_gaussian_parameters_default_refine_fwhm():
    """Default: fwhm_um is a parameter, center_offset is a buffer."""
    beam = Gaussian(2.0)
    pnames = [n for n, _ in beam.named_parameters()]
    assert pnames == ["fwhm_um"]


def test_gaussian_gradcheck_wrt_inputs():
    beam = Gaussian(2.0)
    voxel_size = _t(1.0)

    def f(scan_pos, voxel_center, fwhm):
        beam.fwhm_um.data = fwhm.detach()  # not differentiated; fix fwhm here
        return beam.fraction_over_voxel(scan_pos, voxel_center, voxel_size).sum()

    scan = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    center = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    fwhm = torch.tensor(2.0, dtype=torch.float64)
    # Differentiate only in (scan, center); fwhm fixed via .data assignment.
    assert torch.autograd.gradcheck(
        lambda s, c: f(s, c, fwhm), (scan, center), eps=1e-6, atol=1e-7
    )


def test_gaussian_gradcheck_wrt_fwhm():
    """fwhm_um is an nn.Parameter; gradcheck via a wrapper."""
    def f(fwhm):
        # Build a fresh Gaussian each call (cheap) so that fwhm is the leaf
        beam = Gaussian(fwhm, refine_fwhm=False)
        return beam.fraction_over_voxel(_t(0.0), _t(0.5), _t(0.1)).sum()

    fwhm = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (fwhm,), eps=1e-6, atol=1e-5)


# ----------------------------------------------------------- device portability


@pytest.mark.parametrize("device", _devices())
def test_tophat_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    beam = TopHat(10.0, device=device, dtype=dtype)
    f = beam.fraction_over_voxel(
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(2.0, dtype=dtype, device=device),
        torch.tensor(5.0, dtype=dtype, device=device),
    )
    assert f.device.type == torch.device(device).type
    assert math.isclose(float(f), 1.0, abs_tol=1e-4)


@pytest.mark.parametrize("device", _devices())
def test_gaussian_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    beam = Gaussian(2.0, refine_fwhm=False, device=device, dtype=dtype)
    f = beam.fraction_over_voxel(
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(0.05, dtype=dtype, device=device),
    )
    assert f.device.type == torch.device(device).type
    assert abs(float(f) - 1.0) < 1e-3
