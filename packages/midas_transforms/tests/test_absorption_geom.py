"""Tests for ``midas_transforms.geometry.absorption`` and the
``use_absorption=True`` path of ``radius.forward_model``.

Test layout
-----------
* :func:`path_length_in_sample` analytic checks (axis-aligned, diagonal,
  ray misses, partial mask).
* :func:`absorption_factor` numerical check against ``exp(-μ · L_total)``
  with hand-computed L_total.
* :func:`absorption_factor` gradient flow through ``mu_per_cm``.
* :func:`absorption_factor` requires a regular-grid SampleGrid (error path).
* Forward-model integration: ``use_absorption=True`` multiplies the
  no-absorption prediction by ``exp(-μ L)`` voxel-by-voxel.
* Forward-model: missing args raises with a helpful error.
* Multi-device (CPU + MPS).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import pytest

torch = pytest.importorskip("torch")

from midas_transforms.geometry import (
    SampleGrid, TopHat,
    absorption_factor, path_length_in_sample,
)
from midas_transforms.radius import predicted_spot_intensities


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _t(x, **kw):  # noqa: ANN
    kw.setdefault("dtype", torch.float64)
    return torch.tensor(x, **kw)


# ----------------------------------------------------------- topology factory


def test_regular_grid_topology_basic():
    sg = SampleGrid.from_regular_grid(
        grid_origin_um=[0.0, 0.0, 0.0],
        grid_shape=(3, 2, 1),
        voxel_size_um=10.0,
    )
    assert sg.grid_shape == (3, 2, 1)
    assert sg.n_voxels == 6
    # C-order with i fastest: flat_idx = i + nx*j + nx*ny*k
    # (i=0,j=0,k=0) -> (0,0,0); (i=1,j=0,k=0) -> (10,0,0); (i=0,j=1,k=0) -> (0,10,0)
    pos = sg.voxel_positions
    assert torch.allclose(pos[0], _t([0.0, 0.0, 0.0]))
    assert torch.allclose(pos[1], _t([10.0, 0.0, 0.0]))
    assert torch.allclose(pos[3], _t([0.0, 10.0, 0.0]))


def test_regular_grid_carries_origin_through_to():
    sg = SampleGrid.from_regular_grid(
        grid_origin_um=[5.0, 10.0, 0.0],
        grid_shape=(2, 2, 1),
        voxel_size_um=5.0,
    )
    sg2 = sg.to(device="cpu", dtype=torch.float32)
    assert sg2.grid_origin_um is not None
    assert sg2.grid_origin_um.dtype == torch.float32


# ----------------------------------------------------------- path_length


def _slab_5x1x1():
    """5-voxel slab along +x, voxel_size=10, origin at voxel-0 center (0,0,0).
    Bounding box is [-5, 45] x [-5, 5] x [-5, 5]."""
    return SampleGrid.from_regular_grid(
        grid_origin_um=[0.0, 0.0, 0.0],
        grid_shape=(5, 1, 1),
        voxel_size_um=10.0,
    )


def test_path_length_axis_aligned_forward():
    sg = _slab_5x1x1()
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        _t([[0.0, 0.0, 0.0]]),
        _t([[1.0, 0.0, 0.0]]),
    )
    assert math.isclose(float(L[0]), 45.0, abs_tol=1e-12)


def test_path_length_axis_aligned_backward():
    sg = _slab_5x1x1()
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        _t([[0.0, 0.0, 0.0]]),
        _t([[-1.0, 0.0, 0.0]]),
    )
    assert math.isclose(float(L[0]), 5.0, abs_tol=1e-12)


def test_path_length_diagonal():
    sg = _slab_5x1x1()
    d = 1.0 / math.sqrt(2.0)
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        _t([[0.0, 0.0, 0.0]]),
        _t([[d, d, 0.0]]),
    )
    # Exits when y reaches +5 at t = 5/d = 5*sqrt(2) ≈ 7.0711 (since dy/dt = d)
    assert math.isclose(float(L[0]), 5.0 * math.sqrt(2.0), abs_tol=1e-9)


def test_path_length_ray_misses_returns_zero():
    sg = _slab_5x1x1()
    # Ray below the grid (y=-100) going parallel to grid -> miss
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        _t([[0.0, -100.0, 0.0]]),
        _t([[1.0, 0.0, 0.0]]),
    )
    assert float(L[0]) == 0.0


def test_path_length_partial_mask():
    """Mask out the last 2 voxels -> path along +x from origin should be 25 µm
    (3 voxels: voxel 0 covers [-5, 5], 1 covers [5, 15], 2 covers [15, 25])."""
    sg = SampleGrid.from_regular_grid(
        grid_origin_um=[0.0, 0.0, 0.0],
        grid_shape=(5, 1, 1),
        voxel_size_um=10.0,
        sample_mask=[True, True, True, False, False],
    )
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        _t([[0.0, 0.0, 0.0]]),
        _t([[1.0, 0.0, 0.0]]),
    )
    assert math.isclose(float(L[0]), 25.0, abs_tol=1e-9)


def test_path_length_batched():
    sg = _slab_5x1x1()
    origins = _t([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -100.0, 0.0]])
    dirs = _t([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        origins, dirs,
    )
    assert L.shape == (3,)
    assert torch.allclose(L, _t([45.0, 5.0, 0.0]), atol=1e-9)


# ----------------------------------------------------------- absorption_factor


def test_absorption_factor_matches_analytic():
    sg = _slab_5x1x1()
    # Origin voxel 0 at (0,0,0); -inc = +x and dif = +x
    # L_in = 45 µm, L_out = 45 µm, total = 90 µm = 9e-3 cm
    # μ = 10 cm⁻¹ -> A = exp(-10 * 9e-3) = exp(-0.09)
    A = absorption_factor(
        sg, _t([0], dtype=torch.int64),
        _t([[-1.0, 0.0, 0.0]]),
        _t([[1.0, 0.0, 0.0]]),
        _t(10.0),
    )
    assert math.isclose(float(A[0]), math.exp(-0.09), rel_tol=1e-9)


def test_absorption_factor_gradflow_to_mu():
    sg = _slab_5x1x1()
    mu = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    A = absorption_factor(
        sg, _t([0], dtype=torch.int64),
        _t([[-1.0, 0.0, 0.0]]),
        _t([[1.0, 0.0, 0.0]]),
        mu,
    )
    A.sum().backward()
    # dA/dμ = -L_total · exp(-μ · L_total) = -9e-3 · exp(-0.09)
    expected = -9e-3 * math.exp(-0.09)
    assert math.isclose(float(mu.grad), expected, rel_tol=1e-9)


def test_absorption_factor_gradcheck_wrt_mu():
    sg = _slab_5x1x1()
    inc = _t([[-1.0, 0.0, 0.0]])
    dif = _t([[1.0, 0.0, 0.0]])
    voxel_idx = _t([0], dtype=torch.int64)

    def f(mu):
        return absorption_factor(sg, voxel_idx, inc, dif, mu).sum()

    mu = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (mu,), eps=1e-6, atol=1e-7)


def test_absorption_factor_requires_topology():
    """Flat SampleGrid (no grid_origin / grid_shape) -> error."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=10.0, grain_map=[0],
    )
    with pytest.raises(ValueError, match="from_regular_grid"):
        absorption_factor(
            sg, _t([0], dtype=torch.int64),
            _t([[-1.0, 0.0, 0.0]]),
            _t([[1.0, 0.0, 0.0]]),
            _t(10.0),
        )


def test_absorption_factor_per_spot_mu():
    """``mu_per_cm`` may be a (N,) tensor — vectorize over spots."""
    sg = _slab_5x1x1()
    A = absorption_factor(
        sg, _t([0, 0], dtype=torch.int64),
        _t([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        _t([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        _t([10.0, 20.0]),
    )
    assert math.isclose(float(A[0]), math.exp(-10 * 9e-3), rel_tol=1e-9)
    assert math.isclose(float(A[1]), math.exp(-20 * 9e-3), rel_tol=1e-9)


# --------------------------------------------- forward_model w/ absorption


def test_forward_with_absorption_scales_no_absorption_result():
    """With absorption on, prediction = (no-absorption prediction) · exp(-μ·L)
    voxel-by-voxel.  For a single voxel, ratio is exact."""
    sg = _slab_5x1x1()
    # Set grain id of voxel 0 to 0; mask out the rest of the grain
    grain_map = torch.tensor([0, -1, -1, -1, -1], dtype=torch.int64)
    sg.grain_map = grain_map

    V = _t([1.0, 0.0, 0.0, 0.0, 0.0])
    K = _t([2.0]); I_th = _t([3.0])
    spot_ring  = torch.tensor([0])
    spot_grain = torch.tensor([0])
    # Spot: beam at scan_pos=0 with TopHat width 20 -> voxel 0 fully covered
    # Omega = pi/2 -> proj(v) = v_x, voxel 0 at v_x=0 -> proj=0, in beam
    spot_scan  = _t([0.0])
    spot_omega = _t([math.pi/2])

    I_noabs = predicted_spot_intensities(
        V, K, I_th, spot_ring, spot_grain, spot_scan, spot_omega,
        sg, TopHat(20.0),
    )
    # incident along +y in lab -> -inc = -y; doesn't pass through any voxel since
    # bbox in y is [-5, 5] and origin at (0,0,0) -> path 5 µm one way, 0 going back.
    # Let's pick incident along +x (lab) so absorption is along the long axis.
    inc = _t([[1.0, 0.0, 0.0]])    # beam coming from -x direction, traveling +x
    dif = _t([[1.0, 0.0, 0.0]])    # exit also +x (back-scatter setup)
    mu = _t(10.0)                   # cm⁻¹
    I_abs = predicted_spot_intensities(
        V, K, I_th, spot_ring, spot_grain, spot_scan, spot_omega,
        sg, TopHat(20.0),
        use_absorption=True,
        incident_dirs_per_spot=inc,
        diffracted_dirs_per_spot=dif,
        mu_per_cm=mu,
    )
    # L_in = path from voxel 0 along -inc = -x = 5 µm
    # L_out = path along dif = +x = 45 µm
    # Total = 50 µm = 5e-3 cm, A = exp(-10*5e-3) = exp(-0.05)
    expected_ratio = math.exp(-0.05)
    assert math.isclose(
        float(I_abs[0]) / float(I_noabs[0]), expected_ratio, rel_tol=1e-9
    )


def test_forward_absorption_missing_args_raises():
    sg = _slab_5x1x1()
    sg.grain_map = torch.tensor([0, -1, -1, -1, -1], dtype=torch.int64)
    V = _t([1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="requires incident_dirs"):
        predicted_spot_intensities(
            V, _t([1.0]), _t([1.0]),
            torch.tensor([0]), torch.tensor([0]),
            _t([0.0]), _t([math.pi/2]),
            sg, TopHat(20.0),
            use_absorption=True,
        )


def test_forward_absorption_gradflow_to_mu():
    sg = _slab_5x1x1()
    sg.grain_map = torch.tensor([0, -1, -1, -1, -1], dtype=torch.int64)
    V = _t([1.0, 0.0, 0.0, 0.0, 0.0])
    K = _t([1.0]); I_th = _t([1.0])
    inc = _t([[1.0, 0.0, 0.0]])
    dif = _t([[1.0, 0.0, 0.0]])
    mu = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    I = predicted_spot_intensities(
        V, K, I_th, torch.tensor([0]), torch.tensor([0]),
        _t([0.0]), _t([math.pi/2]), sg, TopHat(20.0),
        use_absorption=True,
        incident_dirs_per_spot=inc, diffracted_dirs_per_spot=dif, mu_per_cm=mu,
    )
    I.sum().backward()
    assert mu.grad is not None
    assert math.isfinite(float(mu.grad))


# ----------------------------------------------------------- device portability


@pytest.mark.parametrize("device", _devices())
def test_path_length_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    sg = SampleGrid.from_regular_grid(
        grid_origin_um=[0.0, 0.0, 0.0], grid_shape=(3, 1, 1),
        voxel_size_um=10.0,
        device=device, dtype=dtype,
    )
    L = path_length_in_sample(
        sg.grid_origin_um, sg.grid_shape, sg.voxel_size_um, sg.sample_mask,
        torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device),
    )
    assert L.device.type == torch.device(device).type
    # Bbox [-5, 25], origin at 0 -> path forward = 25 µm
    tol = 1e-9 if dtype == torch.float64 else 1e-3
    assert math.isclose(float(L[0]), 25.0, abs_tol=tol * 25)


@pytest.mark.parametrize("device", _devices())
def test_absorption_factor_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    sg = SampleGrid.from_regular_grid(
        grid_origin_um=[0.0, 0.0, 0.0], grid_shape=(3, 1, 1),
        voxel_size_um=10.0,
        device=device, dtype=dtype,
    )
    A = absorption_factor(
        sg,
        torch.tensor([0], device=device),
        torch.tensor([[-1.0, 0.0, 0.0]], dtype=dtype, device=device),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device),
        torch.tensor(10.0, dtype=dtype, device=device),
    )
    assert A.device.type == torch.device(device).type
    assert 0.0 < float(A[0]) <= 1.0
