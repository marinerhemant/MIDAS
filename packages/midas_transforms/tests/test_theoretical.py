"""Tests for ``midas_transforms.radius.theoretical``.

Covers:
- Per-ring intensity via ``intensity_from_crystal`` on HCP Ti (physical sanity).
- per_spot_relative_volume — exact division by per-ring reference.
- aggregate_per_voxel / aggregate_per_grain — mean (autograd-safe) + median (hard).
- Differentiability:
    * torch.autograd.gradcheck on (lattice_params, B_iso) → I_ring
    * grad flow from V_rel → wavelength
- Device portability: CPU + MPS (forward), CPU (gradcheck — MPS lacks fp64).
- Bias check vs empirical reference for a few-spot scenario (the *motivation*
  for theoretical normalization).
"""
from __future__ import annotations

import os
# OpenMP runtime collision between torch + numpy on macOS conda envs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, generate_hkls

from midas_transforms.radius import (
    RingTable,
    SpotTensors,
    aggregate_per_grain,
    aggregate_per_voxel,
    load_rings_from_hkls_csv,
    load_spots_from_input_extra_info_csvs,
    per_spot_relative_volume,
    theoretical_intensity_per_ring,
)


# --------------------------------------------------------------- fixtures


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _make_ti_crystal(dtype=torch.float64, device="cpu"):
    """Synthetic CP-Ti — HCP P63/mmc (SG 194), a=2.9508 Å, c=4.6855 Å."""
    sg = SpaceGroup.from_number(194)
    lat = Lattice(2.9508, 2.9508, 4.6855, 90.0, 90.0, 120.0)
    ti = Atom("Ti", (1.0 / 3.0, 2.0 / 3.0, 0.25), B_iso=0.5)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[ti])
    return xt, xt.to_torch(dtype=dtype, device=device)


def _make_ti_ring_table(*, dtype=torch.float64, device="cpu",
                        wavelength_A=0.173, two_theta_max_deg=15.0):
    xt, _ = _make_ti_crystal()
    refs = generate_hkls(
        xt.space_group, xt.lattice,
        wavelength_A=wavelength_A, two_theta_max_deg=two_theta_max_deg,
    )
    uniq = sorted({round(r.two_theta_deg, 4) for r in refs})
    return RingTable(
        ring_numbers=torch.arange(1, len(uniq) + 1, dtype=torch.int64, device=device),
        two_theta_deg=torch.tensor(uniq, dtype=dtype, device=device),
    )


# --------------------------------------------------------------- per-ring I


def test_per_ring_intensity_positive_finite():
    _, xt_t = _make_ti_crystal()
    ring_table = _make_ti_ring_table()
    lam = torch.tensor(0.173, dtype=torch.float64)
    I = theoretical_intensity_per_ring(xt_t, lam, ring_table, two_theta_max_deg=15.0)
    assert I.shape == ring_table.two_theta_deg.shape
    assert torch.isfinite(I).all()
    assert (I > 0).any(), "expected nonzero intensity on at least one HCP-Ti ring"


def test_per_ring_intensity_matches_intensity_from_crystal_sum():
    """Ring totals must equal the sum of per-HKL |F|²·m·Lp from midas-hkls."""
    from midas_hkls.intensity import intensity_from_crystal

    xt, xt_t = _make_ti_crystal()
    ring_table = _make_ti_ring_table()
    lam = torch.tensor(0.173, dtype=torch.float64)
    refs = generate_hkls(
        xt.space_group, xt.lattice,
        wavelength_A=0.173, two_theta_max_deg=15.0,
    )
    _, I_per_hkl = intensity_from_crystal(xt_t, refs, wavelength_A=lam)

    I_ring = theoretical_intensity_per_ring(
        xt_t, lam, ring_table, two_theta_max_deg=15.0
    )
    # Reference computed by Python loop using identical scatter rule.
    ref = np.zeros(len(ring_table.two_theta_deg), dtype=np.float64)
    rt = ring_table.two_theta_deg.numpy()
    for k, r in enumerate(refs):
        j = int(np.abs(rt - r.two_theta_deg).argmin())
        if abs(rt[j] - r.two_theta_deg) < 0.05:
            ref[j] += float(I_per_hkl[k].detach())
    assert np.allclose(I_ring.numpy(), ref, rtol=1e-12, atol=1e-9)


def test_per_ring_intensity_lambda_scales():
    """Halving λ should keep rings non-negative and finite (different 2θ
    coverage may shift which rings are populated, but no NaNs)."""
    _, xt_t = _make_ti_crystal()
    ring_table = _make_ti_ring_table(two_theta_max_deg=10.0)
    for lam in (0.05, 0.10, 0.173, 0.25):
        I = theoretical_intensity_per_ring(
            xt_t, torch.tensor(lam, dtype=torch.float64),
            ring_table, two_theta_max_deg=10.0,
        )
        assert torch.isfinite(I).all(), f"NaN at λ={lam}"
        assert (I >= 0).all(),         f"negative I at λ={lam}"


# --------------------------------------------------------------- V_rel kernel


def test_v_rel_exact_division():
    """V_rel = I_obs / I_ring_theory (per spot)."""
    I_ring = torch.tensor([10.0, 4.0, 0.0, 100.0], dtype=torch.float64)
    spot_ring = torch.tensor([0, 1, 1, 3, -1, 2], dtype=torch.int64)
    spot_I = torch.tensor([10.0, 8.0, 2.0, 50.0, 7.0, 5.0], dtype=torch.float64)
    V = per_spot_relative_volume(spot_ring, spot_I, I_ring)
    # spot 0: 10/10 = 1
    # spot 1: 8/4  = 2
    # spot 2: 2/4  = 0.5
    # spot 3: 50/100 = 0.5
    # spot 4: ring_idx = -1 -> 0
    # spot 5: ring=2 has I_ring=0 -> 0
    assert V.tolist() == [1.0, 2.0, 0.5, 0.5, 0.0, 0.0]


def test_v_rel_grad_flows_to_wavelength():
    """Gradient flows: V_rel.sum() → I_ring → λ (via midas-hkls)."""
    _, xt_t = _make_ti_crystal()
    ring_table = _make_ti_ring_table()
    lam = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)

    I_ring = theoretical_intensity_per_ring(
        xt_t, lam, ring_table, two_theta_max_deg=15.0
    )
    spot_ring = torch.tensor([0, 1, 2], dtype=torch.int64)
    spot_I = torch.tensor([float(I_ring[0].detach()),
                           float(I_ring[1].detach()),
                           float(I_ring[2].detach())], dtype=torch.float64)
    V = per_spot_relative_volume(spot_ring, spot_I, I_ring)
    V.sum().backward()
    assert lam.grad is not None
    assert math.isfinite(float(lam.grad))


# --------------------------------------------------------------- gradcheck


def test_gradcheck_per_ring_intensity_wrt_lattice_a():
    """gradcheck on a → I_ring.sum().  Uses small Ti subset, fp64."""
    sg = SpaceGroup.from_number(194)
    a0, c0 = 2.9508, 4.6855

    def f(a):
        lat = Lattice(float(a.detach()), float(a.detach()), c0, 90.0, 90.0, 120.0)
        xt = Crystal(
            lattice=lat, space_group=sg,
            atoms=[Atom("Ti", (1/3, 2/3, 0.25), B_iso=0.5)],
        )
        # Build a CrystalTensor that takes `a` as the differentiable handle.
        xt_t_static = xt.to_torch(dtype=torch.float64)
        # Replace lattice_params with [a, a, c0, 90, 90, 120] tied to grad input
        lp = torch.stack([
            a, a,
            torch.tensor(c0, dtype=torch.float64),
            torch.tensor(90.0, dtype=torch.float64),
            torch.tensor(90.0, dtype=torch.float64),
            torch.tensor(120.0, dtype=torch.float64),
        ])
        # Swap lattice handle (dataclasses.replace would lose ASU links)
        xt_t = xt_t_static
        xt_t.lattice_params = lp

        rt = _make_ti_ring_table(two_theta_max_deg=10.0)
        I = theoretical_intensity_per_ring(xt_t, torch.tensor(0.173, dtype=torch.float64), rt,
                                            two_theta_max_deg=10.0)
        return I.sum()

    a_in = torch.tensor(2.9508, dtype=torch.float64, requires_grad=True)
    # Loose tolerance: HKL enumeration is discrete + non-differentiable; we only
    # check that the differentiable continuation through |F|²·Lp is consistent.
    assert torch.autograd.gradcheck(f, (a_in,), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_gradcheck_per_ring_intensity_wrt_wavelength():
    """gradcheck on λ → I_ring.sum().  HKL enumeration uses detached λ inside
    ``theoretical_intensity_per_ring``; the continuous λ flows through Lp + |F|²."""
    _, xt_t = _make_ti_crystal()
    rt = _make_ti_ring_table(two_theta_max_deg=10.0, wavelength_A=0.173)

    def f(lam):
        I = theoretical_intensity_per_ring(xt_t, lam, rt, two_theta_max_deg=10.0)
        return I.sum()

    lam = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (lam,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_v_rel_wrt_intensity_inputs():
    """Pure tensor kernel — gradcheck on (spot_I, I_ring)."""
    I_ring = torch.tensor([10.0, 4.0, 1.0], dtype=torch.float64, requires_grad=True)
    spot_I = torch.tensor([8.0, 2.0, 0.7, 5.0], dtype=torch.float64, requires_grad=True)
    spot_ring = torch.tensor([0, 1, 2, -1], dtype=torch.int64)

    def f(spot_I_, I_ring_):
        return per_spot_relative_volume(spot_ring, spot_I_, I_ring_).sum()

    assert torch.autograd.gradcheck(f, (spot_I, I_ring), eps=1e-6, atol=1e-7)


# --------------------------------------------------------------- aggregation


def test_aggregate_per_voxel_mean_matches_python():
    V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    vox = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    out = aggregate_per_voxel(V, vox, n_voxels=4, method="mean")
    assert torch.allclose(out, torch.tensor([1.5, 3.5, 5.0, 0.0], dtype=torch.float64))


def test_aggregate_per_voxel_median_matches_python():
    """``torch.median`` returns the lower of two middle values for even-sized
    inputs (hard pick, non-interpolated) — this is intentional: median is for
    reporting, not refinement, and we want consistent values across devices."""
    V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    vox = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64)
    out = aggregate_per_voxel(V, vox, n_voxels=3, method="median")
    # vox 0: [1,2,3] -> 2 ; vox 1: [4,5] -> 4 (lower-mid) ; vox 2: [6] -> 6
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64))


def test_aggregate_mean_grad_flows():
    """Mean aggregation is differentiable end-to-end."""
    V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)
    vox = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    out = aggregate_per_voxel(V, vox, n_voxels=3, method="mean")
    out.sum().backward()
    # mean over equal counts → grad = 1/group_size for each spot
    assert torch.allclose(V.grad, torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0], dtype=torch.float64))


def test_aggregate_per_grain_mean_unknown_method_raises():
    V = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="unknown aggregation method"):
        aggregate_per_grain(V, g, n_grains=2, method="bogus")


# --------------------------------------------------------------- bias scenario


def test_theoretical_unbiased_for_few_spot_sample():
    """The whole motivation: with a handful of grain spots the empirical
    per-ring powder estimate is a biased reference, but the theoretical
    reference is not.  We don't refit anything here — just demonstrate the
    bias quantitatively for a synthetic 1-grain Ti sample."""
    _, xt_t = _make_ti_crystal()
    rt = _make_ti_ring_table(two_theta_max_deg=10.0)
    I_theory = theoretical_intensity_per_ring(
        xt_t, torch.tensor(0.173, dtype=torch.float64), rt, two_theta_max_deg=10.0
    )
    populated = (I_theory > 0).nonzero().squeeze(-1)
    assert populated.numel() >= 3, "need >=3 rings for the bias demo"

    # Single grain contributes one randomly chosen spot per populated ring.
    # Observed = grain_volume * I_theory_per_ring  (plus noise factor)
    grain_vol_true = 1.0
    g = torch.Generator().manual_seed(0)
    noise = 1.0 + 0.05 * torch.randn(populated.numel(), generator=g, dtype=torch.float64)
    spot_I = I_theory[populated] * grain_vol_true * noise
    V_th = per_spot_relative_volume(populated, spot_I, I_theory)

    # Empirical reference if we naively normalized by observed sum:
    I_emp = torch.zeros_like(I_theory)
    I_emp[populated] = spot_I  # single spot per ring → "powder" = same spot
    V_emp = per_spot_relative_volume(populated, spot_I, I_emp)

    # Theoretical: V_rel ≈ 1 ± noise
    assert torch.allclose(V_th, torch.ones_like(V_th), atol=0.20), \
        f"V_theory deviates: {V_th.tolist()}"
    # Empirical: V_rel ≡ 1 by tautology (no information!).
    assert torch.allclose(V_emp, torch.ones_like(V_emp), atol=1e-12), \
        "empirical few-grain normalization is trivially 1 — illustrating the bias"


# --------------------------------------------------------------- device portability


@pytest.mark.parametrize("device", _devices())
def test_per_ring_intensity_runs_on_device(device):
    """Forward pass on every device.  MPS uses fp32 (no fp64)."""
    dtype = torch.float64 if device != "mps" else torch.float32
    if device == "mps":
        # midas-hkls metric_tensor() currently forces fp64 (lattice_torch.py:31
        # default fallback), which MPS does not support.  Until that's fixed
        # upstream, the per-ring kernel cannot run on MPS.
        pytest.skip("midas-hkls metric_tensor forces fp64; MPS lacks fp64")
    _, xt_t = _make_ti_crystal(dtype=dtype, device=device)
    rt = _make_ti_ring_table(dtype=dtype, device=device, two_theta_max_deg=10.0)
    lam = torch.tensor(0.173, dtype=dtype, device=device)
    I = theoretical_intensity_per_ring(xt_t, lam, rt, two_theta_max_deg=10.0)
    assert I.device.type == torch.device(device).type
    assert I.dtype == dtype
    assert torch.isfinite(I).all()


@pytest.mark.parametrize("device", _devices())
def test_per_spot_relative_volume_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    I_ring = torch.tensor([1.0, 2.0, 0.0], dtype=dtype, device=device)
    spot_ring = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device=device)
    spot_I = torch.tensor([0.5, 6.0, 7.0, 8.0], dtype=dtype, device=device)
    V = per_spot_relative_volume(spot_ring, spot_I, I_ring)
    assert V.device.type == torch.device(device).type
    # spot 0: 0.5/1 = 0.5; spot 1: 6/2 = 3; spot 2: I_ring=0 → 0; spot 3: idx=-1 → 0
    expected = torch.tensor([0.5, 3.0, 0.0, 0.0], dtype=dtype, device=device)
    assert torch.allclose(V, expected)


# --------------------------------------------------------------- I/O helpers


def test_load_rings_from_hkls_csv(tmp_path):
    """Round-trip: synthesize a hkls.csv and parse it back."""
    p = tmp_path / "hkls.csv"
    # Header (skipped) + 4 rows, two share ring 1, then ring 2, ring 3
    p.write_text(
        "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
        "1 0 0 2.5555 1 0 0 0 1.93975 3.8795 0.0\n"
        "0 0 2 2.3428 2 0 0 0 2.1160 4.2320 0.0\n"
        "1 0 1 2.2435 3 0 0 0 2.20965 4.4193 0.0\n"
        "2 0 0 1.2777 1 0 0 0 3.8818 7.7636 0.0\n"
    )
    rt = load_rings_from_hkls_csv(p)
    # First occurrence of each ring wins
    assert rt.ring_numbers.tolist() == [1, 2, 3]
    assert torch.allclose(
        rt.two_theta_deg,
        torch.tensor([3.8795, 4.2320, 4.4193], dtype=torch.float64),
    )


def test_load_spots_from_input_extra_info(tmp_path):
    """Round-trip per-scan spot loading.  Builds two tiny scans + verifies
    SpotID, ring_idx, intensity, omega, eta survive."""
    # Layout: 15 columns matching MIDAS InputAllExtraInfo
    # cols: YLab(0) ZLab(1) Omega(2) GrainRadius(3) SpotID(4) RingNr(5)
    #       Eta(6) Ttheta(7) OmegaIni(8) 9 10 11 12 13 IntInt(14)
    def _row(spot_id, ring_nr, omega, eta, integ_I):
        r = np.zeros(15)
        r[4] = spot_id; r[5] = ring_nr
        r[2] = omega; r[6] = eta
        r[14] = integ_I
        return r

    s1 = np.array([_row(1, 1, 12.5,  30.0, 100.0),
                   _row(2, 3, 45.0, -90.0,  50.0)])
    s2 = np.array([_row(3, 2, 78.0,  60.0,  25.0)])

    (tmp_path / "InputAllExtraInfoFittingAll1.csv").write_text(
        "% header\n" + "\n".join(" ".join(f"{v}" for v in row) for row in s1)
    )
    (tmp_path / "InputAllExtraInfoFittingAll2.csv").write_text(
        "% header\n" + "\n".join(" ".join(f"{v}" for v in row) for row in s2)
    )

    rt = RingTable(
        ring_numbers=torch.tensor([1, 2, 3], dtype=torch.int64),
        two_theta_deg=torch.tensor([3.8, 4.2, 4.4], dtype=torch.float64),
    )
    sp = load_spots_from_input_extra_info_csvs(tmp_path, ring_table=rt)
    # Order: scan 1 (sorted first), then scan 2
    assert sp.spot_id.tolist()    == [1, 2, 3]
    assert sp.scan_nr.tolist()    == [1, 1, 2]
    assert sp.ring_number.tolist() == [1, 3, 2]
    assert sp.ring_idx.tolist()    == [0, 2, 1]
    assert sp.intensity.tolist()   == [100.0, 50.0, 25.0]
    assert sp.omega_deg.tolist()   == [12.5, 45.0, 78.0]
    assert sp.eta_deg.tolist()     == [30.0, -90.0, 60.0]


def test_load_spots_from_input_extra_info_bare_header(tmp_path):
    """The unified midas_transforms writer emits a bare 'YLab ZLab …' header
    with no '%' prefix; the loader must skip it (regression for the V-map
    calc_radius_v stage failing with 'could not convert string YLab to float'
    on pipeline-written CSVs)."""
    def _row(spot_id, ring_nr, omega, eta, integ_I):
        r = np.zeros(15)
        r[4] = spot_id; r[5] = ring_nr
        r[2] = omega; r[6] = eta
        r[14] = integ_I
        return r

    s1 = np.array([_row(1, 1, 12.5, 30.0, 100.0)])
    (tmp_path / "InputAllExtraInfoFittingAll0.csv").write_text(
        "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta\n"
        + "\n".join(" ".join(f"{v}" for v in row) for row in s1)
    )
    rt = RingTable(
        ring_numbers=torch.tensor([1, 2, 3], dtype=torch.int64),
        two_theta_deg=torch.tensor([3.8, 4.2, 4.4], dtype=torch.float64),
    )
    sp = load_spots_from_input_extra_info_csvs(tmp_path, ring_table=rt)
    assert sp.spot_id.tolist() == [1]
    assert sp.intensity.tolist() == [100.0]
    assert sp.omega_deg.tolist() == [12.5]
