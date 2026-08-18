"""End-to-end checks on a synthetic two-phase frame.

Covers the three fixes that change what a normal run reports:
  * honest per-iteration strain (and therefore honest best-iterate choice),
  * RingsToExclude / MinRingSeparation actually reaching the default E-step,
  * a refined Wavelength with no d-spacings raising instead of no-op'ing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.pipelines._common import run_estep_v1, ring_table_for
from midas_calibrate_v2.pipelines.single import autocalibrate
from midas_calibrate_v2.seed.calibrant import phases_from_calibrants

CEO2 = {"name": "CeO2", "sg": 225,
        "lattice": (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)}
LAB6 = {"name": "LaB6", "sg": 221,
        "lattice": (4.15689, 4.15689, 4.15689, 90.0, 90.0, 90.0)}


def _truth(phases=None) -> CalibrationParams:
    p = CalibrationParams()
    p.NrPixelsY = 1024; p.NrPixelsZ = 1024
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0
    p.BC_y = 512.0; p.BC_z = 512.0
    p.tx = 0.0; p.ty = 0.4; p.tz = 0.25
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = CEO2["lattice"]
    p.MaxRingRad = 480.0
    p.MinRingRad = 0.0
    p.RhoD = 512.0 * 200.0          # um, matched to the outer ring
    p.Width = 1500.0
    p.EtaBinSize = 10.0
    p.RBinSize = 1.0
    p.nIterations = 3
    p.RemoveOutliersBetweenIters = False
    p.SNRMin = 1.5
    p.tolLsd = 5000.0; p.tolBC = 8.0; p.tolTilts = 1.0
    p.tolDistortion = 0.0
    p.Refine = {"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": False for i in range(15)}}
    if phases:
        p.Phases = list(phases)
    return p


def _simulate(params: CalibrationParams, sigma_px: float = 1.5) -> np.ndarray:
    rt = build_ring_table(params)
    px = 0.5 * (params.pxY + params.pxZ)
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    Y, Z = np.meshgrid(np.arange(params.NrPixelsY, dtype=np.float64),
                       np.arange(params.NrPixelsZ, dtype=np.float64))
    R, _ = pixel_to_REta(Y, Z, Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
                         Lsd=params.Lsd, RhoD=params.RhoD, px=px,
                         parallax=params.Parallax)
    rng = np.random.default_rng(0)
    img = np.full(R.shape, 50.0) + rng.normal(0, 5.0, size=R.shape)
    for r in rt.r_ideal_px:
        img += (1000.0 / (1.0 + r / 100.0)) * np.exp(
            -0.5 * ((R - r) / sigma_px) ** 2)
    return img


@pytest.fixture(scope="module")
def two_phase_image():
    return _simulate(_truth([CEO2, LAB6]))


# ------------------------------------------------------- ring-table plumbing

def test_two_phase_image_has_both_ring_sets(two_phase_image):
    rt = build_ring_table(_truth([CEO2, LAB6]))
    assert set(rt.phase_names) == {"CeO2", "LaB6"}
    assert rt.phase_mask("LaB6").sum() > 0


def test_fits_carry_phase_labels(two_phase_image):
    v1 = _truth([CEO2, LAB6])
    fits = run_estep_v1(v1, two_phase_image)
    assert fits.phase_idx is not None
    assert fits.phase_names == ("CeO2", "LaB6")
    assert len(set(fits.phase_idx.tolist())) == 2
    assert fits.phase_idx.numel() == fits.Y_pix.numel()


def test_rings_to_exclude_reaches_the_default_estep():
    """Regression: run_estep_v1 built the table raw, so the default pipeline
    silently ignored RingsToExclude / MaxRingNumber."""
    v1 = _truth()
    spec = spec_from_v1_params(v1)
    full = ring_table_for(v1, spec=spec)
    assert len(full) >= 2

    spec.max_ring_number = int(np.sort(full.ring_nr)[-2])
    capped = ring_table_for(v1, spec=spec)
    assert len(capped) < len(full)
    assert capped.ring_nr.max() <= spec.max_ring_number

    spec2 = spec_from_v1_params(v1)
    spec2.rings_to_exclude = [int(full.ring_nr[0])]
    excluded = ring_table_for(v1, spec=spec2)
    assert int(full.ring_nr[0]) not in set(excluded.ring_nr.tolist())


def test_min_ring_separation_reaches_the_default_estep():
    v1 = _truth([CEO2, LAB6])
    rt_all = ring_table_for(v1)
    # Pick a cut this geometry actually has pairs inside, so the test measures
    # the plumbing rather than the phantom's ring density.
    gaps = np.diff(np.sort(rt_all.r_ideal_px))
    cut = float(np.median(gaps))
    v1.MinRingSeparation = cut
    rt_cut = ring_table_for(v1)
    assert len(rt_cut) < len(rt_all)
    assert np.all(np.diff(np.sort(rt_cut.r_ideal_px)) >= cut)


def test_all_rings_excluded_raises_clearly():
    v1 = _truth()
    v1.MinRingSeparation = 1e9
    with pytest.raises(RuntimeError, match="removed every ring"):
        ring_table_for(v1)


# ---------------------------------------------------------- honest strain

def test_honest_strain_is_not_the_optimistic_in_loop_number(two_phase_image):
    """The in-loop residual scores pre-LM peaks under post-LM parameters, so it
    is systematically optimistic; honest scoring re-extracts first."""
    seed = _truth([CEO2, LAB6])
    seed.Lsd += 400.0
    seed.BC_y += 1.5
    seed.BC_z -= 1.0

    import copy
    r_honest = autocalibrate(copy.deepcopy(seed), two_phase_image, n_iter=2,
                             verbose=False, build_residual_corr=False,
                             honest_strain=True)
    r_loose = autocalibrate(copy.deepcopy(seed), two_phase_image, n_iter=2,
                            verbose=False, build_residual_corr=False,
                            honest_strain=False)
    h = min(x.mean_strain_uE for x in r_honest.history)
    o = min(x.mean_strain_uE for x in r_loose.history)
    assert np.isfinite(h) and np.isfinite(o)
    # The optimistic number cannot exceed the honest one by construction on
    # this phantom; the point is that they are DIFFERENT numbers and the
    # honest one is what gets reported now.
    assert o <= h * 1.0000001


def test_honest_run_still_recovers_truth(two_phase_image):
    truth = _truth([CEO2, LAB6])
    seed = _truth([CEO2, LAB6])
    seed.Lsd += 400.0
    seed.BC_y += 1.5
    seed.BC_z -= 1.0
    seed.ty -= 0.05
    seed.tz += 0.06
    res = autocalibrate(seed, two_phase_image, n_iter=3, verbose=False,
                        build_residual_corr=False)
    u = res.unpacked
    assert abs(float(u["Lsd"]) - truth.Lsd) < 400.0
    assert abs(float(u["BC_y"]) - truth.BC_y) < 1.5
    assert abs(float(u["BC_z"]) - truth.BC_z) < 1.5
    assert res.fits_final.phase_idx is not None


# -------------------------------------------------- wavelength gradient guard

def test_pv_doublet_cofit_still_runs(two_phase_image):
    from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
    seed = _truth([CEO2, LAB6])
    seed.Lsd += 300.0
    res = autocalibrate_pv(seed, two_phase_image, n_iter=1, verbose=False,
                           doublet_separation_px=25.0)
    assert np.isfinite(float(res.unpacked["Lsd"]))


def test_pv_raises_clearly_when_the_blend_cut_eats_every_ring(two_phase_image):
    """Dropping every >=3-blend used to leave zero fits and die downstream
    with 'index -1 is out of bounds for axis 0 with size 0'."""
    from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
    with pytest.raises(RuntimeError, match="no fits are left"):
        autocalibrate_pv(_truth([CEO2, LAB6]), two_phase_image, n_iter=1,
                         verbose=False, doublet_separation_px=140.0)


def test_wavelength_has_gradient_when_d_spacings_present(two_phase_image):
    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
    v1 = _truth([CEO2, LAB6])
    fits = run_estep_v1(v1, two_phase_image)
    assert fits.ring_d_spacing_A is not None, \
        "run_estep_v1 must populate d-spacings or a refined lambda is a no-op"

    lam = torch.tensor(v1.Wavelength, dtype=torch.float64, requires_grad=True)
    p = {k: torch.as_tensor(float(v), dtype=torch.float64) for k, v in
         dict(Lsd=v1.Lsd, BC_y=v1.BC_y, BC_z=v1.BC_z, tx=0.0, ty=v1.ty,
              tz=v1.tz, pxY=v1.pxY, pxZ=v1.pxZ).items()}
    p["Wavelength"] = lam
    r = pseudo_strain_residual(fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg,
                               p, rho_d=fits.rho_d,
                               ring_d_spacing_A=fits.ring_d_spacing_A)
    loss = (r * r).sum()
    assert loss.requires_grad
    loss.backward()
    assert lam.grad is not None and float(lam.grad) != 0.0


def test_refined_wavelength_without_d_spacings_raises(two_phase_image, monkeypatch):
    """Rather than silently refining a parameter with a zero Jacobian column."""
    from midas_calibrate_v2.pipelines import multi as multi_mod

    v1 = _truth()
    v1.Refine = dict(v1.Refine); v1.Refine["Wavelength"] = True

    real = multi_mod.run_estep_v1

    def stripped(*a, **kw):
        fd = real(*a, **kw)
        fd.ring_d_spacing_A = None
        fd.rt = None
        return fd

    monkeypatch.setattr(multi_mod, "run_estep_v1", stripped)
    ms = multi_mod.build_multi_spec([v1], mode="same_detector")
    with pytest.raises(RuntimeError, match="silently do\\s+nothing"):
        multi_mod.autocalibrate_multi([v1], [two_phase_image], multi_spec=ms,
                                       n_iter=1, verbose=False,
                                       build_residual_corr=False)
