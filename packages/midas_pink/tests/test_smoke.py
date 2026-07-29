"""Smoke tests for midas_pink: import + minimal forward + minimal inverse."""
from __future__ import annotations

import math

import pytest
import torch


def test_package_imports():
    import midas_pink as mp
    assert hasattr(mp, "ParameterisedSpectrum")
    assert hasattr(mp, "build_pink_bank")
    assert hasattr(mp, "splat_rois")
    assert hasattr(mp, "splat_rois_3d")
    assert hasattr(mp, "recover_grain_state")
    assert hasattr(mp, "recover_joint")


def test_spectrum_construction_and_weights():
    import midas_pink as mp

    spec = mp.ParameterisedSpectrum(
        E0_keV=71.6764, half_bw=0.03, n_samples=51,
        init_kind="gaussian", init_rel_bw=1e-2,
        fixed=True, dtype=torch.float64,
    )
    Es = spec.energies_keV
    lams = spec.lambdas_A
    w = spec.weights()
    assert Es.shape == (51,)
    assert lams.shape == (51,)
    assert w.shape == (51,)
    # weights normalized
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))
    # centroid close to E0 for symmetric Gaussian
    centroid = (w * Es).sum()
    assert abs(float(centroid) - 71.6764) < 1e-4


@pytest.mark.skipif("not __import__('importlib').util.find_spec('midas_diffract')",
                    reason="midas_diffract not installed")
def test_minimal_pink_forward_roundtrip():
    """Build a small pink bank, run forward on one grain, splat ROIs.

    Validates end-to-end glue without claiming any physics result.
    """
    import midas_diffract as md
    import midas_pink as mp
    from midas_hkls import SpaceGroup, Lattice

    DEG2RAD = math.pi / 180.0

    spec = mp.ParameterisedSpectrum(
        E0_keV=71.6764, half_bw=0.02, n_samples=11,
        init_kind="delta", fixed=True, dtype=torch.float64,
    )

    def geom_factory(lam_A):
        return md.HEDMGeometry(
            Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
            omega_start=-180.0, omega_step=0.25, n_frames=1440,
            n_pixels_y=2048, n_pixels_z=2048,
            min_eta=6.0, wavelength=lam_A,
        )

    bank = mp.build_pink_bank(
        spec,
        space_group=SpaceGroup.from_number(225),
        lattice=Lattice.for_system("cubic", a=4.078),
        geom_factory=geom_factory,
        two_theta_max_deg=6.0,
        device="cpu", dtype=torch.float64,
    )
    assert len(bank.models) == 11
    assert bank.hkls_int.shape[0] > 0

    euler = torch.tensor([45.0, 30.0, 60.0],
                          dtype=torch.float64) * DEG2RAD
    pos = torch.zeros(3, dtype=torch.float64)
    latc = torch.tensor([4.078] * 3 + [90.0] * 3, dtype=torch.float64)

    plan = mp.plan_rois_from_state(
        bank, euler.unsqueeze(0), pos.unsqueeze(0),
        lattice_params=latc, roi_h=11, roi_w=11,
    )
    assert plan.n_kept > 0

    rois = mp.splat_rois(
        bank, plan,
        euler.unsqueeze(0), pos.unsqueeze(0), latc,
        sigma_psf_px=1.5,
    )
    assert rois.shape == (plan.n_kept, 11, 11)
    assert torch.isfinite(rois).all()
    assert float(rois.max()) > 0
