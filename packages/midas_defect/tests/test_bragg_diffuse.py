"""Tests for bragg_diffuse: reflection prediction + Bragg/diffuse classification."""
import numpy as np
import pytest
import torch

from midas_defect.lattice import fcc_cu_crystal, cual2_crystal
from midas_defect.bragg_diffuse import (
    enumerate_hkls, predicted_reflection_points, classify_voxels,
    on_lattice_fraction,
)


def test_enumerate_hkls_fcc_parity():
    """FCC allows only all-even / all-odd; no mixed-parity reflection appears."""
    hkls = enumerate_hkls(fcc_cu_crystal(), q_max_inv_A=6.0)
    assert len(hkls) > 0
    par = hkls % 2
    all_even = (par == 0).all(axis=1)
    all_odd = (par == 1).all(axis=1)
    assert (all_even | all_odd).all(), "mixed-parity reflection leaked into FCC set"
    # full multiplicity, not ASU reps: {111} has 8 signed variants
    n111 = ((np.abs(hkls) == 1).all(axis=1)).sum()
    assert n111 == 8


def test_enumerate_hkls_phase_agnostic():
    """CuAl2 (denser, lower symmetry) yields more reflections than FCC."""
    n_fcc = len(enumerate_hkls(fcc_cu_crystal(), q_max_inv_A=6.0))
    n_cual2 = len(enumerate_hkls(cual2_crystal(), q_max_inv_A=6.0))
    assert n_cual2 > n_fcc


def test_predicted_points_shape_and_grad(_device_param, _dtype_param):
    """Prediction is differentiable in the orientation and runs on any device."""
    if _device_param.type == "mps" and _dtype_param == torch.float64:
        pytest.skip("MPS does not support float64")
    cr = fcc_cu_crystal()
    U = torch.eye(3, device=_device_param, dtype=_dtype_param).reshape(1, 3, 3)
    U.requires_grad_(True)
    P = predicted_reflection_points(U, cr, q_max_inv_A=4.0, device=_device_param,
                                    dtype=_dtype_param)
    assert P.shape[1] == 3 and P.shape[0] > 0
    P.sum().backward()
    assert U.grad is not None and torch.isfinite(U.grad).all()


def test_classify_splits_bragg_from_diffuse():
    cr = fcc_cu_crystal()
    P = predicted_reflection_points(np.eye(3)[None], cr, q_max_inv_A=6.0).numpy()
    # half the voxels exactly on lattice, half pushed 0.3 1/A off radially
    on = P[:10]
    off = P[:10] * (1.0 + 0.3 / np.linalg.norm(P[:10], axis=1, keepdims=True))
    q = np.vstack([on, off])
    I = np.concatenate([np.full(10, 5.0), np.full(10, 5.0)])
    split = classify_voxels(q, I, P, tol_inv_A=0.05)
    assert split.on_lattice[:10].all()
    assert not split.on_lattice[10:].any()
    assert abs(split.bragg_intensity_frac - 0.5) < 1e-9
    assert abs(split.bragg_intensity_frac + split.diffuse_intensity_frac - 1.0) < 1e-12


def test_on_lattice_fraction_high_for_correct_geometry():
    """Bright voxels on the lattice ⇒ ~1.0; scrambled directions ⇒ near chance."""
    cr = fcc_cu_crystal()
    P = predicted_reflection_points(np.eye(3)[None], cr, q_max_inv_A=8.0).numpy()
    rng = np.random.default_rng(0)
    # bright voxels sit on reflections; dim background scattered randomly
    bright = P + rng.normal(0, 0.01, P.shape)
    bg = rng.uniform(-5, 5, (2000, 3))
    q = np.vstack([bright, bg])
    I = np.concatenate([np.full(len(bright), 1000.0), rng.uniform(1, 5, len(bg))])
    frac_ok = on_lattice_fraction(q, I, P, bright_percentile=99.0, tol_inv_A=0.1)
    assert frac_ok > 0.9
    # scrambled geometry: a non-symmetry rotation (37° about an arbitrary axis)
    # moves reflections off the lattice (a cubic-symmetry angle would not).
    ang = np.deg2rad(37.0)
    axis = np.array([0.3, 0.5, 0.81]); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rscr = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    frac_bad = on_lattice_fraction((q @ Rscr.T), I, P, bright_percentile=99.0, tol_inv_A=0.1)
    assert frac_bad < frac_ok
