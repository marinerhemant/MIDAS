"""Phase-2 gates for the atomistic SAXS source (PREREGISTER_atomistic.md).

G1  estimator consistency (consistency only, not truth): direct sum vs a brute-force numpy double
    loop, and FFT path vs direct inside the registered range |q| <= q_Nyquist / 4.
G4  (unit part) representation invariance: translation, wrapping, atom order, supercell.
G2  (unit part) a single removed atom gives I = f(q)^2, the absolute-normalisation control.

Boxes here are small on purpose: this suite has to run on a laptop. The big-box timing and the
void / loop / disorder gates are separate registered runs.
"""
import math

import numpy as np
import pytest
import torch

from midas_hkls.form_factors import form_factor
from midas_saxs import SAXSGeometry, simulate_frame
from midas_saxs.atomistic import (
    atomistic_intensity,
    default_resolution_fwhm_inv_A,
    lattice_amplitudes,
    reciprocal_basis,
)
from midas_saxs.md_io import AtomicFrame

pytestmark = pytest.mark.unit

A_CU = 3.615
FCC = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def cu_box(nc=(6, 6, 6), *, void_radius=0.0, shake=0.0, seed=0, drop=()):
    cells = np.array([[i, j, k] for i in range(nc[0]) for j in range(nc[1]) for k in range(nc[2])])
    pos = ((cells[:, None, :] + FCC[None]) * A_CU).reshape(-1, 3)
    cell = np.diag(np.array(nc) * A_CU)
    if void_radius > 0:
        ctr = np.array(nc) * A_CU / 2
        pos = pos[np.linalg.norm(pos - ctr, axis=1) > void_radius]
    if drop:
        pos = np.delete(pos, list(drop), axis=0)
    if shake:
        pos = pos + shake * np.random.default_rng(seed).standard_normal(pos.shape)
    return AtomicFrame(positions=pos, species=np.full(len(pos), "Cu"), cell=cell)


def lattice_h(rmax):
    return np.array([[i, j, k] for i in range(-rmax, rmax + 1) for j in range(-rmax, rmax + 1)
                     for k in range(-rmax, rmax + 1) if (i, j, k) != (0, 0, 0)])


# ----------------------------------------------------------------------------- G1

def test_direct_sum_matches_brute_force_numpy_on_a_triclinic_cell():
    rng = np.random.default_rng(1)
    cell = np.array([[17.0, 0, 0], [3.1, 19.0, 0], [-2.2, 4.4, 23.0]])
    n = 300
    pos = rng.random((n, 3)) @ cell
    species = np.where(rng.random(n) < 0.7, "Cu", "He")
    f = AtomicFrame(positions=pos, species=species, cell=cell)
    h = lattice_h(3)
    A, used = lattice_amplitudes(f, h, method="direct")
    assert used == "direct"
    G = h @ reciprocal_basis(cell)
    gn = np.linalg.norm(G, axis=1)
    ref = np.zeros(len(h), dtype=complex)
    for sym in ("Cu", "He"):
        ff = form_factor((gn / (4 * math.pi)) ** 2, sym)
        ref += np.array([ff[i] * np.exp(1j * (pos[species == sym] @ G[i])).sum()
                         for i in range(len(h))])
    err = np.abs(A.numpy() - ref).max() / np.abs(ref).max()
    assert err < 1e-10, err


def test_fft_path_matches_direct_in_the_registered_range():
    """Median < 1 % for |q| <= q_Nyquist / 4 (|h_i| <= n_i / 8), on a box with a void and noise."""
    f = cu_box((10, 10, 10), void_radius=6.0, shake=0.05)      # n = 40 -> |h| <= 5
    h = lattice_h(5)
    Ad, _ = lattice_amplitudes(f, h, method="direct")
    Af, used = lattice_amplitudes(f, h, method="fft")
    assert used == "fft"
    Id, If = np.abs(Ad.numpy()) ** 2, np.abs(Af.numpy()) ** 2
    assert Id.min() > 0
    rel = np.abs(If - Id) / Id
    assert np.median(rel) < 1e-2, np.median(rel)
    assert rel.max() < 1e-1, rel.max()


def test_fft_path_refuses_points_outside_its_range():
    f = cu_box((10, 10, 10), void_radius=6.0)
    with pytest.raises(ValueError, match="valid only for"):
        lattice_amplitudes(f, lattice_h(8), method="fft")


def test_fft_path_works_on_a_triclinic_cell():
    cell = np.array([[40.0, 0, 0], [6.0, 42.0, 0], [-5.0, 4.0, 44.0]])
    rng = np.random.default_rng(3)
    f = AtomicFrame(positions=rng.random((3000, 3)) @ cell, species=np.full(3000, "Cu"), cell=cell)
    h = lattice_h(4)
    Ad, _ = lattice_amplitudes(f, h, method="direct")
    Af, _ = lattice_amplitudes(f, h, method="fft")
    rel = np.abs(np.abs(Af.numpy()) ** 2 - np.abs(Ad.numpy()) ** 2) / np.abs(Ad.numpy()) ** 2
    assert np.median(rel) < 1e-2


def test_bspline_deposit_conserves_charge():
    from midas_saxs.atomistic import _deposit
    s = torch.rand(500, 3, dtype=torch.float64)
    g = _deposit(s, (12, 14, 15), 6)
    assert abs(float(g.sum()) - 500) < 1e-9


# ----------------------------------------------------------------------------- G4 (unit part)

def _rel_change(a, b):
    sig = max(float(a.max()), float(b.max()))
    m = (a > 1e-8 * sig) | (b > 1e-8 * sig)             # only where there is signal above floor
    assert m.sum() > 20
    return float((np.abs(a - b)[m] / np.maximum(a, b)[m]).max())


def test_amplitude_intensity_is_invariant_to_translation_wrapping_and_order():
    f = cu_box((6, 6, 6), void_radius=5.0, shake=0.1)
    h = lattice_h(4)
    A0, _ = lattice_amplitudes(f, h, method="direct")
    I0 = np.abs(A0.numpy()) ** 2

    shifted = AtomicFrame(positions=f.positions + np.array([3.7, -11.2, 5.5]),
                          species=f.species, cell=f.cell)
    wrapped = AtomicFrame(positions=f.wrapped() + np.array([1, -2, 3]) * np.diag(f.cell),
                          species=f.species, cell=f.cell)
    perm = np.random.default_rng(5).permutation(f.n_atoms)
    permuted = AtomicFrame(positions=f.positions[perm], species=f.species[perm], cell=f.cell)
    for g in (shifted, wrapped, permuted):
        A, _ = lattice_amplitudes(g, h, method="direct")
        assert _rel_change(I0, np.abs(A.numpy()) ** 2) < 1e-9


def test_per_volume_intensity_is_invariant_to_the_supercell():
    """Same structure stored in a 2x1x1 and a 2x2x2 cell: the per-A^3 intensity is unchanged."""
    cell = np.array([[16.0, 0, 0], [0, 17.0, 0], [0, 0, 18.0]])
    rng = np.random.default_rng(11)
    pos = rng.random((60, 3)) @ cell
    sp = np.full(60, "Cu")
    base = AtomicFrame(positions=pos, species=sp, cell=cell)

    def tile(nx, ny, nz):
        shifts = np.array([[i, j, k] for i in range(nx) for j in range(ny) for k in range(nz)])
        P = (pos[None] + shifts[:, None, :] * np.diag(cell)).reshape(-1, 3)
        return AtomicFrame(positions=P, species=np.full(len(P), "Cu"),
                           cell=np.diag(np.diag(cell) * [nx, ny, nz]))

    fwhm = default_resolution_fwhm_inv_A(cell)         # the coarse cell's two spacings
    rq = np.random.default_rng(2)
    qv = torch.as_tensor(rq.standard_normal((40, 3)), dtype=torch.float64)
    qv = qv / qv.norm(dim=1, keepdim=True) * torch.linspace(0.8, 1.6, 40, dtype=torch.float64)[:, None]
    I1, _ = atomistic_intensity(qv, base, resolution_fwhm_inv_A=fwhm, method="direct")
    for rep in ((2, 1, 1), (2, 2, 2)):
        Ir, _ = atomistic_intensity(qv, tile(*rep), resolution_fwhm_inv_A=fwhm, method="direct")
        rel = ((Ir - I1).abs() / I1).max().item()
        assert rel < 1e-4, (rep, rel)


# ----------------------------------------------------------------------------- G2 (unit part)

def test_one_removed_atom_scatters_f_squared():
    """Perfect Cu minus one atom: |A(G)|^2 = f(|G|)^2 away from the crystal Bragg points,
    so the window average per box is f(q)^2 (absolute normalisation, no continuum model)."""
    f = cu_box((6, 6, 6), drop=(37,))
    V = f.volume
    rng = np.random.default_rng(4)
    qv = torch.as_tensor(rng.standard_normal((60, 3)), dtype=torch.float64)
    mag = torch.linspace(0.4, 0.9, 60, dtype=torch.float64)
    qv = qv / qv.norm(dim=1, keepdim=True) * mag[:, None]
    I, info = atomistic_intensity(qv, f, method="direct")
    f2 = form_factor((mag.numpy() / (4 * math.pi)) ** 2, "Cu") ** 2
    ratio = (I.numpy() * V) / f2
    assert abs(ratio.mean() - 1.0) < 0.03, ratio.mean()
    assert ratio.min() > 0.9 and ratio.max() < 1.1, (ratio.min(), ratio.max())
    assert info["method"] == "direct"


def test_perfect_crystal_has_no_small_angle_intensity():
    f = cu_box((6, 6, 6))
    rng = np.random.default_rng(6)
    qv = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float64)
    qv = qv / qv.norm(dim=1, keepdim=True) * torch.linspace(0.4, 1.2, 50, dtype=torch.float64)[:, None]
    I, _ = atomistic_intensity(qv, f, method="direct")
    # one atom's worth of scattering is ~ 29^2 / V per A^3; a perfect crystal is far below it
    one_atom = 29.0 ** 2 / f.volume
    assert float(I.max()) < 1e-12 * one_atom


# ----------------------------------------------------------------------------- plumbing

def test_orientation_convention():
    f = cu_box((6, 6, 6), void_radius=5.0, shake=0.05)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # 90 deg about z
    rng = np.random.default_rng(8)
    qv = torch.as_tensor(rng.standard_normal((30, 3)), dtype=torch.float64) * 0.5
    a, _ = atomistic_intensity(qv, f, lab_from_sample=R, method="direct")
    # q_sample = R^T q_lab
    b, _ = atomistic_intensity(qv @ torch.as_tensor(R), f, method="direct")
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="rotation"):
        atomistic_intensity(qv, f, lab_from_sample=2 * np.eye(3))


def test_frames_are_averaged_incoherently():
    f1 = cu_box((6, 6, 6), void_radius=5.0, shake=0.1, seed=1)
    f2 = cu_box((6, 6, 6), void_radius=5.0, shake=0.1, seed=2)
    qv = torch.as_tensor(np.random.default_rng(9).standard_normal((30, 3)), dtype=torch.float64) * 0.5
    a, _ = atomistic_intensity(qv, f1, method="direct")
    b, _ = atomistic_intensity(qv, f2, method="direct")
    m, info = atomistic_intensity(qv, [f1, f2], method="direct")
    torch.testing.assert_close(m, 0.5 * (a + b), rtol=1e-10, atol=0)
    assert info["n_frames"] == 2


def test_inconsistent_frames_and_free_boundaries_are_refused():
    f = cu_box((6, 6, 6))
    other = AtomicFrame(positions=f.positions, species=f.species, cell=f.cell * 1.01)
    qv = torch.ones(3, 3, dtype=torch.float64) * 0.4
    with pytest.raises(ValueError, match="share the cell"):
        atomistic_intensity(qv, [f, other])
    free = AtomicFrame(positions=f.positions, species=f.species, cell=f.cell,
                       pbc=(True, True, False))
    with pytest.raises(ValueError, match="periodic"):
        atomistic_intensity(qv, free)


def test_narrow_window_and_low_q_warn():
    f = cu_box((6, 6, 6), drop=(3,))
    qv = torch.tensor([[0.05, 0.05, 0.0], [0.9, 0.4, 0.3]], dtype=torch.float64)   # floor = 3 a* = 0.87
    _, info = atomistic_intensity(qv, f, resolution_fwhm_inv_A=0.1, method="direct")
    text = " ".join(info["warnings"])
    assert "reciprocal-lattice spacings" in text        # window below two spacings
    assert info["fraction_below_floor"] == 0.5


def test_simulate_frame_carries_an_atoms_component():
    geom = SAXSGeometry(lsd_um=2.0e6, bcy_px=32, bcz_px=32, px_um=75.0, wavelength_A=0.7293,
                        n_pix_y=64, n_pix_z=64, beamstop_radius_px=6)
    f = cu_box((6, 6, 6), void_radius=5.0, shake=0.05)
    fr = simulate_frame(geom, atoms=f, atoms_options=dict(method="direct"))
    assert "atoms" in fr.components and fr.intensity.shape == (64, 64)
    assert torch.isfinite(fr.intensity).all() and float(fr.intensity.min()) >= 0
    assert fr.info["atoms"]["n_lattice_points"] > 0
    assert (fr.intensity[~fr.mask] == 0).all()
    fr2 = simulate_frame(geom, atoms=f, atoms_options=dict(method="direct"), sample_volume_A3=3.0)
    torch.testing.assert_close(fr2.intensity, 3.0 * fr.intensity)
    assert any("not representative" in w for w in fr.warnings)       # this box's q floor is high


def test_fft_path_reports_a_leakage_bound_that_scales_as_n_squared():
    q = torch.tensor([[0.15, 0.10, 0.05], [0.20, 0.05, 0.10]], dtype=torch.float64)
    box = cu_box((20, 20, 20), void_radius=8.0)                 # n = 75 -> |h| <= 9 reachable
    _, i1 = atomistic_intensity(q, box, method="fft")
    assert i1["fft_leakage_bound_f2"] is not None and i1["fft_leakage_bound_f2"] < 1.0
    assert not any("Bragg leakage" in w for w in i1["warnings"])
    _, i2 = atomistic_intensity(q, box, method="direct")
    assert i2["fft_leakage_bound_f2"] is None
    # the bound is N^2 (h/n)^(2 order): a finer grid with the same atoms must lower it
    _, i3 = atomistic_intensity(q, box, method="fft", fft_spacing_A=0.8)
    assert i3["fft_leakage_bound_f2"] < i1["fft_leakage_bound_f2"]
    with pytest.raises(ValueError, match="valid only for"):
        atomistic_intensity(q, box, method="fft", fft_spacing_A=2.0)         # outside its range
