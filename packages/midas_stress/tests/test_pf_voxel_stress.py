"""Tests for midas_stress.pf.voxel_stress and the Zr / delta-hydride entries."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.materials import (
    STIFFNESS_LIBRARY, cubic_stiffness, get_stiffness, hexagonal_stiffness,
)
from midas_stress.hooke import hooke_stress
from midas_stress.tensor import strain_grain_to_lab
from midas_stress.pf import voxel_stress, read_microstr_full, N_COLS_MICROSTR


def _table(oms, eps_lab_microstrain):
    """Build an (N, 43) microstrFull-layout array."""
    n = len(oms)
    t = np.zeros((n, N_COLS_MICROSTR))
    t[:, 1:10] = np.asarray(oms).reshape(n, 9)
    t[:, 11:14] = np.arange(3 * n).reshape(n, 3)
    t[:, 26] = 0.9
    t[:, 27:36] = np.asarray(eps_lab_microstrain).reshape(n, 9)
    return t


def _random_sym(rng, n, scale=1e-3):
    a = rng.normal(scale=scale, size=(n, 3, 3))
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def test_isotropic_cubic_is_orientation_independent():
    rng = np.random.default_rng(0)
    C11, C12 = 200.0, 100.0
    C = cubic_stiffness(C11, C12, (C11 - C12) / 2.0)
    eps = _random_sym(rng, 1)[0]
    oms = Rotation.random(50, random_state=1).as_matrix()
    out = voxel_stress(_table(oms, np.repeat(eps[None] * 1e6, 50, 0)), C)
    ref = out["stress_lab"][0]
    assert np.allclose(out["stress_lab"], ref[None], atol=1e-9)
    # and equals the isotropic formula lambda tr(e) I + 2 mu e (GPa -> MPa)
    lam, mu = C12, (C11 - C12) / 2.0
    iso = (lam * np.trace(eps) * np.eye(3) + 2 * mu * eps) * 1e3
    assert np.allclose(ref, iso, atol=1e-9)


def test_hydrostatic_strain_cubic():
    e = 1.3e-3
    oms = Rotation.random(20, random_state=2).as_matrix()
    out = voxel_stress(_table(oms, np.repeat((e * np.eye(3))[None] * 1e6, 20, 0)),
                       "ZrH_delta")
    p = STIFFNESS_LIBRARY["ZrH_delta"]
    expect = (p["C11"] + 2 * p["C12"]) * e * 1e3  # MPa
    assert np.allclose(out["hydrostatic"], expect, rtol=1e-12)
    assert np.allclose(out["von_mises"], 0.0, atol=1e-9)


@pytest.mark.parametrize("mat", ["Zr", "ZrH_delta", "Ti"])
def test_crystal_frame_then_rotate_equals_lab(mat):
    rng = np.random.default_rng(3)
    C = get_stiffness(mat)
    n = 40
    oms = Rotation.random(n, random_state=4).as_matrix()
    eps_c = _random_sym(rng, n)
    eps_lab = strain_grain_to_lab(eps_c, oms)
    out = voxel_stress(_table(oms, eps_lab * 1e6), mat)
    sig_c = hooke_stress(eps_c, C, frame="grain")
    sig_lab_ref = strain_grain_to_lab(sig_c, oms) * 1e3
    assert np.allclose(out["stress_lab"], sig_lab_ref, atol=1e-8)
    # independent check without Mandel machinery: explicit 4th-rank C_ijkl
    Cijkl = _mandel_to_cijkl(C)
    for k in range(3):
        U = oms[k]
        s_c = np.einsum("ijkl,kl->ij", Cijkl, eps_c[k])
        assert np.allclose(U @ s_c @ U.T * 1e3, out["stress_lab"][k], atol=1e-8)


def _mandel_to_cijkl(Cm):
    """Mandel 6x6 -> 4th-rank tensor (independent of tensor.py)."""
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    f = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            v = Cm[a, b] / (f[a] * f[b])
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C[p, q, r, s] = v
    return C


def test_new_library_entries_load():
    Zr = get_stiffness("Zr")
    assert np.allclose(Zr, hexagonal_stiffness(143.5, 72.5, 65.4, 164.9, 32.1))
    assert Zr[3, 3] == pytest.approx(143.5 - 72.5)
    assert Zr[4, 4] == pytest.approx(2 * 32.1)
    H = get_stiffness("ZrH_delta")
    assert np.allclose(H, cubic_stiffness(162.0, 103.0, 69.3))
    assert STIFFNESS_LIBRARY["Zr"]["symmetry"] == "hexagonal"
    assert STIFFNESS_LIBRARY["ZrH_delta"]["symmetry"] == "cubic"


def test_units_microstrain_to_mpa():
    # uniaxial 1000 microstrain along crystal x, identity orientation, Zr:
    # sigma_xx = C11 * 1e-3 GPa = 143.5 MPa
    t = _table([np.eye(3)], [np.diag([1000.0, 0, 0])])
    out = voxel_stress(t, "Zr")
    assert out["stress_voigt"][0, 0] == pytest.approx(143.5)
    assert out["stress_voigt"][0, 1] == pytest.approx(72.5)
    assert out["stress_voigt"][0, 2] == pytest.approx(65.4)


def test_zero_strain_table_raises(tmp_path):
    t = _table(Rotation.random(3, random_state=5).as_matrix(), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="zero strain"):
        voxel_stress(t, "Zr")
    assert np.allclose(voxel_stress(t, "Zr", allow_zero_strain=True)["stress_lab"], 0)


def test_read_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    oms = Rotation.random(4, random_state=7).as_matrix()
    t = _table(oms, _random_sym(rng, 4) * 1e6)
    p = tmp_path / "microstrFull.csv"
    np.savetxt(p, t, fmt="%.6f", delimiter=",", header="SpotID,O11")
    back = read_microstr_full(str(p))
    assert back.shape == (4, 43)
    a = voxel_stress(str(p), "Zr")["stress_lab"]
    b = voxel_stress(t, "Zr")["stress_lab"]
    assert np.allclose(a, b, atol=1e-3)  # %.6f print rounding of OM / strain
