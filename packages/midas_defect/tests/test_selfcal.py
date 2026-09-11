"""Crystal self-calibration must recover a planted tilted geometry (2026-09-10 fold-in)."""
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry, qlab_to_pixel
from midas_defect.selfcal import CrystalSpots, _B, selfcalibrate_from_crystals

TRUE = Geometry(lsd_um=349640.6, bcy_px=737.30, bcz_px=810.30, px_um=172.0, wavelength_A=0.42459,
                n_pix_y=1475, n_pix_z=1679, omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40,
                ty_deg=0.20, tz_deg=0.35)


def _spots(U, a, b, c, gamma, geom, seed, noise_px=0.0):
    from midas_diffract.forward import solve_omega
    hkl = np.array([(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-16, 17)
                    if (h, k, l) != (0, 0, 0) and (h + k + l) % 2 == 0], float)
    G = U @ _B(a, b, c, 90.0, 90.0, gamma) @ hkl.T
    K0 = 2 * math.pi / geom.wavelength_A
    v = np.linalg.norm(G, axis=0) ** 2 / (2 * K0)
    wp, wn, ok = (t.numpy() for t in solve_omega(torch.tensor(G[0]), torch.tensor(G[1]), torch.tensor(v)))
    rows, cols, oms, hk = [], [], [], []
    for w in (wp, wn):
        sel = ok & (np.abs(np.degrees(w)) <= 19.5)
        idx = np.flatnonzero(sel)
        cw, sw = np.cos(w[idx]), np.sin(w[idx])
        gl = np.stack([cw * G[0, idx] - sw * G[1, idx], sw * G[0, idx] + cw * G[1, idx], G[2, idx]], 1)
        r, cc = (t.detach().cpu().numpy() for t in qlab_to_pixel(torch.tensor(gl), geom, device="cpu",
                                                                  dtype="float64"))
        on = np.isfinite(r) & np.isfinite(cc) & (r > 5) & (r < 1674) & (cc > 5) & (cc < 1470)
        rows.append(r[on]); cols.append(cc[on]); oms.append(np.degrees(w[idx][on])); hk.append(hkl[idx][on])
    rng = np.random.default_rng(seed)
    row, col = np.concatenate(rows), np.concatenate(cols)
    row = row + rng.normal(0, noise_px, row.shape); col = col + rng.normal(0, noise_px, col.shape)
    return row, col, np.concatenate(oms), np.concatenate(hk)


@pytest.fixture(scope="module")
def planted():
    U1 = Rotation.from_euler("zyx", [12.0, 31.0, -7.0], degrees=True).as_matrix()
    U2 = Rotation.from_euler("zyx", [-40.0, 18.0, 55.0], degrees=True).as_matrix()
    cells = [(U1, 3.580, 3.640, 19.25, 90.30), (U2, 3.600, 3.630, 19.25, 89.80)]
    doms = []
    for i, (U, a, b, c, ga) in enumerate(cells):
        row, col, om, hkl = _spots(U, a, b, c, ga, TRUE, seed=i)
        assert len(row) >= 20, f"domain {i} put only {len(row)} spots on the detector"
        doms.append((row, col, om, hkl, U, a, b, c, ga))
    return doms


def test_recovers_planted_tilts_centre_and_distance_from_a_flat_start(planted):
    start = replace(TRUE, ty_deg=0.0, tz_deg=0.0, bcy_px=TRUE.bcy_px + 0.7, bcz_px=TRUE.bcz_px - 0.5,
                    lsd_um=TRUE.lsd_um + 250.0)
    Upert = Rotation.from_rotvec([2e-3, -1e-3, 1.5e-3]).as_matrix()
    doms = [CrystalSpots(row=r, col=c, omega_deg=o, hkl=h, U=Upert @ U, a=a + 0.01, b=b - 0.01, c=cc, gamma=90.0)
            for (r, c, o, h, U, a, b, cc, ga) in planted]
    res = selfcalibrate_from_crystals(doms, start)
    g = res.geometry
    assert res.rms_after < 1e-3 * res.rms_before, str(res)
    assert abs(g.ty_deg - TRUE.ty_deg) < 0.01 and abs(g.tz_deg - TRUE.tz_deg) < 0.01, str(res)
    assert abs(g.bcy_px - TRUE.bcy_px) < 0.05 and abs(g.bcz_px - TRUE.bcz_px) < 0.05, str(res)
    assert abs(g.lsd_um - TRUE.lsd_um) < 50.0, str(res)
    for cell, (_, _, _, _, U, a, b, _, ga) in zip(res.cells, planted):
        assert abs(cell["a"] - a) < 1e-3 and abs(cell["b"] - b) < 1e-3 and abs(cell["gamma"] - ga) < 0.02
        assert cell["alpha"] == 90.0 and cell["beta"] == 90.0     # pinned by construction: not a check


def test_the_residual_is_the_check_a_wrong_hkl_assignment_does_not_fit(planted):
    row, col, om, hkl, U, a, b, c, ga = planted[0]
    wrong = hkl.copy(); wrong[:, [0, 1]] = wrong[:, [1, 0]]; wrong[:, 2] *= -1     # a different assignment
    res = selfcalibrate_from_crystals([CrystalSpots(row, col, om, wrong, U, a, b, c, ga)],
                                      replace(TRUE, ty_deg=0.0, tz_deg=0.0))
    assert res.rms_after > 0.01, str(res)


def test_rejects_unknown_free_fields_and_tiny_domains(planted):
    row, col, om, hkl, U, a, b, c, ga = planted[0]
    d = CrystalSpots(row, col, om, hkl, U, a, b, c, ga)
    with pytest.raises(ValueError):
        selfcalibrate_from_crystals([d], TRUE, free=("wavelength_A",))
    with pytest.raises(ValueError):
        selfcalibrate_from_crystals([CrystalSpots(row[:4], col[:4], om[:4], hkl[:4], U, a, b, c)], TRUE)
