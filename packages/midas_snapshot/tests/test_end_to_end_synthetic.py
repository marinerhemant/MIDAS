"""End-to-end on a synthetic still-frame series (no real data, generic structures).

Series: 60 frames. Frames 0-19: spotty matrix rings + a population of candidate
spots (a second cubic cell) + low background. Frames 20-39: diffuse halo, no
spots ("melt"). Frames 40-59: matrix at a slightly larger scale, candidate gone.
The pipeline must: fit the matrix scale, see the halo, choose windows by the
stated rule, find the candidate features, confirm on raw photons that they are
present before and absent after, pass the candidate while its rescaled negative
controls fail, and report a non-trivial injection curve.
"""
import json
import os

import numpy as np
import pytest
import tifffile

from midas_snapshot.config import SnapshotConfig

N = 256
LAM = 0.124

MATRIX_CIF = """data_matrix
_cell_length_a 3.60
_cell_length_b 3.60
_cell_length_c 3.60
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 225
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
M1 Ni 0 0 0 1
"""

CAND_CIF = """data_cand
_cell_length_a 4.40
_cell_length_b 4.40
_cell_length_c 4.40
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 225
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
A1 Zr 0 0 0 1
B1 C 0.5 0.5 0.5 1
"""

GEOM = f"""Lsd 250000
BC {N/2} {N/2}
px 172
Wavelength {LAM}
NrPixelsY {N}
NrPixelsZ {N}
"""


def _spot(yy, xx, r0, c0, flux, s=1.7):
    return flux * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def _positions(tth, eta, d_list, n, rng, lam=LAM, avoid=()):
    out = []
    for _ in range(50 * n):
        if len(out) >= n:
            break
        d = rng.choice(d_list)
        t = np.degrees(2 * np.arcsin(lam / (2 * d)))
        band = np.abs(tth - t) < 0.01
        if not band.any():
            continue
        rr, cc = np.nonzero(band)
        k = rng.integers(len(rr))
        r, c = rr[k], cc[k]
        if r < 25 or c < 25 or r > N - 26 or c > N - 26:
            continue
        if any(np.hypot(r - a, c - b) < 12 for a, b in list(out) + list(avoid)):
            continue
        out.append((r + 0.3, c - 0.2))
    return out


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    from midas_snapshot.geometry import build_maps
    from midas_hkls.io.cif import read_cif
    from midas_hkls.feature_phase import allowed_d_lines
    root = tmp_path_factory.mktemp("snap")
    (root / "matrix.cif").write_text(MATRIX_CIF)
    (root / "cand.cif").write_text(CAND_CIF)
    (root / "geom.txt").write_text(GEOM)
    frames = root / "frames"
    frames.mkdir()
    maps = build_maps(str(root / "geom.txt"), 0.005)
    tth, eta = maps.tth, maps.eta
    tmax = float(np.nanmax(tth))
    dmin = LAM / (2 * np.sin(np.radians(tmax / 2)))
    m_lines = allowed_d_lines(read_cif(str(root / "matrix.cif")), dmin, 50)
    c_lines = allowed_d_lines(read_cif(str(root / "cand.cif")), dmin, 50)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:N, 0:N]
    m_before = _positions(tth, eta, m_lines * 1.000, 40, rng)
    cand = _positions(tth, eta, c_lines * 1.010, 12, rng, avoid=m_before)
    m_after = _positions(tth, eta, m_lines * 1.004, 40, rng, avoid=cand)
    halo = 3.0 * np.exp(-0.5 * ((tth - 3.3) / 0.3) ** 2)
    lam_before = 0.1 + sum(_spot(yy, xx, r, c, 300) for r, c in m_before) + sum(_spot(yy, xx, r, c, 60) for r, c in cand)
    lam_melt = 0.1 + halo
    lam_after = 0.1 + sum(_spot(yy, xx, r, c, 300) for r, c in m_after)
    for i in range(60):
        lam = lam_before if i < 20 else (lam_melt if i < 40 else lam_after)
        raw = rng.poisson(lam).astype(np.int32)
        raw[:, 120:123] = -1                                # gap column (raw orientation = geometry)
        tifffile.imwrite(frames / f"f_{i:05d}.tif", raw)
    cfg = SnapshotConfig(frames=str(frames), geometry=str(root / "geom.txt"), out=str(root / "out"),
                         flip=None, matrix_cif=str(root / "matrix.cif"), halo_band=[2.85, 3.10],
                         base_band=[1.5, 2.0], window_sizes=[5], after_len=20, guard=5, rise=0.5,
                         margin_px=6, local_box=21, nproc=2, min_det=2,
                         baseline_until=20, min_before=10)
    os.makedirs(cfg.out, exist_ok=True)
    cfg.save(os.path.join(cfg.out, "snapshot_config.json"))
    return root, cfg, cand


def test_run_windows_scale_and_halo(series):
    from midas_snapshot.pipeline import run
    root, cfg, cand = series
    meta = run(cfg, 5)
    win = np.genfromtxt(os.path.join(cfg.out, "windows_W5.csv"), delimiter=",", names=True)
    assert meta["n_windows"] == 12
    before, melt, after = win[:4], win[4:8], win[8:]
    # ~8 synthetic spots per ring, each within 0.01 deg of its line: ~0.1 % scatter per window
    assert np.all(np.abs(before["scale"] - 1.000) < 2e-3)
    assert np.all(np.abs(after["scale"] - 1.004) < 2e-3)
    assert abs((np.median(after["scale"]) - np.median(before["scale"])) - 0.004) < 1.5e-3
    assert np.all(np.isnan(melt["scale"]))
    assert melt["halo"].min() > 5 * max(before["halo"].max(), after["halo"].max(), 0.01)


def test_analyse_features_phase_and_controls(series):
    from midas_snapshot.cli import main
    root, cfg, cand = series
    cfgp = os.path.join(cfg.out, "snapshot_config.json")
    main(["analyse", cfgp, "--candidates", str(root / "cand.cif"), "--controls", "--n-null", "300"])
    res = json.load(open(os.path.join(cfg.out, "analysis.json")))["W5"]
    assert res["windows"]["before"] == [0, 15] and res["windows"]["after"] == [40, 59]
    f = res["features"]
    found = 0
    for r0, c0 in cand:
        if f["n_features"] and np.min(np.hypot(np.array(f["row"]) - r0, np.array(f["col"]) - c0)) < 2.5:
            found += 1
    assert found >= 9
    vanish = np.array(f["present_before"]) & np.array(f["absent_after"])
    assert vanish.sum() >= 9
    rows = {r["name"]: r for r in res["phase"]["rows"]}
    assert rows["cand"]["passed"]
    assert res["phase"]["valid"]
    assert not any(r["passed"] for r in res["phase"]["rows"] if r["control"])
    inj = res["injection"]["curve"]
    assert inj["320"]["recovered"] > 0.8 and inj["5"]["recovered"] < 0.5
    assert 0 < res["injection"]["pattern_coverage"] <= 1


def test_report(series):
    from midas_snapshot.report import write_report
    root, cfg, cand = series
    p = write_report(cfg)
    rep = json.load(open(p))
    assert "W5" in rep["windows"] and rep["windows"]["W5"]["n_windows"] == 12


def test_static_mode_single_window_no_photon_test(series):
    from midas_snapshot.cli import main
    root, cfg, cand = series
    cfgp = os.path.join(cfg.out, "snapshot_config.json")
    main(["analyse", cfgp, "--static", "--candidates", str(root / "cand.cif"), "--n-null", "200"])
    res = json.load(open(os.path.join(cfg.out, "analysis.json")))["W5"]
    assert res["windows"]["after"] is None and res["windows"]["before"] == [0, 59]
    f = res["features"]
    assert f["n_features"] >= 9 and all(f["present_before"]) and not any(f["absent_after"])
    assert res["phase"]["require_vanishing"] is False and "injection" not in res


def test_threshold_table_over_calibration_windows_and_sigma_is_measured(series):
    """The run calibrates a (background level -> threshold) table on several windows, melt windows
    included, and each window uses the threshold at its own level; with sigma_px 'auto' the width
    is measured and recorded."""
    import dataclasses
    from midas_snapshot.pipeline import run
    root, cfg, cand = series
    c2 = dataclasses.replace(cfg, out=str(root / "out_auto"), sigma_px="auto", n_calib_windows=4,
                             threshold_mode="table")
    os.makedirs(c2.out, exist_ok=True)
    meta = run(c2, 5)
    assert len(meta["thresholds_per_window"]) == 4
    tab = meta["threshold"]
    assert set(tab) == {"b", "T"} and len(tab["b"]) == 4
    from midas_snapshot.pipeline import threshold_at
    lo = int(np.argmin(tab["b"])); hi = int(np.argmax(tab["b"]))
    coarse_lo = np.full((4, 4), tab["b"][lo]); coarse_hi = np.full((4, 4), tab["b"][hi])
    ok = np.ones((4, 4), bool)
    assert abs(threshold_at(tab, coarse_lo, ok) - tab["T"][lo]) < 1e-9
    assert abs(threshold_at(tab, coarse_hi, ok) - tab["T"][hi]) < 1e-9
    assert meta["sigma_px_measurement"]["n_spots"] > 0
    assert 1.2 < meta["sigma_px_used"] < 2.3          # synthetic spots have sigma 1.7


def test_per_window_threshold_follows_each_window(series):
    """Default mode: every window is calibrated on its own Poisson nulls; the melt windows (bright
    diffuse halo) get a different threshold from the solid windows, and one is recorded per window."""
    import dataclasses
    from midas_snapshot.pipeline import run
    root, cfg, cand = series
    c3 = dataclasses.replace(cfg, out=str(root / "out_pw"), threshold_mode="per_window", n_null_window=10)
    os.makedirs(c3.out, exist_ok=True)
    meta = run(c3, 5)
    tw = np.array(meta["thresholds_per_window"])
    assert meta["threshold"] == "per_window" and len(tw) == meta["n_windows"] == 12
    solid, melt = np.r_[tw[:4], tw[8:]], tw[4:8]
    assert abs(np.median(melt) - np.median(solid)) > 0.05 * np.median(solid)
