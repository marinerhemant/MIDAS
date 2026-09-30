"""Map mode (frames = sample positions) on a synthetic map."""
import json
import os

import numpy as np
import tifffile

from midas_snapshot.config import SnapshotConfig

N = 256
LAM = 0.124


def _cif(a, sg, atoms):
    rows = "\n".join(f"{lab} {el} {x} {y} {z} 1" for lab, el, x, y, z in atoms)
    return f"""data_x
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number {sg}
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
{rows}
"""


def _spot(yy, xx, r0, c0, flux, s=1.7):
    return flux * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def _pos(tth, d, rng, avoid):
    t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
    rr, cc = np.nonzero(np.abs(tth - t) < 0.01)
    if len(rr) == 0:
        return None
    for _ in range(500):
        k = rng.integers(len(rr))
        r, c = rr[k], cc[k]
        if 25 < r < N - 26 and 25 < c < N - 26 and all(np.hypot(r - a, c - b) > 12 for a, b in avoid):
            return r + 0.2, c - 0.3
    return None


def test_map_mode_flags_fixed_pixel_and_finds_candidate(tmp_path):
    from midas_hkls.feature_phase import allowed_d_lines
    from midas_hkls.io.cif import read_cif
    from midas_snapshot.cli import main
    from midas_snapshot.geometry import build_maps
    (tmp_path / "matrix.cif").write_text(_cif(3.60, 225, [("M1", "Ni", 0, 0, 0)]))
    (tmp_path / "cand.cif").write_text(_cif(4.40, 225, [("A1", "Zr", 0, 0, 0), ("B1", "C", 0.5, 0.5, 0.5)]))
    (tmp_path / "geom.txt").write_text(f"Lsd 250000\nBC {N/2} {N/2}\npx 172\nWavelength {LAM}\n"
                                       f"NrPixelsY {N}\nNrPixelsZ {N}\n")
    maps = build_maps(str(tmp_path / "geom.txt"), 0.005)
    tmax = float(np.nanmax(maps.tth)); dmin = LAM / (2 * np.sin(np.radians(tmax / 2)))
    m_lines = allowed_d_lines(read_cif(str(tmp_path / "matrix.cif")), dmin, 50)
    c_lines = allowed_d_lines(read_cif(str(tmp_path / "cand.cif")), dmin, 50)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:N, 0:N]
    hot = _pos(maps.tth, (m_lines[-1] + m_lines[-2]) / 2, rng, [])        # off-matrix fixed pixel (d ascending)
    fr = tmp_path / "frames"; fr.mkdir()
    planted = []
    for i in range(30):
        taken = [hot]
        lam = np.full((N, N), 0.1) + _spot(yy, xx, *hot, 400)
        for _ in range(25):
            p = _pos(maps.tth, rng.choice(m_lines), rng, taken)
            if p:
                taken.append(p); lam += _spot(yy, xx, *p, 300)
        if i % 3 == 0:
            for _ in range(2):
                p = _pos(maps.tth, rng.choice(c_lines) * 1.01, rng, taken)
                if p:
                    taken.append(p); planted.append((i, *p)); lam += _spot(yy, xx, *p, 150)
        tifffile.imwrite(fr / f"m_{i:03d}.tif", rng.poisson(lam).astype(np.int32))
    cfg = SnapshotConfig(frames=str(fr), geometry=str(tmp_path / "geom.txt"), out=str(tmp_path / "out"),
                         flip=None, matrix_cif=str(tmp_path / "matrix.cif"), window_sizes=[1],
                         margin_px=6, local_box=21, nproc=2)
    os.makedirs(cfg.out); cp = os.path.join(cfg.out, "snapshot_config.json"); cfg.save(cp)
    main(["run", cp])
    main(["analyse", cp, "--map", "--candidates", str(tmp_path / "cand.cif"), "--n-null", "300"])
    r = json.load(open(os.path.join(cfg.out, "analysis.json")))["W1"]
    f = r["features"]
    assert r["mode"] == "map" and f["n_detector_fixed"] >= 25          # the hot pixel, every frame
    assert not any(np.hypot(rw - hot[0], cl - hot[1]) < 3 for rw, cl in zip(f["row"], f["col"]))
    found = sum(any(fi == i and np.hypot(rw - r0, cl - c0) < 2.5
                    for fi, rw, cl in zip(f["frame"], f["row"], f["col"])) for i, r0, c0 in planted)
    assert found >= 0.8 * len(planted)
    rows = {x["name"]: x for x in r["phase"]["rows"]}
    assert rows["cand"]["passed"] and r["phase"]["valid"]
