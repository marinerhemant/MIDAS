"""Known-phase selection and multi-phase classification (synthetic, generic structures)."""
import os

import numpy as np
import tifffile

from midas_snapshot.config import SnapshotConfig
from midas_snapshot.setup_select import select_setup

N = 256
LAM = 0.124


def _cif(a, sg, el):
    hm = {225: "F m -3 m", 229: "I m -3 m"}[sg]
    return (f"data_x\n_cell_length_a {a}\n_cell_length_b {a}\n_cell_length_c {a}\n"
            f"_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            f"_space_group_IT_number {sg}\n_symmetry_space_group_name_H-M '{hm}'\n"
            "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
            f"_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\nX1 {el} 0 0 0 1\n")


def _setup(tmp_path):
    (tmp_path / "g.txt").write_text(f"Lsd 250000\nBC {N/2} {N/2}\npx 172\nWavelength {LAM}\n"
                                    f"NrPixelsY {N}\nNrPixelsZ {N}\n")
    (tmp_path / "bcc.cif").write_text(_cif(2.87, 229, "Fe"))
    (tmp_path / "fcc.cif").write_text(_cif(3.60, 225, "Ni"))
    (tmp_path / "fcc_big.cif").write_text(_cif(2.87 * np.sqrt(2), 225, "Al"))   # shares bcc lines
    from midas_snapshot.geometry import build_maps, matrix_lines
    maps = build_maps(str(tmp_path / "g.txt"), 0.005)
    tmax = float(np.nanmax(maps.tth))
    L = {k: matrix_lines(str(tmp_path / f"{k}.cif"), LAM, tmax) for k in ("bcc", "fcc", "fcc_big")}
    return maps, L


def _rings(maps, lines, amp=4.0, w=0.02):
    lam = np.zeros(maps.tth.shape)
    for d in lines:
        t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
        lam += amp * np.exp(-0.5 * ((maps.tth - t) / w) ** 2)
    return lam


def _write(folder, lam_list, rng):
    folder.mkdir()
    for i, lam in enumerate(lam_list):
        tifffile.imwrite(folder / f"f_{i:04d}.tif", rng.poisson(0.1 + lam).astype(np.int32))


def test_complete_pattern_beats_subset_coincidence(tmp_path):
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(0)
    _write(tmp_path / "fr", [_rings(maps, L["bcc"] * 1.005)] * 10, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("bcc", "fcc", "fcc_big")},
                       flip=None, n_frames=10)
    assert res["best"]["matrix"] == "bcc" and res["best"]["completeness"] == 1.0
    assert res["phases"] == ["bcc"]                         # the sqrt(2) fcc is a coincidence, not a phase


def test_phase_switch_in_time_gives_both_phases(tmp_path):
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(1)
    frames = [_rings(maps, L["bcc"] * 1.002)] * 20 + [_rings(maps, L["fcc"] * 1.004)] * 20
    _write(tmp_path / "fr", frames, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("bcc", "fcc", "fcc_big")},
                       flip=None, n_frames=10, blocks=(0.0, 1.0))
    assert set(res["phases"]) == {"bcc", "fcc"}
    info = {p["name"]: p for p in res["phase_info"]}
    assert abs(info["bcc"]["cell_a"] - 2.87 * 1.002) < 0.01 and abs(info["fcc"]["cell_a"] - 3.60 * 1.004) < 0.01


def test_second_known_phase_is_not_off_matrix(tmp_path):
    from midas_snapshot.pipeline import run
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:N, 0:N]
    lam = _rings(maps, L["bcc"] * 1.003, amp=3.0) + _rings(maps, L["fcc"] * 1.004, amp=3.0)
    # compact spots on the fcc rings
    for d in L["fcc"][-3:] * 1.004:
        t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
        rr, cc = np.nonzero(np.abs(maps.tth - t) < 0.005)
        for k in rng.choice(len(rr), 3, replace=False):
            if 20 < rr[k] < N - 20 and 20 < cc[k] < N - 20:
                lam += 200 * np.exp(-((yy - rr[k]) ** 2 + (xx - cc[k]) ** 2) / (2 * 1.7 ** 2)) / (2 * np.pi * 1.7 ** 2)
    _write(tmp_path / "fr", [lam] * 5, rng)

    def off_fraction(phase_cifs, out):
        cfg = SnapshotConfig(frames=str(tmp_path / "fr"), geometry=str(tmp_path / "g.txt"), out=str(out),
                             flip=None, matrix_cif=str(tmp_path / "bcc.cif"), phase_cifs=phase_cifs,
                             window_sizes=[5], margin_px=6, local_box=21, nproc=1,
                             # the default 6-px window (+/-0.24 deg at this coarse geometry) spans
                             # the bcc 110 and fcc 111 rings (0.085 deg apart); a blended ring is
                             # rightly dropped, so fit on a window that resolves them
                             fit_window_deg=0.08)
        run(cfg, 5)
        sp = np.load(os.path.join(cfg.out, "spots_W5.npy"))
        # judge only spots on an fcc ring and clear of every bcc ring (at the coarse test
        # geometry close rings merge; resolved rings are where classification is defined)
        d = sp[:, 8]
        near = lambda D, s: np.min(np.abs(d[:, None] / (D[None, :] * s) - 1), axis=1)
        sel = (near(L["fcc"], 1.004) < 0.002) & (near(L["bcc"], 1.003) > 0.01)
        rel = sp[sel, 9]
        return float(np.mean(np.abs(rel) > 0.004)), int(sel.sum())

    off_without, n1 = off_fraction([], tmp_path / "o1")
    off_with, n2 = off_fraction([str(tmp_path / "fcc.cif")], tmp_path / "o2")
    assert n1 >= 3 and n2 >= 3
    assert off_without > 0.8 and off_with < 0.1


def test_two_reference_cells_of_one_pattern_are_one_phase(tmp_path):
    """Two fcc references (different cells) fit the same rings at different scales: one phase."""
    maps, L = _setup(tmp_path)
    (tmp_path / "fcc2.cif").write_text(_cif(3.52, 225, "Ni"))
    rng = np.random.default_rng(4)
    frames = [_rings(maps, L["fcc"] * 1.004)] * 20
    _write(tmp_path / "fr", frames, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("fcc", "fcc2", "bcc")},
                       flip=None, n_frames=10, blocks=(0.0, 1.0))
    assert len(res["phases"]) == 1 and res["phases"][0] in ("fcc", "fcc2")


def test_window_scale_is_nan_when_reference_catches_another_phase(tmp_path):
    """After a bcc -> fcc switch the bcc reference must not report a (wrong) scale."""
    from midas_snapshot.pipeline import run
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(5)
    frames = [_rings(maps, L["bcc"] * 1.002)] * 5 + [_rings(maps, L["fcc"] * 1.004)] * 5
    _write(tmp_path / "fr", frames, rng)
    cfg = SnapshotConfig(frames=str(tmp_path / "fr"), geometry=str(tmp_path / "g.txt"), out=str(tmp_path / "o"),
                         flip=None, matrix_cif=str(tmp_path / "bcc.cif"), window_sizes=[5], margin_px=6,
                         local_box=21, nproc=1)
    run(cfg, 5)
    w = np.genfromtxt(os.path.join(cfg.out, "windows_W5.csv"), delimiter=",", names=True)
    assert abs(w["scale"][0] - 1.002) < 2e-3
    assert np.isnan(w["scale"][1])


def test_same_pattern_cold_and_hot_blocks_is_one_phase(tmp_path):
    """Cold block fits one fcc reference best, hot block another (0.9 % apart): one phase."""
    maps, L = _setup(tmp_path)
    (tmp_path / "fcc2.cif").write_text(_cif(3.52, 225, "Ni"))
    rng = np.random.default_rng(6)
    frames = [_rings(maps, L["fcc"] * 1.000)] * 20 + [_rings(maps, L["fcc"] * 1.009)] * 20
    _write(tmp_path / "fr", frames, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("fcc", "fcc2", "bcc")},
                       flip=None, n_frames=10, blocks=(0.0, 1.0))
    assert len(res["phases"]) == 1


def test_sub_pattern_is_dropped(tmp_path):
    """fcc with a = sqrt(2) a_bcc carries every bcc line: bcc is a sub-pattern, not a phase."""
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(7)
    _write(tmp_path / "fr", [_rings(maps, L["fcc_big"] * 1.004)] * 20, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("fcc_big", "bcc")},
                       flip=None, n_frames=10, blocks=(0.0, 1.0))
    assert res["phases"] == ["fcc_big"]


def test_same_structure_far_apart_is_two_phases(tmp_path):
    """One structure type at two cells 14 % apart, best in different blocks, is two phases
    (not one phase that expanded), and each keeps its own cell."""
    maps, L = _setup(tmp_path)
    (tmp_path / "fcc_b.cif").write_text(_cif(4.10, 225, "Al"))
    Lb = L["fcc"] * 4.10 / 3.60
    rng = np.random.default_rng(5)
    frames = [_rings(maps, L["fcc"] * 1.004)] * 20 + [_rings(maps, Lb * 1.002)] * 20
    _write(tmp_path / "fr", frames, rng)
    res = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")},
                       {k: str(tmp_path / f"{k}.cif") for k in ("fcc", "fcc_b")},
                       flip=None, n_frames=10, blocks=(0.0, 1.0))
    info = {p["name"]: p for p in res["phase_info"]}
    assert set(info) == {"fcc", "fcc_b"}
    assert abs(info["fcc"]["cell_a"] - 3.60 * 1.004) < 0.01 and abs(info["fcc_b"]["cell_a"] - 4.10 * 1.002) < 0.01


def test_unlisted_phase_shows_as_unexplained_lines(tmp_path):
    maps, L = _setup(tmp_path)
    rng = np.random.default_rng(7)
    extra = L["fcc"] * 3.20 / 3.60                           # a phase not in the reference list
    _write(tmp_path / "fa", [_rings(maps, L["fcc_big"])] * 10, rng)
    _write(tmp_path / "fb", [_rings(maps, L["fcc_big"]) + _rings(maps, extra, amp=3.0)] * 10, rng)
    kw = dict(flip=None, n_frames=10)
    refs = {"fcc_big": str(tmp_path / "fcc_big.cif")}
    a = select_setup(str(tmp_path / "fa"), {"g": str(tmp_path / "g.txt")}, refs, **kw)
    b = select_setup(str(tmp_path / "fb"), {"g": str(tmp_path / "g.txt")}, refs, **kw)
    assert a["n_unexplained"] == 0
    got = [u["d"] for u in b["unexplained"][0]]
    assert b["n_unexplained"] >= 3
    assert all(min(abs(g / x - 1) for x in extra) < 0.004 for g in got), (got, b["unexplained"])


def test_harmonic_coincidence_does_not_replace_split_fcc(tmp_path):
    """A hot block: fcc rings split into two cells 0.5 % apart, so the fcc (7 of 8 rings) narrowly
    fails the spread gate. A bcc at a_fcc sqrt(2/3) puts 110/220/400 exactly on fcc 111/222/422 and
    passes it with 3 rings. It must not become the phase (without the veto it does)."""
    from midas_snapshot.geometry import build_maps, matrix_lines
    (tmp_path / "g.txt").write_text(f"Lsd 250000\nBC 256 256\npx 200\nWavelength {LAM}\n"
                                    "NrPixelsY 512\nNrPixelsZ 512\n")       # reaches fcc 422
    (tmp_path / "fcc.cif").write_text(_cif(3.60, 225, "Ni"))
    (tmp_path / "bcch.cif").write_text(_cif(3.60 * np.sqrt(2 / 3), 229, "Fe"))
    maps = build_maps(str(tmp_path / "g.txt"), 0.005)
    lines = matrix_lines(str(tmp_path / "fcc.cif"), LAM, float(np.nanmax(maps.tth)))
    rng = np.random.default_rng(9)
    lam = _rings(maps, lines * 1.000, amp=2.0) + _rings(maps, lines * 1.005, amp=2.0)
    _write(tmp_path / "fr", [lam] * 10, rng)
    kw = dict(flip=None, n_frames=10, spread_tol=0.00033)            # fcc spread 0.00037, bcc 0.00029
    refs = {k: str(tmp_path / f"{k}.cif") for k in ("bcch", "fcc")}
    off = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")}, refs, near_spread=0.0, **kw)
    on = select_setup(str(tmp_path / "fr"), {"g": str(tmp_path / "g.txt")}, refs, **kw)
    assert off["phases"] == ["bcch"]                                   # the trap is real
    assert "bcch" not in on["phases"]


def test_cross_series_recurrence_flags_fixed_pixels():
    from midas_snapshot.analysis import cross_series_recurrence
    pos = {"s1": ([100.0, 400.0, 700.0], [100.0, 400.0, 700.0]),
           "s2": ([101.5, 250.0], [99.0, 250.0]),          # first one within 2.5 px of s1's first
           "s3": ([], [])}
    f = cross_series_recurrence(pos, radius=4.0)
    assert f["s1"].tolist() == [True, False, False]
    assert f["s2"].tolist() == [True, False]
    assert f["s3"].size == 0
