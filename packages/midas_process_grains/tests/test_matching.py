"""``midas_process_grains.matching`` — moved here from ``utils/match_grains.py``.

The regression that motivated the move: the script hardcoded a positional
Grains.csv layout matching **neither** real width. It placed ``eFab11`` at 19,
``Confidence`` at 38 and ``Radius`` at 42, while both the 47- and the
53-column files carry ``DiffPos`` at 19, ``GrainRadius`` at 22 and
``Confidence`` at 23. So ``grain_size = row[42]`` read ``RMSErrorStrain`` and
``--size-filter`` filtered on the wrong quantity — silently, because X/Y/Z and
O11..O33 *are* at the same indices in every width, so matching still worked and
the output looked plausible.

The first two tests below are that regression, in both widths.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.matching import (
    aggregate_grains,
    compute_cost_matrix,
    fit_affine_from_points,
    grain_columns,
    match_grains,
    stitch_layers,
    _AGG_POS,
    _AGG_SIZE,
)

# Real header of a 53-column Grains.csv, and the 47-column one it grew from.
COLS_53 = ("GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33 X Y Z a b c alpha beta "
           "gamma DiffPos DiffOme DiffAngle GrainRadius Confidence "
           "eFab11 eFab12 eFab13 eFab21 eFab22 eFab23 eFab31 eFab32 eFab33 "
           "eKen11 eKen12 eKen13 eKen21 eKen22 eKen23 eKen31 eKen32 eKen33 "
           "RMSErrorStrain PhaseNr Eul0 Eul1 Eul2 "
           "DiffPosPre DiffOmePre DiffAnglePre DiffPosPost DiffOmePost "
           "DiffAnglePost").split()
COLS_47 = COLS_53[:47]

PREAMBLE = ("%NumGrains {n}\n%BeamCenter 0.000000\n%BeamThickness 100.000000\n"
            "%GlobalPosition 0.000000\n%NumPhases 1\n%PhaseInfo\n"
            "%\tSpaceGroup:225\n"
            "%\tLattice Parameter: 3.6 3.6 3.6 90.0 90.0 90.0\n")


def _write_grains(path, cols, grains):
    """``grains`` = list of dicts keyed by column name; unset columns are 0."""
    lines = [PREAMBLE.format(n=len(grains)), "%" + "\t".join(cols) + "\n"]
    for g in grains:
        row = [f"{float(g.get(c, 0.0)):.6f}" for c in cols]
        row[0] = str(int(g.get("GrainID", 1)))
        lines.append("\t".join(row) + "\n")
    path.write_text("".join(lines))
    return path


def _grain(gid, x, y, z, radius, conf, *, rms=999.0, diffpos=888.0):
    """A grain with DISTINCT sentinels in the columns the old layout confused."""
    g = {"GrainID": gid, "X": x, "Y": y, "Z": z,
         "GrainRadius": radius, "Confidence": conf,
         "RMSErrorStrain": rms, "DiffPos": diffpos,
         "O11": 1.0, "O22": 1.0, "O33": 1.0,
         "a": 3.6, "b": 3.6, "c": 3.6,
         "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    return g


# ---------------------------------------------------------------------------
#  The regression: columns by name, in every width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cols,width", [(COLS_47, 47), (COLS_53, 53)])
def test_grain_columns_resolves_by_name(tmp_path, cols, width):
    p = _write_grains(tmp_path / "Grains.csv", cols,
                      [_grain(1, 10.0, 20.0, 0.0, radius=7.5, conf=0.9)])
    col = grain_columns(str(p))
    assert col["GrainRadius"] == 22, f"{width}-col: GrainRadius must be col 22"
    assert col["Confidence"] == 23, f"{width}-col: Confidence must be col 23"
    assert (col["X"], col["Y"], col["Z"]) == (10, 11, 12)
    assert col["O"] == 1
    # the indices the old positional layout used, which must NOT be picked up
    assert col["GrainRadius"] != 42, "col 42 is RMSErrorStrain, not Radius"
    assert col["Confidence"] != 38, "col 38 is an eKen component, not Confidence"


@pytest.mark.parametrize("cols,width", [(COLS_47, 47), (COLS_53, 53)])
def test_aggregate_reads_the_real_grain_size(tmp_path, cols, width):
    """``grain_size`` must be GrainRadius (7.5), never RMSErrorStrain (999)."""
    p = _write_grains(tmp_path / "Grains.csv", cols,
                      [_grain(1, 10.0, 20.0, 0.0, radius=7.5, conf=0.9,
                              rms=999.0, diffpos=888.0)])
    agg = aggregate_grains([str(p)])
    assert agg.shape[0] == 1
    assert agg[0, _AGG_SIZE] == pytest.approx(7.5), (
        f"{width}-col: read {agg[0, _AGG_SIZE]} — 999 means it took "
        "RMSErrorStrain, 888 means DiffPos")
    assert np.allclose(agg[0, _AGG_POS], [10.0, 20.0, 0.0])


def test_headerless_file_falls_back_and_warns(tmp_path, caplog):
    """No header ⇒ legacy positional layout, but it must say so."""
    p = tmp_path / "Grains.csv"
    p.write_text("\n".join(["#pad"] * 9 + ["\t".join(["0.0"] * 47)]) + "\n")
    import logging
    with caplog.at_level(logging.WARNING):
        col = grain_columns(str(p))
    assert col["GrainRadius"] == 42          # the legacy fallback
    assert any("legacy" in r.message.lower() or "legacy" in r.getMessage().lower()
               for r in caplog.records), "the fallback must warn"


# ---------------------------------------------------------------------------
#  Matching behaviour
# ---------------------------------------------------------------------------

def test_match_recovers_a_known_translation(tmp_path):
    """Two states differing by a rigid shift: every grain should pair up."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-300, 300, size=(12, 3))
    shift = np.array([25.0, -10.0, 5.0])
    a = _write_grains(tmp_path / "a.csv", COLS_53,
                      [_grain(i + 1, *pos[i], radius=5.0, conf=0.9)
                       for i in range(len(pos))])
    b = _write_grains(tmp_path / "b.csv", COLS_53,
                      [_grain(i + 1, *(pos[i] + shift), radius=5.0, conf=0.9)
                       for i in range(len(pos))])
    s1, s2 = aggregate_grains([str(a)]), aggregate_grains([str(b)])
    res = match_grains(s1, s2, sg_nr=225, mode="position")
    assert len(res["matches"]) == len(pos)

    i1 = [m[0] for m in res["matches"]]
    i2 = [m[1] for m in res["matches"]]
    aff = fit_affine_from_points(s1[i1][:, _AGG_POS], s2[i2][:, _AGG_POS])
    assert np.allclose(aff[:, :3], np.eye(3), atol=1e-6), "rotation should be I"
    assert np.allclose(aff[:, 3], shift, atol=1e-6), "translation must be recovered"


def test_hungarian_and_greedy_agree_when_well_separated(tmp_path):
    """With unambiguous pairs the optimal and greedy assignments coincide."""
    pos = np.array([[0.0, 0, 0], [400.0, 0, 0], [0, 400.0, 0], [-400.0, 0, 0]])
    a = _write_grains(tmp_path / "a.csv", COLS_53,
                      [_grain(i + 1, *pos[i], radius=5.0, conf=0.9)
                       for i in range(len(pos))])
    b = _write_grains(tmp_path / "b.csv", COLS_53,
                      [_grain(i + 1, *(pos[i] + 2.0), radius=5.0, conf=0.9)
                       for i in range(len(pos))])
    s1, s2 = aggregate_grains([str(a)]), aggregate_grains([str(b)])
    hu = match_grains(s1, s2, 225, mode="position", remove_duplicates=True)
    gr = match_grains(s1, s2, 225, mode="position", remove_duplicates=False)
    assert {(m[0], m[1]) for m in hu["matches"]} == {(m[0], m[1]) for m in gr["matches"]}


def test_size_filter_uses_grain_radius_not_a_neighbouring_column(tmp_path):
    """The point of the whole move: --size-filter must act on GrainRadius.

    Both grains carry RMSErrorStrain 999 (identical) but very different radii.
    A filter keyed on RMSErrorStrain would pass them; one on GrainRadius must
    reject the pair.
    """
    a = _write_grains(tmp_path / "a.csv", COLS_53,
                      [_grain(1, 0.0, 0.0, 0.0, radius=5.0, conf=0.9, rms=999.0)])
    b = _write_grains(tmp_path / "b.csv", COLS_53,
                      [_grain(1, 1.0, 0.0, 0.0, radius=50.0, conf=0.9, rms=999.0)])
    s1, s2 = aggregate_grains([str(a)]), aggregate_grains([str(b)])
    cost = compute_cost_matrix(s1, s2, 225, mode="position", size_filter=10.0)
    assert np.isinf(cost[0, 0]), (
        "a 5 µm and a 50 µm grain must fail a 10% size filter; a finite cost "
        "means the filter read a column where the two agree")


def test_stitch_offsets_layers_by_beam_thickness(tmp_path):
    """``stitch`` must place layer N at Z + N*beam_thickness."""
    files = []
    for L in range(3):
        files.append(str(_write_grains(
            tmp_path / f"L{L}.csv", COLS_53,
            [_grain(1, 100.0 * L, 0.0, 0.0, radius=5.0, conf=0.9)])))
    out = stitch_layers(files, beam_thickness=100.0, sg_nr=225)
    zs = np.sort(out[:, _AGG_POS][:, 2])
    assert np.allclose(zs, [0.0, 100.0, 200.0], atol=1e-6)
