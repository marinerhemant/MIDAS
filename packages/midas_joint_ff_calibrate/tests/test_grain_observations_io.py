"""Grains.csv / SpotMatrix.csv fixtures for the shared Phase-2 loaders.

This package had NO fixture for either file at any width, which is exactly why
``load_grains_csv`` could keep reading the 21-column legacy layout positionally
long after process_grains started writing 53 columns. Against a 53-column file
the old parser returned:

    radius     <- column 19 = DiffPos    (251.3 um instead of 8.6)
    confidence <- column 20 = DiffOme    (0.072 instead of 0.991)

and the ``len(cols) < 21`` guard happily passed at 53 columns, so nothing
raised. ``grain_refine`` selects grains with ``argsort(-confidence)``, so the
package was ranking on DiffOme descending -- the WORST-fitting grains first.

Every fixture below therefore carries the two features that were missing from
every previous fixture in the tree:
  * a ``Matched == 0`` row (a reflection predicted but never found), and
  * a header at the width the pipeline actually writes today.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_joint_ff_calibrate.grain_observations import (
    load_grains_csv,
    load_phase2_grains_and_spots,
    load_ring_two_theta,
    load_spot_matrix,
)

# ---------------------------------------------------------------- fixtures

_PREAMBLE = (
    "%NumGrains 3\n%BeamCenter 0.0 0.0\n%BeamThickness 0.0\n"
    "%GlobalPosition 0.0\n%NumPhases 1\n%PhaseInfo\n%\tSpaceGroup:225\n"
    "%\tLattice Parameter:3.590280 3.590280 3.590280 90.000000 90.000000 90.000000\n"
)

_OM = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

_GRAINS_53_COLS = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
)

#: (id, DiffPos um, DiffOme deg, GrainRadius um, Confidence). The DiffPos /
#: DiffOme values are what the broken positional reader used to hand back as
#: radius / confidence, and they are deliberately ANTI-CORRELATED with the real
#: GrainRadius / Confidence so a confidence-ordered selection changes order.
_G53_ROWS = (
    (1, 251.3, 0.072, 8.6, 0.991),
    (2, 12.4, 0.910, 25.0, 0.640),
    (3, 88.0, 0.455, 14.2, 0.815),
)

#: Kenesei strain, microstrain. Row i is filled with a single value so the
#: dimensionless Voigt pack is trivially checkable.
_KEN_UE = (1000.0, -500.0, 250.0)


def _grains_53(token: str = "ID", trailing_tab: bool = False) -> str:
    hdr = "%" + token + "\t" + "\t".join(_GRAINS_53_COLS) + "\n"
    lines = []
    for (gid, dpos, dome, rad, conf), ken in zip(_G53_ROWS, _KEN_UE):
        vals = (
            [gid] + _OM + [1.0 * gid, 2.0 * gid, 3.0 * gid]
            + [3.59028, 3.59028, 3.59028, 90.0, 90.0, 90.0]     # LATTICE at 13:18
            + [dpos, dome, 0.15, rad, conf]
            + [ken] * 9                                          # eFab, ue
            + [ken] * 9                                          # eKen, ue
            + [12.5, 1, 0.1, 0.2, 0.3]
            + [dpos + 30.0, dome + 0.02, 0.17]
            + [dpos, dome, 0.15]
        )
        lines.append("\t".join(repr(v) if isinstance(v, float) else str(v)
                               for v in vals) + ("\t" if trailing_tab else ""))
    return _PREAMBLE + hdr + "\n".join(lines) + "\n"


def _grains_21() -> str:
    """Genuine legacy width: cols 13-18 ARE the Voigt strain here."""
    cols = (["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
             "X", "Y", "Z", "E11", "E22", "E33", "E12", "E13", "E23",
             "GrainRadius", "Confidence"])
    hdr = "%ID\t" + "\t".join(cols) + "\n"
    lines = []
    for gid, rad, conf in ((1, 8.6, 0.991), (2, 25.0, 0.640), (3, 14.2, 0.815)):
        vals = ([gid] + _OM + [1.0 * gid, 2.0 * gid, 3.0 * gid]
                + [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4] + [rad, conf])
        lines.append("\t".join(str(v) for v in vals))
    return _PREAMBLE + hdr + "\n".join(lines) + "\n"


_SM_28_COLS = (
    ["SpotID", "Omega", "DetectorHor", "DetectorVert", "OmeRaw", "Eta",
     "RingNr", "YLab", "ZLab", "Theta", "StrainError", "Matched",
     "theorSpotID", "theorRingNr", "theorEta", "YExp", "ZExp", "OmegaExp",
     "DiffLen", "DiffOme", "InternalAngle", "YExpPost", "ZExpPost",
     "OmegaExpPost", "DiffLenPost", "DiffOmePost", "InternalAnglePost"]
)

#: Matched rows: (GrainID, SpotID, Omega, RingNr, YLab, ZLab, Theta).
_SM_MATCHED = (
    (1, 101, 10.0, 1, 1000.0, 0.0, 1.5),
    (1, 102, 20.0, 1, 0.0, 1000.0, 1.5),
    (2, 103, 30.0, 2, -1000.0, 0.0, 2.0),
    (2, 104, 40.0, 2, 0.0, -1000.0, 2.0),
    (3, 105, 50.0, 1, 700.0, 700.0, 1.5),
)


def _spotmatrix_28(n_unmatched: int = 2, trailing_tab: bool = True) -> str:
    """SpotMatrix at the width process_grains writes today (28 columns).

    ``n_unmatched`` rows are predicted-but-never-found reflections: -1 in the
    two integer columns (SpotID, RingNr) and NaN in every observed column.
    They are what raised ``KeyError: -1`` in build_observations_and_matches.
    """
    hdr = "%GrainID\t" + "\t".join(_SM_28_COLS) + "\n"
    lines = []
    for gid, sid, ome, ring, ylab, zlab, theta in _SM_MATCHED:
        v = ([gid, sid, ome, 100.0, 200.0, ome, 33.0, ring, ylab, zlab,
              theta, 1.0e-4, 1]
             + [sid, ring, 33.0] + [1.0] * 12)
        lines.append("\t".join(str(x) for x in v))
    for k in range(n_unmatched):
        v = ([1, -1] + ["nan"] * 5 + [-1] + ["nan"] * 4 + [0]
             + [900 + k, 3, 44.0] + ["nan"] * 12)
        lines.append("\t".join(str(x) for x in v))
    tail = "\t" if trailing_tab else ""     # written with newline='\t\n'
    return hdr + "\n".join(line + tail for line in lines) + "\n"


def _spotmatrix_12() -> str:
    """Legacy width: no Matched column, so every row is an observation."""
    hdr = ("%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta\t"
           "RingNr\tYLab\tZLab\tTheta\tStrainError\n")
    lines = []
    for gid, sid, ome, ring, ylab, zlab, theta in _SM_MATCHED:
        v = [gid, sid, ome, 100.0, 200.0, ome, 33.0, ring, ylab, zlab,
             theta, 1.0e-4]
        lines.append("\t".join(str(x) for x in v))
    return hdr + "\n".join(lines) + "\n"


def _hkls_csv() -> str:
    """Real MIDAS hkls.csv layout. load_ring_two_theta reads col 4 (RingNr)
    and col 9 (2Theta, degrees)."""
    hdr = "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
    rows = [
        "1 1 1 2.0990 1 -0.275 -0.275 -0.275 1.5 3.0 53923.4",
        "2 0 0 1.8180 2 -0.317  0.000  0.000 2.0 4.0 62310.1",
    ]
    return hdr + "\n".join(rows) + "\n"


def _layer_dir(tmp_path, grains_text: str, spots_text: str):
    d = tmp_path / "LayerNr_1"
    d.mkdir()
    (d / "Grains.csv").write_text(grains_text)
    (d / "SpotMatrix.csv").write_text(spots_text)
    (d / "hkls.csv").write_text(_hkls_csv())
    return d


# ---------------------------------------------------------------- Grains.csv


@pytest.mark.parametrize("token", ["ID", "GrainID"])
def test_53col_both_header_tokens(tmp_path, token):
    """io/csv writes %ID, c_parity_emit writes %GrainID. Both are real files."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(token=token))
    g = load_grains_csv(p)
    assert g["n_grains"] == 3
    assert g["ids"].tolist() == [1, 2, 3]


def test_53col_radius_is_grainradius_not_diffpos(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = load_grains_csv(p)
    np.testing.assert_allclose(g["radius"], [8.6, 25.0, 14.2])
    # The specific wrong answer the positional reader gave.
    assert not np.allclose(g["radius"], [251.3, 12.4, 88.0])


def test_53col_confidence_is_confidence_not_diffome(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = load_grains_csv(p)
    np.testing.assert_allclose(g["confidence"], [0.991, 0.640, 0.815])
    assert not np.allclose(g["confidence"], [0.072, 0.910, 0.455])


def test_53col_confidence_ordering_selects_best_grains(tmp_path):
    """grain_refine's ``argsort(-confidence)`` must rank the best grain first.

    Reading DiffOme as confidence inverted the ranking: grain 2 (conf 0.640,
    DiffOme 0.910) came first instead of grain 1 (conf 0.991).
    """
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = load_grains_csv(p)
    order = np.argsort(-g["confidence"])
    assert g["ids"][order].tolist() == [1, 3, 2]


def test_53col_strain_is_strain_not_lattice(tmp_path):
    """Cols 13-18 are a b c alpha beta gamma at this width, NOT Voigt strain."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = load_grains_csv(p)
    s = g["strains"]
    assert s.shape == (3, 6)
    # Kenesei microstrain -> dimensionless.
    np.testing.assert_allclose(s[0], [1e-3] * 6)
    np.testing.assert_allclose(s[1], [-5e-4] * 6)
    # The lattice must never appear here: 90 deg and 3.59 A as "strain" is the
    # bug that made phase 5's sequential-vs-joint comparison meaningless.
    assert not np.any(np.isclose(s, 90.0))
    assert not np.any(np.isclose(s, 3.59028))
    # The microstrain blocks are still available unrescaled.
    assert g["strain_ken_ue"].shape == (3, 3, 3)
    np.testing.assert_allclose(g["strain_ken_ue"][0], np.full((3, 3), 1000.0))
    np.testing.assert_allclose(g["lattice_per_grain"][0],
                               [3.59028, 3.59028, 3.59028, 90.0, 90.0, 90.0])


def test_53col_trailing_tab_rows(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(trailing_tab=True))
    g = load_grains_csv(p)
    assert g["n_grains"] == 3
    np.testing.assert_allclose(g["radius"], [8.6, 25.0, 14.2])


def test_53col_preamble_and_shapes(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = load_grains_csv(p)
    assert g["sg"] == 225
    # grain_refine does `grains["lattice"] or tuple(...)`: an ndarray here
    # raises "truth value of an array with more than one element is ambiguous".
    assert isinstance(g["lattice"], tuple)
    np.testing.assert_allclose(g["lattice"], [3.59028] * 3 + [90.0] * 3)
    assert g["orient_mat"].shape == (3, 9)      # (n, 9), not (n, 3, 3)
    np.testing.assert_allclose(g["positions"][1], [2.0, 4.0, 6.0])


def test_21col_legacy_still_reads_voigt_strain(tmp_path):
    """On a genuine 21-column file cols 13-18 ARE E11..E23."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_21())
    g = load_grains_csv(p)
    np.testing.assert_allclose(g["radius"], [8.6, 25.0, 14.2])
    np.testing.assert_allclose(g["confidence"], [0.991, 0.640, 0.815])
    np.testing.assert_allclose(g["strains"][0],
                               [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    assert g["strain_ken_ue"] is None
    assert g["lattice_per_grain"] is None


# ------------------------------------------------------------ SpotMatrix.csv


def test_28col_unmatched_rows_are_dropped(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_unmatched=2))
    s = load_spot_matrix(p)
    assert len(s["spot_id"]) == len(_SM_MATCHED)
    assert s["n_rows_total"] == len(_SM_MATCHED) + 2
    assert s["n_rows_unmatched"] == 2
    # The -1 sentinels are what raised KeyError: -1 in the ring-slot lookup.
    assert -1 not in s["spot_id"].tolist()
    assert -1 not in s["ring_nr"].tolist()
    # eta is recomputed from YLab/ZLab; NaN rows would poison it silently.
    assert np.isfinite(s["eta"]).all()
    assert np.isfinite(s["omega"]).all()
    assert np.isfinite(s["theta"]).all()


def test_28col_eta_recomputed_from_lab_not_column6(tmp_path):
    """SpotMatrix "Eta" (col 6) is a peak-fit diagnostic, not the eta angle."""
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28())
    s = load_spot_matrix(p)
    expected = np.rad2deg(np.arctan2(-s["y_lab"], s["z_lab"]))
    np.testing.assert_allclose(s["eta"], expected)
    # Every fixture row carries 33.0 in the Eta column; none should survive.
    assert not np.any(np.isclose(s["eta"], 33.0))


def test_28col_trailing_tab_rows(tmp_path):
    """SpotMatrix.csv is written with newline='\\t\\n'."""
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(trailing_tab=True))
    s = load_spot_matrix(p)
    assert len(s["spot_id"]) == len(_SM_MATCHED)
    np.testing.assert_allclose(s["omega"], [10.0, 20.0, 30.0, 40.0, 50.0])


def test_12col_legacy_keeps_every_row(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_12())
    s = load_spot_matrix(p)
    assert len(s["spot_id"]) == len(_SM_MATCHED)
    assert s["n_rows_unmatched"] == 0


# ------------------------------------------------------------ layer-dir path


def test_load_phase2_no_phantom_ring_reaches_the_slot_lookup(tmp_path):
    """``build_observations_and_matches`` builds ``obs_ring_nrs`` from the spot
    bags and indexes ``ring_two_theta_by_ring`` with it. An unmatched row's
    RingNr of -1 is not a key in hkls.csv, so it raised ``KeyError: -1``."""
    d = _layer_dir(tmp_path, _grains_53(), _spotmatrix_28(n_unmatched=3))
    eulers, pos, lat, spots, g, s = load_phase2_grains_and_spots(d)

    obs_ring_nrs = sorted({int(r) for bag in spots for r in bag.get("ring_nr", [])})
    assert obs_ring_nrs == [1, 2]
    ring_tt = load_ring_two_theta(d / "hkls.csv")
    # The exact expression that used to blow up.
    assert [ring_tt[r] for r in obs_ring_nrs] == [3.0, 4.0]


def test_load_phase2_shapes_and_bags(tmp_path):
    d = _layer_dir(tmp_path, _grains_53(), _spotmatrix_28())
    eulers, pos, lat, spots, g, s = load_phase2_grains_and_spots(d)
    assert eulers.shape == (3, 3)
    assert pos.shape == (3, 3)
    assert lat.shape == (3, 6)                 # header lattice, tiled
    np.testing.assert_allclose(lat[0], [3.59028] * 3 + [90.0] * 3)
    assert [len(bag["spot_id"]) for bag in spots] == [2, 2, 1]
    # Identity orientation matrix -> zero Euler triple.
    np.testing.assert_allclose(eulers, np.zeros((3, 3)), atol=1e-12)
