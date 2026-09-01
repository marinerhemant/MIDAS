"""Tests for the canonical Grains.csv / SpotMatrix.csv readers.

These exist because the tree had NO checked-in fixture for either file at any
width, and the widths that were generated inline all encoded dead formats --
which is precisely why ~50 positional readers drifted unnoticed.

Every fixture here therefore carries the two features that were missing from
every previous fixture and that hid real bugs:
  * a ``Matched == 0`` row (predicted-but-never-found), and
  * a trailing-tab variant.
"""
import numpy as np
import pytest

from midas_process_grains.io.read import (
    GrainsFormatError,
    read_grains_csv,
    read_spot_matrix,
)

# --- fixtures, one per historical width -------------------------------------

_PREAMBLE = (
    "%NumGrains 2\n%BeamCenter 0.0\n%BeamThickness 0.0\n%GlobalPosition 0.0\n"
    "%NumPhases 1\n%PhaseInfo\n%\tSpaceGroup:225\n"
    "%\tLattice Parameter: 3.590280 3.590280 3.590280 90.000000 90.000000 90.000000\n"
)
_OM = "1 0 0 0 1 0 0 0 1"


def _grains_53(token="GrainID", trailing_tab=False):
    cols = (["O11","O12","O13","O21","O22","O23","O31","O32","O33","X","Y","Z",
             "a","b","c","alpha","beta","gamma","DiffPos","DiffOme","DiffAngle",
             "GrainRadius","Confidence"]
            + [f"eFab{i}{j}" for i in (1,2,3) for j in (1,2,3)]
            + [f"eKen{i}{j}" for i in (1,2,3) for j in (1,2,3)]
            + ["RMSErrorStrain","PhaseNr","Eul0","Eul1","Eul2",
               "DiffPosPre","DiffOmePre","DiffAnglePre",
               "DiffPosPost","DiffOmePost","DiffAnglePost"])
    hdr = "%" + token + "\t" + "\t".join(cols) + "\n"
    rows = []
    for gid, rad, conf in ((1, 27.5, 0.87), (2, 44.0, 0.95)):
        vals = ([gid] + [float(x) for x in _OM.split()] + [1.0, 2.0, 3.0]
                + [3.59028]*3 + [90.0]*3
                + [251.3, 0.072, 0.152, rad, conf]
                + [100.0 + k for k in range(9)]       # eFab, microstrain
                + [200.0 + k for k in range(9)]       # eKen, microstrain
                + [12.5, 1, 0.1, 0.2, 0.3]
                + [283.0, 0.09, 0.17] + [251.3, 0.072, 0.152])
        line = "\t".join(str(v) for v in vals)
        rows.append(line + ("\t" if trailing_tab else ""))
    return _PREAMBLE + hdr + "\n".join(rows) + "\n"


def _grains_21():
    """Legacy width. Cols 13-18 are VOIGT STRAIN here, not the lattice."""
    cols = (["O11","O12","O13","O21","O22","O23","O31","O32","O33","X","Y","Z",
             "E11","E22","E33","E12","E13","E23","GrainRadius","Confidence"])
    hdr = "%ID\t" + "\t".join(cols) + "\n"
    rows = []
    for gid, rad, conf in ((1, 27.5, 0.87), (2, 44.0, 0.95)):
        vals = ([gid] + [float(x) for x in _OM.split()] + [1.0, 2.0, 3.0]
                + [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4] + [rad, conf])
        rows.append("\t".join(str(v) for v in vals))
    return _PREAMBLE + hdr + "\n".join(rows) + "\n"


def _spotmatrix_28(n_matched=3, n_unmatched=2, trailing_tab=True):
    cols = ["SpotID","Omega","DetectorHor","DetectorVert","OmeRaw","Eta","RingNr",
            "YLab","ZLab","Theta","StrainError","Matched","theorSpotID",
            "theorRingNr","theorEta","YExp","ZExp","OmegaExp","DiffLen","DiffOme",
            "InternalAngle","YExpPost","ZExpPost","OmegaExpPost","DiffLenPost",
            "DiffOmePost","InternalAnglePost"]
    hdr = "%GrainID\t" + "\t".join(cols) + "\n"
    rows = []
    for k in range(n_matched):
        v = [7, 100+k, 2.0+k, 3.0, 4.0, 5.0, 6.0, 3, 8.0, 9.0, 2.4, 0.0, 1,
             1.0, 3.0, 5.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
             22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
        rows.append("\t".join(str(x) for x in v))
    for _ in range(n_unmatched):
        v = ([7, -1, "nan", "nan", "nan", "nan", "nan", -1, "nan", "nan", "nan",
              "nan", 0] + [1.0, 3.0, 5.0] + ["nan"]*12)
        rows.append("\t".join(str(x) for x in v))
    tail = "\t" if trailing_tab else ""
    return hdr + "\n".join(r + tail for r in rows) + "\n"


# --- Grains.csv -------------------------------------------------------------

@pytest.mark.parametrize("token", ["GrainID", "ID"])
def test_both_header_tokens_read(tmp_path, token):
    """c_parity_emit writes %GrainID, io/csv writes %ID. Both are real."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(token=token))
    t = read_grains_csv(p)
    assert t.n_grains == 2
    assert t.header_token == token
    np.testing.assert_allclose(t.grain_radius, [27.5, 44.0])
    np.testing.assert_allclose(t.confidence, [0.87, 0.95])


def test_53_col_radius_and_confidence_are_not_diffpos_diffome(tmp_path):
    """The exact silent-wrong bug: a 21-col reader takes DiffPos as the radius
    and DiffOme as the confidence. Assert on the values, not the width."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    t = read_grains_csv(p)
    assert t.grain_radius[0] == pytest.approx(27.5)      # NOT 251.3 (DiffPos)
    assert t.confidence[0] == pytest.approx(0.87)        # NOT 0.072 (DiffOme)
    assert t.diff_pos[0] == pytest.approx(251.3)
    assert t.diff_ome[0] == pytest.approx(0.072)


def test_cols_13_18_are_lattice_on_wide_and_strain_on_legacy(tmp_path):
    """Same positions, different meaning. Only name resolution separates them."""
    wide = tmp_path / "w.csv"; wide.write_text(_grains_53())
    old = tmp_path / "o.csv"; old.write_text(_grains_21())
    tw, to = read_grains_csv(wide), read_grains_csv(old)
    assert tw.lattice is not None and tw.strain_voigt is None
    np.testing.assert_allclose(tw.lattice[0], [3.59028]*3 + [90.0]*3)
    assert to.strain_voigt is not None and to.lattice is None
    np.testing.assert_allclose(to.strain_voigt[0],
                               [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    assert "legacy" in to.strain_units


def test_legacy_21_col_radius_confidence(tmp_path):
    p = tmp_path / "Grains.csv"; p.write_text(_grains_21())
    t = read_grains_csv(p)
    assert t.n_columns == 21
    np.testing.assert_allclose(t.grain_radius, [27.5, 44.0])
    np.testing.assert_allclose(t.confidence, [0.87, 0.95])


def test_trailing_tab_rows_do_not_break_parsing(tmp_path):
    """The C writer emits a trailing tab; a naive split('\\t') then hands
    float() an empty string. midas_plotting still dies on this."""
    p = tmp_path / "Grains.csv"; p.write_text(_grains_53(trailing_tab=True))
    t = read_grains_csv(p)
    assert t.n_grains == 2
    np.testing.assert_allclose(t.grain_radius, [27.5, 44.0])


def test_optional_blocks_are_none_not_misread(tmp_path):
    p = tmp_path / "Grains.csv"; p.write_text(_grains_21())
    t = read_grains_csv(p)
    for attr in ("strain_fab", "strain_ken", "euler", "diff_pos_post", "phase_nr"):
        assert getattr(t, attr) is None, f"{attr} should be absent at 21 cols"


def test_strain_blocks_and_euler_on_wide(tmp_path):
    p = tmp_path / "Grains.csv"; p.write_text(_grains_53())
    t = read_grains_csv(p)
    assert t.strain_fab.shape == (2, 3, 3)
    assert t.strain_ken.shape == (2, 3, 3)
    assert t.strain_fab[0, 0, 0] == pytest.approx(100.0)
    assert t.strain_ken[0, 0, 0] == pytest.approx(200.0)
    np.testing.assert_allclose(t.euler[0], [0.1, 0.2, 0.3])
    assert t.strain_units == "microstrain"


def test_require_raises_instead_of_returning_wrong_column(tmp_path):
    p = tmp_path / "Grains.csv"; p.write_text(_grains_21())
    with pytest.raises(GrainsFormatError, match="required column"):
        read_grains_csv(p, require=("eKen11",))


def test_prose_header_is_rejected_loudly(tmp_path):
    """NF's mic2grains.py writes a prose header. Better to raise than to
    guess at columns."""
    p = tmp_path / "Grains.csv"
    p.write_text("%GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence\n"
                 "1 1 0 0 0 1 0 0 0 1 1 2 3 3.6 3.6 3.6 90 90 90 0 0 0 5 0.9\n")
    with pytest.raises(GrainsFormatError, match="orientation columns"):
        read_grains_csv(p)


def test_metadata_from_preamble(tmp_path):
    p = tmp_path / "Grains.csv"; p.write_text(_grains_53())
    t = read_grains_csv(p)
    assert t.space_group == 225
    assert t.num_phases == 1
    np.testing.assert_allclose(t.lattice_parameter, [3.59028]*3 + [90.0]*3)


# --- SpotMatrix.csv ---------------------------------------------------------

def test_matched_only_is_the_default_and_drops_sentinels(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_matched=3, n_unmatched=2))
    t = read_spot_matrix(p)
    assert t.matched_only is True
    assert len(t.spot_id) == 3
    assert t.n_rows_total == 5 and t.n_rows_unmatched == 2
    assert t.spot_id.min() >= 0          # no -1
    assert t.ring_nr.min() >= 0
    assert not np.isnan(t.omega).any()


def test_unmatched_rows_available_on_request(tmp_path):
    """The un-found population is real data -- the per-ring completeness
    deficit lives in it. It must be reachable, just not by accident."""
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_matched=3, n_unmatched=2))
    t = read_spot_matrix(p, matched_only=False)
    assert len(t.spot_id) == 5
    assert (t.spot_id == -1).sum() == 2
    assert np.isnan(t.omega).sum() == 2


def test_spotmatrix_trailing_tab(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(trailing_tab=True))
    t = read_spot_matrix(p)
    assert len(t.spot_id) == 3
    assert t.n_columns == 28


def test_spotmatrix_theta_omega_are_finite_after_filter(tmp_path):
    """A NaN theta reaching a fit is how the un-found rows poison a residual."""
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_matched=4, n_unmatched=3))
    t = read_spot_matrix(p)
    for arr in (t.omega, t.eta, t.theta, t.y_lab, t.z_lab):
        assert np.isfinite(arr).all()
