"""A per-layer paramstest.txt (written by the FF pipeline before it emitted
canonical keys) spells the geometry it ran with as LsdFit / YBCFit,ZBCFit /
txFit,tyFit,tzFit / LatticeParameter. Read here without aliases it gave Lsd = 0
and tx = 0, silently (grain-tx then matched 0 spots and wrote ``tx 0``)."""
from midas_calibrate.params import CalibrationParams

LAYER = """LatticeParameter 3.590000 3.590000 3.590000 90.000000 90.000000 90.000000;
Wavelength 0.172979;
Distance 998809.737000;
px 150.000000;
Wedge 0.000000;
LsdFit 998809.737000
YBCFit 1430.431081
ZBCFit 1342.479441
txFit -0.024622
tyFit -0.462852
tzFit -0.222085
OmegaStart -179.749486
OmegaStep 0.250514
"""


def test_layer_paramstest_fit_keys_map_to_canonical(tmp_path):
    f = tmp_path / "paramstest.txt"; f.write_text(LAYER)
    p = CalibrationParams.from_file(f)
    assert p.Lsd == 998809.737
    assert (p.BC_y, p.BC_z) == (1430.431081, 1342.479441)
    assert (p.tx, p.ty, p.tz) == (-0.024622, -0.462852, -0.222085)
    assert tuple(p.LatticeConstant)[:3] == (3.59, 3.59, 3.59)
    assert "Lsd<-" in p.extra["AliasedKeys"] and "tx<-txFit" in p.extra["AliasedKeys"]


def test_canonical_key_wins_over_fit_alias(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text("Lsd 1000000\ntx 0.1\nBC 1000 1001\n" + LAYER)
    p = CalibrationParams.from_file(f)
    assert p.Lsd == 1000000.0 and p.tx == 0.1 and (p.BC_y, p.BC_z) == (1000.0, 1001.0)
    assert "Lsd<-" not in p.extra.get("AliasedKeys", "")
