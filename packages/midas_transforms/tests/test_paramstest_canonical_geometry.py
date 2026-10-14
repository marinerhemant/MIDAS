"""write_paramstest emits the geometry under the canonical keys (Lsd, BC,
tx/ty/tz, LatticeConstant, ResidualCorrectionMap) beside the *Fit echoes, so a
reader that only knows canonical keys gets the geometry the run used."""
from midas_transforms.params import ParamsTest, read_paramstest, write_paramstest


def _pt():
    p = ParamsTest(Wavelength=0.172979, Lsd=998809.737, px=150.0)
    p.LatticeConstant = (3.59, 3.59, 3.59, 90.0, 90.0, 90.0)
    p.LsdFit = 998809.737; p.YBCFit = 1430.431081; p.ZBCFit = 1342.479441
    p.txFit = -0.024622; p.tyFit = -0.462852; p.tzFit = -0.222085
    p.RingNumbers = [1]; p.RingRadii = [83575.3]; p.BeamSize = 100.0
    return p


def test_canonical_block_written_with_fit_values(tmp_path):
    p = _pt(); p.ResidualCorrectionMap = "/abs/residual_corr.bin"
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    kv = {}
    for line in f.read_text().splitlines():
        t = line.rstrip(";").split()
        if t:
            kv.setdefault(t[0], t[1:])
    assert float(kv["Lsd"][0]) == float(kv["LsdFit"][0]) == 998809.737
    assert [float(v) for v in kv["BC"]] == [1430.431081, 1342.479441]
    assert float(kv["tx"][0]) == -0.024622 and float(kv["tz"][0]) == -0.222085
    assert [float(v) for v in kv["LatticeConstant"]][:3] == [3.59, 3.59, 3.59]
    assert kv["ResidualCorrectionMap"] == ["/abs/residual_corr.bin"]
    # legacy names still present for older readers
    for k in ("Distance", "LsdFit", "YBCFit", "ZBCFit", "txFit", "LatticeParameter"):
        assert k in kv


def test_no_residual_map_line_when_unset(tmp_path):
    f = tmp_path / "paramstest.txt"
    write_paramstest(_pt(), f)
    assert "ResidualCorrectionMap" not in f.read_text()


def test_roundtrip_reads_same_geometry(tmp_path):
    p = _pt(); p.ResidualCorrectionMap = "/abs/map.bin"
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    q = read_paramstest(f)
    assert q.Lsd == 998809.737 and q.LsdFit == 998809.737
    assert q.txFit == -0.024622 and q.ResidualCorrectionMap == "/abs/map.bin"


def test_midas_calibrate_reader_sees_the_run_geometry(tmp_path):
    from midas_calibrate.params import CalibrationParams
    f = tmp_path / "paramstest.txt"
    write_paramstest(_pt(), f)
    c = CalibrationParams.from_file(f)
    assert c.Lsd == 998809.737 and c.tx == -0.024622
    assert (c.BC_y, c.BC_z) == (1430.431081, 1342.479441)
    assert "AliasedKeys" not in c.extra        # canonical keys present: no aliasing needed
