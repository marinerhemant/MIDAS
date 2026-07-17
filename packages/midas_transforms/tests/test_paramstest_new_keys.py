"""P0-1 regression tests: paramstest.txt must carry the full raw-frame
geometry — ``txFit``, ALL distortion coefficients (v1 p0..p14 and, when
present, the canonical v2 names), and ``OmegaStart``/``OmegaStep``.

The C FitSetupZarr wrote LsdFit/YBCFit/ZBCFit/tyFit/tzFit + p0..p3 only.
Any raw-frame consumer (midas_pf_odf, midas_grain_odf) rebuilding geometry
from such a file got tx=0 — a ~0.27° in-plane rotation ≈ 3-4e3 µε of fake
strain on real Varex data — and an unusable omega description (P0-3:
inferring the step from shadow-gapped OmegaRange spans gave 0.0514° instead
of 0.25°).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_transforms.params import (
    ParamsTest, ZarrParams, read_paramstest, write_paramstest,
)


def _minimal_pt(**kw) -> ParamsTest:
    p = ParamsTest()
    p.Wavelength = 0.2254
    p.Lsd = 1_000_000.0
    p.RingNumbers = [1, 2]
    p.RingRadii = [500.0, 700.0]
    p.RingToIndex = 2
    p.BeamSize = 2000.0
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_txfit_written_and_read_back(tmp_path: Path):
    p = _minimal_pt(txFit=-0.2737, tyFit=0.1, tzFit=-0.05)
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    text = f.read_text()
    assert "txFit -0.273700" in text
    back = read_paramstest(f)
    assert back.txFit == pytest.approx(-0.2737)
    assert back.tyFit == pytest.approx(0.1)
    assert back.tzFit == pytest.approx(-0.05)


def test_full_distortion_p0_to_p14_round_trip(tmp_path: Path):
    vals = {f"p{i}": (i + 1) * 0.5 for i in range(15)}
    vals["p3"] = 35.5          # phi4-style phase (the emerson-scale value)
    vals["p2"] = -1.1e-9       # small-amplitude harmonic — %f would flush to 0
    p = _minimal_pt(**vals)
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    back = read_paramstest(f)
    for k, v in vals.items():
        assert getattr(back, k) == pytest.approx(v, rel=1e-9), k


def test_omega_start_step_round_trip(tmp_path: Path):
    p = _minimal_pt(OmegaStart=-180.0, OmegaStep=0.25)
    # Shadow-gapped multi-range scan (the P0-3 trap): spans must NOT be the
    # step source once explicit keys exist.
    p.OmegaRanges = [(-180.0, -106.0), (-76.0, 74.0), (105.0, 180.0)]
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    back = read_paramstest(f)
    assert back.OmegaStart == pytest.approx(-180.0)
    assert back.OmegaStep == pytest.approx(0.25)


def test_omega_keys_omitted_when_step_unknown(tmp_path: Path):
    """A false ``OmegaStep 0.0`` is worse than absence (E5 class)."""
    p = _minimal_pt()          # OmegaStep defaults to 0.0
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    text = f.read_text()
    assert "OmegaStep" not in text
    assert "OmegaStart" not in text


def test_v2_distortion_names_written_when_native(tmp_path: Path):
    from midas_distortion import P_COEF_NAMES
    coeffs = np.linspace(0.01, 0.15, 15)
    p = _minimal_pt(dist_coeffs_v2=coeffs)
    f = tmp_path / "paramstest.txt"
    write_paramstest(p, f)
    text = f.read_text()
    for nm in P_COEF_NAMES:
        assert f"\n{nm} " in "\n" + text, f"missing v2 key {nm}"
    back = read_paramstest(f)
    np.testing.assert_allclose(back.dist_coeffs_v2, coeffs, rtol=1e-9)


def test_zarr_params_to_paramstest_carries_geometry():
    zp = ZarrParams()
    zp.Lsd = 1_000_000.0
    zp.Wavelength = 0.2254
    zp.YCen = 1024.0
    zp.ZCen = 1023.5
    zp.tx = -0.2737
    zp.ty = 0.12
    zp.tz = -0.08
    zp.OmegaStart = -180.0
    zp.OmegaStep = 0.25
    for i in range(15):
        setattr(zp, f"p{i}", 0.1 * (i + 1))
    pt = zp.to_paramstest()
    assert pt.txFit == pytest.approx(-0.2737)
    assert pt.tyFit == pytest.approx(0.12)
    assert pt.tzFit == pytest.approx(-0.08)
    assert pt.OmegaStart == pytest.approx(-180.0)
    assert pt.OmegaStep == pytest.approx(0.25)
    for i in range(15):
        assert getattr(pt, f"p{i}") == pytest.approx(0.1 * (i + 1)), f"p{i}"


_MIDAS_BIN = Path.home() / "opt" / "MIDAS" / "build" / "bin"


@pytest.mark.parametrize("binary,args", [
    ("IndexerOMP", ["0", "1", "10", "2"]),
    ("FitPosOrStrainsOMP", ["1", "0", "10", "2"]),
])
def test_c_binaries_tolerate_new_keys(tmp_path: Path, binary: str, args):
    """E6 smoke: the C parsers keyword-match and skip unknown keys, so a
    paramstest carrying txFit / p4..p14 / v2 names / OmegaStart/Step must
    behave IDENTICALLY to a legacy file (same output, same exit) — both
    fail at the same missing-binary-inputs point on this bare fixture."""
    import subprocess
    exe = _MIDAS_BIN / binary
    if not exe.exists():
        pytest.skip(f"{binary} not built in ~/opt/MIDAS/build/bin")

    p = _minimal_pt(txFit=-0.2737, OmegaStart=-180.0, OmegaStep=0.25,
                    **{f"p{i}": 0.1 * (i + 1) for i in range(15)})
    p.dist_coeffs_v2 = np.linspace(0.01, 0.15, 15)
    new_f = tmp_path / "paramstest_new.txt"
    write_paramstest(p, new_f)

    legacy = _minimal_pt(**{f"p{i}": 0.1 * (i + 1) for i in range(4)})
    legacy_f = tmp_path / "paramstest_legacy.txt"
    write_paramstest(legacy, legacy_f)
    # Strip the new keys the writer now always emits, to get a true
    # legacy-shaped file.
    kept = [ln for ln in legacy_f.read_text().splitlines()
            if not ln.startswith(("txFit", "p4 ", "p5 ", "p6 ", "p7 ",
                                  "p8 ", "p9 ", "p10 ", "p11 ", "p12 ",
                                  "p13 ", "p14 ", "OmegaStart", "OmegaStep"))]
    legacy_f.write_text("\n".join(kept) + "\n")

    outs = {}
    for f in (legacy_f, new_f):
        r = subprocess.run([str(exe), str(f), *args], cwd=tmp_path,
                           capture_output=True, text=True, timeout=60)
        outs[f.name] = (r.returncode, r.stdout, r.stderr)
    assert outs["paramstest_legacy.txt"] == outs["paramstest_new.txt"], (
        "C binary behaviour changed when the new paramstest keys are present"
    )


def test_legacy_c_paramstest_still_parses(tmp_path: Path):
    """A C-written file (p0..p3 only, no txFit, trailing ';') must still
    read cleanly with benign defaults for the new fields."""
    f = tmp_path / "paramstest.txt"
    f.write_text(
        "LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
        "SpaceGroup 225;\n"
        "Wavelength 0.2254;\n"
        "Distance 1000000.0;\n"
        "RingNumbers 2;\n"
        "RingRadii 700.0;\n"
        "LsdFit 1000000.0\n"
        "YBCFit 1024.0\nZBCFit 1023.5\n"
        "tyFit 0.12\ntzFit -0.08\n"
        "p0 0.1\np1 0.2\np2 0.3\np3 35.5\n"
    )
    back = read_paramstest(f)
    assert back.txFit == 0.0
    assert back.p3 == pytest.approx(35.5)
    assert back.p14 == 0.0
    assert back.OmegaStep == 0.0
