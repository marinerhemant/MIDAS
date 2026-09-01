"""Grains.csv readers here must resolve columns by NAME, at every width.

``tests/conftest.py``'s demk fixture and the two ``dev/paper/smoke_3d_dpdf*``
scripts all sliced positionally behind a ``len(parts) < 21`` guard. That guard
is not a width check -- a 47- or 53-column file passes it -- and columns 13..18
changed MEANING at 47:

    21 columns: 13..18 = E11 E22 E33 E12 E13 E23   (a Voigt STRAIN)
    47/53     : 13..18 = a b c alpha beta gamma    (a LATTICE)

and ``vals[19]``/``vals[20]``, read as GrainRadius/Confidence on a 21-column
file, are DiffPos/DiffOme on a modern one. None of that raises. It is correct
today only because ``Grains_L1_local.csv`` really is 21 columns.

These tests pin the name-driven behaviour at both widths, under both ID
spellings, with and without the trailing tab the C writer emits.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import _read_grains_by_name

_DEV = Path(__file__).resolve().parents[1] / "dev" / "paper"

_COLS_21 = ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
            "X", "Y", "Z", "E11", "E22", "E33", "E12", "E13", "E23",
            "GrainRadius", "Confidence"]
_COLS_53 = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
)

#: Distinct per-column values so a one-off slice cannot pass by coincidence.
_VALUES = {
    "O11": 1.0, "O12": 0.0, "O13": 0.0,
    "O21": 0.0, "O22": 1.0, "O23": 0.0,
    "O31": 0.0, "O32": 0.0, "O33": 1.0,
    "X": 11.0, "Y": 22.0, "Z": 33.0,
    "E11": 1e-4, "E22": 2e-4, "E33": 3e-4,
    "E12": 4e-4, "E13": 5e-4, "E23": 6e-4,
    "a": 3.6356, "b": 3.6356, "c": 3.6356,
    "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    "DiffPos": 7.7, "DiffOme": 0.123, "DiffAngle": 0.456,
    "GrainRadius": 31.5, "Confidence": 0.42,
    "RMSErrorStrain": 410.0, "PhaseNr": 1.0,
}


def _write(path, cols, *, id_token="ID", trailing_tab=False, n=3):
    lines = [
        f"%NumGrains {n}", "%BeamCenter 0.0 0.0", "%BeamThickness 0.0",
        "%GlobalPosition 0.0", "%NumPhases 1", "%PhaseInfo",
        "%\tSpaceGroup:225",
        "%\tLattice Parameter: 3.6356 3.6356 3.6356 90.0 90.0 90.0",
        "%" + "\t".join([id_token] + cols),
    ]
    for k in range(n):
        row = [str(k + 1)] + [f"{_VALUES.get(c, 0.0):.6f}" for c in cols]
        lines.append("\t".join(row) + ("\t" if trailing_tab else ""))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_legacy_21col_reads_the_voigt_strain(tmp_path, id_token, trailing_tab):
    p = _write(tmp_path / "Grains.csv", _COLS_21, id_token=id_token,
               trailing_tab=trailing_tab)
    OM, pos, eps, radii, confs, n_cols = _read_grains_by_name(p)
    assert n_cols == 21
    assert OM.shape == (3, 3, 3)
    np.testing.assert_allclose(OM[0], np.eye(3))
    np.testing.assert_allclose(pos[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(eps[0], [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    np.testing.assert_allclose(radii, 31.5)
    np.testing.assert_allclose(confs, 0.42)


@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_wide_53col_never_reports_a_lattice_as_strain(tmp_path, id_token,
                                                      trailing_tab):
    """The whole point: eps must be None, NOT [3.6356, 3.6356, ..., 90.0]."""
    p = _write(tmp_path / "Grains.csv", _COLS_53, id_token=id_token,
               trailing_tab=trailing_tab)
    OM, pos, eps, radii, confs, n_cols = _read_grains_by_name(p)
    assert n_cols == 53
    assert eps is None
    np.testing.assert_allclose(pos[0], [11.0, 22.0, 33.0])
    # GrainRadius/Confidence come from the NAMED columns, not 19/20 (which on
    # this width are DiffOme = 0.123 and DiffAngle = 0.456).
    np.testing.assert_allclose(radii, 31.5)
    np.testing.assert_allclose(confs, 0.42)


def test_demk_fixture_fails_loudly_on_a_regenerated_file(tmp_path):
    """The fixture's ground truth is tied to the legacy Voigt block; a wide
    file must stop the comparison rather than silently change its meaning."""
    p = _write(tmp_path / "Grains.csv", _COLS_53)
    _, _, eps, _, _, n_cols = _read_grains_by_name(p)
    assert eps is None and n_cols == 53
    # conftest turns exactly that into a pytest.fail with an actionable
    # message; assert the branch condition rather than re-running the fixture.


@pytest.mark.parametrize("cols,has_eps", [(_COLS_21, True), (_COLS_53, False)],
                         ids=["21col", "53col"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_inline_fallback_agrees_with_the_canonical_reader(tmp_path, monkeypatch,
                                                          cols, has_eps,
                                                          trailing_tab):
    """The no-midas_process_grains path must give identical answers.

    ``midas_process_grains`` is not a declared dependency of midas_defect, so
    the fallback is what runs in a minimal install -- and an untested fallback
    is how a positional slice creeps back in.
    """
    p = _write(tmp_path / "Grains.csv", cols, trailing_tab=trailing_tab)
    ref = _read_grains_by_name(p)
    # Setting a sys.modules entry to None makes `import` raise ImportError.
    monkeypatch.setitem(sys.modules, "midas_process_grains.io", None)
    got = _read_grains_by_name(p)

    np.testing.assert_allclose(got[0], ref[0])          # OM
    np.testing.assert_allclose(got[1], ref[1])          # pos
    if has_eps:
        np.testing.assert_allclose(got[2], ref[2])      # Voigt strain
    else:
        assert got[2] is None and ref[2] is None
    np.testing.assert_allclose(got[3], ref[3])          # GrainRadius
    np.testing.assert_allclose(got[4], ref[4])          # Confidence
    assert got[5] == ref[5] == len(cols) + 1   # +1 for the ID column


def _load_smoke(name):
    p = _DEV / name
    if not p.exists():
        pytest.skip(f"dev script not present: {p}")
    src = p.read_text()
    # These scripts import optional heavy deps at module scope; pull just the
    # loader out rather than executing the whole file.
    import ast
    tree = ast.parse(src)
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "load_grains"),
                None)
    assert node is not None, f"{p}: no top-level def load_grains"
    ns = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), ns)
    return ns["load_grains"]


@pytest.mark.parametrize("script", ["smoke_3d_dpdf.py",
                                    "smoke_3d_dpdf_corrected.py"])
@pytest.mark.parametrize("cols,width", [(_COLS_21, 21), (_COLS_53, 53)],
                         ids=["21col", "53col"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_smoke_script_loader_reads_confidence_by_name(tmp_path, script, cols,
                                                      width, trailing_tab):
    """``vals[20]`` is Confidence at 21 columns and DiffOme at 53."""
    load_grains = _load_smoke(script)
    p = _write(tmp_path / "Grains.csv", cols, trailing_tab=trailing_tab)
    OM, pos, conf = load_grains(p)
    assert OM.shape == (3, 3, 3)
    np.testing.assert_allclose(OM[0], np.eye(3))
    np.testing.assert_allclose(pos[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(conf, 0.42)   # never 0.123 (DiffOme)


@pytest.mark.parametrize("script", ["smoke_3d_dpdf.py",
                                    "smoke_3d_dpdf_corrected.py"])
def test_smoke_script_loader_matches_the_real_file(script):
    """Same answers as before the change, on the file actually used."""
    from .conftest import DEMK_FCC_ROOT
    real = DEMK_FCC_ROOT / "Grains_L1_local.csv"
    if not real.exists():
        pytest.skip(f"demk FCC L1 data not mounted at {DEMK_FCC_ROOT}")
    load_grains = _load_smoke(script)
    OM, pos, conf = load_grains(real)
    # Reference: the old positional slice, on a file that really is 21 columns.
    ref_om, ref_pos, ref_conf = [], [], []
    for line in real.read_text().splitlines():
        if line.startswith("%") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        if len(v) < 21:
            continue
        ref_om.append(np.array(v[1:10]).reshape(3, 3))
        ref_pos.append(v[10:13])
        ref_conf.append(v[20])
    np.testing.assert_allclose(OM, np.stack(ref_om))
    np.testing.assert_allclose(pos, np.array(ref_pos))
    np.testing.assert_allclose(conf, np.array(ref_conf))
