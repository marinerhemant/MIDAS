"""The paper/dev Grains.csv loaders must follow the file's own header.

Both ``dev/paper/figures/plot_grain_irrad_compare.py`` (which feeds a paper
figure) and ``dev/paper/grain_recovery/local_m3_m6.py`` carried a hard-coded
47-name column tuple and did ``df.columns = COLS``. Grains.csv was widened to
53 columns on 2026-08-21, so both raised

    ValueError: Length mismatch: Expected axis has 53 elements,
                new values have 47

and, worse, a widening that happened to preserve the count would have
relabelled every column in silence. These tests pin the header-driven
behaviour at BOTH widths, under both ID spellings, with and without the
trailing tab the C writer emits.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")

_DEV = Path(__file__).resolve().parents[1] / "dev" / "paper"
_FIG = _DEV / "figures" / "plot_grain_irrad_compare.py"
_M3M6 = _DEV / "grain_recovery" / "local_m3_m6.py"

_COLS_47 = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2"]
)
_COLS_53 = _COLS_47 + ["DiffPosPre", "DiffOmePre", "DiffAnglePre",
                       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]


def _write_grains(path, cols, *, id_token="ID", trailing_tab=False, n=3):
    """A Grains.csv of the requested width.

    ``trailing_tab`` reproduces the C ProcessGrains writer, which terminates
    every data row with a tab.
    """
    lines = [
        f"%NumGrains {n}", "%BeamCenter 0", "%BeamThickness 200",
        "%GlobalPosition 0", "%NumPhases 1", "%PhaseInfo",
        "%\tSpaceGroup:225",
        "%\tLattice Parameter: 4.0 4.0 4.0 90.0 90.0 90.0",
        "%" + "\t".join([id_token] + cols),
    ]
    base = {"O11": 1, "O22": 1, "O33": 1, "X": 10.0, "Y": 20.0, "Z": 30.0,
            "a": 4.0, "b": 4.0, "c": 4.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "GrainRadius": 2.0, "Confidence": 1.0, "PhaseNr": 1,
            "eKen11": 300.0, "eKen22": 300.0, "eKen33": 300.0,
            "DiffPos": 5.0, "DiffOme": 0.05, "DiffAngle": 0.08,
            "RMSErrorStrain": 400.0}
    for k in range(n):
        row = [str(k + 1)] + [f"{float(base.get(c, 0.0)):.6f}" for c in cols]
        lines.append("\t".join(row) + ("\t" if trailing_tab else ""))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _extract_function(path, func_name):
    """Pull one top-level function out of a bare script.

    ``local_m3_m6.py`` runs its analysis at import time (and needs .npz files
    that are not in the repo), so it cannot simply be imported. Exec just the
    function definition instead.
    """
    src = Path(path).read_text()
    tree = ast.parse(src)
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert node is not None, f"{path}: no top-level def {func_name}"
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(path), "exec"), ns)
    return ns[func_name]


@pytest.mark.parametrize("cols", [_COLS_47, _COLS_53], ids=["47col", "53col"])
@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_figure_loader_follows_the_file_header(tmp_path, cols, id_token,
                                               trailing_tab):
    if not _FIG.exists():
        pytest.skip(f"dev script not present: {_FIG}")
    mod = _load_module(_FIG, "_pf_plot_grain_irrad_compare")
    p = _write_grains(tmp_path / "Grains.csv", cols, id_token=id_token,
                      trailing_tab=trailing_tab)
    assert mod._grains_columns(p) == [id_token] + cols
    df = mod._load(str(p))
    assert list(df.columns)[:len(cols) + 1] == [id_token] + cols
    assert len(df) == 3
    np.testing.assert_allclose(df["GrainRadius"].values, 2.0)
    # eps_h is the eKen trace mean minus its volume-weighted mean -> 0 here.
    np.testing.assert_allclose(df["eps_h"].values, 0.0, atol=1e-9)


@pytest.mark.parametrize("cols", [_COLS_47, _COLS_53], ids=["47col", "53col"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_m3m6_loader_follows_the_file_header(tmp_path, cols, trailing_tab):
    if not _M3M6.exists():
        pytest.skip(f"dev script not present: {_M3M6}")
    grains_columns = _extract_function(_M3M6, "grains_columns")
    p = _write_grains(tmp_path / "Grains.csv", cols, trailing_tab=trailing_tab)
    got = grains_columns(p)
    assert got == ["ID"] + cols

    # ...and the pandas read it feeds must line up with those names.
    import pandas as pd
    g = pd.read_csv(p, sep=r"\s+", comment="%", header=None, names=got)
    assert g.shape == (3, len(cols) + 1)
    np.testing.assert_allclose(g["GrainRadius"].values, 2.0)


def test_prose_header_raises_rather_than_guessing(tmp_path):
    if not _FIG.exists():
        pytest.skip(f"dev script not present: {_FIG}")
    mod = _load_module(_FIG, "_pf_plot_grain_irrad_compare")
    p = tmp_path / "Grains.csv"
    p.write_text("%NumGrains 1\n%GrainID OrientMat(9) X Y Z LatC(6)\n"
                 "1 1 0 0 0 1 0 0 0 1 1 2 3 4 4 4 90 90 90\n")
    with pytest.raises(ValueError, match="O11"):
        mod._grains_columns(p)
