"""Every in-package call of a default-cell function forwards the space group in scope (2026-09-10).

`midas_defect.rows` defaults six functions to La3Ni2O7 (`space_group_number=139`, I-centring). A caller
that HAS a space group but does not pass it silently applies I4/mmm extinctions. Four such calls were
found on 2026-09-10 -- one inside `index_from_cloud`'s cell convergence, three inside `index_from_pairs`
and `index_by_grid` -- the F-vs-I centring error that once halved S5 indexing. The static test fails on
any new one; the runtime test proves the right value reaches the call.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import midas_defect

PKG = Path(midas_defect.__file__).parent
DEFAULT_CELL = {"find_lattice_rows", "index_from_row", "index_from_pairs", "match_mask",
                "cell_from_row", "refine_to_convergence"}


def _drops():
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            if not params & {"space_group_number", "crystal"}:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    nm = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                    if nm in DEFAULT_CELL and "space_group_number" not in {k.arg for k in node.keywords}:
                        bad.append(f"{f.name}:{node.lineno} {fn.name}() -> {nm}")
    return sorted(set(bad))


def test_no_internal_call_drops_an_in_scope_space_group():
    assert _drops() == []


def test_index_from_cloud_converges_the_cell_in_the_crystals_own_space_group(monkeypatch):
    import midas_hkls as mh
    import midas_defect.rows as rows
    import midas_defect.seed_index as seed_index
    from midas_defect.geometry import Geometry
    from midas_defect.indexing import index_from_cloud

    class _Stop(Exception):
        pass

    seen = {}
    monkeypatch.setattr(seed_index, "find_seed_orientation",
                        lambda *a, **k: type("R", (), {"U": np.eye(3), "a": 5.27, "c": 20.5})())

    def _capture(*a, **k):
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(rows, "refine_to_convergence", _capture)
    fmmm = mh.Crystal(lattice=mh.Lattice(a=5.2739, b=5.2384, c=20.5, alpha=90., beta=90., gamma=90.),
                      space_group=mh.SpaceGroup.from_number(69),
                      atoms=[mh.Atom(element="La", fract=(0., 0., 0.), label="La")])
    g = Geometry(lsd_um=349640.6, bcy_px=737.0, bcz_px=810.0, px_um=172.0, wavelength_A=0.42459,
                 n_pix_y=1475, n_pix_z=1679, omega_first_deg=-5.5, omega_step_deg=1.0, n_frames=12)
    z = np.zeros(10)
    with pytest.raises(_Stop):
        index_from_cloud(np.ones((10, 3)), np.ones(10), z, z, z, fmmm, g, np.zeros((1679, 1475), bool),
                         d_min=1.1, max_two_theta_rad=5e-3, max_eta_rad=3e-2, max_omega_rad=3e-2)
    assert seen.get("space_group_number") == 69


def test_index_from_cloud_forwards_the_residual_budget(monkeypatch):
    """PACKAGE_NOTES §10 'still open': the convergence ran on La3Ni2O7's sigma_rtn whatever the sample."""
    import midas_hkls as mh
    import midas_defect.rows as rows
    import midas_defect.seed_index as seed_index
    from midas_defect.geometry import Geometry
    from midas_defect.indexing import index_from_cloud

    class _Stop(Exception):
        pass

    seen = {}
    monkeypatch.setattr(seed_index, "find_seed_orientation",
                        lambda *a, **k: type("R", (), {"U": np.eye(3), "a": 3.6, "c": 19.2})())

    def _capture(*a, **k):
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(rows, "refine_to_convergence", _capture)
    cry = mh.Crystal(lattice=mh.Lattice(a=3.6116, b=3.6116, c=19.2516, alpha=90., beta=90., gamma=90.),
                     space_group=mh.SpaceGroup.from_number(139),
                     atoms=[mh.Atom(element="La", fract=(0., 0., 0.), label="La")])
    g = Geometry(lsd_um=349640.6, bcy_px=737.0, bcz_px=810.0, px_um=172.0, wavelength_A=0.42459,
                 n_pix_y=1475, n_pix_z=1679, omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40)
    z = np.zeros(10)
    with pytest.raises(_Stop):
        index_from_cloud(np.ones((10, 3)), np.ones(10), z, z, z, cry, g, np.zeros((1679, 1475), bool),
                         d_min=1.1, max_two_theta_rad=5e-3, max_eta_rad=3e-2, max_omega_rad=3e-2,
                         sigma_rtn=(0.001, 0.002, 0.003), tol_sigma=5.0)
    assert seen.get("sigma_rtn") == (0.001, 0.002, 0.003) and seen.get("tol_sigma") == 5.0
    seen.clear()
    with pytest.raises(_Stop):
        index_from_cloud(np.ones((10, 3)), np.ones(10), z, z, z, cry, g, np.zeros((1679, 1475), bool),
                         d_min=1.1, max_two_theta_rad=5e-3, max_eta_rad=3e-2, max_omega_rad=3e-2)
    assert "sigma_rtn" not in seen and "tol_sigma" not in seen        # defaults stay the callee's
