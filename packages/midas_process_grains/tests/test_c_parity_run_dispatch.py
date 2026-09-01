"""``ProcessGrains.run(mode="c_parity")`` must work — it is the default mode.

Regression: ``run()`` validated against ``VALID_MODES = ('legacy',
'paper_claim', 'adaptive')``, which does not contain ``c_parity``. So the
documented default of both ``ProcessGrains.run`` and ``midas-pipeline`` raised

    ValueError: mode must be one of ('legacy', 'paper_claim', 'adaptive');
                got 'c_parity'

from the library API, while ``run()``'s own docstring listed ``c_parity`` as
accepted. The shipping path never hit it — the pipeline stage shells out to the
CLI, which dispatches c_parity in an early branch — so only a library caller
following the docstring found it.

``run()`` now dispatches to the same ``run_c_parity_pipeline_from_disk`` the CLI
calls, and builds its result by reading back the ``Grains.csv`` that call wrote,
so there is exactly one implementation of the C-parity strain solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_process_grains.params import ProcessGrainsParams
from midas_process_grains.pipeline import ProcessGrains, _result_from_grains_csv
from midas_process_grains.modes import SPOT_AWARE_DISABLED, VALID_MODES


# ── the schema adapter ───────────────────────────────────────────────────────
def _write_grains_csv(path: Path, rows):
    """A minimal legacy 47-column Grains.csv (8 '%' header lines + rows)."""
    with open(path, "w") as f:
        for line in ("%NumGrains {}".format(len(rows)), "%BeamCenter 0 0",
                     "%BeamThickness 0", "%GlobalPosition 0", "%NumPhases 1",
                     "%PhaseInfo", "%\tSpaceGroup:225",
                     "%\tLattice Parameter:0 0 0 0 0 0", "%GrainID\tO11"):
            f.write(line + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.6f}" for v in r) + "\n")


def _row(gid):
    """One row whose every field is distinguishable by column index."""
    r = np.arange(47, dtype=np.float64)
    r[0] = gid
    return r


def test_adapter_maps_the_47_column_schema(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_csv(p, [_row(11), _row(22)])
    res = _result_from_grains_csv(p)

    assert res.n_grains == 2
    assert res.ids.tolist() == [11, 22]
    # 1..9 orientation, row-major 3x3
    assert res.orient_mat[0].flatten().tolist() == list(range(1, 10))
    assert res.positions[0].tolist() == [10.0, 11.0, 12.0]      # 10..12
    assert res.lattice[0].tolist() == [13.0, 14, 15, 16, 17, 18]  # 13..18
    assert res.diff_pos_um[0] == 19.0
    assert res.diff_ome_deg[0] == 20.0
    assert res.diff_angle_deg[0] == 21.0
    assert res.grain_radius[0] == 22.0
    assert res.confidence[0] == 23.0
    assert res.strain_grain[0].flatten().tolist() == list(range(24, 33))  # eFab
    assert res.strain_lab[0].flatten().tolist() == list(range(33, 42))    # eKen
    assert res.rms_error_strain[0] == 42.0
    assert res.phase_nr[0] == 43


def test_adapter_stamps_the_mode_it_actually_ran():
    """Provenance must not be inherited from a dataclass default."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Grains.csv"
        _write_grains_csv(p, [_row(1)])
        assert _result_from_grains_csv(p).mode == "c_parity"


def test_adapter_does_not_invent_a_rep_row(tmp_path):
    """rep_pos indexes OrientPosFit.bin and is not carried by Grains.csv."""
    p = tmp_path / "Grains.csv"
    _write_grains_csv(p, [_row(1)])
    assert _result_from_grains_csv(p).rep_pos.tolist() == [-1]


def test_adapter_handles_an_empty_grain_list(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_csv(p, [])
    res = _result_from_grains_csv(p)
    assert res.n_grains == 0
    assert res.orient_mat.shape == (0, 3, 3)


# ── dispatch ─────────────────────────────────────────────────────────────────
def _pg(tmp_path, **param_kw):
    return ProcessGrains(
        params=ProcessGrainsParams(**param_kw),
        run_dir=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def _patch_disk_run(monkeypatch, tmp_path, capture):
    """Stand in for the heavy C-replica run; write the Grains.csv it would."""
    import midas_process_grains.compute.c_parity_run as cpr

    def fake(**kwargs):
        capture.update(kwargs)
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_grains_csv(out / "Grains.csv", [_row(7)])
        return None

    monkeypatch.setattr(cpr, "run_c_parity_pipeline_from_disk", fake)


def test_c_parity_no_longer_raises(monkeypatch, tmp_path):
    """The exact regression: the default mode must not be rejected."""
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    res = _pg(tmp_path).run(mode="c_parity")
    assert res.n_grains == 1
    assert res.mode == "c_parity"


def test_c_parity_is_the_default_mode(monkeypatch, tmp_path):
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    assert _pg(tmp_path).run().mode == "c_parity"


def test_c_parity_is_absent_from_valid_modes_by_design():
    """It is dispatched before that check, not added to it."""
    assert "c_parity" not in VALID_MODES


def test_dispatch_passes_the_callers_parameter_file(monkeypatch, tmp_path):
    """Not run_dir/'paramstest.txt' — the same defect fixed twice before."""
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    pg = _pg(tmp_path)
    pg.param_file = tmp_path / "oddly_named.txt"
    pg.run(mode="c_parity")
    assert cap["paramstest"] == tmp_path / "oddly_named.txt"


def test_dispatch_defaults_out_dir_to_run_dir(monkeypatch, tmp_path):
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    _pg(tmp_path).run(mode="c_parity")
    assert cap["out_dir"] == tmp_path


def test_out_dir_is_honoured(monkeypatch, tmp_path):
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    dest = tmp_path / "elsewhere"
    res = _pg(tmp_path).run(mode="c_parity", out_dir=dest)
    assert cap["out_dir"] == dest
    assert (dest / "Grains.csv").exists()
    assert res.n_grains == 1


def test_unset_misori_tol_leaves_the_c_default(monkeypatch, tmp_path):
    """None means 'not set in paramstest' — do not pass it through as a value."""
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    _pg(tmp_path, MisoriTol=None).run(mode="c_parity")
    assert "misori_tol_stage1_deg" not in cap


def test_an_explicit_misori_tol_reaches_the_run(monkeypatch, tmp_path):
    cap = {}
    _patch_disk_run(monkeypatch, tmp_path, cap)
    _pg(tmp_path, MisoriTol=0.9).run(mode="c_parity")
    assert cap["misori_tol_stage1_deg"] == pytest.approx(0.9)


def test_missing_output_is_an_error_not_an_empty_result(monkeypatch, tmp_path):
    """A result describing output that was never written is worse than a raise."""
    import midas_process_grains.compute.c_parity_run as cpr
    monkeypatch.setattr(cpr, "run_c_parity_pipeline_from_disk",
                        lambda **kw: None)
    with pytest.raises(RuntimeError, match="was not written"):
        _pg(tmp_path).run(mode="c_parity")


# ── the other modes are unchanged ────────────────────────────────────────────
def test_spot_aware_still_raises(tmp_path):
    with pytest.raises(ValueError) as ei:
        _pg(tmp_path).run(mode="spot_aware")
    assert "DISABLED" in str(ei.value)
    assert str(ei.value) == SPOT_AWARE_DISABLED


def test_unknown_mode_names_c_parity_in_the_error(tmp_path):
    with pytest.raises(ValueError, match="c_parity"):
        _pg(tmp_path).run(mode="not_a_mode")


def test_out_dir_is_rejected_for_pure_compute_modes(tmp_path):
    """Only c_parity writes during run(); silently ignoring out_dir would lie."""
    with pytest.raises(ValueError, match="pure-compute"):
        _pg(tmp_path).run(mode="legacy", out_dir=tmp_path)
