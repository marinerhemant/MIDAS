"""``Completeness`` must reach the gate, and the gate must actually gate.

C ProcessGrains drops every grain whose completeness falls below the user's
``Completeness`` key. This package parsed the same physical quantity under its
own name ``ConfidenceTol`` -- documented as "minimum Confidence to keep a
grain" -- and then never applied it anywhere. ``Completeness`` itself was not
even a recognised key.

Measured on the datasetA Ni layer, from one refiner output:

    C ProcessGrains, handed the zarr archive (carries MinNrSpots 3,
    Completeness 0.5) ............................................  6132 grains
    C ProcessGrains, handed paramstest.txt (carries neither) ...... 23710 grains

paramstest.txt is what the pipeline hands process-grains, so the python side
was running with no gate at all and the 4x grain-count gap was read as an
algorithmic difference between the two implementations for most of a session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_process_grains.params import read_paramstest_pg


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "ps.txt"
    p.write_text(body)
    return p


# ---------------------------------------------------------------- parsing ---

def test_completeness_key_is_parsed(tmp_path):
    p = _write(tmp_path, "SpaceGroup 225;\nCompleteness 0.5;\n")
    assert read_paramstest_pg(p).Completeness == 0.5


def test_completeness_populates_the_gate(tmp_path):
    """The user writes Completeness; the code reads ConfidenceTol."""
    p = _write(tmp_path, "SpaceGroup 225;\nCompleteness 0.5;\n")
    assert read_paramstest_pg(p).ConfidenceTol == 0.5


def test_an_explicit_confidence_tol_wins(tmp_path):
    p = _write(tmp_path, "SpaceGroup 225;\nCompleteness 0.5;\nConfidenceTol 0.8;\n")
    assert read_paramstest_pg(p).ConfidenceTol == 0.8


def test_absent_completeness_leaves_no_gate(tmp_path):
    """The old behaviour survives for files that set neither key."""
    ps = read_paramstest_pg(_write(tmp_path, "SpaceGroup 225;\n"))
    assert ps.Completeness is None
    assert ps.ConfidenceTol == 0.0


def test_completeness_out_of_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Completeness"):
        read_paramstest_pg(_write(tmp_path, "SpaceGroup 225;\nCompleteness 50;\n"))


def test_midas_punctuation_is_tolerated(tmp_path):
    """MIDAS parameter files carry trailing ';' and '#' comments."""
    p = _write(tmp_path, "Completeness 0.5;  # keep only well-matched grains\n")
    assert read_paramstest_pg(p).ConfidenceTol == 0.5


# ------------------------------------------------------------- the gate -----

def _set_completeness(run_dir: Path, values) -> None:
    """Give the fixture's three seeds distinct completeness AND distinct
    orientations, so they survive clustering as separate grains and the gate
    has something to choose between. All three ship with the identity
    orientation, which collapses them into one grain."""
    opf = np.fromfile(run_dir / "Results" / "OrientPosFit.bin",
                      dtype=np.float64).reshape(-1, 27)
    assert opf.shape[0] == len(values)
    opf[:, 26] = values
    for i in range(opf.shape[0]):
        # rotation by i*30 deg about z — far beyond MisoriTol 0.25 and not a
        # cubic symmetry equivalent of the identity.
        c, s = np.cos(np.radians(30.0 * i)), np.sin(np.radians(30.0 * i))
        opf[i, 1:10] = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
    opf.tofile(run_dir / "Results" / "OrientPosFit.bin")


def _append_param(run_dir: Path, line: str) -> None:
    ps = run_dir / "paramstest.txt"
    ps.write_text(ps.read_text().rstrip("\n") + "\n" + line + "\n")


def _run(run_dir: Path):
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(run_dir / "paramstest.txt", device="cpu")
    return pg.run(mode="adaptive")   # was spot_aware (disabled)


def test_grains_below_the_gate_are_dropped(tiny_run_dir):
    _set_completeness(tiny_run_dir, [0.95, 0.40, 0.95])

    ungated = _run(tiny_run_dir)
    _append_param(tiny_run_dir, "Completeness 0.5")
    gated = _run(tiny_run_dir)

    assert len(gated.confidence) < len(ungated.confidence), (
        "a 0.40-completeness seed must not survive Completeness 0.5"
    )
    assert float(gated.confidence.min()) >= 0.5


def test_no_gate_keeps_everything(tiny_run_dir):
    """Without the key the result is exactly what it was before this change."""
    _set_completeness(tiny_run_dir, [0.95, 0.40, 0.95])
    res = _run(tiny_run_dir)
    assert float(res.confidence.min()) < 0.5


def test_per_grain_diagnostics_stay_in_lockstep(tiny_run_dir):
    """Filtering out_grains without the parallel diag lists misattributes
    every per-grain diagnostic to the wrong grain."""
    _set_completeness(tiny_run_dir, [0.95, 0.40, 0.95])
    _append_param(tiny_run_dir, "Completeness 0.5")
    res = _run(tiny_run_dir)

    n = len(res.confidence)
    diag = getattr(res, "diagnostics", None) or {}
    for name in ("cluster_sizes", "n_resolved_hkls", "n_majority_hkls",
                 "n_residual_tie_hkls", "n_forward_sim_hkls"):
        if name in diag:
            assert len(diag[name]) == n, f"{name} desynchronised from grains"
