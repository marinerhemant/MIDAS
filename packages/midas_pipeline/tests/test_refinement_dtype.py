"""FF refinement must not inherit the run's throughput dtype.

``cli._resolve_dtype`` maps ``--dtype auto`` to float32 on cuda/mps "for
production speed". That is a reasonable trade for peak fitting and a wrong
answer for FF grain position: the X-along-beam Jacobian enters the forward
model only as ``(Lsd − x)/Lsd`` (~1e-3 at Lsd = 1.67 m), so the normal
equations are near-singular in that direction and fp32 cannot resolve it.

Measured on the 1-ID GE5 Au3 scan (2026-07-30), refining the *identical*
seeds from the C ``IndexerOMP``'s ``IndexBest.bin`` and comparing against the
C reference ``FitPosOrStrainsOMP``:

    dtype     device   median |Δposition| vs C   DiffPos median
    float64   cpu               13.4 µm              199.07
    float64   cuda              13.4 µm              199.10
    float32   cpu              158.2 µm              231.89
    float32   cuda             158.2 µm              231.93

cpu and cuda agree to 3 s.f. within each dtype, so this is precision — not a
GPU or TF32 artefact.
"""

from __future__ import annotations

import inspect

from midas_pipeline.config import RefinementConfig
from midas_pipeline.stages import refinement as refinement_stage


def test_refinement_dtype_defaults_to_float64():
    assert RefinementConfig().dtype == "float64", (
        "FF refinement defaulted away from float64; grain positions will be "
        "~158 um off the C reference"
    )


def test_refinement_dtype_is_independent_of_the_run_dtype():
    """It must be its own field, not an alias that a global --dtype can
    silently downgrade."""
    cfg = RefinementConfig()
    cfg.dtype = "float32"
    assert RefinementConfig().dtype == "float64", "dtype leaked between instances"


def test_ff_branch_passes_the_refinement_dtype_not_the_run_dtype():
    """Source guard: the subprocess must be handed
    ``ctx.config.refinement.dtype``. Passing ``ctx.config.dtype`` is exactly
    the regression this file exists to prevent, and it is invisible in any
    output except the grain positions themselves."""
    src = inspect.getsource(refinement_stage)
    i = src.index('"--dtype"')
    window = src[i:i + 120]
    assert "refine_dtype" in window, (
        f"refinement passes {window!r} — must forward "
        f"ctx.config.refinement.dtype, not the run's global dtype"
    )
    # and refine_dtype must be bound from the refinement config
    assert "refine_dtype = ctx.config.refinement.dtype" in src


# ── the failure must reach the log people actually read ──────────────────

def test_unrefined_position_warning_is_surfaced_from_the_subprocess_log(
    tmp_path, caplog
):
    """FF refinement runs in a subprocess whose output goes to
    ``refinement_{out,err}.csv``, which nobody reads. A run where the solver
    silently returned seed positions therefore looked completely normal in
    ``ff_run.log`` — that is how ~158 um of grain-position error shipped
    unnoticed. The stage must promote it."""
    import logging

    from midas_pipeline.stages.refinement import (
        _UNREFINED_MARKER, _surface_unrefined_positions,
    )

    (tmp_path / "refinement_err.csv").write_text(
        "some noise\n"
        f"WARNING midas_fit_grain.driver: {_UNREFINED_MARKER} no grain "
        "position moved a measurable distance (max 0.000493 um, median "
        "0.000493 um over 20 grains; floor 0.2 um = px/1000; 20 "
        "bit-identical to their seed; dtype=float32, solver=lbfgs).\n"
        "more noise\n"
    )
    with caplog.at_level(logging.WARNING):
        _surface_unrefined_positions(tmp_path)

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "no grain position moved" in joined, joined
    assert "dtype=float32" in joined, "the dtype must be named in the warning"
    assert "20 grains" in joined


def test_no_warning_when_the_refiner_did_move_the_grains(tmp_path, caplog):
    """It must stay quiet on a healthy run, or it becomes noise people learn
    to ignore."""
    import logging

    from midas_pipeline.stages.refinement import _surface_unrefined_positions

    (tmp_path / "refinement_out.csv").write_text(
        "INFO midas_fit_grain.driver: refining 20 grains (skipped 0 empty slots)\n"
    )
    with caplog.at_level(logging.WARNING):
        _surface_unrefined_positions(tmp_path)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_surfacing_tolerates_a_missing_log_dir(tmp_path):
    """Diagnostics must never break a run."""
    from midas_pipeline.stages.refinement import _surface_unrefined_positions
    _surface_unrefined_positions(tmp_path / "does-not-exist")
