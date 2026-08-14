"""Grain refinement via ``midas-fit-grain``."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from midas_fit_grain.losses import (
    MULTIDET_LOSS, PANEL_DEPENDENT_LOSSES, resolve as resolve_loss,
)

from ._base import StageContext, env_for_index_refine, run_subprocess
from .._logging import stage_timer
from ..results import RefineResult


def run(ctx: StageContext) -> RefineResult:
    started = time.time()

    paramstest = ctx.layer_dir / "paramstest.txt"
    spots_to_index = ctx.layer_dir / "SpotsToIndex.csv"
    n_seeds = sum(1 for _ in spots_to_index.open() if _.strip())

    output_dir = ctx.layer_dir / "Output"
    results_dir = ctx.layer_dir / "Results"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    from .._logging import LOG

    # Retired names are substituted here rather than passed through to die in
    # the refiner's argparse.
    loss, note = resolve_loss(ctx.config.refine_loss)
    if note:
        LOG.warning("  refinement: %s", note)

    # Multi-detector pinwheel: a PIXEL-BASED loss is per-panel (each panel has
    # its own beam centre + Lsd), so one global residual mixes incompatible
    # frames when the refiner sees spots from several panels. Switch to
    # ``angular`` — (2θ, η, ω), geometry-independent — when the merged
    # paramstest carries DetParams blocks.
    #
    # This used to test ``loss == "pixel"`` only. ``full3d`` is pixel-based too
    # (y_pixel, z_pixel, Δω·r_px — residuals.py:152-168) and is now the
    # default, so the guard has to key on the SET of panel-dependent losses or
    # it silently stops firing exactly when it starts mattering.
    is_multi_det = "\nDetParams " in ("\n" + paramstest.read_text())
    if is_multi_det and loss in PANEL_DEPENDENT_LOSSES:
        LOG.info("  multi-detector run → switching refine loss %r → %r "
                 "(pixel-based losses are per-panel)", loss, MULTIDET_LOSS)
        loss = MULTIDET_LOSS

    with stage_timer("refinement"):
        cmd = [
            sys.executable, "-m", "midas_fit_grain",
            str(paramstest),
            "0",                                   # block_nr
            "1",                                   # n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
            "--solver", ctx.config.refine_solver,
            "--loss", loss,
        ]
        if ctx.config.refine_mode:
            cmd += ["--mode", ctx.config.refine_mode]
        run_subprocess(
            cmd,
            cwd=ctx.layer_dir,
            stdout_path=ctx.log_dir / "refinement_out.csv",
            stderr_path=ctx.log_dir / "refinement_err.csv",
            env=env_for_index_refine(ctx.config),
        )

    finished = time.time()
    # midas-fit-grain honours OutputFolder + ResultFolder from paramstest.txt
    # (set by the transforms stage), so outputs already land in Output/ +
    # Results/ — no relocation needed.
    orient_pos_fit = results_dir / "OrientPosFit.bin"
    n_grains_refined = 0
    if orient_pos_fit.exists():
        # OrientPosFit.bin: per refined seed, several doubles.
        # Just record file size as a proxy for "non-zero output produced."
        n_grains_refined = orient_pos_fit.stat().st_size // 8

    return RefineResult(
        stage_name="refinement",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={
            str(orient_pos_fit): "",
            str(output_dir / "FitBest.bin"): "",
            str(results_dir / "Key.bin"): "",
            str(results_dir / "ProcessKey.bin"): "",
        },
        orient_pos_fit_bin=str(orient_pos_fit),
        n_grains_refined=n_grains_refined,
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [
        ctx.layer_dir / "Results" / "OrientPosFit.bin",
        ctx.layer_dir / "Output" / "FitBest.bin",
    ]
