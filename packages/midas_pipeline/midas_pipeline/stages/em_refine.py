"""Stage: em_refine (PF only, optional).

Runs the EM spot-ownership refinement (port of
``FF_HEDM/workflows/em_pf_integration.py``). Gated on
``cfg.em.enable=True``. The EM kernel itself lives in
``fwd_sim/em_spot_ownership.py`` and is imported lazily — keeping
this stage opt-in.
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import EMRefineResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    if ctx.is_ff or not cfg.em.enable:
        return stub_run("em_refine", ctx)
    from ._voxel_grid import require_positions_grid
    require_positions_grid(ctx.layer_dir, "em_refine")

    started = time.time()
    layer_dir = Path(ctx.layer_dir)

    LOG.info("em_refine(PF): n_iter=%d sigma_init=%.2f sigma_min=%.2f",
             cfg.em.iter, cfg.em.sigma_init, cfg.em.sigma_min)

    from ..em_refine import run_em_spot_ownership
    refined_sinos_paths = run_em_spot_ownership(
        topdir=str(layer_dir),
        n_scans=int(cfg.scan.n_scans),
        n_iter=cfg.em.iter,
        sigma_init=cfg.em.sigma_init,
        sigma_min=cfg.em.sigma_min,
        sigma_decay=cfg.em.sigma_decay,
        refine_orientations=cfg.em.refine_orientations,
        n_opt_steps=cfg.em.opt_steps,
        lr=cfg.em.lr,
    )
    out_map = {str(p): "" for p in (refined_sinos_paths or [])} if refined_sinos_paths else {}
    finished = time.time()
    return EMRefineResult(
        stage_name="em_refine",
        started_at=started, finished_at=finished, duration_s=finished - started,
        refined_sinos_paths={k: k for k in out_map} if out_map else {},
        outputs=out_map,
        metrics={"n_iter": cfg.em.iter, "sigma_init": cfg.em.sigma_init,
                 "sigma_min": cfg.em.sigma_min,
                 "refine_orientations": cfg.em.refine_orientations},
    )
