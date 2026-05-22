"""Stage: grain_geometry (FF-only, optional, OFF by default).

Closes the ``tx=0`` gap. Powder/calibrant calibration is structurally blind to
``tx`` (in-plane detector rotation about the beam) — symmetric rings are
invariant under it — so the first FF reconstruction runs with ``tx=0``. This
stage refines ``tx`` (and optionally ``Wedge``) from the recovered grain spots
via ``midas_joint_ff_calibrate.grain_refine.refine_geometry_from_grains`` and
writes a corrected paramstest. The user then re-runs the pipeline from the
``transforms`` stage with that paramstest (a true second detector-calibration
pass). See ``project_ff_tx_grain_calibration``.

No-op (skipped StageResult) unless ``config.grain_geometry.run`` is True.
"""
from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config.grain_geometry
    if not cfg.run:
        return stub_run("grain_geometry", ctx)
    if not ctx.is_ff:
        LOG.info("grain_geometry: FF-only stage, scan_mode=%s → skip.", ctx.scan_mode)
        return stub_run("grain_geometry", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    paramstest = layer_dir / "paramstest.txt"
    grains = layer_dir / "Grains.csv"
    spots = layer_dir / "SpotMatrix.csv"
    missing = [p.name for p in (paramstest, grains, spots) if not p.exists()]
    if missing:
        LOG.info("grain_geometry: missing %s in %s → skip.", missing, layer_dir)
        return stub_run("grain_geometry", ctx)

    from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains

    out_path = layer_dir / cfg.out_name
    device = "cpu" if str(ctx.config.device).lower() not in ("cuda", "mps") else str(ctx.config.device)
    LOG.info("grain_geometry: refining %s from grains (kind=%s, max_grains=%d) …",
             tuple(cfg.refine_params), cfg.kind, cfg.max_grains)
    res = refine_geometry_from_grains(
        paramstest=paramstest, layer_dir=layer_dir,
        refine_params=tuple(cfg.refine_params), kind=cfg.kind,
        max_grains=cfg.max_grains, max_iter=cfg.max_iter,
        two_theta_max_deg=cfg.two_theta_max_deg,
        refine_grain_strain=cfg.refine_strain, with_powder=False,
        out_paramstest=out_path, device=device,
    )
    LOG.info("grain_geometry: %s  cost %.3e → %.3e  (%d grains, %d spots) → %s",
             {k: round(v, 6) for k, v in res.refined.items()},
             res.cost_init, res.cost_final, res.n_grains, res.n_spots_matched,
             out_path.name)

    finished = time.time()
    return StageResult(
        stage_name="grain_geometry",
        started_at=started, finished_at=finished, duration_s=finished - started,
        inputs={"paramstest": str(paramstest), "grains_csv": str(grains),
                "spot_matrix": str(spots)},
        outputs={"paramstest_refined": str(out_path)},
        metrics={"refined": res.refined, "cost_init": res.cost_init,
                 "cost_final": res.cost_final, "n_grains": res.n_grains,
                 "n_spots_matched": res.n_spots_matched, "rc": res.rc},
    )
