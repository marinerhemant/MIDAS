"""Stage: grain_geometry (FF-only, optional, OFF by default).

Closes the ``tx=0`` gap. Powder/calibrant calibration is structurally blind to
``tx`` (in-plane detector rotation about the beam) — symmetric rings are
invariant under it — so the first FF reconstruction runs with ``tx=0``. This
stage measures the absolute rotation-axis ``Wedge`` (and the OmegaStep
scale) from Friedel pairs (``midas_joint_ff_calibrate.friedel_wedge``), then
refines ``tx`` from the recovered grain spots with that Wedge pinned
(``midas_joint_ff_calibrate.grain_refine.refine_geometry_from_grains``), and
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
    refine = tuple(cfg.refine_params)
    tx_input = paramstest
    fw_metrics = {}
    if getattr(cfg, "friedel_wedge", True):
        # Wedge from Friedel pairs (first-order, omega channel); grain-tx then
        # refines tx with that Wedge pinned in its input file.
        from midas_joint_ff_calibrate.friedel_wedge import friedel_wedge
        wedge_path = layer_dir / "paramstest_wedge.txt"
        fw = friedel_wedge(paramstest, layer_dir, out_paramstest=wedge_path,
                           write_omega_step=bool(getattr(cfg, "write_omega_step", False)))
        LOG.info("grain_geometry: Friedel Wedge %+.5f +- %.5f deg (%d pairs, %d grains); "
                 "OmegaStep %.6f -> %.6f", fw.wedge_deg, fw.wedge_se_deg, fw.n_pairs,
                 fw.n_grains, fw.omega_step_in, fw.omega_step_fit)
        for msg in fw.warnings:
            LOG.warning("grain_geometry: %s", msg)
        fw_metrics = {"friedel_wedge_deg": fw.wedge_deg, "friedel_wedge_se_deg": fw.wedge_se_deg,
                      "friedel_n_pairs": fw.n_pairs, "omega_step_fit": fw.omega_step_fit,
                      "omega_scale_eps": fw.omega_scale_eps}
        tx_input = wedge_path
        if "Wedge" in refine:
            LOG.info("grain_geometry: Wedge measured from Friedel pairs; dropping it from "
                     "the grain-tx refine set.")
            refine = tuple(r for r in refine if r != "Wedge")
    LOG.info("grain_geometry: refining %s from grains (kind=%s, max_grains=%d) …",
             refine, cfg.kind, cfg.max_grains)
    res = refine_geometry_from_grains(
        paramstest=tx_input, layer_dir=layer_dir,
        refine_params=refine, kind=cfg.kind,
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
        metrics={**fw_metrics, "refined": res.refined, "cost_init": res.cost_init,
                 "cost_final": res.cost_final, "n_grains": res.n_grains,
                 "n_spots_matched": res.n_spots_matched, "rc": res.rc},
    )
