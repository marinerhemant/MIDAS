"""Stage: seeding (PF only).

Dispatches on :attr:`midas_pipeline.config.SeedingConfig.mode`:

- ``unseeded``  → no-op (skipped stub). The indexer runs its full
                  orientation grid per voxel.
- ``ff``        → expect a user-supplied ``GrainsFile`` (path in
                  ``SeedingConfig.grains_file``); convert it to
                  ``UniqueOrientations.csv`` via
                  :func:`midas_pipeline.seeding.handoff.grains_csv_to_unique_orientations`.
- ``merged-ff`` → run the four-stage merged-FF seeding orchestrator:
                  align (per-scan rotation-axis correction)
                  → merge_all (merge all scans into one FF-style spot file)
                  → ff_index (run midas-index FF on the merged file)
                  → handoff (Grains.csv → UniqueOrientations.csv).

``ff_index`` is wired and shells out to ``python -m midas_index`` on
the merged spot file. ``align.method`` other than ``"none"`` is the
only sub-stage that still raises ``NotImplementedError`` — those
methods (``ring-center``, ``cross-correlation``) require
midas-calibrate-v2 integration that hasn't landed yet. Use
``method="none"`` for drift-free synthetic data or when alignment
has been done upstream. Soft skip otherwise.
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def _read_space_group(layer_dir: Path) -> int:
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return 225
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "SpaceGroup":
            try:
                return int(toks[1])
            except ValueError:
                continue
    return 225


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    if ctx.is_ff or cfg.seeding.mode == "unseeded":
        return stub_run("seeding", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    seed_csv = layer_dir / "UniqueOrientations.csv"
    space_group = _read_space_group(layer_dir)

    from ..seeding import grains_csv_to_unique_orientations

    if cfg.seeding.mode == "ff":
        # User-supplied GrainsFile path → handoff.
        if not cfg.seeding.grains_file:
            LOG.info(
                "seeding(ff): no grains_file configured; "
                "indexer will fall back to unseeded behavior."
            )
            return stub_run("seeding", ctx)
        grains_path = Path(cfg.seeding.grains_file)
        if not grains_path.exists():
            raise FileNotFoundError(
                f"seeding(ff): grains_file {grains_path} not on disk."
            )
        aug = None
        if cfg.seeding.augment_ff_layer:
            # The merged-FF seed is what survives process-grains' clustering;
            # on ESRF ma5608 that dropped real crystals (incl. twin domains).
            # Add back the ones that beat an omega-shuffled null run.
            from ..seeding.augment import augment_from_ff_layer
            aug = augment_from_ff_layer(
                cfg.seeding.augment_ff_layer, layer_dir / "seed_augment",
                sg=space_group, n_null=cfg.seeding.augment_null_runs,
                n_cpus=int(getattr(cfg, "n_cpus", 8) or 8))
            LOG.info("seeding(ff): augmented seed %s (+%d orientations, gate %.4f)",
                     aug["out_csv"], aug["added"], aug["gate"])
            grains_path = Path(aug["out_csv"])
        LOG.info("seeding(ff): handoff %s → %s", grains_path, seed_csv)
        n_seeds = grains_csv_to_unique_orientations(
            grains_path, seed_csv, space_group=space_group,
        )
        finished = time.time()
        metrics = {"mode": "ff", "n_seed_grains": n_seeds, "space_group": space_group}
        if aug is not None:
            metrics.update(augment_added=aug["added"], augment_gate=aug["gate"],
                           augment_null_runs=len(aug["null_distinct_far"]))
        return StageResult(
            stage_name="seeding",
            started_at=started, finished_at=finished, duration_s=finished - started,
            outputs={str(seed_csv): ""},
            metrics=metrics,
        )

    if cfg.seeding.mode == "merged-ff":
        from ..seeding.align import align_per_scan
        from ..seeding.merge_all import merge_all_scans
        from ..seeding.ff_index import run_ff_indexer_on_merged

        n_scans = int(cfg.scan.n_scans)
        ref_scan = cfg.merged_ref_scan()
        LOG.info(
            "seeding(merged-ff): align=%s ref_scan=%d n_scans=%d",
            cfg.seeding.merged_align_method, ref_scan, n_scans,
        )
        # Stage A — alignment. method='none' is the production fallback
        # for drift-free synthetic data; ring-center / cross-correlation
        # raise NotImplementedError until midas-calibrate-v2 wires up.
        diags = align_per_scan(
            layer_dir=layer_dir,
            n_scans=n_scans,
            method=cfg.seeding.merged_align_method,
            reference_scan=ref_scan,
        )
        LOG.info("seeding(merged-ff): %d alignment diagnostics", len(diags))

        # Stage B — merge all scans into one FF spot file.
        tol_px = cfg.seeding.merged_tol_px if cfg.seeding.merged_tol_px > 0 else 2.0
        tol_ome = cfg.seeding.merged_tol_ome if cfg.seeding.merged_tol_ome > 0 else 2.0
        merge_summary = merge_all_scans(
            layer_dir=layer_dir, n_scans=n_scans,
            tol_px=tol_px, tol_ome=tol_ome,
        )
        LOG.info(
            "seeding(merged-ff): merge_all %d → %d spots",
            merge_summary.n_spots_in, merge_summary.n_spots_out,
        )

        # Stage C — FF indexer on the merged file. Not yet wired.
        ff_result = run_ff_indexer_on_merged(
            layer_dir=layer_dir,
            merged_csv=merge_summary.merged_csv,
            paramstest_in=layer_dir / "paramstest.txt",
            min_n_hkls=cfg.seeding.merged_min_nhkls,
        )

        # Stage D — handoff Grains.csv → UniqueOrientations.csv.
        grains_csv = layer_dir / "Grains.csv"
        if not grains_csv.exists():
            raise FileNotFoundError(
                f"seeding(merged-ff): missing Grains.csv (was Stage C "
                f"ff_index wired up?). Expected at {grains_csv}."
            )
        n_seeds = grains_csv_to_unique_orientations(
            grains_csv, seed_csv, space_group=space_group,
        )
        # The merged-FF Grains.csv is indexed with a 1-row positions.csv, which
        # puts the indexer in FF mode: no beam-position gate, so every scan's
        # spots can support every orientation. It is a SEED list for the
        # per-voxel indexer, not a grain count. On ma5608 it held 568 rows
        # against 204 find_grains grains; on the reference campaign its
        # omega-shuffle null found MORE grains than the real data.
        LOG.warning(
            "seeding(merged-ff): %d seed orientations written to %s. This is a "
            "seed list indexed with the beam gate OFF (FF mode), not a grain "
            "count; quote find_grains' Output/UniqueOrientations.csv instead.",
            n_seeds, seed_csv,
        )
        finished = time.time()
        return StageResult(
            stage_name="seeding",
            started_at=started, finished_at=finished, duration_s=finished - started,
            outputs={str(seed_csv): "",
                     str(merge_summary.merged_csv): ""},
            metrics={"mode": "merged-ff",
                     "n_seed_grains": n_seeds,
                     "n_merged_spots": merge_summary.n_spots_out,
                     "align_method": cfg.seeding.merged_align_method,
                     "space_group": space_group},
        )

    raise ValueError(f"seeding: unknown mode {cfg.seeding.mode!r}")
