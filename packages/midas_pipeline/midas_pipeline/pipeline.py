"""Top-level Pipeline class.

The pipeline iterates the selected layers, runs each stage in
``STAGE_ORDER`` for the active scan-mode, records provenance, and rolls
results up into ``LayerResult``.

Stage order is mode-dependent:

- ``scan_mode="ff"`` runs the FF path (peakfit → … → process_grains →
  consolidation), with PF-only stages skipped entirely.
- ``scan_mode="pf"`` runs the PF path (peakfit → … → merge_scans → …
  → find_grains → sinogen → reconstruct → fuse → potts → em_refine →
  consolidation), with FF-only ``process_grains`` skipped.

A stage is "FF-only" if it only makes sense for single-scan runs (e.g.,
``process_grains`` is the FF grain consolidation; PF uses
``consolidation`` directly). A stage is "PF-only" if it's meaningless
without scanning (e.g., ``merge_scans``, ``find_grains``, ``sinogen``,
``reconstruct``, ``fuse``, ``potts``, ``em_refine``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from . import stages
from ._logging import LOG, configure_logging, stage_timer
from .config import PipelineConfig, ScanMode
from .dispatch import configure_dispatch
from .provenance import ProvenanceStore
from .results import LayerResult, StageResult
from .stages._base import StageContext


# ---------------------------------------------------------------------------
# Stage registry — (stage_name, stage_module.run, applies_in)
# ---------------------------------------------------------------------------

StageFn = Callable[[StageContext], StageResult]
StageEntry = Tuple[str, StageFn, Tuple[ScanMode, ...]]


_BOTH: Tuple[ScanMode, ...] = ("ff", "pf")
_FF_ONLY: Tuple[ScanMode, ...] = ("ff",)
_PF_ONLY: Tuple[ScanMode, ...] = ("pf",)


_STAGES: List[StageEntry] = [
    # Shared front half
    ("zip_convert",        stages.zip_convert.run,        _BOTH),
    ("hkl",                stages.hkl.run,                _BOTH),
    ("peakfit",            stages.peakfit.run,            _BOTH),
    ("merge_overlaps",     stages.merge_overlaps.run,     _BOTH),
    ("calc_radius",        stages.calc_radius.run,        _BOTH),
    ("transforms",         stages.transforms.run,         _BOTH),
    ("cross_det_merge",    stages.cross_det_merge.run,    _BOTH),
    ("global_powder",      stages.global_powder.run,      _BOTH),

    # PF-only: per-scan merge before binning
    ("merge_scans",        stages.merge_scans.run,        _PF_ONLY),

    # PF-only seeding (unseeded | ff | merged-ff). Runs AFTER merge_scans
    # so merged-FF seeding can consume the per-scan spot lists, and
    # BEFORE binning so the indexer's seeded path can read the resulting
    # UniqueOrientations.csv.
    ("seeding",            stages.seeding.run,            _PF_ONLY),

    # Shared: binning → indexing → refinement
    ("binning",            stages.binning.run,            _BOTH),
    ("indexing",           stages.indexing.run,           _BOTH),
    ("refinement",         stages.refinement.run,         _BOTH),

    # PF-only tail
    ("find_grains",        stages.find_grains_stage.run,  _PF_ONLY),
    # Optional missing-spot directionality voxel cleanup (OFF by default).
    # Runs after find_grains so the cleaned voxel_grid.csv feeds the V-map path.
    ("voxel_cleanup",      stages.voxel_cleanup.run,      _PF_ONLY),
    ("sinogen",            stages.sinogen.run,            _PF_ONLY),
    ("reconstruct",        stages.reconstruct.run,        _PF_ONLY),
    ("fuse",               stages.fuse.run,               _PF_ONLY),
    ("potts",              stages.potts.run,              _PF_ONLY),
    ("em_refine",          stages.em_refine.run,          _PF_ONLY),

    # FF-only grain consolidation; PF uses `consolidation` directly.
    ("process_grains",     stages.process_grains.run,     _FF_ONLY),

    # FF-only optional: grain-based tx/Wedge geometry refinement (powder is
    # blind to tx). No-op unless grain_geometry.run. Runs after process_grains
    # (needs Grains.csv); writes a corrected paramstest for a re-run.
    ("grain_geometry",     stages.grain_geometry.run,     _FF_ONLY),

    # Shared finalizer
    ("consolidation",      stages.consolidation.run,      _BOTH),

    # P8: V-map joint refinement (PF + FF compact-grain).  Both stages
    # are clean no-ops when ``PipelineConfig.vmap.run`` is False.
    ("calc_radius_v",      stages.calc_radius_v.run,      _BOTH),
    ("refine_vmap",        stages.refine_vmap.run,        _BOTH),
]


def stage_order_for(scan_mode: ScanMode) -> List[Tuple[str, StageFn]]:
    """Return ``[(stage_name, run_fn), ...]`` for the given scan mode."""
    if scan_mode not in _BOTH:
        raise ValueError(f"unknown scan_mode={scan_mode!r}")
    return [(name, fn) for (name, fn, modes) in _STAGES if scan_mode in modes]


def all_stage_names() -> List[str]:
    """Every stage name across both modes (used by CLI validation)."""
    return [name for (name, _, _) in _STAGES]


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    """Top-level orchestrator.

    Usage::

        cfg = PipelineConfig(result_dir=..., params_file=..., scan=ScanGeometry.ff())
        pipe = Pipeline(cfg)
        results = pipe.run()           # list[LayerResult]
    """

    config: PipelineConfig

    def run(self) -> List[LayerResult]:
        configure_logging()
        configure_dispatch(self.config.machine.name,
                           self.config.machine.n_nodes,
                           self.config.n_cpus)
        results: List[LayerResult] = []
        for layer_nr in self.config.layer_selection.layers():
            results.append(self._run_layer(layer_nr))
        return results

    # --- internals ---

    def _make_context(self, layer_nr: int) -> StageContext:
        from .stages._base import resolve_layer_dir
        layer_dir = resolve_layer_dir(self.config.result_path, layer_nr)
        layer_dir.mkdir(parents=True, exist_ok=True)
        log_dir = layer_dir / "midas_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(
            config=self.config,
            layer_nr=layer_nr,
            layer_dir=layer_dir,
            log_dir=log_dir,
        )

    def _run_layer(self, layer_nr: int) -> LayerResult:
        ctx = self._make_context(layer_nr)
        store = ProvenanceStore(ctx.layer_dir)
        layer_result = LayerResult(layer_nr=layer_nr,
                                   layer_dir=str(ctx.layer_dir))

        scan_mode = self.config.scan.scan_mode
        stage_list = stage_order_for(scan_mode)

        LOG.info("=" * 60)
        LOG.info("Layer %d — scan_mode=%s, %d stages",
                 layer_nr, scan_mode, len(stage_list))
        LOG.info("=" * 60)

        # Resume-from handling: skip stages strictly before resume_from_stage
        resume_after_idx = -1
        if self.config.resume == "from" and self.config.resume_from_stage:
            for i, (name, _) in enumerate(stage_list):
                if name == self.config.resume_from_stage:
                    resume_after_idx = i - 1
                    break

        for idx, (name, fn) in enumerate(stage_list):
            if name in self.config.skip_stages:
                LOG.info("  skipping stage '%s' (--skip)", name)
                continue
            if self.config.only_stages and name not in self.config.only_stages:
                continue

            # Auto-resume: skip stages already complete + outputs match hash
            if self.config.resume == "auto" and store.is_complete(name):
                LOG.info("  resume: '%s' already complete, skipping", name)
                rec = store.read(name) or {}
                result = StageResult(
                    stage_name=name,
                    started_at=rec.get("started_at", 0.0),
                    finished_at=rec.get("finished_at", 0.0),
                    duration_s=rec.get("duration_s", 0.0),
                    inputs=rec.get("inputs", {}),
                    outputs=rec.get("outputs", {}),
                    metrics=rec.get("metrics", {}),
                    skipped=True,
                )
                _attach(layer_result, name, result)
                continue

            # Resume-from: skip stages before the target
            if self.config.resume == "from" and idx <= resume_after_idx:
                continue

            with stage_timer(name) as timing:
                result = fn(ctx)
            # stage_timer fills timing["finished_at"] + timing["duration_s"] on exit.
            if result.started_at == 0.0:
                result.started_at = timing["started_at"]
            if result.finished_at == 0.0:
                result.finished_at = timing["finished_at"]
            if result.duration_s == 0.0:
                result.duration_s = timing["duration_s"]

            store.record(
                name,
                status="complete" if not result.skipped else "skipped",
                started_at=result.started_at,
                finished_at=result.finished_at,
                duration_s=result.duration_s,
                inputs=result.inputs,
                outputs=result.outputs,
                metrics=result.metrics,
            )
            _attach(layer_result, name, result)

        return layer_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LAYER_RESULT_FIELD_BY_STAGE = {
    "zip_convert":     "zip_convert",
    "hkl":             "hkl",
    "peakfit":         "peakfit",
    "merge_overlaps":  "merge_overlaps",
    "calc_radius":     "calc_radius",
    "transforms":      "transforms",
    "cross_det_merge": "cross_det_merge",
    "global_powder":   "global_powder",
    "binning":         "binning",
    "indexing":        "indexing",
    "refinement":      "refinement",
    "process_grains":  "process_grains",
    "merge_scans":     "merge_scans",
    "find_grains":     "find_grains",
    "voxel_cleanup":   "voxel_cleanup",
    "sinogen":         "sinogen",
    "reconstruct":     "reconstruct",
    "fuse":            "fuse",
    "potts":           "potts",
    "em_refine":       "em_refine",
    "consolidation":   "consolidation",
    "calc_radius_v":   "calc_radius_v",
    "refine_vmap":     "refine_vmap",
}


def _attach(layer_result: LayerResult, stage_name: str, result: StageResult) -> None:
    field = _LAYER_RESULT_FIELD_BY_STAGE.get(stage_name)
    if field is None:
        return
    setattr(layer_result, field, result)
