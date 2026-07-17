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
from .detector import DetectorConfig
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


# N4: per-stage upstream requirements (the stage whose OUTPUT files each
# stage consumes). Used to validate ``--only`` allowlists — both the SOH
# and Ni campaigns independently produced broken recons from an ``--only``
# set that omitted merge_scans/seeding/refinement (each omitted stage
# soft-skips, the run "succeeds"). Mode-dependent entries use a dict.
_STAGE_DEPS: dict = {
    "peakfit":         ("zip_convert", "hkl"),
    "merge_overlaps":  ("peakfit",),
    "calc_radius":     ("peakfit",),
    "transforms":      ("peakfit", "calc_radius", "hkl"),
    "cross_det_merge": ("transforms",),
    "global_powder":   ("calc_radius",),
    "merge_scans":     ("transforms",),
    "seeding":         ("merge_scans",),
    "binning":         {"ff": ("transforms",), "pf": ("merge_scans", "seeding")},
    "indexing":        ("binning",),
    "refinement":      ("indexing", "binning"),
    "find_grains":     ("refinement",),
    "voxel_cleanup":   ("find_grains",),
    "sinogen":         ("find_grains",),
    "reconstruct":     ("sinogen",),
    "fuse":            ("reconstruct",),
    "potts":           ("fuse",),
    "em_refine":       ("fuse",),
    "process_grains":  ("refinement",),
    "grain_geometry":  ("process_grains",),
}


def stage_deps_for(name: str, scan_mode: ScanMode) -> Tuple[str, ...]:
    """Upstream stage names ``name`` consumes, restricted to ``scan_mode``."""
    deps = _STAGE_DEPS.get(name, ())
    if isinstance(deps, dict):
        deps = deps.get(scan_mode, ())
    mode_stages = {n for (n, _) in stage_order_for(scan_mode)}
    return tuple(d for d in deps if d in mode_stages)


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
    detectors: Sequence[DetectorConfig] = ()

    def __post_init__(self) -> None:
        if not self.detectors:
            self.detectors = self._discover_detectors()

    def _discover_detectors(self) -> List[DetectorConfig]:
        """Resolve detector geometry list (tolerant — empty on failure).

        Priority:
          1. ``config.detectors_json`` (multi-det JSON spec).
          2. ``DetParams N ...`` rows in the paramstest file (FF multi-det).
          3. Single-detector fallback built from the global ``Lsd`` / ``BC``
             / tilt keys in the paramstest.

        If paramstest is missing/incomplete (common for PF runs whose
        geometry comes from the merged paramstest written later by the
        peakfit/cross_det_merge stages), returns ``[]``. Stages that
        actually need detector geometry will fail when invoked.
        """
        if self.config.detectors_json:
            return DetectorConfig.load_many(self.config.detectors_json)
        try:
            dets = DetectorConfig.load_from_paramstest(
                self.config.params_file,
                zarr_path=self.config.zarr_path,
            )
        except (FileNotFoundError, ValueError):
            dets = []
        if dets:
            if self.config.zarr_path:
                for d in dets:
                    if not d.zarr_path:
                        d.zarr_path = self.config.zarr_path
            return dets
        try:
            return [DetectorConfig.single_from_paramstest(
                self.config.params_file,
                zarr_path=self.config.zarr_path,
            )]
        except (FileNotFoundError, ValueError):
            return []

    def run(self) -> List[LayerResult]:
        configure_logging()
        # Resolve cluster defaults + load parsl config; push back resolved
        # (n_cpus, n_nodes) so downstream stages see the cluster-aware
        # values, not whatever the caller passed (often -1 / 0 for "auto").
        n_cpus, n_nodes = configure_dispatch(self.config.machine.name,
                                             self.config.machine.n_nodes,
                                             self.config.n_cpus)
        self.config.n_cpus = n_cpus
        self.config.machine.n_nodes = n_nodes
        results: List[LayerResult] = []
        for layer_nr in self.config.layer_selection.layers():
            results.append(self._run_layer(layer_nr))
        return results

    # --- internals ---

    def _make_context(self, layer_nr: int) -> StageContext:
        from .stages._base import resolve_layer_dir
        layer_dir = resolve_layer_dir(self.config.result_path, layer_nr)
        layer_dir.mkdir(parents=True, exist_ok=True)
        # P0-2: materialize positions.csv at layer setup for PF runs.
        # Every early PF stage (zip_convert/hkl/peakfit/transforms)
        # iterates scans from this file; before this, a missing file made
        # them all skip and the run exited 0 having done nothing. File
        # order = acquisition order (sign per --scan-step); never
        # overwrite a pre-seeded file. FF is untouched — the transforms
        # dump writes its own 1-row sentinel (midas_transforms
        # pipeline.py:214-217).
        if self.config.scan.is_pf:
            lines = "".join(
                f"{y:.6f}\n" for y in self.config.scan.scan_positions
            )
            layer_pcsv = layer_dir / "positions.csv"
            if not layer_pcsv.exists():
                layer_pcsv.write_text(lines)
                LOG.info("materialized %s (%d scans, acquisition order)",
                         layer_pcsv, self.config.scan.n_scans)
            root_pcsv = Path(self.config.result_path) / "positions.csv"
            if not root_pcsv.exists():
                root_pcsv.write_text(lines)
        log_dir = layer_dir / "midas_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(
            config=self.config,
            layer_nr=layer_nr,
            layer_dir=layer_dir,
            log_dir=log_dir,
            detectors=list(self.detectors),
        )

    def _run_layer(self, layer_nr: int) -> LayerResult:
        ctx = self._make_context(layer_nr)
        store = ProvenanceStore(ctx.layer_dir)
        layer_result = LayerResult(layer_nr=layer_nr,
                                   layer_dir=str(ctx.layer_dir))

        # Per-layer FF seed-grain resolution (NF→FF handoff or fixed file).
        # Idempotent: writes GrainsFile + MinNrSpots 1 into the params file
        # so downstream midas-fit-setup picks up the seed.
        if self.config.is_ff and (
            self.config.nf_result_dir or self.config.grains_file
        ):
            from pathlib import Path as _P
            from .ff_seeding import patch_params_with_grains, resolve_grains_file_for_layer
            seed = resolve_grains_file_for_layer(
                layer_nr=layer_nr,
                grains_file=self.config.grains_file,
                nf_result_dir=self.config.nf_result_dir,
            )
            if seed:
                patch_params_with_grains(_P(self.config.params_file), seed)

        scan_mode = self.config.scan.scan_mode
        stage_list = stage_order_for(scan_mode)

        LOG.info("=" * 60)
        LOG.info("Layer %d — scan_mode=%s, %d stages",
                 layer_nr, scan_mode, len(stage_list))
        LOG.info("=" * 60)

        # N4: validate an ``--only`` allowlist against the per-mode stage
        # dependency graph. A dependency is satisfied when it is in the
        # only-set, or already complete in the provenance store (resume
        # workflows), or produced by a previous partial run. Anything else
        # is a hard error — an omitted dependency makes each downstream
        # stage soft-skip and the run "succeed" with a broken recon (hit
        # independently by two campaigns from the same handoff doc).
        if self.config.only_stages:
            missing: dict = {}
            only = set(self.config.only_stages)
            for name, _fn in stage_list:
                if name not in only or name in self.config.skip_stages:
                    continue
                unmet = [
                    d for d in stage_deps_for(name, scan_mode)
                    if d not in only
                    and d not in self.config.skip_stages
                    and not store.is_complete(d)
                ]
                if unmet:
                    missing[name] = unmet
            if missing:
                detail = "; ".join(
                    f"{k} needs {', '.join(v)}" for k, v in missing.items()
                )
                raise RuntimeError(
                    f"--only allowlist omits required upstream stages "
                    f"({detail}) that are neither selected nor already "
                    "complete for this layer. Each omitted stage would "
                    "soft-skip and the run would 'succeed' with a broken "
                    "recon. Prefer --skip on the unwanted tail stages, or "
                    "add the missing stages to --only."
                )

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
