"""Per-stage result dataclasses for midas-pipeline.

Each stage's ``run(ctx)`` returns one of these. The pipeline persists
them into ``midas_state.h5`` for resume + inspection.

Extends the midas-ff-pipeline result set with PF-specific stages
(``merge_scans``, ``find_grains``, ``sinogen``, ``reconstruct``,
``fuse``, ``potts``, ``em_refine``, ``consolidation_pf``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Common fields across stages."""

    stage_name: str
    started_at: float
    finished_at: float
    duration_s: float
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False


# ---------------------------------------------------------------------------
# Shared stage results (used by both FF and PF modes)
# ---------------------------------------------------------------------------


@dataclass
class HKLResult(StageResult):
    hkls_csv: str = ""


@dataclass
class PerDetPeakFitResult:
    det_id: int
    n_peaks: int
    zip_path: str


@dataclass
class PeakFitResult(StageResult):
    per_detector: List[PerDetPeakFitResult] = field(default_factory=list)


@dataclass
class MergeOverlapsResult(StageResult):
    n_peaks_after_merge: int = 0


@dataclass
class CalcRadiusResult(StageResult):
    n_spots: int = 0


@dataclass
class TransformsResult(StageResult):
    paramstest_path: str = ""


@dataclass
class CrossDetMergeResult(StageResult):
    n_total_spots: int = 0
    n_per_detector: List[int] = field(default_factory=list)
    spots_bin: str = ""
    spots_det_bin: str = ""


@dataclass
class BinningResult(StageResult):
    n_bins: int = 0


@dataclass
class IndexResult(StageResult):
    index_best_bin: str = ""               # FF-flavor path (single-solution)
    index_best_all_bin: str = ""           # PF-flavor consolidated path
    n_seeds_attempted: int = 0
    n_seeds_indexed: int = 0
    n_voxels_indexed: int = 0              # PF mode


@dataclass
class RefineResult(StageResult):
    orient_pos_fit_bin: str = ""           # FF-flavor
    results_dir: str = ""                  # PF: dir of per-voxel Result_*.csv
    n_grains_refined: int = 0
    n_voxels_refined: int = 0              # PF mode


@dataclass
class ProcessGrainsResult(StageResult):
    """FF-mode only — calls ``midas-process-grains``."""

    grains_csv: str = ""
    n_grains: int = 0


# ---------------------------------------------------------------------------
# PF-only stage results
# ---------------------------------------------------------------------------


@dataclass
class MergeScansResult(StageResult):
    merged_csv: str = ""
    n_spots_in: int = 0
    n_spots_out: int = 0


@dataclass
class FindGrainsResult(StageResult):
    unique_orientations_csv: str = ""
    unique_index_single_key_bin: str = ""
    spots_to_index_csv: str = ""           # multiple-mode output
    n_unique_grains: int = 0


@dataclass
class SinogenResult(StageResult):
    sinos_paths: Dict[str, str] = field(default_factory=dict)   # variant → path
    omegas_path: str = ""
    nr_hkls_path: str = ""
    n_grains: int = 0
    max_n_hkls: int = 0


@dataclass
class ReconResult(StageResult):
    method: str = ""
    per_grain_tifs: List[str] = field(default_factory=list)
    full_recon_max_project_grid_tif: str = ""


@dataclass
class FuseResult(StageResult):
    posterior_path: str = ""


@dataclass
class PottsResult(StageResult):
    grain_id_map_path: str = ""
    n_flips: int = 0
    n_iter: int = 0


@dataclass
class EMRefineResult(StageResult):
    refined_sinos_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class CalcRadiusVResult(StageResult):
    """PF-mode calc_radius (V-map foundation, P8).

    Writes a per-spot V_rel CSV and a per-ring theoretical-intensity CSV
    used by :class:`RefineVmapResult`.  ``n_spots`` is the number of
    spots that received a finite V_rel.
    """

    radius_csv: str = ""               # per-spot Radius_V.csv
    theory_csv: str = ""               # per-ring I_theory.csv
    n_spots: int = 0
    n_rings: int = 0


@dataclass
class RefineVmapResult(StageResult):
    """PF-mode V-map joint refinement (P8).

    Writes per-voxel V (.h5 dataset) plus per-ring K and a loss history.
    Loss is log-space residual ``log I_obs - log I_pred`` per
    :func:`midas_transforms.radius.refine_vmap_joint`.
    """

    v_map_h5: str = ""
    k_ring_csv: str = ""
    loss_history_csv: str = ""
    n_voxels: int = 0
    n_rings: int = 0
    n_iterations: int = 0
    final_loss: float = 0.0
    converged: bool = False


@dataclass
class ConsolidationResult(StageResult):
    """PF mode: pure-Python port of pf_MIDAS.py:2429–2519.

    FF mode shares the dataclass; ``microstr_full_csv`` is empty there.
    """

    microstr_full_csv: str = ""
    microstructure_hdf: str = ""
    full_recon_max_project_grid_tif: str = ""


# ---------------------------------------------------------------------------
# LayerResult / RunResult — all-stage roll-ups
# ---------------------------------------------------------------------------


@dataclass
class LayerResult:
    """All-stage roll-up for one layer (FF) or one scan-mode run (PF)."""

    layer_nr: int
    layer_dir: str

    # Shared
    zip_convert: Optional[StageResult] = None
    hkl: Optional[HKLResult] = None
    peakfit: Optional[PeakFitResult] = None
    merge_overlaps: Optional[MergeOverlapsResult] = None
    calc_radius: Optional[CalcRadiusResult] = None
    transforms: Optional[TransformsResult] = None
    cross_det_merge: Optional[CrossDetMergeResult] = None
    global_powder: Optional[StageResult] = None
    binning: Optional[BinningResult] = None
    indexing: Optional[IndexResult] = None
    refinement: Optional[RefineResult] = None

    # FF only
    process_grains: Optional[ProcessGrainsResult] = None

    # PF only
    merge_scans: Optional[MergeScansResult] = None
    find_grains: Optional[FindGrainsResult] = None
    sinogen: Optional[SinogenResult] = None
    reconstruct: Optional[ReconResult] = None
    fuse: Optional[FuseResult] = None
    potts: Optional[PottsResult] = None
    em_refine: Optional[EMRefineResult] = None
    # PF V-map (P8 of the V-map plan)
    calc_radius_v: Optional["CalcRadiusVResult"] = None
    refine_vmap: Optional["RefineVmapResult"] = None

    # Shared end
    consolidation: Optional[ConsolidationResult] = None

    @property
    def n_grains(self) -> int:
        if self.process_grains:
            return self.process_grains.n_grains
        if self.find_grains:
            return self.find_grains.n_unique_grains
        return 0

    def all_stage_results(self) -> List[StageResult]:
        out: list[StageResult] = []
        for f in (
            self.zip_convert, self.hkl, self.peakfit, self.merge_overlaps,
            self.calc_radius, self.transforms, self.cross_det_merge,
            self.global_powder, self.binning,
            self.merge_scans,
            self.indexing, self.refinement,
            self.find_grains, self.sinogen, self.reconstruct,
            self.fuse, self.potts, self.em_refine,
            self.calc_radius_v, self.refine_vmap,
            self.process_grains, self.consolidation,
        ):
            if f is not None:
                out.append(f)
        return out

    def total_duration_s(self) -> float:
        return sum(r.duration_s for r in self.all_stage_results())
