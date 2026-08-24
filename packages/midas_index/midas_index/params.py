"""IndexerParams dataclass.

Mirrors `struct TParams` from `FF_HEDM/src/IndexerOMP.c:119`. All keys parsed
by `ReadParams` (lines 1281-1547) map to fields here.

Multi-detector additions (per-panel pinwheel) are appended at the bottom:
``DetParams``, ``EtaCoverage_DetN``, ``RingRadii_DetN`` blocks emitted by
midas-ff-pipeline's transforms + cross_det_merge stages. ``DetParams`` is
empty for single-detector runs and the indexer falls back to its global
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class IndexerParams:
    """All paramstest.txt fields consumed by the indexer.

    Aliases (from C): Distance/Lsd, LatticeParameter/LatticeConstant,
    MarginRadius/MarginRad, Completeness/MinMatchesToAcceptFrac,
    StepsizeOrient/StepSizeOrient, MinEta/ExcludePoleAngle. The reader
    folds aliases onto the canonical field below.

    `BigDetSize` is parsed for backward compat but ignored (deprecated).
    """

    # --- Geometry ---
    px: float = 0.0
    Distance: float = 0.0          # also "Lsd"
    Wavelength: float = 0.0
    Rsample: float = 0.0
    Hbeam: float = 0.0

    # --- Crystal ---
    SpaceGroup: int = 0
    LatticeConstant: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )                                # also "LatticeParameter"

    # --- Grid stepping ---
    StepsizePos: float = 0.0
    StepsizeOrient: float = 0.0      # also "StepSizeOrient"

    # --- Margins ---
    MarginOme: float = 0.0
    MarginRad: float = 0.0           # also "MarginRadius"
    MarginRadial: float = 0.0
    MarginEta: float = 0.0
    EtaBinSize: float = 0.0
    OmeBinSize: float = 0.0
    ExcludePoleAngle: float = 0.0    # also "MinEta" — same field in C
    MinMatchesToAcceptFrac: float = 0.0  # also "Completeness"

    # --- Rings / detectors ---
    RingNumbers: list[int] = field(default_factory=list)
    RingsToReject: list[int] = field(default_factory=list)
    # RingRadii is sparse-by-ring-index: RingRadii[ring_nr] -> radius (um).
    # Built from parallel parse-time lists RingNumbers + RingRadiiUser.
    RingRadii: dict[int, float] = field(default_factory=dict)

    # --- Omega / box ranges (parallel lists) ---
    OmegaRanges: list[tuple[float, float]] = field(default_factory=list)
    BoxSizes: list[tuple[float, float, float, float]] = field(default_factory=list)

    # --- Mode A vs B ---
    UseFriedelPairs: int = 0  # 0 | 1 | 2
    isGrainsInput: bool = False
    GrainsFileName: str = ""
    SpotsFileName: str = "Spots.bin"
    IDsFileName: str = "IDsHash.csv"
    OutputFolder: str = "."

    # --- Multi-detector (pinwheel / hydra) extensions ---
    # ``DetParams[det_id] = {lsd, y_bc, z_bc, tx, ty, tz, p_distortion}``.
    # Empty when running single-detector. midas-ff-pipeline's
    # cross_det_merge stage emits one ``DetParams N ...`` row per panel.
    DetParams: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # ``RingRadiiPerDet[det_id][ring_nr] = radius_um``. Per-panel ring
    # radii from each panel's hkls.csv (each panel has its own Lsd, so
    # ring radii differ slightly).
    RingRadiiPerDet: Dict[int, Dict[int, float]] = field(default_factory=dict)
    # ``EtaCoverage[det_id]`` is a list of contiguous ``(ring_nr, eta_lo_deg,
    # eta_hi_deg)`` arcs. midas-ff-pipeline's transforms stage computes
    # these by pixel-enumeration on each panel.
    EtaCoverage: Dict[int, List[Tuple[int, float, float]]] = field(
        default_factory=dict,
    )

    # --- Scan-aware (pf-HEDM) extensions ---
    # Default values keep behavior identical to FF mode:
    #   scan_pos_tol_um = 0.0  → filter disabled
    #   multi_solution_output = False → today's single-solution IndexBest.bin
    # When ``scan_positions_path`` is set, the indexer reads a 1-D Y-position
    # array (n_scans,) and iterates over the (n_scans × n_scans) Cartesian
    # voxel grid; the matching kernel applies the per-voxel scan-position
    # filter ``|s_proj − ypos| < tol`` (Friedel-symmetric by default; flip
    # off via ``friedel_symmetric_scan_filter=False`` for the C-parity gate).
    # See P0 audit §1b/§1d in the plan file for the binary-layout contract.
    scan_positions_path: str = ""           # path to positions.csv; "" disables scan mode
    scan_pos_tol_um: float = 0.0            # 0 ⇒ kernel disables filter (FF default)
    friedel_symmetric_scan_filter: bool = False  # single-sided = correct physics + matches C
    multi_solution_output: bool = False     # True → emit IndexBest_all.bin + friends
    # Ring number used for seed selection in scanning mode (mirrors
    # IndexerScanningOMP.c:1687-1693: the seed pool is all obs spots with
    # ObsSpotsLab[5] == RingToIndex). FF mode ignores this and uses
    # SpotsToIndex.csv directly.
    RingToIndex: int = 0
    # PF seed-strength floor, in micrometres of equivalent grain radius
    # (``Spots.bin`` col 3, ``GrainRadius``). A spot below this is not tried as
    # a SEED but stays available for matching, so no grain's completeness can
    # change. GrainRadius is used rather than raw intensity because
    # ``midas_transforms/radius`` divides out the ring's ``powder_int`` and
    # ``m_hkl``, making it comparable across rings — a raw cut would conflate a
    # small grain with a low-|F| reflection of a large one.
    # 0.0 ⇒ disabled, historical seed set.
    MinSeedGrainRadius: float = 0.0
    # PREFERRED form of the same floor: drop the weakest fraction of the
    # RingToIndex seed pool, resolved once against that pool's own
    # distribution. The absolute GrainRadius scale is uncalibrated (guessed
    # ``Vgauge``, no polarization term), so a fixed number means a different
    # cut on every layer — on bt_1id_jun25b s1 the per-layer median ranged 16.5 to
    # 68.8. A fraction is the only form that means the same thing everywhere,
    # and it is the reportable statement: "the weakest N % of candidate seed
    # spots were not used as seeds". 0.0 ⇒ disabled.
    SeedDropWeakestFrac: float = 0.0

    def get_ring_radius(self, ring_nr: int) -> float:
        """Sparse lookup mirroring `Params.RingRadii[ring_nr]` in C."""
        return self.RingRadii.get(ring_nr, 0.0)

    def highest_ring_nr(self) -> int:
        """Mirrors `HighestRingNo` computation in IndexerOMP.c:2238."""
        return max(self.RingRadii.keys(), default=0)

    @property
    def is_multi_detector(self) -> bool:
        """True iff at least one ``DetParams`` block was parsed."""
        return bool(self.DetParams)

    def panels_covering(self, ring_nr: int, eta_deg: float) -> List[int]:
        """Return all panel ids whose ``EtaCoverage`` arcs contain (ring, η).

        Empty when (a) no panels are configured, (b) the ring has no
        coverage on any panel, (c) ``eta_deg`` falls in a gap between
        arcs. Used by the indexer's forward adapter to mask predicted
        spots that wouldn't land on any detector.
        """
        out: List[int] = []
        for det_id, arcs in self.EtaCoverage.items():
            for ring_d, lo, hi in arcs:
                if ring_d != ring_nr:
                    continue
                if lo <= eta_deg <= hi:
                    out.append(det_id)
                    break
        return out

    def panel_ring_radius(self, det_id: int, ring_nr: int) -> float:
        """Per-(det, ring) radius from ``RingRadiiPerDet`` with global fallback."""
        per_det = self.RingRadiiPerDet.get(det_id, {})
        if ring_nr in per_det:
            return per_det[ring_nr]
        return self.RingRadii.get(ring_nr, 0.0)
