"""Pipeline configuration dataclasses — frozen interface contracts.

These dataclasses are the **single source of truth** for a midas-pipeline run.
They are the public ABI that downstream stages, parallel-stream developers,
and CLI bindings rely on. **Adding fields is fine; renaming or removing
fields breaks every parallel stream.** Coordinate before changing field
names.

Two scan modes — FF is structurally a single-scan PF:

- ``scan_mode="ff"``: ``n_scans=1``, ``scan_pos_tol_um=0``, kernel
  behavior identical to ``midas-ff-pipeline`` today.
- ``scan_mode="pf"``: ``n_scans>1``, ``scan_pos_tol_um>0`` (defaults to
  ``beam_size_um/2`` inside the kernel), scan-position filter applied
  per voxel.

Constructable explicitly from a notebook or via the CLI (see
``midas_pipeline.cli.build_config``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Literal types (single source of truth for the CLI's choice= lists)
# ---------------------------------------------------------------------------

ScanMode = Literal["ff", "pf"]
ResumeMode = Literal["none", "auto", "from"]
Device = Literal["cpu", "cuda", "mps"]
Dtype = Literal["float32", "float64"]
ReconMethod = Literal["fbp", "mlem", "osem", "voxelmap", "bayesian"]
SinoType = Literal["raw", "norm", "abs", "normabs"]
SinoSource = Literal["tolerance", "indexing"]
SeedingMode = Literal["unseeded", "ff", "merged-ff"]
AlignMethod = Literal["ring-center", "cross-correlation", "none"]
RefinePositionMode = Literal["fixed", "voxel_bounded"]
RefineSolver = Literal["lbfgs", "lm", "nelder_mead", "adam", "lm_batched"]
RefineLoss = Literal["pixel", "angular", "internal_angle"]
RefineMode = Literal["", "iterative", "all_at_once"]
ProcessGrainsMode = Literal["spot_aware", "legacy", "paper_claim"]


# ---------------------------------------------------------------------------
# ScanGeometry — the FF/PF distinction lives here
# ---------------------------------------------------------------------------


@dataclass
class ScanGeometry:
    """One-dimensional (Y-axis) scan geometry.

    PF-HEDM in the C codebase scans along Y only; the 2-D voxel grid is
    the Cartesian product of two sorted Y arrays (P0 audit §1a, evidence
    at SaveBinDataScanning.c:1667–1683).

    For FF mode use ``ScanGeometry.ff()`` (n_scans=1, single zero scan
    position, scan-position tolerance 0).

    Attributes
    ----------
    scan_mode : "ff" | "pf"
        Drives ``midas_pipeline.pipeline.STAGE_ORDER`` selection.
    n_scans : int
        Number of scan positions (== ``len(scan_positions)``).
    scan_positions : np.ndarray[float64], shape (n_scans,)
        1-D Y positions in micrometers.
    beam_size_um : float
        Beam half-width along Y in micrometers. Drives the default scan
        tolerance: when ``scan_pos_tol_um == 0`` the kernel uses
        ``beam_size_um / 2`` (matches C ``ScanPosTol > 0 ? ScanPosTol
        : BeamSize / 2`` precedent).
    scan_pos_tol_um : float
        Spot-to-voxel scan-position-consistency tolerance. ``0`` →
        defer to kernel default (``beam_size_um / 2``).
    friedel_symmetric_scan_filter : bool
        **Defaults to False (single-sided)** to match the C indexer's
        physics. The OR-form
        ``(|s_proj − ypos| < tol) OR (|−s_proj − ypos| < tol)`` was
        once the default but was wrong physics: it accepts spots that
        could be Friedel pairs of OTHER voxels' grains, which doesn't
        correspond to "is THIS voxel in the beam when THIS spot was
        observed." The OR-form is kept available as an opt-in for
        experimental modes (e.g. sinogram cell masking,
        ``pf_MIDAS.py:242``).
    """

    scan_mode: ScanMode
    n_scans: int
    scan_positions: np.ndarray
    beam_size_um: float
    scan_pos_tol_um: float = 0.0
    friedel_symmetric_scan_filter: bool = False

    def __post_init__(self) -> None:
        self.scan_positions = np.asarray(self.scan_positions, dtype=np.float64).ravel()
        if self.scan_positions.shape[0] != self.n_scans:
            raise ValueError(
                f"ScanGeometry.scan_positions has {self.scan_positions.shape[0]} "
                f"entries but n_scans={self.n_scans}"
            )
        if self.scan_mode == "ff" and self.n_scans != 1:
            raise ValueError(
                f"ScanGeometry.scan_mode='ff' requires n_scans=1, got {self.n_scans}"
            )
        if self.scan_mode == "pf" and self.n_scans < 2:
            raise ValueError(
                f"ScanGeometry.scan_mode='pf' requires n_scans>=2, got {self.n_scans}"
            )
        if self.beam_size_um < 0:
            raise ValueError("ScanGeometry.beam_size_um must be >= 0")
        if self.scan_pos_tol_um < 0:
            raise ValueError("ScanGeometry.scan_pos_tol_um must be >= 0")

    @classmethod
    def ff(cls, *, beam_size_um: float = 0.0) -> "ScanGeometry":
        """The FF-HEDM degenerate case: 1 scan position at origin, tol 0."""
        return cls(
            scan_mode="ff",
            n_scans=1,
            scan_positions=np.zeros(1, dtype=np.float64),
            beam_size_um=beam_size_um,
            scan_pos_tol_um=0.0,
            friedel_symmetric_scan_filter=False,
        )

    @classmethod
    def pf_uniform(
        cls,
        *,
        n_scans: int,
        scan_step_um: float,
        beam_size_um: float,
        start_um: Optional[float] = None,
        scan_pos_tol_um: float = 0.0,
        friedel_symmetric_scan_filter: bool = False,
    ) -> "ScanGeometry":
        """Construct a PF scan geometry from uniform step parameters.

        Default centers positions on 0: ``start_um = -(n_scans-1)*step/2``.
        """
        if start_um is None:
            start_um = -(n_scans - 1) * scan_step_um / 2.0
        positions = start_um + np.arange(n_scans, dtype=np.float64) * scan_step_um
        return cls(
            scan_mode="pf",
            n_scans=n_scans,
            scan_positions=positions,
            beam_size_um=beam_size_um,
            scan_pos_tol_um=scan_pos_tol_um,
            friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
        )

    @property
    def is_ff(self) -> bool:
        return self.scan_mode == "ff"

    @property
    def is_pf(self) -> bool:
        return self.scan_mode == "pf"


# ---------------------------------------------------------------------------
# Refinement, reconstruction, fusion, EM, seeding — per-stage knob bundles
# ---------------------------------------------------------------------------


@dataclass
class RefinementConfig:
    """Per-voxel refinement knobs.

    The PF refinement has two position modes, locked in P0 audit §1e:

    - ``"fixed"`` (default): voxel position is locked to the scan grid.
      Bit-or-tolerance parity gate vs C ``FitOrStrainsScanningOMP`` at
      P5.
    - ``"voxel_bounded"``: position is jointly optimized inside
      ``voxel_center ± beam_size/2``. New functionality, no C parity
      reference; tested against synthetic ground truth.

    All refinement uses ``midas_fit_grain.refine`` with
    ``mode="all_at_once"`` (single joint fit over orientation, position,
    and strain; observed↔predicted association frozen at entry).
    """

    position_mode: RefinePositionMode = "fixed"
    solver: RefineSolver = "lbfgs"
    loss: RefineLoss = "pixel"
    mode: RefineMode = "all_at_once"
    # Sigmoid box-bound reparameterization (torch-native, autograd).
    # Bounds the optimizer around the seed so it cannot drift to
    # alternative minima; preserves device portability (no scipy).
    use_bounds: bool = False
    bound_euler_deg: float = 5.0
    bound_lat_abc_pct: float = 0.01
    bound_lat_angle_deg: float = 2.0


@dataclass
class ReconConfig:
    """Tomographic reconstruction knobs (PF-only)."""

    do_tomo: bool = True
    method: ReconMethod = "fbp"
    mlem_iter: int = 50
    osem_subsets: int = 4
    sino_type: SinoType = "raw"
    sino_source: SinoSource = "tolerance"
    sino_conf_min: float = 0.5            # MIDAS_PF_SINO_CONF_MIN
    sino_scan_tol_um: float = 1.5         # MIDAS_PF_SINO_SCAN_TOL
    cull_min_size: int = 0                # drop CCs smaller than this many voxels


@dataclass
class FusionConfig:
    """Bayesian fusion + Potts ICM smoothing (PF-only)."""

    enable_bayesian: bool = False
    max_ang_deg: float = 1.0
    min_conf: float = 0.5
    cw_potts_lambda: float = 0.0          # 0 ⇒ disabled
    cw_potts_max_iter: int = 30
    cw_potts_conf_floor: float = 0.05


@dataclass
class EMConfig:
    """EM spot-ownership refinement (PF-only, optional)."""

    enable: bool = False
    iter: int = 50
    sigma_init: float = 5.0
    sigma_min: float = 0.5
    sigma_decay: float = 0.95
    refine_orientations: bool = False
    opt_steps: int = 50
    lr: float = 1e-3


@dataclass
class SoftAttributionConfig:
    """Soft beam attribution (P6/P7 of the V-map plan).

    When ``enable=True``, the indexer's binary scan-position filter is
    replaced by a continuous weight function and the sinogen stage also
    emits a sum-pooled ``sinos_softsum_*.bin`` variant.  Both default to
    off to preserve bit-exact C parity.
    """

    enable: bool = False
    profile: str = "gaussian"          # "tophat" | "gaussian" | "tophat-ramp"
    fwhm_um: float = 0.0               # 0 ⇒ default to scan.beam_size_um
    tophat_fall_off_um: float = 0.0    # smooth ramp added past beam half-width
    truncate_at_um: float = 0.0        # 0 ⇒ no truncation (Gaussian tails kept)
    omega_sigma_deg: float = 0.0       # 0 ⇒ sinogen uses uniform weights


@dataclass
class VMapConfig:
    """V-map foundation knobs (P8 of the V-map plan).

    Drives the ``calc_radius`` and ``refine_vmap`` stages.  Defaults are
    chosen so existing pipelines that don't enable ``run`` skip the new
    work entirely.
    """

    run: bool = False
    crystal_cif: Optional[str] = None              # path to a CIF for I_theory
    wavelength_A: float = 0.0                       # 0 ⇒ read from scan
    polarization: float = 0.5
    two_theta_max_deg: float = 0.0                  # 0 ⇒ infer from rings
    two_theta_tol_deg: float = 0.05
    # Refinement
    refine_K: bool = True                           # default: closed-form init + LBFGS K
    refine_V: bool = True
    refine_mu: bool = False
    refine_beam: bool = False
    use_absorption: bool = False
    element: str = ""                               # for absorption (μ via NIST)
    max_iter: int = 80
    loss_kind: str = "log_l2"                       # "log_l2" | "huber_log"
    tolerance: float = 1e-8
    # Diagnostics (P9) — figures + tables under ``<layer_dir>/diag/``.
    emit_diagnostics: bool = True
    diag_axes: tuple = (0, 1)                       # which lab axes for V-map 2-D image
    # Beam-voxel projection mode (P5): ``"pf"`` xy rotation, ``"z"`` FF height
    # scan, ``"none"`` FF compact (every grain voxel fully in beam).
    # ``"auto"`` (default) picks "pf" for ``scan_mode=='pf'`` and "none" for FF.
    scan_axis: str = "auto"


@dataclass
class SeedingConfig:
    """Indexer seeding mode.

    ``"merged-ff"`` runs four sub-stages (align → merge_all → ff_index →
    handoff) before the per-voxel indexing pass. See plan §9.
    """

    mode: SeedingMode = "unseeded"
    grains_file: Optional[str] = None
    mic_file: Optional[str] = None

    # merged-ff sub-args (ignored unless mode == "merged-ff")
    merged_align_method: AlignMethod = "ring-center"
    merged_ref_scan: int = -1             # -1 ⇒ n_scans // 2
    merged_min_nhkls: int = -1            # -1 ⇒ MinNHKLs // 2
    merged_tol_px: float = -1.0           # -1 ⇒ 2 * pixel_size
    merged_tol_ome: float = -1.0          # -1 ⇒ 2 * omega_step

    def __post_init__(self) -> None:
        if self.grains_file:
            self.grains_file = str(Path(self.grains_file).resolve())
        if self.mic_file:
            self.mic_file = str(Path(self.mic_file).resolve())


@dataclass
class LayerSelection:
    """Inclusive layer range (FF-style multi-layer scans)."""

    start: int = 1
    end: int = 1

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("LayerSelection.start must be >= 1")
        if self.end < self.start:
            raise ValueError("LayerSelection.end must be >= start")

    def layers(self) -> list[int]:
        return list(range(self.start, self.end + 1))


@dataclass
class MachineConfig:
    """Dispatch target (parsl cluster name + node count)."""

    name: str = "local"
    n_nodes: int = 1


# ---------------------------------------------------------------------------
# PipelineConfig — top-level frozen-by-convention dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """The single source of truth for a pipeline run.

    Constructable explicitly from a notebook or via the CLI. Field
    additions are safe; renaming or removing fields breaks parallel
    streams — coordinate before touching.
    """

    # Required
    result_dir: str
    params_file: str
    scan: ScanGeometry

    # Detector discovery
    zarr_path: Optional[str] = None
    detectors_json: Optional[str] = None

    # Sub-configs
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    recon: ReconConfig = field(default_factory=ReconConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    em: EMConfig = field(default_factory=EMConfig)
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    soft_attribution: SoftAttributionConfig = field(default_factory=SoftAttributionConfig)
    vmap: VMapConfig = field(default_factory=VMapConfig)
    layer_selection: LayerSelection = field(default_factory=LayerSelection)
    machine: MachineConfig = field(default_factory=MachineConfig)

    # Compute
    n_cpus: int = 16
    n_cpus_local: int = 4
    device: Device = "cuda"
    dtype: Dtype = "float64"

    # Resume / staging
    resume: ResumeMode = "auto"
    resume_from_stage: Optional[str] = None
    only_stages: list[str] = field(default_factory=list)
    skip_stages: list[str] = field(default_factory=list)

    # Indexer
    indexer_group_size: int = 4
    shard_gpus: Optional[str] = None

    # Process-grains (FF mode only — see plan §3d)
    process_grains_mode: ProcessGrainsMode = "spot_aware"

    # Raw-data overrides
    raw_dir: Optional[str] = None

    # Ingestion
    num_frame_chunks: int = -1
    preproc_thresh: int = -1
    convert_files: bool = True
    file_name: Optional[str] = None
    num_files_per_scan: int = 1
    normalize_intensities: int = 2
    do_peak_search: int = 1
    one_sol_per_vox: bool = True
    peak_fit_gpu: bool = False             # midas-peakfit torch backend

    # sr-midas
    run_sr: bool = False
    srfac: int = 8
    sr_config_path: str = "auto"
    save_sr_patches: bool = False
    save_frame_good_coords: bool = False

    # Consolidation
    generate_h5: bool = False

    # Validation
    skip_validation: bool = False
    strict_validation: bool = False

    # Misc
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.result_dir = str(Path(self.result_dir).resolve())
        self.params_file = str(Path(self.params_file).resolve())
        if self.zarr_path:
            self.zarr_path = str(Path(self.zarr_path).resolve())
        if self.detectors_json:
            self.detectors_json = str(Path(self.detectors_json).resolve())
        if self.raw_dir:
            self.raw_dir = str(Path(self.raw_dir).resolve())
        if self.resume == "from" and not self.resume_from_stage:
            raise ValueError("resume='from' requires resume_from_stage")
        if self.resume_from_stage and self.resume != "from":
            self.resume = "from"
        if self.layer_selection is None:
            self.layer_selection = LayerSelection()
        # Cross-config sanity: PF-only sub-configs must not be enabled in FF mode.
        if self.scan.is_ff and self.recon.do_tomo:
            # Soft: tomography is meaningless in FF; disable silently.
            self.recon = ReconConfig(do_tomo=False, method=self.recon.method)

    # --- Convenience -------------------------------------------------

    @property
    def result_path(self) -> Path:
        return Path(self.result_dir)

    @property
    def is_ff(self) -> bool:
        return self.scan.is_ff

    @property
    def is_pf(self) -> bool:
        return self.scan.is_pf

    def layer_dir(self, layer_nr: int) -> Path:
        return self.result_path / f"LayerNr_{layer_nr}"

    def merged_ref_scan(self) -> int:
        """Resolve the -1 sentinel to n_scans // 2 at use-site."""
        ref = self.seeding.merged_ref_scan
        return ref if ref >= 0 else self.scan.n_scans // 2


# ---------------------------------------------------------------------------
# Scan-mode sniffing from a legacy MIDAS parameter file
# ---------------------------------------------------------------------------


def sniff_scan_mode_from_paramfile(path: str | Path) -> ScanMode:
    """Heuristic: ``nScans > 1`` or scanning keys present → 'pf', else 'ff'.

    Used by the CLI when ``--scan-mode`` is omitted. Not authoritative —
    if the caller knows otherwise they should pass ``--scan-mode``
    explicitly.
    """
    p = Path(path)
    if not p.exists():
        return "ff"
    n_scans = 1
    has_scanning_key = False
    for line in p.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "nScans" and len(parts) >= 2:
            try:
                n_scans = int(parts[1])
            except ValueError:
                pass
        if key in {"BeamSize", "ScanPosTol", "px", "ScanStep"}:
            has_scanning_key = True
    if n_scans > 1:
        return "pf"
    if has_scanning_key and n_scans > 1:
        return "pf"
    return "ff"


def read_scan_geometry_from_paramfile(path: str | Path) -> Optional[dict]:
    """Extract scan-geometry knobs from a paramstest.txt for ScanGeometry.pf_uniform.

    Returns a dict with the keys ScanGeometry.pf_uniform takes, or None
    when the file is missing or contains no scanning keys. Used by the
    CLI to default ``--n-scans`` / ``--scan-step`` / ``--beam-size`` from
    the params file so the user doesn't have to repeat them on the command
    line. CLI flags override anything sniffed here.

    Keys recognised (matches the C / pf_MIDAS.py convention):
      nScans     → n_scans
      ScanStep   → scan_step_um (do NOT confuse with ``px`` — that's the
                   detector pixel pitch, a completely different number)
      BeamSize   → beam_size_um
      ScanPosTol → scan_pos_tol_um (0 ⇒ kernel defaults to BeamSize/2)
    """
    p = Path(path)
    if not p.exists():
        return None
    found = False
    out: dict = {}
    for raw in p.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        key = parts[0]
        if len(parts) < 2:
            continue
        token = parts[1].rstrip(";")
        try:
            if key == "nScans":
                out["n_scans"] = int(token); found = True
            elif key == "ScanStep":
                out["scan_step_um"] = float(token); found = True
            elif key == "BeamSize":
                out["beam_size_um"] = float(token); found = True
            elif key == "ScanPosTol":
                out["scan_pos_tol_um"] = float(token); found = True
        except ValueError:
            continue
    return out if found else None
