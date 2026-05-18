"""Pipeline configuration dataclasses.

``PipelineConfig`` is the single source of truth for a pipeline run.
Constructable explicitly from a notebook, or via the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

ResumeMode = Literal["none", "auto", "from"]
Device = Literal["cpu", "cuda", "mps"]
Dtype = Literal["float32", "float64"]


@dataclass
class LayerSelection:
    """Inclusive layer range to process. ``start == end == 1`` for single-layer."""

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
    """Where the run executes.

    For now only ``name="local"`` is wired (in-process, no scheduler).
    Cluster names (polaris, alleppey, copland) reserve room for future
    setsid-bash dispatch like ff_MIDAS does today.
    """

    name: str = "local"
    n_nodes: int = 1


@dataclass
class PipelineConfig:
    """Top-level configuration."""

    result_dir: str                                # top-level run directory
    params_file: str                               # Parameters.txt or paramstest.txt
    zarr_path: Optional[str] = None                # single-detector .MIDAS.zip (overrides discovery)
    detectors_json: Optional[str] = None           # path to detectors.json (multi-det)
    n_cpus: int = 16
    device: Device = "cuda"
    dtype: Dtype = "float64"
    layer_selection: LayerSelection = field(default_factory=LayerSelection)
    machine: MachineConfig = field(default_factory=MachineConfig)

    # Resume / state
    resume: ResumeMode = "auto"
    resume_from_stage: Optional[str] = None        # only when resume == "from"

    # Stage selection — empty list means "all"
    only_stages: list[str] = field(default_factory=list)
    skip_stages: list[str] = field(default_factory=list)

    # Indexer / refiner knobs
    refine_solver: Literal["lbfgs", "lm", "nelder_mead", "adam", "lm_batched"] = "lbfgs"
    refine_loss: Literal["pixel", "angular", "internal_angle"] = "pixel"
    refine_mode: Literal["", "iterative", "all_at_once"] = ""
    indexer_group_size: int = 4                    # default to small group for fp64 safety
    process_grains_mode: Literal["spot_aware", "legacy", "paper_claim"] = "spot_aware"

    # Multi-GPU sharding for the indexer stage. Comma-separated list of
    # CUDA device indices, e.g. "0,1" to fan the indexer across two GPUs.
    # Each shard handles a disjoint slice of seeds (--block-nr / --n-blocks)
    # and pwrites into the shared IndexBest.bin file. None / empty = single
    # GPU. Refinement and downstream stages still run single-GPU.
    shard_gpus: Optional[str] = None

    # Multi-process CPU sharding for the indexer stage. midas-index uses
    # torch intra-op threading; scaling collapses past ~16 threads on the
    # small per-seed ops the indexer runs. With cpu_shards > 1 the pipeline
    # spawns N concurrent midas-index processes, each with
    # ``torch.set_num_threads(n_cpus // N)`` over a disjoint seed slice.
    # Ignored on GPU. 0 / 1 disables sharding (single process gets all cpus).
    cpu_shards: int = 1

    # --- Layer-aware seeding (gap #3, #4) ---
    grains_file: Optional[str] = None              # explicit GrainsFile to seed indexer/refiner
    nf_result_dir: Optional[str] = None            # dir of NF GrainsLayer{N}.csv files; per-layer override

    # --- Raw-data overrides (gap #5) ---
    raw_dir: Optional[str] = None                  # overrides RawFolder (and Dark path) before zip_convert

    # --- Validation (gap #10) ---
    skip_validation: bool = False
    strict_validation: bool = False

    # --- Ingestion (gap #1) ---
    num_frame_chunks: int = -1                     # -1 → resolve_runtime_defaults picks n_cpus*4
    preproc_thresh: int = -1                       # -1 → resolve_runtime_defaults picks min(RingThresh)
    convert_files: bool = True                     # set False to skip zip_convert (zarr already exists)
    file_name: Optional[str] = None                # single-file override; layer_nr derived from FileNr
    num_files_per_scan: int = 1                    # passes through to ffGenerateZipRefactor

    # --- Consolidation (gap #11) ---
    generate_h5: bool = False                      # emit consolidated grain↔peak HDF5

    # --- sr-midas (gap #9) ---
    run_sr: bool = False
    srfac: int = 8
    sr_config_path: str = "auto"
    save_sr_patches: bool = False
    save_frame_good_coords: bool = False

    # --- Misc ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Coerce paths to absolute strings for consistent downstream use.
        self.result_dir = str(Path(self.result_dir).resolve())
        self.params_file = str(Path(self.params_file).resolve())
        if self.zarr_path:
            self.zarr_path = str(Path(self.zarr_path).resolve())
        if self.detectors_json:
            self.detectors_json = str(Path(self.detectors_json).resolve())
        if self.grains_file:
            self.grains_file = str(Path(self.grains_file).resolve())
        if self.nf_result_dir:
            self.nf_result_dir = str(Path(self.nf_result_dir).resolve())
        if self.raw_dir:
            self.raw_dir = str(Path(self.raw_dir).resolve())
        if self.resume == "from" and not self.resume_from_stage:
            raise ValueError(
                "resume='from' requires resume_from_stage to name the stage"
            )
        if self.resume_from_stage and self.resume != "from":
            self.resume = "from"
        if self.layer_selection is None:
            self.layer_selection = LayerSelection()

    @property
    def result_path(self) -> Path:
        return Path(self.result_dir)

    def layer_dir(self, layer_nr: int) -> Path:
        return self.result_path / f"LayerNr_{layer_nr}"
