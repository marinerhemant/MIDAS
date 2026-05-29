"""midas-pipeline CLI.

Subcommands:
    run         end-to-end pipeline (ff or pf scan-mode)
    status      show resume state of a run directory
    resume      restart from a named stage
    reprocess   rerun consolidation / recon on a completed dir
    inspect     dump per-stage timings + grain count for one layer
    simulate    forward-simulate synthetic data for tests
    seed        run the merged-FF seeding sub-pipeline standalone

For the FF degenerate case, ``midas-pipeline run --scan-mode ff …`` is
equivalent to today's ``midas-ff-pipeline run`` invocation. The
``midas-ff-pipeline`` console-script (defined in the
``midas-ff-pipeline`` package) is a thin shim that injects
``--scan-mode ff`` and delegates here — see the back-compat note in the
README.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from ._logging import LOG, configure_logging
from .config import (
    AlignMethod,
    Device,
    Dtype,
    EMConfig,
    FusionConfig,
    LayerSelection,
    MachineConfig,
    PipelineConfig,
    ProcessGrainsMode,
    ReconConfig,
    ReconMethod,
    RefineLoss,
    RefineMode,
    RefinePositionMode,
    RefineSolver,
    RefinementConfig,
    ResumeMode,
    ScanGeometry,
    GrainGeometryConfig,
    ScanMode,
    SeedingConfig,
    SeedingMode,
    SinoSource,
    SinoType,
    SoftAttributionConfig,
    VMapConfig,
    VoxelCleanupConfig,
    read_scan_geometry_from_paramfile,
    sniff_scan_mode_from_paramfile,
)
from .pipeline import Pipeline, all_stage_names
from .provenance import ProvenanceStore


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-pipeline",
        description="Unified MIDAS HEDM orchestrator (FF + PF, single source).",
    )
    p.add_argument("--version", action="version", version=f"midas-pipeline {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ----- run -----
    run = sub.add_parser("run", help="execute the pipeline end-to-end")
    run.add_argument("--params", required=True, help="path to Parameters.txt / paramstest.txt")
    run.add_argument("--result", required=True, help="result directory (LayerNr_N subdirs created here)")

    run.add_argument("--scan-mode", choices=["ff", "pf", "auto"], default="auto",
                     help="ff: single-scan / today's FF-HEDM. pf: multi-scan PF-HEDM. "
                          "auto: sniff from params (nScans>1 → pf).")

    # PF-mode scan geometry
    run.add_argument("--n-scans", type=int, default=None,
                     help="(pf) number of scan positions; required if scan-mode=pf")
    run.add_argument("--scan-step", type=float, default=None,
                     help="(pf) Y step between scans, micrometers")
    run.add_argument("--beam-size", type=float, default=0.0,
                     help="beam half-width along Y, micrometers")
    run.add_argument("--scan-pos-tol", type=float, default=0.0,
                     help="(pf) scan-position-filter tolerance (0 → beam_size/2)")
    run.add_argument("--friedel-symmetric-scan-filter", dest="friedel",
                     action="store_true",
                     help="(pf) enable OR-form scan filter: "
                          "(|s_proj − ypos| < tol) OR (|−s_proj − ypos| < tol). "
                          "Default is single-sided, which matches C and the "
                          "correct physics for 'voxel illuminated when spot "
                          "observed'. OR-form is an experimental opt-in.")
    # Kept for backwards-compat (no-op now since the default is already
    # single-sided), but allowed so old invocations don't fail.
    run.add_argument("--no-friedel-symmetric-scan-filter", dest="friedel",
                     action="store_false",
                     help=argparse.SUPPRESS)
    run.set_defaults(friedel=False)

    # Compute
    run.add_argument("--n-cpus", type=int, default=16)
    run.add_argument("--n-cpus-local", type=int, default=4)
    run.add_argument("--machine", default="local")
    run.add_argument("--n-nodes", type=int, default=1)
    run.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    run.add_argument("--dtype", choices=["auto", "float32", "float64"],
                     default="auto",
                     help="'auto' = float32 on cuda/mps (production speed), "
                          "float64 on cpu (matches the fp64 parity gate).")

    # Resume / stage selection
    run.add_argument("--resume", choices=["none", "auto", "from"], default="auto")
    run.add_argument("--from", dest="resume_from_stage", default=None,
                     metavar="STAGE",
                     help=f"stage to resume from (requires --resume=from). "
                          f"valid stages: {', '.join(all_stage_names())}")
    run.add_argument("--only", action="append", default=[], metavar="STAGE",
                     help="run only these stages (repeatable)")
    run.add_argument("--skip", action="append", default=[], metavar="STAGE",
                     help="skip these stages (repeatable)")

    # Refinement
    run.add_argument("--pf-refine-mode", choices=["fixed", "voxel_bounded"],
                     default="fixed",
                     help="(pf) position-refinement mode")
    run.add_argument("--refine-solver", choices=["lbfgs", "lm", "nelder_mead", "adam", "lm_batched"],
                     default="lbfgs")
    run.add_argument("--refine-loss",
                     choices=["full3d", "angular", "internal_angle"],
                     default="full3d")  # 2D 'pixel' loss disabled in fit-grain
    run.add_argument("--refine-mode", choices=["", "iterative", "all_at_once"],
                     default="all_at_once",
                     help="refinement strategy; default 'all_at_once' (single joint fit)")
    run.add_argument("--use-bounds", action="store_true",
                     help="bound refinement via sigmoid reparam (torch-native, "
                          "autograd-preserving); recommended for PF to prevent "
                          "boundary-voxel drift")
    run.add_argument("--bound-euler-deg", type=float, default=5.0,
                     help="±half-width on each Euler component (default 5°)")
    run.add_argument("--bound-lat-abc-pct", type=float, default=0.01,
                     help="±fractional half-width on a, b, c (default 0.01 = 1%%)")
    run.add_argument("--bound-lat-angle-deg", type=float, default=2.0,
                     help="±half-width on α, β, γ (default 2°)")

    # Indexer
    run.add_argument("--group-size", default="auto",
                     help="indexer seed group size. 'auto' picks based on the "
                          "smallest visible GPU's memory tier (≥70GB→8, ≥32GB→4, "
                          "≥16GB→2, <16GB→1), then down-scales for datasets "
                          "denser than the park22 baseline (8 rings, 720 frames). "
                          "An integer overrides the auto pick.")
    run.add_argument("--shard-gpus", default="auto",
                     help="comma-separated CUDA indices for multi-GPU indexing. "
                          "'auto' uses every visible CUDA device when --device=cuda; "
                          "'none' / empty disables sharding.")
    run.add_argument("--cpu-shards", default="auto",
                     help="how many CPU-only indexer shards to run in parallel. "
                          "'auto' picks max(1, n_cpus // 16) on CPU, 1 elsewhere.")
    run.add_argument("--indexer-backend", choices=["python", "c-omp"],
                     default="c-omp",
                     help="indexing backend: 'c-omp' (default, bundled unified "
                          "C binary, requires OpenMP-built midas-index install) "
                          "or 'python' (in-process numba/torch; needed for GPU "
                          "runs and the fp64 parity gate).")
    run.add_argument("--refine-backend", choices=["python", "c-omp"],
                     default="python",
                     help="refinement backend: 'python' (default, in-process "
                          "PyTorch refiner; differentiable, GPU/MPS, UQ) or "
                          "'c-omp' (bundled unified C binary midas_fitgrain; "
                          "FF refines position, PF fixes it; requires "
                          "OpenMP-built midas-fit-grain install).")

    # Recon (PF)
    run.add_argument("--do-tomo", type=int, default=1, choices=[0, 1])
    run.add_argument("--recon-method",
                     choices=["fbp", "mlem", "osem", "voxelmap", "bayesian"],
                     default="fbp")
    run.add_argument("--mlem-iter", type=int, default=50)
    run.add_argument("--osem-subsets", type=int, default=4)
    run.add_argument("--sino-type", choices=["raw", "norm", "abs", "normabs"],
                     default="raw")
    run.add_argument("--sino-source", choices=["tolerance", "indexing"],
                     default="tolerance")
    run.add_argument("--sino-conf-min", type=float, default=0.5)
    run.add_argument("--sino-scan-tol", type=float, default=1.5)
    run.add_argument("--cull-min-size", type=int, default=0)

    # Fusion (PF)
    run.add_argument("--cw-potts-lambda", type=float, default=0.0,
                     help="(pf) Potts ICM strength; 0 disables")
    run.add_argument("--max-ang-deg", type=float, default=1.0)
    run.add_argument("--min-conf", type=float, default=0.5)

    # EM (PF)
    run.add_argument("--use-em", type=int, default=0, choices=[0, 1])
    run.add_argument("--em-iter", type=int, default=50)
    run.add_argument("--em-sigma-init", type=float, default=5.0)
    run.add_argument("--em-sigma-min", type=float, default=0.5)
    run.add_argument("--em-sigma-decay", type=float, default=0.95)
    run.add_argument("--em-refine-orientations", type=int, default=0, choices=[0, 1])
    run.add_argument("--em-opt-steps", type=int, default=50)
    run.add_argument("--em-lr", type=float, default=1e-3)

    # Seeding
    run.add_argument("--seeding-mode", choices=["unseeded", "ff", "merged-ff"],
                     default="unseeded")
    run.add_argument("--grains-file", default=None)
    run.add_argument("--mic-file", default=None)
    run.add_argument("--merged-align-method",
                     choices=["ring-center", "cross-correlation", "none"],
                     default="ring-center")
    run.add_argument("--merged-ref-scan", type=int, default=-1,
                     help="-1 ⇒ n_scans // 2")
    run.add_argument("--merged-min-nhkls", type=int, default=-1)
    run.add_argument("--merged-tol-px", type=float, default=-1.0)
    run.add_argument("--merged-tol-ome", type=float, default=-1.0)

    # V-map joint refinement (P8/P9 of the V-map plan)
    run.add_argument("--vmap-run", action="store_true",
                     help="enable calc_radius_v + refine_vmap V-map stages")
    run.add_argument("--vmap-crystal-cif", default=None,
                     help="path to CIF for theoretical I_ring computation")
    run.add_argument("--vmap-wavelength", type=float, default=0.0,
                     help="λ in Å for I_theory; 0 ⇒ read from paramstest.txt")
    run.add_argument("--vmap-refine-V", type=int, default=1, choices=[0, 1])
    run.add_argument("--vmap-refine-K", type=int, default=1, choices=[0, 1])
    run.add_argument("--vmap-refine-mu", type=int, default=0, choices=[0, 1])
    run.add_argument("--vmap-refine-beam", type=int, default=0, choices=[0, 1])
    run.add_argument("--vmap-use-absorption", action="store_true")
    run.add_argument("--vmap-element", default="",
                     help="single element for absorption μ via NIST table")
    run.add_argument("--vmap-max-iter", type=int, default=80)
    run.add_argument("--vmap-loss-kind", choices=["log_l2", "huber_log"],
                     default="log_l2")
    run.add_argument("--vmap-tolerance", type=float, default=1e-8)
    run.add_argument("--vmap-gauge-reg", type=float, default=1e-2,
                     help="Tikhonov penalty weight on log(V) to fix the V·K "
                          "scale degeneracy (mean V→1). 0 disables; ignored "
                          "when refine-K=0 since K then fixes the scale.")
    run.add_argument("--vmap-beam-geometry", choices=["center", "siddon"],
                     default="center",
                     help="Beam-voxel weight geometry: 'center' (original "
                          "voxel-centre box) or 'siddon' (exact rotated-square "
                          "trapezoidal footprint; reduces diagonal streaks).")
    run.add_argument("--vmap-soft-grain-attribution", action="store_true",
                     help="Blend each spot across the grains whose voxels "
                          "matched it (weight ∝ matched-voxel count) instead "
                          "of forcing it onto a single grain.")
    # --- grain-based tx/Wedge geometry refine (FF; OFF by default) ---------
    run.add_argument("--grain-geometry-run", action="store_true",
                     help="FF only: after process_grains, refine tx (powder is "
                          "blind to it) — and optionally Wedge — from the "
                          "recovered grain spots; writes a corrected paramstest "
                          "for a re-run from the transforms stage.")
    run.add_argument("--grain-geometry-refine", default="tx",
                     help="comma-separated geometry blocks to refine "
                          "(default 'tx'; e.g. 'tx,Wedge').")
    run.add_argument("--grain-geometry-kind", default="angular",
                     choices=["angular", "internal_angle"],
                     help="η-sensitive loss; 'pixel' is disabled (blind to tx).")
    run.add_argument("--grain-geometry-max-grains", type=int, default=50)
    run.add_argument("--grain-geometry-out", default="paramstest_graintx.txt",
                     help="corrected paramstest filename (under the layer dir).")

    # --- missing-spot directionality voxel cleanup (PF; OFF by default) ----
    run.add_argument("--voxel-cleanup", action="store_true",
                     help="Enable missing-spot directionality voxel cleanup "
                          "after find_grains (compact-grain regime). Removes/"
                          "reassigns voxels whose predicted spots are unmatched "
                          "in directions where the grain's sinogram is empty.")
    run.add_argument("--voxel-cleanup-threshold", type=float, default=0.30,
                     help="Min geometry-miss fraction to act on a voxel.")
    run.add_argument("--voxel-cleanup-max-neighbours", type=int, default=1,
                     help="Connectivity gate: only act on voxels with at most "
                          "this many same-grain neighbours. 1=isolated/jutting "
                          "(speckle mode); 4=disable gate (extent-clip mode).")
    run.add_argument("--voxel-cleanup-max-iters", type=int, default=8)
    run.add_argument("--voxel-cleanup-action", choices=["reassign", "remove"],
                     default="reassign",
                     help="reassign flagged voxels to majority neighbour grain, "
                          "or remove (set grain_id=-1).")
    run.add_argument("--vmap-emit-diagnostics", type=int, default=1, choices=[0, 1])
    run.add_argument("--vmap-diag-axes", default="0,1",
                     help="comma-pair (a,b) selecting lab axes for the 2-D "
                          "V-map image (default '0,1' = x–y)")

    # Soft beam attribution (P6/P7)
    run.add_argument("--soft-attribution", action="store_true",
                     help="enable continuous beam-weight in indexer + sinogen")
    run.add_argument("--soft-profile",
                     choices=["gaussian", "tophat", "tophat-ramp"],
                     default="gaussian")
    run.add_argument("--soft-fwhm-um", type=float, default=0.0,
                     help="beam FWHM in µm; 0 ⇒ defaults to scan.beam_size_um")
    run.add_argument("--soft-tophat-fall-off-um", type=float, default=0.0)
    run.add_argument("--soft-truncate-at-um", type=float, default=0.0)
    run.add_argument("--soft-omega-sigma-deg", type=float, default=0.0,
                     help="sino-soft ω-Gaussian σ; 0 ⇒ uniform sum-pool")

    # Process-grains (FF only)
    run.add_argument("--pg-mode", choices=["spot_aware", "legacy", "paper_claim"],
                     default="spot_aware")

    # Layers
    run.add_argument("--layers", default="1-1",
                     help="layer range, e.g. '1-1' or '1-5'")

    # Ingestion
    run.add_argument("--num-frame-chunks", type=int, default=-1)
    run.add_argument("--preproc-thresh", type=int, default=-1)
    run.add_argument("--no-convert", dest="convert_files", action="store_false")
    run.set_defaults(convert_files=True)
    run.add_argument("--file-name", default=None)
    run.add_argument("--num-files-per-scan", type=int, default=1)
    run.add_argument("--normalize-intensities", type=int, default=2,
                     choices=[0, 1, 2, 3])
    run.add_argument("--do-peak-search", type=int, default=1, choices=[0, 1, -1])
    run.add_argument("--one-sol-per-vox", type=int, default=1, choices=[0, 1])
    run.add_argument("--peak-fit-gpu", action="store_true")

    # sr-midas
    run.add_argument("--run-sr", action="store_true")
    run.add_argument("--srfac", type=int, default=8)
    run.add_argument("--sr-config", default="auto")
    run.add_argument("--save-sr-patches", action="store_true")
    run.add_argument("--save-frame-good-coords", action="store_true")

    # Output
    run.add_argument("--generate-h5", action="store_true")

    # Detector discovery
    run.add_argument("--zarr", default=None,
                     help="single-detector .MIDAS.zip (overrides discovery)")
    run.add_argument("--detectors", default=None,
                     help="detectors.json for multi-detector run")
    run.add_argument("--raw-dir", default=None)

    # Batch mode (FF; mirrors legacy ff_MIDAS.py -batchMode)
    run.add_argument("--batch", action="store_true",
                     help="(FF) auto-discover one raw file per layer in "
                          "RawFolder; iterate FileStem + per-layer seed "
                          "resolution across the layer range.")
    run.add_argument("--nf-result-dir", default=None,
                     help="(FF) directory containing GrainsLayer{N}.csv "
                          "seed files (NF→FF handoff).")
    run.add_argument("--ff-grains-file", default=None,
                     help="(FF) one GrainsFile applied to every layer; "
                          "alternative to --nf-result-dir.")

    # Validation
    run.add_argument("--skip-validation", action="store_true")
    run.add_argument("--strict-validation", action="store_true")

    # Logging
    run.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ----- status / resume / reprocess / inspect / simulate / seed -----
    st = sub.add_parser("status", help="show resume state of a run directory")
    st.add_argument("result_dir")
    st.add_argument("--layers", default=None)
    st.add_argument("--json", action="store_true")

    rs = sub.add_parser("resume", help="resume a run from a stage")
    rs.add_argument("result_dir")
    rs.add_argument("--params", default=None)
    rs.add_argument("--from", dest="resume_from_stage", required=True, metavar="STAGE")
    rs.add_argument("--layers", default=None)

    rp = sub.add_parser("reprocess",
                        help="rerun merge_overlaps + consolidation on completed dir")
    rp.add_argument("result_dir")
    rp.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    rp.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    rp.add_argument("--n-cpus", type=int, default=16)

    ins = sub.add_parser("inspect", help="dump per-stage timings + grain count")
    ins.add_argument("layer_dir")
    ins.add_argument("--json", action="store_true")

    sim = sub.add_parser("simulate", help="forward-simulate a synthetic dataset")
    sim.add_argument("--out", required=True)
    sim.add_argument("--params", required=True)
    sim.add_argument("--n-grains", type=int, default=5)
    sim.add_argument("--n-scans", type=int, default=15)
    sim.add_argument("--scan-range", type=float, default=70.0)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--n-cpus", type=int, default=8)
    sim.add_argument("--n-detectors", type=int, default=1,
                     help="number of detectors. >1 → multi-det layout")
    sim.add_argument("--mode", default="ff_compressed",
                     choices=["ff_compressed", "diffract_pinwheel"],
                     help="forward-sim backend: 'ff_compressed' (legacy "
                          "ForwardSimulationCompressed, single-panel per call) "
                          "or 'diffract_pinwheel' (midas_diffract HEDMForwardModel "
                          "with multi_mode='panel'; required for true pinwheel).")
    sim.add_argument("--device", default="cuda",
                     help="GPU device for diffract_pinwheel (cuda / cuda:0 / cpu)")
    sim.add_argument("--no-hydra-geometry", action="store_true",
                     help="for diffract_pinwheel: use simplified pinwheel "
                          "(shared BC, no Lsd jitter) instead of hydra-real geometry")

    sd = sub.add_parser("seed",
                        help="run the merged-FF seeding sub-pipeline standalone")
    sd.add_argument("--params", required=True)
    sd.add_argument("--result", required=True)
    sd.add_argument("--output", default="UniqueOrientations.csv")

    return p


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _parse_layers(spec: str) -> LayerSelection:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return LayerSelection(start=int(start), end=int(end))
    n = int(spec)
    return LayerSelection(start=n, end=n)


# ---------------------------------------------------------------------------
# Auto-resolution helpers (cluster + GPU + dataset density)
# ---------------------------------------------------------------------------


def _resolve_dtype(device: str, dtype_arg: str) -> str:
    """``auto`` → float32 on cuda/mps (production), float64 on cpu (parity).
    Explicit values pass through.
    """
    if dtype_arg != "auto":
        return dtype_arg
    return "float32" if device in ("cuda", "mps") else "float64"


def _resolve_shard_gpus(device: str, shard_arg: str | None) -> str | None:
    """``auto`` → all visible CUDA devices when device=cuda. ``none``/empty
    disables sharding. Explicit comma list (e.g. ``'0,1'``) passes through.
    """
    if shard_arg in ("none", "None", "", None):
        return None
    if shard_arg != "auto":
        return shard_arg
    if device != "cuda":
        return None
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        return None
    if n <= 1:
        return None
    return ",".join(str(i) for i in range(n))


def _resolve_cpu_shards(device: str, n_cpus: int, cpu_shards_arg: str) -> int:
    """``auto`` → ``max(1, n_cpus // 16)`` on CPU, 1 elsewhere."""
    if device != "cpu":
        return 1
    if cpu_shards_arg in ("auto", ""):
        return max(1, n_cpus // 16)
    try:
        return max(1, int(cpu_shards_arg))
    except ValueError:
        raise SystemExit(
            f"--cpu-shards must be 'auto' or an integer, got {cpu_shards_arg!r}"
        )


_BASELINE_N_RINGS = 8        # park22 FCC reference
_BASELINE_N_OMEGA = 720      # park22 reference scan length


def _count_dataset_density(params_file: str | None) -> tuple[int, int]:
    """Parse paramstest to return ``(n_rings, n_omega_steps)``. ``(0, 0)`` on
    any failure (caller treats as no density info).
    """
    if params_file is None:
        return 0, 0
    try:
        with open(params_file) as f:
            text = f.read()
    except OSError:
        return 0, 0
    ring_keys = {"RingNumbers", "RingThresh", "RingsToIndex", "RingsToUse"}
    rings_per_key: dict[str, int] = {}
    ostart = ostop = ostep = None
    for raw in text.splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]
        if key in ring_keys:
            rings_per_key[key] = rings_per_key.get(key, 0) + 1
        elif key == "OmegaStart" and len(parts) >= 2:
            try: ostart = float(parts[1])
            except ValueError: pass
        elif key == "OmegaEnd" and len(parts) >= 2:
            try: ostop = float(parts[1])
            except ValueError: pass
        elif key == "OmegaStep" and len(parts) >= 2:
            try: ostep = float(parts[1])
            except ValueError: pass
    n_rings = max(rings_per_key.values()) if rings_per_key else 0
    if ostart is None or ostop is None or ostep is None or ostep == 0:
        n_omega = 0
    else:
        n_omega = int(abs(ostop - ostart) / abs(ostep))
    return n_rings, n_omega


def _resolve_group_size(device: str, shard_gpus: str | None,
                        group_size_arg: str,
                        *,
                        params_file: str | None = None) -> int:
    """``auto`` → memory-tier baseline (≥70GB→8, ≥32GB→4, ≥16GB→2, <16GB→1)
    down-scaled by dataset density (rings × ω-steps / park22 baseline).
    Falls back to 4 (the legacy default) if memory probing fails or
    device is non-cuda. Explicit integer passes through.
    """
    if group_size_arg != "auto":
        try:
            return int(group_size_arg)
        except ValueError:
            raise SystemExit(
                f"--group-size must be 'auto' or an integer, got {group_size_arg!r}"
            )
    if device != "cuda":
        return 4
    try:
        import torch
        if not torch.cuda.is_available():
            return 4
        if shard_gpus:
            indices = [int(x) for x in shard_gpus.split(",") if x.strip()]
        else:
            indices = list(range(torch.cuda.device_count()))
        if not indices:
            return 4
        min_mem_gb = min(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in indices
        )
    except Exception:
        return 4

    if min_mem_gb >= 70:
        baseline_gs = 8
    elif min_mem_gb >= 32:
        baseline_gs = 4
    elif min_mem_gb >= 16:
        baseline_gs = 2
    else:
        baseline_gs = 1

    n_rings, n_omega = _count_dataset_density(params_file)
    rings_factor = (n_rings / _BASELINE_N_RINGS) if n_rings > 0 else 1.0
    omega_factor = (n_omega / _BASELINE_N_OMEGA) if n_omega > 0 else 1.0
    if n_rings == 0 and n_omega == 0:
        return baseline_gs
    density = max(1.0, rings_factor * omega_factor)
    return max(1, int(baseline_gs / density))


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Translate argparse Namespace → PipelineConfig.

    Sniffs scan-mode from the params file when ``--scan-mode auto``.
    """
    # Resolve scan mode
    scan_mode: ScanMode = args.scan_mode
    if scan_mode == "auto":
        scan_mode = sniff_scan_mode_from_paramfile(args.params)
        LOG.info("scan-mode=auto → sniffed '%s' from %s", scan_mode, args.params)

    # Build ScanGeometry
    if scan_mode == "ff":
        scan = ScanGeometry.ff(beam_size_um=args.beam_size or 0.0)
    else:
        # Fall back to the params file for nScans / ScanStep / BeamSize
        # when the CLI didn't provide them. CLI flags always override.
        sniffed = read_scan_geometry_from_paramfile(args.params) or {}
        n_scans = args.n_scans if args.n_scans is not None else sniffed.get("n_scans")
        scan_step = (args.scan_step if args.scan_step is not None
                     else sniffed.get("scan_step_um"))
        beam_size = args.beam_size if args.beam_size else sniffed.get("beam_size_um", 0.0)
        scan_pos_tol = (args.scan_pos_tol if args.scan_pos_tol
                        else sniffed.get("scan_pos_tol_um", 0.0))
        missing = []
        if n_scans is None:
            missing.append("--n-scans (or nScans in paramstest)")
        if scan_step is None:
            missing.append("--scan-step (or ScanStep/px in paramstest)")
        if missing:
            raise SystemExit(
                f"scan_mode=pf needs {', '.join(missing)}. "
                "Either pass on the CLI or include in the params file."
            )
        if args.n_scans is None and "n_scans" in sniffed:
            LOG.info("scan-mode=pf: using nScans=%d from %s",
                     sniffed["n_scans"], args.params)
        if args.scan_step is None and "scan_step_um" in sniffed:
            LOG.info("scan-mode=pf: using scan_step=%s µm from %s",
                     sniffed["scan_step_um"], args.params)
        scan = ScanGeometry.pf_uniform(
            n_scans=int(n_scans),
            scan_step_um=float(scan_step),
            beam_size_um=float(beam_size),
            scan_pos_tol_um=float(scan_pos_tol),
            friedel_symmetric_scan_filter=args.friedel,
        )

    refinement = RefinementConfig(
        position_mode=args.pf_refine_mode,
        solver=args.refine_solver,
        loss=args.refine_loss,
        mode=args.refine_mode,
        use_bounds=bool(args.use_bounds),
        bound_euler_deg=float(args.bound_euler_deg),
        bound_lat_abc_pct=float(args.bound_lat_abc_pct),
        bound_lat_angle_deg=float(args.bound_lat_angle_deg),
    )
    recon = ReconConfig(
        do_tomo=bool(args.do_tomo),
        method=args.recon_method,
        mlem_iter=args.mlem_iter,
        osem_subsets=args.osem_subsets,
        sino_type=args.sino_type,
        sino_source=args.sino_source,
        sino_conf_min=args.sino_conf_min,
        sino_scan_tol_um=args.sino_scan_tol,
        cull_min_size=args.cull_min_size,
    )
    fusion = FusionConfig(
        enable_bayesian=(args.recon_method == "bayesian"),
        max_ang_deg=args.max_ang_deg,
        min_conf=args.min_conf,
        cw_potts_lambda=args.cw_potts_lambda,
    )
    em = EMConfig(
        enable=bool(args.use_em),
        iter=args.em_iter,
        sigma_init=args.em_sigma_init,
        sigma_min=args.em_sigma_min,
        sigma_decay=args.em_sigma_decay,
        refine_orientations=bool(args.em_refine_orientations),
        opt_steps=args.em_opt_steps,
        lr=args.em_lr,
    )
    seeding = SeedingConfig(
        mode=args.seeding_mode,
        grains_file=args.grains_file,
        mic_file=args.mic_file,
        merged_align_method=args.merged_align_method,
        merged_ref_scan=args.merged_ref_scan,
        merged_min_nhkls=args.merged_min_nhkls,
        merged_tol_px=args.merged_tol_px,
        merged_tol_ome=args.merged_tol_ome,
    )

    diag_axes_parts = [int(x) for x in str(args.vmap_diag_axes).split(",")]
    if len(diag_axes_parts) != 2:
        raise ValueError(
            f"--vmap-diag-axes must be 'a,b' with two ints; got "
            f"{args.vmap_diag_axes!r}"
        )
    vmap = VMapConfig(
        run=bool(args.vmap_run),
        crystal_cif=args.vmap_crystal_cif,
        wavelength_A=float(args.vmap_wavelength),
        refine_V=bool(args.vmap_refine_V),
        refine_K=bool(args.vmap_refine_K),
        refine_mu=bool(args.vmap_refine_mu),
        refine_beam=bool(args.vmap_refine_beam),
        use_absorption=bool(args.vmap_use_absorption),
        element=args.vmap_element,
        max_iter=int(args.vmap_max_iter),
        loss_kind=args.vmap_loss_kind,
        tolerance=float(args.vmap_tolerance),
        v_gauge_reg=float(args.vmap_gauge_reg),
        beam_geometry=args.vmap_beam_geometry,
        soft_grain_attribution=bool(args.vmap_soft_grain_attribution),
        emit_diagnostics=bool(args.vmap_emit_diagnostics),
        diag_axes=tuple(diag_axes_parts),
    )
    grain_geometry = GrainGeometryConfig(
        run=bool(args.grain_geometry_run),
        refine_params=tuple(s.strip() for s in args.grain_geometry_refine.split(",")
                            if s.strip()),
        kind=args.grain_geometry_kind,
        max_grains=int(args.grain_geometry_max_grains),
        out_name=args.grain_geometry_out,
    )
    soft_attribution = SoftAttributionConfig(
        enable=bool(args.soft_attribution),
        profile=args.soft_profile,
        fwhm_um=float(args.soft_fwhm_um),
        tophat_fall_off_um=float(args.soft_tophat_fall_off_um),
        truncate_at_um=float(args.soft_truncate_at_um),
        omega_sigma_deg=float(args.soft_omega_sigma_deg),
    )

    cfg = PipelineConfig(
        result_dir=args.result,
        params_file=args.params,
        scan=scan,
        zarr_path=args.zarr,
        detectors_json=args.detectors,
        refinement=refinement,
        recon=recon,
        fusion=fusion,
        em=em,
        seeding=seeding,
        vmap=vmap,
        grain_geometry=grain_geometry,
        soft_attribution=soft_attribution,
        voxel_cleanup=VoxelCleanupConfig(
            run=bool(args.voxel_cleanup),
            score_threshold=float(args.voxel_cleanup_threshold),
            max_same_neighbours=int(args.voxel_cleanup_max_neighbours),
            max_iters=int(args.voxel_cleanup_max_iters),
            action=args.voxel_cleanup_action,
        ),
        layer_selection=_parse_layers(args.layers),
        machine=MachineConfig(name=args.machine, n_nodes=args.n_nodes),
        n_cpus=args.n_cpus,
        n_cpus_local=args.n_cpus_local,
        device=args.device,
        dtype=args.dtype,
        resume=args.resume,
        resume_from_stage=args.resume_from_stage,
        only_stages=list(args.only),
        skip_stages=list(args.skip),
        indexer_group_size=args.group_size,
        indexer_backend=args.indexer_backend,
        refine_backend=args.refine_backend,
        shard_gpus=args.shard_gpus,
        process_grains_mode=args.pg_mode,
        raw_dir=args.raw_dir,
        grains_file=args.ff_grains_file,
        nf_result_dir=args.nf_result_dir,
        num_frame_chunks=args.num_frame_chunks,
        preproc_thresh=args.preproc_thresh,
        convert_files=args.convert_files,
        file_name=args.file_name,
        num_files_per_scan=args.num_files_per_scan,
        normalize_intensities=args.normalize_intensities,
        do_peak_search=args.do_peak_search,
        one_sol_per_vox=bool(args.one_sol_per_vox),
        peak_fit_gpu=args.peak_fit_gpu,
        run_sr=args.run_sr,
        srfac=args.srfac,
        sr_config_path=args.sr_config,
        save_sr_patches=args.save_sr_patches,
        save_frame_good_coords=args.save_frame_good_coords,
        generate_h5=args.generate_h5,
        skip_validation=args.skip_validation,
        strict_validation=args.strict_validation,
        log_level=args.log_level,
    )
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    configure_logging(level=getattr(logging, args.log_level))
    # Resolve "auto" knobs in-place so build_config sees concrete values.
    args.dtype = _resolve_dtype(args.device, args.dtype)
    args.shard_gpus = _resolve_shard_gpus(args.device, args.shard_gpus)
    args.group_size = _resolve_group_size(
        args.device, args.shard_gpus, args.group_size, params_file=args.params,
    )
    args.cpu_shards = _resolve_cpu_shards(args.device, args.n_cpus, args.cpu_shards)
    cfg = build_config(args)
    LOG.info("midas-pipeline run: scan_mode=%s, layers=%s, device=%s, "
             "dtype=%s, group_size=%d, shard_gpus=%s, cpu_shards=%d",
             cfg.scan.scan_mode, args.layers, cfg.device, cfg.dtype,
             cfg.indexer_group_size, cfg.shard_gpus, args.cpu_shards)
    pipeline = Pipeline(cfg)
    if getattr(args, "batch", False):
        if cfg.scan.scan_mode != "ff":
            raise ValueError("--batch is only valid for --scan-mode ff")
        from .discovery import run_batch
        run_batch(pipeline, args)
    else:
        results = pipeline.run()
        LOG.info("done. layers processed: %d", len(results))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    configure_logging()
    rd = Path(args.result_dir)
    layer_dirs = sorted(rd.glob("LayerNr_*"))
    if not layer_dirs:
        # Possibly PF mode where state.h5 lives at the top level
        if (rd / "midas_state.h5").exists():
            layer_dirs = [rd]
    report = {}
    for ld in layer_dirs:
        store = ProvenanceStore(ld)
        report[ld.name] = store.all_stages()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for layer_name, stages in report.items():
            print(f"\n{layer_name}:")
            for sname, rec in stages.items():
                print(f"  {sname:<22} {rec.get('status'):<10} {rec.get('duration_s'):.2f}s")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    print("resume: not yet implemented in P1 scaffold; "
          "use `run --resume=from --from <stage>` instead.", file=sys.stderr)
    return 2


def _cmd_reprocess(args: argparse.Namespace) -> int:
    from .reprocess import reprocess_dir
    reprocess_dir(
        Path(args.result_dir),
        n_cpus=args.n_cpus,
        device=args.device,
        dtype=args.dtype,
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    configure_logging()
    store = ProvenanceStore(args.layer_dir)
    stages = store.all_stages()
    if args.json:
        print(json.dumps(stages, indent=2, default=str))
    else:
        print(f"{args.layer_dir}:")
        for sname, rec in stages.items():
            print(f"  {sname:<22} {rec.get('status'):<10} {rec.get('duration_s'):.2f}s")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Forward-simulate a synthetic dataset.

    Single-detector: one ``.MIDAS.zip``. Multi-detector (n_detectors>1):
    one zip per detector + a ``detectors.json`` ready for ``run --detectors``.
    """
    if args.n_detectors > 1:
        if args.mode == "diffract_pinwheel":
            from .testing import generate_pinwheel_synthetic_dataset
            zips, dets_json = generate_pinwheel_synthetic_dataset(
                out_dir=Path(args.out),
                n_grains=args.n_grains,
                seed=args.seed,
                n_panels=args.n_detectors,
                use_hydra_geometry=not args.no_hydra_geometry,
                device=args.device,
            )
        else:
            from .testing import generate_multidet_synthetic_dataset
            zips, dets_json = generate_multidet_synthetic_dataset(
                out_dir=Path(args.out),
                params_template=Path(args.params),
                n_grains=args.n_grains,
                seed=args.seed,
                n_cpus=args.n_cpus,
                n_detectors=args.n_detectors,
            )
        for z in zips:
            print(f"  zip: {z}")
        print(f"detectors.json: {dets_json}")
        return 0

    from .testing import generate_synthetic_dataset
    out_zip = generate_synthetic_dataset(
        out_dir=Path(args.out),
        params_template=Path(args.params),
        n_grains=args.n_grains,
        seed=args.seed,
        n_cpus=args.n_cpus,
    )
    print(f"synthetic dataset: {out_zip}")
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    print("seed: standalone merged-FF seeding sub-pipeline lands in P7.",
          file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_DISPATCH = {
    "run": _cmd_run,
    "status": _cmd_status,
    "resume": _cmd_resume,
    "reprocess": _cmd_reprocess,
    "inspect": _cmd_inspect,
    "simulate": _cmd_simulate,
    "seed": _cmd_seed,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH[args.cmd]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
