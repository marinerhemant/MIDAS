"""``midas-ff-pipeline`` console script.

Subcommands:
  run        — execute the full pipeline
  status     — show per-layer / per-stage resume state
  resume     — re-run starting from a named stage
  inspect    — dump a layer's stage timings + n_grains
  simulate   — generate a synthetic 50-grain Au dataset for testing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import LayerSelection, MachineConfig, PipelineConfig
from .pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midas-ff-pipeline",
        description="End-to-end pure-Python FF-HEDM workflow (1-N detectors).",
    )
    parser.add_argument("--version", action="version",
                        version=f"midas-ff-pipeline {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="run the pipeline end-to-end")
    p_run.add_argument("--params", required=True,
                       help="path to Parameters.txt or paramstest.txt")
    p_run.add_argument("--result", required=True,
                       help="result directory; LayerNr_<N> subdirs created here")
    p_run.add_argument("--zarr", default=None,
                       help="path to .MIDAS.zip (single-detector); overrides params discovery")
    p_run.add_argument("--detectors", default=None,
                       help="optional detectors.json for multi-detector runs")
    p_run.add_argument("--layers", default="1-1",
                       help="layer range, e.g. '1-1', '1-5'")
    p_run.add_argument("--n-cpus", type=int, default=16)
    p_run.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p_run.add_argument("--dtype", default="auto",
                       choices=["auto", "float32", "float64"],
                       help="auto = float32 on cuda/mps, float64 on cpu "
                            "(matches the rest of MIDAS production defaults)")
    p_run.add_argument("--resume", default="auto", choices=["none", "auto", "from"])
    p_run.add_argument("--from", dest="resume_from", default=None,
                       help="when --resume=from, the stage to resume from")
    p_run.add_argument("--only", action="append", default=[],
                       help="run only this stage (repeatable)")
    p_run.add_argument("--skip", action="append", default=[],
                       help="skip this stage (repeatable)")
    p_run.add_argument("--solver", default="lbfgs",
                       choices=["lbfgs", "lm", "nelder_mead", "adam", "lm_batched"],
                       help="midas-fit-grain solver")
    p_run.add_argument("--loss", default="pixel",
                       choices=["pixel", "angular", "internal_angle"],
                       help="midas-fit-grain residual definition")
    p_run.add_argument("--mode", default="",
                       choices=["", "iterative", "all_at_once"],
                       help="midas-fit-grain mode")
    p_run.add_argument("--group-size", default="auto",
                       help="midas-index seed group size. 'auto' picks based "
                            "on the smallest visible-GPU memory across the "
                            "shard set: ≥70GB→8, ≥32GB→4, ≥16GB→2, else 1. "
                            "Pass an integer to override.")
    p_run.add_argument("--shard-gpus", default="auto",
                       help="comma-separated CUDA indices to fan the indexer "
                            "across (e.g. '0,1' for two GPUs). Each shard "
                            "handles a disjoint slice of seeds; results land "
                            "in the same IndexBest.bin via pwrite. Refinement "
                            "and downstream stages stay single-GPU. "
                            "'auto' uses every visible CUDA device when "
                            "--device=cuda; 'none' or '' disables sharding.")
    p_run.add_argument("--cpu-shards", default="auto",
                       help="number of midas-index processes to run in "
                            "parallel on CPU. 'auto' picks max(1, n_cpus // "
                            "16) — intra-op threading stops scaling past ~16 "
                            "threads on small per-seed ops, so multi-process "
                            "sharding is faster than a single 96-thread "
                            "process. Ignored on GPU. '1' or '0' disables.")
    p_run.add_argument("--pg-mode", default="spot_aware",
                       choices=["spot_aware", "legacy", "paper_claim"],
                       help="midas_process_grains mode")
    p_run.add_argument("--machine", default="local")
    p_run.add_argument("--n-nodes", type=int, default=1)
    p_run.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # --- gap #5: raw-dir override ---
    p_run.add_argument("--raw-dir", default=None,
                       help="override RawFolder + Dark dir in the parameter file "
                            "(equivalent to ff_MIDAS.py -rawDir).")
    # --- gaps #3/#4: seed grains ---
    p_run.add_argument("--grains-file", default=None,
                       help="explicit GrainsFile (Grains.csv) to seed indexer + "
                            "refiner. Sets MinNrSpots=1 in paramstest.")
    p_run.add_argument("--nf-result-dir", default=None,
                       help="directory of NF results; per layer N looks for "
                            "GrainsLayer{N}.csv as the seed (overrides --grains-file).")
    # --- gap #10: midas-params preflight ---
    p_run.add_argument("--skip-validation", action="store_true",
                       help="skip midas-params preflight on the parameter file.")
    p_run.add_argument("--strict-validation", action="store_true",
                       help="exit on midas-params validation errors (default: warn).")
    # --- gap #1: ingestion controls (zip_convert lands in pass 2) ---
    p_run.add_argument("--num-frame-chunks", type=int, default=-1,
                       help="frame chunking for raw→zarr conversion. -1 → auto.")
    p_run.add_argument("--preproc-thresh", type=int, default=-1,
                       help="dark-subtraction threshold for raw→zarr. -1 → auto.")
    p_run.add_argument("--no-convert", action="store_true",
                       help="do not run raw→zarr; assume zarr exists.")
    p_run.add_argument("--file-name", default=None,
                       help="single raw file override (sets layer_nr from FileNr).")
    p_run.add_argument("--num-files-per-scan", type=int, default=1)
    # --- gap #11: consolidated HDF5 ---
    p_run.add_argument("--generate-h5", action="store_true",
                       help="emit a consolidated grain↔peak HDF5 after process_grains.")
    # --- gap #9: sr-midas (lands in pass 4) ---
    p_run.add_argument("--run-sr", action="store_true",
                       help="run sr-midas super-resolution peak search (requires sr-midas).")
    p_run.add_argument("--srfac", type=int, default=8,
                       help="sr-midas super-resolution factor.")
    p_run.add_argument("--sr-config", default="auto",
                       help="path to sr-midas config or 'auto'.")
    p_run.add_argument("--save-sr-patches", action="store_true")
    p_run.add_argument("--save-frame-good-coords", action="store_true")
    # --- gap #2: batch mode (driver lands in pass 2) ---
    p_run.add_argument("--batch", action="store_true",
                       help="auto-detect raw files in RawFolder; loops over file-nr range.")

    p_run.set_defaults(func=_cmd_run)

    # --- status ---
    p_status = sub.add_parser("status", help="show resume state of a run dir")
    p_status.add_argument("result_dir", help="result directory of a previous run")
    p_status.add_argument("--layers", default=None,
                          help="layer range (default: discover from disk)")
    p_status.add_argument("--json", action="store_true",
                          help="dump structured JSON instead of human-readable text")
    p_status.set_defaults(func=_cmd_status)

    # --- resume ---
    p_resume = sub.add_parser("resume", help="resume a run from a stage")
    p_resume.add_argument("result_dir")
    p_resume.add_argument("--params", default=None,
                          help="paramstest.txt to use; defaults to <result_dir>/<LayerNr_*>/paramstest.txt")
    p_resume.add_argument("--detectors", default=None)
    p_resume.add_argument("--from", dest="from_stage", required=True,
                          help="stage to resume from")
    p_resume.add_argument("--layers", default="1-1")
    p_resume.add_argument("--n-cpus", type=int, default=16)
    p_resume.add_argument("--device", default="cuda")
    p_resume.add_argument("--dtype", default="float64")
    p_resume.set_defaults(func=_cmd_resume)

    # --- reprocess ---
    p_reprocess = sub.add_parser(
        "reprocess",
        help="re-run merge_overlaps + consolidation on completed result dirs.",
    )
    p_reprocess.add_argument("result_dir",
                             help="result directory of a previous run "
                                  "(or a parent of LayerNr_*).")
    p_reprocess.add_argument("--device", default="cuda",
                             choices=["cpu", "cuda", "mps"])
    p_reprocess.add_argument("--dtype", default="float64",
                             choices=["float32", "float64"])
    p_reprocess.add_argument("--n-cpus", type=int, default=8)
    p_reprocess.set_defaults(func=_cmd_reprocess)

    # --- inspect ---
    p_inspect = sub.add_parser("inspect", help="dump per-stage timings + grain count")
    p_inspect.add_argument("layer_dir", help="path to a LayerNr_<N> directory")
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(func=_cmd_inspect)

    # --- simulate ---
    p_sim = sub.add_parser("simulate",
                           help="forward-simulate a synthetic dataset for testing")
    p_sim.add_argument("--out", required=True, help="output directory")
    p_sim.add_argument("--params", required=True, help="Parameters.txt template")
    p_sim.add_argument("--n-grains", type=int, default=50)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--n-cpus", type=int, default=8)
    p_sim.add_argument("--n-detectors", type=int, default=1,
                       help="number of detectors. >1 → multi-det layout")
    p_sim.add_argument("--mode", default="ff_compressed",
                       choices=["ff_compressed", "diffract_pinwheel"],
                       help="forward-sim backend: 'ff_compressed' (legacy "
                            "ForwardSimulationCompressed, single-panel per call) "
                            "or 'diffract_pinwheel' (midas_diffract HEDMForwardModel "
                            "with multi_mode='panel'; required for true pinwheel).")
    p_sim.add_argument("--device", default="cuda",
                       help="GPU device for diffract_pinwheel (cuda / cuda:0 / cpu)")
    p_sim.add_argument("--no-hydra-geometry", action="store_true",
                       help="for diffract_pinwheel: use simplified pinwheel "
                            "(shared BC, no Lsd jitter) instead of hydra-real geometry")
    p_sim.set_defaults(func=_cmd_simulate)

    return parser


# ---- subcommand handlers ----


def _parse_layer_range(spec: str) -> LayerSelection:
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        return LayerSelection(start=int(start_s), end=int(end_s))
    n = int(spec)
    return LayerSelection(start=n, end=n)


def _resolve_dtype(device: str, dtype_arg: str) -> str:
    """``auto`` → float32 on cuda/mps (matches the rest of MIDAS production),
    float64 on cpu (where parity tests run). Explicit values pass through.
    """
    if dtype_arg != "auto":
        return dtype_arg
    return "float32" if device in ("cuda", "mps") else "float64"


def _resolve_shard_gpus(device: str, shard_arg: str) -> str | None:
    """``auto`` → all visible CUDA devices when device=cuda. ``none``/empty
    disables sharding. Explicit comma list (e.g. ``'0,1'``) passes through.
    """
    if shard_arg in ("none", "None", ""):
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
    """``auto`` → ``max(1, n_cpus // 16)`` on CPU, else 1.

    Intra-op threading scales poorly past ~16 threads on small per-seed
    ops; multi-process sharding (each shard ``set_num_threads(n_cpus // N)``)
    keeps each shard in the well-scaled regime while still using every core.
    """
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


_BASELINE_N_RINGS = 8       # park22 FCC reference
_BASELINE_N_OMEGA = 720     # park22 reference scan length


def _count_dataset_density(params_file: str | None) -> tuple[int, int]:
    """Parse paramstest to return ``(n_rings, n_omega_steps)``.

    The indexer's per-seed matching tensor (and therefore peak-group memory)
    scales roughly with ``n_rings × n_omega_steps`` — more rings means a
    bigger theoretical-spot pool per candidate orientation, and longer scans
    mean more observed spots per ring. Returns ``(0, 0)`` on any failure;
    callers treat that as ``no density information available``.
    """
    if params_file is None:
        return 0, 0
    try:
        with open(params_file) as f:
            text = f.read()
    except OSError:
        return 0, 0
    # Ring-list keys differ across paramstest variants:
    #   midas-ff-pipeline params.txt → ``RingThresh <ring> <thresh>``
    #   indexer-style paramstest.txt → ``RingNumbers <ring>``
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
    # Pick the largest count across ring-list keys (one file may use only one;
    # if multiple appear, the largest is the canonical refinement set).
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
    """``auto`` → pick from the smallest visible-GPU memory across the shard
    set, then *down*-scale for datasets denser than the calibration baseline.

    Memory-tier baseline (park22 FCC, fp32, gs=4 → ~47 GB peak):

      ≥ 70 GB (H100, A100-80) → 8
      ≥ 32 GB (A6000, A100-40) → 4
      ≥ 16 GB (V100, A5000)    → 2
      < 16 GB                  → 1

    Density factor (only down-scales, never up):

      density = max(1, (n_rings / 8) × (n_omega_steps / 720))
      group_size = max(1, baseline // density)

    Where ``n_rings`` is the count of ``RingNumbers`` lines in the
    paramstest and ``n_omega_steps`` is ``|OmegaEnd − OmegaStart| /
    OmegaStep``. Datasets at the park22 baseline (8 rings, 720 frames) are
    unchanged; Ti-7Al-class datasets (12 rings, 1440 frames → density 3)
    get one fp64 group at gs=4-tier GPUs instead of OOM'ing.

    Falls back to 4 (the FF pipeline's previous default) if memory probing
    fails or device is non-cuda. Explicit integer passes through.
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


def _cmd_run(args: argparse.Namespace) -> int:
    # --- midas-params preflight (gap #10) — soft dependency ---
    import logging
    _logger = logging.getLogger("midas_ff_pipeline")
    try:
        from midas_params.hook import preflight_validate, resolve_runtime_defaults
    except ImportError:
        preflight_validate = None
        resolve_runtime_defaults = None

    if preflight_validate is not None:
        ok = preflight_validate(
            param_file=args.params,
            pipeline="ff",
            skip=args.skip_validation,
            strict=args.strict_validation,
            logger=_logger,
        )
        if not ok:
            return 1

    n_chunks = args.num_frame_chunks
    preproc = args.preproc_thresh
    if resolve_runtime_defaults is not None:
        try:
            n_chunks, preproc = resolve_runtime_defaults(
                param_file=args.params,
                num_frame_chunks=n_chunks,
                pre_proc_thresh=preproc,
                n_cpus=args.n_cpus,
                logger=_logger,
            )
        except Exception:
            pass

    # Resolve auto-detect knobs.
    resolved_dtype = _resolve_dtype(args.device, args.dtype)
    resolved_shard = _resolve_shard_gpus(args.device, args.shard_gpus)
    resolved_gs = _resolve_group_size(args.device, resolved_shard, args.group_size,
                                      params_file=args.params)
    resolved_cpu_shards = _resolve_cpu_shards(args.device, args.n_cpus, args.cpu_shards)
    if args.dtype == "auto":
        _logger.info("auto: dtype=%s (device=%s)", resolved_dtype, args.device)
    if args.shard_gpus == "auto" and resolved_shard:
        _logger.info("auto: shard-gpus=%s", resolved_shard)
    if args.group_size == "auto":
        _logger.info("auto: group-size=%d", resolved_gs)
    if args.cpu_shards == "auto" and resolved_cpu_shards > 1:
        _logger.info("auto: cpu-shards=%d (n_cpus=%d, %d threads/shard)",
                     resolved_cpu_shards, args.n_cpus,
                     args.n_cpus // resolved_cpu_shards)

    config = PipelineConfig(
        result_dir=args.result,
        params_file=args.params,
        zarr_path=args.zarr,
        detectors_json=args.detectors,
        n_cpus=args.n_cpus,
        device=args.device,
        dtype=resolved_dtype,
        layer_selection=_parse_layer_range(args.layers),
        machine=MachineConfig(name=args.machine, n_nodes=args.n_nodes),
        resume=args.resume,
        resume_from_stage=args.resume_from,
        only_stages=args.only,
        skip_stages=args.skip,
        refine_solver=args.solver,
        refine_loss=args.loss,
        refine_mode=args.mode,
        indexer_group_size=resolved_gs,
        shard_gpus=resolved_shard,
        cpu_shards=resolved_cpu_shards,
        process_grains_mode=args.pg_mode,
        log_level=args.log_level,
        # gap #5
        raw_dir=args.raw_dir,
        # gaps #3/#4
        grains_file=args.grains_file,
        nf_result_dir=args.nf_result_dir,
        # gap #10
        skip_validation=args.skip_validation,
        strict_validation=args.strict_validation,
        # gap #1
        num_frame_chunks=n_chunks,
        preproc_thresh=preproc,
        convert_files=not args.no_convert,
        file_name=args.file_name,
        num_files_per_scan=args.num_files_per_scan,
        # gap #11
        generate_h5=args.generate_h5,
        # gap #9
        run_sr=args.run_sr,
        srfac=args.srfac,
        sr_config_path=args.sr_config,
        save_sr_patches=args.save_sr_patches,
        save_frame_good_coords=args.save_frame_good_coords,
    )
    pipe = Pipeline(config=config)
    if args.batch:
        from .discovery import run_batch
        run_batch(pipe, args)
    else:
        pipe.run()
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    result_dir = Path(args.result_dir)
    if args.layers:
        layer_sel = _parse_layer_range(args.layers)
        layer_nrs = layer_sel.layers()
    else:
        # discover LayerNr_<N> dirs from disk
        layer_nrs = sorted(
            int(p.name.split("_")[-1])
            for p in result_dir.glob("LayerNr_*") if p.is_dir()
        )
        if not layer_nrs:
            print(f"no LayerNr_* dirs in {result_dir}", file=sys.stderr)
            return 2
    out = {"result_dir": str(result_dir), "layers": []}
    from .provenance import ProvenanceStore
    for ln in layer_nrs:
        layer_dir = result_dir / f"LayerNr_{ln}"
        store = ProvenanceStore(layer_dir)
        stages = store.all_stages()
        out["layers"].append({
            "layer_nr": ln,
            "layer_dir": str(layer_dir),
            "stages": stages,
        })
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    # Human-readable table.
    for layer in out["layers"]:
        print(f"\nLayer {layer['layer_nr']}: {layer['layer_dir']}")
        if not layer["stages"]:
            print("  (no provenance — never run)")
            continue
        for name, info in layer["stages"].items():
            status = info.get("status", "?")
            dur = info.get("duration_s", 0.0)
            metrics = info.get("metrics") or {}
            extras = " ".join(f"{k}={v}" for k, v in metrics.items()
                              if not isinstance(v, (list, dict)))
            print(f"  {name:18s} {status:10s} {dur:6.2f}s  {extras}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    layer_sel = _parse_layer_range(args.layers)
    # Find a paramstest if not supplied — pick the first layer's.
    params = args.params
    if not params:
        first = Path(args.result_dir) / f"LayerNr_{layer_sel.start}" / "paramstest.txt"
        if not first.exists():
            print(f"--params not given and {first} not found", file=sys.stderr)
            return 2
        params = str(first)
    config = PipelineConfig(
        result_dir=args.result_dir,
        params_file=params,
        detectors_json=args.detectors,
        n_cpus=args.n_cpus,
        device=args.device,
        dtype=args.dtype,
        layer_selection=layer_sel,
        resume="from",
        resume_from_stage=args.from_stage,
    )
    pipe = Pipeline(config=config)
    pipe.run()
    return 0


def _cmd_reprocess(args: argparse.Namespace) -> int:
    """Re-run merge_overlaps + consolidation on existing result dirs.

    Mirrors ``ff_MIDAS.py -reprocess 1``. Operates on every ``LayerNr_*``
    subdirectory of ``args.result_dir`` (or directly on ``args.result_dir``
    if it itself looks like a layer dir).
    """
    from .reprocess import reprocess_dir
    root = Path(args.result_dir).resolve()
    if not root.exists():
        print(f"result_dir not found: {root}", file=sys.stderr)
        return 2

    layer_dirs = sorted(p for p in root.glob("LayerNr_*") if p.is_dir())
    if not layer_dirs:
        # Treat the path itself as the layer dir.
        layer_dirs = [root]

    rc = 0
    for ld in layer_dirs:
        try:
            reprocess_dir(ld, n_cpus=args.n_cpus,
                          device=args.device, dtype=args.dtype)
        except Exception as e:
            print(f"reprocess failed for {ld}: {e}", file=sys.stderr)
            rc = 1
    return rc


def _cmd_inspect(args: argparse.Namespace) -> int:
    layer_dir = Path(args.layer_dir)
    from .provenance import ProvenanceStore
    store = ProvenanceStore(layer_dir)
    stages = store.all_stages()
    if args.json:
        print(json.dumps({"layer_dir": str(layer_dir), "stages": stages}, indent=2,
                         default=str))
        return 0
    if not stages:
        print(f"no provenance found in {layer_dir / 'midas_state.h5'}")
        return 1
    grains_csv = layer_dir / "Grains.csv"
    n_grains = 0
    if grains_csv.exists():
        with grains_csv.open() as fp:
            head = fp.readline()
            if head.startswith("%NumGrains"):
                try:
                    n_grains = int(head.split()[1])
                except ValueError:
                    pass
    total = sum(s.get("duration_s", 0.0) for s in stages.values())
    print(f"\n{layer_dir}: {n_grains} grains, total {total:.1f}s")
    for name, info in stages.items():
        status = info.get("status", "?")
        dur = info.get("duration_s", 0.0)
        metrics = info.get("metrics") or {}
        extras = " ".join(f"{k}={v}" for k, v in metrics.items()
                          if not isinstance(v, (list, dict)))
        print(f"  {name:18s} {status:10s} {dur:6.2f}s  {extras}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Forward-simulate a synthetic dataset using ForwardSimulationCompressed.

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


# ---- main ----


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
