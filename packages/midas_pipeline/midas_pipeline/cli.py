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
    ScanMode,
    SeedingConfig,
    SeedingMode,
    SinoSource,
    SinoType,
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
    run.add_argument("--dtype", choices=["float32", "float64"], default="float64")

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
    run.add_argument("--refine-loss", choices=["pixel", "angular", "internal_angle"],
                     default="pixel")
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
    run.add_argument("--group-size", type=int, default=4,
                     help="indexer seed group size (default 4 for fp64 safety)")
    run.add_argument("--shard-gpus", default=None,
                     help="comma-separated CUDA indices for multi-GPU indexing")

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
    sim.add_argument("--n-detectors", type=int, default=1)

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
        shard_gpus=args.shard_gpus,
        process_grains_mode=args.pg_mode,
        raw_dir=args.raw_dir,
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
    cfg = build_config(args)
    LOG.info("midas-pipeline run: scan_mode=%s, layers=%s, device=%s",
             cfg.scan.scan_mode, args.layers, cfg.device)
    pipeline = Pipeline(cfg)
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
    print("reprocess: not yet implemented in P1 scaffold "
          "(lands when consolidation_pf goes live).", file=sys.stderr)
    return 2


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
    print("simulate: P1 scaffold — wraps tests/test_pf_hedm.py harness "
          "in a later phase. Use the existing test harness for now.",
          file=sys.stderr)
    return 2


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
