"""CLI: ``midas-process-grains`` (and ``python -m midas_process_grains``).

Mirrors the C ``ProcessGrains`` invocation pattern (single positional arg:
the parameter file path) with optional flags to override mode, device,
dtype, and a couple of merge knobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-process-grains"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _build_parser() -> argparse.ArgumentParser:
    p = _midas_make_parser(
        prog="midas-process-grains",
        description=(
            "Pure-Python FF-HEDM grain-determination + strain pipeline "
            "(drop-in for ProcessGrains)."
        ),
    )
    p.add_argument(
        "param_file",
        type=Path,
        help="Path to paramstest.txt (the same file IndexerOMP/FitPosOrStrains "
             "consumed for this run).",
    )
    p.add_argument(
        "num_procs", type=int, nargs="?", default=1,
        help="CPU thread count (used only on cpu device). Default 1.",
    )
    p.add_argument(
        # 'spot_aware' is DISABLED and removed from the choices; it was also
        # the DEFAULT here, so anything invoking this CLI without --mode got
        # it. See the adjudication in the help text below.
        "--mode", choices=("legacy", "paper_claim", "adaptive",
                           "physics", "c_parity"),
        default="c_parity",
        help="Pipeline mode. 'adaptive' derives the misori threshold from the "
             "antimode of the pairwise-misorientation histogram of the alive "
             "candidates (no hand-tuned MisoriTol). 'physics' enables the v4 "
             "physics-bounded pipeline: per-candidate seed-(h,k,l) recovery "
             "(FZ-canonical), orientation-aware expected-hkl prediction, "
             "variant-constrained Hungarian split for over-merged clusters, "
             "twin + sub-grain labeling, NNLS grain-size recompute on splits, "
             "and a hierarchical GrainsV4.csv emitter. "
                  "Default 'c_parity' is a bit-level replica of the C ProcessGrains "
             "pipeline. 'spot_aware' is DISABLED and rejected: against EBSD on "
             "shade_LSHR only 7.2%% of the 691 grains it adds over c_parity had "
             "a partner (vs 80.4%% for the shared population), and on 20-ID "
             "alumina it returned 1652 grains against c_parity's 533 while "
             "placing 4.1%% of them outside the physical sample.",
    )
    p.add_argument(
        "--min-nr-spots", type=int, default=None,
        help="MinNrSpots threshold (Stage 1 cluster-size cutoff). C ProcessGrains "
             "default is 1; the original peakfit_hard run used 3.",
    )
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    p.add_argument("--dtype", choices=("float32", "float64"), default=None)
    p.add_argument("--misori-tol", type=float, default=None,
                   help="Override the Phase 1 misorientation tolerance (degrees).")
    p.add_argument(
        "--merge-primitive", choices=("misori", "forward_predict", "consensus_anchor"),
        default="misori",
        help="(mode=physics only) Pass-1 clustering primitive. 'misori' (default) "
             "uses the smart-antimode pairwise-misorientation threshold. "
             "'forward_predict' uses midas_diffract to predict each candidate's "
             "ring-spots and merges only when same-variant evidence agrees on "
             "K+ spots AND no cross-variant disagreement — symmetric by "
             "construction, immune to refiner-asymmetric matched lists, and "
             "robust against the chain-fusion giant-component pathology on "
             "heavily-twinned datasets.",
    )
    p.add_argument(
        "--k-agree", type=int, default=None,
        help="(mode=physics + --merge-primitive=forward_predict) Same-variant "
             "agreement threshold for a forward-predict merge edge. None = "
             "auto-select (smallest K such that the largest connected component "
             "is below max(100, n_alive/100)). Typical: K=4 for cubic-FCC, K=5 "
             "for heavily-twinned LMO/oxide samples.",
    )
    p.add_argument(
        "--consensus-qmin", type=int, default=6,
        help="(--merge-primitive=consensus_anchor) Minimum tight-snap quality for "
             "a candidate to SEED a grain. Higher = stricter (fewer spurious "
             "grains, lower recovery). Default 6.",
    )
    p.add_argument(
        "--consensus-tau-deg", type=float, default=1.0,
        help="(--merge-primitive=consensus_anchor) Misorientation radius (deg) "
             "within which an anchor absorbs sibling candidates. Default 1.0.",
    )
    p.add_argument(
        "--strain-method",
        choices=(
            "kenesei", "kenesei_unbounded", "fable_beaudoin", "both",
            # backwards-compat aliases (resolved in params.validated())
            "lstsq", "lattice",
        ),
        default=None,
        help="Per-grain strain solver. Default: kenesei (bounded ±0.01, "
             "matches C reference). Use fable_beaudoin for the lattice-"
             "parameter route, or both to emit each.",
    )
    p.add_argument("--material", default=None,
                   help="Material name for stiffness lookup (e.g. Cu, Ni, Fe).")
    p.add_argument("--stiffness-file", type=Path, default=None,
                   help="Path to a 6×6 stiffness matrix (CSV/TXT/NPY).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write outputs. Default: param-file directory.")
    p.add_argument("--no-h5", action="store_true",
                   help="Skip writing data_consolidated.h5.")
    p.add_argument("--no-diagnostics-h5", action="store_true",
                   help="Skip writing processgrains_diagnostics.h5 (the "
                        "signed per-spot residual sidecar every downstream "
                        "diagnostic reads). Honoured by every mode except "
                        "'physics', which never builds the residual table.")
    p.add_argument("--max-seeds", type=int, default=None,
                   help="Process only the first N alive seeds (smoke / dev).")
    p.add_argument("--nnls-volume", action="store_true",
                   help="After grain emission, run joint non-negative least "
                        "squares to correct GrainRadius for shared-spot "
                        "intensity attribution between twin partners and "
                        "overlapping grains. See compute/volume_nnls.py. "
                        "Off by default for byte-level compatibility with C.")
    p.add_argument("--physical-K", action="store_true",
                   help="Use physical K(ring) = mult·|F|²·LP·DWF instead of "
                        "the empirical median-intensity K for NNLS volume "
                        "correction. Theoretically more rigorous; on dense "
                        "datasets the two agree to ~0.3% at the population "
                        "level. Implies --nnls-volume.")
    p.add_argument("--version", action="version",
                   version=f"midas-process-grains {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns process exit code."""
    args = _build_parser().parse_args(argv)

    from .device import apply_cpu_threads, resolve_device, resolve_dtype
    from .pipeline import ProcessGrains

    # ── physics mode: dispatch to the v4 physics-bounded pipeline ──────────
    if args.mode == "physics":
        from .v4_pipeline import run_v4_pipeline
        run_dir = args.param_file.parent
        out_dir = args.out_dir if args.out_dir is not None else run_dir
        # Map --min-nr-spots onto the v4 min_n_unique_hkls filter — they serve
        # the same purpose: reject single-spot indexing artifacts. When the
        # flag is absent, take the user's MinNrSpots from the parameter file
        # rather than substituting a literal; hardcoding it here is the same
        # defect that made c_parity return 23138 grains instead of 6157 on the
        # datasetA Ni layer, because the pipeline propagates MinNrSpots into
        # the file precisely so it will be honoured.
        min_unique = args.min_nr_spots
        if min_unique is None:
            from .params import read_paramstest_pg
            try:
                _p = read_paramstest_pg(args.param_file)
                min_unique = (int(_p.MinNrSpots)
                              if "MinNrSpots" in getattr(_p, "raw", {}) else 2)
            except Exception:
                min_unique = 2
        paths = run_v4_pipeline(
            layer_dir=run_dir,
            out_dir=out_dir,
            paramstest=args.param_file,   # the file the user actually named

            trust_scheme="strict",
            min_n_unique_hkls=min_unique,
            merge_primitive=args.merge_primitive,
            k_agree=args.k_agree,
            consensus_qmin=args.consensus_qmin,
            consensus_tau_deg=args.consensus_tau_deg,
        )
        print(
            f"midas-process-grains {__version__} (mode=physics): "
            f"wrote {paths['leaf']}",
            file=sys.stderr,
        )
        return 0

    # ── c_parity mode: dispatch to the C-replica pipeline and return ────────
    if args.mode == "c_parity":
        from .compute.c_parity_run import run_c_parity_pipeline_from_disk
        run_dir = args.param_file.parent
        out_dir = args.out_dir if args.out_dir is not None else run_dir
        device = resolve_device(args.device)
        # torch device strings: "cpu" / "cuda" / "cuda:0" / "mps"
        device_str = str(device) if not hasattr(device, "type") else (
            device.type if device.index is None else f"{device.type}:{device.index}"
        )
        apply_cpu_threads(args.num_procs, device)
        run_c_parity_pipeline_from_disk(
            run_dir=run_dir,
            out_dir=out_dir,
            paramstest=args.param_file,
            # None → read MinNrSpots from the parameter file. Hardcoding 1 here
            # meant the user's own `MinNrSpots 3` was propagated into the file
            # by the pipeline and then ignored, and only an explicit
            # --min-nr-spots on the command line had any effect.
            min_nr_spots=args.min_nr_spots,
            write_diagnostics=not args.no_diagnostics_h5,
            device=device_str,
        )
        return 0

    pg = ProcessGrains.from_param_file(
        args.param_file,
        device=args.device,
        dtype=args.dtype,
    )
    apply_cpu_threads(args.num_procs, pg.device)

    # CLI overrides on top of paramstest.
    if args.misori_tol is not None:
        pg.params.MisoriTol = float(args.misori_tol)
    if args.strain_method is not None:
        pg.params.StrainMethod = args.strain_method
    if args.material is not None:
        pg.params.MaterialName = args.material
    if args.stiffness_file is not None:
        pg.params.StiffnessFile = str(args.stiffness_file)
    if args.nnls_volume or args.physical_K:
        pg.params.NnlsVolume = True
    if args.physical_K:
        pg.params.PhysicalK = True
    pg.params = pg.params.validated()

    if args.max_seeds is not None:
        pg.params.raw["__max_seeds__"] = [str(args.max_seeds)]

    result = pg.run(mode=args.mode)
    out_dir = args.out_dir if args.out_dir is not None else pg.run_dir
    result.write(
        out_dir,
        h5=not args.no_h5,
        diagnostics_h5=not args.no_diagnostics_h5,
    )
    print(
        f"midas-process-grains {__version__}: "
        f"{result.n_grains} grains written to {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
