#!/usr/bin/env python
"""
Compare iterative vs all-at-once refinement in FitPosOrStrainsOMP.

Runs the FF-HEDM pipeline twice (iterative refinement, then all-at-once with
G-vector angular minimization) and compares both against the GrainsSim.csv
ground truth.  Optionally adds detector noise via SimNoiseSigma.

Usage:
    python tests/compare_fit_modes.py -nCPUs 8 --noiseSigma 200
"""
import argparse
import os
import sys
import subprocess
import shutil
import numpy as np
from pathlib import Path

# ── Import helpers from the main FF test ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ff_hedm import (
    parse_parameter_file,
    create_testing_env,
    run_forward_simulation,
    enrich_zarr_metadata,
    parse_grains_csv,
    PRESERVE,
)

# Import symmetry-aware misorientation from MIDAS utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from calcMiso import GetMisOrientationAngleOM
from generate_grains import generate_grains_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare iterative vs all-at-once FF-HEDM refinement"
    )
    parser.add_argument("-nCPUs", type=int, default=4)
    parser.add_argument(
        "-paramFN",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "FF_HEDM"
            / "Example"
            / "Parameters.txt"
        ),
    )
    parser.add_argument(
        "--noiseSigma",
        type=float,
        default=0.0,
        help="Spot position noise sigma in pixels (0 = no noise)",
    )
    parser.add_argument(
        "--omegaSigma",
        type=float,
        default=0.0,
        help="Omega Gaussian width in degrees for 3D spots (0 = single frame)",
    )
    parser.add_argument(
        "--nGrains",
        type=int,
        default=0,
        help="Generate N random grains instead of using existing GrainsSim.csv (0 = use existing)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grain generation")
    parser.add_argument("--no-cleanup", action="store_true")
    return parser.parse_args()


def cleanup_result_dir(work_dir):
    """Remove generated analysis files so the pipeline can re-run refinement."""
    layer_dir = work_dir / "LayerNr_1"
    if not layer_dir.exists():
        return
    for fn in [
        "Key.bin", "ProcessKey.bin", "OrientPosFit.bin", "FitBest.bin",
        "Grains.csv", "SpotMatrix.csv",
    ]:
        p = layer_dir / fn
        if p.exists():
            p.unlink()
    for hf in layer_dir.glob("*_consolidated.h5"):
        hf.unlink()


def run_pipeline(work_dir, test_param_file, zarr_name, nCPUs):
    """Run ff_MIDAS.py and return the Grains.csv path."""
    midas_home = Path(os.environ.get(
        "MIDAS_HOME", str(Path(__file__).resolve().parent.parent)
    ))
    ff_script = midas_home / "FF_HEDM" / "workflows" / "ff_MIDAS.py"
    cmd = [
        sys.executable, str(ff_script),
        "-paramFN", test_param_file.name,
        "-nCPUs", str(nCPUs),
        "-dataFN", zarr_name,
        "-convertFiles", "0",
    ]
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(work_dir))
    if res.returncode != 0:
        print("ERROR: ff_MIDAS.py failed.")
        sys.exit(1)
    grains = work_dir / "LayerNr_1" / "Grains.csv"
    if not grains.exists():
        print(f"ERROR: {grains} not found after pipeline run.")
        sys.exit(1)
    return grains


def run_pipeline_restart(work_dir, test_param_file, zarr_name, nCPUs):
    """Re-run ff_MIDAS.py from refinement stage only."""
    midas_home = Path(os.environ.get(
        "MIDAS_HOME", str(Path(__file__).resolve().parent.parent)
    ))
    ff_script = midas_home / "FF_HEDM" / "workflows" / "ff_MIDAS.py"
    cmd = [
        sys.executable, str(ff_script),
        "-paramFN", test_param_file.name,
        "-nCPUs", str(nCPUs),
        "-dataFN", zarr_name,
        "-convertFiles", "0",
        "-restartFrom", "refinement",
    ]
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(work_dir))
    if res.returncode != 0:
        print("ERROR: ff_MIDAS.py -restartFrom refinement failed.")
        sys.exit(1)
    grains = work_dir / "LayerNr_1" / "Grains.csv"
    if not grains.exists():
        print(f"ERROR: {grains} not found after restart run.")
        sys.exit(1)
    return grains


def match_grains(ref_grains, fit_grains):
    """Match fitted grains to reference grains by nearest 3D position.
    Returns list of (ref_grain, fit_grain) tuples."""
    matched = []
    used = set()
    for rg in ref_grains:
        best_dist = 1e30
        best_idx = -1
        for idx, fg in enumerate(fit_grains):
            if idx in used:
                continue
            d = np.linalg.norm(rg["pos"] - fg["pos"])
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx >= 0:
            matched.append((rg, fit_grains[best_idx]))
            used.add(best_idx)
    return matched


def compute_metrics(ref_grain, fit_grain, sg_num=225):
    """Compute position error, orientation error, and lattice errors."""
    pos_err = np.linalg.norm(ref_grain["pos"] - fit_grain["pos"])
    # Symmetry-aware misorientation using MIDAS C library
    om1 = ref_grain["orient"].flatten().tolist()
    om2 = fit_grain["orient"].flatten().tolist()
    miso_rad, _ = GetMisOrientationAngleOM(om1, om2, sg_num)
    orient_err = np.degrees(miso_rad)
    ref_lat = ref_grain["lattice"]
    fit_lat = fit_grain["lattice"]
    lat_err = np.abs(fit_lat[:3] - ref_lat[:3]) / ref_lat[:3] * 100  # percent
    return pos_err, orient_err, lat_err


def print_comparison(gt_grains, iter_grains, aao_grains, noise_sigma, omega_sigma=0):
    """Print paper-ready comparison table."""
    iter_matched = match_grains(gt_grains, iter_grains)
    aao_matched = match_grains(gt_grains, aao_grains)

    w = 72
    print(f"\n{'=' * w}")
    print(f"  Refinement Mode Comparison (vs Ground Truth)")
    print(f"  SimNoiseSigma = {noise_sigma} pixels, OmegaSigma = {omega_sigma} deg")
    print(f"{'=' * w}")
    print(f"{'Grain':>5} | {'Mode':<13} | {'Pos(um)':>8} | {'Orient(deg)':>11} | "
          f"{'da/a%':>6} | {'db/b%':>6} | {'dc/c%':>6}")
    print(f"{'-' * 5}-+-{'-' * 13}-+-{'-' * 8}-+-{'-' * 11}-+-"
          f"{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}")

    iter_metrics = []
    aao_metrics = []

    for (rg_i, fg_i), (rg_a, fg_a) in zip(iter_matched, aao_matched):
        pe_i, oe_i, le_i = compute_metrics(rg_i, fg_i)
        pe_a, oe_a, le_a = compute_metrics(rg_a, fg_a)
        iter_metrics.append((pe_i, oe_i, le_i))
        aao_metrics.append((pe_a, oe_a, le_a))

        gid = rg_i["id"]
        print(f"{gid:>5} | {'Iterative':<13} | {pe_i:>8.2f} | {oe_i:>11.6f} | "
              f"{le_i[0]:>6.3f} | {le_i[1]:>6.3f} | {le_i[2]:>6.3f}")
        print(f"{gid:>5} | {'All-at-once':<13} | {pe_a:>8.2f} | {oe_a:>11.6f} | "
              f"{le_a[0]:>6.3f} | {le_a[1]:>6.3f} | {le_a[2]:>6.3f}")

    # Summary
    print(f"{'=' * w}")
    print(f"  Summary (mean across {len(iter_matched)} grains)")
    print(f"{'=' * w}")
    if iter_metrics:
        mp_i = np.mean([m[0] for m in iter_metrics])
        mo_i = np.mean([m[1] for m in iter_metrics])
        ms_i = np.mean([np.mean(m[2]) for m in iter_metrics])
        mp_a = np.mean([m[0] for m in aao_metrics])
        mo_a = np.mean([m[1] for m in aao_metrics])
        ms_a = np.mean([np.mean(m[2]) for m in aao_metrics])
        print(f"  Iterative:    pos={mp_i:.2f} um, orient={mo_i:.6f} deg, "
              f"strain={ms_i:.4f}%")
        print(f"  All-at-once:  pos={mp_a:.2f} um, orient={mo_a:.6f} deg, "
              f"strain={ms_a:.4f}%")
    print(f"{'=' * w}")


def main():
    args = parse_args()
    param_path = Path(args.paramFN).resolve()
    example_dir = param_path.parent
    work_dir = example_dir

    print(f"Parameter file: {param_path}")
    print(f"Work directory: {work_dir}")
    print(f"Noise sigma:    {args.noiseSigma} pixels")
    print(f"Omega sigma:    {args.omegaSigma} degrees")
    print(f"nCPUs:          {args.nCPUs}")

    # ── Step 0: Clean stale results ──
    layer_dir = work_dir / "LayerNr_1"
    if layer_dir.exists():
        shutil.rmtree(layer_dir)

    # ── Step 1: Create test environment ──
    test_param_file, params, out_file_name = create_testing_env(param_path, work_dir)

    # ── Step 1b: Generate random grains if requested ──
    if args.nGrains > 0:
        grains_path = example_dir / "GrainsSim.csv"
        # Back up original
        backup = example_dir / "GrainsSim_original.csv"
        if not backup.exists() and grains_path.exists():
            shutil.copy2(str(grains_path), str(backup))
        rsample = params.get("Rsample", 2000)
        hbeam = params.get("Hbeam", 2000)
        beam_thickness = params.get("BeamThickness", 200)
        sg = params.get("SpaceGroup", 225)
        if isinstance(rsample, list): rsample = rsample[0]
        if isinstance(hbeam, list): hbeam = hbeam[0]
        if isinstance(beam_thickness, list): beam_thickness = beam_thickness[0]
        if isinstance(sg, list): sg = sg[0]
        lat_str = params.get("LatticeConstant", [4.08, 4.08, 4.08, 90, 90, 90])
        if not isinstance(lat_str, list):
            lat_str = [lat_str]
        lat = [float(v) for v in lat_str]
        if len(lat) < 6:
            lat = [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
        generate_grains_csv(
            grains_path, args.nGrains, lat,
            float(rsample), float(hbeam), float(beam_thickness),
            space_group=int(sg), seed=args.seed,
        )
        # Update the test param file's InFileName to point to new grains
        # (create_testing_env already set it to the absolute path)

    # Append simulation parameters if requested
    extra_params = []
    if args.noiseSigma > 0:
        extra_params.append(f"SimNoiseSigma {args.noiseSigma}")
    if args.omegaSigma > 0:
        extra_params.append(f"OmegaSigma {args.omegaSigma}")
    if extra_params:
        with open(test_param_file, "a") as f:
            f.write("\n" + "\n".join(extra_params) + "\n")
        print(f"Appended to {test_param_file.name}: {', '.join(extra_params)}")

    # ── Step 2: Forward simulation ──
    print("\n=== Forward Simulation ===")
    run_forward_simulation(test_param_file, args.nCPUs, work_dir)

    # ── Step 3: Enrich Zarr and rename ──
    # ForwardSimulation produces {OutFileName}_scanNr_0.zip
    generated_zip = work_dir / f"{out_file_name}_scanNr_0.zip"
    if not generated_zip.exists():
        print(f"ERROR: Expected simulation output {generated_zip} not found.")
        sys.exit(1)
    enrich_zarr_metadata(generated_zip, params)
    # ff_MIDAS.py expects {OutFileName}.analysis.MIDAS.zip
    zarr_name = f"{out_file_name}.analysis.MIDAS.zip"
    zarr_path = work_dir / zarr_name
    shutil.move(str(generated_zip), str(zarr_path))
    print(f"Simulation complete: {zarr_path}")

    # ── Step 4: Run pipeline with iterative refinement (default) ──
    import time
    print("\n=== Run 1: Iterative Refinement ===")
    t0_iter = time.time()
    grains_iter_path = run_pipeline(work_dir, test_param_file, zarr_name, args.nCPUs)
    t_iter = time.time() - t0_iter
    saved_iter = work_dir / "Grains_iterative.csv"
    shutil.copy2(str(grains_iter_path), str(saved_iter))
    print(f"Saved iterative result to {saved_iter}")
    print(f"Iterative pipeline time: {t_iter:.1f} s")

    # Save iterative refining log before it gets overwritten
    iter_reflog = work_dir / "LayerNr_1" / "midas_log" / "refining_out0.csv"
    saved_reflog = work_dir / "refining_out_iterative.csv"
    if iter_reflog.exists():
        shutil.copy2(str(iter_reflog), str(saved_reflog))

    # ── Step 5: Modify paramstest.txt for all-at-once, restart from refinement ──
    print("\n=== Run 2: All-at-Once Refinement ===")
    # Append to both the test param file AND paramstest.txt in LayerNr_1
    with open(test_param_file, "a") as f:
        f.write("\nFitAllAtOnce 1\n")
    paramstest = work_dir / "LayerNr_1" / "paramstest.txt"
    if paramstest.exists():
        with open(paramstest, "a") as f:
            f.write("\nFitAllAtOnce 1\n")
        print(f"Appended FitAllAtOnce 1 to paramstest.txt")
    else:
        print(f"WARNING: {paramstest} not found, appending to test param file only")

    cleanup_result_dir(work_dir)
    t0_aao = time.time()
    grains_aao_path = run_pipeline_restart(
        work_dir, test_param_file, zarr_name, args.nCPUs
    )
    t_aao = time.time() - t0_aao
    saved_aao = work_dir / "Grains_allatonce.csv"
    shutil.copy2(str(grains_aao_path), str(saved_aao))
    print(f"Saved all-at-once result to {saved_aao}")
    print(f"All-at-once pipeline time: {t_aao:.1f} s (refinement+consolidation only)")

    # ── Step 6: Load ground truth and compare ──
    gt_path = example_dir / "GrainsSim.csv"
    if not gt_path.exists():
        print(f"ERROR: Ground truth {gt_path} not found.")
        sys.exit(1)

    gt_grains = parse_grains_csv(gt_path)
    iter_grains = parse_grains_csv(saved_iter)
    aao_grains = parse_grains_csv(saved_aao)

    print(f"\nGround truth:  {len(gt_grains)} grains")
    print(f"Iterative:     {len(iter_grains)} grains")
    print(f"All-at-once:   {len(aao_grains)} grains")

    print_comparison(gt_grains, iter_grains, aao_grains, args.noiseSigma, args.omegaSigma)

    # ── Cleanup ──
    if not args.no_cleanup:
        for fn in work_dir.iterdir():
            if fn.name not in PRESERVE and fn.name not in {
                "Grains_iterative.csv", "Grains_allatonce.csv",
            }:
                if fn.is_dir():
                    shutil.rmtree(fn, ignore_errors=True)
                elif fn.is_file():
                    fn.unlink(missing_ok=True)
        print("\nCleaned up working directory (kept Grains_*.csv).")


if __name__ == "__main__":
    main()
