#!/usr/bin/env python3
"""
MIDAS Phase Identification Benchmark Test

Runs phase_id.py on CeO2 calibration data with CeO2 + Au phases
and verifies that CeO2 is correctly identified as PRESENT and Au
as ABSENT.

Tests BOTH pipelines:
  A) Zarr pipeline  (create_zarr_zip + run_cpu_pipeline)
  B) Direct pipeline (run_detector_mapper + process_single_file)

DetectorMapper is regenerated between the two pipelines.

Usage:
    python test_phase_id.py
    python test_phase_id.py -nCPUs 8
    python test_phase_id.py --keep-work-dir
"""

import argparse
import math
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
CALIB_DIR = MIDAS_HOME / "FF_HEDM" / "Example" / "Calibration"

# Import phase_id components for direct programmatic access
sys.path.insert(0, str(SCRIPT_DIR))
from phase_id import (
    parse_phases_file, predict_rings_for_phase, merge_and_deduplicate,
    read_geometry, write_peak_params, create_zarr_zip, run_cpu_pipeline,
    run_detector_mapper, process_single_file,
    read_fit_bin, back_calculate_lattice, print_results,
    compute_confidence,
    PhaseInfo, RingEntry, FitResult,
)


def write_test_phases(path: Path):
    """Write a test phases file with CeO2 (present) and Au (absent)."""
    with open(path, 'w') as f:
        f.write("# Test phases for CeO2 calibration benchmark\n")
        f.write("# name  spacegroup  lattice_a(Å)\n")
        f.write("CeO2    225         5.4116\n")
        f.write("Au      225         4.0782\n")


def check_assertions(fits, rings, phases, geom, snr_threshold,
                     rel_intensity_threshold, pipeline_label):
    """Run assertions on fit results. Returns True if all pass."""
    global_max_imax = max((f.Imax for f in fits if f.Imax > 0), default=1.0)
    min_sigma = 0.5 * geom['RBinSize']
    phase_detected = {p.name: 0 for p in phases}
    phase_total = {p.name: 0 for p in phases}
    phase_a = {p.name: [] for p in phases}
    phase_excl_det = {p.name: 0 for p in phases}
    phase_excl_tot = {p.name: 0 for p in phases}
    phase_intensities = {p.name: [] for p in phases}

    for i, ring in enumerate(rings):
        if i >= len(fits):
            break
        fit = fits[i]
        auc = math.pi * fit.Imax * fit.Sigma if fit.Imax > 0 else 0.0
        det = (fit.Imax > 0 and
               fit.SNR >= snr_threshold and
               fit.Imax >= rel_intensity_threshold * global_max_imax and
               fit.Sigma >= min_sigma)

        for ref in ring.reflections:
            phase_total[ref.phase] += 1
            if not ring.is_overlap:
                phase_excl_tot[ref.phase] += 1
            if det:
                phase_detected[ref.phase] += 1
                a = back_calculate_lattice(
                    fit.Center, ref.h, ref.k, ref.l, geom)
                phase_a[ref.phase].append(a)
                phase_intensities[ref.phase].append(auc)
                if not ring.is_overlap:
                    phase_excl_det[ref.phase] += 1

    ceo2_det = phase_detected.get('CeO2', 0)
    ceo2_tot = phase_total.get('CeO2', 1)
    ceo2_ratio = ceo2_det / ceo2_tot if ceo2_tot > 0 else 0
    ceo2_pass = ceo2_ratio >= 0.3

    au_det = phase_detected.get('Au', 0)
    au_pass = au_det == 0

    ceo2_a_vals = phase_a.get('CeO2', [])
    if ceo2_a_vals:
        mean_a = statistics.mean(ceo2_a_vals)
        delta_ppm = abs(mean_a - 5.4116) / 5.4116 * 1e6
        a_pass = delta_ppm < 500
    else:
        mean_a = 0
        delta_ppm = float('inf')
        a_pass = False

    total_intensity = (sum(phase_intensities.get('CeO2', []))
                       + sum(phase_intensities.get('Au', [])))
    ceo2_int_frac = (sum(phase_intensities.get('CeO2', []))
                     / total_intensity if total_intensity > 0 else 0)
    ceo2_conf = compute_confidence(
        ceo2_det, ceo2_tot,
        phase_excl_det.get('CeO2', 0), phase_excl_tot.get('CeO2', 0),
        ceo2_a_vals, 5.4116, ceo2_int_frac, True)
    conf_pass = ceo2_conf >= 50

    print()
    print("-" * 70)
    pass_all = True

    if ceo2_pass:
        print(f"  ✅ PASS: CeO2 detected ({ceo2_det}/{ceo2_tot}, "
              f"{ceo2_ratio*100:.0f}%)")
    else:
        print(f"  ❌ FAIL: CeO2 NOT detected ({ceo2_det}/{ceo2_tot})")
        pass_all = False

    if au_pass:
        print(f"  ✅ PASS: Au absent (0 detected)")
    else:
        print(f"  ❌ FAIL: Au detected ({au_det} peaks)")
        pass_all = False

    if a_pass:
        print(f"  ✅ PASS: CeO2 a = {mean_a:.4f} Å "
              f"(Δa/a = {delta_ppm:.0f} ppm < 500 ppm)")
    else:
        print(f"  ❌ FAIL: CeO2 a = {mean_a:.4f} Å "
              f"(Δa/a = {delta_ppm:.0f} ppm ≥ 500 ppm)")
        pass_all = False

    if conf_pass:
        print(f"  ✅ PASS: CeO2 confidence = {ceo2_conf} (≥ 50)")
    else:
        print(f"  ❌ FAIL: CeO2 confidence = {ceo2_conf} (< 50)")
        pass_all = False

    print("-" * 70)
    if pass_all:
        print(f"  ✅ [{pipeline_label}] ALL ASSERTIONS PASSED")
    else:
        print(f"  ❌ [{pipeline_label}] SOME ASSERTIONS FAILED")
    print()
    return pass_all


def run_test(n_cpus=4, max_rings=15, keep_work=False,
             work_dir_override=None, snr_threshold=5.0,
             rel_intensity_threshold=0.01):
    """Run the phase identification benchmark on both pipelines."""
    param_path = CALIB_DIR / "parameters.txt"
    data_file = CALIB_DIR / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
    dark_file = CALIB_DIR / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"

    if not param_path.exists():
        print(f"ERROR: {param_path} not found")
        sys.exit(1)
    if not data_file.exists():
        print(f"ERROR: {data_file} not found")
        sys.exit(1)

    if work_dir_override:
        work_dir = Path(work_dir_override)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix='midas_phase_id_test_'))

    print("=" * 70)
    print("  MIDAS Phase Identification Benchmark")
    print("=" * 70)
    print(f"  Calibration dir: {CALIB_DIR}")
    print(f"  CPUs: {n_cpus}")
    print(f"  Work dir: {work_dir}")
    print()

    try:
        # ── Shared: predict rings and deduplicate ─────────────────────
        phases_file = work_dir / "test_phases.txt"
        write_test_phases(phases_file)
        phases = parse_phases_file(phases_file)
        geom = read_geometry(param_path)

        print("[1/6] Predicting ring positions...")
        all_refs = []
        for phase in phases:
            refs = predict_rings_for_phase(phase, param_path, geom)
            refs = refs[:max_rings]
            all_refs.extend(refs)
            print(f"  {phase.name}: {len(refs)} rings")

        print("\n[2/6] Deduplicating rings...")
        merge_thresh = 2.0 * geom['RBinSize']
        rings = merge_and_deduplicate(all_refs, merge_thresh)
        n_overlaps = sum(1 for r in rings if r.is_overlap)
        print(f"  {len(rings)} peaks ({n_overlaps} overlapping)")

        # ── Shared: create working param file ─────────────────────────
        peak_params = work_dir / "peak_params.txt"
        n_peaks = write_peak_params(rings, peak_params)

        work_param = work_dir / "test_params.txt"
        param_dir = param_path.parent
        with open(param_path) as fin, open(work_param, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    fout.write(line)
                    continue
                key = parts[0]
                if key in ('MaskFile', 'MaskFN') and len(parts) > 1:
                    fout.write(f"{key} {(param_dir / parts[1]).resolve()}\n")
                elif key == 'Dark' and len(parts) > 1:
                    fout.write(f"Dark {(param_dir / parts[1]).resolve()}\n")
                elif key == 'Folder':
                    fout.write(f"Folder {work_dir}\n")
                else:
                    fout.write(line)

        # ==============================================================
        #  PIPELINE A: Zarr pipeline
        # ==============================================================
        print("\n" + "=" * 70)
        print("  PIPELINE A: Zarr (create_zarr_zip + IntegratorZarrOMP)")
        print("=" * 70)

        print("\n[3/6] Running Zarr integration + peak fitting...")
        zip_file = create_zarr_zip(data_file, dark_file, work_param, work_dir)
        fit_bin_zarr = run_cpu_pipeline(zip_file, peak_params, work_dir, n_cpus)

        print("\n[A] Analyzing Zarr pipeline results...")
        fits_zarr = read_fit_bin(fit_bin_zarr, n_peaks)
        if not fits_zarr:
            print("  FAIL: No fit results from Zarr pipeline")
            return False

        print_results(rings, fits_zarr, geom, snr_threshold,
                      rel_intensity_threshold, phases)
        pass_zarr = check_assertions(fits_zarr, rings, phases, geom,
                                     snr_threshold, rel_intensity_threshold,
                                     "Zarr")

        # ==============================================================
        #  Clean up Zarr artifacts, regenerate DetectorMapper
        # ==============================================================
        print("[4/6] Cleaning Zarr artifacts & regenerating DetectorMapper...")
        # Remove Zarr-generated files to ensure clean slate
        for stale in work_dir.glob("*.zarr.zip"):
            stale.unlink()
        for stale in work_dir.glob("*_fit.bin"):
            stale.unlink()
        for stale in work_dir.glob("*_lineout.bin"):
            stale.unlink()
        for stale in work_dir.glob("*_fit_per_eta.csv"):
            stale.unlink()
        for stale in ("Map.bin", "nMap.bin",
                      "REtaMap.bin", "nREtaMap.bin"):
            p = work_dir / stale
            if p.exists():
                p.unlink()

        # Regenerate DetectorMapper for direct mode
        print("  Running DetectorMapper (non-Zarr)...")
        run_detector_mapper(work_param, work_dir)

        map_bin = work_dir / "Map.bin"
        nmap_bin = work_dir / "nMap.bin"
        if not map_bin.exists() or not nmap_bin.exists():
            print("  FAIL: DetectorMapper did not produce Map.bin / nMap.bin")
            return False
        print(f"  Map.bin:  {map_bin.stat().st_size:,} bytes")
        print(f"  nMap.bin: {nmap_bin.stat().st_size:,} bytes")

        # ==============================================================
        #  PIPELINE B: Direct mode (non-Zarr)
        # ==============================================================
        print("\n" + "=" * 70)
        print("  PIPELINE B: Direct (IntegratorZarrOMP -paramFN -dataFN)")
        print("=" * 70)

        print("\n[5/6] Running direct-mode integration + peak fitting...")
        log_text, results_text, summary_text, tm, _peak_rows = process_single_file(
            data_file, work_dir, work_param, peak_params,
            rings, n_peaks, geom, dark_file, n_cpus, 'cpu',
            snr_threshold, rel_intensity_threshold, phases,
        )

        print("\n[B] Analyzing direct pipeline results...")
        fit_bin_direct = work_dir / f"{data_file.stem}_fit.bin"
        fits_direct = read_fit_bin(fit_bin_direct, n_peaks)
        if not fits_direct:
            print("  FAIL: No fit results from direct pipeline")
            return False

        print_results(rings, fits_direct, geom, snr_threshold,
                      rel_intensity_threshold, phases)
        pass_direct = check_assertions(fits_direct, rings, phases, geom,
                                       snr_threshold, rel_intensity_threshold,
                                       "Direct")

        # ==============================================================
        #  Final verdict
        # ==============================================================
        print("[6/6] Final verdict...")
        print("=" * 70)
        if pass_zarr and pass_direct:
            print("  ✅ BOTH PIPELINES PASSED")
        else:
            if not pass_zarr:
                print("  ❌ Zarr pipeline FAILED")
            if not pass_direct:
                print("  ❌ Direct pipeline FAILED")
        print("=" * 70)
        print()

        return pass_zarr and pass_direct

    finally:
        if not keep_work and not work_dir_override:
            print(f"Cleaning up: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"Work directory preserved: {work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase Identification Benchmark Test')
    parser.add_argument('-nCPUs', type=int, default=4)
    parser.add_argument('--keep-work-dir', action='store_true')
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--max-rings', type=int, default=15)
    args = parser.parse_args()

    success = run_test(
        n_cpus=args.nCPUs,
        max_rings=args.max_rings,
        keep_work=args.keep_work_dir,
        work_dir_override=args.work_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
