#!/usr/bin/env python3
"""
MIDAS Multi-Phase Identification Tool

Identifies crystallographic phases in 2D diffraction images by:
1. Predicting ring positions for each candidate phase via GetHKLList
2. Deduplicating overlapping rings across phases
3. Running integration + pseudo-Voigt peak fitting (CPU or GPU backend)
4. Back-calculating lattice parameters from fitted peak centers (cubic)
5. Reporting per-ring results and per-phase detection summary

Usage:
    python phase_id.py -paramFN geometry.txt -dataFN image.tif -phases phases.txt
    python phase_id.py -paramFN geometry.txt -dataFN image.tif -phases phases.txt -backend gpu
"""

import argparse
import concurrent.futures
import contextlib
import glob as glob_mod
import io
import json
import math
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # macOS: prevent dual-libomp abort
import re
import shutil
import statistics as stats_mod
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"

# Log levels
QUIET, NORMAL, VERBOSE = 0, 1, 2
_log_level = NORMAL


def qprint(*args, level=NORMAL, **kwargs):
    """Print only if current log level >= required level."""
    if _log_level >= level:
        print(*args, **kwargs)


# Peak fit binary format: 7 doubles per peak
PF_PARAMS_PER_PEAK = 7
# [0]=Imax, [1]=BG, [2]=Mix, [3]=Center, [4]=Sigma, [5]=SNR, [6]=Area


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class PhaseInfo:
    """Crystal phase definition."""
    name: str
    spacegroup: int
    lattice_a: float  # Å (cubic)


@dataclass
class HKLReflection:
    """Single reflection from GetHKLList."""
    phase: str
    h: int
    k: int
    l: int
    d_spacing: float   # Å
    ring_nr: int
    R_um: float         # radius in µm
    R_px: float         # radius in pixels


@dataclass
class RingEntry:
    """A (possibly merged) peak position for fitting."""
    R_px: float                             # peak center in pixels
    R_um: float                             # peak center in µm
    reflections: List[HKLReflection]        # contributing reflections
    is_overlap: bool = False                # True if multiple phases contribute

    @property
    def hkl_label(self):
        if len(self.reflections) == 1:
            r = self.reflections[0]
            return f"{r.h}{r.k}{r.l}"
        return "+".join(f"{r.phase}({r.h}{r.k}{r.l})" for r in self.reflections)

    @property
    def phase_names(self):
        return list(set(r.phase for r in self.reflections))


@dataclass
class FitResult:
    """Fitted peak result."""
    Imax: float = 0.0
    BG: float = 0.0
    Mix: float = 0.0
    Center: float = 0.0   # R_fitted in pixels
    Sigma: float = 0.0
    SNR: float = 0.0
    Area: float = 0.0


# =========================================================================
# Utility functions
# =========================================================================

def run_cmd(cmd, cwd=None, check=True, log=None):
    """Run a command and return stdout.

    If *log* is a list, diagnostic output is appended there instead of
    being printed, which allows parallel workers to collect output.
    """
    cmd_str = f"  $ {' '.join(str(c) for c in cmd)}"
    if log is not None:
        log.append(cmd_str)
    else:
        qprint(cmd_str, level=VERBOSE)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                            errors='replace')
    if check and result.returncode != 0:
        err_msg = f"  STDERR: {result.stderr[-500:]}"
        if log is not None:
            log.append(err_msg)
        else:
            print(err_msg)
        raise RuntimeError(f"Command failed (rc={result.returncode}): "
                           f"{' '.join(str(c) for c in cmd)}")
    return result.stdout


def read_param_value(param_file: Path, key: str, default=None):
    """Read a single parameter value from a MIDAS parameter file."""
    with open(param_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == key and len(parts) >= 2:
                return parts[1]
    return default


def read_geometry(param_file: Path) -> dict:
    """Read key geometry parameters from a MIDAS parameter file."""
    geom = {'px': 172.0, 'Lsd': 0.0, 'Wavelength': 0.0,
             'RMin': 10.0, 'RMax': 1200.0, 'RBinSize': 0.25,
             'MultFactor': 0.0}
    with open(param_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0]
            if key in geom and len(parts) >= 2:
                geom[key] = float(parts[1])
    return geom


# =========================================================================
# Phase file parsing
# =========================================================================

def parse_phases_file(phases_path: Path) -> List[PhaseInfo]:
    """Parse the phases definition file.

    Format: name  spacegroup  lattice_a(Å)
    Lines starting with # are comments.
    """
    phases = []
    with open(phases_path) as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                print(f"  WARNING: Skipping malformed line {line_num}: {stripped}")
                continue
            try:
                phases.append(PhaseInfo(
                    name=parts[0],
                    spacegroup=int(parts[1]),
                    lattice_a=float(parts[2]),
                ))
            except ValueError as e:
                print(f"  WARNING: Could not parse line {line_num}: {e}")
    return phases


# =========================================================================
# Ring prediction via GetHKLList
# =========================================================================

def predict_rings_for_phase(phase: PhaseInfo, param_file: Path,
                            geom: dict) -> List[HKLReflection]:
    """Run GetHKLList for a single phase and return reflections."""
    # Create temp parameter file with this phase's crystal structure
    tmp_fd = tempfile.NamedTemporaryFile(suffix='_phase.txt', delete=False)
    tmp_param = Path(tmp_fd.name)
    tmp_fd.close()
    try:
        with open(param_file) as fin, open(tmp_param, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    fout.write(line)
                    continue
                key = parts[0]
                if key == 'SpaceGroup':
                    fout.write(f"SpaceGroup {phase.spacegroup}\n")
                elif key == 'LatticeConstant':
                    a = phase.lattice_a
                    fout.write(f"LatticeConstant {a} {a} {a} 90.0 90.0 90.0\n")
                else:
                    fout.write(line)

        # Run GetHKLList (suppress $ line from screen)
        hkl_bin = MIDAS_BIN / "GetHKLList"
        _sink: List[str] = []  # absorb diagnostic output
        stdout = run_cmd([str(hkl_bin), str(tmp_param), "--stdout"],
                         log=_sink)

        # Parse output: "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
        reflections = []
        seen_rings = set()
        px = geom['px']
        for line in stdout.strip().split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('h '):
                continue
            parts = stripped.split()
            if len(parts) >= 11:
                try:
                    h = int(float(parts[0]))
                    k = int(float(parts[1]))
                    l = int(float(parts[2]))
                    d_spacing = float(parts[3])
                    ring_nr = int(float(parts[4]))
                    radius_um = float(parts[10])
                    # Only keep one reflection per unique ring
                    if ring_nr not in seen_rings:
                        seen_rings.add(ring_nr)
                        reflections.append(HKLReflection(
                            phase=phase.name,
                            h=abs(h), k=abs(k), l=abs(l),
                            d_spacing=d_spacing,
                            ring_nr=ring_nr,
                            R_um=radius_um,
                            R_px=radius_um / px,
                        ))
                except (ValueError, IndexError):
                    continue
        return reflections
    finally:
        tmp_param.unlink(missing_ok=True)


def merge_and_deduplicate(all_reflections: List[HKLReflection],
                          merge_threshold_px: float = 0.5
                          ) -> List[RingEntry]:
    """Sort all reflections by R, merge overlapping ones into RingEntries."""
    # Sort by R_px
    sorted_refs = sorted(all_reflections, key=lambda r: r.R_px)
    if not sorted_refs:
        return []

    # Build initial ring entries
    entries = [RingEntry(R_px=r.R_px, R_um=r.R_um, reflections=[r])
               for r in sorted_refs]

    # Merge adjacent entries within threshold
    merged = [entries[0]]
    for entry in entries[1:]:
        prev = merged[-1]
        if abs(entry.R_px - prev.R_px) < merge_threshold_px:
            # Merge: weighted midpoint
            n1 = len(prev.reflections)
            n2 = len(entry.reflections)
            prev.R_px = (prev.R_px * n1 + entry.R_px * n2) / (n1 + n2)
            prev.R_um = (prev.R_um * n1 + entry.R_um * n2) / (n1 + n2)
            prev.reflections.extend(entry.reflections)
            prev.is_overlap = True
        else:
            merged.append(entry)

    return merged


# =========================================================================
# CPU backend: ffGenerateZipRefactor → DetectorMapperZarr → IntegratorZarrOMP
# =========================================================================

def write_peak_params(rings: List[RingEntry], out_path: Path,
                      roi_padding: int = 30) -> int:
    """Write peak_params.txt for IntegratorZarrOMP."""
    with open(out_path, 'w') as f:
        f.write("# Auto-generated by phase_id.py\n")
        f.write("DoPeakFit 1\n")
        f.write(f"FitROIPadding {roi_padding}\n")
        for ring in rings:
            f.write(f"PeakLocation {ring.R_px:.6f}\n")
    return len(rings)


def create_zarr_zip(data_file: Path, dark_file: Optional[Path],
                    param_file: Path, work_dir: Path, log=None) -> Path:
    """Create a Zarr zip from a TIFF/HDF5 data file."""
    gen_script = SCRIPT_DIR / "ffGenerateZipRefactor.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"ffGenerateZipRefactor.py not found at {gen_script}")

    cmd = [
        sys.executable, str(gen_script),
        '-paramFN', str(param_file.resolve()),
        '-dataFN', str(data_file.resolve()),
        '-resultFolder', str(work_dir.resolve()),
    ]
    if dark_file and dark_file.exists():
        cmd.extend(['-darkFN', str(dark_file.resolve())])

    run_cmd(cmd, cwd=str(work_dir.resolve()), log=log)

    zips = list(work_dir.glob("*.MIDAS.zip"))
    if not zips:
        zips = list(work_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No Zarr zip generated in {work_dir}")
    return zips[0]


def run_detector_mapper(param_file: Path, work_dir: Path, n_cpus: int = 0, log=None):
    """Run DetectorMapper (non-Zarr) to produce Map.bin + nMap.bin.

    Uses the parameter-file-based DetectorMapper which reads geometry
    directly from the text file, avoiding the need for Zarr creation.
    """
    mapper = MIDAS_BIN / "DetectorMapper"
    msg = "  Running DetectorMapper (non-Zarr)..."
    if log is not None:
        log.append(msg)
    else:
        print(msg)
    cmd = [str(mapper), str(param_file.resolve())]
    if n_cpus > 0:
        cmd += ["-nCPUs", str(n_cpus)]
    run_cmd(cmd, cwd=str(work_dir.resolve()), log=log)


def run_cpu_pipeline(zip_file: Path, peak_params: Path,
                     work_dir: Path, n_cpus: int, log=None) -> Path:
    """Run DetectorMapperZarr + IntegratorZarrOMP. Returns path to fit.bin.

    Skips DetectorMapperZarr if Map.bin and nMap.bin already exist.
    If *log* is a list, output is captured there instead of printed.
    """
    # Resolve all paths to absolute (run_cmd uses cwd=work_dir)
    zip_file = zip_file.resolve()
    peak_params = peak_params.resolve()
    work_dir = work_dir.resolve()

    def _log(msg):
        if log is not None:
            log.append(msg)
        else:
            print(msg)

    map_bin = work_dir / "Map.bin"
    nmap_bin = work_dir / "nMap.bin"

    if map_bin.exists() and nmap_bin.exists():
        _log("  DetectorMapperZarr: skipped (Map.bin + nMap.bin exist)")
        # Validate parameter headers (capture prints)
        from map_header import check_map_header
        hdr_buf = io.StringIO()
        with contextlib.redirect_stdout(hdr_buf):
            check_map_header(map_bin, "Map.bin")
            check_map_header(nmap_bin, "nMap.bin")
        hdr_out = hdr_buf.getvalue().strip()
        if hdr_out:
            for hl in hdr_out.split('\n'):
                _log(hl)
    else:
        mapper = MIDAS_BIN / "DetectorMapperZarr"
        _log("  Running DetectorMapperZarr...")
        cmd = [str(mapper), str(zip_file)]
        if n_cpus > 0:
            cmd += ["-nCPUs", str(n_cpus)]
        run_cmd(cmd, cwd=str(work_dir), log=log)

    integrator = MIDAS_BIN / "IntegratorZarrOMP"
    _log("  Running IntegratorZarrOMP with peak fitting...")
    stdout = run_cmd([str(integrator), str(zip_file), str(n_cpus),
                      str(peak_params)], cwd=str(work_dir), log=log)

    # Diagnostic lines
    for line in stdout.split('\n'):
        stripped = line.strip()
        if stripped and any(k in stripped for k in
                           ['peak', 'Peak', 'nPeaks', 'nRBins', 'PeakFit',
                            'Warning', 'Error', 'REJECTED', 'FAILED']):
            _log(f"    ▸ {stripped}")

    fit_bin = work_dir / "fit.bin"
    if fit_bin.exists():
        _log(f"  fit.bin: {fit_bin.stat().st_size} bytes")
    else:
        _log("  WARNING: fit.bin not generated")
    return fit_bin


# =========================================================================
# GPU backend: integrator_batch_process.py
# =========================================================================

def run_gpu_pipeline(data_file: Path, dark_file: Optional[Path],
                     param_file: Path, rings: List[RingEntry],
                     work_dir: Path, n_cpus: int,
                     roi_padding: int = 30) -> Path:
    """Run GPU pipeline via integrator_batch_process.py. Returns path to fit.bin."""
    # Resolve all paths to absolute
    data_file = data_file.resolve()
    param_file = param_file.resolve()
    work_dir = work_dir.resolve()
    if dark_file:
        dark_file = dark_file.resolve()

    # Create temp param file with PeakLocation lines injected
    gpu_param = work_dir / "gpu_params.txt"
    with open(param_file) as fin, open(gpu_param, 'w') as fout:
        for line in fin:
            fout.write(line)
        fout.write("\n# Phase ID peak fitting\n")
        fout.write("DoPeakFit 1\n")
        fout.write(f"FitROIPadding {roi_padding}\n")
        for ring in rings:
            fout.write(f"PeakLocation {ring.R_px:.6f}\n")

    # Set up data folder
    data_folder = data_file.parent

    batch_script = SCRIPT_DIR / "integrator_batch_process.py"
    if not batch_script.exists():
        raise FileNotFoundError(f"integrator_batch_process.py not found")

    output_h5 = work_dir / "phase_id_output.h5"
    cmd = [
        sys.executable, str(batch_script),
        '--param-file', str(gpu_param),
        '--folder', str(data_folder),
        '--output-dir', str(work_dir),
        '--output-h5', str(output_h5),
        '--no-zarr',
    ]
    if dark_file and dark_file.exists():
        cmd.extend(['--dark', str(dark_file)])

    print("  Running GPU pipeline (integrator_batch_process.py)...")
    run_cmd(cmd, cwd=str(work_dir))

    fit_bin = work_dir / "fit.bin"
    if fit_bin.exists():
        print(f"  fit.bin: {fit_bin.stat().st_size} bytes")
    else:
        print("  WARNING: fit.bin not generated")
    return fit_bin


# =========================================================================
# fit.bin parsing and analysis
# =========================================================================

def read_fit_bin(fit_bin: Path, n_peaks: int) -> List[FitResult]:
    """Read fit.bin and return per-peak FitResult objects."""
    if not fit_bin.exists():
        return []

    data = fit_bin.read_bytes()
    expected_size = n_peaks * PF_PARAMS_PER_PEAK * 8

    if len(data) < expected_size:
        print(f"  WARNING: fit.bin size ({len(data)}) < expected ({expected_size})")
        return []

    values = struct.unpack(f'{n_peaks * PF_PARAMS_PER_PEAK}d',
                           data[:expected_size])
    results = []
    for i in range(n_peaks):
        base = i * PF_PARAMS_PER_PEAK
        results.append(FitResult(
            Imax=values[base], BG=values[base + 1], Mix=values[base + 2],
            Center=values[base + 3], Sigma=values[base + 4],
            SNR=values[base + 5], Area=values[base + 6],
        ))
    return results


def read_fit_per_eta(csv_path: Path) -> dict:
    """Parse fit_per_eta.csv produced by IntegratorZarrOMP.

    Returns a dict keyed by PeakIdx (int), where each value is a list
    of dicts with keys: EtaCen, R_px, R_um, TwoTheta_deg, Imax,
    Sigma_px, FWHM_px, SNR, Mix, GoF, Area.
    Only frame 0 rows are returned (phase_id processes single-frame
    images).
    """
    if not csv_path.exists():
        return {}
    import csv
    result: dict = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['Frame'])
                if frame != 0:
                    continue
                pidx = int(row['PeakIdx'])
                entry = {
                    'EtaCen': float(row['EtaCen']),
                    'R_px': float(row['R_px']),
                    'R_um': float(row['R_um']),
                    'TwoTheta_deg': float(row['TwoTheta_deg']),
                    'Imax': float(row['Imax']),
                    'Sigma_px': float(row['Sigma_px']),
                    'FWHM_px': float(row['FWHM_px']),
                    'SNR': float(row['SNR']),
                }
                result.setdefault(pidx, []).append(entry)
            except (KeyError, ValueError):
                continue
    return result


def compute_filtered_peak_stats(
        per_eta: dict, rings: List['RingEntry'], geom: dict,
        mult_factor: float) -> dict:
    """Compute filtered per-peak lattice parameters from per-eta-bin data.

    For each ring, collects all per-eta R_px values, computes ``a`` for
    each eta bin, applies iterative MultFactor rejection on |Δa/a_mean|,
    and returns filtered statistics.

    Returns a dict keyed by peak index (int), with values:
        mean_R_px, mean_a, std_a, n_total, n_excluded,
        mean_tth_deg, mean_fwhm_px, mean_imax
    Peaks without per-eta data are omitted.
    """
    result = {}
    for pidx, eta_rows in per_eta.items():
        if pidx >= len(rings):
            continue
        ring = rings[pidx]
        ref = ring.reflections[0]  # use first reflection for back-calc

        # Compute a from each eta bin
        a_per_eta = []
        r_per_eta = []
        tth_per_eta = []
        fwhm_per_eta = []
        imax_per_eta = []
        for row in eta_rows:
            a = back_calculate_lattice(row['R_px'], ref.h, ref.k, ref.l, geom)
            if a > 0:
                a_per_eta.append(a)
                r_per_eta.append(row['R_px'])
                tth_per_eta.append(row['TwoTheta_deg'])
                fwhm_per_eta.append(row['FWHM_px'])
                imax_per_eta.append(row['Imax'])

        if not a_per_eta:
            continue

        n_total = len(a_per_eta)
        n_excluded = 0

        if mult_factor > 0 and len(a_per_eta) >= 2:
            # Iterative rejection: threshold = mult_factor × mean(|Δa/a_mean|)
            indices = list(range(len(a_per_eta)))
            for _iteration in range(3):
                if len(indices) < 2:
                    break
                vals = [a_per_eta[j] for j in indices]
                mean_a = stats_mod.mean(vals)
                if mean_a <= 0:
                    break
                deltas = [abs(v - mean_a) / mean_a for v in vals]
                mean_delta = stats_mod.mean(deltas)
                threshold = mult_factor * mean_delta
                kept = [j for j, d in zip(indices, deltas) if d <= threshold]
                if len(kept) == len(indices) or len(kept) < 1:
                    break
                indices = kept
            n_excluded = n_total - len(indices)
            # Use only kept indices
            a_per_eta = [a_per_eta[j] for j in indices]
            r_per_eta = [r_per_eta[j] for j in indices]
            tth_per_eta = [tth_per_eta[j] for j in indices]
            fwhm_per_eta = [fwhm_per_eta[j] for j in indices]
            imax_per_eta = [imax_per_eta[j] for j in indices]

        mean_a = stats_mod.mean(a_per_eta)
        std_a = stats_mod.stdev(a_per_eta) if len(a_per_eta) > 1 else 0.0

        result[pidx] = {
            'mean_R_px': stats_mod.mean(r_per_eta),
            'mean_a': mean_a,
            'std_a': std_a,
            'n_total': n_total,
            'n_excluded': n_excluded,
            'mean_tth_deg': stats_mod.mean(tth_per_eta),
            'mean_fwhm_px': stats_mod.mean(fwhm_per_eta),
            'mean_imax': stats_mod.mean(imax_per_eta),
        }
    return result


def back_calculate_lattice(R_fitted_px: float, h: int, k: int, l: int,
                           geom: dict) -> float:
    """Back-calculate cubic lattice parameter from fitted Rcen.

    R(px) → R(µm) → 2θ → d-spacing → a = d × √(h²+k²+l²)
    """
    px = geom['px']
    Lsd = geom['Lsd']
    wl = geom['Wavelength']

    R_um = R_fitted_px * px
    two_theta = math.atan(R_um / Lsd)
    theta = two_theta / 2.0
    if theta <= 0 or theta >= math.pi / 2:
        return 0.0
    d_fitted = wl / (2.0 * math.sin(theta))
    hkl_norm = math.sqrt(h * h + k * k + l * l)
    if hkl_norm == 0:
        return 0.0
    return d_fitted * hkl_norm


def compute_confidence(det, tot, excl_det, excl_tot, a_values,
                       lattice_a, intensity_frac, first_rings_detected):
    """Compute 0–100 confidence score for phase presence.

    Components:
    - Coverage (40%):  detected / total rings, saturates at 50%.
    - Exclusive (25%): exclusively-detected / exclusive-total rings.
    - Lattice (20%):   consistency of back-calculated a with nominal.
    - Intensity (15%): AUC fraction, saturates at 30%.
    """
    # Coverage component
    coverage = (det / tot) if tot > 0 else 0
    coverage_score = min(coverage / 0.5, 1.0) * 40

    # Exclusive-peak component
    if excl_tot > 0:
        excl_score = (excl_det / excl_tot) * 25
    else:
        excl_score = coverage * 12.5  # partial credit

    # Lattice consistency component
    if a_values and lattice_a > 0:
        mean_a = stats_mod.mean(a_values)
        delta_ppm = abs(mean_a - lattice_a) / lattice_a * 1e6
        lattice_score = max(0, 1.0 - delta_ppm / 5000) * 20
    else:
        lattice_score = 0

    # Intensity fraction component
    int_score = min(intensity_frac / 0.3, 1.0) * 15

    # Penalty: no first-ring detection
    if not first_rings_detected and det > 0:
        coverage_score *= 0.5

    total = coverage_score + excl_score + lattice_score + int_score
    return min(100, max(0, round(total)))


# =========================================================================
# Peak table output
# =========================================================================

PEAK_TABLE_COLUMNS = [
    'Filename', 'R_px', 'R_um', '2theta_deg', 'FWHM_px', 'FWHM_2theta_deg',
    'Intensity', 'Imax', 'SNR', 'Phase', 'HKL', 'Flag'
]


def build_peak_rows(filename: str, rings: List[RingEntry],
                    fits: List[FitResult], geom: dict,
                    snr_threshold: float,
                    rel_intensity_threshold: float,
                    filtered_stats: dict = None) -> List[dict]:
    """Build peak table rows for one data file.

    Returns one dict per peak (per reflection for overlaps), containing
    all columns in PEAK_TABLE_COLUMNS.  All peaks are included regardless
    of detection status; the Flag column distinguishes them.
    When *filtered_stats* is provided, per-eta-bin filtered values are
    used for R, 2θ, and FWHM.
    """
    px = geom['px']
    Lsd = geom['Lsd']

    min_sigma = 0.5 * geom['RBinSize']
    global_max_imax = max((f.Imax for f in fits if f.Imax > 0), default=1.0)

    n = min(len(rings), len(fits))
    rows: List[dict] = []

    for i in range(n):
        ring = rings[i]
        fit = fits[i]

        # Detection flag (same logic as print_results)
        snr_ok = fit.SNR >= snr_threshold
        rel_ok = fit.Imax >= rel_intensity_threshold * global_max_imax
        sigma_ok = fit.Sigma >= min_sigma
        detected = fit.Imax > 0 and snr_ok and rel_ok and sigma_ok

        # Use per-eta filtered values when available
        if filtered_stats and i in filtered_stats:
            fs = filtered_stats[i]
            R_px = fs['mean_R_px']
            tth_deg = fs['mean_tth_deg']
            fwhm_px = fs['mean_fwhm_px']
        else:
            R_px = fit.Center
            tth_deg = math.degrees(math.atan(R_px * px / Lsd)) if Lsd > 0 else 0.0
            fwhm_px = 2.355 * fit.Sigma

        R_um = R_px * px
        fwhm_um = fwhm_px * px
        if Lsd > 0:
            dtth_dR = 1.0 / (Lsd * (1.0 + (R_um / Lsd) ** 2))
            fwhm_tth_deg = math.degrees(fwhm_um * dtth_dR)
        else:
            fwhm_tth_deg = 0.0

        # One row per reflection (overlapping rings get multiple rows)
        for ref in ring.reflections:
            hkl = f"{ref.h}{ref.k}{ref.l}"
            flag = "DETECTED" if detected else "NOT_DET"
            rows.append({
                'Filename': filename,
                'R_px': R_px,
                'R_um': R_um,
                '2theta_deg': tth_deg,
                'FWHM_px': fwhm_px,
                'FWHM_2theta_deg': fwhm_tth_deg,
                'Intensity': fit.Area,
                'Imax': fit.Imax,
                'SNR': fit.SNR,
                'Phase': ref.phase,
                'HKL': hkl,
                'Flag': flag,
            })

    return rows


def write_peak_tables(rows: List[dict], base_path: Path):
    """Write combined peak table as both CSV and space-separated TXT."""
    csv_path = base_path.with_suffix('.csv')
    txt_path = base_path.with_suffix('.txt')

    # CSV
    with open(csv_path, 'w', newline='') as f:
        f.write(','.join(PEAK_TABLE_COLUMNS) + '\n')
        for row in rows:
            vals = []
            for col in PEAK_TABLE_COLUMNS:
                v = row[col]
                if isinstance(v, float):
                    vals.append(f'{v:.6f}')
                else:
                    vals.append(str(v))
            f.write(','.join(vals) + '\n')

    # Space-separated TXT with comment header
    with open(txt_path, 'w') as f:
        f.write('# ' + '  '.join(f'{c:>16}' for c in PEAK_TABLE_COLUMNS) + '\n')
        for row in rows:
            parts = []
            for col in PEAK_TABLE_COLUMNS:
                v = row[col]
                if isinstance(v, float):
                    parts.append(f'{v:>16.6f}')
                else:
                    parts.append(f'{str(v):>16}')
            f.write('  '.join(parts) + '\n')

    return csv_path, txt_path


SINGLE_PHASE_COLUMNS = ['R_px', 'two_theta_deg', 'FWHM_2theta_deg', 'intensity']


def write_single_phase_peak_table(fits: List[FitResult], rings: List[RingEntry],
                                  geom: dict, out_path: Path,
                                  filtered_stats: dict = None,
                                  intensity_threshold: float = 0.0):
    """Write minimal peak table for single-phase mode.

    Columns: R_px, two_theta_deg, FWHM_2theta_deg, intensity
    Only rows with intensity > intensity_threshold are written.
    When *filtered_stats* is provided, uses per-eta-bin filtered values.
    """
    px = geom['px']
    Lsd = geom['Lsd']
    n = min(len(rings), len(fits))
    with open(out_path, 'w') as f:
        f.write('  '.join(f'{c:>18}' for c in SINGLE_PHASE_COLUMNS) + '\n')
        for i in range(n):
            fit = fits[i]
            if fit.Area <= intensity_threshold:
                continue
            if filtered_stats and i in filtered_stats:
                fs = filtered_stats[i]
                R_px = fs['mean_R_px']
                tth_deg = fs['mean_tth_deg']
                fwhm_px = fs['mean_fwhm_px']
            else:
                R_px = fit.Center
                R_um = R_px * px
                tth_deg = math.degrees(math.atan(R_um / Lsd)) if Lsd > 0 else 0.0
                fwhm_px = 2.355 * fit.Sigma
            R_um = R_px * px
            fwhm_um = fwhm_px * px
            if Lsd > 0:
                dtth_dR = 1.0 / (Lsd * (1.0 + (R_um / Lsd) ** 2))
                fwhm_tth = math.degrees(fwhm_um * dtth_dR)
            else:
                fwhm_tth = 0.0
            f.write(f'{R_px:>18.6f}  {tth_deg:>18.6f}  {fwhm_tth:>18.6f}  {fit.Area:>18.6f}\n')
    return out_path


def write_lineout_comparison(fits: List[FitResult], geom: dict,
                             lineout_bin: Path, out_path: Path,
                             roi_padding: int = 30):
    """Write measured vs calculated 1D profile for single-phase mode.

    Reads the eta-averaged lineout from lineout.bin, reconstructs
    the calculated profile from the pseudo-Voigt fit parameters
    **only within each peak's ROI** (±roi_padding bins from center),
    using the per-job background stored in each peak's BG field.

    The pseudo-Voigt model matches PeakFit.c exactly:
        I(R) = BG_job + Σ { Imax * [Mix*L(R) + (1-Mix)*G(R)] }
        L(R) = 1 / (1 + 4*(R-c)²/Γ²)
        G(R) = exp(-4·ln(2)·(R-c)²/Γ²)
        Γ    = Sigma * 2.355   (stored Sigma is Gaussian-equiv σ)

    Bins outside all ROIs have I_calculated = NaN.
    """
    if not lineout_bin.exists():
        return None

    px = geom['px']
    Lsd = geom['Lsd']
    RMin = geom['RMin']
    RBinSize = geom['RBinSize']

    # Read lineout: nRBins doubles
    data = lineout_bin.read_bytes()
    nRBins = len(data) // 8
    if nRBins == 0:
        return None
    measured = struct.unpack(f'{nRBins}d', data[:nRBins * 8])

    # Compute R bin centers
    R_centers = [RMin + (i + 0.5) * RBinSize for i in range(nRBins)]

    # Build per-peak ROI ranges and merge overlapping ones into jobs
    # (mirrors PeakFit.c's job merging logic)
    C0 = 4.0 * math.log(2.0)
    calculated = [float('nan')] * nRBins

    # Build jobs: groups of peaks with overlapping ROIs
    valid_peaks = []
    for fit in fits:
        if fit.Imax <= 0:
            continue
        # Find nearest bin to peak center
        best_bin = -1
        min_diff = 1e10
        for r in range(nRBins):
            d = abs(R_centers[r] - fit.Center)
            if d < min_diff:
                min_diff = d
                best_bin = r
        if best_bin >= 0:
            roi_start = max(0, best_bin - roi_padding)
            roi_end = min(nRBins - 1, best_bin + roi_padding)
            valid_peaks.append((fit, roi_start, roi_end))

    if not valid_peaks:
        return None

    # Sort by roi_start and merge overlapping ROIs into jobs
    valid_peaks.sort(key=lambda x: x[1])
    jobs = []  # list of (roi_start, roi_end, [fits])
    first_fit, cur_start, cur_end = valid_peaks[0]
    cur_fits_list = [first_fit]
    for fit, rs, re in valid_peaks[1:]:
        if rs <= cur_end:
            # Merge
            cur_end = max(cur_end, re)
            cur_fits_list.append(fit)
        else:
            jobs.append((cur_start, cur_end, cur_fits_list))
            cur_start, cur_end = rs, re
            cur_fits_list = [fit]
    jobs.append((cur_start, cur_end, cur_fits_list))

    # Compute calculated profile within each job's ROI
    for roi_start, roi_end, job_fits in jobs:
        # Use BG from first peak in job (all peaks in a job share BG)
        bg = job_fits[0].BG

        for i in range(roi_start, roi_end + 1):
            val = bg
            for fit in job_fits:
                Imax = fit.Imax
                Mix = max(0.0, min(1.0, fit.Mix))
                c = fit.Center
                Gamma = fit.Sigma * 2.355
                invG2 = 1.0 / max(Gamma * Gamma, 1e-18)
                diff = R_centers[i] - c
                diff_sq = diff * diff
                L = 1.0 / (1.0 + 4.0 * diff_sq * invG2)
                G = math.exp(-C0 * diff_sq * invG2)
                val += Imax * (Mix * L + (1.0 - Mix) * G)
            calculated[i] = val

    # Convert R to 2θ and write
    with open(out_path, 'w') as f:
        f.write(f'{"two_theta_deg":>18}  {"I_measured":>18}  {"I_calculated":>18}\n')
        for i in range(nRBins):
            R_um = R_centers[i] * px
            tth_deg = math.degrees(math.atan(R_um / Lsd)) if Lsd > 0 else 0.0
            calc_str = f'{calculated[i]:>18.6f}' if not math.isnan(calculated[i]) else f'{"NaN":>18}'
            f.write(f'{tth_deg:>18.6f}  {measured[i]:>18.6f}  {calc_str}\n')

    return out_path


# =========================================================================
# Reporting
# =========================================================================

def print_results(rings: List[RingEntry], fits: List[FitResult],
                  geom: dict, snr_threshold: float,
                  rel_intensity_threshold: float,
                  phases: List[PhaseInfo], out=None,
                  summary_out=None, mult_factor: float = 0.0,
                  filtered_stats: dict = None):
    """Print per-ring results and per-phase summary with dual filters.

    If *out* is provided, output is written there instead of stdout.
    If *summary_out* is provided, the Phase Summary table (Table B)
    is also written there for compact screen display.
    If *filtered_stats* is provided (from compute_filtered_peak_stats),
    per-eta-bin filtered lattice parameters are used instead of single-fit values.
    """

    if out is None:
        out = sys.stdout

    phase_nominal = {p.name: p.lattice_a for p in phases}

    # Compute AUC (area under curve) = pi * Imax * Sigma for each fit
    aucs = [math.pi * f.Imax * f.Sigma if f.Imax > 0 else 0.0 for f in fits]
    min_sigma = 0.5 * geom['RBinSize']  # peaks narrower than this are fitting artifacts

    # Find global max Imax for relative intensity filter
    # (AUC is NOT used for gating — broad-sigma fitter artifacts inflate it)
    global_max_imax = max((f.Imax for f in fits if f.Imax > 0), default=1.0)

    # Determine detection per peak
    n = min(len(rings), len(fits))
    detected_flags = []
    for i in range(n):
        fit = fits[i]
        snr_ok = fit.SNR >= snr_threshold
        rel_ok = fit.Imax >= rel_intensity_threshold * global_max_imax
        sigma_ok = fit.Sigma >= min_sigma
        detected_flags.append(fit.Imax > 0 and snr_ok and rel_ok and sigma_ok)

    # Collect per-phase statistics
    phase_stats = {}
    for p in phases:
        phase_stats[p.name] = {
            'detected': 0, 'total': 0, 'a_values': [], 'intensities': [],
            'exclusive_detected': 0, 'exclusive_total': 0,
            'ring_order': [],  # list of (R_px, detected_bool) in R order
        }

    # ── Table A: Per-ring results ──────────────────────────────────────
    print(file=out)
    print("=" * 120, file=out)
    print("  MULTI-PHASE IDENTIFICATION RESULTS", file=out)
    print("=" * 120, file=out)
    print(f"  Filters: SNR ≥ {snr_threshold},  "
          f"Imax ≥ {rel_intensity_threshold*100:.0f}% of max "
          f"(= {global_max_imax * rel_intensity_threshold:.0f} counts)", file=out)
    print(file=out)
    header = (f"{'Phase':<8} {'(hkl)':<8} {'R_theory':>9} {'R_fitted':>9} "
              f"{'Imax':>10} {'AUC':>10} {'BG':>8} {'Sigma':>6} {'SNR':>8} "
              f"{'a_fitted':>9} {'Δa/a(ppm)':>10}  {'Notes'}")
    print(header, file=out)
    print("-" * 120, file=out)

    for i in range(n):
        ring = rings[i]
        fit = fits[i]
        det = detected_flags[i]

        for ref in ring.reflections:
            phase_stats[ref.phase]['total'] += 1
            if not ring.is_overlap:
                phase_stats[ref.phase]['exclusive_total'] += 1

        if det:
            for ref in ring.reflections:
                # Use per-eta filtered value if available, else single fit
                if filtered_stats and i in filtered_stats:
                    a_fitted = filtered_stats[i]['mean_a']
                else:
                    a_fitted = back_calculate_lattice(
                        fit.Center, ref.h, ref.k, ref.l, geom)
                a_nom = phase_nominal[ref.phase]
                delta_ppm = ((a_fitted - a_nom) / a_nom * 1e6
                             if a_nom > 0 else 0)

                phase_stats[ref.phase]['detected'] += 1
                phase_stats[ref.phase]['a_values'].append(a_fitted)
                phase_stats[ref.phase]['intensities'].append(aucs[i])
                if not ring.is_overlap:
                    phase_stats[ref.phase]['exclusive_detected'] += 1
                phase_stats[ref.phase]['ring_order'].append(
                    (ref.R_px, True))
                # Track per-eta exclusions for this ring/phase
                if filtered_stats and i in filtered_stats:
                    phase_stats[ref.phase].setdefault(
                        'eta_excluded', 0)
                    phase_stats[ref.phase]['eta_excluded'] += \
                        filtered_stats[i]['n_excluded']

                notes = ""
                if ring.is_overlap:
                    others = [r for r in ring.reflections if r.phase != ref.phase]
                    if others:
                        notes = "⚠️ " + "+".join(
                            f"{r.phase}({r.h}{r.k}{r.l})" for r in others)

                print(f"  {ref.phase:<8} {ref.h}{ref.k}{ref.l:<7} "
                      f"{ref.R_px:>9.2f} {fit.Center:>9.2f} "
                      f"{fit.Imax:>10.1f} {aucs[i]:>10.1f} {fit.BG:>8.1f} "
                      f"{fit.Sigma:>6.3f} {fit.SNR:>8.1f} "
                      f"{a_fitted:>9.4f} {delta_ppm:>10.1f}  {notes}", file=out)
        else:
            ref = ring.reflections[0]
            phase_label = ref.phase
            if ring.is_overlap:
                phase_label = "+".join(r.phase for r in ring.reflections)

            # Track non-detected rings per phase
            for ref_nd in ring.reflections:
                phase_stats[ref_nd.phase]['ring_order'].append(
                    (ref_nd.R_px, False))

            # Show why it was rejected
            reason = "NOT DET"
            auc_i = aucs[i] if i < len(aucs) else 0
            if fit.Imax > 0:
                if fit.Sigma < min_sigma:
                    reason = f"σ={fit.Sigma:.3f}<{min_sigma:.3f}"
                elif fit.SNR < snr_threshold:
                    reason = f"SNR={fit.SNR:.1f}<{snr_threshold}"
                elif fit.Imax < rel_intensity_threshold * global_max_imax:
                    rel_pct = fit.Imax / global_max_imax * 100
                    reason = f"Imax={rel_pct:.2f}%<{rel_intensity_threshold*100:.0f}%"
            print(f"  {phase_label:<8} {ref.h}{ref.k}{ref.l:<7} "
                  f"{ref.R_px:>9.2f} {'':>9} "
                  f"{'':>10} {'':>10} {'':>8} "
                  f"{'':>6} {'':>8} "
                  f"{'':>9} {'':>10}  {reason}", file=out)

    # ── Table B: Per-phase summary ────────────────────────────────────
    # Full output (out) gets the decorated header; summary_out gets
    # only the data rows so the caller can print one shared header.
    col_header = (f"  {'Phase':<8} {'Det':<6} {'Cov':<6} {'Uniq':<5}"
                  f"{'Mean a(Å)':<10} {'Std a(Å)':<10} "
                  f"{'Min a(Å)':<10} {'Max a(Å)':<10} "
                  f"{'Δa/a(ppm)':<12} "
                  f"{'Sum AUC':>10} {'Frac':>6}  {'Excl':>4}  {'Conf':>4}  {'Status'}")
    W = len(col_header) + 4
    print(file=out)
    print("=" * W, file=out)
    print("  PHASE SUMMARY", file=out)
    print("=" * W, file=out)
    print(col_header, file=out)
    print("-" * W, file=out)

    for p in phases:
        st = phase_stats[p.name]
        det = st['detected']
        tot = st['total']
        coverage = f"{det}/{tot}" if tot > 0 else "0/0"
        pct = (det / tot * 100) if tot > 0 else 0

        excl_det = st['exclusive_detected']
        excl_tot = st['exclusive_total']
        ratio = (excl_det / excl_tot) if excl_tot > 0 else (det / tot if tot > 0 else 0)

        # ── Two-level outlier rejection ─────────────────────────────
        # Level 1: per-eta-bin exclusions (already applied in
        #          compute_filtered_peak_stats, counted here)
        # Level 2: per-ring rejection — remove entire rings whose
        #          mean-a deviates too far from the phase mean.
        #          Catches misattributed overlapping rings (e.g.
        #          CeO2 ring counted as Ta).
        a_values = list(st['a_values'])
        n_outliers = st.get('eta_excluded', 0)

        if mult_factor > 0 and len(a_values) >= 2 and p.lattice_a > 0:
            # Use the NOMINAL lattice parameter as reference (not the
            # computed mean) — prevents bimodal distributions from
            # masking outliers (e.g. CeO2 ring misattributed to Ta).
            deltas = [abs(a - p.lattice_a) / p.lattice_a
                      for a in a_values]
            mean_delta = stats_mod.mean(deltas)
            threshold = mult_factor * mean_delta
            kept = [a for a, d in zip(a_values, deltas)
                    if d <= threshold]
            if 0 < len(kept) < len(a_values):
                a_values = kept
            n_ring_excl = len(st['a_values']) - len(a_values)
            n_outliers += n_ring_excl

        if a_values:
            mean_a = stats_mod.mean(a_values)
            std_a = stats_mod.stdev(a_values) if len(a_values) > 1 else 0.0
            min_a = min(a_values)
            max_a = max(a_values)
            d_ppm = (mean_a - p.lattice_a) / p.lattice_a * 1e6
        else:
            mean_a = std_a = min_a = max_a = d_ppm = 0.0

        # Intensity fraction for this phase
        total_intensity = sum(sum(phase_stats[pp.name]['intensities'])
                              for pp in phases
                              if phase_stats[pp.name]['intensities'])
        phase_intensity = sum(st['intensities']) if st['intensities'] else 0
        intensity_frac = phase_intensity / total_intensity if total_intensity > 0 else 0

        # Classification logic:
        #   - exclusive_total > 0 but exclusive_detected == 0 means ALL
        #     this phase's unique peaks failed detection — its signal is
        #     entirely explained by overlap with another phase → capped
        #     at MARGINAL.
        #   - Otherwise, PRESENT if exclusive ratio ≥ 30%, or strong
        #     overall coverage + intensity.
        has_exclusive_opportunity = excl_tot > 0
        has_exclusive_evidence = excl_det > 0

        # Check if any of the first 2 (lowest-angle) rings were detected.
        # Low-angle peaks are always strongest; if both missed, phase is
        # likely not real.
        sorted_rings = sorted(st['ring_order'], key=lambda x: x[0])
        first_n = min(2, len(sorted_rings))
        first_rings_detected = any(d for _, d in sorted_rings[:first_n])

        if has_exclusive_opportunity and not has_exclusive_evidence:
            status = "⚠️  MARGINAL" if det > 0 else "❌ ABSENT"
        elif not first_rings_detected and det > 0:
            # Only high-order peaks matched — likely coincidental
            status = "⚠️  MARGINAL"
        elif (ratio >= 0.3
                or (pct >= 50 and intensity_frac >= 0.2)
                or (pct >= 40 and intensity_frac >= 0.4)):
            status = "✅ PRESENT"
        elif det > 0:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ ABSENT"

        auc_str = f"{phase_intensity:10.1f}" if phase_intensity > 0 else f"{'--':>10}"
        frac_str = f"{intensity_frac*100:5.1f}%" if phase_intensity > 0 else f"{'--':>6}"
        excl_str = f"{n_outliers:>4}" if n_outliers > 0 else f"{'--':>4}"

        conf = compute_confidence(det, tot, excl_det, excl_tot,
                                  a_values, p.lattice_a,
                                  intensity_frac, first_rings_detected)

        if mean_a > 0:
            line = (f"  {p.name:<8} {coverage:<6} {pct:>4.0f}%  {excl_det:<5}"
                    f"{mean_a:<10.4f} {std_a:<10.4f} "
                    f"{min_a:<10.4f} {max_a:<10.4f} "
                    f"{d_ppm:<12.1f} "
                    f"{auc_str} {frac_str}  {excl_str}  {conf:>4}  {status}")
        else:
            line = (f"  {p.name:<8} {coverage:<6} {pct:>4.0f}%  {excl_det:<5}"
                    f"{'--':<10} {'--':<10} "
                    f"{'--':<10} {'--':<10} "
                    f"{'--':<12} "
                    f"{auc_str} {frac_str}  {excl_str}  {conf:>4}  {status}")
        print(line, file=out)
        if summary_out is not None:
            print(line, file=summary_out)

    # ── Table C: Per-phase intensity statistics ───────────────────────
    has_any = any(phase_stats[p.name]['intensities'] for p in phases)
    if has_any:
        print(file=out)
        print("=" * 120, file=out)
        print("  INTENSITY STATISTICS", file=out)
        print("=" * 120, file=out)
        header = (f"{'Phase':<8} {'Sum AUC':>14} {'Mean AUC':>14} "
                  f"{'Max AUC':>14} {'Min AUC':>14} "
                  f"{'Frac of Total':>14}")
        print(header, file=out)
        print("-" * 120, file=out)

        total_intensity = sum(sum(phase_stats[p.name]['intensities'])
                              for p in phases)
        for p in phases:
            ints = phase_stats[p.name]['intensities']
            if ints:
                s = sum(ints)
                frac = s / total_intensity * 100 if total_intensity > 0 else 0
                print(f"  {p.name:<8} {s:>14.1f} {stats_mod.mean(ints):>14.1f} "
                      f"{max(ints):>14.1f} {min(ints):>14.1f} "
                      f"{frac:>13.1f}%", file=out)
            else:
                print(f"  {p.name:<8} {'--':>14} {'--':>14} "
                      f"{'--':>14} {'--':>14} {'--':>14}", file=out)

    print(file=out)
    if summary_out is not None:
        print(file=summary_out)


def phase_summary_header() -> str:
    """Return the column header block for the Phase Summary table."""
    col = (f"  {'Phase':<8} {'Det':<6} {'Cov':<6} {'Uniq':<5}"
           f"{'Mean a(Å)':<10} {'Std a(Å)':<10} "
           f"{'Min a(Å)':<10} {'Max a(Å)':<10} "
           f"{'Δa/a(ppm)':<12} "
           f"{'Sum AUC':>10} {'Frac':>6}  {'Excl':>4}  {'Conf':>4}  {'Status'}")
    W = len(col) + 4
    return (f"{'=' * W}\n"
            f"  PHASE SUMMARY\n"
            f"{'=' * W}\n"
            f"{col}\n"
            f"{'-' * W}")


# =========================================================================
# Data file resolution
# =========================================================================

def resolve_data_files(data_fns, start_nr, end_nr, data_folder):
    """Resolve the list of data files from the various input modes.

    Modes (in priority order):
    1. -dataFolder: all TIFF/HDF5 files in the folder (sorted)
    2. -startNr/-endNr with -dataFN template: number substitution
    3. -dataFN as explicit file list

    For mode 2, the number in the template filename is detected and
    replaced with each number in [startNr, endNr], zero-padded to
    match the original digit width.
    """


    # Mode 1: folder scan
    if data_folder:
        folder = Path(data_folder).resolve()
        if not folder.is_dir():
            print(f"ERROR: Data folder not found: {folder}")
            sys.exit(1)
        exts = ('.tif', '.tiff', '.hdf', '.hdf5', '.h5', '.cbf')
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        )
        if not files:
            print(f"ERROR: No data files found in {folder}")
            sys.exit(1)
        return files

    # Mode 2: number range from template
    if start_nr is not None and end_nr is not None:
        if not data_fns or len(data_fns) != 1:
            print("ERROR: -startNr/-endNr requires exactly one -dataFN as template")
            sys.exit(1)
        template = str(Path(data_fns[0]).resolve())

        # Find the last group of digits in the filename (the frame number)
        basename = Path(template).name
        matches = list(re.finditer(r'\d+', basename))
        if not matches:
            print(f"ERROR: No number found in filename: {basename}")
            sys.exit(1)
        last_match = matches[-1]
        digit_width = len(last_match.group())
        prefix = basename[:last_match.start()]
        suffix = basename[last_match.end():]
        parent = str(Path(template).parent)

        files = []
        for nr in range(start_nr, end_nr + 1):
            padded = str(nr).zfill(digit_width)
            fname = f"{prefix}{padded}{suffix}"
            fpath = Path(parent) / fname
            if fpath.exists():
                files.append(fpath)
            else:
                print(f"  WARNING: File not found, skipping: {fpath.name}")
        if not files:
            print(f"ERROR: No files found in range [{start_nr}, {end_nr}]")
            sys.exit(1)
        print(f"  Number range: {len(files)} files "
              f"({start_nr}–{end_nr}, {digit_width}-digit padding)")
        return files

    # Mode 3: explicit list
    files = [Path(f).resolve() for f in data_fns]
    return files


# =========================================================================
# Per-file worker  (used by both sequential and parallel modes)
# =========================================================================

def process_single_file(data_file: Path, work_dir: Path, cur_param: Path,
                        peak_params_src: Path, rings: List[RingEntry],
                        n_peaks: int, geom: dict, dark_file: Optional[Path],
                        n_cpus: int, backend: str, snr_threshold: float,
                        rel_intensity_threshold: float,
                        phases: List[PhaseInfo],
                        roi_padding: int = 30,
                        mult_factor: float = 0.0,
                        ) -> Tuple[str, str, str, dict]:
    """Process a single data file.

    Returns ``(log_text, results_text, summary_text, timings)``.
    *timings* is a dict with per-stage wall-clock seconds.
    """
    timings: dict = {}
    log: List[str] = []

    def _log(msg):
        log.append(msg)

    t_total = time.monotonic()
    _log(f"\n[3/4] Running {backend.upper()} integration + peak fitting...")

    if backend == 'cpu':
        # Ensure peak_params is in the per-file directory
        file_peak_params = work_dir / "peak_params.txt"
        if file_peak_params != peak_params_src:
            shutil.copy2(str(peak_params_src), str(file_peak_params))

        # ── Direct mode: skip zarr creation ────────────────────────
        # DetectorMapper must have already run (Map.bin/nMap.bin in work_dir)
        t0 = time.monotonic()

        integrator = MIDAS_BIN / "IntegratorZarrOMP"
        cmd = [
            str(integrator),
            "-paramFN", str(cur_param.resolve()),
            "-dataFN", str(data_file.resolve()),
            "-nCPUs", "1",
            "-PeakParamsFN", str(file_peak_params.resolve()),
        ]
        if dark_file and dark_file.exists():
            cmd.extend(["-darkFN", str(dark_file.resolve())])

        _log("  Running IntegratorZarrOMP (direct mode)...")
        stdout = run_cmd(cmd, cwd=str(work_dir.resolve()), log=log)

        # Diagnostic lines
        for line in stdout.split('\n'):
            stripped = line.strip()
            if stripped and any(k in stripped for k in
                               ['peak', 'Peak', 'nPeaks', 'nRBins', 'PeakFit',
                                'Warning', 'Error', 'REJECTED', 'FAILED']):
                _log(f"    ▸ {stripped}")

        fit_bin = work_dir / "fit.bin"
        if fit_bin.exists():
            _log(f"  fit.bin: {fit_bin.stat().st_size} bytes")
        else:
            _log("  WARNING: fit.bin not generated")

        timings['integrate'] = time.monotonic() - t0
    else:
        # GPU backend
        t0 = time.monotonic()
        fit_bin = run_gpu_pipeline(data_file, dark_file,
                                  cur_param, rings, work_dir,
                                  n_cpus, roi_padding)
        timings['gpu_pipeline'] = time.monotonic() - t0

    # ── Parse results and report ─────────────────────────────────
    t0 = time.monotonic()
    _log(f"\n[4/4] Analyzing fit results...")
    fits = read_fit_bin(fit_bin, n_peaks)
    if not fits:
        timings['analysis'] = time.monotonic() - t0
        timings['total'] = time.monotonic() - t_total
        _log(f"WARNING: No fit results for {data_file.name}")
        return "\n".join(log), "", "", timings, []

    _log(f"  Read {len(fits)} peak fit results")

    # Per-eta-bin filtering (if MultFactor enabled and per-eta CSV exists)
    filtered_stats = None
    if mult_factor > 0:
        fit_per_eta_csv = work_dir / "fit_per_eta.csv"
        per_eta = read_fit_per_eta(fit_per_eta_csv)
        if per_eta:
            filtered_stats = compute_filtered_peak_stats(
                per_eta, rings, geom, mult_factor)
            n_filtered = sum(1 for v in filtered_stats.values()
                             if v['n_excluded'] > 0)
            _log(f"  Per-eta filtering (MultFactor={mult_factor}): "
                 f"{len(per_eta)} peaks, "
                 f"{n_filtered} peaks had outlier bins rejected")

    results_buf = io.StringIO()
    summary_buf = io.StringIO()
    print_results(rings, fits, geom, snr_threshold,
                  rel_intensity_threshold, phases,
                  out=results_buf, summary_out=summary_buf,
                  mult_factor=mult_factor,
                  filtered_stats=filtered_stats)
    timings['analysis'] = time.monotonic() - t0
    timings['total'] = time.monotonic() - t_total

    # Build peak table rows
    peak_rows = build_peak_rows(data_file.name, rings, fits, geom,
                                snr_threshold, rel_intensity_threshold,
                                filtered_stats=filtered_stats)

    # Append timing to log
    parts = []
    for k in ('zarr', 'integrate', 'gpu_pipeline', 'analysis'):
        if k in timings:
            parts.append(f"{k}={timings[k]:.2f}s")
    _log(f"  Timing: {', '.join(parts)}  total={timings['total']:.2f}s")

    return "\n".join(log), results_buf.getvalue(), summary_buf.getvalue(), timings, peak_rows


# =========================================================================
# Helper functions for main
# =========================================================================

def prepare_work_dirs(data_files, base_work_dir, work_param, parallel, backend):
    """Create per-file work directories, linking shared Map.bin if needed."""
    n_files = len(data_files)
    file_work_dirs = []
    file_params = []
    for data_file in data_files:
        if n_files > 1:
            wd = base_work_dir / data_file.stem
            wd.mkdir(parents=True, exist_ok=True)
            fp = wd / "phase_id_params.txt"
            with open(work_param) as fin, open(fp, 'w') as fout:
                for line in fin:
                    if line.strip().startswith('Folder '):
                        fout.write(f"Folder {wd}\n")
                    else:
                        fout.write(line)
            # Hard-link Map.bin/nMap.bin if in parallel mode (zero-copy)
            if parallel and backend == 'cpu':
                for mf in ("Map.bin", "nMap.bin"):
                    src = base_work_dir / mf
                    dst = wd / mf
                    if not dst.exists():
                        os.link(str(src), str(dst))
            file_work_dirs.append(wd)
            file_params.append(fp)
        else:
            file_work_dirs.append(base_work_dir)
            file_params.append(work_param)
    return file_work_dirs, file_params


def dispatch_files(data_files, file_work_dirs, file_params, peak_params,
                   rings, n_peaks, geom, dark_file, n_cpus,
                   backend, snr_threshold, rel_intensity_threshold,
                   phases, roi_padding, parallel, n_workers,
                   mult_factor=0.0):
    """Run per-file processing, returning results in file order.

    Returns a list of (data_file, log_text, results_text, summary_text,
    timings, peak_rows) tuples ordered by file index.
    """
    n_files = len(data_files)
    if parallel:
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers) as pool:
            for idx, (df, wd, fp) in enumerate(
                    zip(data_files, file_work_dirs, file_params)):
                fut = pool.submit(
                    process_single_file,
                    df, wd, fp, peak_params, rings, n_peaks, geom,
                    dark_file, n_cpus, backend,
                    snr_threshold, rel_intensity_threshold,
                    phases, roi_padding, mult_factor,
                )
                futures[fut] = (idx, df)

        results_by_idx = {}
        for fut in futures:
            idx, df = futures[fut]
            try:
                log_text, results_text, summary_text, tm, peak_rows = \
                    fut.result()
                results_by_idx[idx] = (df, log_text, results_text,
                                       summary_text, tm, peak_rows)
            except Exception as exc:
                results_by_idx[idx] = (df, f"  ERROR: {exc}", "", "", {}, [])

        return [results_by_idx[i] for i in range(n_files)]
    else:
        results = []
        for df, wd, fp in zip(data_files, file_work_dirs, file_params):
            log_text, results_text, summary_text, tm, peak_rows = \
                process_single_file(
                    df, wd, fp, peak_params,
                    rings, n_peaks, geom, dark_file, n_cpus,
                    backend, snr_threshold,
                    rel_intensity_threshold, phases, roi_padding,
                    mult_factor,
                )
            results.append((df, log_text, results_text, summary_text,
                            tm, peak_rows))
        return results


def emit_file_result(idx, n_files, df_name, summary_text, tm, multi=True):
    """Print compact per-file result to screen."""
    t_file = tm.get('total', 0)
    timing_parts = []
    for k in ('zarr', 'integrate', 'gpu_pipeline', 'analysis'):
        if k in tm:
            timing_parts.append(f"{k}={tm[k]:.2f}s")
    timing_str = (f"  [{', '.join(timing_parts)}, total={t_file:.2f}s]"
                  if timing_parts else "")
    if multi:
        qprint(f"  ── [{idx+1}/{n_files}] {df_name}{timing_str}")
        if summary_text.strip():
            qprint(summary_text, end="")
        else:
            qprint("  (no results)")
    else:
        if timing_str:
            qprint(f"  {df_name}{timing_str}")
        qprint(summary_text)


def write_results_file(output_path, parts, fmt='table'):
    """Save results to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'json':
        # For JSON, the caller already built a JSON string in parts[0]
        with open(output_path, 'w') as f:
            f.write(parts[0])
    else:
        with open(output_path, 'w') as f:
            f.write("\n".join(parts))


def build_json_results(data_files, results_ordered, rings, phases, geom,
                       snr_threshold, rel_intensity_threshold):
    """Build structured JSON output from results."""
    phase_nominal = {p.name: p.lattice_a for p in phases}
    min_sigma = 0.5 * geom['RBinSize']
    output = {
        'geometry': geom,
        'phases': [{'name': p.name, 'spacegroup': p.spacegroup,
                    'lattice_a': p.lattice_a} for p in phases],
        'rings': [{'R_px': r.R_px, 'R_um': r.R_um,
                   'hkl_label': r.hkl_label,
                   'phase_names': r.phase_names,
                   'is_overlap': r.is_overlap} for r in rings],
        'files': [],
    }

    for df, log_text, results_text, summary_text, tm, _pk in results_ordered:
        # Re-read fit.bin for structured data
        n_peaks = len(rings)
        fit_bin = Path(log_text.split("fit.bin:")[0]).parent / "fit.bin" \
            if "fit.bin:" in log_text else None

        # Build per-phase stats from fit results
        file_entry = {
            'filename': df.name,
            'timings': tm,
            'phases': [],
        }

        # Parse summary_text for phase data (robust approach)
        for p in phases:
            phase_data = {
                'name': p.name,
                'status': 'UNKNOWN',
                'confidence': 0,
            }
            # Extract from summary text lines
            for line in summary_text.split('\n'):
                if line.strip().startswith(p.name):
                    parts = line.split()
                    if len(parts) >= 4:
                        phase_data['detection'] = parts[1] if len(parts) > 1 else ''
                        # Find status keyword
                        if 'PRESENT' in line:
                            phase_data['status'] = 'PRESENT'
                        elif 'MARGINAL' in line:
                            phase_data['status'] = 'MARGINAL'
                        elif 'ABSENT' in line:
                            phase_data['status'] = 'ABSENT'
                    break
            file_entry['phases'].append(phase_data)

        output['files'].append(file_entry)

    return json.dumps(output, indent=2)


# =========================================================================
# Main
# =========================================================================


def main():
    global _log_level

    parser = argparse.ArgumentParser(
        description='MIDAS Multi-Phase Identification Tool')
    parser.add_argument('-paramFN', required=True,
                        help='Geometry parameter file')
    parser.add_argument('-dataFN', nargs='+', default=None,
                        help='Data file(s) (TIFF or HDF5). Multiple files '
                             'may be specified. With -startNr/-endNr, '
                             'provide one file as a template.')
    parser.add_argument('-dataFolder', default=None,
                        help='Process all data files in this folder')
    parser.add_argument('-startNr', type=int, default=None,
                        help='Starting file number (requires -dataFN template)')
    parser.add_argument('-endNr', type=int, default=None,
                        help='Ending file number (requires -startNr)')
    parser.add_argument('-phases', required=True,
                        help='Phase definitions file (name SG a)')
    parser.add_argument('-darkFN', default=None,
                        help='Dark frame file')
    parser.add_argument('-nCPUs', type=int, default=4,
                        help='Number of CPUs per integration job (default: 4)')
    parser.add_argument('-backend', choices=['cpu', 'gpu'], default='cpu',
                        help='Backend: cpu (IntegratorZarrOMP) or gpu '
                             '(IntegratorFitPeaksGPUStream)')
    parser.add_argument('--snr-threshold', type=float, default=5.0,
                        help='SNR threshold for peak detection (default: 5.0)')
    parser.add_argument('--rel-intensity-threshold', type=float, default=0.01,
                        help='Min Imax as fraction of strongest peak '
                             '(default: 0.01 = 1%%)')
    parser.add_argument('--mult-factor', type=float, default=None,
                        help='Outlier rejection factor for lattice '
                             'parameters (like CalibrantPanelShiftsOMP). '
                             'Rings with |Δa/a| > MultFactor × mean(|Δa/a|) '
                             'are excluded. Overrides MultFactor from param '
                             'file. (default: 0 = disabled)')
    parser.add_argument('--max-rings', type=int, default=20,
                        help='Max rings per phase (default: 20)')
    parser.add_argument('--roi-padding', type=int, default=30,
                        help='Peak fit ROI half-width in bins (default: 30)')
    parser.add_argument('--merge-threshold', type=float, default=None,
                        help='Ring merge threshold in pixels '
                             '(default: 2 × RBinSize)')
    parser.add_argument('--keep-work-dir', action='store_true',
                        help='Keep temp working directory')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Use a specific working directory')
    parser.add_argument('--multi-cpu', type=int, default=0, metavar='N',
                        help='Process N files in parallel. Uses '
                             'DetectorMapper (non-Zarr) once, then runs '
                             'integration in parallel. (default: 0 = sequential)')
    parser.add_argument('--output', type=str, default=None, metavar='FILE',
                        help='Save results to this file '
                             '(default: <work-dir>/phase_id_results.txt)')
    parser.add_argument('--format', choices=['table', 'json'], default='table',
                        help='Output format: table (default) or json')
    parser.add_argument('--single-phase', action='store_true',
                        help='Single-phase mode: each line in -phases maps '
                             'to one data file (1:1). Output is a minimal '
                             'peak table per file: pixel, 2theta, FWHM, intensity.')
    parser.add_argument('--plot', action='store_true',
                        help='(Single-phase only) Show measured vs calculated '
                             'lineout plot for each file.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output except the phase summary table')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed diagnostic output')
    args = parser.parse_args()

    # Set log level
    if args.quiet:
        _log_level = QUIET
    elif args.verbose:
        _log_level = VERBOSE

    param_file = Path(args.paramFN).resolve()
    phases_file = Path(args.phases).resolve()
    dark_file = Path(args.darkFN).resolve() if args.darkFN else None

    if not param_file.exists():
        print(f"ERROR: Parameter file not found: {param_file}")
        sys.exit(1)
    if not phases_file.exists():
        print(f"ERROR: Phases file not found: {phases_file}")
        sys.exit(1)

    # Validate input mode
    if not args.dataFN and not args.dataFolder:
        print("ERROR: Must specify either -dataFN or -dataFolder")
        sys.exit(1)
    if args.startNr is not None and args.endNr is None:
        print("ERROR: -startNr requires -endNr")
        sys.exit(1)
    if args.endNr is not None and args.startNr is None:
        print("ERROR: -endNr requires -startNr")
        sys.exit(1)

    # Resolve data files
    data_files = resolve_data_files(args.dataFN, args.startNr, args.endNr,
                                    args.dataFolder)
    for df in data_files:
        if not df.exists():
            print(f"ERROR: Data file not found: {df}")
            sys.exit(1)

    # Read geometry
    geom = read_geometry(param_file)
    merge_threshold = (args.merge_threshold if args.merge_threshold is not None
                       else 2.0 * geom['RBinSize'])

    # Create base work directory
    if args.work_dir:
        base_work_dir = Path(args.work_dir)
        base_work_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_work_dir = Path(tempfile.mkdtemp(prefix='midas_phase_id_'))

    parallel = args.multi_cpu > 0 and len(data_files) > 1
    n_workers = min(args.multi_cpu, len(data_files)) if parallel else 1

    # Determine output file path
    output_path = (Path(args.output) if args.output
                   else base_work_dir / "phase_id_results.txt")

    # ── Header (file only, compact screen) ──────────────────────────
    header_buf = io.StringIO()
    def _hdr(msg=""):
        print(msg, file=header_buf)

    _hdr("=" * 70)
    _hdr("  MIDAS Multi-Phase Identification")
    _hdr("=" * 70)
    _hdr(f"  Parameter file: {param_file}")
    _hdr(f"  Data file(s):   {len(data_files)} file(s)")
    for df in data_files:
        _hdr(f"                  {df.name}")
    _hdr(f"  Phases file:    {phases_file}")
    _hdr(f"  Backend:        {args.backend}")
    _hdr(f"  CPUs/job:       {args.nCPUs}")
    if parallel:
        _hdr(f"  Parallel jobs:  {n_workers}")
    _hdr(f"  SNR threshold:  {args.snr_threshold}")
    _hdr(f"  Max rings/phase:{args.max_rings}")
    _hdr(f"  Merge threshold:{merge_threshold:.2f} px")
    _hdr(f"  Work dir:       {base_work_dir}")
    _hdr(f"  Output file:    {output_path}")
    _hdr(f"  Geometry: px={geom['px']}, Lsd={geom['Lsd']:.1f} µm, "
         f"λ={geom['Wavelength']:.6f} Å")
    _hdr()

    # Compact screen header
    mode_str = f", {n_workers} parallel" if parallel else ""
    qprint(f"  MIDAS Phase ID: {len(data_files)} file(s), "
           f"{args.backend.upper()}{mode_str}")
    qprint(f"  Output → {output_path}")

    t_wall_start = time.monotonic()

    try:
        # ==============================================================
        # Step 1: Parse phases and predict rings
        # ==============================================================
        t0 = time.monotonic()
        _hdr("[1/4] Parsing phases and predicting ring positions...")
        phases = parse_phases_file(phases_file)
        if not phases:
            print("ERROR: No phases found in phases file")
            sys.exit(1)

        for p in phases:
            _hdr(f"  Phase: {p.name}  SG={p.spacegroup}  a={p.lattice_a} Å")

        # Determine MultFactor: CLI overrides param file
        mult_factor = (args.mult_factor if args.mult_factor is not None
                       else geom.get('MultFactor', 0.0))

        # ── Single-phase mode validation ──────────────────────────
        if args.single_phase:
            if len(phases) != len(data_files):
                print(f"ERROR: --single-phase requires phases file to have "
                      f"exactly {len(data_files)} entries (one per data file), "
                      f"got {len(phases)}")
                sys.exit(1)

        # ==============================================================
        # SINGLE-PHASE MODE
        # ==============================================================
        if args.single_phase:
            qprint(f"  Single-phase mode: {len(data_files)} file(s), "
                   f"1 phase per file")

            # Create working param file with absolute paths
            work_param = base_work_dir / "phase_id_params.txt"
            param_dir = param_file.parent
            with open(param_file) as fin, open(work_param, 'w') as fout:
                for line in fin:
                    parts = line.strip().split()
                    if not parts:
                        fout.write(line)
                        continue
                    key = parts[0]
                    if key in ('MaskFile', 'MaskFN') and len(parts) > 1:
                        abs_path = (param_dir / parts[1]).resolve()
                        fout.write(f"{key} {abs_path}\n")
                    elif key == 'Dark' and len(parts) > 1:
                        abs_path = (param_dir / parts[1]).resolve()
                        fout.write(f"Dark {abs_path}\n")
                    elif key == 'Folder':
                        fout.write(f"Folder {base_work_dir}\n")
                    else:
                        fout.write(line)

            # Predict rings per unique phase and cache mapper results
            phase_ring_cache = {}   # (name, sg, a) → List[RingEntry]

            # DetectorMapper depends only on geometry (from param file),
            # not on which phase/peaks we're looking for.  Run once.
            mapper_done = False

            t_rings = time.monotonic() - t0
            n_files = len(data_files)
            lineout_files = []  # (filename, lineout_path) for --plot

            for idx, (df, phase) in enumerate(zip(data_files, phases)):
                phase_key = (phase.name, phase.spacegroup, phase.lattice_a)
                qprint(f"  [{idx+1}/{n_files}] {df.name}  →  {phase.name} "
                       f"(SG={phase.spacegroup}, a={phase.lattice_a} Å)")

                # Predict rings (cached per unique phase)
                if phase_key not in phase_ring_cache:
                    refs = predict_rings_for_phase(phase, param_file, geom)
                    refs = refs[:args.max_rings]
                    rings = merge_and_deduplicate(refs, merge_threshold)
                    phase_ring_cache[phase_key] = rings
                    qprint(f"    Predicted {len(rings)} rings")
                else:
                    rings = phase_ring_cache[phase_key]

                n_peaks = len(rings)

                # Per-file work directory
                file_dir = base_work_dir / f"file_{idx:04d}"
                file_dir.mkdir(parents=True, exist_ok=True)

                # Write per-file peak_params.txt
                file_peak_params = file_dir / "peak_params.txt"
                write_peak_params(rings, file_peak_params, args.roi_padding)

                # Copy work param
                file_param = file_dir / "phase_id_params.txt"
                shutil.copy2(str(work_param), str(file_param))

                # Run DetectorMapper ONCE (geometry-only, same for all phases)
                if args.backend == 'cpu':
                    if (file_dir / "Map.bin").exists():
                        # Reuse existing Map.bin from previous run
                        if not mapper_done:
                            mapper_done = file_dir
                    elif not mapper_done:
                        qprint(f"    Running DetectorMapper...", end="",
                               flush=True)
                        t0m = time.monotonic()
                        run_detector_mapper(file_param, file_dir)
                        qprint(f" done [{time.monotonic()-t0m:.2f}s]")
                        mapper_done = file_dir  # remember where Map.bin lives
                    else:
                        # Symlink Map.bin + nMap.bin from first run
                        for mf in ('Map.bin', 'nMap.bin'):
                            src = mapper_done / mf
                            dst = file_dir / mf
                            if not dst.exists() and src.exists():
                                os.symlink(str(src), str(dst))

                    if not (file_dir / "Map.bin").exists():
                        print(f"ERROR: DetectorMapper failed for {df.name}")
                        continue

                # Run integrator
                t0i = time.monotonic()
                integrator = MIDAS_BIN / "IntegratorZarrOMP"
                cmd = [
                    str(integrator),
                    "-paramFN", str(file_param.resolve()),
                    "-dataFN", str(df.resolve()),
                    "-nCPUs", "1",
                    "-PeakParamsFN", str(file_peak_params.resolve()),
                ]
                if dark_file and dark_file.exists():
                    cmd.extend(["-darkFN", str(dark_file.resolve())])
                run_cmd(cmd, cwd=str(file_dir.resolve()))
                t_int = time.monotonic() - t0i

                # Read fit results
                fit_bin = file_dir / "fit.bin"
                fits = read_fit_bin(fit_bin, n_peaks)
                if not fits:
                    qprint(f"    WARNING: No fit results")
                    continue

                # Per-eta-bin filtering for single-phase mode
                sp_filtered = None
                if mult_factor > 0:
                    per_eta_csv = file_dir / "fit_per_eta.csv"
                    per_eta = read_fit_per_eta(per_eta_csv)
                    if per_eta:
                        sp_filtered = compute_filtered_peak_stats(
                            per_eta, rings, geom, mult_factor)

                # Write minimal peak table
                out_name = df.stem + "_peaks.txt"
                out_path_sp = base_work_dir / out_name
                write_single_phase_peak_table(fits, rings, geom, out_path_sp,
                                              filtered_stats=sp_filtered)

                # Write lineout comparison (measured vs calculated)
                lineout_name = df.stem + "_lineout.txt"
                lineout_path = base_work_dir / lineout_name
                lineout_bin = file_dir / "lineout.bin"
                lo_result = write_lineout_comparison(
                    fits, geom, lineout_bin, lineout_path,
                    roi_padding=args.roi_padding)

                extra = f", lineout" if lo_result else ""
                qprint(f"    → {out_path_sp.name}  "
                       f"({n_peaks} peaks{extra}, {t_int:.2f}s)")
                if lo_result:
                    lineout_files.append((df.name, lineout_path))

            t_wall = time.monotonic() - t_wall_start
            qprint(f"\n  Total wall time: {t_wall:.2f}s")
            qprint(f"  Peak tables in: {base_work_dir}")

            # Optional plot of measured vs calculated lineouts
            if args.plot and lineout_files:
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.gridspec as gridspec
                    import numpy as np_plot
                    n_plots = len(lineout_files)
                    fig = plt.figure(figsize=(12, 4.5 * n_plots))
                    gs = gridspec.GridSpec(n_plots * 2, 1,
                                          height_ratios=[3, 1] * n_plots,
                                          hspace=0.08)
                    for ax_idx, (fname, lpath) in enumerate(lineout_files):
                        ax_main = fig.add_subplot(gs[ax_idx * 2])
                        ax_resid = fig.add_subplot(gs[ax_idx * 2 + 1],
                                                   sharex=ax_main)
                        tth, meas, calc = [], [], []
                        with open(lpath) as lf:
                            next(lf)  # skip header
                            for line in lf:
                                parts = line.split()
                                if len(parts) >= 3:
                                    tth.append(float(parts[0]))
                                    meas.append(float(parts[1]))
                                    try:
                                        calc.append(float(parts[2]))
                                    except ValueError:
                                        calc.append(float('nan'))
                        tth_a = np_plot.array(tth)
                        meas_a = np_plot.array(meas)
                        calc_a = np_plot.array(calc)
                        resid = meas_a - calc_a  # NaN where calc is NaN

                        # Main plot
                        ax_main.scatter(tth_a, meas_a, s=2, alpha=0.5,
                                        label='Measured', color='steelblue')
                        ax_main.plot(tth_a, calc_a, linewidth=1.2,
                                     label='Calculated', color='crimson')
                        ax_main.set_title(fname, fontsize=10)
                        ax_main.set_ylabel('Intensity')
                        ax_main.set_yscale('log')
                        ax_main.legend(fontsize=8)
                        plt.setp(ax_main.get_xticklabels(), visible=False)

                        # Residual plot
                        ax_resid.plot(tth_a, resid, linewidth=0.8,
                                      color='forestgreen', alpha=0.8)
                        ax_resid.axhline(0, color='gray', linewidth=0.5,
                                         linestyle='--')
                        ax_resid.set_ylabel('Residual')
                        ax_resid.set_xlabel('2θ (°)')
                    fig.tight_layout()
                    plot_path = base_work_dir / "lineout_comparison.png"
                    fig.savefig(str(plot_path), dpi=150)
                    qprint(f"  Plot saved: {plot_path}")
                    plt.show()
                except ImportError:
                    qprint("  WARNING: matplotlib not available, skipping plot")

            return  # Done — skip multi-phase reporting

        # ==============================================================
        # MULTI-PHASE MODE (original behavior)
        # ==============================================================

        # Predict rings for all phases in parallel
        all_reflections = []
        if len(phases) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(phases)) as pool:
                futures = {
                    pool.submit(predict_rings_for_phase, phase,
                                param_file, geom): phase
                    for phase in phases
                }
                phase_refs = {}
                for fut in concurrent.futures.as_completed(futures):
                    phase = futures[fut]
                    refs = fut.result()[:args.max_rings]
                    phase_refs[phase.name] = refs
            # Maintain original phase order
            for phase in phases:
                refs = phase_refs[phase.name]
                all_reflections.extend(refs)
                _hdr(f"\n  Computing rings for {phase.name} "
                     f"(SG={phase.spacegroup}, a={phase.lattice_a} Å)...")
                _hdr(f"    → {len(refs)} rings")
        else:
            phase = phases[0]
            _hdr(f"\n  Computing rings for {phase.name} "
                 f"(SG={phase.spacegroup}, a={phase.lattice_a} Å)...")
            refs = predict_rings_for_phase(phase, param_file, geom)
            refs = refs[:args.max_rings]
            all_reflections.extend(refs)
            _hdr(f"    → {len(refs)} rings")

        # ==============================================================
        # Step 2: Deduplicate overlapping rings
        # ==============================================================
        _hdr(f"\n[2/4] Deduplicating {len(all_reflections)} reflections "
             f"(threshold={merge_threshold:.2f} px)...")
        rings = merge_and_deduplicate(all_reflections, merge_threshold)
        n_overlaps = sum(1 for r in rings if r.is_overlap)
        _hdr(f"  → {len(rings)} deduplicated peaks "
             f"({n_overlaps} overlapping)")
        for ring in rings[:10]:
            label = ring.hkl_label
            phases_str = ",".join(ring.phase_names)
            ol = " [OVERLAP]" if ring.is_overlap else ""
            _hdr(f"    R={ring.R_px:.2f} px  {phases_str}({label}){ol}")
        if len(rings) > 10:
            _hdr(f"    ... and {len(rings) - 10} more")

        t_rings = time.monotonic() - t0
        # Brief screen summary for steps 1-2
        phase_names = ", ".join(p.name for p in phases)
        qprint(f"  Phases: {phase_names}  |  "
               f"{len(rings)} peaks ({n_overlaps} overlapping)  "
               f"[{t_rings:.2f}s]")

        # ==============================================================
        # Step 3+4: Run integration + peak fitting per data file
        # ==============================================================
        # Write peak params file (shared across files)
        peak_params = base_work_dir / "peak_params.txt"
        n_peaks = write_peak_params(rings, peak_params, args.roi_padding)

        # Create a working param file with absolute paths (shared)
        work_param = base_work_dir / "phase_id_params.txt"
        param_dir = param_file.parent
        with open(param_file) as fin, open(work_param, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    fout.write(line)
                    continue
                key = parts[0]
                if key in ('MaskFile', 'MaskFN') and len(parts) > 1:
                    abs_path = (param_dir / parts[1]).resolve()
                    fout.write(f"{key} {abs_path}\n")
                elif key == 'Dark' and len(parts) > 1:
                    abs_path = (param_dir / parts[1]).resolve()
                    fout.write(f"Dark {abs_path}\n")
                elif key == 'Folder':
                    fout.write(f"Folder {base_work_dir}\n")
                else:
                    fout.write(line)

        # ── Run DetectorMapper once if in parallel mode ──────────
        t_mapper = 0.0
        if args.backend == 'cpu':
            map_bin = base_work_dir / "Map.bin"
            nmap_bin = base_work_dir / "nMap.bin"
            if map_bin.exists() and nmap_bin.exists():
                _hdr(f"\n[2.5/4] DetectorMapper: skipped "
                     f"(Map.bin + nMap.bin exist)")
            else:
                _hdr(f"\n[2.5/4] Running DetectorMapper "
                     f"once (shared map)...")
                qprint("  Running DetectorMapper...", end="", flush=True)
                t0 = time.monotonic()
                run_detector_mapper(work_param, base_work_dir)
                t_mapper = time.monotonic() - t0
                if not map_bin.exists() or not nmap_bin.exists():
                    print(" FAILED")
                    print("ERROR: DetectorMapper did not produce "
                          "Map.bin / nMap.bin")
                    sys.exit(1)
                qprint(f" done [{t_mapper:.2f}s]")
            _hdr(f"  Map.bin:  {map_bin.stat().st_size:,} bytes")
            _hdr(f"  nMap.bin: {nmap_bin.stat().st_size:,} bytes")

        # ── Prepare per-file work dirs ──────────────────────────
        file_work_dirs, file_params = prepare_work_dirs(
            data_files, base_work_dir, work_param, parallel, args.backend)

        # Collect all output for the results file
        all_output_parts: List[str] = [header_buf.getvalue()]
        all_timings: List[Tuple[str, dict]] = []


        # ── Dispatch files (unified parallel/sequential) ─────────
        n_files = len(data_files)
        results_ordered = dispatch_files(
            data_files, file_work_dirs, file_params, peak_params,
            rings, n_peaks, geom, dark_file, args.nCPUs,
            args.backend, args.snr_threshold,
            args.rel_intensity_threshold, phases, args.roi_padding,
            parallel, n_workers, mult_factor)

        # ── Emit results ─────────────────────────────────────────
        if n_files > 1 or parallel:
            qprint(f"\n{phase_summary_header()}")
        all_peak_rows: List[dict] = []
        for idx in range(n_files):
            df, log_text, results_text, summary_text, tm, peak_rows = \
                results_ordered[idx]
            emit_file_result(idx, n_files, df.name, summary_text, tm,
                             multi=(n_files > 1))
            full_header = (f"\n{'='*70}\n"
                           f"  [{idx+1}/{n_files}] {df.name}\n"
                           f"{'='*70}")
            all_output_parts.append(full_header)
            all_output_parts.append(log_text)
            all_output_parts.append(results_text)
            all_timings.append((df.name, tm))
            all_peak_rows.extend(peak_rows)

        # ── Timing summary ───────────────────────────────────────
        t_wall = time.monotonic() - t_wall_start
        qprint(f"\n  Total wall time: {t_wall:.2f}s  "
               f"(rings={t_rings:.2f}s"
               + (f", mapper={t_mapper:.2f}s" if t_mapper > 0 else "")
               + f", files={sum(t.get('total', 0) for _, t in all_timings):.2f}s)")

        # ── Save results ─────────────────────────────────────────
        if args.format == 'json':
            json_str = build_json_results(
                data_files, results_ordered, rings, phases, geom,
                args.snr_threshold, args.rel_intensity_threshold)
            write_results_file(output_path, [json_str], fmt='json')
        else:
            write_results_file(output_path, all_output_parts)
        qprint(f"  Results saved to: {output_path}")

        # ── Write combined peak table ─────────────────────────────
        if all_peak_rows:
            peak_base = output_path.with_name('peak_table')
            csv_path, txt_path = write_peak_tables(all_peak_rows, peak_base)
            qprint(f"  Peak table ({len(all_peak_rows)} rows): {csv_path}")
            qprint(f"  Peak table ({len(all_peak_rows)} rows): {txt_path}")

    finally:
        if not args.keep_work_dir and not args.work_dir:
            qprint(f"  Cleaning up: {base_work_dir}")
            shutil.rmtree(base_work_dir, ignore_errors=True)
        else:
            qprint(f"  Work directory: {base_work_dir}")


if __name__ == '__main__':
    main()


