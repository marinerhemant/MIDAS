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
import math
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # macOS: prevent dual-libomp abort
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"

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

def run_cmd(cmd, cwd=None, check=True):
    """Run a command and return stdout."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                            errors='replace')
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
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
             'RMin': 10.0, 'RMax': 1200.0, 'RBinSize': 0.25}
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
    tmp_param = Path(tempfile.mktemp(suffix='_phase.txt'))
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

        # Run GetHKLList
        hkl_bin = MIDAS_BIN / "GetHKLList"
        stdout = run_cmd([str(hkl_bin), str(tmp_param), "--stdout"])

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
                    param_file: Path, work_dir: Path) -> Path:
    """Create a Zarr zip from a TIFF/HDF5 data file."""
    gen_script = SCRIPT_DIR / "ffGenerateZipRefactor.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"ffGenerateZipRefactor.py not found at {gen_script}")

    cmd = [
        sys.executable, str(gen_script),
        '-paramFN', str(param_file),
        '-dataFN', str(data_file),
        '-resultFolder', str(work_dir),
    ]
    if dark_file and dark_file.exists():
        cmd.extend(['-darkFN', str(dark_file)])

    run_cmd(cmd, cwd=str(work_dir))

    zips = list(work_dir.glob("*.MIDAS.zip"))
    if not zips:
        zips = list(work_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No Zarr zip generated in {work_dir}")
    return zips[0]


def run_cpu_pipeline(zip_file: Path, peak_params: Path,
                     work_dir: Path, n_cpus: int) -> Path:
    """Run DetectorMapperZarr + IntegratorZarrOMP. Returns path to fit.bin.

    Skips DetectorMapperZarr if Map.bin and nMap.bin already exist.
    """
    map_bin = work_dir / "Map.bin"
    nmap_bin = work_dir / "nMap.bin"

    if map_bin.exists() and nmap_bin.exists():
        print("  DetectorMapperZarr: skipped (Map.bin + nMap.bin exist)")
        # Validate parameter headers
        from map_header import check_map_header
        check_map_header(map_bin, "Map.bin")
        check_map_header(nmap_bin, "nMap.bin")
    else:
        mapper = MIDAS_BIN / "DetectorMapperZarr"
        print("  Running DetectorMapperZarr...")
        run_cmd([str(mapper), str(zip_file)], cwd=str(work_dir))

    integrator = MIDAS_BIN / "IntegratorZarrOMP"
    print("  Running IntegratorZarrOMP with peak fitting...")
    stdout = run_cmd([str(integrator), str(zip_file), str(n_cpus),
                      str(peak_params)], cwd=str(work_dir))

    # Print diagnostic lines
    for line in stdout.split('\n'):
        stripped = line.strip()
        if stripped and any(k in stripped for k in
                           ['peak', 'Peak', 'nPeaks', 'nRBins', 'PeakFit',
                            'Warning', 'Error', 'REJECTED', 'FAILED']):
            print(f"    ▸ {stripped}")

    fit_bin = work_dir / "fit.bin"
    if fit_bin.exists():
        print(f"  fit.bin: {fit_bin.stat().st_size} bytes")
    else:
        print("  WARNING: fit.bin not generated")
    return fit_bin


# =========================================================================
# GPU backend: integrator_batch_process.py
# =========================================================================

def run_gpu_pipeline(data_file: Path, dark_file: Optional[Path],
                     param_file: Path, rings: List[RingEntry],
                     work_dir: Path, n_cpus: int,
                     roi_padding: int = 30) -> Path:
    """Run GPU pipeline via integrator_batch_process.py. Returns path to fit.bin."""
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


# =========================================================================
# Reporting
# =========================================================================

def print_results(rings: List[RingEntry], fits: List[FitResult],
                  geom: dict, snr_threshold: float,
                  rel_intensity_threshold: float,
                  phases: List[PhaseInfo]):
    """Print per-ring results and per-phase summary with dual filters."""
    import statistics as stats_mod

    phase_nominal = {p.name: p.lattice_a for p in phases}

    # Find global max Imax for relative intensity filter
    global_max_imax = max((f.Imax for f in fits if f.Imax > 0), default=1.0)

    # Determine detection per peak
    n = min(len(rings), len(fits))
    detected_flags = []
    for i in range(n):
        fit = fits[i]
        snr_ok = fit.SNR >= snr_threshold
        rel_ok = fit.Imax >= rel_intensity_threshold * global_max_imax
        detected_flags.append(fit.Imax > 0 and snr_ok and rel_ok)

    # Collect per-phase statistics
    phase_stats = {}
    for p in phases:
        phase_stats[p.name] = {
            'detected': 0, 'total': 0, 'a_values': [], 'intensities': [],
            'exclusive_detected': 0, 'exclusive_total': 0,
        }

    # ── Table A: Per-ring results ──────────────────────────────────────
    print()
    print("=" * 120)
    print("  MULTI-PHASE IDENTIFICATION RESULTS")
    print("=" * 120)
    print(f"  Filters: SNR ≥ {snr_threshold},  "
          f"Imax ≥ {rel_intensity_threshold*100:.0f}% of max "
          f"(= {global_max_imax * rel_intensity_threshold:.0f} counts)")
    print()
    header = (f"{'Phase':<8} {'(hkl)':<8} {'R_theory':>9} {'R_fitted':>9} "
              f"{'Imax':>12} {'BG':>10} {'Sigma':>6} {'SNR':>8} "
              f"{'a_fitted':>9} {'Δa/a(ppm)':>10}  {'Notes'}")
    print(header)
    print("-" * 120)

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
                a_fitted = back_calculate_lattice(
                    fit.Center, ref.h, ref.k, ref.l, geom)
                a_nom = phase_nominal[ref.phase]
                delta_ppm = ((a_fitted - a_nom) / a_nom * 1e6
                             if a_nom > 0 else 0)

                phase_stats[ref.phase]['detected'] += 1
                phase_stats[ref.phase]['a_values'].append(a_fitted)
                phase_stats[ref.phase]['intensities'].append(fit.Imax)
                if not ring.is_overlap:
                    phase_stats[ref.phase]['exclusive_detected'] += 1

                notes = ""
                if ring.is_overlap:
                    others = [r for r in ring.reflections if r.phase != ref.phase]
                    if others:
                        notes = "⚠️ " + "+".join(
                            f"{r.phase}({r.h}{r.k}{r.l})" for r in others)

                print(f"  {ref.phase:<8} {ref.h}{ref.k}{ref.l:<7} "
                      f"{ref.R_px:>9.2f} {fit.Center:>9.2f} "
                      f"{fit.Imax:>12.1f} {fit.BG:>10.1f} "
                      f"{fit.Sigma:>6.3f} {fit.SNR:>8.1f} "
                      f"{a_fitted:>9.4f} {delta_ppm:>10.1f}  {notes}")
        else:
            ref = ring.reflections[0]
            phase_label = ref.phase
            if ring.is_overlap:
                phase_label = "+".join(r.phase for r in ring.reflections)

            # Show why it was rejected
            reason = "NOT DET"
            if fit.Imax > 0:
                if fit.SNR < snr_threshold:
                    reason = f"SNR={fit.SNR:.1f}<{snr_threshold}"
                elif fit.Imax < rel_intensity_threshold * global_max_imax:
                    rel_pct = fit.Imax / global_max_imax * 100
                    reason = f"Imax={rel_pct:.2f}%<{rel_intensity_threshold*100:.0f}%"
            print(f"  {phase_label:<8} {ref.h}{ref.k}{ref.l:<7} "
                  f"{ref.R_px:>9.2f} {'':>9} "
                  f"{'':>12} {'':>10} "
                  f"{'':>6} {'':>8} "
                  f"{'':>9} {'':>10}  {reason}")

    # ── Table B: Per-phase summary ────────────────────────────────────
    print()
    print("=" * 120)
    print("  PHASE SUMMARY")
    print("=" * 120)
    header = (f"{'Phase':<8} {'Detected':<10} {'Coverage':<9} "
              f"{'Mean a(Å)':<10} {'Std a(Å)':<10} "
              f"{'Min a(Å)':<10} {'Max a(Å)':<10} "
              f"{'Δa/a_nom(ppm)':<14} {'Status'}")
    print(header)
    print("-" * 120)

    for p in phases:
        st = phase_stats[p.name]
        det = st['detected']
        tot = st['total']
        coverage = f"{det}/{tot}" if tot > 0 else "0/0"
        pct = (det / tot * 100) if tot > 0 else 0

        excl_det = st['exclusive_detected']
        excl_tot = st['exclusive_total']
        ratio = (excl_det / excl_tot) if excl_tot > 0 else (det / tot if tot > 0 else 0)

        if st['a_values']:
            mean_a = stats_mod.mean(st['a_values'])
            std_a = stats_mod.stdev(st['a_values']) if len(st['a_values']) > 1 else 0.0
            min_a = min(st['a_values'])
            max_a = max(st['a_values'])
            d_ppm = (mean_a - p.lattice_a) / p.lattice_a * 1e6
        else:
            mean_a = std_a = min_a = max_a = d_ppm = 0.0

        if ratio >= 0.3:
            status = "✅ PRESENT"
        elif det > 0:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ ABSENT"

        if mean_a > 0:
            print(f"  {p.name:<8} {coverage:<10} {pct:>5.0f}%   "
                  f"{mean_a:<10.4f} {std_a:<10.4f} "
                  f"{min_a:<10.4f} {max_a:<10.4f} "
                  f"{d_ppm:<14.1f} {status}")
        else:
            print(f"  {p.name:<8} {coverage:<10} {pct:>5.0f}%   "
                  f"{'--':<10} {'--':<10} "
                  f"{'--':<10} {'--':<10} "
                  f"{'--':<14} {status}")

    # ── Table C: Per-phase intensity statistics ───────────────────────
    has_any = any(phase_stats[p.name]['intensities'] for p in phases)
    if has_any:
        print()
        print("=" * 120)
        print("  INTENSITY STATISTICS")
        print("=" * 120)
        header = (f"{'Phase':<8} {'Sum Imax':>14} {'Mean Imax':>14} "
                  f"{'Max Imax':>14} {'Min Imax':>14} "
                  f"{'Frac of Total':>14}")
        print(header)
        print("-" * 120)

        total_intensity = sum(sum(phase_stats[p.name]['intensities'])
                              for p in phases)
        for p in phases:
            ints = phase_stats[p.name]['intensities']
            if ints:
                s = sum(ints)
                frac = s / total_intensity * 100 if total_intensity > 0 else 0
                print(f"  {p.name:<8} {s:>14.1f} {stats_mod.mean(ints):>14.1f} "
                      f"{max(ints):>14.1f} {min(ints):>14.1f} "
                      f"{frac:>13.1f}%")
            else:
                print(f"  {p.name:<8} {'--':>14} {'--':>14} "
                      f"{'--':>14} {'--':>14} {'--':>14}")

    print()


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
    import glob as glob_mod
    import re

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
        print(f"  Folder scan: found {len(files)} files in {folder}")
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
# Main
# =========================================================================

def main():
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
                        help='Number of CPUs (default: 4)')
    parser.add_argument('-backend', choices=['cpu', 'gpu'], default='cpu',
                        help='Backend: cpu (IntegratorZarrOMP) or gpu '
                             '(IntegratorFitPeaksGPUStream)')
    parser.add_argument('--snr-threshold', type=float, default=5.0,
                        help='SNR threshold for peak detection (default: 5.0)')
    parser.add_argument('--rel-intensity-threshold', type=float, default=0.01,
                        help='Min Imax as fraction of strongest peak '
                             '(default: 0.01 = 1%%)')
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
    args = parser.parse_args()

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

    print("=" * 70)
    print("  MIDAS Multi-Phase Identification")
    print("=" * 70)
    print(f"  Parameter file: {param_file}")
    print(f"  Data file(s):   {len(data_files)} file(s)")
    for df in data_files:
        print(f"                  {df.name}")
    print(f"  Phases file:    {phases_file}")
    print(f"  Backend:        {args.backend}")
    print(f"  CPUs:           {args.nCPUs}")
    print(f"  SNR threshold:  {args.snr_threshold}")
    print(f"  Max rings/phase:{args.max_rings}")
    print(f"  Merge threshold:{merge_threshold:.2f} px")
    print(f"  Work dir:       {base_work_dir}")
    print(f"  Geometry: px={geom['px']}, Lsd={geom['Lsd']:.1f} µm, "
          f"λ={geom['Wavelength']:.6f} Å")
    print()

    try:
        # ==============================================================
        # Step 1: Parse phases and predict rings
        # ==============================================================
        print("[1/4] Parsing phases and predicting ring positions...")
        phases = parse_phases_file(phases_file)
        if not phases:
            print("ERROR: No phases found in phases file")
            sys.exit(1)

        for p in phases:
            print(f"  Phase: {p.name}  SG={p.spacegroup}  a={p.lattice_a} Å")

        all_reflections = []
        for phase in phases:
            print(f"\n  Computing rings for {phase.name} "
                  f"(SG={phase.spacegroup}, a={phase.lattice_a} Å)...")
            refs = predict_rings_for_phase(phase, param_file, geom)
            # Limit per phase
            refs = refs[:args.max_rings]
            all_reflections.extend(refs)
            print(f"    → {len(refs)} rings")

        # ==============================================================
        # Step 2: Deduplicate overlapping rings
        # ==============================================================
        print(f"\n[2/4] Deduplicating {len(all_reflections)} reflections "
              f"(threshold={merge_threshold:.2f} px)...")
        rings = merge_and_deduplicate(all_reflections, merge_threshold)
        n_overlaps = sum(1 for r in rings if r.is_overlap)
        print(f"  → {len(rings)} deduplicated peaks "
              f"({n_overlaps} overlapping)")
        for ring in rings[:10]:
            label = ring.hkl_label
            phases_str = ",".join(ring.phase_names)
            ol = " [OVERLAP]" if ring.is_overlap else ""
            print(f"    R={ring.R_px:.2f} px  {phases_str}({label}){ol}")
        if len(rings) > 10:
            print(f"    ... and {len(rings) - 10} more")

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

        for file_idx, data_file in enumerate(data_files):
            n_files = len(data_files)
            # Per-file work directory (if multiple files)
            if n_files > 1:
                work_dir = base_work_dir / data_file.stem
                work_dir.mkdir(parents=True, exist_ok=True)
                # Update Folder in the param for this sub-dir
                file_param = work_dir / "phase_id_params.txt"
                with open(work_param) as fin, open(file_param, 'w') as fout:
                    for line in fin:
                        if line.strip().startswith('Folder '):
                            fout.write(f"Folder {work_dir}\n")
                        else:
                            fout.write(line)
                cur_param = file_param
            else:
                work_dir = base_work_dir
                cur_param = work_param

            if n_files > 1:
                print(f"\n{'='*70}")
                print(f"  [{file_idx+1}/{n_files}] {data_file.name}")
                print(f"{'='*70}")

            print(f"\n[3/4] Running {args.backend.upper()} integration + "
                  f"peak fitting...")

            if args.backend == 'cpu':
                # Copy shared peak_params to per-file dir if needed
                file_peak_params = work_dir / "peak_params.txt"
                if n_files > 1:
                    shutil.copy2(str(peak_params), str(file_peak_params))
                else:
                    file_peak_params = peak_params

                # Create zarr zip
                print("  Generating Zarr zip...")
                zip_file = create_zarr_zip(data_file, dark_file,
                                           cur_param, work_dir)

                # Run pipeline (DetectorMapper skips if Map.bin exists)
                fit_bin = run_cpu_pipeline(zip_file, file_peak_params,
                                          work_dir, args.nCPUs)
            else:
                # GPU backend
                n_peaks = len(rings)
                fit_bin = run_gpu_pipeline(data_file, dark_file, param_file,
                                          rings, work_dir, args.nCPUs,
                                          args.roi_padding)

            # ==============================================================
            # Step 4: Parse results and report
            # ==============================================================
            print(f"\n[4/4] Analyzing fit results...")
            fits = read_fit_bin(fit_bin, n_peaks)
            if not fits:
                print(f"WARNING: No fit results for {data_file.name}")
                continue

            print(f"  Read {len(fits)} peak fit results")
            print_results(rings, fits, geom, args.snr_threshold,
                          args.rel_intensity_threshold, phases)

    finally:
        if not args.keep_work_dir and not args.work_dir:
            print(f"Cleaning up: {base_work_dir}")
            shutil.rmtree(base_work_dir, ignore_errors=True)
        else:
            print(f"Work directory preserved: {base_work_dir}")


if __name__ == '__main__':
    main()
