#!/usr/bin/env python3
"""
MIDAS vs pyFAI Radial Integration Comparison
=============================================
Comprehensive validation and comparison script for the Radial Integration paper.

Experiments:
  1. Varex CeO₂: MIDAS vs pyFAI with zero tilts, ΔR = {1.0, 2.0}
  2. Pilatus CeO₂: MIDAS vs pyFAI with mask, real tilts/distortion
  3. Precision test: float32 vs float64 accumulation on Pilatus data
  4. pyFAI cake aliasing: shows cardinal-angle oscillations in pyFAI cakes

Generates all publication-quality figures for the paper:
  figures/compare_cake_dr*.{pdf,png}
  figures/compare_scatter_dr*.{pdf,png}
  figures/compare_hist_dr*.{pdf,png}
  figures/compare_lineouts_dr*.{pdf,png}
  figures/compare_1d_dr*.{pdf,png}
  figures/pilatus_cake_comparison.{pdf,png}
  figures/pilatus_1d_comparison.{pdf,png}
  figures/pilatus_scatter.{pdf,png}
  figures/precision_f32_vs_f64.{pdf,png}
  figures/pyfai_cake_aliasing.{pdf,png}

Usage:
    python radial_integration_comparison.py
"""

import os
import sys
import subprocess
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# ─── Varex Configuration ──────────────────────────────────────────
DATA_DIR   = '/Users/hsharma/Desktop/analysis/calibration/andrew'
PARAM_FILE = 'ps_local.txt'
DATA_FILE  = 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'
MIDAS_BIN  = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, 'figures')
WORK_BASE  = os.path.join(SCRIPT_DIR, 'test_runs')

# Varex geometry parameters (from ps_local.txt)
BC_Y       = 1447.068247746732   # pixels
BC_Z       = 1468.589839947911   # pixels
LSD_UM     = 900450.947810575482  # micrometers
PX_UM      = 150.0               # micrometers
WL_A       = 0.19582             # Angstroms
NPY        = 2880
NPZ        = 2880

# Integration ranges (Varex)
R_MIN      = 10.25   # pixels
R_MAX      = 1500.25 # pixels
ETA_MIN    = -180.0
ETA_MAX    = 180.0
ETA_BIN    = 1.0     # degrees

# ΔR values to compare
DELTA_RS   = [1.0, 2.0]

# Ring radius for lineout analysis (CeO₂ (220), ~955 px)
R_RING     = 955.0

# ─── Pilatus Configuration ────────────────────────────────────────
PIL_DATA_DIR  = os.path.expanduser('~/opt/MIDAS/FF_HEDM/Example/Calibration')
PIL_DATA_FILE = 'CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif'
PIL_DARK_FILE = 'dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif'
PIL_MASK_FILE = 'mask_upd.tif'
PIL_PARAM_FILE = 'parameters.txt'

# Pilatus geometry (from parameters.txt — calibrated values)
PIL_BC_Y   = 685.485459654125
PIL_BC_Z   = 921.034377043941
PIL_LSD_UM = 657436.895687981043
PIL_PX_UM  = 172.0
PIL_WL_A   = 0.172973
PIL_NPY    = 1475
PIL_NPZ    = 1679
PIL_TX     = 0.0
PIL_TY     = 0.200888234849
PIL_TZ     = 0.446902376310
PIL_P0     = 0.000230535992
PIL_P1     = 0.000172564332
PIL_P2     = -0.000542224078
PIL_P3     = -13.773706892191
PIL_P4     = 0.001909017437
PIL_P5     = 0.0

# Pilatus integration ranges
PIL_R_MIN   = 10
PIL_R_MAX   = 1200
PIL_ETA_BIN = 1.0
PIL_DR      = 0.25

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(WORK_BASE, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════

def read_param_lines(fn):
    with open(fn) as f:
        return f.readlines()


def write_param_file(src_lines, dst_fn, overrides):
    written = set()
    with open(dst_fn, 'w') as f:
        for line in src_lines:
            key = line.strip().split()[0] if line.strip() and not line.strip().startswith('#') else ''
            if key in overrides:
                f.write(f'{key} {overrides[key]}\n')
                written.add(key)
            else:
                f.write(line)
        for key, val in overrides.items():
            if key not in written:
                f.write(f'{key} {val}\n')


def run_midas_case(work_dir, param_lines, overrides, timeout_mapper=300):
    """Create param file, run DetectorMapper + IntegratorZarrOMP."""
    os.makedirs(work_dir, exist_ok=True)

    # Symlink data files
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.tif'):
            dst = os.path.join(work_dir, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(DATA_DIR, fn), dst)

    # Write param file
    all_overrides = {'Folder': os.path.abspath(work_dir)}
    all_overrides.update(overrides)
    pf = os.path.join(work_dir, 'ps_test.txt')
    write_param_file(param_lines, pf, all_overrides)

    mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
    integrator = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')

    # Run DetectorMapper
    map_bin = os.path.join(work_dir, 'Map.bin')
    if not os.path.isfile(map_bin):
        print(f"    Running DetectorMapper ...", end='', flush=True)
        t0 = time.time()
        r1 = subprocess.run([mapper, 'ps_test.txt'],
                            capture_output=True, text=True,
                            timeout=timeout_mapper, cwd=work_dir)
        dt = time.time() - t0
        if r1.returncode != 0 or not os.path.isfile(map_bin):
            print(f" FAILED (rc={r1.returncode}, {dt:.1f}s)")
            if r1.stderr:
                print(f"      stderr: {r1.stderr[:300]}")
            return None
        print(f" done ({dt:.1f}s)")
    else:
        print(f"    DetectorMapper: reusing existing Map.bin")

    # Run IntegratorZarrOMP
    data_path = os.path.join(work_dir, DATA_FILE)
    print(f"    Running IntegratorZarrOMP ...", end='', flush=True)
    t0 = time.time()
    r2 = subprocess.run([integrator, '-paramFN', 'ps_test.txt',
                         '-dataFN', data_path, '-nCPUs', '8'],
                        capture_output=True, text=True, timeout=60,
                        cwd=work_dir)
    dt = time.time() - t0
    caked = os.path.join(work_dir, DATA_FILE + '.caked.hdf')
    if r2.returncode != 0 or not os.path.isfile(caked):
        print(f" FAILED (rc={r2.returncode}, {dt:.1f}s)")
        if r2.stderr:
            print(f"      stderr: {r2.stderr[:300]}")
        return None
    print(f" done ({dt:.1f}s)")
    return caked


def read_caked_hdf(fn):
    """Read MIDAS .caked.hdf → R, Eta, I_normalized, Area (weight sums)."""
    with h5py.File(fn, 'r') as f:
        remap = f['REtaMap'][:]
        I_norm = f['IntegrationResult/FrameNr_0'][:]
    R    = remap[0, :, 0]   # radial bins (pixels)
    Eta  = remap[2, 0, :]   # azimuthal bins (degrees)
    Area = remap[3, :, :]   # per-bin total fractional area (weight sum)
    return R, Eta, I_norm, Area


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════
#  Experiment: Run MIDAS with zero tilts
# ═══════════════════════════════════════════════════════════════════

def run_midas_experiments():
    print("\n" + "=" * 60)
    print("MIDAS Integration (zero tilts)")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    results = {}

    for dr in DELTA_RS:
        tag = f"compare_dr_{str(dr).replace('.', 'p')}"
        work_dir = os.path.join(WORK_BASE, tag)

        neta = int((ETA_MAX - ETA_MIN) / ETA_BIN)
        overrides = {
            'RBinSize': str(dr),
            'RMin': str(R_MIN),
            'RMax': str(R_MAX),
            'EtaBinSize': str(ETA_BIN),
            'EtaMin': str(ETA_MIN),
            'EtaMax': str(ETA_MAX),
            'ty': '0',
            'tz': '0',
            'p0': '0',
            'p1': '0',
            'p2': '0',
            'p3': '0',
            'p4': '0',
            'p5': '0',
            'Normalize': '1',
            'SubPixelLevel': '1',
        }

        print(f"\n  ΔR = {dr} px:")
        caked = run_midas_case(work_dir, base_lines, overrides)
        if caked:
            R, Eta, I_norm, Area = read_caked_hdf(caked)
            # Reconstruct raw (unnormalized) weighted sum
            I_raw = I_norm * Area
            results[dr] = {
                'R': R, 'Eta': Eta,
                'I_norm': I_norm, 'I_raw': I_raw,
                'Area': Area,
            }
            print(f"    Cake shape: {I_norm.shape}, "
                  f"R range: [{R.min():.2f}, {R.max():.2f}], "
                  f"Eta range: [{Eta.min():.2f}, {Eta.max():.2f}]")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Experiment: Run pyFAI with matched parameters
# ═══════════════════════════════════════════════════════════════════

def run_pyfai_experiments():
    print("\n" + "=" * 60)
    print("pyFAI Integration (matched parameters)")
    print("=" * 60)

    try:
        import pyFAI
        try:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        except ImportError:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        import tifffile
        print(f"  pyFAI version: {pyFAI.version}")
    except ImportError:
        print("  pyFAI not installed — skipping.")
        return None

    px_m = PX_UM * 1e-6
    lsd_m = LSD_UM * 1e-6
    wl_m = WL_A * 1e-10

    # Load image and apply MIDAS transforms:
    #   ImTransOpt 2 = flipud, then MIDAS always transposes
    img_path = os.path.join(DATA_DIR, DATA_FILE)
    img_raw = tifffile.imread(img_path).astype(np.float64)
    img = np.flipud(img_raw)   # ImTransOpt 2
    img = img.T                # MIDAS always transposes after ImTransOpt
    print(f"  Image shape after transform: {img.shape}")

    # Dark subtraction
    dark_files = [fn for fn in os.listdir(DATA_DIR)
                  if fn.startswith('dark') and fn.endswith('.tif')]
    if dark_files:
        dark_raw = tifffile.imread(os.path.join(DATA_DIR, dark_files[0])).astype(np.float64)
        dark = np.flipud(dark_raw).T
        img -= dark
        print(f"  Dark subtracted: {dark_files[0]}")

    # pyFAI geometry (zero tilts)
    # In MIDAS after transforms, image is [y, z] with BC = (bc_y, bc_z)
    # pyFAI PONI1 = vertical distance (dim-0) = bc_y * px_m
    # pyFAI PONI2 = horizontal distance (dim-1) = bc_z * px_m
    #
    # CRITICAL: pixel-center convention difference!
    # pyFAI places pixel center at (i+0.5)*pixel_size (half-pixel offset)
    # MIDAS uses i*pixel_size (pixel index IS the center coordinate)
    # We must add 0.5 px to both PONI coordinates to match.
    det = Detector(pixel1=px_m, pixel2=px_m,
                   max_shape=(img.shape[0], img.shape[1]))
    poni1 = (BC_Y + 0.5) * px_m
    poni2 = (BC_Z + 0.5) * px_m
    ai = AzimuthalIntegrator(
        dist=lsd_m,
        poni1=poni1,
        poni2=poni2,
        wavelength=wl_m,
        detector=det
    )
    print(f"  pyFAI setup: dist={lsd_m:.6f}m, "
          f"PONI1={poni1:.6f}m, PONI2={poni2:.6f}m"
          f" (BC+0.5 px for pixel-center convention)")

    r_min_mm = R_MIN * px_m * 1e3
    r_max_mm = R_MAX * px_m * 1e3
    npt_azim = int((ETA_MAX - ETA_MIN) / ETA_BIN)

    results = {}

    for dr in DELTA_RS:
        npt_rad = int((R_MAX - R_MIN) / dr)
        print(f"\n  ΔR = {dr} px (npt_rad={npt_rad}, npt_azim={npt_azim}):")

        try:
            # correctSolidAngle=False: MIDAS does not apply solid-angle
            # correction (cos³(2θ)), so we disable it in pyFAI to match.
            # With it enabled, there is a systematic ~2.5% offset.
            res = ai.integrate2d(
                img, npt_rad=npt_rad, npt_azim=npt_azim,
                unit="r_mm",
                method=("full", "histogram", "cython"),
                radial_range=(r_min_mm, r_max_mm),
                azimuth_range=(ETA_MIN, ETA_MAX),
                correctSolidAngle=False,
            )
            # Convert radial axis from mm to pixels
            R_px = res.radial / (px_m * 1e3)
            Eta = res.azimuthal
            I_norm = res.intensity   # pyFAI normalizes by weight sum

            # Get unnormalized (raw weighted sum) via integrate2d_ng
            # or by multiplying by count. pyFAI's integrate2d normalizes
            # by the sum of weights per bin. We can get raw by computing
            # sum image separately, but for simplicity we'll use the
            # normalized output and note the differences.

            results[dr] = {
                'R': R_px, 'Eta': Eta,
                'I_norm': I_norm,
                'npt_rad': npt_rad,
            }
            print(f"    Cake shape: {I_norm.shape}, "
                  f"R range: [{R_px.min():.2f}, {R_px.max():.2f}] px, "
                  f"Eta range: [{Eta.min():.2f}, {Eta.max():.2f}]°")

        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════════════════
#  Comparison and Plotting
# ═══════════════════════════════════════════════════════════════════

def compute_statistics(midas_2d, pyfai_2d, label=""):
    """Compute pixel-by-pixel comparison statistics."""
    # Mask: require both non-zero, finite, and above a noise floor
    MIN_I = 1.0  # minimum intensity to include in relative stats
    valid = (midas_2d > 0) & (pyfai_2d > MIN_I) & np.isfinite(midas_2d) & np.isfinite(pyfai_2d)
    m = midas_2d[valid]
    p = pyfai_2d[valid]

    if len(m) == 0:
        print(f"  {label}: No valid bins to compare!")
        return {}

    diff = m - p
    rel_diff = diff / p * 100  # percent relative to pyFAI

    stats = {
        'n_valid': len(m),
        'n_total': midas_2d.size,
        'mean_midas': np.mean(m),
        'mean_pyfai': np.mean(p),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mae': np.mean(np.abs(diff)),
        'max_abs_diff': np.max(np.abs(diff)),
        'mean_rel_diff': np.mean(rel_diff),
        'median_rel_diff': np.median(rel_diff),
        'std_rel_diff': np.std(rel_diff),
        'max_rel_diff': np.max(np.abs(rel_diff)),
        'correlation': np.corrcoef(m, p)[0, 1],
        'rel_diff_values': rel_diff,
        'diff_values': diff,
    }

    print(f"\n  === {label} ===")
    print(f"    Valid bins:      {stats['n_valid']} / {stats['n_total']}")
    print(f"    Mean intensity:  MIDAS={stats['mean_midas']:.2f}, "
          f"pyFAI={stats['mean_pyfai']:.2f}")
    print(f"    RMSE:            {stats['rmse']:.4f}")
    print(f"    MAE:             {stats['mae']:.4f}")
    print(f"    Max |diff|:      {stats['max_abs_diff']:.4f}")
    print(f"    Mean rel diff:   {stats['mean_rel_diff']:.4f}%")
    print(f"    Median rel diff: {stats['median_rel_diff']:.4f}%")
    print(f"    Std rel diff:    {stats['std_rel_diff']:.4f}%")
    print(f"    Max |rel diff|:  {stats['max_rel_diff']:.2f}%")
    print(f"    Correlation:     {stats['correlation']:.8f}")

    return stats


def interpolate_pyfai_to_midas_grid(midas_R, midas_Eta, pyfai_R, pyfai_Eta, pyfai_I):
    """Interpolate pyFAI cake onto MIDAS R-Eta grid for pixel comparison."""
    from scipy.interpolate import RegularGridInterpolator

    # pyFAI: I[eta_idx, r_idx], axes are (eta, r)
    # MIDAS: I[r_idx, eta_idx], axes are (r, eta)
    # Create interpolator on pyFAI grid (eta, r)
    interp = RegularGridInterpolator(
        (pyfai_Eta, pyfai_R),
        pyfai_I,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # Evaluate on MIDAS grid
    R_grid, Eta_grid = np.meshgrid(midas_R, midas_Eta, indexing='ij')
    coords = np.column_stack([Eta_grid.ravel(), R_grid.ravel()])
    pyfai_on_midas = interp(coords).reshape(len(midas_R), len(midas_Eta))

    return pyfai_on_midas


def plot_cake_comparison(midas_data, pyfai_data, dr, fname):
    """Side-by-side 2D cakes + difference map."""
    mR, mEta = midas_data['R'], midas_data['Eta']
    mI = midas_data['I_norm']

    pR, pEta = pyfai_data['R'], pyfai_data['Eta']
    pI = pyfai_data['I_norm']

    # Interpolate pyFAI onto MIDAS grid
    pI_interp = interpolate_pyfai_to_midas_grid(mR, mEta, pR, pEta, pI)

    # Compute difference
    diff = mI - pI_interp
    valid = np.isfinite(diff) & (mI > 0) & np.isfinite(pI_interp) & (pI_interp > 1.0)
    rel_diff = np.full_like(diff, np.nan)
    rel_diff[valid] = diff[valid] / pI_interp[valid] * 100

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MIDAS vs pyFAI — ΔR = {dr} px (zero tilts, zero distortion)',
                 fontsize=14, fontweight='bold')

    # Common vmin/vmax from valid data
    all_valid = np.concatenate([mI[mI > 0], pI_interp[np.isfinite(pI_interp) & (pI_interp > 0)]])
    vmin, vmax = np.percentile(all_valid, [1, 99])

    # (a) MIDAS cake
    ax = axes[0, 0]
    im1 = ax.imshow(mI, origin='lower', cmap='viridis', aspect='auto',
                    extent=[mEta.min(), mEta.max(), mR.min(), mR.max()],
                    vmin=vmin, vmax=vmax)
    ax.set_xlabel('η (degrees)')
    ax.set_ylabel('R (pixels)')
    ax.set_title('(a) MIDAS')
    plt.colorbar(im1, ax=ax, label='Intensity')

    # (b) pyFAI cake (on MIDAS grid)
    ax = axes[0, 1]
    im2 = ax.imshow(pI_interp, origin='lower', cmap='viridis', aspect='auto',
                    extent=[mEta.min(), mEta.max(), mR.min(), mR.max()],
                    vmin=vmin, vmax=vmax)
    ax.set_xlabel('η (degrees)')
    ax.set_ylabel('R (pixels)')
    ax.set_title('(b) pyFAI')
    plt.colorbar(im2, ax=ax, label='Intensity')

    # (c) Absolute difference
    ax = axes[1, 0]
    dlim = np.nanpercentile(np.abs(diff[valid]), 99)
    im3 = ax.imshow(diff, origin='lower', cmap='RdBu_r', aspect='auto',
                    extent=[mEta.min(), mEta.max(), mR.min(), mR.max()],
                    vmin=-dlim, vmax=dlim)
    ax.set_xlabel('η (degrees)')
    ax.set_ylabel('R (pixels)')
    ax.set_title('(c) MIDAS − pyFAI')
    plt.colorbar(im3, ax=ax, label='Δ Intensity')

    # (d) Relative difference (%)
    ax = axes[1, 1]
    rlim = min(np.nanpercentile(np.abs(rel_diff[valid]), 95), 50)
    im4 = ax.imshow(rel_diff, origin='lower', cmap='RdBu_r', aspect='auto',
                    extent=[mEta.min(), mEta.max(), mR.min(), mR.max()],
                    vmin=-rlim, vmax=rlim)
    ax.set_xlabel('η (degrees)')
    ax.set_ylabel('R (pixels)')
    ax.set_title('(d) Relative difference (%)')
    plt.colorbar(im4, ax=ax, label='Δ / pyFAI (%)')

    fig.tight_layout()
    save(fig, fname)

    return pI_interp, diff, rel_diff


def plot_scatter(midas_I, pyfai_I, dr, fname):
    """Scatter plot: MIDAS vs pyFAI bin-by-bin."""
    valid = (midas_I > 0) & np.isfinite(pyfai_I) & (pyfai_I > 0)
    m, p = midas_I[valid], pyfai_I[valid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(p, m, s=0.3, alpha=0.2, c='steelblue', rasterized=True)

    lims = [min(p.min(), m.min()), max(p.max(), m.max())]
    ax.plot(lims, lims, 'r-', lw=1, label='1:1 line')
    ax.set_xlabel('pyFAI intensity', fontsize=12)
    ax.set_ylabel('MIDAS intensity', fontsize=12)
    ax.set_title(f'Bin-by-bin comparison — ΔR = {dr} px', fontsize=13)
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation annotation
    corr = np.corrcoef(m, p)[0, 1]
    ax.text(0.05, 0.92, f'r = {corr:.6f}\nN = {len(m):,}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.tight_layout()
    save(fig, fname)


def plot_histogram(rel_diff_values, dr, fname):
    """Histogram of relative differences."""
    v = rel_diff_values[np.isfinite(rel_diff_values)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.linspace(np.percentile(v, 0.5), np.percentile(v, 99.5), 100)
    ax.hist(v, bins=bins, color='steelblue', alpha=0.7, edgecolor='white',
            linewidth=0.3)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.axvline(np.mean(v), color='orange', ls='-', lw=1.5,
               label=f'Mean = {np.mean(v):.3f}%')
    ax.axvline(np.median(v), color='green', ls='-', lw=1.5,
               label=f'Median = {np.median(v):.3f}%')

    ax.set_xlabel('Relative difference (MIDAS − pyFAI) / pyFAI  (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of relative differences — ΔR = {dr} px', fontsize=13)
    ax.legend(fontsize=11)

    fig.tight_layout()
    save(fig, fname)


def plot_lineouts(midas_data, pyfai_data, pyfai_interp, dr, fname):
    """Compare I(η) lineouts at selected R values."""
    mR, mEta = midas_data['R'], midas_data['Eta']
    mI = midas_data['I_norm']

    # Selected R values: near the (220) ring and off-ring
    r_targets = [R_RING, R_RING + 50, 377.0, 617.0]  # CeO₂ ring radii
    r_targets = [r for r in r_targets if r >= mR.min() and r <= mR.max()]

    n = len(r_targets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, r_target in enumerate(r_targets):
        ridx = np.argmin(np.abs(mR - r_target))
        ax = axes[i]

        midas_lineout = mI[ridx, :]
        pyfai_lineout = pyfai_interp[ridx, :]

        ax.plot(mEta, midas_lineout, '-', color='C0', lw=0.8,
                label='MIDAS', alpha=0.8)
        ax.plot(mEta, pyfai_lineout, '-', color='C1', lw=0.8,
                label='pyFAI', alpha=0.8)
        ax.set_ylabel('I (norm.)', fontsize=11)
        ax.set_title(f'R = {mR[ridx]:.1f} px', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('η (degrees)', fontsize=12)
    fig.suptitle(f'I(η) lineouts — ΔR = {dr} px', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, fname)


def plot_1d_profiles(midas_data, pyfai_data, dr, fname):
    """Compare 1D I(R) profiles (azimuthally averaged)."""
    mR = midas_data['R']
    mI = midas_data['I_norm']
    mI_1d = np.nanmean(mI, axis=1)  # average over η

    pR = pyfai_data['R']
    pI = pyfai_data['I_norm']
    pI_1d = np.nanmean(pI, axis=0)  # pyFAI: I[eta, r]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    ax1.plot(mR, mI_1d, '-', color='C0', lw=0.8, label='MIDAS')
    ax1.plot(pR, pI_1d, '-', color='C1', lw=0.8, label='pyFAI', alpha=0.7)
    ax1.set_ylabel('I (norm.)', fontsize=12)
    ax1.set_title(f'1D Radial profiles (azimuthally averaged) — ΔR = {dr} px',
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Residual — interpolate pyFAI to MIDAS R grid
    from scipy.interpolate import interp1d
    pI_interp1d = interp1d(pR, pI_1d, kind='linear', bounds_error=False,
                            fill_value=np.nan)(mR)
    residual = mI_1d - pI_interp1d
    valid = np.isfinite(residual) & (mI_1d > 0)
    rel_residual = np.where(valid, residual / pI_interp1d * 100, np.nan)

    ax2.plot(mR[valid], rel_residual[valid], '-', color='C2', lw=0.5)
    ax2.axhline(0, color='red', ls='--', lw=0.8)
    ax2.set_xlabel('R (pixels)', fontsize=12)
    ax2.set_ylabel('(M−P)/P (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2: Pilatus CeO₂ with mask and real geometry
# ═══════════════════════════════════════════════════════════════════

def run_pilatus_midas():
    """Run MIDAS on Pilatus CeO₂ data with mask and real tilts/distortion."""
    print("\n" + "=" * 60)
    print("MIDAS Pilatus Integration (with mask, real geometry)")
    print("=" * 60)

    work_dir = os.path.join(WORK_BASE, 'pilatus')
    os.makedirs(work_dir, exist_ok=True)

    # Symlink data files
    for fn in [PIL_DATA_FILE, PIL_DARK_FILE, PIL_MASK_FILE,
               'panelshiftsCalibrant.txt']:
        src = os.path.join(PIL_DATA_DIR, fn)
        dst = os.path.join(work_dir, fn)
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst)

    # Build param file
    lines = read_param_lines(os.path.join(PIL_DATA_DIR, PIL_PARAM_FILE))
    overrides = {
        'Folder': os.path.abspath(work_dir),
        'RBinSize': str(PIL_DR),
        'RMin': str(PIL_R_MIN),
        'RMax': str(PIL_R_MAX),
        'EtaBinSize': str(PIL_ETA_BIN),
        'EtaMin': str(ETA_MIN),
        'EtaMax': str(ETA_MAX),
        'Normalize': '1',
        'SolidAngleCorrection': '0',  # match pyFAI correctSolidAngle=False
    }
    pf = os.path.join(work_dir, 'ps_test.txt')
    write_param_file(lines, pf, overrides)

    mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
    integrator = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')

    map_bin = os.path.join(work_dir, 'Map.bin')
    if not os.path.isfile(map_bin):
        print(f"    Running DetectorMapper ...", end='', flush=True)
        t0 = time.time()
        r1 = subprocess.run([mapper, 'ps_test.txt'],
                            capture_output=True, text=True,
                            timeout=600, cwd=work_dir)
        dt = time.time() - t0
        if r1.returncode != 0 or not os.path.isfile(map_bin):
            print(f" FAILED (rc={r1.returncode}, {dt:.1f}s)")
            if r1.stderr:
                print(f"      stderr: {r1.stderr[:300]}")
            return None
        print(f" done ({dt:.1f}s)")
    else:
        print(f"    DetectorMapper: reusing existing Map.bin")

    data_path = os.path.join(work_dir, PIL_DATA_FILE)
    print(f"    Running IntegratorZarrOMP ...", end='', flush=True)
    t0 = time.time()
    r2 = subprocess.run([integrator, '-paramFN', 'ps_test.txt',
                         '-dataFN', data_path, '-nCPUs', '8'],
                        capture_output=True, text=True, timeout=60,
                        cwd=work_dir)
    dt = time.time() - t0
    caked = os.path.join(work_dir, PIL_DATA_FILE + '.caked.hdf')
    if r2.returncode != 0 or not os.path.isfile(caked):
        print(f" FAILED (rc={r2.returncode}, {dt:.1f}s)")
        if r2.stderr:
            print(f"      stderr: {r2.stderr[:300]}")
        return None
    print(f" done ({dt:.1f}s)")

    R, Eta, I_norm, Area = read_caked_hdf(caked)
    return {'R': R, 'Eta': Eta, 'I_norm': I_norm, 'Area': Area}


def run_pilatus_pyfai():
    """Run pyFAI on Pilatus CeO₂ with matched geometry."""
    print("\n" + "=" * 60)
    print("pyFAI Pilatus Integration")
    print("=" * 60)

    try:
        import pyFAI
        try:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        except ImportError:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        import tifffile
    except ImportError:
        print("  pyFAI not installed — skipping.")
        return None

    px_m = PIL_PX_UM * 1e-6
    lsd_m = PIL_LSD_UM * 1e-6
    wl_m = PIL_WL_A * 1e-10

    img_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_DATA_FILE)).astype(np.float64)
    dark_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_DARK_FILE)).astype(np.float64)
    mask_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_MASK_FILE))

    # ImTransOpt 2 = flipud, then transpose
    img = np.flipud(img_raw).T
    dark = np.flipud(dark_raw).T
    img -= dark

    # Mask: in MIDAS mask, 0 = masked pixel, nonzero = valid
    # pyFAI: 1 = masked, 0 = valid — so invert
    mask_midas = np.flipud(mask_raw.astype(np.float64)).T
    mask_pyfai = (mask_midas == 0).astype(np.int8)
    print(f"  Image shape: {img.shape}, mask: {mask_pyfai.sum()} masked pixels")

    det = Detector(pixel1=px_m, pixel2=px_m,
                   max_shape=(img.shape[0], img.shape[1]))
    poni1 = (PIL_BC_Y + 0.5) * px_m
    poni2 = (PIL_BC_Z + 0.5) * px_m
    ai = AzimuthalIntegrator(
        dist=lsd_m,
        poni1=poni1,
        poni2=poni2,
        wavelength=wl_m,
        detector=det
    )

    r_min_mm = PIL_R_MIN * px_m * 1e3
    r_max_mm = PIL_R_MAX * px_m * 1e3
    npt_rad = int((PIL_R_MAX - PIL_R_MIN) / PIL_DR)
    npt_azim = int((ETA_MAX - ETA_MIN) / PIL_ETA_BIN)

    print(f"  npt_rad={npt_rad}, npt_azim={npt_azim}")

    res = ai.integrate2d(
        img, npt_rad=npt_rad, npt_azim=npt_azim,
        unit="r_mm",
        method=("full", "histogram", "cython"),
        radial_range=(r_min_mm, r_max_mm),
        azimuth_range=(ETA_MIN, ETA_MAX),
        correctSolidAngle=False,
        mask=mask_pyfai,
    )
    R_px = res.radial / (px_m * 1e3)
    Eta = res.azimuthal
    I_norm = res.intensity

    return {'R': R_px, 'Eta': Eta, 'I_norm': I_norm}


def plot_pilatus_comparison(midas, pyfai):
    """Generate Pilatus comparison figures.
    
    Note: Because pyFAI cannot model MIDAS's per-panel shifts and distortion
    polynomial, a bin-for-bin 2D cake comparison is not meaningful. We compare
    azimuthally-averaged 1D profiles instead, where geometry differences
    largely cancel out.
    """
    mR, mEta = midas['R'], midas['Eta']
    mI = midas['I_norm']
    pR, pEta = pyfai['R'], pyfai['Eta']
    pI = pyfai['I_norm']

    # 1D profiles — azimuthal average (NaN-aware for masked bins)
    mI_1d = np.nanmean(mI, axis=1)
    pI_1d = np.nanmean(pI, axis=0)  # pyFAI: I[eta, r]

    from scipy.interpolate import interp1d
    pI_interp1d = interp1d(pR, pI_1d, kind='linear', bounds_error=False,
                            fill_value=np.nan)(mR)

    # Stats on 1D profiles
    valid_1d = np.isfinite(mI_1d) & np.isfinite(pI_interp1d) & (mI_1d > 1) & (pI_interp1d > 1)
    if valid_1d.sum() > 0:
        m1d = mI_1d[valid_1d]
        p1d = pI_interp1d[valid_1d]
        rel_1d = (m1d - p1d) / p1d * 100
        corr_1d = np.corrcoef(m1d, p1d)[0, 1]
        print(f"\n  === Pilatus CeO₂ — 1D Profiles ===")
        print(f"    Valid R bins: {valid_1d.sum()} / {len(mR)}")
        print(f"    Mean rel diff: {np.mean(rel_1d):.4f}%")
        print(f"    Std rel diff:  {np.std(rel_1d):.4f}%")
        print(f"    Correlation:   {corr_1d:.8f}")

    # (a) 1D profiles + residual
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    ax1.plot(mR, mI_1d, '-', color='C0', lw=0.8, label='MIDAS')
    ax1.plot(pR, pI_1d, '-', color='C1', lw=0.8, label='pyFAI', alpha=0.7)
    ax1.set_ylabel('I (norm.)', fontsize=12)
    ax1.set_title('Pilatus CeO₂ — 1D Radial Profiles (azimuthal average)',
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    residual = mI_1d - pI_interp1d
    rel_residual = np.full_like(mI_1d, np.nan)
    safe = valid_1d & (np.abs(pI_interp1d) > 1e-6)
    rel_residual[safe] = residual[safe] / pI_interp1d[safe] * 100

    ax2.plot(mR[safe], rel_residual[safe], '-', color='C2', lw=0.5)
    ax2.axhline(0, color='red', ls='--', lw=0.8)
    ax2.set_xlabel('R (pixels)', fontsize=12)
    ax2.set_ylabel('(M−P)/P (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # Annotate mean
    if safe.sum() > 0:
        ax2.text(0.02, 0.9, f'Mean: {np.nanmean(rel_residual[safe]):.3f}%\n'
                 f'Std: {np.nanstd(rel_residual[safe]):.3f}%',
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                 va='top')
    fig.tight_layout()
    save(fig, 'pilatus_1d_comparison')

    # (b) 1D scatter plot
    if valid_1d.sum() > 0:
        fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(pI_interp1d[valid_1d], mI_1d[valid_1d], s=3, alpha=0.4,
                   c='steelblue', rasterized=True)
        lims = [min(pI_interp1d[valid_1d].min(), mI_1d[valid_1d].min()),
                max(pI_interp1d[valid_1d].max(), mI_1d[valid_1d].max())]
        ax.plot(lims, lims, 'r-', lw=1, label='1:1 line')
        ax.set_xlabel('pyFAI I(R)', fontsize=12)
        ax.set_ylabel('MIDAS I(R)', fontsize=12)
        ax.set_title(f'Pilatus CeO₂ — 1D Profile Comparison (ΔR={PIL_DR})',
                     fontsize=13)
        ax.text(0.05, 0.92, f'r = {corr_1d:.6f}\nN = {valid_1d.sum():,}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        fig2.tight_layout()
        save(fig2, 'pilatus_scatter')


# ═══════════════════════════════════════════════════════════════════
#  Experiment 3: Precision test (float32 vs float64)
# ═══════════════════════════════════════════════════════════════════

def run_precision_test():
    """Run IntegratorZarrOMP and IntegratorZarrOMP_f32 on same data."""
    print("\n" + "=" * 60)
    print("Precision Test: float64 vs float32 Accumulation")
    print("=" * 60)

    work_dir = os.path.join(WORK_BASE, 'pilatus')
    if not os.path.isfile(os.path.join(work_dir, 'Map.bin')):
        print("  ERROR: Run Pilatus MIDAS first (need Map.bin)")
        return None

    integrator_f64 = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')
    integrator_f32 = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP_f32')
    if not os.path.isfile(integrator_f32):
        print(f"  ERROR: {integrator_f32} not found. Build with: "
              f"cmake --build . --target IntegratorZarrOMP_f32")
        return None

    data_path = os.path.join(work_dir, PIL_DATA_FILE)
    results = {}

    for label, exe in [('f64', integrator_f64), ('f32', integrator_f32)]:
        # Use different output dir to avoid overlap
        prec_dir = os.path.join(work_dir, f'precision_{label}')
        os.makedirs(prec_dir, exist_ok=True)

        # Symlink map files
        for fn in ['Map.bin', 'nMap.bin', 'binMask.bin', 'MapHeader']:
            src = os.path.join(work_dir, fn)
            dst = os.path.join(prec_dir, fn)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        # Symlink data and dark
        for fn in [PIL_DATA_FILE, PIL_DARK_FILE]:
            src = os.path.join(work_dir, fn)
            dst = os.path.join(prec_dir, fn)
            if not os.path.exists(dst) and os.path.exists(src):
                os.symlink(src, dst)

        # Write param file with Folder = prec_dir
        lines = read_param_lines(os.path.join(PIL_DATA_DIR, PIL_PARAM_FILE))
        overrides = {
            'Folder': os.path.abspath(prec_dir),
            'RBinSize': str(PIL_DR),
            'RMin': str(PIL_R_MIN),
            'RMax': str(PIL_R_MAX),
            'EtaBinSize': str(PIL_ETA_BIN),
            'EtaMin': str(ETA_MIN),
            'EtaMax': str(ETA_MAX),
            'Normalize': '1',
        }
        pf = os.path.join(prec_dir, 'ps_test.txt')
        write_param_file(lines, pf, overrides)

        data_fn = os.path.join(prec_dir, PIL_DATA_FILE)
        print(f"  Running {label.upper()} integrator ...", end='', flush=True)
        t0 = time.time()
        r = subprocess.run([exe, '-paramFN', 'ps_test.txt',
                            '-dataFN', data_fn, '-nCPUs', '8'],
                           capture_output=True, text=True, timeout=60,
                           cwd=prec_dir)
        dt = time.time() - t0
        caked = os.path.join(prec_dir, PIL_DATA_FILE + '.caked.hdf')
        if r.returncode != 0 or not os.path.isfile(caked):
            print(f" FAILED (rc={r.returncode}, {dt:.1f}s)")
            if r.stderr:
                print(f"      stderr: {r.stderr[:300]}")
            continue
        print(f" done ({dt:.1f}s)")
        R, Eta, I_norm, Area = read_caked_hdf(caked)
        results[label] = {'R': R, 'Eta': Eta, 'I_norm': I_norm, 'Area': Area}

    return results


def plot_precision(results):
    """Plot float32 vs float64 precision comparison."""
    if 'f64' not in results or 'f32' not in results:
        print("  Missing precision data — skipping plot.")
        return

    I64 = results['f64']['I_norm']
    I32 = results['f32']['I_norm']
    R = results['f64']['R']

    valid = (I64 > 0) & (I32 > 0) & np.isfinite(I64) & np.isfinite(I32)
    diff = (I32[valid] - I64[valid])
    rel_diff = diff / I64[valid] * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('float32 vs float64 Accumulation Precision Test',
                 fontsize=14, fontweight='bold')

    # (a) Histogram of relative differences
    ax = axes[0, 0]
    bins = np.linspace(np.percentile(rel_diff, 0.1),
                       np.percentile(rel_diff, 99.9), 100)
    ax.hist(rel_diff, bins=bins, color='steelblue', alpha=0.7,
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('(f32 − f64) / f64  (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'(a) Distribution of relative error\n'
                 f'mean={np.mean(rel_diff):.6f}%, '
                 f'std={np.std(rel_diff):.6f}%')

    # (b) Scatter plot
    ax = axes[0, 1]
    subsample = np.random.choice(len(I64[valid]), min(50000, len(I64[valid])),
                                  replace=False)
    ax.scatter(I64[valid][subsample], I32[valid][subsample],
              s=0.3, alpha=0.2, c='steelblue', rasterized=True)
    lims = [I64[valid].min(), I64[valid].max()]
    ax.plot(lims, lims, 'r-', lw=1, label='1:1')
    ax.set_xlabel('f64 intensity', fontsize=11)
    ax.set_ylabel('f32 intensity', fontsize=11)
    ax.set_title(f'(b) Bin-by-bin correlation\nr = {np.corrcoef(I64[valid], I32[valid])[0,1]:.10f}')
    ax.set_aspect('equal')
    ax.legend()

    # (c) Error vs R (radial dependence)
    ax = axes[1, 0]
    R_grid = np.broadcast_to(R[:, None], I64.shape)
    ax.scatter(R_grid[valid], np.abs(rel_diff), s=0.1, alpha=0.1,
              c='steelblue', rasterized=True)
    ax.set_xlabel('R (pixels)', fontsize=11)
    ax.set_ylabel('|relative error| (%)', fontsize=11)
    ax.set_title('(c) Error vs radial position')
    ax.set_yscale('log')

    # (d) CDF of absolute relative error
    ax = axes[1, 1]
    sorted_err = np.sort(np.abs(rel_diff))
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cdf, '-', color='C0', lw=1)
    ax.axvline(0.01, color='gray', ls=':', lw=0.8, label='0.01%')
    ax.axvline(0.1, color='gray', ls='--', lw=0.8, label='0.1%')
    pct_below_001 = np.sum(np.abs(rel_diff) < 0.01) / len(rel_diff) * 100
    pct_below_01 = np.sum(np.abs(rel_diff) < 0.1) / len(rel_diff) * 100
    ax.set_xlabel('|relative error| (%)', fontsize=11)
    ax.set_ylabel('Cumulative fraction (%)', fontsize=11)
    ax.set_title(f'(d) CDF: {pct_below_001:.1f}% below 0.01%, '
                 f'{pct_below_01:.1f}% below 0.1%')
    ax.legend()
    ax.set_xscale('log')

    fig.tight_layout()
    save(fig, 'precision_f32_vs_f64')


# ═══════════════════════════════════════════════════════════════════
#  Experiment 4: pyFAI cake aliasing visualization
# ═══════════════════════════════════════════════════════════════════

def plot_pyfai_cake_aliasing():
    """Generate pyFAI cake plot showing cardinal-angle aliasing oscillations.
    Similar to Fig. 4a in the paper but for pyFAI, demonstrating that the
    oscillation artifact is inherent to histogram-based integration."""
    print("\n" + "=" * 60)
    print("pyFAI Cake Aliasing Visualization")
    print("=" * 60)

    try:
        import pyFAI
        try:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        except ImportError:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        import tifffile
    except ImportError:
        print("  pyFAI not installed — skipping.")
        return

    # Use Pilatus data (panel gaps make aliasing more visible)
    px_m = PIL_PX_UM * 1e-6
    lsd_m = PIL_LSD_UM * 1e-6
    wl_m = PIL_WL_A * 1e-10

    img_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_DATA_FILE)).astype(np.float64)
    dark_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_DARK_FILE)).astype(np.float64)
    mask_raw = tifffile.imread(os.path.join(PIL_DATA_DIR, PIL_MASK_FILE))

    img = np.flipud(img_raw).T
    dark = np.flipud(dark_raw).T
    img -= dark
    mask_pyfai = (np.flipud(mask_raw.astype(np.float64)).T == 0).astype(np.int8)

    det = Detector(pixel1=px_m, pixel2=px_m,
                   max_shape=(img.shape[0], img.shape[1]))
    poni1 = (PIL_BC_Y + 0.5) * px_m
    poni2 = (PIL_BC_Z + 0.5) * px_m
    ai = AzimuthalIntegrator(
        dist=lsd_m, poni1=poni1, poni2=poni2,
        wavelength=wl_m, detector=det
    )

    r_min_mm = PIL_R_MIN * px_m * 1e3
    r_max_mm = PIL_R_MAX * px_m * 1e3
    npt_rad = int((PIL_R_MAX - PIL_R_MIN) / PIL_DR)
    npt_azim = int((ETA_MAX - ETA_MIN) / PIL_ETA_BIN)

    res = ai.integrate2d(
        img, npt_rad=npt_rad, npt_azim=npt_azim,
        unit="r_mm",
        method=("full", "histogram", "cython"),
        radial_range=(r_min_mm, r_max_mm),
        azimuth_range=(ETA_MIN, ETA_MAX),
        correctSolidAngle=False,
        mask=mask_pyfai,
    )
    R_px = res.radial / (px_m * 1e3)
    Eta = res.azimuthal
    I = res.intensity

    # Find a CeO₂ ring — look for peak in 1D profile
    I_1d = np.nanmean(I, axis=0)
    # Use the strongest peak in R range [200, 800]
    r_mask = (R_px > 200) & (R_px < 800)
    peak_idx = np.nanargmax(I_1d[r_mask])
    r_peak = R_px[r_mask][peak_idx]
    print(f"  Strongest ring at R = {r_peak:.1f} px")

    # Extract a narrow R band around the ring
    r_width = 5  # pixels
    r_lo = r_peak - r_width
    r_hi = r_peak + r_width
    r_sel = (R_px >= r_lo) & (R_px <= r_hi)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle('pyFAI Cake: Cardinal-Angle Aliasing Oscillations',
                 fontsize=14, fontweight='bold')

    # (a) Full cake (R vs η) — zoomed to ring
    ax = axes[0]
    cake_slice = I[np.ix_(range(len(Eta)), np.where(r_sel)[0])]
    im = ax.imshow(cake_slice, origin='lower', cmap='viridis', aspect='auto',
                   extent=[R_px[r_sel].min(), R_px[r_sel].max(),
                           Eta.min(), Eta.max()])
    ax.set_xlabel('R (pixels)', fontsize=12)
    ax.set_ylabel('η (degrees)', fontsize=12)
    ax.set_title(f'(a) pyFAI cake — R ∈ [{r_lo:.0f}, {r_hi:.0f}] px')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Mark cardinal angles
    for eta_card in [-180, -90, 0, 90, 180]:
        if ETA_MIN <= eta_card <= ETA_MAX:
            ax.axhline(eta_card, color='red', ls='--', lw=0.5, alpha=0.5)

    # (b) I(η) lineout at ring peak
    ax = axes[1]
    ring_idx = np.argmin(np.abs(R_px - r_peak))
    lineout = I[:, ring_idx]
    ax.plot(Eta, lineout, '-', color='C0', lw=0.8)
    ax.set_xlabel('η (degrees)', fontsize=12)
    ax.set_ylabel('I (norm.)', fontsize=12)
    ax.set_title(f'(b) I(η) at R = {r_peak:.1f} px — note oscillations at η = 0°, ±90°')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(30))

    # Annotate cardinal angles
    for eta_card in [-90, 0, 90]:
        ax.axvline(eta_card, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.annotate(f'η={eta_card}°', xy=(eta_card, lineout[np.argmin(np.abs(Eta - eta_card))]),
                    fontsize=8, color='red', ha='left', va='bottom')

    fig.tight_layout()
    save(fig, 'pyfai_cake_aliasing')




# ═══════════════════════════════════════════════════════════════════
#  Experiment 5: CPU Runtime Benchmark (MIDAS vs pyFAI)
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_ITERS   = 100
BENCHMARK_WARMUP  = 5
BENCHMARK_NCPUS   = 8
BENCHMARK_DR      = 1.0
BENCHMARK_MODE    = 'cpu'   # 'cpu' or 'gpu'

# Set OMP_NUM_THREADS for pyFAI (must be done before pyFAI import)
def _set_omp_threads(n):
    """Set OMP_NUM_THREADS for both MIDAS and pyFAI."""
    os.environ['OMP_NUM_THREADS'] = str(n)
    global BENCHMARK_NCPUS, TABLE2_NCPUS
    BENCHMARK_NCPUS = n
    TABLE2_NCPUS = n


def _ensure_varex_workdir():
    """Ensure Varex benchmark workdir exists with Map.bin.

    If real CeO₂ data is available, uses existing Map.bin.
    Otherwise, generates synthetic Varex data + runs DetectorMapper
    so benchmarks work on any machine.
    """
    tag = f"compare_dr_{str(BENCHMARK_DR).replace('.', 'p')}"
    work_dir = os.path.join(WORK_BASE, tag)
    map_bin = os.path.join(work_dir, 'Map.bin')

    if os.path.isfile(map_bin):
        return work_dir

    # Map.bin doesn't exist — try to generate from synthetic data
    print(f"  Map.bin not found — generating synthetic Varex workdir...")
    import tifffile
    os.makedirs(work_dir, exist_ok=True)

    # Generate synthetic image if real data not available
    data_path = os.path.join(work_dir, DATA_FILE)
    if not os.path.isfile(data_path):
        real_data = os.path.join(DATA_DIR, DATA_FILE)
        if os.path.isfile(real_data):
            import shutil
            shutil.copy2(real_data, data_path)
            print(f"    Copied real CeO₂ data")
        else:
            print(f"    Generating synthetic 2880×2880 image...")
            img = generate_synthetic_image(2880, 2880)
            tifffile.imwrite(data_path, img)

    # Generate param file
    det = {'name': 'Varex 2923', 'px_um': 150, 'width': 2880, 'height': 2880}
    ps_fn = os.path.join(work_dir, 'ps_test.txt')
    if not os.path.isfile(ps_fn):
        generate_param_file(det, work_dir)
        # Rename ps_bench.txt to ps_test.txt for Experiment 5 compatibility
        bench_fn = os.path.join(work_dir, 'ps_bench.txt')
        if os.path.isfile(bench_fn):
            os.rename(bench_fn, ps_fn)

    # Run DetectorMapper
    mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
    cmd = [mapper, ps_fn, '-nCPUs', str(BENCHMARK_NCPUS)]
    print(f"    Running DetectorMapper...")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=600, cwd=work_dir)
    if r.returncode != 0:
        print(f"    DetectorMapper FAILED (rc={r.returncode})")
        if r.stderr:
            print(f"    stderr: {r.stderr[:300]}")
        return None
    print(f"    DetectorMapper done, Map.bin created")
    return work_dir


def benchmark_midas_subprocess(n_iters=50, warmup=3):
    """Approach 1: Time IntegratorZarrOMP via repeated subprocess calls.

    Each call forks a new process, so this includes ~130ms of process
    startup, file I/O, map mmap, image read, dark subtract, and HDF5 write
    overhead on top of the ~11ms integration kernel.
    """
    print(f"\n  ── MIDAS Subprocess Benchmark ({n_iters} iters, "
          f"{warmup} warmup, {BENCHMARK_NCPUS} CPUs) ──")

    work_dir = _ensure_varex_workdir()
    if not work_dir:
        return None

    integrator = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')
    data_path = os.path.join(work_dir, DATA_FILE)
    cmd = [integrator, '-paramFN', 'ps_test.txt',
           '-dataFN', data_path, '-nCPUs', str(BENCHMARK_NCPUS)]

    timings = []
    total = warmup + n_iters
    for i in range(total):
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=120, cwd=work_dir)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"    iter {i}: FAILED (rc={r.returncode})")
            continue
        if i >= warmup:
            timings.append(dt)
        if i < warmup:
            tag = "warmup"
        else:
            tag = f"iter {i - warmup}/{n_iters}"
        if i < 3 or i == warmup or (i % 25 == 0):
            print(f"    {tag}: {dt*1000:.1f} ms")

    return np.array(timings) if timings else None


def benchmark_midas_charness(n_iters=BENCHMARK_ITERS,
                              warmup=BENCHMARK_WARMUP):
    """Approach 2: Time integration kernel via -benchmark flag (in-process).

    Runs the integration kernel n_iters times inside a single process
    invocation. Supports both CPU (IntegratorZarrOMP) and GPU
    (IntegratorFitPeaksGPUStream) mode via BENCHMARK_MODE.
    """
    is_gpu = (BENCHMARK_MODE == 'gpu')
    mode_label = 'GPU' if is_gpu else 'CPU'
    exe_name = 'IntegratorFitPeaksGPUStream' if is_gpu else 'IntegratorZarrOMP'

    print(f"\n  ── MIDAS {mode_label} C-Harness Benchmark ({n_iters} iters, "
          f"{warmup} warmup{'' if is_gpu else f', {BENCHMARK_NCPUS} CPUs'}) ──")

    work_dir = _ensure_varex_workdir()
    if not work_dir:
        return None

    integrator = os.path.join(MIDAS_BIN, exe_name)
    data_path = os.path.join(work_dir, DATA_FILE)
    total = warmup + n_iters
    cmd = [integrator, '-paramFN', 'ps_test.txt',
           '-dataFN', data_path, '-benchmark', str(total)]
    if not is_gpu:
        cmd.extend(['-nCPUs', str(BENCHMARK_NCPUS)])

    print(f"    Running {total} kernel iterations in a single process...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=600, cwd=work_dir)
    wall = time.time() - t0
    print(f"    Total wall time: {wall:.1f} s")

    if r.returncode != 0:
        print(f"    FAILED (rc={r.returncode})")
        if r.stderr:
            print(f"    stderr: {r.stderr[:500]}")
        return None

    # Parse BENCHMARK_CSV lines from stdout
    # New format: iteration,h2d_ms,kernel_ms,d2h_ms,total_ms
    # Old format: iteration,total_ms
    timings_total = []
    timings_h2d = []
    timings_kernel = []
    timings_d2h = []
    for line in r.stdout.split('\n'):
        if line.startswith('BENCHMARK_CSV,') and \
           not line.startswith('BENCHMARK_CSV_HEADER'):
            parts = line.strip().split(',')
            if len(parts) == 5:  # new format
                timings_h2d.append(float(parts[1]))
                timings_kernel.append(float(parts[2]))
                timings_d2h.append(float(parts[3]))
                timings_total.append(float(parts[4]))
            elif len(parts) == 3:  # old format
                timings_total.append(float(parts[2]))

    if len(timings_total) < total:
        print(f"    WARNING: only parsed {len(timings_total)} of {total} "
              f"timings")

    # Discard warmup
    timings_total = timings_total[warmup:]
    timings_h2d = timings_h2d[warmup:]
    timings_kernel = timings_kernel[warmup:]
    timings_d2h = timings_d2h[warmup:]
    if timings_total:
        print(f"    Parsed {len(timings_total)} benchmark timings "
              f"(after {warmup} warmup)")
        if timings_h2d:
            h2d_med = np.median(timings_h2d) * 1000
            ker_med = np.median(timings_kernel) * 1000
            d2h_med = np.median(timings_d2h) * 1000
            tot_med = np.median(timings_total) * 1000
            print(f"    Breakdown: H→D {h2d_med:.3f} ms | "
                  f"Kernel {ker_med:.3f} ms | "
                  f"D→H {d2h_med:.3f} ms | "
                  f"Total {tot_med:.3f} ms")
    if not timings_total:
        return None
    result = {'total': np.array(timings_total)}
    if timings_h2d:
        result['h2d'] = np.array(timings_h2d)
        result['kernel'] = np.array(timings_kernel)
        result['d2h'] = np.array(timings_d2h)
    return result


def benchmark_pyfai_method(method_tuple, n_iters=BENCHMARK_ITERS,
                            warmup=BENCHMARK_WARMUP):
    """Time pyFAI integrate2d with a specific method.

    Parameters
    ----------
    method_tuple : tuple of str
        pyFAI method specification, e.g. ("full", "csr", "cython") for
        cached sparse-matrix integration (comparable to MIDAS Map.bin),
        or ("full", "histogram", "cython") for uncached histogram mode.
    """
    import signal
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)  # pyFAI safety

    try:
        import pyFAI
        try:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        except ImportError:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        from pyFAI.detectors import Detector
        import tifffile

        mstr = '-'.join(method_tuple)
        print(f"\n  ── pyFAI Benchmark [{mstr}] ({n_iters} iters, "
              f"{warmup} warmup) ──")
        print(f"    pyFAI version: {pyFAI.version}")
    except ImportError:
        print("    pyFAI not installed — skipping.")
        signal.signal(signal.SIGINT, old_handler)
        return None

    px_m = PX_UM * 1e-6
    lsd_m = LSD_UM * 1e-6
    wl_m = WL_A * 1e-10

    # Use real data if available, otherwise fall back to synthetic
    real_img_path = os.path.join(DATA_DIR, DATA_FILE)
    if os.path.isfile(real_img_path):
        img_raw = tifffile.imread(real_img_path).astype(np.float64)
        img = np.flipud(img_raw).T

        dark_files = [fn for fn in os.listdir(DATA_DIR)
                      if fn.startswith('dark') and fn.endswith('.tif')]
        if dark_files:
            dark_raw = tifffile.imread(
                os.path.join(DATA_DIR, dark_files[0])).astype(np.float64)
            dark = np.flipud(dark_raw).T
            img -= dark
    else:
        # Synthetic fallback — use the workdir image
        work_dir = _ensure_varex_workdir()
        if not work_dir:
            signal.signal(signal.SIGINT, old_handler)
            return None
        synth_path = os.path.join(work_dir, DATA_FILE)
        if not os.path.isfile(synth_path):
            synth_path = os.path.join(work_dir, 'synthetic.tif')
        img = tifffile.imread(synth_path).astype(np.float64)
        # Synthetic uses centered BC and 150µm pixel
        px_m = 150e-6
        print(f"    Using synthetic data (no real CeO₂ available)")

    det = Detector(pixel1=px_m, pixel2=px_m,
                   max_shape=(img.shape[0], img.shape[1]))
    # Beam center: real data uses BC_Y/BC_Z, synthetic uses image center
    if os.path.isfile(real_img_path):
        poni1 = (BC_Y + 0.5) * px_m
        poni2 = (BC_Z + 0.5) * px_m
    else:
        poni1 = (img.shape[0] / 2.0 + 0.5) * px_m
        poni2 = (img.shape[1] / 2.0 + 0.5) * px_m
    ai = AzimuthalIntegrator(
        dist=lsd_m, poni1=poni1, poni2=poni2,
        wavelength=wl_m, detector=det
    )

    dr = BENCHMARK_DR
    r_max_px = np.sqrt((img.shape[0]/2)**2 + (img.shape[1]/2)**2)
    npt_rad = int((r_max_px - R_MIN) / dr)
    npt_azim = int((ETA_MAX - ETA_MIN) / ETA_BIN)
    r_min_mm = R_MIN * px_m * 1e3
    r_max_mm = r_max_px * px_m * 1e3
    kwargs = dict(npt_rad=npt_rad, npt_azim=npt_azim, unit="r_mm",
                  method=method_tuple,
                  radial_range=(r_min_mm, r_max_mm),
                  azimuth_range=(ETA_MIN, ETA_MAX),
                  correctSolidAngle=False)

    # First call to build cache (LUT/CSR) if applicable
    print(f"    Cache build...", end=' ', flush=True)
    t0 = time.time()
    ai.integrate2d(img, **kwargs)
    print(f"{time.time() - t0:.1f}s")

    timings = []
    total = warmup + n_iters
    for i in range(total):
        t0 = time.time()
        ai.integrate2d(img, **kwargs)
        dt = time.time() - t0
        if i >= warmup:
            timings.append(dt)
        if i < 3 or i == warmup or (i % 25 == 0):
            tag = "warmup" if i < warmup else f"iter {i - warmup}/{n_iters}"
            print(f"    {tag}: {dt*1000:.1f} ms")

    signal.signal(signal.SIGINT, old_handler)
    return np.array(timings) if timings else None


def compute_benchmark_stats(timings, label):
    """Compute and print benchmark statistics."""
    if timings is None or len(timings) == 0:
        print(f"  {label}: No timing data available.")
        return None

    t_ms = timings * 1000  # convert to ms
    stats = {
        'n': len(t_ms),
        'mean_ms': np.mean(t_ms),
        'median_ms': np.median(t_ms),
        'std_ms': np.std(t_ms),
        'min_ms': np.min(t_ms),
        'max_ms': np.max(t_ms),
        'p5_ms': np.percentile(t_ms, 5),
        'p25_ms': np.percentile(t_ms, 25),
        'p75_ms': np.percentile(t_ms, 75),
        'p95_ms': np.percentile(t_ms, 95),
        'iqr_ms': np.percentile(t_ms, 75) - np.percentile(t_ms, 25),
        'fps': 1000.0 / np.median(t_ms),
        'raw_ms': t_ms,
    }

    print(f"\n  ┌── {label} ({stats['n']} iterations) ──")
    print(f"  │  Median:  {stats['median_ms']:.3f} ms  "
          f"({stats['fps']:.0f} fps)")
    print(f"  │  Mean:    {stats['mean_ms']:.3f} ms  "
          f"± {stats['std_ms']:.3f} ms")
    print(f"  │  Min/Max: {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
    print(f"  │  P5/P95:  {stats['p5_ms']:.3f} / {stats['p95_ms']:.3f} ms")
    print(f"  │  IQR:     {stats['iqr_ms']:.3f} ms  "
          f"(P25={stats['p25_ms']:.3f}, P75={stats['p75_ms']:.3f})")
    print(f"  └──────────────────────")

    return stats


def plot_benchmarks(all_stats, fname='benchmark_runtime'):
    """Generate benchmark comparison plots."""
    labels = list(all_stats.keys())
    stats_list = [all_stats[k] for k in labels]

    # Filter out None entries
    valid = [(l, s) for l, s in zip(labels, stats_list) if s is not None]
    if len(valid) < 2:
        print("  Not enough benchmark data to plot.")
        return
    labels, stats_list = zip(*valid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CPU Runtime Benchmark: MIDAS IntegratorZarrOMP vs pyFAI\n'
                 f'Varex CeO₂ 2880×2880, ΔR={BENCHMARK_DR} px, '
                 f'MIDAS: {BENCHMARK_NCPUS} OMP threads',
                 fontsize=13, fontweight='bold')

    colors = {'MIDAS (subprocess)': '#1f77b4',
              'MIDAS (C-harness)': '#2ca02c',
              'pyFAI (CSR)': '#ff7f0e',
              'pyFAI (histogram)': '#d62728'}

    # (a) Box plot (log scale so all methods visible)
    ax = axes[0]
    data = [s['raw_ms'] for s in stats_list]
    bp = ax.boxplot(data, tick_labels=[l.replace(' ', '\n') for l in labels],
                    patch_artist=True, widths=0.6,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, '#999999'))
        patch.set_alpha(0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Time per frame (ms, log)', fontsize=11)
    ax.set_title('(a) Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    for ii, s in enumerate(stats_list):
        ax.text(ii + 1, s['median_ms'] * 1.15,
                f"{s['median_ms']:.1f} ms\n({s['fps']:.0f} fps)",
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    # (b) Histogram overlay
    ax = axes[1]
    for label, s in zip(labels, stats_list):
        ax.hist(s['raw_ms'], bins=30, alpha=0.5,
                color=colors.get(label, '#999999'),
                label=f"{label} (med={s['median_ms']:.1f}ms)",
                edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Time per frame (ms)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(b) Histogram', fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Time series (log scale)
    ax = axes[2]
    for label, s in zip(labels, stats_list):
        ax.plot(s['raw_ms'], '-', alpha=0.5, lw=0.5,
                color=colors.get(label, '#999999'), label=label)
        w = min(10, len(s['raw_ms']) // 5)
        if w > 1 and len(s['raw_ms']) > w:
            running = np.convolve(s['raw_ms'], np.ones(w)/w, mode='valid')
            ax.plot(np.arange(w-1, len(s['raw_ms'])), running, '-',
                    color=colors.get(label, '#999999'), lw=1.5, alpha=0.9)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Time per frame (ms, log)', fontsize=11)
    ax.set_title('(c) Time series', fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save(fig, fname)


def run_benchmark_experiment():
    """Run complete CPU runtime benchmark: MIDAS vs pyFAI.

    Benchmarks four configurations:
    1. MIDAS C-harness  — pure integration kernel, no process overhead
    2. pyFAI CSR         — cached sparse matrix (fair comparison to MIDAS)
    3. pyFAI histogram   — uncached, recomputes pixel splitting each call
    4. MIDAS subprocess  — full end-to-end including process startup/IO
    """
    import json

    print("\n" + "=" * 60)
    print("  Experiment 5: CPU Runtime Benchmark")
    print("=" * 60)
    print(f"  Config: {BENCHMARK_ITERS} iters, {BENCHMARK_WARMUP} warmup, "
          f"ΔR={BENCHMARK_DR}, ncpus={BENCHMARK_NCPUS}")
    print(f"  Data: Varex CeO₂ 2880×2880")

    all_stats = {}

    # 1. MIDAS C-harness (pure kernel, ~11ms/frame)
    t_cha_result = benchmark_midas_charness()
    t_cha = t_cha_result['total'] if isinstance(t_cha_result, dict) \
        else t_cha_result
    s_cha = compute_benchmark_stats(t_cha, 'MIDAS (C-harness)')
    all_stats['MIDAS (C-harness)'] = s_cha

    # 2. pyFAI CSR (cached sparse matrix, ~119ms/frame)
    t_csr = benchmark_pyfai_method(("full", "csr", "cython"),
                                    n_iters=BENCHMARK_ITERS,
                                    warmup=BENCHMARK_WARMUP)
    s_csr = compute_benchmark_stats(t_csr, 'pyFAI (CSR)')
    all_stats['pyFAI (CSR)'] = s_csr

    # 3. pyFAI histogram (uncached, ~1066ms/frame — fewer iters)
    t_hist = benchmark_pyfai_method(("full", "histogram", "cython"),
                                     n_iters=10, warmup=2)
    s_hist = compute_benchmark_stats(t_hist, 'pyFAI (histogram)')
    all_stats['pyFAI (histogram)'] = s_hist

    # 4. MIDAS subprocess (full end-to-end, ~146ms/frame)
    t_sub = benchmark_midas_subprocess()
    s_sub = compute_benchmark_stats(t_sub, 'MIDAS (subprocess)')
    all_stats['MIDAS (subprocess)'] = s_sub

    # Summary table
    print("\n" + "=" * 75)
    fmt = "  {:<24} {:>12} {:>12} {:>10} {:>8}"
    print(fmt.format('Method', 'Median (ms)', 'Mean (ms)', 'Std (ms)', 'FPS'))
    print("-" * 75)
    for label in ['MIDAS (C-harness)', 'MIDAS (subprocess)',
                  'pyFAI (CSR)', 'pyFAI (histogram)']:
        s = all_stats.get(label)
        if s:
            print(fmt.format(label, f"{s['median_ms']:.3f}",
                             f"{s['mean_ms']:.3f}", f"{s['std_ms']:.3f}",
                             f"{s['fps']:.0f}"))
    print("=" * 75)

    # Speedup ratios
    if s_cha and s_csr:
        print(f"\n  MIDAS kernel vs pyFAI-CSR:  "
              f"{s_csr['median_ms'] / s_cha['median_ms']:.1f}× faster")
    if s_cha and s_hist:
        print(f"  MIDAS kernel vs pyFAI-hist: "
              f"{s_hist['median_ms'] / s_cha['median_ms']:.0f}× faster")
    if s_sub and s_csr:
        print(f"  MIDAS subproc vs pyFAI-CSR: "
              f"{s_csr['median_ms'] / s_sub['median_ms']:.1f}× faster")

    # Generate plots
    plot_benchmarks(all_stats)

    # Save raw data as JSON
    json_out = {}
    for label, s in all_stats.items():
        if s is not None:
            json_out[label] = {k: v.tolist() if isinstance(v, np.ndarray)
                               else v for k, v in s.items()}
    json_fn = os.path.join(WORK_BASE, 'benchmark_results.json')
    with open(json_fn, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Raw data saved to: {json_fn}")

    return all_stats



# ═══════════════════════════════════════════════════════════════════
#  Experiment 6: Table 2 — Throughput for all detector configs
# ═══════════════════════════════════════════════════════════════════

# Detector configurations from the paper Table 1 + Varex validation detector
TABLE2_DETECTORS = [
    {'name': 'EIGER2 500K',  'px_um': 75,  'width': 1028, 'height':  512},
    {'name': 'PILATUS3 1M',  'px_um': 172, 'width':  981, 'height': 1043},
    {'name': 'PILATUS3 2M',  'px_um': 172, 'width': 1475, 'height': 1679},
    {'name': 'EIGER2 4M',    'px_um': 75,  'width': 2068, 'height': 2162},
    {'name': 'PILATUS3 6M',  'px_um': 172, 'width': 2463, 'height': 2527},
    {'name': 'Varex 2923',   'px_um': 150, 'width': 2880, 'height': 2880},
    {'name': 'EIGER2 9M',    'px_um': 75,  'width': 3110, 'height': 3269},
    {'name': 'EIGER2 16M',   'px_um': 75,  'width': 4148, 'height': 4362},
]

# Paper integration parameters
TABLE2_DR      = 1.0     # radial bin size (px)
TABLE2_DETA    = 5.0     # azimuthal bin size (deg)
TABLE2_ETA_MIN = -180.0
TABLE2_ETA_MAX = 180.0
TABLE2_R_MIN   = 10.0    # pixels
TABLE2_LSD_UM  = 500000  # 500 mm
TABLE2_NCPUS   = 8
TABLE2_ITERS   = 100
TABLE2_WARMUP  = 5


def generate_synthetic_image(width, height, n_rings=10, seed=42):
    """Generate synthetic uint16 image with Debye-Scherrer rings.

    Matches the paper methodology: 10 rings at random radial positions,
    ring thickness ~5 px, pixel values uniform [2000, 16000], beam center
    at geometric center.
    """
    rng = np.random.RandomState(seed)
    img = rng.randint(50, 200, size=(height, width), dtype=np.uint16)

    cy, cx = height / 2.0, width / 2.0
    max_r = np.sqrt(cx**2 + cy**2) * 0.95
    ring_radii = rng.uniform(0.05 * max_r, 0.95 * max_r, n_rings)
    ring_width = 5.0  # px

    yy, xx = np.mgrid[:height, :width]
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    for r0 in ring_radii:
        mask = np.abs(rr - r0) <= ring_width / 2
        ring_vals = rng.randint(2000, 16000, size=mask.sum()).astype(np.uint16)
        img[mask] = ring_vals

    return img


def generate_param_file(det, work_dir):
    """Write a parameter file for DetectorMapper + IntegratorZarrOMP."""
    px = det['px_um']
    w, h = det['width'], det['height']
    cx, cy = w / 2.0, h / 2.0
    r_max = np.sqrt(cx**2 + cy**2)

    params = f"""NrPixelsY {w}
NrPixelsZ {h}
px {px}
BC {cx} {cy}
Lsd {TABLE2_LSD_UM}
RhoD {int(r_max + 100)}
RBinSize {TABLE2_DR}
EtaBinSize {TABLE2_DETA}
EtaMin {TABLE2_ETA_MIN}
EtaMax {TABLE2_ETA_MAX}
RMin {TABLE2_R_MIN}
RMax {r_max:.1f}
tx 0
ty 0
tz 0
p0 0
p1 0
p2 0
p3 0
"""
    fn = os.path.join(work_dir, 'ps_bench.txt')
    with open(fn, 'w') as f:
        f.write(params)
    return fn


def benchmark_detector_config(det, n_iters=TABLE2_ITERS, warmup=TABLE2_WARMUP):
    """Run full benchmark for a single detector config.

    1. Generate synthetic image
    2. Run DetectorMapper to create Map.bin / nMap.bin
    3. Run IntegratorZarrOMP -benchmark N for kernel timing
    """
    import tifffile

    name = det['name']
    tag = name.replace(' ', '_').lower()
    work_dir = os.path.join(WORK_BASE, f'bench_{tag}')
    os.makedirs(work_dir, exist_ok=True)

    mpx = det['width'] * det['height'] / 1e6
    print(f"\n  ── {name} ({det['width']}×{det['height']}, {mpx:.2f} Mpx) ──")

    # 1. Generate synthetic image
    img_fn = os.path.join(work_dir, 'synthetic.tif')
    if not os.path.isfile(img_fn):
        print(f"    Generating synthetic image...")
        img = generate_synthetic_image(det['width'], det['height'])
        tifffile.imwrite(img_fn, img)

    # 2. Generate parameter file
    ps_fn = generate_param_file(det, work_dir)

    # 3. Run DetectorMapper to create Map.bin
    map_fn = os.path.join(work_dir, 'Map.bin')
    if not os.path.isfile(map_fn):
        mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
        cmd = [mapper, ps_fn, '-nCPUs', str(TABLE2_NCPUS)]
        print(f"    Running DetectorMapper...")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=600, cwd=work_dir)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"    DetectorMapper FAILED (rc={r.returncode})")
            if r.stderr:
                print(f"    stderr: {r.stderr[:300]}")
            return None
        print(f"    DetectorMapper done in {dt:.1f}s")
    else:
        print(f"    Map.bin exists, skipping DetectorMapper")

    # 4. Run integrator -benchmark N
    is_gpu = (BENCHMARK_MODE == 'gpu')
    exe_name = 'IntegratorFitPeaksGPUStream' if is_gpu else 'IntegratorZarrOMP'
    integrator = os.path.join(MIDAS_BIN, exe_name)
    total = warmup + n_iters
    cmd = [integrator, '-paramFN', 'ps_bench.txt',
           '-dataFN', img_fn, '-benchmark', str(total)]
    if not is_gpu:
        cmd.extend(['-nCPUs', str(TABLE2_NCPUS)])

    mode_label = 'GPU' if is_gpu else 'CPU'
    print(f"    Running {exe_name} -benchmark {total} ({mode_label})...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=600, cwd=work_dir)
    wall = time.time() - t0

    if r.returncode != 0:
        print(f"    {exe_name} FAILED (rc={r.returncode})")
        if r.stderr:
            print(f"    stderr: {r.stderr[:300]}")
        return None

    # Parse BENCHMARK_CSV timings
    # New format: iteration,h2d_ms,kernel_ms,d2h_ms,total_ms
    # Old format: iteration,total_ms
    timings_total = []
    timings_h2d = []
    timings_kernel = []
    timings_d2h = []
    for line in r.stdout.split('\n'):
        if line.startswith('BENCHMARK_CSV,') and \
           not line.startswith('BENCHMARK_CSV_HEADER'):
            parts = line.strip().split(',')
            if len(parts) == 5:  # new format
                timings_h2d.append(float(parts[1]))
                timings_kernel.append(float(parts[2]))
                timings_d2h.append(float(parts[3]))
                timings_total.append(float(parts[4]))
            elif len(parts) == 3:  # old format
                timings_total.append(float(parts[2]))

    timings_total = timings_total[warmup:]
    timings_h2d = timings_h2d[warmup:]
    timings_kernel = timings_kernel[warmup:]
    timings_d2h = timings_d2h[warmup:]
    if not timings_total:
        print(f"    WARNING: no benchmark timings parsed")
        print(f"    HINT: Rebuild MIDAS binary — the -benchmark flag may not "
              f"be compiled in.")
        print(f"      cd ~/opt/MIDAS/build && cmake --build . "
              f"--target {exe_name} -j")
        if r.stderr:
            print(f"    stderr: {r.stderr[:500]}")
        if r.stdout:
            print(f"    stdout (last 500): {r.stdout[-500:]}")
        return None

    t_arr = np.array(timings_total)
    med_ms = np.median(t_arr) * 1000
    fps = 1000.0 / med_ms
    print(f"    {len(timings_total)} iters → {med_ms:.2f} ms/frame "
          f"({fps:.0f} fps)")

    result = {
        'name': name,
        'width': det['width'],
        'height': det['height'],
        'px_um': det['px_um'],
        'mpx': mpx,
        'median_ms': med_ms,
        'mean_ms': np.mean(t_arr) * 1000,
        'std_ms': np.std(t_arr) * 1000,
        'fps': fps,
        'raw_s': t_arr,
    }

    if timings_h2d:
        h2d_med = np.median(timings_h2d) * 1000
        ker_med = np.median(timings_kernel) * 1000
        d2h_med = np.median(timings_d2h) * 1000
        result['h2d_ms'] = h2d_med
        result['kernel_ms'] = ker_med
        result['d2h_ms'] = d2h_med
        print(f"    Breakdown: H→D {h2d_med:.3f} ms | "
              f"Kernel {ker_med:.3f} ms | "
              f"D→H {d2h_med:.3f} ms")

    return result


def plot_table2_results(results, fname='benchmark_table2'):
    """Plot throughput vs megapixels for all detector configs."""
    if len(results) < 2:
        return

    # Sort by megapixels so the line connects in order
    results = sorted(results, key=lambda r: r['mpx'])
    mpx = [r['mpx'] for r in results]
    fps = [r['fps'] for r in results]
    names = [r['name'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'MIDAS CPU Integration Throughput — {TABLE2_NCPUS} OMP threads\n'
                 f'ΔR={TABLE2_DR}px, Δη={TABLE2_DETA}°, '
                 f'synthetic data, 100 iterations',
                 fontsize=13, fontweight='bold')

    # (a) FPS vs Megapixels (log-log)
    ax1.loglog(mpx, fps, 'o-', color='#2ca02c', lw=2, markersize=8, zorder=5)
    for x, y, n in zip(mpx, fps, names):
        ax1.annotate(n, (x, y), textcoords="offset points",
                     xytext=(5, 8), fontsize=7, fontweight='bold')
    ax1.set_xlabel('Megapixels', fontsize=11)
    ax1.set_ylabel('Frames per second (fps)', fontsize=11)
    ax1.set_title('(a) Throughput vs detector size', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')

    # (b) ms/frame vs Megapixels (log-log)
    ms = [r['median_ms'] for r in results]
    ax2.loglog(mpx, ms, 's-', color='#1f77b4', lw=2, markersize=8, zorder=5)
    for x, y, n in zip(mpx, ms, names):
        ax2.annotate(f'{n}\n{y:.1f}ms', (x, y), textcoords="offset points",
                     xytext=(5, 8), fontsize=7)
    ax2.set_xlabel('Megapixels', fontsize=11)
    ax2.set_ylabel('Time per frame (ms)', fontsize=11)
    ax2.set_title('(b) Latency vs detector size', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save(fig, fname)


def run_table2_benchmark():
    """Run Experiment 6: benchmark all Table 2 detector configs.

    This is the comprehensive throughput benchmark from the paper,
    using synthetic data for all 7 detector models. Results are
    comparable to the co-author's numbers on different hardware.
    """
    import json

    print("\n" + "=" * 60)
    print("  Experiment 6: Table 2 — Integration Throughput")
    print("=" * 60)
    print(f"  Config: {TABLE2_ITERS} iters, {TABLE2_WARMUP} warmup, "
          f"{TABLE2_NCPUS} OMP threads")
    print(f"  ΔR={TABLE2_DR}px, Δη={TABLE2_DETA}°, "
          f"[{TABLE2_ETA_MIN}°, {TABLE2_ETA_MAX}°]")

    results = []
    for det in TABLE2_DETECTORS:
        r = benchmark_detector_config(det)
        if r:
            results.append(r)

    if not results:
        print("  No results — check MIDAS build.")
        return None

    # Print Table 2
    print("\n" + "=" * 75)
    fmt = "  {:<18} {:>6} {:>12} {:>12} {:>10} {:>8}"
    print(fmt.format('Detector', 'Mpx', 'Median (ms)',
                     'Mean (ms)', 'Std (ms)', 'FPS'))
    print("-" * 75)
    for r in results:
        print(fmt.format(r['name'], f"{r['mpx']:.2f}",
                         f"{r['median_ms']:.2f}", f"{r['mean_ms']:.2f}",
                         f"{r['std_ms']:.2f}", f"{r['fps']:.0f}"))
    print("=" * 75)

    # Pixel throughput (Gpx/s)
    if results:
        best = max(results, key=lambda r: r['mpx'] * r['fps'])
        gpxs = best['mpx'] * best['fps'] / 1000.0
        print(f"\n  Peak pixel throughput: {gpxs:.2f} Gpx/s "
              f"({best['name']}, {best['fps']:.0f} fps)")

    # Plot
    plot_table2_results(results)

    # Save JSON
    json_out = []
    for r in results:
        entry = {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in r.items()}
        json_out.append(entry)
    json_fn = os.path.join(WORK_BASE, 'table2_results.json')
    with open(json_fn, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"  Raw data saved to: {json_fn}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 62 + "╗")
    print("║  MIDAS Radial Integration — Comprehensive Validation Suite ║")
    print("╚" + "═" * 62 + "╝")

    # Check if real Varex data is available for Experiments 1-4
    has_real_data = os.path.isdir(DATA_DIR) and os.path.isfile(
        os.path.join(DATA_DIR, PARAM_FILE))

    if has_real_data:
        # ── Experiment 1: Varex MIDAS vs pyFAI ──
        midas_results = run_midas_experiments()
        pyfai_results = run_pyfai_experiments()

        if pyfai_results:
            for dr in DELTA_RS:
                if dr not in midas_results or dr not in pyfai_results:
                    print(f"\n  Skipping ΔR={dr}: missing data")
                    continue

                print(f"\n{'='*60}")
                print(f"  VAREX COMPARISON: ΔR = {dr} px")
                print(f"{'='*60}")

                md = midas_results[dr]
                pd = pyfai_results[dr]

                pI_interp = interpolate_pyfai_to_midas_grid(
                    md['R'], md['Eta'], pd['R'], pd['Eta'], pd['I_norm'])

                stats = compute_statistics(md['I_norm'], pI_interp,
                                           label=f"Normalized, ΔR={dr}")

                tag = str(dr).replace('.', 'p')
                plot_cake_comparison(md, pd, dr, f'compare_cake_dr{tag}')
                plot_scatter(md['I_norm'], pI_interp, dr,
                             f'compare_scatter_dr{tag}')
                if stats and 'rel_diff_values' in stats:
                    plot_histogram(stats['rel_diff_values'], dr,
                                  f'compare_hist_dr{tag}')
                plot_lineouts(md, pd, pI_interp, dr,
                              f'compare_lineouts_dr{tag}')
                plot_1d_profiles(md, pd, dr, f'compare_1d_dr{tag}')
        else:
            print("\nWARNING: pyFAI not available — skipping comparison.")

        # ── Experiment 2: Pilatus CeO₂ ──
        pil_midas = run_pilatus_midas()

        # ── Experiment 3: Precision test ──
        prec = run_precision_test()
        if prec:
            plot_precision(prec)

        # ── Experiment 4: pyFAI cake aliasing ──
        plot_pyfai_cake_aliasing()
    else:
        print(f"\n  NOTE: Real Varex data not found at {DATA_DIR}")
        print(f"  Skipping Experiments 1-4 (require real CeO₂ data).")
        print(f"  Running Experiments 5-6 with synthetic data.\n")

    # ── Experiment 5: CPU runtime benchmark ──
    run_benchmark_experiment()

    # ── Experiment 6: Table 2 throughput benchmark ──
    run_table2_benchmark()

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='MIDAS Radial Integration — Comprehensive Validation Suite')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='cpu',
                        help='Benchmark mode: cpu (IntegratorZarrOMP) or '
                             'gpu (IntegratorFitPeaksGPUStream). Default: cpu')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of OMP threads for MIDAS and pyFAI. '
                             'Default: 8')
    parser.add_argument('--only', choices=['5', '6', 'all'], default='all',
                        help='Run only a specific experiment: '
                             '5=MIDAS vs pyFAI, 6=Table2, all=everything. '
                             'Default: all')
    args = parser.parse_args()

    # Apply CLI settings
    BENCHMARK_MODE = args.mode
    _set_omp_threads(args.threads)

    if args.only == '5':
        run_benchmark_experiment()
    elif args.only == '6':
        run_table2_benchmark()
    else:
        main()
