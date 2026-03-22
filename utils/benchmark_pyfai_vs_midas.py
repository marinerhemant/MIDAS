#!/usr/bin/env python3
"""
Paper 3 — pyFAI vs MIDAS Calibration Benchmark
================================================
Compares pyFAI's geometry refinement against MIDAS's powder calibration
pipeline on the same CeO₂ calibrant images.

Approach:
  - Both tools start with the SAME initial geometry (from MIDAS auto-guess)
  - pyFAI: ring-masked peak extraction → GeometryRefinement (L-BFGS)
  - MIDAS: AutoCalibrateZarr (full automated pipeline)
  - Compare final geometry parameters and pseudo-strain

Usage:
    python benchmark_pyfai_vs_midas.py [--pilatus] [--varex] [--all]

Coordinate conventions:
    - Image is transformed with ImTransOpt then transposed (MIDAS convention)
    - pyFAI PONI1/PONI2 use +0.5px center offset vs MIDAS integer centers
"""

import os
import sys
import time
import argparse
import subprocess
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDAS_HOME = os.path.dirname(SCRIPT_DIR)

# ─── Dataset configurations ───────────────────────────────────────────
DATASETS = {
    'pilatus': {
        'label': 'Pilatus CeO₂ (172µm, 71.7keV)',
        'data_dir': os.path.join(MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration'),
        'data_file': 'CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif',
        'dark_file': 'dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif',
        'mask_file': 'mask_upd.tif',
        'px_um': 172.0,
        'wavelength_A': 0.17297,
        'im_trans': 2,
        'n_iterations': 30,
        'seed_lsd_um': 657437,
        'seed_bc_y': 685,
        'seed_bc_z': 921,
    },
    'varex': {
        'label': 'Varex CeO₂ (150µm, 63keV)',
        'data_dir': os.path.join(MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration'),
        'data_file': 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif',
        'dark_file': None,
        'mask_file': None,
        'px_um': 150.0,
        'wavelength_A': 0.19582,
        'im_trans': 2,
        'n_iterations': 30,
        'seed_lsd_um': 900363,
        'seed_bc_y': 1447,
        'seed_bc_z': 1469,
    },
    'ge5': {
        'label': 'GE5 CeO₂ non-summed (200µm, 71.7keV)',
        'data_dir': os.path.expanduser(
            '~/Desktop/analysis/lieghanne/LCG_MIDAS_TEST_03092026-selected_no_sum'),
        'data_file': 'ceria_1dfocusbeam_0deg_10f0p2s_000417.ge5.h5',
        'dark_file': 'dark_ceria_1dfocusbeam_0deg_10f0p2s_000418.ge5.h5',
        'mask_file': None,
        'px_um': 200.0,
        'wavelength_A': 0.173058,
        'im_trans': 0,
        'n_iterations': 30,
        'data_loc': '/exchange/data',
        'dark_loc': '/exchange/data',
        'seed_lsd_um': 278039,
        'seed_bc_y': 1022,
        'seed_bc_z': 1124,
    },
    'ge5_summed': {
        'label': 'GE5 CeO₂ summed (200µm, 71.7keV)',
        'data_dir': os.path.expanduser(
            '~/Desktop/analysis/lieghanne/LCG_MIDAS_TEST_03092026-selected_sum'),
        'data_file': 'summed_ceria_1dfocusbeam_0deg_10f0p2s_000417.ge5.h5',
        'dark_file': 'summed_dark_ceria_1dfocusbeam_0deg_10f0p2s_000418.ge5.h5',
        'mask_file': None,
        'px_um': 200.0,
        'wavelength_A': 0.173058,
        'im_trans': 0,
        'n_iterations': 30,
        'data_loc': '/exchange/data',
        'dark_loc': '/exchange/data',
        'seed_lsd_um': 278035,
        'seed_bc_y': 1022,
        'seed_bc_z': 1124,
    },
}


def load_and_transform_image(dataset_cfg):
    """Load image and apply MIDAS image transforms."""
    import tifffile

    data_dir = dataset_cfg['data_dir']
    data_path = os.path.join(data_dir, dataset_cfg['data_file'])

    # Load based on file format
    ext = os.path.splitext(data_path)[1].lower()
    if ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
        import h5py
        data_loc = dataset_cfg.get('data_loc', '/exchange/data').lstrip('/')
        with h5py.File(data_path, 'r') as f:
            ds = f[data_loc]
            if ds.ndim == 3:
                img = ds[0].astype(np.float64)  # first frame
            else:
                img = ds[()].astype(np.float64)  # already 2D (summed)
        # Dark from separate file
        if dataset_cfg['dark_file']:
            dark_path = os.path.join(data_dir, dataset_cfg['dark_file'])
            dark_loc = dataset_cfg.get('dark_loc', data_loc).lstrip('/')
            if os.path.exists(dark_path):
                with h5py.File(dark_path, 'r') as f:
                    ds = f[dark_loc]
                    if ds.ndim == 3:
                        dark = ds[0].astype(np.float64)
                    else:
                        dark = ds[()].astype(np.float64)
                img = img - dark
                img[img < 0] = 0
    else:
        img = tifffile.imread(data_path).astype(np.float64)
        # Dark subtraction
        if dataset_cfg['dark_file']:
            dark_path = os.path.join(data_dir, dataset_cfg['dark_file'])
            if os.path.exists(dark_path):
                dark = tifffile.imread(dark_path).astype(np.float64)
                img = img - dark
                img[img < 0] = 0

    # ImTransOpt
    im_trans = dataset_cfg['im_trans']
    if im_trans == 1:
        img = np.fliplr(img)
    elif im_trans == 2:
        img = np.flipud(img)
    elif im_trans == 3:
        img = np.flipud(np.fliplr(img))

    # MIDAS always transposes
    img = img.T
    return img


def run_pyfai_calibration(dataset_cfg):
    """Run pyFAI's geometry refinement on a dataset.

    Steps:
      1. Load image with MIDAS transforms
      2. Ring-masked peak extraction via Massif
      3. GeometryRefinement with 2-stage fitting

    Returns dict with geometry parameters.
    """
    import pyFAI
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import Detector
    from pyFAI.geometryRefinement import GeometryRefinement
    from pyFAI.massif import Massif

    print(f"\n  pyFAI version: {pyFAI.version}")

    px_um = dataset_cfg['px_um']
    wl_A = dataset_cfg['wavelength_A']
    px_m = px_um * 1e-6
    wl_m = wl_A * 1e-10

    img = load_and_transform_image(dataset_cfg)
    ny, nz = img.shape
    print(f"  Image shape: {img.shape}")

    # Set up CeO2 calibrant
    cal = get_calibrant('CeO2')
    cal.wavelength = wl_m
    tth_values = cal.get_2th()

    # Initial guess from seed values
    bc_y_seed = dataset_cfg['seed_bc_y']
    bc_z_seed = dataset_cfg['seed_bc_z']
    dist_seed = dataset_cfg['seed_lsd_um'] * 1e-6

    poni1_init = (bc_y_seed + 0.5) * px_m
    poni2_init = (bc_z_seed + 0.5) * px_m

    print(f"  Seed: dist={dist_seed*1e6:.0f} µm, BC=({bc_y_seed}, {bc_z_seed}) px")

    t0 = time.time()

    # Compute expected ring radii
    ring_r_px = [dist_seed * np.tan(tth) / px_m for tth in tth_values]

    # Pixel coordinate arrays for ring masking
    yy, zz = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    R_img = np.sqrt((yy - bc_y_seed)**2 + (zz - bc_z_seed)**2)

    # Extract peaks per ring using Massif
    massif = Massif(img)
    control_points = []
    n_total_points = 0
    n_rings_found = 0
    max_rings = 20
    max_extent = min(ny - bc_y_seed, bc_y_seed, nz - bc_z_seed, bc_z_seed) * 0.95

    for i, r_px in enumerate(ring_r_px[:max_rings]):
        if r_px > max_extent:
            break

        # Annulus half-width: ±max(4, 2% of radius)
        hw = max(4, int(0.02 * r_px))
        ring_mask = ((R_img >= r_px - hw) & (R_img <= r_px + hw)).astype(np.int8)

        try:
            pts = massif.peaks_from_area(ring_mask, Imin=float(np.median(img)),
                                         keep=1000, dmin=2.0)
            if pts is not None and len(pts) > 3:
                for pt in pts:
                    control_points.append((pt[0], pt[1], i))
                n_total_points += len(pts)
                n_rings_found += 1
                print(f"    Ring {i+1} (2θ={np.degrees(tth_values[i]):.2f}°, "
                      f"R≈{r_px:.0f}px): {len(pts)} peaks")
        except Exception as e:
            print(f"    Ring {i+1}: failed ({e})")
            continue

    t_extract = time.time() - t0
    print(f"  Peak extraction: {t_extract:.1f}s, "
          f"{n_total_points} points on {n_rings_found} rings")

    if n_total_points < 20:
        print("  ERROR: Too few control points")
        return None

    gr_data = np.array(control_points, dtype=np.float64)

    # Geometry refinement
    t1 = time.time()
    try:
        det = Detector(pixel1=px_m, pixel2=px_m)
        gr = GeometryRefinement(
            data=gr_data,
            dist=dist_seed,
            poni1=poni1_init,
            poni2=poni2_init,
            rot1=0.0, rot2=0.0, rot3=0.0,
            detector=det,
            wavelength=wl_m,
            calibrant=cal,
        )

        # Stage 1: center + distance only
        gr.refine2(fix=['rot1', 'rot2', 'rot3', 'wavelength'])
        s1_lsd = gr.dist * 1e6
        s1_bc_y = gr.poni1 / px_m - 0.5
        s1_bc_z = gr.poni2 / px_m - 0.5
        print(f"  Stage 1 (center+dist): dist={s1_lsd:.1f} µm, "
              f"BC=({s1_bc_y:.2f}, {s1_bc_z:.2f}) px")

        # Stage 2: tilts + dist with BC constrained to ±1px from Stage 1
        # pyFAI's default poni bounds are extremely wide (meters), allowing
        # tilt-BC coupling to produce spurious solutions. Set tight bounds
        # on PONI1/PONI2 using pyFAI's native bounds API.
        gr._poni1_min = gr.poni1 - 1.0 * px_m
        gr._poni1_max = gr.poni1 + 1.0 * px_m
        gr._poni2_min = gr.poni2 - 1.0 * px_m
        gr._poni2_max = gr.poni2 + 1.0 * px_m
        gr.refine2(fix=['wavelength'])
        t_refine = time.time() - t1
        total_time = time.time() - t0

        bc_y = gr.poni1 / px_m - 0.5
        bc_z = gr.poni2 / px_m - 0.5

        print(f"  Stage 2 (BC±1px bounds, {t_refine:.1f}s):")
        print(f"    dist  = {gr.dist*1e6:.1f} µm")
        print(f"    BC    = ({bc_y:.2f}, {bc_z:.2f}) px")
        print(f"    rot1  = {np.degrees(gr.rot1):.4f}°")
        print(f"    rot2  = {np.degrees(gr.rot2):.4f}°")

        # Compute residual
        chi2 = gr.chi2()
        print(f"    chi²  = {chi2:.6f}")

        return {
            'lsd_um': gr.dist * 1e6,
            'bc_y_px': bc_y,
            'bc_z_px': bc_z,
            'rot1_deg': np.degrees(gr.rot1),
            'rot2_deg': np.degrees(gr.rot2),
            'rot3_deg': np.degrees(gr.rot3),
            'chi2': chi2,
            'runtime_s': total_time,
            'n_rings': n_rings_found,
            'n_points': n_total_points,
            # Stage 1 (zero-tilt) results
            's1_lsd_um': s1_lsd,
            's1_bc_y_px': s1_bc_y,
            's1_bc_z_px': s1_bc_z,
        }

    except Exception as e:
        import traceback
        print(f"  GeometryRefinement failed: {e}")
        traceback.print_exc()
        return None


CACHE_DIR = os.path.join(SCRIPT_DIR, '.benchmark_cache')


def _midas_cache_path(dataset_name):
    return os.path.join(CACHE_DIR, f'midas_{dataset_name}.json')


def _load_midas_cache(dataset_name):
    """Load cached MIDAS result if available."""
    path = _midas_cache_path(dataset_name)
    if os.path.exists(path):
        try:
            with open(path) as f:
                result = json.load(f)
            print(f"  Using cached MIDAS result from {path}")
            print(f"  MIDAS cached result:")
            print(f"    Lsd   = {result['lsd_um']:.1f} µm")
            print(f"    BC    = ({result['bc_y_px']:.2f}, {result['bc_z_px']:.2f}) px")
            if result.get('mean_strain'):
                print(f"    Strain = {result['mean_strain']*1e6:.1f} µε")
            return result
        except Exception as e:
            print(f"  WARNING: Cache load failed: {e}")
    return None


def _save_midas_cache(dataset_name, result):
    """Save MIDAS result to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _midas_cache_path(dataset_name)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Cached MIDAS result to {path}")


def run_midas_calibration(dataset_cfg, dataset_name=None, use_cache=True):
    """Run MIDAS AutoCalibrateZarr (fully automated, no seeds needed).

    Returns dict with geometry parameters parsed from output.
    """
    # Try cache first
    if use_cache and dataset_name:
        cached = _load_midas_cache(dataset_name)
        if cached is not None:
            return cached

    data_dir = dataset_cfg['data_dir']
    data_file = dataset_cfg['data_file']

    cmd = [
        sys.executable,
        os.path.join(MIDAS_HOME, 'utils', 'AutoCalibrateZarr.py'),
        '--data', data_file,
        '--material', 'ceo2',
        '--wavelength', str(dataset_cfg['wavelength_A']),
        '--im-trans', str(dataset_cfg['im_trans']),
        '--n-iterations', str(dataset_cfg['n_iterations']),
    ]
    if dataset_cfg['dark_file']:
        cmd.extend(['--dark', dataset_cfg['dark_file']])
    if dataset_cfg['mask_file']:
        cmd.extend(['--mask', dataset_cfg['mask_file']])
    if dataset_cfg.get('data_loc'):
        cmd.extend(['--data-loc', dataset_cfg['data_loc']])
    if dataset_cfg.get('dark_loc'):
        cmd.extend(['--dark-loc', dataset_cfg['dark_loc']])

    print(f"\n  Command: {os.path.basename(cmd[1])} --data {data_file} ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=data_dir, capture_output=True, text=True,
                            timeout=600)
    total_time = time.time() - t0

    if result.returncode != 0:
        print(f"  MIDAS failed (exit {result.returncode})")
        stderr = result.stderr[-500:] if result.stderr else ""
        print(stderr)
        return None

    # Parse the "Converged" block from combined output
    output = result.stdout + '\n' + result.stderr
    lsd = bc_y = bc_z = ty = tz = mean_strain = None

    in_converged = False
    for line in output.split('\n'):
        line = line.strip()
        if 'Converged' in line and 'Best' in line:
            in_converged = True
            continue
        if in_converged:
            if line.startswith('Lsd'):
                try: lsd = float(line.split()[1])
                except: pass
            elif line.startswith('BC'):
                try:
                    parts = line.split()
                    bc_y = float(parts[1])
                    bc_z = float(parts[2])
                except: pass
            elif line.startswith('ty'):
                try: ty = float(line.split()[1])
                except: pass
            elif line.startswith('tz'):
                try: tz = float(line.split()[1])
                except: pass
            elif line.startswith('Mean Strain'):
                try: mean_strain = float(line.split()[-1])
                except: pass
            elif line.startswith('===='):
                if lsd is not None:
                    in_converged = False

    if lsd is None:
        print("  WARNING: Could not parse MIDAS output from Converged block")
        # Fallback: check INFO lines
        for line in output.split('\n'):
            if 'INFO - Lsd' in line and lsd is None:
                try: lsd = float(line.strip().split()[-1])
                except: pass
            elif 'INFO - BC' in line and bc_y is None:
                try:
                    parts = line.strip().split()
                    bc_y = float(parts[-2])
                    bc_z = float(parts[-1])
                except: pass
            elif 'INFO - Mean Strain' in line and mean_strain is None:
                try: mean_strain = float(line.strip().split()[-1])
                except: pass

    if lsd is None:
        print("  ERROR: Failed to parse any MIDAS output")
        return None

    print(f"  MIDAS result ({total_time:.1f}s):")
    print(f"    Lsd   = {lsd:.1f} µm")
    print(f"    BC    = ({bc_y:.2f}, {bc_z:.2f}) px")
    print(f"    ty    = {ty:.4f}°" if ty else "    ty    = N/A")
    print(f"    tz    = {tz:.4f}°" if tz else "    tz    = N/A")
    if mean_strain:
        print(f"    Strain = {mean_strain*1e6:.1f} µε")

    result = {
        'lsd_um': lsd,
        'bc_y_px': bc_y,
        'bc_z_px': bc_z,
        'ty_deg': ty if ty else 0.0,
        'tz_deg': tz if tz else 0.0,
        'mean_strain': mean_strain,
        'runtime_s': total_time,
    }

    # Cache for future runs
    if dataset_name:
        _save_midas_cache(dataset_name, result)

    return result


def compare_results(pyfai_result, midas_result, dataset_cfg):
    """Print comparison table."""
    print(f"\n{'='*72}")
    print(f"  COMPARISON: {dataset_cfg['label']}")
    print(f"{'='*72}")

    if pyfai_result is None:
        print("  pyFAI calibration failed")
        return
    if midas_result is None:
        print("  MIDAS calibration failed")
        return

    lsd_d = pyfai_result['lsd_um'] - midas_result['lsd_um']
    bc_y_d = pyfai_result['bc_y_px'] - midas_result['bc_y_px']
    bc_z_d = pyfai_result['bc_z_px'] - midas_result['bc_z_px']

    # Stage 1 (zero-tilt) comparison
    s1_lsd_d = pyfai_result.get('s1_lsd_um', 0) - midas_result['lsd_um']
    s1_bc_y_d = pyfai_result.get('s1_bc_y_px', 0) - midas_result['bc_y_px']
    s1_bc_z_d = pyfai_result.get('s1_bc_z_px', 0) - midas_result['bc_z_px']

    print(f"\n  {'Parameter':<20s}  {'pyFAI(notilt)':<14s}  {'pyFAI(tilt)':<14s}  {'MIDAS':<14s}")
    print(f"  {'-'*72}")
    print(f"  {'Lsd (µm)':<20s}  {pyfai_result.get('s1_lsd_um',0):>14.1f}  "
          f"{pyfai_result['lsd_um']:>14.1f}  {midas_result['lsd_um']:>14.1f}")
    print(f"  {'BC_Y (px)':<20s}  {pyfai_result.get('s1_bc_y_px',0):>14.2f}  "
          f"{pyfai_result['bc_y_px']:>14.2f}  {midas_result['bc_y_px']:>14.2f}")
    print(f"  {'BC_Z (px)':<20s}  {pyfai_result.get('s1_bc_z_px',0):>14.2f}  "
          f"{pyfai_result['bc_z_px']:>14.2f}  {midas_result['bc_z_px']:>14.2f}")

    print(f"\n  {'Tilts':<20s}  {'pyFAI':>14s}  {'MIDAS':>14s}")
    print(f"  {'-'*50}")
    print(f"  {'rot1 / ty (°)':<20s}  {pyfai_result['rot1_deg']:>+14.4f}  "
          f"{midas_result['ty_deg']:>+14.4f}")
    print(f"  {'rot2 / tz (°)':<20s}  {pyfai_result['rot2_deg']:>+14.4f}  "
          f"{midas_result['tz_deg']:>+14.4f}")
    if 'rot3_deg' in pyfai_result:
        print(f"  {'rot3 (°)':<20s}  {pyfai_result['rot3_deg']:>+14.4f}  {'N/A':>14s}")

    print(f"\n  {'Metric':<20s}  {'pyFAI':>14s}  {'MIDAS':>14s}")
    print(f"  {'-'*50}")
    print(f"  {'Runtime (s)':<20s}  {pyfai_result['runtime_s']:>14.1f}  "
          f"{midas_result['runtime_s']:>14.1f}")
    print(f"  {'Control pts':<20s}  {pyfai_result['n_points']:>14d}  {'auto':>14s}")
    print(f"  {'Rings used':<20s}  {pyfai_result['n_rings']:>14d}  {'auto':>14s}")

    if pyfai_result.get('chi2') is not None:
        print(f"  {'χ² (pyFAI)':<20s}  {pyfai_result['chi2']:>14.6f}")
    if midas_result.get('mean_strain') is not None:
        print(f"  {'MeanStrain (µε)':<20s}  {'—':>14s}  "
              f"{midas_result['mean_strain']*1e6:>14.1f}")


# ── Pseudo-strain computation ──────────────────────────────────────────

def _pseudo_voigt(x, amp, cen, fwhm, eta_mix, bg):
    """Pseudo-Voigt profile: eta_mix*Lorentzian + (1-eta_mix)*Gaussian.
    Matches MIDAS's GSAS-II style peak fitting."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
    gamma = fwhm / 2.0  # Lorentzian HWHM
    G = np.exp(-(x - cen)**2 / (2 * sigma**2))
    L = gamma**2 / ((x - cen)**2 + gamma**2)
    return amp * (eta_mix * L + (1 - eta_mix) * G) + bg


def compute_pseudostrain(img, geom_dict, calibrant, wl_A, px_m,
                         n_eta_bins=360, max_rings=15, label=""):
    """Compute per-ring pseudo-strain using pyFAI's integrate2d.

    For each eta bin, fit pseudo-Voigt to each ring, compute d-spacing,
    compare with known CeO₂ d-spacings. Uses median for robustness.

    Parameters
    ----------
    img : 2D array
        Image after MIDAS transforms (transposed)
    geom_dict : dict
        Keys: lsd_um, bc_y_px, bc_z_px (used with zero tilts)
    calibrant : pyFAI calibrant
    wl_A : float
        Wavelength in Angstroms
    px_m : float
        Pixel size in meters
    n_eta_bins : int
        Number of azimuthal bins for 2D integration
    max_rings : int
        Maximum rings to fit
    label : str
        Label for printing

    Returns
    -------
    dict with:
        'strain_1d' : per-ring strain from 1D lineout
        'strain_2d' : (n_rings, n_eta) strain array from 2D cake
        'median_1d' : median |strain| from 1D
        'median_2d' : median |strain| from 2D
        'ring_tth'  : 2θ positions of fitted rings
    """
    from scipy.optimize import curve_fit
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    ny, nz = img.shape
    det = Detector(pixel1=px_m, pixel2=px_m, max_shape=(ny, nz))

    lsd_m = geom_dict['lsd_um'] * 1e-6
    bc_y = geom_dict['bc_y_px']
    bc_z = geom_dict['bc_z_px']
    wl_m = wl_A * 1e-10

    ai = AzimuthalIntegrator(
        dist=lsd_m,
        poni1=(bc_y + 0.5) * px_m,
        poni2=(bc_z + 0.5) * px_m,
        wavelength=wl_m, detector=det)

    # Known d-spacings and 2θ values
    tth_known_rad = np.array(calibrant.get_2th()[:max_rings])
    tth_known_deg = np.degrees(tth_known_rad)
    d_known = np.array(calibrant.get_dSpacing()[:max_rings])

    # 1D lineout
    tth_1d, I_1d = ai.integrate1d(img, 3000, unit='2th_deg')

    # 2D cake
    I_2d, tth_2d, eta_2d = ai.integrate2d(img, 3000, n_eta_bins,
                                            unit='2th_deg')
    # integrate2d returns bin edges (len n+1); I_2d has shape (n_eta, n_tth)
    n_tth_bins = I_2d.shape[1]
    n_eta_act = I_2d.shape[0]
    # Use bin centers: midpoint of edges
    if len(tth_2d) == n_tth_bins + 1:
        tth_centers = (tth_2d[:-1] + tth_2d[1:]) / 2
    else:
        tth_centers = tth_2d[:n_tth_bins]
    if len(eta_2d) == n_eta_act + 1:
        eta_centers = (eta_2d[:-1] + eta_2d[1:]) / 2
    else:
        eta_centers = eta_2d[:n_eta_act]

    # Fit peaks in 1D lineout
    strain_1d = []
    ring_fitted_tth = []
    ring_indices_used = []

    for i in range(len(tth_known_deg)):
        tth_exp = tth_known_deg[i]
        if tth_exp < tth_1d[0] + 0.2 or tth_exp > tth_1d[-1] - 0.2:
            continue

        hw = 0.12  # degrees half-width for peak fitting
        mask = (tth_1d > tth_exp - hw) & (tth_1d < tth_exp + hw)
        if np.sum(mask) < 5:
            continue

        x, y = tth_1d[mask], I_1d[mask]
        idx_max = np.argmax(y)
        amp0 = y[idx_max] - np.min(y)
        if amp0 <= 0:
            continue

        try:
            popt, _ = curve_fit(_pseudo_voigt, x, y,
                                p0=[amp0, x[idx_max], 0.02, 0.5, np.min(y)],
                                bounds=([0, tth_exp-hw, 0.002, 0, -np.inf],
                                        [np.inf, tth_exp+hw, hw, 1, np.inf]),
                                maxfev=5000)
            tth_fit = popt[1]
        except Exception:
            tth_fit = x[idx_max]

        d_meas = wl_A / (2 * np.sin(np.radians(tth_fit / 2)))
        strain = (d_meas - d_known[i]) / d_known[i]
        strain_1d.append(strain)
        ring_fitted_tth.append(tth_fit)
        ring_indices_used.append(i)

    strain_1d = np.array(strain_1d)
    median_1d = np.median(np.abs(strain_1d)) if len(strain_1d) > 0 else np.nan

    # Fit peaks in 2D cake (per eta bin)
    n_rings_used = len(ring_indices_used)
    strain_2d = np.full((n_rings_used, n_eta_act), np.nan)

    for j in range(n_eta_act):
        profile = I_2d[j, :]  # [n_tth_bins]
        if np.max(profile) <= 0:
            continue

        for ri, ring_idx in enumerate(ring_indices_used):
            tth_exp = tth_known_deg[ring_idx]
            hw = 0.12
            mask = (tth_centers > tth_exp - hw) & (tth_centers < tth_exp + hw)
            if np.sum(mask) < 5:
                continue

            x, y = tth_centers[mask], profile[mask]
            idx_max = np.argmax(y)
            amp0 = y[idx_max] - np.min(y)
            if amp0 <= 0:
                continue

            try:
                popt, _ = curve_fit(_pseudo_voigt, x, y,
                                    p0=[amp0, x[idx_max], 0.02, 0.5, np.min(y)],
                                    bounds=([0, tth_exp-hw, 0.002, 0, -np.inf],
                                            [np.inf, tth_exp+hw, hw, 1, np.inf]),
                                    maxfev=3000)
                tth_fit = popt[1]
            except Exception:
                continue

            d_meas = wl_A / (2 * np.sin(np.radians(tth_fit / 2)))
            strain_2d[ri, j] = (d_meas - d_known[ring_idx]) / d_known[ring_idx]

    # 2D median: per ring, median over valid eta bins (robust to outlier bins)
    ring_median_2d = np.nanmedian(np.abs(strain_2d), axis=1)
    median_2d = np.nanmedian(np.abs(strain_2d)) if not np.all(np.isnan(strain_2d)) else np.nan

    return {
        'strain_1d': strain_1d,
        'strain_2d': strain_2d,
        'ring_median_2d': ring_median_2d,
        'median_1d': median_1d,
        'median_2d': median_2d,
        'ring_indices': ring_indices_used,
        'ring_tth': np.array(ring_fitted_tth),
        'd_known': d_known[ring_indices_used] if len(ring_indices_used) > 0 else np.array([]),
        'eta_centers': eta_centers,
    }


def compare_pseudostrain(img, pyfai_result, midas_result, dataset_cfg, calibrant):
    """Compute and compare pseudo-strain for pyFAI and MIDAS geometries."""
    px_m = dataset_cfg['px_um'] * 1e-6
    wl_A = dataset_cfg['wavelength_A']
    wl_m = wl_A * 1e-10

    calibrant.wavelength = wl_m

    print(f"\n{'─'*72}")
    print(f"  PSEUDO-STRAIN COMPARISON: {dataset_cfg['label']}")
    print(f"{'─'*72}")

    # pyFAI geometry (Stage 1 = zero tilts)
    pyfai_geom = {
        'lsd_um': pyfai_result.get('s1_lsd_um', pyfai_result['lsd_um']),
        'bc_y_px': pyfai_result.get('s1_bc_y_px', pyfai_result['bc_y_px']),
        'bc_z_px': pyfai_result.get('s1_bc_z_px', pyfai_result['bc_z_px']),
    }

    # MIDAS geometry (zero tilts for fair comparison via pyFAI integrator)
    midas_geom = {
        'lsd_um': midas_result['lsd_um'],
        'bc_y_px': midas_result['bc_y_px'],
        'bc_z_px': midas_result['bc_z_px'],
    }

    print(f"\n  Computing pyFAI pseudo-strain (pyFAI geom, zero tilts)...")
    ps_pyfai = compute_pseudostrain(img, pyfai_geom, calibrant, wl_A, px_m,
                                     label="pyFAI")

    print(f"  Computing MIDAS pseudo-strain (MIDAS geom, zero tilts, pyFAI integrator)...")
    ps_midas = compute_pseudostrain(img, midas_geom, calibrant, wl_A, px_m,
                                     label="MIDAS")

    # Print per-ring comparison
    n = min(len(ps_pyfai['strain_1d']), len(ps_midas['strain_1d']))
    print(f"\n  {'Ring':<6s}  {'2θ (°)':<8s}  {'d (Å)':<10s}  "
          f"{'pyFAI ε (µε)':<14s}  {'MIDAS ε (µε)':<14s}  "
          f"{'pyFAI 2D (µε)':<14s}  {'MIDAS 2D (µε)':<14s}")
    print(f"  {'-'*80}")
    for i in range(n):
        ri = ps_pyfai['ring_indices'][i]
        p_1d = ps_pyfai['strain_1d'][i] * 1e6
        m_1d = ps_midas['strain_1d'][i] * 1e6
        p_2d = ps_pyfai['ring_median_2d'][i] * 1e6 if i < len(ps_pyfai['ring_median_2d']) else np.nan
        m_2d = ps_midas['ring_median_2d'][i] * 1e6 if i < len(ps_midas['ring_median_2d']) else np.nan
        tth = ps_pyfai['ring_tth'][i]
        d_k = ps_pyfai['d_known'][i]
        print(f"  {ri:<6d}  {tth:<8.4f}  {d_k:<10.5f}  "
              f"{p_1d:<+14.1f}  {m_1d:<+14.1f}  "
              f"{p_2d:<14.1f}  {m_2d:<14.1f}")

    print(f"\n  Summary (median |ε|):")
    print(f"    1D median |ε|:  pyFAI = {ps_pyfai['median_1d']*1e6:.1f} µε,  "
          f"MIDAS = {ps_midas['median_1d']*1e6:.1f} µε")
    print(f"    2D median |ε|:  pyFAI = {ps_pyfai['median_2d']*1e6:.1f} µε,  "
          f"MIDAS = {ps_midas['median_2d']*1e6:.1f} µε")
    if midas_result.get('mean_strain') is not None:
        print(f"    MIDAS CalibrantPanelShiftsOMP: {midas_result['mean_strain']*1e6:.1f} µε")

    return {'pyfai': ps_pyfai, 'midas': ps_midas}


def main():
    parser = argparse.ArgumentParser(
        description='pyFAI vs MIDAS calibration benchmark')
    parser.add_argument('--pilatus', action='store_true')
    parser.add_argument('--varex', action='store_true')
    parser.add_argument('--ge5', action='store_true')
    parser.add_argument('--ge5-summed', action='store_true',
                        dest='ge5_summed')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--strain', action='store_true',
                        help='Also compute pseudo-strain comparison')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-run MIDAS (ignore cached results)')
    args = parser.parse_args()

    if args.all:
        args.pilatus = True
        args.varex = True
        args.ge5 = True
        args.ge5_summed = True
    if not (args.pilatus or args.varex or args.ge5 or args.ge5_summed):
        args.pilatus = True

    print("=" * 72)
    print("  Paper 3: pyFAI vs MIDAS Calibration Benchmark")
    print("=" * 72)

    all_results = {}

    for name, cfg in DATASETS.items():
        if not getattr(args, name, False):
            continue

        data_path = os.path.join(cfg['data_dir'], cfg['data_file'])
        if not os.path.exists(data_path):
            print(f"\n  SKIP: {cfg['label']} — data not found")
            continue

        print(f"\n{'#'*72}")
        print(f"  Dataset: {cfg['label']}")
        print(f"{'#'*72}")

        print(f"\n[1/2] pyFAI geometry refinement...")
        pyfai_r = run_pyfai_calibration(cfg)

        print(f"\n[2/2] MIDAS AutoCalibrateZarr (fully automated)...")
        midas_r = run_midas_calibration(cfg, dataset_name=name,
                                         use_cache=not args.no_cache)

        compare_results(pyfai_r, midas_r, cfg)

        # Pseudo-strain comparison
        if args.strain and pyfai_r and midas_r:
            from pyFAI.calibrant import get_calibrant
            cal = get_calibrant('CeO2')
            img = load_and_transform_image(cfg)
            ps_results = compare_pseudostrain(
                img, pyfai_r, midas_r, cfg, cal)
            all_results[name] = {'pyfai': pyfai_r, 'midas': midas_r,
                                 'strain': ps_results}
        else:
            all_results[name] = {'pyfai': pyfai_r, 'midas': midas_r}

    # Final summary
    if len(all_results) > 1:
        print(f"\n\n{'='*72}")
        print("  CROSS-DATASET SUMMARY")
        print(f"{'='*72}")
        for name, res in all_results.items():
            cfg = DATASETS[name]
            p, m = res['pyfai'], res['midas']
            if p and m:
                lsd_d = abs(p['lsd_um'] - m['lsd_um'])
                bc_d = np.sqrt((p['bc_y_px']-m['bc_y_px'])**2 +
                               (p['bc_z_px']-m['bc_z_px'])**2)
                line = f"  {cfg['label']:<40s}  ΔLsd={lsd_d:.0f}µm  ΔBC={bc_d:.2f}px"
                if 'strain' in res:
                    ps = res['strain']
                    line += (f"  pyFAI_ε={ps['pyfai']['median_2d']*1e6:.0f}µε"
                             f"  MIDAS_ε={ps['midas']['median_2d']*1e6:.0f}µε")
                print(line)


if __name__ == '__main__':
    main()
