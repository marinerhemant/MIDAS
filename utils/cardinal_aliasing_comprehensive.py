#!/usr/bin/env python3
"""
Cardinal-Angle Aliasing — Comprehensive Test Suite
====================================================
Demonstrates the cardinal-angle aliasing artifact using:
  1. MIDAS DetectorMapper + IntegratorZarrOMP  (real CeO₂ data)
  2. pyFAI AzimuthalIntegrator                 (same real data)
  3. BinArea flatness analysis                 (proves the map is correct)

Generates publication-quality figures.

Usage:
    python cardinal_aliasing_comprehensive.py
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

# ─── Configuration ─────────────────────────────────────────────────
DATA_DIR   = '/Users/hsharma/Desktop/analysis/calibration/andrew'
PARAM_FILE = 'ps_local.txt'
DATA_FILE  = 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'
MIDAS_BIN  = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin')
FIG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
WORK_BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_runs')
R_RING     = 955.0
CARDINALS  = [-180, -90, 0, 90, 180]
ETA_BIN_SIZE = 0.5

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(WORK_BASE, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════

def cardinal_boxes(ax, eta_range, box_hw=10):
    """Draw red boxes (±box_hw°) around cardinal angles on a cake axis."""
    from matplotlib.patches import Rectangle
    xlim = ax.get_xlim()
    for ca in [0, 90, -90]:
        if eta_range[0] <= ca <= eta_range[1]:
            rect = Rectangle((xlim[0], ca - box_hw),
                             xlim[1] - xlim[0], 2 * box_hw,
                             linewidth=1.2, edgecolor='red',
                             facecolor='none', linestyle='-', alpha=0.7)
            ax.add_patch(rect)


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

    # Run DetectorMapper (only if Map.bin doesn't already exist)
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
        return None
    print(f" done ({dt:.1f}s)")
    return caked


def read_caked_hdf(fn):
    """Read .caked.hdf → R(nR,), Eta(nEta,), I(nR,nEta), Area(nR,nEta)."""
    with h5py.File(fn, 'r') as f:
        remap = f['REtaMap'][:]
        I = f['IntegrationResult/FrameNr_0'][:]
    R    = remap[0, :, 0]
    Eta  = remap[2, 0, :]
    Area = remap[3, :, :]
    return R, Eta, I, Area


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1: MIDAS ΔR sweep
# ═══════════════════════════════════════════════════════════════════

def experiment_midas():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: MIDAS ΔR sweep")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))

    # ΔR values to test: sub-pixel (aliased) and ≥1 px (clean)
    delta_rs = [0.25, 0.5, 0.75, 1.0, 2.0]
    results = {}

    for dr in delta_rs:
        print(f"\n  ΔR = {dr} px:")
        work_dir = os.path.join(WORK_BASE, f'dr_{str(dr).replace(".", "p")}')
        caked = run_midas_case(work_dir, base_lines, {
            'RBinSize': str(dr),
            'SubPixelLevel': '1',
            'EtaBinSize': str(ETA_BIN_SIZE),
        })
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            ridx = np.argmin(np.abs(R - R_RING))
            results[dr] = {
                'R': R, 'Eta': Eta, 'I': I, 'Area': Area,
                'I_eta': I[ridx, :], 'Area_eta': Area[ridx, :],
                'R_actual': R[ridx], 'ridx': ridx,
            }
            print(f"    Ring at R = {R[ridx]:.1f} px (index {ridx})")
        else:
            print(f"    FAILED")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2: Tilt dependence
# ═══════════════════════════════════════════════════════════════════

def experiment_tiltstudy():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Tilt dependence")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))

    # Parse real BC, tilts from param file
    real_bc_y, real_bc_z, real_ty, real_tz = 0, 0, 0, 0
    for line in base_lines:
        toks = line.strip().split()
        if not toks or toks[0].startswith('#'):
            continue
        if toks[0] == 'BC' and len(toks) >= 3:
            real_bc_y, real_bc_z = float(toks[1]), float(toks[2])
        elif toks[0] == 'ty':
            real_ty = float(toks[1])
        elif toks[0] == 'tz':
            real_tz = float(toks[1])

    int_bc_y, int_bc_z = round(real_bc_y), round(real_bc_z)

    cases = [
        ('Int BC, no tilt',  int_bc_y, int_bc_z, 0, 0),
        ('Real BC, no tilt', real_bc_y, real_bc_z, 0, 0),
        ('Int BC, real tilt', int_bc_y, int_bc_z, real_ty, real_tz),
        ('Real BC, real tilt', real_bc_y, real_bc_z, real_ty, real_tz),
    ]

    results = {}
    for label, bcy, bcz, ty, tz in cases:
        tag = label.replace(' ', '_').replace(',', '')
        print(f"\n  {label}:")
        work_dir = os.path.join(WORK_BASE, f'tilt_{tag}')
        caked = run_midas_case(work_dir, base_lines, {
            'RBinSize': '0.5',
            'SubPixelLevel': '1',
            'BC': f'{bcy} {bcz}',
            'ty': str(ty),
            'tz': str(tz),
            'EtaBinSize': str(ETA_BIN_SIZE),
        })
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            ridx = np.argmin(np.abs(R - R_RING))
            results[label] = {'Eta': Eta, 'I_eta': I[ridx, :]}

    return results


# ═══════════════════════════════════════════════════════════════════
#  Experiment 4: tx rotation
# ═══════════════════════════════════════════════════════════════════

def experiment_tx_rotation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Detector rotation (tx)")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    tx_values = [0, 30, 45, 90]

    results = {}
    for tx in tx_values:
        print(f"\n  tx = {tx}°:")
        work_dir = os.path.join(WORK_BASE, f'tx_{tx}')
        caked = run_midas_case(work_dir, base_lines, {
            'RBinSize': '0.5',
            'SubPixelLevel': '1',
            'ty': '0',
            'tz': '0',
            'tx': str(tx),
            'EtaBinSize': str(ETA_BIN_SIZE),
        })
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            ridx = np.argmin(np.abs(R - R_RING))
            results[tx] = {'Eta': Eta, 'I_eta': I[ridx, :]}

    return results


# ═══════════════════════════════════════════════════════════════════
#  Experiment 5: pyFAI comparison
# ═══════════════════════════════════════════════════════════════════

def experiment_pyfai():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: pyFAI comparison (zero tilts)")
    print("=" * 60)

    # ── Run MIDAS with zero tilts for matched comparison ──────────
    print("\n  Running MIDAS with zero tilts for matched comparison...")
    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    zero_tilt_overrides = {
        'ty': '0', 'tz': '0',
        'p0': '0', 'p1': '0', 'p2': '0', 'p3': '0', 'p4': '0', 'p5': '0',
        'SubPixelLevel': '1',
        'EtaBinSize': str(ETA_BIN_SIZE),
    }
    midas_zt = {}  # MIDAS zero-tilt results
    for dr in [0.5, 1.0, 2.0]:
        work_dir = os.path.join(WORK_BASE, f'zt_dr_{str(dr).replace(".", "p")}')
        overrides = dict(zero_tilt_overrides)
        overrides['RBinSize'] = str(dr)
        caked = run_midas_case(work_dir, base_lines, overrides)
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            ridx = np.argmin(np.abs(R - R_RING))
            midas_zt[dr] = {
                'R': R, 'Eta': Eta, 'I': I, 'Area': Area,
                'I_eta': I[ridx, :], 'Area_eta': Area[ridx, :],
                'R_actual': R[ridx], 'ridx': ridx,
            }
            print(f"    ΔR={dr}: ring at R={R[ridx]:.1f}")

    # ── Run pyFAI with zero tilts ──────────────────────────────────
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
        return None, midas_zt

    # Read MIDAS parameters
    bc_y_px = bc_z_px = lsd_um = px_um = wl_A = 0
    npy = npz = 2880
    rmin = 10.25; rmax = 1500.25; eta_bs = 1.0
    with open(os.path.join(DATA_DIR, PARAM_FILE)) as f:
        for line in f:
            toks = line.strip().split()
            if not toks or toks[0].startswith('#'):
                continue
            if toks[0] == 'BC':
                bc_y_px, bc_z_px = float(toks[1]), float(toks[2])
            elif toks[0] == 'Lsd':
                lsd_um = float(toks[1])
            elif toks[0] == 'px':
                px_um = float(toks[1])
            elif toks[0] == 'Wavelength':
                wl_A = float(toks[1])
            elif toks[0] == 'NrPixelsY':
                npy = int(toks[1])
            elif toks[0] == 'NrPixelsZ':
                npz = int(toks[1])
            elif toks[0] == 'RMin':
                rmin = float(toks[1])
            elif toks[0] == 'RMax':
                rmax = float(toks[1])
            elif toks[0] == 'EtaBinSize':
                eta_bs = float(toks[1])

    px_m = px_um * 1e-6
    lsd_m = lsd_um * 1e-6
    wl_m = wl_A * 1e-10

    # Load image and apply MIDAS transforms:
    #   ImTransOpt 2 = flipud,  then MIDAS always transposes
    # So the image pyFAI sees must match what MIDAS feeds to its integrator.
    img_path = os.path.join(DATA_DIR, DATA_FILE)
    img_raw = tifffile.imread(img_path).astype(np.float64)
    img = np.flipud(img_raw)  # ImTransOpt 2
    img = img.T               # MIDAS always transposes after ImTransOpt
    print(f"  Image shape after transform: {img.shape}")
    print(f"  MIDAS BC: y={bc_y_px:.2f}, z={bc_z_px:.2f}")
    print(f"  Lsd={lsd_um:.1f} µm, px={px_um:.1f} µm, λ={wl_A:.5f} Å")

    # In MIDAS, after transforms, the image is addressed as [y, z]
    # with BC = (bc_y_px, bc_z_px).
    # pyFAI PONI1 = vertical distance from corner (dim-0 direction)
    # pyFAI PONI2 = horizontal distance from corner (dim-1 direction)
    #
    # CRITICAL: pixel-center convention difference!
    # pyFAI places pixel center at (i+0.5)*pixel_size (half-pixel offset)
    # MIDAS uses i*pixel_size (pixel index IS the center coordinate)
    # We must add 0.5 px to both PONI coordinates to match.
    det = Detector(pixel1=px_m, pixel2=px_m,
                   max_shape=(img.shape[0], img.shape[1]))
    ai = AzimuthalIntegrator(
        dist=lsd_m,
        poni1=(bc_y_px + 0.5) * px_m,
        poni2=(bc_z_px + 0.5) * px_m,
        wavelength=wl_m,
        detector=det
    )

    r_min_mm = rmin * px_m * 1e3
    r_max_mm = rmax * px_m * 1e3
    npt_azim = 361

    delta_rs = [0.5, 1.0, 2.0]
    results = {}

    for dr in delta_rs:
        npt_rad = int((rmax - rmin) / dr)
        print(f"\n  ΔR ≈ {dr} px (npt_rad={npt_rad}) ...", end='', flush=True)
        try:
            # correctSolidAngle=False: MIDAS does not apply solid-angle
            # correction (cos³(2θ)), so we disable it here to match.
            res = ai.integrate2d(
                img, npt_rad=npt_rad, npt_azim=npt_azim,
                unit="r_mm",
                method=("full", "histogram", "cython"),
                radial_range=(r_min_mm, r_max_mm),
                azimuth_range=(-180, 180),
                correctSolidAngle=False,
            )
            r_ring_mm = R_RING * px_m * 1e3
            ridx = np.argmin(np.abs(res.radial - r_ring_mm))
            I_eta_ring = res.intensity[:, ridx]
            # Diagnostics
            mask0 = np.abs(res.azimuthal) < 5
            vals0 = I_eta_ring[mask0]
            nz = np.count_nonzero(vals0)
            print(f" done (ring at r_mm={res.radial[ridx]:.3f}, "
                  f"I_eta near 0°: mean={np.mean(vals0):.1f}, "
                  f"nonzero={nz}/{len(vals0)})")
            results[dr] = {
                'eta': res.azimuthal,
                'I_eta': I_eta_ring,
                'I2d': res.intensity,
                'r_mm': res.radial,
                'ridx': ridx,
                'npt_rad': npt_rad,
            }
        except Exception as e:
            import traceback
            print(f" FAILED: {e}")
            traceback.print_exc()

    return results, midas_zt


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png}}")


def plot_schematic(fname):
    """Fig 1: schematic of pixel splitting at cardinal vs non-cardinal angles.
    Two panels showing a 6×6 pixel grid with ΔR=0.5 bin boundaries."""
    from matplotlib.patches import Rectangle

    ny, nz = 5, 6  # pixel grid dimensions (Y cols, Z rows)
    # Pixel-averaged intensities: ring at z=5,6 (high), background elsewhere
    pixel_vals = np.full((nz, ny), 25, dtype=int)
    pixel_vals[2:4, :] = 444  # ring region (z-rows 5 and 6 in display coords)

    bg_color = '#4C566A'     # dark grey for background
    ring_color = '#FF9F43'   # orange for ring

    def draw_panel(ax, title, angle_deg):
        """Draw pixel grid with bin boundaries at given angle."""
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Draw pixel rectangles with values
        for iz in range(nz):
            for iy in range(ny):
                val = pixel_vals[iz, iy]
                color = ring_color if val > 100 else bg_color
                rect = Rectangle((iy - 0.5, nz - 1 - iz - 0.5), 1, 1,
                                 facecolor=color, edgecolor='grey',
                                 lw=0.5, alpha=0.85)
                ax.add_patch(rect)
                ax.text(iy, nz - 1 - iz, str(val),
                        ha='center', va='center', fontsize=9,
                        color='white' if val < 100 else 'black',
                        fontweight='bold')

        # Draw bin boundaries (cyan dashed)
        dr = 0.5  # bin width in pixel units
        if angle_deg == 0:
            # Horizontal lines (R-bins parallel to pixel rows)
            for r in np.arange(-0.5 - dr, nz + 0.5 + dr, dr):
                ax.axhline(r, color='cyan', ls='--', lw=2.0, alpha=0.8)
        elif angle_deg == 45:
            # Diagonal lines at 45°
            for offset in np.arange(-nz - ny, nz + ny + 1, dr):
                y_start = -1
                y_end = ny + 1
                z_start = offset + y_start
                z_end = offset + y_end
                ax.plot([y_start, y_end], [z_start, z_end],
                        color='cyan', ls='--', lw=2.0, alpha=0.8)

        ax.set_xlim(-0.7, ny - 0.3)
        ax.set_ylim(-0.7 + (nz - 1 - (nz - 1)), nz - 0.3)
        ax.set_xlim(-0.7, ny - 0.3)
        ax.set_ylim(-0.7, nz - 0.3)
        ax.set_xlabel('Y (pixels)', fontsize=11)
        ax.set_ylabel('Z (pixels)', fontsize=11)
        ax.set_xticks(range(ny))
        ax.set_yticks(range(nz))
        ax.set_yticklabels([str(i + 3) for i in range(nz)])
        ax.set_aspect('equal')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Pixel splitting geometry at cardinal vs.\\ non-cardinal angles',
                 fontsize=14, fontweight='bold', y=1.02)

    draw_panel(ax1, '(a) Cardinal angle (η = 0°): R-bins ∥ pixel rows', 0)
    draw_panel(ax2, '(b) Non-cardinal (η = 45°): R-bins cross pixel rows', 45)

    fig.tight_layout()
    save(fig, fname)


def plot_cake(midas_results, dr_key, title_suffix, fname):
    """2D cake (X=R, Y=η) zoomed around the ring — like the user's image."""
    if dr_key not in midas_results:
        return
    d = midas_results[dr_key]
    R, Eta, I = d['R'], d['Eta'], d['I']

    r_half = 25
    rmask = (R >= R_RING - r_half) & (R <= R_RING + r_half)
    r_idx = np.where(rmask)[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    Rsub = R[r_idx]
    Isub = I[r_idx, :].T.astype(float)
    Isub[Isub <= 0] = np.nan
    from matplotlib.colors import LogNorm
    vmin_log = max(1, np.nanpercentile(Isub[np.isfinite(Isub)], 1))
    vmax = np.nanpercentile(Isub[np.isfinite(Isub)], 99.5)
    im = ax.pcolormesh(Rsub, Eta, Isub, shading='auto', cmap='viridis',
                       norm=LogNorm(vmin=vmin_log, vmax=vmax), rasterized=True)
    ax.set_xlabel('R (pixels)', fontsize=12)
    ax.set_ylabel('η (°)', fontsize=12)
    ax.set_title(f'MIDAS cake (ΔR = {dr_key} px) — {title_suffix}',
                 fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Intensity (counts)', pad=0.02)
    cardinal_boxes(ax, (Eta.min(), Eta.max()))
    fig.tight_layout()
    save(fig, fname)


def plot_raw_vs_caked(midas_results, fname):
    """Generate raw detector image (Cartesian) alongside R-η cake."""
    from matplotlib.patches import Circle
    from PIL import Image as PILImage

    dr = 0.5
    if dr not in midas_results:
        return

    d = midas_results[dr]
    R, Eta, I_cake = d['R'], d['Eta'], d['I']

    # Load raw image
    img_path = os.path.join(DATA_DIR, DATA_FILE)
    raw = np.array(PILImage.open(img_path), dtype=np.float64)

    # Dark subtraction
    dark_files = [fn for fn in os.listdir(DATA_DIR)
                  if fn.startswith('dark') and fn.endswith('.tif')]
    if dark_files:
        dark = np.array(PILImage.open(os.path.join(DATA_DIR, dark_files[0])),
                        dtype=np.float64)
        raw -= dark

    # Parse beam center and ImTransOpt
    bc_y_midas, bc_z_midas = 0, 0
    im_trans_opt = 0
    npz, npy = raw.shape
    with open(os.path.join(DATA_DIR, PARAM_FILE)) as f:
        for line in f:
            toks = line.strip().split()
            if not toks or toks[0].startswith('#'):
                continue
            if toks[0] == 'BC':
                bc_y_midas, bc_z_midas = float(toks[1]), float(toks[2])
            elif toks[0] == 'ImTransOpt':
                im_trans_opt = int(toks[1])

    # Convert BC from MIDAS (post-transform) to raw image coordinates
    bc_y_raw = bc_y_midas
    bc_z_raw = bc_z_midas
    if im_trans_opt == 1:    # flip Y
        bc_y_raw = npy - 1 - bc_y_midas
    elif im_trans_opt == 2:  # flip Z
        bc_z_raw = npz - 1 - bc_z_midas

    ring_Rs = [377, 617, 955]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={'width_ratios': [1, 1.3]})

    vmax = np.percentile(raw[raw > 0], 99.5)
    ax1.imshow(raw, origin='lower', cmap='inferno', vmin=0, vmax=vmax,
               aspect='equal')
    ax1.set_xlabel('Y (pixels)', fontsize=11)
    ax1.set_ylabel('Z (pixels)', fontsize=11)
    ax1.set_title('(a) Raw detector image (Cartesian)', fontsize=12)
    ax1.plot(bc_y_raw, bc_z_raw, 'c+', ms=12, mew=1.5)

    for r_px in ring_Rs:
        circ = Circle((bc_y_raw, bc_z_raw), r_px, fill=False, color='cyan',
                      ls='--', lw=0.8, alpha=0.7)
        ax1.add_patch(circ)

    # η direction overlay
    import matplotlib.patches as mpatches
    eta_r = 250  # radius for η labels
    # MIDAS convention: Y_det (right) is flipped to Y_lab (left).
    # Z_lab = Z_det (up).  η = atan2(-Y_lab, Z_lab).
    # Since Y_lab is to the LEFT, -Y_lab is to the RIGHT.
    # η=+90° at RIGHT (in image: bc_y + r), η=-90° at LEFT (bc_y - r)
    # With origin='lower': z=0 at bottom, z increases upward
    labels = [
        (bc_y_raw,          bc_z_raw + eta_r, 'η=0°'),
        (bc_y_raw + eta_r,  bc_z_raw,         'η=+90°'),
        (bc_y_raw,          bc_z_raw - eta_r, 'η=±180°'),
        (bc_y_raw - eta_r,  bc_z_raw,         'η=−90°'),
    ]
    for (lx, ly, txt) in labels:
        ax1.annotate(txt, xy=(lx, ly), fontsize=8, color='white',
                     fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', fc='black',
                               alpha=0.6, ec='none'))

    # Curved arrow showing clockwise direction (from 0° toward +90°, top toward right)
    arc = mpatches.FancyArrowPatch(
        (bc_y_raw, bc_z_raw + eta_r * 0.7),
        (bc_y_raw + eta_r * 0.7, bc_z_raw),
        connectionstyle="arc3,rad=-0.4",
        arrowstyle='->', color='white', lw=1.5, mutation_scale=12)
    ax1.add_patch(arc)

    # Y, Z coordinate arrows near bottom-left corner
    arr_x, arr_y = 150, 150  # bottom-left in display coords
    arr_len = 120
    ax1.annotate('', xy=(arr_x, arr_y + arr_len), xytext=(arr_x, arr_y),
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
    ax1.text(arr_x + 10, arr_y + arr_len + 15, 'Z', color='yellow',
             fontsize=9, fontweight='bold')
    ax1.annotate('', xy=(arr_x + arr_len, arr_y), xytext=(arr_x, arr_y),
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
    ax1.text(arr_x + arr_len + 10, arr_y + 5, 'Y', color='yellow',
             fontsize=9, fontweight='bold')

    from matplotlib.colors import LogNorm as _LN
    cake_display = I_cake.copy().astype(float)
    cake_display[cake_display <= 0] = np.nan
    vmin_cake = max(1, np.nanpercentile(cake_display[np.isfinite(cake_display)], 1))
    vmax_cake = np.nanpercentile(cake_display[np.isfinite(cake_display)], 99.5)
    ax2.imshow(cake_display, origin='lower', cmap='viridis',
               norm=_LN(vmin=vmin_cake, vmax=vmax_cake), aspect='auto',
               extent=[Eta.min(), Eta.max(), R.min(), R.max()])
    ax2.set_xlabel('η (degrees)', fontsize=11)
    ax2.set_ylabel('R (pixels)', fontsize=11)
    ax2.set_title('(b) Azimuthal integration (R–η cake)', fontsize=12)

    for r_px in ring_Rs:
        ax2.axhline(r_px, color='cyan', ls='--', lw=0.5, alpha=0.5)

    fig.tight_layout()
    save(fig, fname)


def plot_cake_comparison(midas_results, fname):
    """Multi-panel cake showing ΔR=0.5 raw + residual, 1.0, 2.0.
    The residual panel row-normalizes each R-row to reveal azimuthal
    modulation as ±% deviations, making cardinal-angle striping
    dramatically visible."""
    drs = [0.5, 1.0, 2.0]
    available = [dr for dr in drs if dr in midas_results]
    if not available:
        return

    # Extra panel for ΔR=0.5 residual
    nrows = len(available)
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(12, 3.5 * nrows),
                             sharex=False)
    if nrows == 1:
        axes = [axes]
    fig.suptitle('2D Cake: CeO₂ (220) — effect of radial bin width',
                 fontsize=13, fontweight='bold')

    r_half = 25
    ax_idx = 0

    for dr in available:
        d = midas_results[dr]
        R, Eta, I = d['R'], d['Eta'], d['I']
        rmask = (R >= R_RING - r_half) & (R <= R_RING + r_half)
        r_idx = np.where(rmask)[0]
        Rsub = R[r_idx]
        Isub = I[r_idx, :].T  # shape (nEta, nR_sub)

        Isub = Isub.astype(float)
        Isub[Isub <= 0] = np.nan
        from matplotlib.colors import LogNorm as _LN2
        vmin_log = max(1, np.nanpercentile(Isub[np.isfinite(Isub)], 1))
        vmax = np.nanpercentile(Isub[np.isfinite(Isub)], 99.5)

        ax = axes[ax_idx]
        im = ax.pcolormesh(Rsub, Eta, Isub, shading='auto', cmap='viridis',
                           norm=_LN2(vmin=vmin_log, vmax=vmax), rasterized=True)
        ax.set_ylabel('η (°)', fontsize=11)
        label = '← ALIASED (striping at cardinal angles)' if dr < 1.0 else '← CLEAN'
        ax.set_title(f'ΔR = {dr} px  {label}', fontsize=11, loc='left')
        fig.colorbar(im, ax=ax, label='Intensity (counts)', pad=0.02, shrink=0.9)
        cardinal_boxes(ax, (Eta.min(), Eta.max()))
        ax_idx += 1

    axes[-1].set_xlabel('R (pixels)', fontsize=11)
    fig.tight_layout()
    save(fig, fname)


def plot_pyfai_cake(midas_zt_results, pyfai_results, fname):
    """Side-by-side MIDAS vs pyFAI 2D cake zoomed around the ring at ΔR=0.5.

    Shows that both integrators produce the same cardinal-angle striping
    in the 2D cake representation.
    """
    dr = 0.5
    if dr not in pyfai_results or dr not in midas_zt_results:
        return

    from matplotlib.colors import LogNorm as _LN3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f'2D Cake comparison at ΔR = {dr} px — cardinal aliasing in both integrators',
                 fontsize=13, fontweight='bold')

    r_half = 25

    # ── Panel (a): MIDAS zero-tilt ─────────────────────────────
    dm = midas_zt_results[dr]
    R_m, Eta_m, I_m = dm['R'], dm['Eta'], dm['I']
    rmask_m = (R_m >= R_RING - r_half) & (R_m <= R_RING + r_half)
    r_idx_m = np.where(rmask_m)[0]
    Rsub_m = R_m[r_idx_m]
    Isub_m = I_m[r_idx_m, :].T.astype(float)
    Isub_m[Isub_m <= 0] = np.nan
    vmin_m = max(1, np.nanpercentile(Isub_m[np.isfinite(Isub_m)], 1))
    vmax_m = np.nanpercentile(Isub_m[np.isfinite(Isub_m)], 99.5)

    im1 = ax1.pcolormesh(Rsub_m, Eta_m, Isub_m, shading='auto', cmap='viridis',
                         norm=_LN3(vmin=vmin_m, vmax=vmax_m), rasterized=True)
    ax1.set_xlabel('R (pixels)', fontsize=11)
    ax1.set_ylabel('η (°)', fontsize=11)
    ax1.set_title(f'(a) MIDAS — ΔR = {dr} px', fontsize=12)
    fig.colorbar(im1, ax=ax1, label='Intensity', pad=0.02, shrink=0.85)
    cardinal_boxes(ax1, (Eta_m.min(), Eta_m.max()))

    # ── Panel (b): pyFAI ────────────────────────────────────────
    dp = pyfai_results[dr]
    # pyFAI stores I2d as (npt_azim, npt_rad) and r_mm
    px_m = 150e-6  # pixel pitch in metres
    r_px = dp['r_mm'] / (px_m * 1e3)  # convert mm back to pixels
    eta_p = dp['eta']
    I2d_p = dp['I2d'].astype(float)
    I2d_p[I2d_p <= 0] = np.nan

    # Zoom to ring
    rmask_p = (r_px >= R_RING - r_half) & (r_px <= R_RING + r_half)
    r_idx_p = np.where(rmask_p)[0]
    Rsub_p = r_px[r_idx_p]
    Isub_p = I2d_p[:, r_idx_p]  # shape (nEta, nR_sub)

    vmin_p = max(1, np.nanpercentile(Isub_p[np.isfinite(Isub_p)], 1))
    vmax_p = np.nanpercentile(Isub_p[np.isfinite(Isub_p)], 99.5)

    im2 = ax2.pcolormesh(Rsub_p, eta_p, Isub_p, shading='auto', cmap='viridis',
                         norm=_LN3(vmin=vmin_p, vmax=vmax_p), rasterized=True)
    ax2.set_xlabel('R (pixels)', fontsize=11)
    ax2.set_ylabel('η (°)', fontsize=11)
    ax2.set_title(f'(b) pyFAI — ΔR = {dr} px', fontsize=12)
    fig.colorbar(im2, ax=ax2, label='Intensity', pad=0.02, shrink=0.85)
    cardinal_boxes(ax2, (eta_p.min(), eta_p.max()))

    fig.tight_layout()
    save(fig, fname)


def plot_ieta_sweep(results, label, fname):
    """I(η) at the ring — all ΔR values overlaid on one plot."""
    colours = {0.25: '#e63946', 0.5: '#457b9d', 0.75: '#f4a261',
               1.0: '#2a9d8f', 2.0: '#264653'}
    sorted_drs = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'{label}: I(η) at R ≈ {R_RING} px (CeO₂ 220)',
                 fontsize=13, fontweight='bold')

    for dr in sorted_drs:
        if 'Eta' in results[dr]:
            eta = results[dr]['Eta']
        else:
            eta = results[dr]['eta']
        I_eta = results[dr]['I_eta']
        col = colours.get(dr, 'black')

        mask0 = np.abs(eta) < 5
        vals = I_eta[mask0]
        std_lbl = ''
        if len(vals) > 2 and np.mean(vals) > 0:
            std_rel = np.std(vals) / np.mean(vals)
            std_lbl = f' (σ/μ={std_rel:.3f})'

        ax.plot(eta, I_eta, lw=1.5, color=col, alpha=0.85,
                label=f'ΔR = {dr} px{std_lbl}')

    for ca in CARDINALS:
        ax.axvline(ca, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_ylabel('I (a.u.)', fontsize=11)
    ax.set_xlabel('η (°)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(MultipleLocator(45))

    fig.tight_layout()
    save(fig, fname)


def plot_binarea(midas_results, fname):
    """BinArea residual vs η — all ΔR values overlaid on one plot.
    Plots ΔA = A(η) - <A> centered at 0."""
    colours = {0.5: '#457b9d', 1.0: '#2a9d8f', 2.0: '#264653'}
    sorted_drs = sorted(midas_results.keys())

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'BinArea residual (ΔA = A − ⟨A⟩) vs η at R ≈ {R_RING} px\n'
                 'Near-zero residual proves pixel-splitting geometry is exact — '
                 'intensity oscillations are NOT a map error',
                 fontsize=11, fontweight='bold')

    for dr in sorted_drs:
        d = midas_results[dr]
        Eta = d['Eta']
        Area_eta = d['Area_eta']
        col = colours.get(dr, 'black')

        valid = Area_eta[Area_eta > 0]
        mean_area = np.mean(valid) if len(valid) > 0 else 0
        residual = Area_eta - mean_area
        dev_pct = (valid.max() - valid.min()) / mean_area * 100 if mean_area > 0 else 0

        ax.plot(Eta, residual, lw=1.5, color=col, alpha=0.85,
                label=f'ΔR = {dr} px  (p-p = {dev_pct:.4f}%, ⟨A⟩ = {mean_area:.1f} px²)')

    ax.axhline(0, color='grey', ls='-', lw=0.3, alpha=0.5)
    for ca in CARDINALS:
        ax.axvline(ca, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_ylabel('ΔA (px²)', fontsize=11)
    ax.set_xlabel('η (°)', fontsize=11)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    fig.tight_layout()
    save(fig, fname)


def plot_midas_vs_pyfai(midas_results, pyfai_results, fname):
    """MIDAS vs pyFAI at ΔR=0.5 — single plot, rolling σ on right axis."""
    dr = 0.5
    if dr not in midas_results:
        return

    has_pyfai = pyfai_results and dr in pyfai_results

    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    fig.suptitle(f'MIDAS vs pyFAI at ΔR = {dr} px (zero tilts, same geometry)',
                 fontsize=13, fontweight='bold')

    m = midas_results[dr]
    ax1.plot(m['Eta'], m['I_eta'], color='#1f77b4', lw=1.5, alpha=0.85,
             label='MIDAS')

    if has_pyfai:
        p = pyfai_results[dr]
        ax1.plot(p['eta'], p['I_eta'], color='#d62728', lw=1.5, alpha=0.85,
                 label='pyFAI')

    ax1.set_ylabel('Intensity (counts)', fontsize=11)
    ax1.set_xlabel('η (°)', fontsize=11)
    for ca in CARDINALS:
        ax1.axvline(ca, color='grey', ls=':', lw=0.8, alpha=0.5)

    # Rolling σ on right axis
    def rolling_std(arr, w=7):
        from numpy.lib.stride_tricks import sliding_window_view
        padded = np.pad(arr, w // 2, mode='edge')
        windows = sliding_window_view(padded, w)
        return np.std(windows, axis=1)[:len(arr)]

    ax2 = ax1.twinx()
    rstd_m = rolling_std(m['I_eta'])
    ax2.plot(m['Eta'], rstd_m, color='#1f77b4', lw=1.0, ls='--', alpha=0.6,
             label='MIDAS σ')
    if has_pyfai:
        rstd_p = rolling_std(p['I_eta'])
        ax2.plot(p['eta'], rstd_p, color='#d62728', lw=1.0, ls='--', alpha=0.6,
                 label='pyFAI σ')
    ax2.set_ylabel('Rolling σ (7-bin)', fontsize=11, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=10, framealpha=0.9)

    ax1.xaxis.set_major_locator(MultipleLocator(45))
    fig.tight_layout()
    save(fig, fname)


def plot_tiltstudy(tilt_results, fname):
    """2×2 zoom layout: full range, η=0°, η=-90°, η=±180° (broken axis)."""
    if not tilt_results:
        return

    colours_tilt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = list(tilt_results.keys())

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('BC integer/real × tilts on/off (ΔR = 0.5 px)',
                 fontsize=13, fontweight='bold')

    # Top-left: full η range
    ax_full = fig.add_subplot(2, 2, 1)
    ax_full.set_title('Full η range', fontsize=11)
    for i, label in enumerate(labels):
        d = tilt_results[label]
        ax_full.plot(d['Eta'], d['I_eta'], lw=0.7, color=colours_tilt[i],
                     label=label.replace(', ', ', '))
    for ca in CARDINALS:
        ax_full.axvline(ca, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax_full.set_xlabel('Eta (deg)')
    ax_full.set_ylabel('Intensity (a.u.)')
    ax_full.legend(fontsize=7, loc='lower left')

    # Top-right: zoom η = 0°
    ax_z0 = fig.add_subplot(2, 2, 2)
    ax_z0.set_title('Zoom: η = 0°', fontsize=11)
    for i, label in enumerate(labels):
        d = tilt_results[label]
        mask = np.abs(d['Eta']) <= 15
        ax_z0.plot(d['Eta'][mask], d['I_eta'][mask], lw=0.8,
                   color=colours_tilt[i], label=label)
    ax_z0.axvline(0, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax_z0.set_xlabel('Eta (deg)')
    ax_z0.set_ylabel('Intensity (a.u.)')
    ax_z0.legend(fontsize=7)

    # Bottom-left: zoom η = -90°
    ax_z90 = fig.add_subplot(2, 2, 3)
    ax_z90.set_title('Zoom: η = −90°', fontsize=11)
    for i, label in enumerate(labels):
        d = tilt_results[label]
        mask = (d['Eta'] >= -97.5) & (d['Eta'] <= -82.5)
        ax_z90.plot(d['Eta'][mask], d['I_eta'][mask], lw=0.8,
                    color=colours_tilt[i], label=label)
    ax_z90.axvline(-90, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax_z90.set_xlabel('Eta (deg)')
    ax_z90.set_ylabel('Intensity (a.u.)')
    ax_z90.legend(fontsize=7)

    # Bottom-right: zoom η = ±180° — broken axis (two windows)
    # Use two sub-axes side by side to show near +180° and near -180°
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    gs_br = gs[1, 1].subgridspec(1, 2, wspace=0.08)
    ax_left = fig.add_subplot(gs_br[0, 0])
    ax_right = fig.add_subplot(gs_br[0, 1])

    ax_left.set_title('Zoom: η = ±180°', fontsize=11, loc='center')
    hw = 8  # half-width of window in degrees

    for i, label in enumerate(labels):
        d = tilt_results[label]
        # Near -180°
        mask_neg = (d['Eta'] >= -180 - hw) & (d['Eta'] <= -180 + hw)
        ax_left.plot(d['Eta'][mask_neg], d['I_eta'][mask_neg], lw=0.8,
                     color=colours_tilt[i])
        # Near +180°
        mask_pos = (d['Eta'] >= 180 - hw) & (d['Eta'] <= 180 + hw)
        ax_right.plot(d['Eta'][mask_pos], d['I_eta'][mask_pos], lw=0.8,
                      color=colours_tilt[i])

    ax_left.axvline(-180, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax_right.axvline(180, color='grey', ls=':', lw=0.5, alpha=0.5)

    # Broken-axis styling
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(left=False, labelleft=False)
    ax_left.set_ylabel('Intensity (a.u.)')

    # Diagonal break marks
    d_len = 0.02
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, lw=1)
    ax_left.plot((1 - d_len, 1 + d_len), (-d_len, +d_len), **kwargs)
    ax_left.plot((1 - d_len, 1 + d_len), (1 - d_len, 1 + d_len), **kwargs)
    kwargs['transform'] = ax_right.transAxes
    ax_right.plot((-d_len, +d_len), (-d_len, +d_len), **kwargs)
    ax_right.plot((-d_len, +d_len), (1 - d_len, 1 + d_len), **kwargs)

    fig.tight_layout()
    save(fig, fname)


def plot_tx_rotation(tx_results, fname):
    """2×2 panel tx rotation."""
    if not tx_results:
        return
    tx_values = sorted(tx_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Detector rotation experiment: aliasing shifts with pixel grid',
                 fontsize=13, fontweight='bold')
    for ax_idx, tx in enumerate(tx_values):
        ax = axes.flat[ax_idx]
        d = tx_results[tx]
        ax.plot(d['Eta'], d['I_eta'], 'k-', lw=0.5, alpha=0.85)
        # Predicted aliasing angles
        pred = sorted([(a + tx) % 360 for a in [0, 90, 180, 270]])
        pred = [a - 360 if a > 180 else a for a in pred]
        for ang in pred:
            ax.axvline(ang, color='blue', ls='--', lw=1.5, alpha=0.7)
        for ang in [0, 90, -90, 180, -180]:
            ax.axvline(ang, color='red', ls=':', lw=0.5, alpha=0.3)
        ax.set_xlabel('η (°)')
        ax.set_ylabel('Intensity')
        pred_str = ', '.join([f'{a:.0f}' for a in pred])
        ax.set_title(f'tx={tx}°: predicted at [{pred_str}]')
        ax.grid(True, alpha=0.15)
        ax.set_xlim(-185, 185)
    fig.tight_layout()
    save(fig, fname)


def print_summary(midas_results, pyfai_results):
    print("\n" + "=" * 70)
    print("SUMMARY: Oscillation amplitude at η ≈ 0°")
    print("=" * 70)
    hdr = f"{'ΔR (px)':<10}{'MIDAS σ/μ':<14}{'MIDAS pp/μ':<14}"
    if pyfai_results:
        hdr += f"{'pyFAI σ/μ':<14}{'pyFAI pp/μ':<14}"
    print(hdr)
    print("-" * len(hdr))

    for dr in sorted(midas_results.keys()):
        m = midas_results[dr]
        mask0 = np.abs(m['Eta']) < 5
        vals = m['I_eta'][mask0]
        mu = np.mean(vals)
        m_std = np.std(vals) / mu if mu > 0 else 0
        m_pp  = (vals.max() - vals.min()) / mu if mu > 0 else 0
        row = f"{dr:<10.2f}{m_std:<14.4f}{m_pp:<14.4f}"

        if pyfai_results and dr in pyfai_results:
            p = pyfai_results[dr]
            mask0p = np.abs(p['eta']) < 5
            pv = p['I_eta'][mask0p]
            pmu = np.mean(pv) if len(pv) > 0 else 0
            p_std = np.std(pv) / pmu if pmu > 0 else 0
            p_pp  = (pv.max() - pv.min()) / pmu if pmu > 0 else 0
            row += f"{p_std:<14.6f}{p_pp:<14.6f}"

        print(row)
    print()

# ═══════════════════════════════════════════════════════════════════
#  Experiment 6: Gradient Correction A/B test
# ═══════════════════════════════════════════════════════════════════

def experiment_gradient_correction():
    """Run ΔR=0.5 with GradientCorrection=0 vs GradientCorrection=1, analyze multiple rings."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Gradient Correction A/B Test (ΔR=0.5)")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    results = {}
    for gc_val, label in [(0, 'no_grad'), (1, 'grad')]:
        print(f"\n  GradientCorrection={gc_val} ({label}):")
        work_dir = os.path.join(WORK_BASE, f'gradient_{label}')
        # Force fresh Map.bin (v3 format with deltaR)
        map_bin = os.path.join(work_dir, 'Map.bin')
        if os.path.isfile(map_bin):
            os.remove(map_bin)
        nmap_bin = os.path.join(work_dir, 'nMap.bin')
        if os.path.isfile(nmap_bin):
            os.remove(nmap_bin)
        caked = run_midas_case(work_dir, base_lines, {
            'RBinSize': '0.5',
            'SubPixelLevel': '1',
            'GradientCorrection': str(gc_val),
            'EtaBinSize': str(ETA_BIN_SIZE),
        })
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            results[label] = {'R': R, 'Eta': Eta, 'I': I, 'Area': Area}

            # Find bright rings: take radial mean profile, find peaks
            radial_mean = np.nanmean(I, axis=1)
            # Simple peak detection: local maxima above median + 2*std
            threshold = np.nanmedian(radial_mean) + 2 * np.nanstd(radial_mean)
            ring_indices = []
            for i in range(2, len(radial_mean) - 2):
                if (radial_mean[i] > threshold and
                    radial_mean[i] > radial_mean[i-1] and
                    radial_mean[i] > radial_mean[i+1] and
                    radial_mean[i] > radial_mean[i-2] and
                    radial_mean[i] > radial_mean[i+2]):
                    ring_indices.append(i)
            # Merge nearby peaks (within 5 bins)
            merged = []
            for idx in ring_indices:
                if not merged or idx - merged[-1] > 10:
                    merged.append(idx)
                elif radial_mean[idx] > radial_mean[merged[-1]]:
                    merged[-1] = idx
            results[label]['ring_indices'] = merged
            results[label]['ring_R'] = [R[i] for i in merged]
            print(f"    Found {len(merged)} rings: R = {[f'{R[i]:.0f}' for i in merged]}")

            # Print overall stats for the primary ring
            ridx = np.argmin(np.abs(R - R_RING))
            mu = np.nanmean(I[ridx, :])
            sigma = np.nanstd(I[ridx, :])
            print(f"    Primary ring R={R[ridx]:.0f}: σ/μ = {sigma/mu*100:.3f}%  (μ={mu:.1f}, σ={sigma:.3f})")
        else:
            print(f"    FAILED")

    # Multi-ring cardinal-angle comparison
    if 'no_grad' in results and 'grad' in results:
        ng = results['no_grad']
        gc = results['grad']
        # Use same hardcoded radii as the plot
        ring_R_values = [376, 434, 955]

        print(f"\n  Multi-ring cardinal-angle comparison (±5° windows):")
        print(f"  {'Ring R':>8s}  {'Angle':>6s}  {'Std pp':>8s}  {'Corr pp':>8s}  {'pp red':>8s}  {'Std σ/μ':>9s}  {'Corr σ/μ':>9s}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")

        results['ring_metrics'] = []
        for ring_R in ring_R_values:
            ng_ridx = np.argmin(np.abs(ng['R'] - ring_R))
            gc_ridx = np.argmin(np.abs(gc['R'] - ring_R))
            ng_I_eta = ng['I'][ng_ridx, :]
            gc_I_eta = gc['I'][gc_ridx, :]

            # Skip if ring is too faint (mostly NaN or near zero)
            if np.nanmean(ng_I_eta) < 50:
                continue

            for angle in [0, 90, -90]:
                mask = np.abs(ng['Eta'] - angle) <= 5.0
                if np.sum(mask) < 3:
                    continue
                ng_vals = ng_I_eta[mask]
                gc_vals = gc_I_eta[mask]
                ng_pp = np.nanmax(ng_vals) - np.nanmin(ng_vals)
                gc_pp = np.nanmax(gc_vals) - np.nanmin(gc_vals)
                ng_cv = np.nanstd(ng_vals) / np.nanmean(ng_vals) * 100
                gc_cv = np.nanstd(gc_vals) / np.nanmean(gc_vals) * 100
                reduction = (1 - gc_pp / ng_pp) * 100 if ng_pp > 0 else 0
                print(f"  {ring_R:>7.0f}  {angle:>5d}°  {ng_pp:>8.1f}  {gc_pp:>8.1f}  {reduction:>7.1f}%  {ng_cv:>8.2f}%  {gc_cv:>8.2f}%")
                results['ring_metrics'].append({
                    'R': ring_R, 'angle': angle,
                    'ng_pp': ng_pp, 'gc_pp': gc_pp,
                    'ng_cv': ng_cv, 'gc_cv': gc_cv,
                    'reduction': reduction
                })

        # Store I_eta for primary ring for backward compatibility
        ridx_ng = np.argmin(np.abs(ng['R'] - R_RING))
        ridx_gc = np.argmin(np.abs(gc['R'] - R_RING))
        results['no_grad']['I_eta'] = ng['I'][ridx_ng, :]
        results['no_grad']['R_actual'] = ng['R'][ridx_ng]
        results['grad']['I_eta'] = gc['I'][ridx_gc, :]
        results['grad']['R_actual'] = gc['R'][ridx_gc]

    return results


def plot_gradient_comparison(grad_results, fname):
    """Compare I(η) with/without gradient correction across multiple rings.
    Uses 3 panels (one per ring) with secondary Y-axis for ΔI."""
    if 'no_grad' not in grad_results or 'grad' not in grad_results:
        print("  Skipping gradient comparison plot (missing data)")
        return

    ng = grad_results['no_grad']
    gc = grad_results['grad']

    # Hardcoded CeO₂ ring radii to analyze
    plot_rings = [376, 434, 955]

    n_rings = len(plot_rings)
    fig, axes = plt.subplots(n_rings, 1, figsize=(14, 3.5 * n_rings), sharex=True)
    if n_rings == 1:
        axes = [axes]
    fig.suptitle('Radial Resampling Correction A/B Test (ΔR = 0.5 px)',
                 fontsize=14, fontweight='bold')

    for i, ring_R in enumerate(plot_rings):
        ng_ridx = np.argmin(np.abs(ng['R'] - ring_R))
        gc_ridx = np.argmin(np.abs(gc['R'] - ring_R))
        ng_I = ng['I'][ng_ridx, :]
        gc_I = gc['I'][gc_ridx, :]

        ax1 = axes[i]
        ng_cv = np.nanstd(ng_I) / np.nanmean(ng_I) * 100
        gc_cv = np.nanstd(gc_I) / np.nanmean(gc_I) * 100
        ax1.plot(ng['Eta'], ng_I, '-', lw=1.5, alpha=0.85, color='#457b9d',
                 label=f'Standard (σ/μ = {ng_cv:.2f}%)')
        ax1.plot(gc['Eta'], gc_I, '-', lw=1.5, alpha=0.85, color='#e63946',
                 label=f'Corrected (σ/μ = {gc_cv:.2f}%)')
        ax1.set_ylabel('Intensity (counts)', fontsize=10)
        ax1.set_title(f'R ≈ {ring_R:.0f} px', fontsize=11, loc='left')
        ax1.legend(fontsize=9, loc='upper right')

        # Secondary Y-axis for ΔI
        ax2 = ax1.twinx()
        diff = gc_I - ng_I
        ax2.plot(gc['Eta'], diff, '-', lw=1.2, color='#2a9d8f', alpha=0.7,
                 label='ΔI')
        ax2.axhline(0, color='grey', ls='--', lw=0.5, alpha=0.5)
        ax2.set_ylabel('ΔI (counts)', fontsize=10, color='#2a9d8f')
        ax2.tick_params(axis='y', labelcolor='#2a9d8f')
        ax2.legend(fontsize=8, loc='lower right')

        for angle in [0, 90, -90]:
            ax1.axvline(angle, color='red', alpha=0.3, ls=':', lw=0.8)

    axes[-1].set_xlabel('η (degrees)', fontsize=11)
    fig.tight_layout()
    save(fig, fname)


def plot_correction_cake(grad_results, fname):
    """3-panel cake: (a) standard, (b) corrected, (c) difference.
    Shows where the resampling correction acts in (R, η) space."""
    if 'no_grad' not in grad_results or 'grad' not in grad_results:
        print("  Skipping correction cake plot (missing data)")
        return

    ng = grad_results['no_grad']
    gc = grad_results['grad']

    r_half = 25
    rmask = (ng['R'] >= R_RING - r_half) & (ng['R'] <= R_RING + r_half)
    r_idx = np.where(rmask)[0]
    Rsub = ng['R'][r_idx]
    Eta = ng['Eta']

    I_std = ng['I'][r_idx, :].T.astype(float)   # (nEta, nR_sub)
    I_cor = gc['I'][r_idx, :].T.astype(float)

    I_std[I_std == 0] = np.nan
    I_cor[I_cor == 0] = np.nan
    diff = I_cor - I_std

    from matplotlib.colors import LogNorm as _LN3
    vmin_log = max(1, np.nanpercentile(I_std[np.isfinite(I_std)], 1))
    vmax = np.nanpercentile(I_std[np.isfinite(I_std)], 99.5)
    dlim = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 97)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    fig.suptitle(f'Resampling correction effect on 2D cake (ΔR = 0.5 px, R ≈ {R_RING} px)',
                 fontsize=13, fontweight='bold')

    # (a) Standard
    im0 = axes[0].pcolormesh(Rsub, Eta, I_std, shading='auto', cmap='viridis',
                              norm=_LN3(vmin=vmin_log, vmax=vmax), rasterized=True)
    axes[0].set_title('(a) Standard integration', fontsize=11, loc='left')
    axes[0].set_ylabel('η (°)', fontsize=11)
    fig.colorbar(im0, ax=axes[0], label='Intensity (counts)', pad=0.02, shrink=0.9)

    # (b) Corrected
    im1 = axes[1].pcolormesh(Rsub, Eta, I_cor, shading='auto', cmap='viridis',
                              norm=_LN3(vmin=vmin_log, vmax=vmax), rasterized=True)
    axes[1].set_title('(b) With resampling correction', fontsize=11, loc='left')
    axes[1].set_ylabel('η (°)', fontsize=11)
    fig.colorbar(im1, ax=axes[1], label='Intensity (counts)', pad=0.02, shrink=0.9)

    # (c) Difference
    im2 = axes[2].pcolormesh(Rsub, Eta, diff, shading='auto', cmap='RdBu_r',
                              vmin=-dlim, vmax=dlim, rasterized=True)
    axes[2].set_title('(c) Difference (corrected − standard)', fontsize=11, loc='left')
    axes[2].set_ylabel('η (°)', fontsize=11)
    axes[2].set_xlabel('R (pixels)', fontsize=11)
    fig.colorbar(im2, ax=axes[2], label='ΔI (counts)', pad=0.02, shrink=0.9)

    for ax in axes:
        cardinal_boxes(ax, (Eta.min(), Eta.max()))

    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 7: Sub-pixel peak fitting — R_center(η) wobble
# ═══════════════════════════════════════════════════════════════════

def pseudo_voigt(R, amplitude, center, sigma, eta_pv, bg):
    """Pseudo-Voigt profile: eta_pv * Lorentzian + (1-eta_pv) * Gaussian."""
    dR = R - center
    G = np.exp(-0.5 * (dR / sigma) ** 2)
    L = 1.0 / (1.0 + (dR / sigma) ** 2)
    return bg + amplitude * (eta_pv * L + (1 - eta_pv) * G)


def experiment_peak_fitting(grad_results):
    """Fit pseudo-Voigt to each η-slice of the (220) ring, extract R_center(η)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Sub-pixel peak fitting R_center(η)")
    print("=" * 60)

    if not grad_results or 'no_grad' not in grad_results or 'grad' not in grad_results:
        print("  Skipping — gradient correction data not available")
        return None

    from scipy.optimize import curve_fit

    ng = grad_results['no_grad']
    gc = grad_results['grad']
    R = ng['R']
    Eta = ng['Eta']

    # Focus on the (220) ring — find peak in radial profile
    ring_R = R_RING
    r_window = 15  # px half-width for fitting window
    rmask = (R >= ring_R - r_window) & (R <= ring_R + r_window)
    r_idx = np.where(rmask)[0]
    R_fit = R[r_idx]

    results = {}
    for label, data in [('Standard', ng), ('Corrected', gc)]:
        print(f"\n  Fitting {label}...")
        centers = []
        fwhms = []
        etas_good = []
        for ei in range(len(Eta)):
            I_slice = data['I'][r_idx, ei]
            if np.all(I_slice == 0) or np.any(np.isnan(I_slice)):
                continue
            # Initial guesses
            bg0 = np.min(I_slice)
            amp0 = np.max(I_slice) - bg0
            cen0 = R_fit[np.argmax(I_slice)]
            sig0 = 2.0
            try:
                popt, _ = curve_fit(pseudo_voigt, R_fit, I_slice,
                                   p0=[amp0, cen0, sig0, 0.5, bg0],
                                   bounds=([0, ring_R - r_window, 0.1, 0, -np.inf],
                                           [np.inf, ring_R + r_window, 20, 1, np.inf]),
                                   maxfev=2000)
                centers.append(popt[1])
                fwhm = 2.355 * popt[2]  # approximate FWHM
                fwhms.append(fwhm)
                etas_good.append(Eta[ei])
            except (RuntimeError, ValueError):
                continue

        centers = np.array(centers)
        fwhms = np.array(fwhms)
        etas_good = np.array(etas_good)
        results[label] = {
            'eta': etas_good, 'R_center': centers, 'FWHM': fwhms
        }
        # Print stats
        print(f"    Fitted {len(centers)}/{len(Eta)} slices")
        print(f"    R_center: mean={np.mean(centers):.4f}, "
              f"std={np.std(centers):.4f}, "
              f"peak-to-peak={np.ptp(centers):.4f}")
        # Cardinal-angle wobble
        for angle in [0, 90, -90]:
            mask = np.abs(etas_good - angle) <= 5
            if np.sum(mask) >= 3:
                local_std = np.std(centers[mask])
                local_pp = np.ptp(centers[mask])
                print(f"    η≈{angle:+d}°: local std={local_std:.4f}, pp={local_pp:.4f}")

    # Compute artificial strain metrics
    if 'Standard' in results and 'Corrected' in results:
        for label in ['Standard', 'Corrected']:
            R_c = results[label]['R_center']
            strain_std = np.std(R_c) / np.mean(R_c) * 1e6  # microstrain
            print(f"\n  {label} artificial strain from R_center scatter: "
                  f"{strain_std:.1f} µε")

    return results


def plot_peak_fitting(pf_results, fname):
    """Plot R_center(η): raw + detrended residuals + cardinal-angle zooms."""
    if not pf_results or 'Standard' not in pf_results:
        print("  Skipping peak fitting plot (missing data)")
        return

    from scipy.ndimage import median_filter

    colors = {'Standard': '#457b9d', 'Corrected': '#e63946'}

    # ── Detrend: subtract a wide median filter to remove smooth geometry ──
    detrended = {}
    for label in ['Standard', 'Corrected']:
        if label not in pf_results:
            continue
        d = pf_results[label]
        # Median filter with ~30° window to remove smooth sinusoidal trend
        kernel = max(31, len(d['R_center']) // 12) | 1  # odd integer
        trend = median_filter(d['R_center'], size=kernel)
        residual = d['R_center'] - trend
        detrended[label] = {
            'eta': d['eta'], 'R_center': d['R_center'],
            'trend': trend, 'residual': residual,
            'FWHM': d['FWHM'],
        }

    # ── Layout: 2 rows top (full-width) + 3 columns bottom (zooms) ──
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.3,
                          height_ratios=[1, 1, 1.2])
    ax_raw = fig.add_subplot(gs[0, :])
    ax_res = fig.add_subplot(gs[1, :])
    ax_z0  = fig.add_subplot(gs[2, 0])
    ax_z90 = fig.add_subplot(gs[2, 1])
    ax_zm90 = fig.add_subplot(gs[2, 2])

    fig.suptitle('Sub-pixel Peak Fitting: CeO₂ (220) — Detrended R$_{center}$(η)',
                 fontsize=14, fontweight='bold')

    # ── Panel 1: Raw R_center(η) with smooth trend ──
    for label, col in colors.items():
        if label in detrended:
            d = detrended[label]
            ax_raw.plot(d['eta'], d['R_center'], '.', ms=5, color=col,
                        alpha=0.5, label=label)
            ax_raw.plot(d['eta'], d['trend'], '-', lw=1.5, color=col,
                        alpha=0.8)
    ax_raw.set_ylabel('R$_{center}$ (px)', fontsize=11)
    ax_raw.legend(fontsize=9, loc='upper right')
    ax_raw.set_title('(a) Fitted peak center with smooth geometry trend',
                     fontsize=11, loc='left')
    for angle in [0, 90, -90]:
        ax_raw.axvline(angle, color='grey', ls=':', lw=0.5, alpha=0.5)

    # ── Panel 2: Detrended residuals (full range) ──
    for label, col in colors.items():
        if label in detrended:
            d = detrended[label]
            res_std = np.std(d['residual']) * 1000  # milli-pixels
            ax_res.plot(d['eta'], d['residual'] * 1000, '.', ms=5,
                        color=col, alpha=0.6,
                        label=f'{label} (σ = {res_std:.1f} mpx)')
    ax_res.axhline(0, color='grey', ls='--', lw=0.5)
    ax_res.set_ylabel('ΔR$_{center}$ residual (mpx)', fontsize=11)
    ax_res.set_xlabel('η (degrees)', fontsize=11)
    ax_res.legend(fontsize=9, loc='upper right')
    ax_res.set_title('(b) Detrended residuals — high-frequency wobble at cardinal angles',
                     fontsize=11, loc='left')
    for angle in [0, 90, -90]:
        ax_res.axvline(angle, color='red', ls='--', lw=1.0, alpha=0.4)

    # ── Bottom panels: Zoom into ±10° around each cardinal angle ──
    zoom_hw = 10  # half-width in degrees
    zoom_axes = [(ax_z0, 0, 'η = 0°'), (ax_z90, 90, 'η = +90°'),
                 (ax_zm90, -90, 'η = −90°')]

    for ax, angle, title in zoom_axes:
        for label, col in colors.items():
            if label in detrended:
                d = detrended[label]
                mask = np.abs(d['eta'] - angle) <= zoom_hw
                if np.sum(mask) < 3:
                    continue
                eta_z = d['eta'][mask]
                res_z = d['residual'][mask] * 1000  # milli-pixels
                ax.plot(eta_z, res_z, 'o-', ms=6, lw=1.0, color=col,
                        alpha=0.7, label=label)
                # Local stats
                local_std = np.std(d['residual'][mask]) * 1000
                local_pp = np.ptp(d['residual'][mask]) * 1000
                ax.annotate(f'{label}: σ={local_std:.1f}, pp={local_pp:.1f} mpx',
                           xy=(0.03, 0.95 - 0.12 * list(colors.keys()).index(label)),
                           xycoords='axes fraction', fontsize=7.5,
                           color=col, va='top',
                           bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                     ec=col, alpha=0.7))

        ax.axvline(angle, color='red', ls='--', lw=1.0, alpha=0.4)
        ax.axhline(0, color='grey', ls='--', lw=0.5)
        ax.set_title(f'({"cde"[zoom_axes.index((ax, angle, title))]}) Zoom: {title}',
                     fontsize=11, loc='left')
        ax.set_xlabel('η (degrees)', fontsize=10)
        ax.set_ylabel('Residual (mpx)', fontsize=10)

    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 8: Throughput benchmark (Phase 2 timing)
# ═══════════════════════════════════════════════════════════════════

def experiment_benchmark(n_repeats=5):
    """Time IntegratorZarrOMP Phase 2 with and without GradientCorrection."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 8: Throughput Benchmark ({n_repeats} repeats)")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    results = {}

    for gc_val, label in [(0, 'Standard'), (1, 'Corrected')]:
        print(f"\n  {label} (GradientCorrection={gc_val}):")
        work_dir = os.path.join(WORK_BASE, f'bench_{label.lower()}')
        os.makedirs(work_dir, exist_ok=True)

        # Symlink data files
        for fn in os.listdir(DATA_DIR):
            if fn.endswith('.tif'):
                dst = os.path.join(work_dir, fn)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(DATA_DIR, fn), dst)

        # Write param file
        all_overrides = {
            'Folder': os.path.abspath(work_dir),
            'RBinSize': '0.5',
            'SubPixelLevel': '1',
            'GradientCorrection': str(gc_val),
            'EtaBinSize': str(ETA_BIN_SIZE),
        }
        pf = os.path.join(work_dir, 'ps_test.txt')
        write_param_file(base_lines, pf, all_overrides)

        # Run DetectorMapper once (reuse Map.bin)
        mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
        integrator = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')
        map_bin = os.path.join(work_dir, 'Map.bin')
        if not os.path.isfile(map_bin):
            print(f"    Building map...", end='', flush=True)
            subprocess.run([mapper, 'ps_test.txt'],
                          capture_output=True, text=True, timeout=300,
                          cwd=work_dir)
            print(" done")

        data_path = os.path.join(work_dir, DATA_FILE)
        times = []
        for rep in range(n_repeats):
            # Remove previous output to force re-integration
            caked = os.path.join(work_dir, DATA_FILE + '.caked.hdf')
            lineout = os.path.join(work_dir,
                                   DATA_FILE.replace('.tif', '_lineout.bin'))
            for f in [caked, lineout]:
                if os.path.isfile(f):
                    os.remove(f)

            r = subprocess.run([integrator, '-paramFN', 'ps_test.txt',
                               '-dataFN', data_path, '-nCPUs', '8'],
                              capture_output=True, text=True, timeout=60,
                              cwd=work_dir)
            # Parse "Time for integration: X seconds." from stdout
            dt = None
            for line in r.stdout.splitlines():
                if 'Time for integration' in line:
                    try:
                        dt = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
            if dt is None:
                # Fallback: wall-clock
                dt = 0.0
            times.append(dt)
            print(f"    Rep {rep+1}/{n_repeats}: {dt*1000:.1f} ms")

        median_ms = np.median(times) * 1000
        results[label] = {
            'times': times,
            'median_ms': median_ms,
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
        }
        print(f"    Median: {median_ms:.1f} ms/frame")

    if 'Standard' in results and 'Corrected' in results:
        s = results['Standard']['median_ms']
        c = results['Corrected']['median_ms']
        overhead = (c - s) / s * 100
        print(f"\n  Overhead: {overhead:+.1f}% "
              f"({s:.1f} → {c:.1f} ms/frame)")
        results['overhead_pct'] = overhead

    return results


# ═══════════════════════════════════════════════════════════════════
#  Experiment 9: Sub-pixel splitting test
# ═══════════════════════════════════════════════════════════════════

def experiment_subpixel_splitting():
    """Run ΔR=0.5 with SubPixelLevel=1,2,4 to show artifact persists."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: Sub-pixel splitting test (ΔR=0.5)")
    print("=" * 60)

    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    results = {}
    target_R = 955  # (220) ring

    for spl in [1, 2, 4]:
        label = f'spl{spl}'
        print(f"\n  SubPixelLevel={spl}:")
        work_dir = os.path.join(WORK_BASE, f'subpixel_{label}')
        # Force fresh Map.bin
        for fname in ['Map.bin', 'nMap.bin']:
            f = os.path.join(work_dir, fname)
            if os.path.isfile(f):
                os.remove(f)
        caked = run_midas_case(work_dir, base_lines, {
            'RBinSize': '0.5',
            'SubPixelLevel': str(spl),
            'GradientCorrection': '0',
            'EtaBinSize': str(ETA_BIN_SIZE),
        })
        if caked:
            R, Eta, I, Area = read_caked_hdf(caked)
            # Find closest R to target
            r_idx = np.argmin(np.abs(R - target_R))
            ring_I = I[r_idx, :]
            ring_R = R[r_idx]

            # Stats within ±5° of cardinal angles
            stats = {}
            for angle in [0, 90, -90]:
                mask = np.abs(Eta - angle) <= 5
                if np.sum(mask) < 3:
                    continue
                local = ring_I[mask]
                valid = local[np.isfinite(local)]
                if len(valid) > 0:
                    stats[angle] = {
                        'sigma_mu': np.std(valid) / np.mean(valid) * 100,
                        'pp': np.ptp(valid),
                        'mean': np.mean(valid),
                    }

            results[label] = {
                'spl': spl, 'R': ring_R, 'Eta': Eta, 'I_ring': ring_I,
                'stats': stats,
            }
            print(f"    Ring R={ring_R:.0f}: σ/μ = {np.std(ring_I[np.isfinite(ring_I)])/np.mean(ring_I[np.isfinite(ring_I)])*100:.3f}%")
            for angle, s in stats.items():
                print(f"    η≈{angle:+d}°: σ/μ={s['sigma_mu']:.2f}%, pp={s['pp']:.1f}")

    # Summary table
    if results:
        print(f"\n  Summary (±5° of η=0°):")
        print(f"  {'SPL':>5s}  {'σ/μ (%)':>10s}  {'Peak-to-peak':>12s}")
        print(f"  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*12:>12s}")
        for label in sorted(results.keys()):
            r = results[label]
            s = r['stats'].get(0, {})
            if s:
                print(f"  {r['spl']:5d}  {s['sigma_mu']:10.2f}  {s['pp']:12.1f}")

    return results


def plot_subpixel_splitting(spl_results, fname):
    """Plot I(η) for SubPixelLevel=1,2,4 to show artifact persists."""
    if not spl_results:
        print("  Skipping sub-pixel plot (missing data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('Sub-pixel Splitting Does Not Remove Cardinal-Angle Aliasing',
                 fontsize=14, fontweight='bold')

    colors = {'spl1': '#457b9d', 'spl2': '#e76f51', 'spl4': '#2a9d8f'}
    labels = {'spl1': 'SPL = 1', 'spl2': 'SPL = 2', 'spl4': 'SPL = 4'}

    # Cardinal angle windows
    windows = [(0, 'η = 0°'), (90, 'η = +90°'), (-90, 'η = −90°')]

    for ax, (angle, title) in zip(axes, windows):
        for key in ['spl1', 'spl2', 'spl4']:
            if key not in spl_results:
                continue
            r = spl_results[key]
            mask = np.abs(r['Eta'] - angle) <= 15
            if np.sum(mask) < 3:
                continue
            eta_w = r['Eta'][mask]
            I_w = r['I_ring'][mask]
            s = r['stats'].get(angle, {})
            lbl = labels[key]
            if s:
                lbl += f' (σ/μ={s["sigma_mu"]:.1f}%)'
            ax.plot(eta_w, I_w, '-', lw=1.2, color=colors[key],
                    alpha=0.8, label=lbl)

        ax.axvline(angle, color='red', ls='--', lw=1.0, alpha=0.4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('η (degrees)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        if ax == axes[0]:
            ax.set_ylabel('Intensity (counts)', fontsize=11)

    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 10: Bicubic vs bilinear interpolation
# ═══════════════════════════════════════════════════════════════════

MAP_HEADER_MAGIC = 0x3050414D
MAP_HEADER_SIZE = 64
# struct data { float y, z; double frac; float deltaR, _reserved; } = 24 bytes
MAP_ENTRY_DTYPE = np.dtype([
    ('y', np.float32), ('z', np.float32),
    ('frac', np.float64),
    ('deltaR', np.float32), ('_reserved', np.float32),
])


def read_map_bin(work_dir):
    """Read Map.bin and nMap.bin, return pixel list and bin index."""
    map_path = os.path.join(work_dir, 'Map.bin')
    nmap_path = os.path.join(work_dir, 'nMap.bin')

    # Read Map.bin
    with open(map_path, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        f.seek(0)
        if magic == MAP_HEADER_MAGIC:
            f.seek(MAP_HEADER_SIZE)
        else:
            f.seek(0)
        px_data = np.frombuffer(f.read(), dtype=MAP_ENTRY_DTYPE)

    # Read nMap.bin
    with open(nmap_path, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        f.seek(0)
        if magic == MAP_HEADER_MAGIC:
            f.seek(MAP_HEADER_SIZE)
        else:
            f.seek(0)
        n_data = np.frombuffer(f.read(), dtype=np.int32)

    # n_data is pairs: [count, offset, count, offset, ...]
    n_bins = len(n_data) // 2
    bin_counts = n_data[0::2]
    bin_offsets = n_data[1::2]
    return px_data, bin_counts, bin_offsets, n_bins


def read_raw_image(work_dir, dark_sub=True):
    """Read the raw TIFF image and subtract dark."""
    from PIL import Image as PILImage
    # Find the data file
    data_path = os.path.join(work_dir, DATA_FILE)
    img = PILImage.open(data_path)
    image = np.array(img, dtype=np.float64)

    # Find and subtract dark
    dark_files = [f for f in os.listdir(work_dir)
                  if f.startswith('dark') and f.endswith('.tif')]
    if dark_sub and dark_files:
        dark_img = PILImage.open(os.path.join(work_dir, dark_files[0]))
        image -= np.array(dark_img, dtype=np.float64)

    return image


def experiment_bicubic_comparison():
    """Compare bilinear vs bicubic interpolation to disentangle
    interpolation error from Nyquist limit in the residual aliasing."""
    from scipy.ndimage import map_coordinates

    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Bicubic vs Bilinear Interpolation")
    print("=" * 60)

    # First, run MIDAS normally to get the map at ΔR=0.5 with GradientCorrection=1
    base_lines = read_param_lines(os.path.join(DATA_DIR, PARAM_FILE))
    work_dir = os.path.join(WORK_BASE, 'gradient_grad')

    # Make sure Map.bin exists (from the gradient correction experiment)
    map_bin = os.path.join(work_dir, 'Map.bin')
    if not os.path.isfile(map_bin):
        print("  Need gradient_grad run first. Running...")
        for fname in ['Map.bin', 'nMap.bin']:
            f = os.path.join(work_dir, fname)
            if os.path.isfile(f):
                os.remove(f)
        run_midas_case(work_dir, base_lines, {
            'RBinSize': '0.5',
            'SubPixelLevel': '1',
            'GradientCorrection': '1',
        })

    # Read the map and image
    print("  Reading Map.bin and nMap.bin...")
    px_data, bin_counts, bin_offsets, n_bins = read_map_bin(work_dir)
    print(f"    {len(px_data)} map entries across {n_bins} bins")

    print("  Reading raw image...")
    image = read_raw_image(work_dir)
    NrPixelsY, NrPixelsZ = image.shape[1], image.shape[0]
    print(f"    Image shape: {image.shape}")

    # Read the caked output to get R and Eta axes
    caked_path = os.path.join(work_dir, DATA_FILE + '.caked.hdf')
    with h5py.File(caked_path, 'r') as hf:
        remap = hf['REtaMap'][:]
        R_axis = remap[0, :, 0]
        Eta_axis = remap[2, 0, :]
        nRBins = len(R_axis)
        nEtaBins = len(Eta_axis)

    # Parse beam center from params
    with open(os.path.join(work_dir, 'ps_test.txt')) as f:
        BC_y, BC_z = 0, 0
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                if parts[0] == 'BC':
                    vals = parts[1].split(',') if ',' in parts[1] else parts[1:]
                    if len(vals) >= 2:
                        BC_y, BC_z = float(vals[0]), float(vals[1])
                    elif len(parts) >= 3:
                        BC_y, BC_z = float(parts[1]), float(parts[2])

    print(f"    BC = ({BC_y}, {BC_z}), grid = {nRBins}×{nEtaBins}")

    # Python re-integration with different interpolation orders
    # For each bin, gather the shifted read coordinates and fracs
    target_R = R_RING
    r_idx = np.argmin(np.abs(R_axis - target_R))
    print(f"\n  Re-integrating ring R={R_axis[r_idx]:.1f} ({nEtaBins} eta bins)")

    results = {}
    for order, label in [(1, 'bilinear'), (3, 'bicubic')]:
        for gc_mode, gc_label in [(False, 'standard'), (True, 'corrected')]:
            key = f'{label}_{gc_label}'
            I_ring = np.zeros(nEtaBins)
            for ei in range(nEtaBins):
                pos = r_idx * nEtaBins + ei
                if pos >= n_bins:
                    continue
                n_px = bin_counts[pos]
                offset = bin_offsets[pos]
                if n_px == 0:
                    continue

                entries = px_data[offset:offset + n_px]
                read_y = entries['y'].astype(np.float64)
                read_z = entries['z'].astype(np.float64)
                fracs = entries['frac']
                deltaR = entries['deltaR'].astype(np.float64)

                if gc_mode:
                    dy = read_y - BC_y
                    dz = read_z - BC_z
                    R_val = np.sqrt(dy**2 + dz**2)
                    valid = R_val > 1.0
                    read_y = read_y.copy()
                    read_z = read_z.copy()
                    read_y[valid] -= deltaR[valid] * dy[valid] / R_val[valid]
                    read_z[valid] -= deltaR[valid] * dz[valid] / R_val[valid]

                # Clip to image bounds
                read_y = np.clip(read_y, 0, NrPixelsY - 1.001)
                read_z = np.clip(read_z, 0, NrPixelsZ - 1.001)

                # scipy.ndimage.map_coordinates expects (row, col) = (z, y)
                coords = np.array([read_z, read_y])
                vals = map_coordinates(image, coords, order=order, mode='nearest')

                I_ring[ei] = np.sum(vals * fracs)

            results[key] = I_ring

    # Print comparison
    print(f"\n  Cardinal-angle statistics (±5° of η=0°):")
    print(f"  {'Method':<25s}  {'σ/μ (%)':>10s}  {'Peak-to-peak':>12s}")
    print(f"  {'-'*25:<25s}  {'-'*10:>10s}  {'-'*12:>12s}")
    for key in ['bilinear_standard', 'bilinear_corrected',
                'bicubic_standard', 'bicubic_corrected']:
        I = results[key]
        mask = np.abs(Eta_axis) <= 5
        valid = I[mask]
        valid = valid[valid > 0]
        if len(valid) > 0:
            sm = np.std(valid) / np.mean(valid) * 100
            pp = np.ptp(valid)
            print(f"  {key:<25s}  {sm:10.2f}  {pp:12.1f}")

    # Full-ring stats
    print(f"\n  Full-ring statistics:")
    for key in ['bilinear_standard', 'bilinear_corrected',
                'bicubic_standard', 'bicubic_corrected']:
        I = results[key]
        valid = I[I > 0]
        sm = np.std(valid) / np.mean(valid) * 100
        print(f"  {key:<25s}: σ/μ = {sm:.3f}%")

    results['Eta'] = Eta_axis
    results['R'] = R_axis[r_idx]
    return results


def plot_bicubic_comparison(bic_results, fname):
    """Plot bilinear vs bicubic at cardinal angles."""
    if not bic_results:
        print("  Skipping bicubic plot (missing data)")
        return

    Eta = bic_results['Eta']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f'Bilinear vs Bicubic Interpolation — R ≈ {bic_results["R"]:.0f} px, ΔR = 0.5 px',
        fontsize=14, fontweight='bold')

    colors = {
        'bilinear_standard': '#457b9d',
        'bilinear_corrected': '#e63946',
        'bicubic_standard': '#2a9d8f',
        'bicubic_corrected': '#e76f51',
    }
    labels = {
        'bilinear_standard': 'Bilinear, standard',
        'bilinear_corrected': 'Bilinear, corrected',
        'bicubic_standard': 'Bicubic, standard',
        'bicubic_corrected': 'Bicubic, corrected',
    }
    lstyles = {
        'bilinear_standard': '-',
        'bilinear_corrected': '-',
        'bicubic_standard': '--',
        'bicubic_corrected': '--',
    }

    windows = [(0, 'η = 0°'), (90, 'η = +90°'), (-90, 'η = −90°')]

    for ax, (angle, title) in zip(axes, windows):
        mask = np.abs(Eta - angle) <= 15
        if np.sum(mask) < 3:
            continue
        eta_w = Eta[mask]
        for key in ['bilinear_standard', 'bilinear_corrected',
                    'bicubic_standard', 'bicubic_corrected']:
            I_w = bic_results[key][mask]
            # Stats within ±5°
            sm_mask = np.abs(eta_w - angle) <= 5
            valid = I_w[sm_mask]
            valid = valid[valid > 0]
            sm = np.std(valid) / np.mean(valid) * 100 if len(valid) > 0 else 0
            lbl = f'{labels[key]} (σ/μ={sm:.1f}%)'
            ax.plot(eta_w, I_w, lstyles[key], lw=1.2,
                    color=colors[key], alpha=0.8, label=lbl)

        ax.axvline(angle, color='red', ls='--', lw=1.0, alpha=0.4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('η (degrees)', fontsize=10)
        ax.legend(fontsize=7, loc='best')
        if ax == axes[0]:
            ax.set_ylabel('Intensity (counts)', fontsize=11)

    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 11: Fourier analysis of I(η) — aliasing frequency
# ═══════════════════════════════════════════════════════════════════

def experiment_fourier_analysis(midas_results):
    """Compute 1D FFT of I(η) at the ring radius to show aliasing peak."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 11: Fourier Analysis of I(η)")
    print("=" * 60)

    drs_to_show = [0.5, 1.0, 2.0]
    results = {}

    for dr in drs_to_show:
        if dr not in midas_results:
            continue
        d = midas_results[dr]
        Eta, I_eta = d['Eta'], d['I_eta']
        N = len(I_eta)
        # Zero-mean before FFT
        signal = I_eta.astype(float) - np.mean(I_eta)
        # Compute FFT
        freqs = np.fft.rfftfreq(N, d=(Eta[1]-Eta[0]))
        power = np.abs(np.fft.rfft(signal))**2
        power_norm = power / power.max()  # normalize
        results[dr] = {'freqs': freqs, 'power': power_norm, 'Eta': Eta, 'I_eta': I_eta}
        # Find dominant peak
        peak_idx = np.argmax(power_norm[1:]) + 1  # skip DC
        peak_freq = freqs[peak_idx]
        peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
        print(f"  ΔR = {dr} px: dominant frequency = {peak_freq:.4f} /°, "
              f"period = {peak_period:.2f}°")

    return results


def plot_fourier_spectrum(fourier_results, fname):
    """Plot power spectra of I(η) for different ΔR values."""
    if not fourier_results:
        return

    drs = sorted(fourier_results.keys())
    fig, axes = plt.subplots(len(drs), 1, figsize=(10, 3.5 * len(drs)),
                             sharex=True)
    if len(drs) == 1:
        axes = [axes]

    colors = {0.5: '#e74c3c', 1.0: '#3498db', 2.0: '#2ecc71'}

    for ax, dr in zip(axes, drs):
        d = fourier_results[dr]
        freqs, power = d['freqs'], d['power']
        color = colors.get(dr, 'gray')
        ax.semilogy(freqs, power, color=color, lw=1.2)
        ax.set_ylabel('Normalized power', fontsize=11)
        ax.set_title(f'ΔR = {dr} px', fontsize=11, loc='left')
        ax.set_xlim(0, 1.05)  # Up to just past Nyquist
        ax.axvline(1.0 / (2 * ETA_BIN_SIZE), color='red', ls='--',
                  lw=0.8, alpha=0.6, label=f'Nyquist = {1.0/(2*ETA_BIN_SIZE):.1f} /°')
        # Mark the 2-bin period frequency = 1/(2 * EtaBinSize)
        if dr < 1.0:
            f_alias = 1.0 / (2 * ETA_BIN_SIZE)
            ax.annotate(f'2-bin aliasing\n(f = {f_alias:.1f} /°)',
                       xy=(f_alias, power[np.argmin(np.abs(freqs - f_alias))]),
                       xytext=(f_alias + 0.15, 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=9, color='red')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Spatial frequency (cycles / °)', fontsize=11)
    fig.suptitle('Power spectrum of I(η) at R ≈ 955 px', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    save(fig, fname)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 12: Flux conservation and FWHM analysis
# ═══════════════════════════════════════════════════════════════════

def experiment_flux_fwhm(grad_results):
    """Compare mean intensity and fitted FWHM with/without correction."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 12: Flux Conservation & FWHM Analysis")
    print("=" * 60)

    if 'no_grad' not in grad_results or 'grad' not in grad_results:
        print("  Need gradient results. Skipping.")
        return None

    ng = grad_results['no_grad']
    gc = grad_results['grad']
    R, Eta = ng['R'], ng['Eta']
    I_std = ng['I']  # (nR, nEta)
    I_cor = gc['I']

    # --- Flux conservation: global mean intensity change ---
    # Sum over all bins
    total_std = I_std[I_std > 0].sum()
    total_cor = I_cor[I_cor > 0].sum()
    flux_change = (total_cor - total_std) / total_std * 100
    print(f"  Total flux: Standard = {total_std:.0f}, Corrected = {total_cor:.0f}")
    print(f"  Flux change: {flux_change:+.3f}%")

    # Mean intensity at the ring
    ridx = np.argmin(np.abs(R - R_RING))
    mean_std = np.mean(I_std[ridx, :])
    mean_cor = np.mean(I_cor[ridx, :])
    mean_change = (mean_cor - mean_std) / mean_std * 100
    print(f"  Ring mean intensity: Standard = {mean_std:.1f}, Corrected = {mean_cor:.1f}")
    print(f"  Ring mean change: {mean_change:+.3f}%")

    # --- FWHM analysis: fit Gaussian to radial profile at several η values ---
    from scipy.optimize import curve_fit

    def gauss(x, a, mu, sigma, bg):
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

    fwhm_std_list = []
    fwhm_cor_list = []
    eta_fitted = []

    for j in range(0, len(Eta), max(1, len(Eta) // 72)):
        i_std_r = I_std[:, j].astype(float)
        i_cor_r = I_cor[:, j].astype(float)

        # Only fit around the ring
        r_half = 15
        rmask = (R >= R_RING - r_half) & (R <= R_RING + r_half)
        ri = np.where(rmask)[0]
        Rsub = R[ri]
        is_r = i_std_r[ri]
        ic_r = i_cor_r[ri]

        try:
            p0 = [is_r.max() - is_r.min(), R_RING, 2.0, is_r.min()]
            popt_s, _ = curve_fit(gauss, Rsub, is_r, p0=p0, maxfev=5000)
            popt_c, _ = curve_fit(gauss, Rsub, ic_r, p0=p0, maxfev=5000)
            fwhm_s = abs(popt_s[2]) * 2.3548
            fwhm_c = abs(popt_c[2]) * 2.3548
            fwhm_std_list.append(fwhm_s)
            fwhm_cor_list.append(fwhm_c)
            eta_fitted.append(Eta[j])
        except Exception:
            pass

    fwhm_std_arr = np.array(fwhm_std_list)
    fwhm_cor_arr = np.array(fwhm_cor_list)
    eta_f = np.array(eta_fitted)

    mean_fwhm_std = np.median(fwhm_std_arr)
    mean_fwhm_cor = np.median(fwhm_cor_arr)
    fwhm_change = (mean_fwhm_cor - mean_fwhm_std) / mean_fwhm_std * 100
    print(f"  Median FWHM: Standard = {mean_fwhm_std:.4f} px, "
          f"Corrected = {mean_fwhm_cor:.4f} px")
    print(f"  FWHM change: {fwhm_change:+.2f}%")

    return {
        'flux_change_pct': flux_change,
        'mean_change_pct': mean_change,
        'fwhm_std': mean_fwhm_std,
        'fwhm_cor': mean_fwhm_cor,
        'fwhm_change_pct': fwhm_change,
        'fwhm_std_arr': fwhm_std_arr,
        'fwhm_cor_arr': fwhm_cor_arr,
        'eta_fitted': eta_f,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Cardinal-Angle Aliasing — Comprehensive Test Suite")
    print("=" * 60)

    # ── Run all experiments ──────────────────────────────────────
    midas_results  = experiment_midas()       # Exp 1: ΔR sweep (5 values)
    tilt_results   = experiment_tiltstudy()    # Exp 2: tilt dependence
    tx_results     = experiment_tx_rotation()  # Exp 4: tx rotation
    pyfai_results, midas_zt = experiment_pyfai()  # Exp 5: pyFAI comparison (+ MIDAS zero-tilt)
    grad_results   = experiment_gradient_correction()  # Exp 6: gradient A/B
    pf_results     = experiment_peak_fitting(grad_results)  # Exp 7: peak fitting
    bench_results  = experiment_benchmark()    # Exp 8: throughput benchmark
    fourier_results = experiment_fourier_analysis(midas_results) if midas_results else {}
    flux_results   = experiment_flux_fwhm(grad_results) if grad_results else None

    # ── Generate all paper figures ───────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Fig 1: schematic.pdf (pixel splitting geometry diagram)
    plot_schematic('schematic')

    if midas_results:
        # Fig 2: raw_vs_caked.pdf (raw detector image + R-η cake)
        plot_raw_vs_caked(midas_results, 'raw_vs_caked')

        # Fig 3: aliasing_cake_comparison.pdf (4-panel cake: ΔR=0.5 + residual, 1.0, 2.0)
        plot_cake_comparison(midas_results, 'aliasing_cake_comparison')

        # Fig 3: rbinsweep.pdf (I(η) for 5 ΔR values)
        plot_ieta_sweep(midas_results, 'MIDAS — CeO₂ (220)', 'rbinsweep')

        # Fig 7: aliasing_binarea.pdf (bin area flatness proof)
        plot_binarea(midas_results, 'aliasing_binarea')

        # Supplementary: individual cake plots
        plot_cake(midas_results, 0.5, 'CeO₂ (220) — aliasing visible', 'aliasing_cake_dr0p5')
        plot_cake(midas_results, 1.0, 'aliasing-free reference', 'aliasing_cake_dr1p0')
        plot_cake(midas_results, 2.0, 'aliasing-free reference', 'aliasing_cake_dr2p0')

        # Also generate the generic MIDAS sweep plot
        plot_ieta_sweep(midas_results, 'MIDAS', 'aliasing_midas_sweep')

    # Fig 4: tiltstudy.pdf (2×2 zoom: BC/tilt combos)
    if tilt_results:
        plot_tiltstudy(tilt_results, 'tiltstudy')

    # Fig 5: tx_rotation.pdf (2×2: tx=0,30,45,90)
    if tx_results:
        plot_tx_rotation(tx_results, 'tx_rotation')

    # Gradient correction comparison
    if grad_results:
        plot_gradient_comparison(grad_results, 'gradient_correction')
        plot_correction_cake(grad_results, 'correction_cake')

    # Peak fitting R_center(η) wobble
    if pf_results:
        plot_peak_fitting(pf_results, 'peak_fitting_wobble')

    # Fig 6: aliasing_midas_vs_pyfai.pdf (MIDAS vs pyFAI side-by-side)
    if pyfai_results:
        plot_ieta_sweep(pyfai_results, 'pyFAI', 'aliasing_pyfai_sweep')

    # Use zero-tilt MIDAS for fair comparison with pyFAI
    midas_for_comp = midas_zt if midas_zt else midas_results
    if midas_for_comp:
        plot_midas_vs_pyfai(midas_for_comp, pyfai_results, 'aliasing_midas_vs_pyfai')
        print_summary(midas_for_comp, pyfai_results)

    # Fig 6b: MIDAS vs pyFAI 2D cake side-by-side
    if pyfai_results and midas_for_comp:
        plot_pyfai_cake(midas_for_comp, pyfai_results, 'aliasing_pyfai_cake')

    # Fourier spectrum of I(η)
    if fourier_results:
        plot_fourier_spectrum(fourier_results, 'fourier_spectrum')

    # ── Benchmark summary ────────────────────────────────────────
    if bench_results and 'Standard' in bench_results and 'Corrected' in bench_results:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        s = bench_results['Standard']
        c = bench_results['Corrected']
        print(f"  Standard:  {s['median_ms']:.1f} ms/frame (median)")
        print(f"  Corrected: {c['median_ms']:.1f} ms/frame (median)")
        overhead = bench_results.get('overhead_pct', 0)
        print(f"  Overhead:  {overhead:+.1f}%")

    # ── Summary of paper figures ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER FIGURE CHECKLIST")
    print("=" * 60)
    paper_figs = [
        ('Fig 1', 'schematic.pdf', 'Pixel splitting schematic'),
        ('Fig 2', 'aliasing_cake_comparison.pdf', 'Cake comparison ΔR=0.5/1.0/2.0'),
        ('Fig 3', 'rbinsweep.pdf', 'I(η) sweep for 5 ΔR values'),
        ('Fig 4', 'tiltstudy.pdf', 'Tilt dependence (2×2 zoom)'),
        ('Fig 5', 'tx_rotation.pdf', 'Detector rotation tx=0/30/45/90°'),
        ('Fig 6', 'aliasing_midas_vs_pyfai.pdf', 'MIDAS vs pyFAI'),
        ('Fig 7', 'aliasing_binarea.pdf', 'BinArea flatness proof'),
        ('Fig 8', 'gradient_correction.png', 'Gradient correction A/B'),
        ('Fig 9', 'peak_fitting_wobble.png', 'Peak fitting R_center(η) wobble'),
    ]
    for label, fn, desc in paper_figs:
        path = os.path.join(FIG_DIR, fn)
        status = '✓' if os.path.isfile(path) else '✗ MISSING'
        print(f"  {status}  {label}: {fn}  ({desc})")

    print(f"\nDone! All figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    main()

