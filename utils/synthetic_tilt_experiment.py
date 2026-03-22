#!/usr/bin/env python3
"""
Synthetic Tilted-Detector Experiment
=====================================
Generates a synthetic CeO₂-like diffraction image and integrates it
with MIDAS at multiple detector tilt angles, with and without the
radial resampling (GradientCorrection).

The synthetic image uses the MIDAS forward geometry to compute the
exact 2θ at each pixel for a given tilt, then places Gaussian rings
at the known CeO₂ d-spacings. This provides a ground truth against
which we can measure the cardinal-angle aliasing and correction
effectiveness.

Usage:
    python synthetic_tilt_experiment.py
"""

import os
import sys
import subprocess
import time
import struct

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ─── Configuration ─────────────────────────────────────────────────
MIDAS_BIN  = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin')
FIG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
WORK_BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_runs', 'synthetic_tilt')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(WORK_BASE, exist_ok=True)

# Detector geometry matching the real CeO₂ experiment
NPY = 2880
NPZ = 2880
PX_UM   = 150.0        # pixel pitch (µm)
LSD_UM  = 900450.95     # sample–detector distance (µm)
WAVELENGTH_A = 0.19582  # wavelength (Å)
BC_Y = 1447.07          # beam center Y (px)
BC_Z = 1468.59          # beam center Z (px)

# CeO₂ lattice constant (Å) and space group 225 (Fm-3m)
# Allowed reflections: h²+k²+l² all even or all odd → h+k+l = 4n gives 111,200,220,311,222,...
A0 = 5.4116
RING_HKL = [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0), (3,3,1), (4,2,0)]
RING_PEAK_COUNTS = 50000   # HIGH peak intensity to isolate aliasing from noise
RING_FWHM_PX = 2.7         # FWHM in pixels
BG_COUNTS = 100             # background
ADD_NOISE = True            # Enable Poisson noise — needed to see aliasing

# Bin parameters
DR = 0.5                # radial bin size (px)
ETA_BIN = 0.5           # azimuthal bin size (deg)
R_MIN = 10.25
R_MAX = 1500.25

# Tilt angles to test (degrees)
TILT_CASES = [
    ('no_tilt',     0.0,    0.0),
    ('small_tilt',  -0.236, 0.403),     # real calibrated tilts
    ('medium_tilt', 2.0,    3.0),
    ('large_tilt',  5.0,    8.0),
    ('extreme_tilt', 10.0,  15.0),
]

# Cardinal angles for analysis
CARDINALS = [0, 90, -90]
NON_CARDINALS = [30, 45, 60, 120, 135, 150, -30, -45, -60, -120, -135, -150]
WINDOW_HW = 5  # ±5° for sigma/mu calculation


def deg2rad(d):
    return d * np.pi / 180.0


def compute_ring_radii():
    """Compute ring radii in pixels for CeO₂ reflections."""
    radii = []
    for h, k, l in RING_HKL:
        d_hkl = A0 / np.sqrt(h**2 + k**2 + l**2)
        two_theta = 2 * np.arcsin(WAVELENGTH_A / (2 * d_hkl))
        R_px = (LSD_UM / PX_UM) * np.tan(two_theta)
        if R_px < R_MAX:
            radii.append((R_px, f'{h}{k}{l}'))
    return radii


def generate_synthetic_image(ty_deg, tz_deg):
    """Generate a synthetic diffraction image for given detector tilts.

    Uses the EXACT same coordinate transform as MIDAS's
    dg_pixel_to_REta() in DetectorGeometry.c:
      Yc = (-Y + Ycen) * px       (Y is flipped!)
      Zc = (Z - Zcen) * px
      ABC = [0, Yc, Zc]
      ABCPr = TRs @ ABC           where TRs = Rx @ (Ry @ Rz)
      XYZ = [Lsd + ABCPr[0], ABCPr[1], ABCPr[2]]
      R_um = (Lsd / XYZ[0]) * sqrt(XYZ[1]^2 + XYZ[2]^2)
      R_px = R_um / px
    """
    # Build tilt matrix TRs = Rx @ (Ry @ Rz), with tx=0
    txr = 0.0  # tx is always 0 in our experiments
    tyr = deg2rad(ty_deg)
    tzr = deg2rad(tz_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(txr), -np.sin(txr)],
                   [0, np.sin(txr),  np.cos(txr)]])
    Ry = np.array([[ np.cos(tyr), 0, np.sin(tyr)],
                   [0,            1, 0],
                   [-np.sin(tyr), 0, np.cos(tyr)]])
    Rz = np.array([[np.cos(tzr), -np.sin(tzr), 0],
                   [np.sin(tzr),  np.cos(tzr), 0],
                   [0,            0,            1]])
    TRs = Rx @ (Ry @ Rz)

    ring_radii = compute_ring_radii()
    sigma_px = RING_FWHM_PX / (2 * np.sqrt(2 * np.log(2)))

    img = np.full((NPY, NPZ), BG_COUNTS, dtype=np.float64)

    # Pixel coordinate arrays
    y_arr = np.arange(NPY, dtype=np.float64)
    z_arr = np.arange(NPZ, dtype=np.float64)
    YY, ZZ = np.meshgrid(y_arr, z_arr, indexing='ij')

    # MIDAS convention (DetectorGeometry.c lines 88-89):
    #   Yc = (-Y + Ycen) * px    ← Y is FLIPPED
    #   Zc = (Z - Zcen) * px
    Yc_um = (-YY + BC_Y) * PX_UM
    Zc_um = (ZZ - BC_Z) * PX_UM

    # ABC = [0, Yc, Zc], then ABCPr = TRs @ ABC
    # XYZ = [Lsd + ABCPr[0], ABCPr[1], ABCPr[2]]
    ABCPr_0 = TRs[0,1] * Yc_um + TRs[0,2] * Zc_um
    ABCPr_1 = TRs[1,1] * Yc_um + TRs[1,2] * Zc_um
    ABCPr_2 = TRs[2,1] * Yc_um + TRs[2,2] * Zc_um

    XYZ_0 = LSD_UM + ABCPr_0
    XYZ_1 = ABCPr_1
    XYZ_2 = ABCPr_2

    # R in µm, then convert to pixels
    R_um = (LSD_UM / XYZ_0) * np.sqrt(XYZ_1**2 + XYZ_2**2)
    R_px = R_um / PX_UM

    # Place Gaussian rings at known R values
    for R_ring, hkl_label in ring_radii:
        ring_intensity = RING_PEAK_COUNTS * np.exp(
            -0.5 * ((R_px - R_ring) / sigma_px)**2
        )
        img += ring_intensity

    # Optionally add Poisson noise
    img = np.maximum(img, 0)
    if ADD_NOISE:
        img = np.random.poisson(img.astype(int)).astype(np.float32)
    else:
        img = img.astype(np.float32)

    # CRITICAL: Transpose before saving!
    # The image is indexed as img[y, z], but MIDAS reads TIFFs with
    # rows = Z (slow axis j) and columns = Y (fast axis i), i.e.
    # data[j * NrPixelsY + i]. So the TIFF must have Z as rows.
    img = img.T  # now img[z, y] — matches TIFF layout for MIDAS

    return img, ring_radii


# ── Panel gap mask and calibration image for Paper 3 ──

def generate_panel_gap_mask(ny=NPY, nz=NPZ, n_panels_y=3, n_panels_z=4,
                            gap_y=17, gap_z=17):
    """Generate a Pilatus-like panel gap mask.

    Parameters
    ----------
    ny, nz : int
        Detector dimensions (pixels).
    n_panels_y, n_panels_z : int
        Number of panels along Y and Z.
    gap_y, gap_z : int
        Gap width in pixels between panels.

    Returns
    -------
    mask : ndarray (nz, ny), uint8
        0 = good pixel, 1 = gap/bad pixel.  Shape is (nz, ny) to match
        TIFF row-major layout used by MIDAS.
    """
    mask = np.zeros((nz, ny), dtype=np.uint8)

    # Panel sizes (excluding gaps)
    total_gaps_y = gap_y * (n_panels_y - 1)
    total_gaps_z = gap_z * (n_panels_z - 1)
    panel_h = (ny - total_gaps_y) // n_panels_y  # panel height in Y
    panel_w = (nz - total_gaps_z) // n_panels_z  # panel width in Z

    # Horizontal gaps (along Y direction — column gaps in TIFF)
    for i in range(1, n_panels_y):
        y_start = i * panel_h + (i - 1) * gap_y
        y_end = y_start + gap_y
        mask[:, y_start:y_end] = 1

    # Vertical gaps (along Z direction — row gaps in TIFF)
    for i in range(1, n_panels_z):
        z_start = i * panel_w + (i - 1) * gap_z
        z_end = z_start + gap_z
        mask[z_start:z_end, :] = 1

    return mask


def generate_synthetic_calibration_image(ty_deg=0.0, tz_deg=0.0,
                                          add_gaps=False,
                                          n_panels_y=3, n_panels_z=4,
                                          gap_y=17, gap_z=17,
                                          output_dir=None, tag='synth'):
    """Generate a synthetic calibration image with optional panel gaps.

    Saves:
      - {tag}.tif : synthetic image (TIFF, uint32)
      - {tag}_mask.tif : panel gap mask if add_gaps=True
      - {tag}_ground_truth.txt : ground truth parameters

    Returns
    -------
    img_path : str
        Path to the saved TIFF image.
    truth : dict
        Ground truth geometry parameters.
    """
    from PIL import Image as PILImage

    if output_dir is None:
        output_dir = os.path.join(WORK_BASE, 'calibration_test')
    os.makedirs(output_dir, exist_ok=True)

    # Generate the image
    img, ring_radii = generate_synthetic_image(ty_deg, tz_deg)

    # Apply panel gap mask if requested
    mask = None
    if add_gaps:
        mask = generate_panel_gap_mask(
            ny=NPY, nz=NPZ, n_panels_y=n_panels_y, n_panels_z=n_panels_z,
            gap_y=gap_y, gap_z=gap_z)
        img[mask == 1] = -1  # sentinel value for gap pixels

    # Save as TIFF
    img_path = os.path.join(output_dir, f'{tag}.tif')
    pil_img = PILImage.fromarray(img.astype(np.int32))
    pil_img.save(img_path)

    # Save mask
    mask_path = None
    if mask is not None:
        mask_path = os.path.join(output_dir, f'{tag}_mask.tif')
        pil_mask = PILImage.fromarray(mask)
        pil_mask.save(mask_path)

    # Save ground truth
    truth = {
        'ty_deg': ty_deg, 'tz_deg': tz_deg,
        'bc_y': BC_Y, 'bc_z': BC_Z,
        'lsd_um': LSD_UM, 'px_um': PX_UM,
        'wavelength_a': WAVELENGTH_A,
        'npy': NPY, 'npz': NPZ,
        'a0': A0, 'space_group': 225,
        'add_gaps': add_gaps,
        'n_panels_y': n_panels_y if add_gaps else 0,
        'n_panels_z': n_panels_z if add_gaps else 0,
    }
    truth_path = os.path.join(output_dir, f'{tag}_ground_truth.txt')
    with open(truth_path, 'w') as f:
        for k, v in truth.items():
            f.write(f'{k} {v}\n')

    print(f"Saved: {img_path}")
    if mask_path:
        print(f"Saved: {mask_path}")
    print(f"Saved: {truth_path}")
    print(f"Ground truth: ty={ty_deg:.3f}° tz={tz_deg:.3f}°, "
          f"BC=({BC_Y:.2f}, {BC_Z:.2f}), Lsd={LSD_UM:.2f}µm")

    return img_path, truth


def save_raw_image(img, path, fig_dir, tag):
    """Save raw 2D synthetic image as PNG for visual inspection."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # Linear scale
    ax = axes[0]
    im = ax.imshow(img, origin='lower', cmap='viridis',
                   vmin=0, vmax=np.percentile(img[img > 0], 99))
    ax.set_title(f'{tag} — linear scale', fontsize=12)
    ax.set_xlabel('Z (px)'); ax.set_ylabel('Y (px)')
    fig.colorbar(im, ax=ax, shrink=0.7)
    # Log scale
    ax = axes[1]
    img_log = np.log10(np.maximum(img, 1).astype(float))
    im = ax.imshow(img_log, origin='lower', cmap='viridis')
    ax.set_title(f'{tag} — log₁₀ scale', fontsize=12)
    ax.set_xlabel('Z (px)'); ax.set_ylabel('Y (px)')
    fig.colorbar(im, ax=ax, shrink=0.7, label='log₁₀(I)')
    # Mark beam center
    for ax in axes:
        ax.axhline(BC_Y, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.axvline(BC_Z, color='red', ls='--', lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'synthetic_raw_{tag}.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved synthetic_raw_{tag}.png")


def save_as_tif(img, path):
    """Save a 2D float32 array as a 32-bit TIFF."""
    try:
        import tifffile
        tifffile.imwrite(path, img.astype(np.float32))
    except ImportError:
        from PIL import Image
        # PIL doesn't support float32 TIF well, use uint16
        img_uint = np.clip(img, 0, 65535).astype(np.uint16)
        Image.fromarray(img_uint).save(path)


def write_param_file(dst_fn, ty, tz, gradient_correction):
    """Write a MIDAS parameter file for the synthetic experiment."""
    gc = 1 if gradient_correction else 0
    content = f"""Folder .
FileStem synthetic
Ext .tif
StartNr 1
EndNr 1
ImTransOpt 0
DataType 6
Lsd        {LSD_UM}
BC         {BC_Y} {BC_Z}
ty         {ty}
tz         {tz}
p0         0
p1         0
p2         0
p3         0
p4         0
p5         0
RhoD       309094.286
Wedge 0
EtaBinSize {ETA_BIN}
Padding 6
NrPixelsY {NPY}
NrPixelsZ {NPZ}
px {PX_UM}
Wavelength {WAVELENGTH_A}
SpaceGroup 225
LatticeConstant {A0} {A0} {A0} 90.0 90.0 90.0
RMin {R_MIN}
RMax {R_MAX}
RBinSize {DR}
EtaMin -180.5
EtaMax 180.5
DoPeakFit 0
skipFrame 0
Normalize 1
SubPixelLevel 1
GradientCorrection {gc}
"""
    with open(dst_fn, 'w') as f:
        f.write(content)


def run_midas(work_dir, ty, tz, gradient_correction, img_src_dir=None):
    """Run DetectorMapper + IntegratorZarrOMP on synthetic data."""
    os.makedirs(work_dir, exist_ok=True)

    pf = os.path.join(work_dir, 'ps_test.txt')
    write_param_file(pf, ty, tz, gradient_correction)

    # Symlink the synthetic image into this work directory
    tif_name = 'synthetic_000001.tif'
    dst_tif = os.path.join(work_dir, tif_name)
    if img_src_dir and not os.path.isfile(dst_tif):
        src_tif = os.path.join(img_src_dir, tif_name)
        if os.path.isfile(src_tif):
            os.symlink(os.path.abspath(src_tif), dst_tif)

    mapper = os.path.join(MIDAS_BIN, 'DetectorMapper')
    integrator = os.path.join(MIDAS_BIN, 'IntegratorZarrOMP')

    # Run DetectorMapper
    map_bin = os.path.join(work_dir, 'Map.bin')
    if not os.path.isfile(map_bin):
        print(f"    Running DetectorMapper ...", end='', flush=True)
        t0 = time.time()
        r1 = subprocess.run([mapper, 'ps_test.txt'],
                            capture_output=True, text=True,
                            timeout=300, cwd=work_dir)
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
    data_path = os.path.join(work_dir, tif_name)
    print(f"    Running IntegratorZarrOMP (GC={'ON' if gradient_correction else 'OFF'}) ...",
          end='', flush=True)
    t0 = time.time()
    r2 = subprocess.run([integrator, '-paramFN', 'ps_test.txt',
                         '-dataFN', data_path, '-nCPUs', '8'],
                        capture_output=True, text=True, timeout=60,
                        cwd=work_dir)
    dt = time.time() - t0
    caked = data_path + '.caked.hdf'
    if r2.returncode != 0 or not os.path.isfile(caked):
        print(f" FAILED (rc={r2.returncode}, {dt:.1f}s)")
        if r2.stderr:
            print(f"      stderr: {r2.stderr[:300]}")
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


def window_metrics(Eta, I_eta, center, hw=WINDOW_HW):
    """Compute sigma/mu and peak-to-peak in a window around center."""
    mask = np.abs(Eta - center) < hw
    vals = I_eta[mask]
    vals = vals[vals > 0]
    if len(vals) < 5:
        return None
    mu = np.mean(vals)
    sigma = np.std(vals)
    ptp = np.max(vals) - np.min(vals)
    return {
        'sigma_mu': 100 * sigma / mu if mu > 0 else 0,
        'ptp_rel': 100 * ptp / mu if mu > 0 else 0,
        'ptp': ptp,
        'mu': mu,
    }


def all_metrics(Eta, I_eta, hw=WINDOW_HW):
    """Compute metrics at cardinal AND non-cardinal angles."""
    results = {'cardinal': {}, 'non_cardinal': {}}
    for ca in CARDINALS:
        m = window_metrics(Eta, I_eta, ca, hw)
        if m:
            results['cardinal'][ca] = m
    for nc in NON_CARDINALS:
        m = window_metrics(Eta, I_eta, nc, hw)
        if m:
            results['non_cardinal'][nc] = m
    return results


def experiment_synthetic_tilt():
    """Main experiment: generate synthetic data and test correction
    across multiple tilt angles."""
    print("\n" + "=" * 60)
    print("SYNTHETIC TILTED-DETECTOR EXPERIMENT")
    print("=" * 60)

    ring_radii = compute_ring_radii()
    print(f"\nCeO₂ ring radii (px): "
          + ", ".join(f"{R:.1f} ({hkl})" for R, hkl in ring_radii))

    # Pick the (220) ring for analysis (same as the real experiment)
    R_RING = None
    for R, hkl in ring_radii:
        if hkl == '220':
            R_RING = R
            break
    if R_RING is None:
        R_RING = ring_radii[2][0]  # fallback to 3rd ring
    print(f"Analysis ring: R = {R_RING:.1f} px (220)")

    all_results = {}

    for case_name, ty_deg, tz_deg in TILT_CASES:
        print(f"\n{'─'*50}")
        print(f"  Case: {case_name} (ty={ty_deg}°, tz={tz_deg}°)")
        print(f"{'─'*50}")

        # Generate synthetic image
        work_dir = os.path.join(WORK_BASE, case_name)
        os.makedirs(work_dir, exist_ok=True)
        img_path = os.path.join(work_dir, 'synthetic_000001.tif')

        if not os.path.isfile(img_path):
            print(f"  Generating synthetic image ...", end='', flush=True)
            t0 = time.time()
            img, _ = generate_synthetic_image(ty_deg, tz_deg)
            save_as_tif(img, img_path)
            print(f" done ({time.time()-t0:.1f}s, shape={img.shape})")
            # Save raw 2D image for inspection
            save_raw_image(img, img_path, FIG_DIR, case_name)
        else:
            print(f"  Reusing existing synthetic image")

        case_results = {}

        for gc_label, gc_flag, sub_dir in [
            ('standard', False, 'standard'),
            ('corrected', True, 'corrected'),
        ]:
            work_sub = os.path.join(work_dir, sub_dir)
            caked = run_midas(work_sub, ty_deg, tz_deg,
                             gradient_correction=gc_flag,
                             img_src_dir=work_dir)
            if not caked:
                continue

            R, Eta, I, Area = read_caked_hdf(caked)
            # Find peak R-bin (highest mean intensity near expected ring)
            r_window = np.abs(R - R_RING) < 5
            r_candidates = np.where(r_window)[0]
            if len(r_candidates) == 0:
                continue
            # Pick the R-bin with the highest mean intensity
            mean_I = [np.nanmean(I[ri, I[ri,:] > 0]) if np.any(I[ri,:] > 0)
                      else 0 for ri in r_candidates]
            ridx = r_candidates[np.argmax(mean_I)]
            I_eta = I[ridx, :]

            metrics = all_metrics(Eta, I_eta)
            case_results[gc_label] = {
                'R': R, 'Eta': Eta, 'I_eta': I_eta, 'I_full': I,
                'ridx': ridx, 'R_actual': R[ridx], 'metrics': metrics,
            }

            # Print cardinal vs non-cardinal comparison
            card_sm = [m['sigma_mu'] for m in metrics['cardinal'].values()]
            noncard_sm = [m['sigma_mu'] for m in metrics['non_cardinal'].values()
                          if m['sigma_mu'] < 50]  # filter outliers
            avg_card = np.mean(card_sm) if card_sm else 0
            avg_noncard = np.mean(noncard_sm) if noncard_sm else 0
            ratio = avg_card / avg_noncard if avg_noncard > 0 else 0

            print(f"    {gc_label.upper()} (R={R[ridx]:.1f} px, "
                  f"mean I={np.mean(I_eta[I_eta>0]):.0f}):")
            print(f"      Cardinal avg σ/μ = {avg_card:.2f}%")
            print(f"      Non-cardinal avg σ/μ = {avg_noncard:.2f}%")
            print(f"      Cardinal/Non-cardinal ratio = {ratio:.2f}")
            for ca, m in metrics['cardinal'].items():
                print(f"        η={ca:+4d}°: σ/μ={m['sigma_mu']:.2f}%, "
                      f"p-p/μ={m['ptp_rel']:.2f}%")

        # Save cake image for this case
        if 'standard' in case_results:
            _save_cake_image(case_results, case_name, R_RING, FIG_DIR)

        all_results[case_name] = case_results

    return all_results, R_RING


def _save_cake_image(case_results, case_name, R_RING, fig_dir):
    """Save 2D cake image (R vs η) around the analysis ring.
    Uses tight zoom on ring peak with narrow vmin/vmax to reveal modulations."""
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)
    r_half = 5  # tight zoom on ring (FWHM ≈ 2.7 px)

    for ax, (key, label) in zip(axes, [('standard', 'Standard'),
                                        ('corrected', 'Corrected')]):
        if key not in case_results:
            continue
        cr = case_results[key]
        R = cr['R']
        Eta = cr['Eta']
        I = cr['I_full']
        rmask = (R >= R_RING - r_half) & (R <= R_RING + r_half)
        r_idx = np.where(rmask)[0]
        Isub = I[r_idx, :].T.astype(float)
        Isub[Isub <= 0] = np.nan

        # Full-region percentiles for color scale
        peak_vals = Isub[np.isfinite(Isub)]
        vmin_lin = np.percentile(peak_vals, 5)
        vmax_lin = np.percentile(peak_vals, 95)

        im = ax.pcolormesh(R[r_idx], Eta, Isub, shading='auto', cmap='inferno',
                           vmin=vmin_lin, vmax=vmax_lin, rasterized=True)
        ax.set_ylabel('η (°)')
        ax.set_title(f'{case_name.replace("_"," ").title()} — {label}')
        fig.colorbar(im, ax=ax, label='I', pad=0.02, shrink=0.85)
        # Cardinal boxes (±10°)
        for ca in CARDINALS:
            xlim = ax.get_xlim()
            rect = Rectangle((xlim[0], ca - 10), xlim[1] - xlim[0], 20,
                              lw=1.0, edgecolor='red', facecolor='none',
                              ls='--', alpha=0.7)
            ax.add_patch(rect)
        # Non-cardinal boxes
        for nc in [45, -45, 135, -135]:
            xlim = ax.get_xlim()
            rect = Rectangle((xlim[0], nc - 5), xlim[1] - xlim[0], 10,
                              lw=1.0, edgecolor='cyan', facecolor='none',
                              ls=':', alpha=0.5)
            ax.add_patch(rect)

    axes[-1].set_xlabel('R (px)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'synthetic_cake_{case_name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved synthetic_cake_{case_name}.{{pdf,png}}")


def plot_tilt_comparison(results, R_RING, fname):
    """Plot I(η) for standard vs corrected at each tilt."""
    ncases = len(results)
    fig, axes = plt.subplots(ncases, 1, figsize=(14, 3.5 * ncases),
                              sharex=True)
    if ncases == 1:
        axes = [axes]

    fig.suptitle(
        f'Synthetic Tilted-Detector Experiment: I(η) at R ≈ {R_RING:.0f} px\n'
        f'Standard (blue) vs Corrected (red)',
        fontsize=13, fontweight='bold')

    for ax, (case_name, ty_deg, tz_deg) in zip(axes, TILT_CASES):
        if case_name not in results:
            continue
        cr = results[case_name]

        if 'standard' in cr:
            ax.plot(cr['standard']['Eta'], cr['standard']['I_eta'],
                    '-', color='steelblue', lw=0.5, alpha=0.8,
                    label='Standard')
        if 'corrected' in cr:
            ax.plot(cr['corrected']['Eta'], cr['corrected']['I_eta'],
                    '-', color='#e63946', lw=0.5, alpha=0.8,
                    label='Corrected')

        # Mark cardinals
        for ca in CARDINALS:
            ax.axvline(ca, color='grey', ls=':', lw=0.5, alpha=0.5)

        ax.set_ylabel('I (counts)', fontsize=10)
        label = f'{case_name.replace("_", " ").title()} (ty={ty_deg}°, tz={tz_deg}°)'
        # Add metrics to title
        if 'standard' in cr and 'corrected' in cr:
            m_std = cr['standard']['metrics']['cardinal'].get(0, {})
            m_cor = cr['corrected']['metrics']['cardinal'].get(0, {})
            if m_std and m_cor:
                label += (f'  |  η=0°: σ/μ {m_std["sigma_mu"]:.1f}% → '
                         f'{m_cor["sigma_mu"]:.1f}%')
        ax.set_title(label, fontsize=10, loc='left')
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('η (°)', fontsize=11)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\n  Saved {fname}.{{pdf,png}}")


def plot_tilt_zoom(results, R_RING, fname):
    """Plot zoomed views around cardinal angles for each tilt."""
    ncases = len(results)
    fig, axes = plt.subplots(ncases, 3, figsize=(16, 3 * ncases),
                              sharex='col')
    if ncases == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Zoomed Cardinal Angles: Standard (blue) vs Corrected (red)',
        fontsize=13, fontweight='bold')

    for row, (case_name, ty_deg, tz_deg) in enumerate(TILT_CASES):
        if case_name not in results:
            continue
        cr = results[case_name]

        for col, ca in enumerate(CARDINALS):
            ax = axes[row, col]
            for key, color, label in [
                ('standard', 'steelblue', 'Std'),
                ('corrected', '#e63946', 'Corr'),
            ]:
                if key not in cr:
                    continue
                eta = cr[key]['Eta']
                ie = cr[key]['I_eta']
                mask = np.abs(eta - ca) < 15
                ax.plot(eta[mask], ie[mask], '-', color=color,
                        lw=0.8, alpha=0.9, label=label)

            ax.axvline(ca, color='grey', ls=':', lw=0.5)
            if row == 0:
                ax.set_title(f'η = {ca}°', fontsize=10)
            if col == 0:
                short = case_name.replace('_', ' ').title()
                ax.set_ylabel(f'{short}\nI', fontsize=9)
            if row == ncases - 1:
                ax.set_xlabel('η (°)', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}.{{pdf,png}}")


def plot_improvement_summary(results, fname):
    """Bar chart: σ/μ reduction (%) at η=0° as a function of tilt."""
    tilts = []
    reductions_0 = []
    reductions_90 = []
    reductions_m90 = []

    for case_name, ty_deg, tz_deg in TILT_CASES:
        if case_name not in results:
            continue
        cr = results[case_name]
        if 'standard' not in cr or 'corrected' not in cr:
            continue
        total_tilt = np.sqrt(ty_deg**2 + tz_deg**2)
        tilts.append(f'{case_name.replace("_"," ").title()}\n'
                    f'({total_tilt:.1f}°)')

        for ca, arr in [(0, reductions_0), (90, reductions_90),
                        (-90, reductions_m90)]:
            ms = cr['standard']['metrics']['cardinal'].get(ca)
            mc = cr['corrected']['metrics']['cardinal'].get(ca)
            if ms and mc:
                s = ms['sigma_mu']
                c = mc['sigma_mu']
                arr.append(100 * (1 - c/s) if s > 0 else 0)
            else:
                arr.append(0)

    x = np.arange(len(tilts))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, reductions_0, w, label='η = 0°', color='#e63946')
    ax.bar(x, reductions_90, w, label='η = +90°', color='#457b9d')
    ax.bar(x + w, reductions_m90, w, label='η = −90°', color='#2a9d8f')

    ax.set_ylabel('σ/μ Reduction (%)', fontsize=12)
    ax.set_xlabel('Tilt Configuration', fontsize=12)
    ax.set_title('Radial Resampling Correction Effectiveness vs. Detector Tilt',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tilts, fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}.{{pdf,png}}")


def _plot_paper_ieta_comparison(case_results, R_RING, fname):
    """I(η) at cardinal (η=0°) and non-cardinal (η=45°) for paper figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    fig.suptitle(f'I(η) at R ≈ {R_RING:.0f} px', fontsize=11, fontweight='bold')

    for ax, center, title in [(ax1, 0, 'η = 0° (cardinal)'),
                                (ax2, 45, 'η = 45° (non-cardinal)')]:
        window = 15  # ±15° view
        for key, color, label in [('standard', 'steelblue', 'Standard'),
                                   ('corrected', '#e63946', 'Corrected')]:
            if key not in case_results:
                continue
            eta = case_results[key]['Eta']
            ie = case_results[key]['I_eta']
            mask = np.abs(eta - center) < window
            ax.plot(eta[mask], ie[mask], '-', color=color, lw=0.8,
                    alpha=0.9, label=label)
        ax.axvline(center, color='grey', ls=':', lw=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('η (°)', fontsize=9)
        ax.legend(fontsize=7)
    ax1.set_ylabel('I (counts)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}.{{pdf,png}}")


def _plot_paper_ratio_barchart(results, fname):
    """Cardinal / non-cardinal σ/μ ratio as a function of tilt for paper."""
    tilts = []
    std_ratios = []
    cor_ratios = []

    for case_name, ty_deg, tz_deg in TILT_CASES:
        if case_name not in results:
            continue
        cr = results[case_name]
        if 'standard' not in cr or 'corrected' not in cr:
            continue
        total_tilt = np.sqrt(ty_deg**2 + tz_deg**2)
        tilts.append(f'{total_tilt:.1f}°')

        for key, arr in [('standard', std_ratios), ('corrected', cor_ratios)]:
            card_vals = [m['sigma_mu'] for m in cr[key]['metrics']['cardinal'].values()]
            nc_vals = [m['sigma_mu'] for m in cr[key]['metrics']['non_cardinal'].values()
                       if m['sigma_mu'] < 50]
            avg_c = np.mean(card_vals) if card_vals else 0
            avg_nc = np.mean(nc_vals) if nc_vals else 1
            arr.append(avg_c / avg_nc if avg_nc > 0 else 0)

    x = np.arange(len(tilts))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - w/2, std_ratios, w, label='Standard', color='steelblue')
    ax.bar(x + w/2, cor_ratios, w, label='Corrected', color='#e63946')
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('Cardinal / Non-cardinal σ/μ ratio', fontsize=10)
    ax.set_xlabel('Total tilt (°)', fontsize=10)
    ax.set_title('Cardinal-angle aliasing ratio vs. detector tilt',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tilts, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{fname}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}.{{pdf,png}}")


def main():
    results, R_RING = experiment_synthetic_tilt()

    print("\n" + "=" * 60)
    print("PLOTTING")
    print("=" * 60)

    plot_tilt_comparison(results, R_RING, 'synthetic_tilt_comparison')
    plot_tilt_zoom(results, R_RING, 'synthetic_tilt_zoom')
    plot_improvement_summary(results, 'synthetic_tilt_improvement')

    # Generate paper-specific figures with the names referenced by LaTeX
    import shutil
    # paper_cake_no_tilt = the no_tilt case cake
    src = os.path.join(FIG_DIR, 'synthetic_cake_no_tilt.pdf')
    dst = os.path.join(FIG_DIR, 'paper_cake_no_tilt.pdf')
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  Copied → paper_cake_no_tilt.pdf")

    # paper_ieta_comparison: I(η) at cardinal vs non-cardinal for no_tilt
    if 'no_tilt' in results:
        _plot_paper_ieta_comparison(results['no_tilt'], R_RING, 'paper_ieta_comparison')

    # paper_ratio_barchart: cardinal/non-cardinal ratio vs tilt
    _plot_paper_ratio_barchart(results, 'paper_ratio_barchart')

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Case':<20s} {'η':>4s}  {'σ/μ std':>9s}  {'σ/μ cor':>9s}  "
          f"{'Reduction':>10s}")
    print("-" * 60)
    for case_name, ty_deg, tz_deg in TILT_CASES:
        if case_name not in results:
            continue
        cr = results[case_name]
        for ca in CARDINALS:
            if 'standard' in cr and 'corrected' in cr:
                ms = cr['standard']['metrics']['cardinal'].get(ca)
                mc = cr['corrected']['metrics']['cardinal'].get(ca)
                if ms and mc:
                    red = 100 * (1 - mc['sigma_mu']/ms['sigma_mu']) \
                          if ms['sigma_mu'] > 0 else 0
                    print(f"{case_name:<20s} {ca:+4d}°  "
                          f"{ms['sigma_mu']:8.2f}%  {mc['sigma_mu']:8.2f}%  "
                          f"{red:9.1f}%")


if __name__ == '__main__':
    main()
