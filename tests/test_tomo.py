#!/usr/bin/env python3
"""
MIDAS Tomography Reconstruction Benchmark Test

Generates a synthetic Shepp-Logan phantom, computes projections via the
Radon transform (scikit-image), converts them to simulated raw detector data
(with dark/white normalization), and reconstructs using the full MIDAS_TOMO
pipeline via the run_tomo() Python API.

The reconstruction quality is validated via Pearson correlation against
the ground-truth phantom (robust to FBP ringing and scaling) and absolute
centre-value accuracy.

Usage:
    python test_tomo.py
    python test_tomo.py -nCPUs 8
    python test_tomo.py --keep-work-dir
    python test_tomo.py --phantom-size 256
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate MIDAS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent

sys.path.insert(0, str(MIDAS_HOME / "TOMO"))
from midas_tomo_python import run_tomo, run_tomo_from_sinos


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------
def shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate a Shepp-Logan phantom.  Returns float32 in [0, 1]."""
    ellipses = [
        ( 1.0,   0.6900, 0.9200,  0.0000,  0.0000,    0),
        (-0.8,   0.6624, 0.8740,  0.0000, -0.0184,    0),
        (-0.2,   0.1100, 0.3100,  0.2200,  0.0000,  -18),
        (-0.2,   0.1600, 0.4100, -0.2200,  0.0000,   18),
        ( 0.1,   0.2100, 0.2500,  0.0000,  0.3500,    0),
        ( 0.1,   0.0460, 0.0460,  0.0000,  0.1000,    0),
        ( 0.1,   0.0460, 0.0460,  0.0000, -0.1000,    0),
        ( 0.1,   0.0460, 0.0230, -0.0800, -0.6050,    0),
        ( 0.1,   0.0230, 0.0230,  0.0000, -0.6060,    0),
        ( 0.1,   0.0230, 0.0460,  0.0600, -0.6050,    0),
    ]

    img = np.zeros((size, size), dtype=np.float64)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xg, yg = np.meshgrid(x, y)

    for intensity, a, b, x0, y0, theta_deg in ellipses:
        cos_t = np.cos(np.radians(theta_deg))
        sin_t = np.sin(np.radians(theta_deg))
        xr = cos_t * (xg - x0) + sin_t * (yg - y0)
        yr = -sin_t * (xg - x0) + cos_t * (yg - y0)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        img[mask] += intensity

    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def radon_transform(image: np.ndarray, thetas_deg: np.ndarray) -> np.ndarray:
    """Compute Radon transform using scikit-image (with scipy fallback)."""
    try:
        from skimage.transform import radon
        sino = radon(image, theta=thetas_deg, circle=True)
        return sino.T.astype(np.float32)
    except ImportError:
        from scipy.ndimage import rotate
        N = image.shape[0]
        sinogram = np.zeros((len(thetas_deg), N), dtype=np.float32)
        for i, theta in enumerate(thetas_deg):
            rotated = rotate(image, theta, reshape=False, order=1)
            sinogram[i, :] = np.sum(rotated, axis=0)
        return sinogram


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two images."""
    a64 = a.ravel().astype(np.float64)
    b64 = b.ravel().astype(np.float64)
    a64 -= a64.mean()
    b64 -= b64.mean()
    num = np.dot(a64, b64)
    den = np.sqrt(np.dot(a64, a64) * np.dot(b64, b64))
    return float(num / den) if den > 0 else 0.0


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalise image to [0, 1]."""
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)


def crop_to_phantom(recon_slice: np.ndarray, phantom_size: int) -> np.ndarray:
    """Crop power-of-2 padded reconstruction back to phantom size.

    Also applies a 90-degree rotation to correct for the angle-convention
    difference between skimage.radon and MIDAS_TOMO/gridrec.
    """
    xDimNew = recon_slice.shape[0]
    offset = (xDimNew - phantom_size) // 2
    cropped = recon_slice[offset:offset+phantom_size, offset:offset+phantom_size]
    return np.rot90(cropped, 1)


# ---------------------------------------------------------------------------
# Test runner
def plot_results(phantom, sinogram, recon_full, recon_sino, thetas):
    """Show phantom, sinogram, and both reconstructions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(phantom, cmap='gray', origin='lower')
    axes[0, 0].set_title('Phantom (ground truth)')

    extent = [thetas[0], thetas[-1], 0, sinogram.shape[1]]
    axes[0, 1].imshow(sinogram, cmap='gray', aspect='auto', extent=extent)
    axes[0, 1].set_title('Sinogram')
    axes[0, 1].set_xlabel('Angle (deg)')
    axes[0, 1].set_ylabel('Detector pixel')

    im2 = axes[1, 0].imshow(recon_full, cmap='gray', origin='lower')
    axes[1, 0].set_title('Recon: Full pipeline (run_tomo)')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(recon_sino, cmap='gray', origin='lower')
    axes[1, 1].set_title('Recon: Sinogram pipeline')
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
def run_test(n_cpus: int = 4, phantom_size: int = 256,
             n_thetas: int = 1800, keep_work: bool = False,
             plot: bool = False) -> bool:
    """Run the tomography benchmark.

    Metrics validated:
        - Pearson correlation > 0.85 (structural match, robust to FBP ringing)
        - Centre value within 10% of phantom centre (quantitative accuracy)
    """
    CORR_THRESHOLD = 0.85
    CENTRE_TOL = 0.10   # relative tolerance on centre pixel value

    print("=" * 70)
    print("  MIDAS Tomography Reconstruction Benchmark")
    print("=" * 70)
    print(f"  Phantom size: {phantom_size} x {phantom_size}")
    print(f"  Projections:  {n_thetas}")
    print(f"  CPUs:         {n_cpus}")
    print()

    # 1. Generate phantom
    print("[1/6] Generating Shepp-Logan phantom...", flush=True)
    phantom = shepp_logan_phantom(phantom_size)
    c = phantom_size // 2
    phantom_centre_val = float(phantom[c, c])
    print(f"  Shape: {phantom.shape}, range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    print(f"  Centre pixel value: {phantom_centre_val:.4f}")

    # 2. Compute sinograms
    print("[2/6] Computing Radon transform projections...", flush=True)
    thetas = np.linspace(0, 360 - 360.0 / n_thetas, n_thetas)
    t0 = time.time()
    sinogram = radon_transform(phantom, thetas)
    print(f"  Sinogram shape: {sinogram.shape}, computed in {time.time()-t0:.1f}s")
    print(f"  Sinogram range: [{sinogram.min():.1f}, {sinogram.max():.1f}]")

    # 3. Convert to raw detector data
    print("[3/6] Preparing simulated detector data...", flush=True)
    I0 = 30000.0
    dark_level = 100.0
    mu_scale = 3.0 / (sinogram.max() + 1e-6)
    projections = I0 * np.exp(-sinogram * mu_scale) + dark_level
    projections = np.clip(projections, 0, 65535).astype(np.uint16)

    xDim = phantom_size
    nSlices = 2
    proj_3d = np.stack([projections, projections], axis=1)
    data = np.concatenate([proj_3d[:2], proj_3d], axis=0)
    dark = np.full((nSlices, xDim), dark_level, dtype=np.float32)
    whites = np.full((2, nSlices, xDim), I0 + dark_level, dtype=np.float32)

    print(f"  data shape:   {data.shape} (nThetas+2, nSlices, xDim)")
    print(f"  Projection range: [{projections.min()}, {projections.max()}]")

    work_dir = Path(tempfile.mkdtemp(prefix='midas_tomo_test_'))
    print(f"  Work dir: {work_dir}")

    pass_full = False
    pass_sino = False

    try:
        # 4. Full pipeline (run_tomo)
        print("\n[4/6] Reconstructing via run_tomo() (full pipeline)...", flush=True)
        recon_full = run_tomo(
            data, dark, whites,
            workingdir=str(work_dir / "full"),
            thetas=thetas, shifts=0.0, filterNr=2, doLog=1,
            numCPUs=n_cpus, doCleanup=0,
        )
        print(f"  Reconstruction shape: {recon_full.shape}")
        recon_cropped = crop_to_phantom(recon_full[0, 0], phantom_size)
        recon_norm = normalize_image(recon_cropped)
        recon_centre = float(recon_norm[c, c])

        corr = pearson_correlation(phantom, recon_norm)
        centre_err = abs(recon_centre - phantom_centre_val) / (phantom_centre_val + 1e-9)
        corr_ok = corr > CORR_THRESHOLD
        centre_ok = centre_err < CENTRE_TOL
        pass_full = corr_ok and centre_ok

        status = "PASS \u2705" if pass_full else "FAIL \u274c"
        print(f"\n  [{status}]  Full pipeline")
        print(f"    Correlation:  {corr:.4f}  (threshold > {CORR_THRESHOLD})"
              f"  {'PASS' if corr_ok else 'FAIL'}")
        print(f"    Centre value: {recon_centre:.4f} vs {phantom_centre_val:.4f}"
              f"  (err={centre_err:.1%}, tol {CENTRE_TOL:.0%})"
              f"  {'PASS' if centre_ok else 'FAIL'}")

        # 5. Sinogram pipeline
        print("\n[5/6] Reconstructing via run_tomo_from_sinos()...", flush=True)
        recon_sino = run_tomo_from_sinos(
            sinogram * mu_scale,
            workingdir=str(work_dir / "sino"),
            thetas=thetas, shifts=0.0, filterNr=2, doLog=0,
            numCPUs=n_cpus, doCleanup=0,
        )
        print(f"  Reconstruction shape: {recon_sino.shape}")
        sino_cropped = crop_to_phantom(recon_sino[0, 0], phantom_size)
        sino_norm = normalize_image(sino_cropped)
        sino_centre = float(sino_norm[c, c])

        corr_s = pearson_correlation(phantom, sino_norm)
        centre_err_s = abs(sino_centre - phantom_centre_val) / (phantom_centre_val + 1e-9)
        corr_ok_s = corr_s > CORR_THRESHOLD
        centre_ok_s = centre_err_s < CENTRE_TOL
        pass_sino = corr_ok_s and centre_ok_s

        status = "PASS \u2705" if pass_sino else "FAIL \u274c"
        print(f"\n  [{status}]  Sinogram pipeline")
        print(f"    Correlation:  {corr_s:.4f}  (threshold > {CORR_THRESHOLD})"
              f"  {'PASS' if corr_ok_s else 'FAIL'}")
        print(f"    Centre value: {sino_centre:.4f} vs {phantom_centre_val:.4f}"
              f"  (err={centre_err_s:.1%}, tol {CENTRE_TOL:.0%})"
              f"  {'PASS' if centre_ok_s else 'FAIL'}")

        # 6. Cross-check: both pipelines should agree closely
        cross_corr = pearson_correlation(recon_norm, sino_norm)
        print(f"\n  Cross-pipeline correlation: {cross_corr:.4f}")

        # Optional visualisation
        if plot:
            plot_results(phantom, sinogram, recon_norm, sino_norm, thetas)

        # 6. Verdict
        print("\n[6/6] Final verdict")
        print("=" * 70)
        if pass_full and pass_sino:
            print("  \u2705 BOTH PIPELINES PASSED")
        else:
            if not pass_full:
                print("  \u274c Full pipeline FAILED")
            if not pass_sino:
                print("  \u274c Sinogram pipeline FAILED")
        print("=" * 70)

    finally:
        if not keep_work:
            print(f"\nCleaning up: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\nWork directory preserved: {work_dir}")

    return pass_full and pass_sino


def main():
    parser = argparse.ArgumentParser(
        description='MIDAS Tomography Reconstruction Benchmark Test')
    parser.add_argument('-nCPUs', type=int, default=4)
    parser.add_argument('--phantom-size', type=int, default=256)
    parser.add_argument('--n-thetas', type=int, default=1800,
                        help='Number of projection angles (default 1800)')
    parser.add_argument('--keep-work-dir', action='store_true')
    parser.add_argument('--plot', action='store_true',
                        help='Show phantom and reconstruction plots')
    args = parser.parse_args()

    success = run_test(
        n_cpus=args.nCPUs,
        phantom_size=args.phantom_size,
        n_thetas=args.n_thetas,
        keep_work=args.keep_work_dir,
        plot=args.plot,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
