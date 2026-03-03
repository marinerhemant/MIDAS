#!/usr/bin/env python

"""
Sinogram cleanup and tomographic reconstruction for grain shape recovery.

Reads sinograms produced by pf_MIDAS.py / findSingleSolutionPFRefactored,
applies column-wise intensity normalization (streak removal) and angular
hole-filling interpolation, then reconstructs using the MIDAS_TOMO C binary
(Gridrec-based filtered back-projection).

Usage:
    python sino_cleanup_tomo.py -topDir /path/to/result -sinoType normabs
    python sino_cleanup_tomo.py -topDir . -sinoType raw -grainNrs 0,1,5
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import (median_filter, gaussian_filter, gaussian_filter1d,
                            binary_dilation, binary_opening, binary_closing,
                            generate_binary_structure)
from scipy.interpolate import griddata

# Import MIDAS_TOMO reconstruction
tomo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TOMO')
if tomo_dir not in sys.path:
    sys.path.insert(0, tomo_dir)
from midas_tomo_python import run_tomo_from_sinos


def normalize_columns(sino, min_fill_frac=0.1, window=5):
    """Normalize sinogram columns to remove vertical streak intensity variations.

    Uses a sliding-window robust normalization: each column's non-zero mean is
    divided by the local median of neighboring column means (within a window).
    This handles slowly varying intensity trends better than a single global target.
    Only columns with at least min_fill_frac of non-zero pixels are normalized.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Input sinogram.
    min_fill_frac : float
        Minimum fraction of non-zero pixels a column must have to be normalized.
    window : int
        Sliding window size for local median target (odd number recommended).

    Returns
    -------
    sino_out : ndarray
        Normalized sinogram (same shape).
    """
    sino_out = sino.copy()
    nScans, nThetas = sino_out.shape

    col_means = np.zeros(nThetas)
    col_fill = np.zeros(nThetas)
    for j in range(nThetas):
        col = sino_out[:, j]
        nz = col[col > 0]
        col_fill[j] = len(nz) / nScans
        if len(nz) > 0:
            col_means[j] = np.mean(nz)

    valid_mask = (col_means > 0) & (col_fill >= min_fill_frac)
    valid_means = col_means[valid_mask]
    if len(valid_means) == 0:
        return sino_out

    # Build local median target using sliding window
    half_w = window // 2
    local_target = np.zeros(nThetas)
    for j in range(nThetas):
        lo = max(0, j - half_w)
        hi = min(nThetas, j + half_w + 1)
        neighbors = col_means[lo:hi]
        neighbor_valid = neighbors[(neighbors > 0)]
        if len(neighbor_valid) > 0:
            local_target[j] = np.median(neighbor_valid)
        else:
            local_target[j] = np.median(valid_means)  # global fallback

    for j in range(nThetas):
        if col_means[j] > 0 and col_fill[j] >= min_fill_frac and local_target[j] > 0:
            sino_out[:, j] *= local_target[j] / col_means[j]

    return sino_out


def fill_holes(sino, max_gap=2):
    """Fill small zero-valued gaps within the sinogram signal trace.

    For each row, identifies runs of consecutive zeros. Only gaps of
    max_gap or fewer consecutive zeros that are flanked on BOTH sides
    by non-zero values are filled (by linear interpolation). This leaves
    genuinely empty regions (outside the sinusoidal trace) untouched.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Sinogram with potential holes (zeros).
    max_gap : int
        Maximum number of consecutive zeros to fill. Larger gaps are
        assumed to be genuinely empty and are left as-is.

    Returns
    -------
    sino_out : ndarray
        Sinogram with small holes filled.
    """
    sino_out = sino.copy()
    nScans, nThetas = sino_out.shape

    for i in range(nScans):
        row = sino_out[i, :]
        if not np.any(row > 0):
            continue

        # Find runs of zeros
        is_zero = (row == 0)
        if not np.any(is_zero):
            continue

        # Label consecutive zero runs
        # Find start and end of each zero run
        diff = np.diff(is_zero.astype(int))
        # starts: where diff == 1 (transition from nonzero to zero)
        starts = np.where(diff == 1)[0] + 1
        # ends: where diff == -1 (transition from zero to nonzero)
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases: row starts or ends with zeros
        if is_zero[0]:
            starts = np.concatenate([[0], starts])
        if is_zero[-1]:
            ends = np.concatenate([ends, [nThetas]])

        for s, e in zip(starts, ends):
            gap_len = e - s
            # Only fill small gaps that are flanked on both sides
            if gap_len <= max_gap and s > 0 and e < nThetas:
                # Linear interpolation between left and right neighbors
                left_val = row[s - 1]
                right_val = row[e]
                row[s:e] = np.linspace(left_val, right_val, gap_len + 2)[1:-1]

        sino_out[i, :] = row

    return sino_out


def despeckle(sino, size=3):
    """Remove isolated bright noise pixels using a median filter.

    Replaces pixels that significantly exceed their local median neighborhood,
    killing salt-and-pepper noise and false-assignment outliers while
    preserving the sinusoidal trace.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Input sinogram.
    size : int
        Median filter kernel size.

    Returns
    -------
    sino_out : ndarray
        Denoised sinogram.
    """
    med = median_filter(sino, size=size)
    sino_out = sino.copy()
    # Replace pixels that are >3x the local median (isolated speckle)
    mask = (sino > 0) & (med > 0) & (sino > 3.0 * med)
    sino_out[mask] = med[mask]
    # Kill isolated bright pixels where median is zero (noise outside trace)
    mask_isolated = (sino > 0) & (med == 0)
    sino_out[mask_isolated] = 0
    return sino_out


def smooth_columns(sino, sigma=0.7):
    """Apply light Gaussian smoothing along the column (angle) axis.

    Suppresses small-scale column-to-column noise without blurring
    the sinusoidal trace (which runs along the row axis).

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Input sinogram.
    sigma : float
        Gaussian sigma for smoothing along axis=1 (angles).

    Returns
    -------
    sino_out : ndarray
        Smoothed sinogram.
    """
    sino_out = sino.copy()
    mask = sino_out > 0
    smoothed = gaussian_filter1d(sino_out, sigma=sigma, axis=1)
    # Only apply smoothing where we had signal
    sino_out[mask] = smoothed[mask]
    return sino_out


def trace_inpaint(sino, dilation_iters=5):
    """Fill gaps within the sinusoidal trace using 2D nearest-neighbor interpolation.

    The trace region is estimated by morphological dilation of the non-zero
    pixel mask. All zero pixels within the dilated trace are then filled
    by interpolating from known (non-zero) pixels using scipy griddata.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Input sinogram with gaps.
    dilation_iters : int
        Number of binary dilation iterations to expand the trace mask.

    Returns
    -------
    sino_out : ndarray
        Sinogram with gaps within the trace filled.
    trace_mask : ndarray, shape (nScans, nThetas), dtype bool
        The dilated trace region mask.
    """
    sino_out = sino.copy()
    nScans, nThetas = sino_out.shape

    has_data = sino_out > 0
    if not np.any(has_data):
        return sino_out, has_data

    # Dilate the existing data mask to estimate the full trace envelope
    struct = generate_binary_structure(2, 2)  # 3x3 connectivity
    trace_mask = binary_dilation(has_data, structure=struct,
                                 iterations=dilation_iters)

    # Find gap pixels: in the trace but currently zero
    gap_mask = trace_mask & ~has_data
    n_gaps = np.count_nonzero(gap_mask)
    if n_gaps == 0:
        return sino_out, trace_mask

    # Interpolate gaps from known pixels using nearest-neighbor
    known_rows, known_cols = np.where(has_data)
    known_vals = sino_out[has_data]
    gap_rows, gap_cols = np.where(gap_mask)

    filled_vals = griddata(
        (known_rows, known_cols), known_vals,
        (gap_rows, gap_cols), method='nearest')

    sino_out[gap_rows, gap_cols] = filled_vals
    return sino_out, trace_mask


def smooth_2d(sino, sigma=0.8, trace_mask=None):
    """Apply 2D Gaussian smoothing within the trace region.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
    sigma : float
        Gaussian sigma for 2D smoothing.
    trace_mask : ndarray, optional
        Boolean mask of the trace region.

    Returns
    -------
    sino_out : ndarray
    """
    if trace_mask is None:
        trace_mask = sino > 0

    sino_out = sino.copy()
    smoothed = gaussian_filter(sino_out, sigma=sigma)
    sino_out[trace_mask] = smoothed[trace_mask]
    return sino_out


def fill_missing_columns(sino, min_fill_frac=0.1):
    """Interpolate missing angular projections (columns) from neighbors.

    A column is considered missing if fewer than min_fill_frac of its
    pixels are non-zero. Missing columns are replaced by linear
    interpolation (pixel-by-pixel) from the nearest non-missing columns
    on each side.

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Sinogram.
    min_fill_frac : float
        Minimum fraction of non-zero pixels for a column to be considered valid.

    Returns
    -------
    sino_out : ndarray
        Sinogram with missing columns interpolated.
    """
    sino_out = sino.copy()
    nScans, nThetas = sino_out.shape

    # Classify columns as valid or missing
    col_fill = np.array([np.count_nonzero(sino_out[:, j]) / nScans
                         for j in range(nThetas)])
    valid = col_fill >= min_fill_frac

    if np.all(valid) or not np.any(valid):
        return sino_out

    valid_indices = np.where(valid)[0]
    missing_indices = np.where(~valid)[0]

    # For each missing column, interpolate pixel-by-pixel from nearest valid neighbors
    for j in missing_indices:
        # Find nearest valid column to the left and right
        left_candidates = valid_indices[valid_indices < j]
        right_candidates = valid_indices[valid_indices > j]

        if len(left_candidates) > 0 and len(right_candidates) > 0:
            jl = left_candidates[-1]
            jr = right_candidates[0]
            # Linear interpolation weight
            w = (j - jl) / (jr - jl)
            sino_out[:, j] = (1 - w) * sino_out[:, jl] + w * sino_out[:, jr]
        elif len(left_candidates) > 0:
            sino_out[:, j] = sino_out[:, left_candidates[-1]]
        elif len(right_candidates) > 0:
            sino_out[:, j] = sino_out[:, right_candidates[0]]

    return sino_out


def reconstruct_midas(sino, thetas, workingdir, filter_nr=2,
                      ring_removal=0.0, stripe_removal=True):
    """Reconstruct a 2D slice from a sinogram using MIDAS_TOMO (Gridrec FBP).

    Parameters
    ----------
    sino : ndarray, shape (nScans, nThetas)
        Sinogram (rows = detector positions, columns = angles).
    thetas : 1D array
        Rotation angles in degrees.
    workingdir : str
        Temporary working directory for MIDAS_TOMO files.
    filter_nr : int
        Filter for FBP: 0=none, 1=Shepp-Logan, 2=Hann, 3=Hamming, 4=Ramp.
    ring_removal : float
        Ring removal coefficient (0 to disable, 1.0 typical).
    stripe_removal : bool
        Enable Vo et al. 2018 stripe removal in MIDAS_TOMO.

    Returns
    -------
    recon : ndarray, shape (nScans, nScans)
        Reconstructed slice.
    """
    nScans, nThetas = sino.shape

    # MIDAS_TOMO expects (nThetas, detXdim) for a single slice
    sino_transposed = sino.T.copy()  # (nThetas, nScans)

    # Write config manually to include stripe removal (not in wrapper API)
    import tempfile
    os.makedirs(workingdir, exist_ok=True)
    infn = os.path.join(workingdir, 'input_sino.bin')
    sino_transposed.astype(np.float32).tofile(infn)
    outfnstr = os.path.join(workingdir, 'output')

    # Next power of 2 (MIDAS_TOMO output dimension)
    xDimNew = 1
    while xDimNew < nScans:
        xDimNew <<= 1

    # Pad odd slices
    nSlices = 1
    if nSlices % 2 != 0:
        # Write 2 copies for the even-slice requirement
        combined = np.stack([sino_transposed, sino_transposed], axis=0)  # (2, nThetas, nScans)
        combined.astype(np.float32).tofile(infn)
        nSlices = 2
    else:
        sino_transposed.astype(np.float32).tofile(infn)

    # Write thetas
    thetasFN = os.path.join(workingdir, 'midastomo_thetas.txt')
    with open(thetasFN, 'w') as f:
        for theta in thetas:
            f.write(f'{theta}\n')

    configFN = os.path.join(workingdir, 'midastomo.par')
    with open(configFN, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 1\n')
        f.write(f'detXdim {nScans}\n')
        f.write(f'detYdim {nSlices}\n')
        f.write(f'thetaFileName {thetasFN}\n')
        f.write(f'filter {filter_nr}\n')
        f.write(f'shiftValues 0 0 1\n')
        if ring_removal > 0:
            f.write(f'ringRemovalCoefficient {ring_removal}\n')
        f.write(f'doLog 0\n')
        f.write('slicesToProcess -1\n')
        f.write('ExtraPad 0\n')
        f.write('AutoCentering 0\n')
        if stripe_removal:
            f.write('doStripeRemoval 1\n')

    import subprocess
    tomo_dir_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TOMO')
    utils_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    try:
        import midas_config
        tomo_exe = os.path.join(midas_config.MIDAS_TOMO_BIN_DIR, 'MIDAS_TOMO')
    except ImportError:
        tomo_exe = os.path.expanduser('~/opt/MIDAS/TOMO/bin/MIDAS_TOMO')

    subprocess.run([tomo_exe, configFN, '1'], check=True, cwd=workingdir)

    # Read result
    outfn = (f'{outfnstr}_NrShifts_001'
             f'_NrSlices_{str(nSlices).zfill(5)}'
             f'_XDim_{str(xDimNew).zfill(6)}'
             f'_YDim_{str(xDimNew).zfill(6)}_float32.bin')
    recon_data = np.fromfile(outfn, dtype=np.float32,
                             count=nSlices * xDimNew * xDimNew)
    recon_all = recon_data.reshape((1, nSlices, xDimNew, xDimNew))
    recon_full = recon_all[0, 0, :, :]  # first shift, first slice

    # Cleanup temp files (wisdom files may be in workingdir or cwd)
    cleanup_files = [outfn, configFN, thetasFN, infn]
    for wdir in [workingdir, '.']:
        cleanup_files.append(os.path.join(wdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'))
        cleanup_files.append(os.path.join(wdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt'))
    for fn in cleanup_files:
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    # Center-crop from power-of-2 output to nScans × nScans
    rsize = recon_full.shape[0]
    if rsize > nScans:
        start = (rsize - nScans) // 2
        recon = recon_full[start:start + nScans, start:start + nScans]
    elif rsize < nScans:
        pad = nScans - rsize
        recon = np.pad(recon_full, ((pad // 2, pad - pad // 2),
                                    (pad // 2, pad - pad // 2)))
    else:
        recon = recon_full

    return recon


def clean_reconstruction(recon, morph_size=3):
    """Post-reconstruction cleanup: threshold, morphological clean, normalize.

    PF-HEDM sinograms are inherently sparse and noisy, so the raw
    reconstruction will have streak artifacts and background noise.
    This function extracts the grain shape by:
    1. Clipping negative values
    2. Thresholding at the 75th percentile of positive values
    3. Morphological opening (removes small artifacts)
    4. Morphological closing (fills small holes in grain)
    5. Multiplying binary mask by smoothed reconstruction for clean output

    Parameters
    ----------
    recon : ndarray, shape (N, N)
        Raw reconstruction.
    morph_size : int
        Size of the morphological structuring element.

    Returns
    -------
    recon_clean : ndarray, shape (N, N)
        Cleaned reconstruction with grain shape extracted.
    """
    # Clip negatives
    recon_pos = np.clip(recon, 0, None)

    # Threshold to separate grain from background noise
    pos_vals = recon_pos[recon_pos > 0]
    if len(pos_vals) == 0:
        return recon_pos

    # Use 75th percentile as threshold (grain pixels are brightest)
    threshold = np.percentile(pos_vals, 75)
    grain_mask = recon_pos > threshold

    # Morphological cleanup
    struct = generate_binary_structure(2, 1)  # cross connectivity
    # Opening: remove small isolated noise clusters
    grain_mask = binary_opening(grain_mask, structure=struct, iterations=1)
    # Closing: fill small holes within grain
    grain_mask = binary_closing(grain_mask, structure=struct, iterations=1)

    # Apply smoothed reconstruction weighted by grain mask
    recon_smooth = gaussian_filter(recon_pos, sigma=1.0)
    recon_clean = recon_smooth * grain_mask.astype(np.float64)

    return recon_clean


FILTER_MAP = {0: None, 1: 'shepp-logan', 2: 'hann', 3: 'hamming', 4: 'ramp'}


def process_grain(grNr, topdir, sinoType, filterNr):
    """Process a single grain: clean sinogram and reconstruct.

    Returns
    -------
    dict with keys: grNr, nScans, nThetas, orig_fill_pct, clean_fill_pct,
                    max_intensity, recon (ndarray)
    """
    grStr = str(grNr).zfill(4)
    sino_fn = os.path.join(topdir, f'Sinos/sino_{sinoType}_grNr_{grStr}.tif')
    theta_fn = os.path.join(topdir, f'Thetas/thetas_grNr_{grStr}.txt')

    if not os.path.exists(sino_fn):
        print(f"  [SKIP] Sinogram not found: {sino_fn}")
        return None
    if not os.path.exists(theta_fn):
        print(f"  [SKIP] Thetas not found: {theta_fn}")
        return None

    # Load
    sino = np.array(Image.open(sino_fn), dtype=np.float64)
    thetas = np.loadtxt(theta_fn)
    nScans, nThetas = sino.shape

    # Stats before
    total_pixels = sino.size
    orig_nonzero = np.count_nonzero(sino)
    orig_fill_pct = 100.0 * orig_nonzero / total_pixels

    # Step 1: Despeckle — kill isolated bright noise pixels
    sino_clean = despeckle(sino)

    # Step 2: Trace-aware inpainting (fill gaps within dilated trace envelope)
    sino_clean, trace_mask = trace_inpaint(sino_clean, dilation_iters=5)

    # Step 3: Normalize columns (sliding-window, 2 passes for convergence)
    sino_clean = normalize_columns(sino_clean)
    sino_clean = normalize_columns(sino_clean)

    # Step 4: Fill remaining row-wise gaps
    sino_clean = fill_holes(sino_clean, max_gap=6)

    # Step 5: 2D Gaussian smoothing within trace
    sino_clean = smooth_2d(sino_clean, sigma=0.8, trace_mask=trace_mask)

    # Step 6: Zero out everything outside the trace (kill outer noise)
    sino_clean[~trace_mask] = 0

    # Stats after
    clean_nonzero = np.count_nonzero(sino_clean)
    clean_fill_pct = 100.0 * clean_nonzero / total_pixels
    max_intensity = np.max(sino_clean)

    # Save cleaned sinogram
    os.makedirs(os.path.join(topdir, 'Sinos'), exist_ok=True)
    Image.fromarray(sino_clean).save(
        os.path.join(topdir, f'Sinos/sino_{sinoType}_clean_grNr_{grStr}.tif'))

    # Reconstruct using MIDAS_TOMO (with ring + stripe removal)
    recon_workdir = os.path.join(topdir, f'Recons/.tomo_work_grNr_{grStr}')
    os.makedirs(recon_workdir, exist_ok=True)
    recon = reconstruct_midas(sino_clean, thetas, recon_workdir,
                              filter_nr=filterNr, ring_removal=1.0,
                              stripe_removal=True)

    # Post-reconstruction cleanup
    recon_clean = clean_reconstruction(recon)

    # Save reconstruction (both raw and cleaned)
    os.makedirs(os.path.join(topdir, 'Recons'), exist_ok=True)
    Image.fromarray(recon).save(
        os.path.join(topdir, f'Recons/recon_raw_grNr_{grStr}.tif'))
    Image.fromarray(recon_clean).save(
        os.path.join(topdir, f'Recons/recon_clean_grNr_{grStr}.tif'))

    print(f"  Grain {grNr:4d}: fill {orig_fill_pct:5.1f}% -> {clean_fill_pct:5.1f}%, "
          f"maxI={max_intensity:.2f}, recon {recon.shape[0]}x{recon.shape[1]}")

    return {
        'grNr': grNr, 'nScans': nScans, 'nThetas': nThetas,
        'orig_fill_pct': orig_fill_pct, 'clean_fill_pct': clean_fill_pct,
        'max_intensity': max_intensity, 'recon': recon_clean
    }


def main():
    parser = argparse.ArgumentParser(
        description='Sinogram cleanup and tomo reconstruction for grain shapes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-topDir', type=str, required=True,
                        help='Working directory with Sinos/ and Thetas/ folders')
    parser.add_argument('-sinoType', type=str, required=True,
                        choices=['raw', 'norm', 'abs', 'normabs'],
                        help='Sinogram type to process')
    parser.add_argument('-grainNrs', type=str, default='all',
                        help='Comma-separated grain numbers, or "all"')
    parser.add_argument('-filterNr', type=int, default=2,
                        help='Tomo filter: 0=none, 1=Shepp-Logan, 2=Hann, 3=Hamming, 4=Ramp')
    args = parser.parse_args()

    topdir = os.path.abspath(args.topDir)
    sinoType = args.sinoType

    # Discover grains
    if args.grainNrs == 'all':
        pattern = os.path.join(topdir, f'Sinos/sino_{sinoType}_grNr_*.tif')
        files = sorted(glob.glob(pattern))
        grain_nrs = []
        for f in files:
            base = os.path.basename(f)
            # Filter out cleaned sinograms
            if '_clean_' in base:
                continue
            nr_str = base.split('_grNr_')[1].replace('.tif', '')
            grain_nrs.append(int(nr_str))
    else:
        grain_nrs = [int(x.strip()) for x in args.grainNrs.split(',')]

    if not grain_nrs:
        print(f"No sinograms found for type '{sinoType}' in {topdir}/Sinos/")
        sys.exit(1)

    print(f"Processing {len(grain_nrs)} grains (type: {sinoType}) in {topdir}")
    print(f"  Filter: {FILTER_MAP.get(args.filterNr, 'hann')}")
    print("=" * 70)

    t0 = time.time()
    results = []

    for grNr in grain_nrs:
        result = process_grain(grNr, topdir, sinoType, args.filterNr)
        if result is not None:
            results.append(result)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Processed {len(results)}/{len(grain_nrs)} grains in {elapsed:.1f}s")

    # Create max-projection across all grains (using cleaned recons)
    if results:
        nScans = results[0]['nScans']
        recon_stack = np.zeros((len(results), nScans, nScans))
        for i, r in enumerate(results):
            recon_stack[i, :, :] = r['recon']

        # Threshold: only keep pixels above small positive value for max
        recon_stack[recon_stack < 0] = 0

        full_recon = np.max(recon_stack, axis=0)
        max_id = np.argmax(recon_stack, axis=0).astype(np.int32)
        # Mark background as -1 (where no grain has significant signal)
        bg_threshold = np.max(full_recon) * 0.05 if np.max(full_recon) > 0 else 0
        max_id[full_recon <= bg_threshold] = -1
        full_recon[full_recon <= bg_threshold] = 0

        Image.fromarray(full_recon).save(
            os.path.join(topdir, 'Recons/Full_recon_clean_max_project.tif'))
        Image.fromarray(max_id).save(
            os.path.join(topdir, 'Recons/Full_recon_clean_max_project_grID.tif'))

        # Save all recons as multi-page TIF
        im_list = [Image.fromarray(r['recon']) for r in results]
        if len(im_list) > 1:
            im_list[0].save(
                os.path.join(topdir, 'Recons/all_recons_clean_together.tif'),
                compression="tiff_deflate", save_all=True,
                append_images=im_list[1:])

        print(f"Saved max projection and {len(im_list)} individual recons to Recons/")

    # Summary table
    if results:
        print("\n--- Summary ---")
        print(f"{'Grain':>6s}  {'OrigFill%':>9s}  {'CleanFill%':>10s}  {'MaxIntensity':>12s}")
        for r in results:
            print(f"{r['grNr']:6d}  {r['orig_fill_pct']:9.1f}  {r['clean_fill_pct']:10.1f}  {r['max_intensity']:12.2f}")


if __name__ == '__main__':
    main()
