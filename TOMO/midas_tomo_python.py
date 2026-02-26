"""
MIDAS Tomography Reconstruction Python Library

Provides two entry points for tomographic reconstruction using the MIDAS_TOMO
C binary (Gridrec-based filtered back-projection):

  run_tomo()            — from raw projection data (dark + 2 whites + images)
  run_tomo_from_sinos() — from pre-formed sinograms (areSinos=1 mode)

Usage examples
--------------
From raw projections::

    from midas_tomo_python import run_tomo
    recon = run_tomo(data, dark, whites, '/tmp/work', thetas, shifts=1.0)

From sinograms::

    from midas_tomo_python import run_tomo_from_sinos
    recon = run_tomo_from_sinos(sino_2d, '/tmp/work', thetas)
"""

import collections.abc
import os
import subprocess
import sys
import time
from math import ceil, log2

import numpy as np


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _next_power_of_2(n):
    """Return the smallest power of 2 >= *n*."""
    if n <= 0:
        return 1
    return 1 << int(ceil(log2(n))) if n > 1 else 1


def _find_tomo_exe():
    """Locate the MIDAS_TOMO binary."""
    try:
        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'utils')
        if utils_dir not in sys.path:
            sys.path.append(utils_dir)
        import midas_config
        return os.path.join(midas_config.MIDAS_TOMO_BIN_DIR, 'MIDAS_TOMO')
    except ImportError:
        return os.path.expanduser("~/opt/MIDAS/TOMO/bin/MIDAS_TOMO")


def _write_thetas(thetas, path):
    """Write one angle per line to *path*."""
    with open(path, 'w') as f:
        for theta in thetas:
            f.write(f'{theta}\n')


def _parse_shift_arg(shifts):
    """Return (shift_string_for_config, nrShifts)."""
    if not isinstance(shifts, collections.abc.Sequence):
        return f'{shifts} {shifts} 1', 1
    nrShifts = round(abs((shifts[1] - shifts[0])) / shifts[2]) + 1
    return f'{shifts[0]} {shifts[1]} {shifts[2]}', nrShifts


def _read_recon(outfnstr, nrShifts, nrSlices, xDimNew):
    """Read the reconstruction binary written by MIDAS_TOMO."""
    outfn = (f'{outfnstr}_NrShifts_{str(nrShifts).zfill(3)}'
             f'_NrSlices_{str(nrSlices).zfill(5)}'
             f'_XDim_{str(xDimNew).zfill(6)}'
             f'_YDim_{str(xDimNew).zfill(6)}_float32.bin')
    recon = np.fromfile(outfn, dtype=np.float32,
                        count=nrSlices * nrShifts * xDimNew * xDimNew)
    return recon.reshape((nrShifts, nrSlices, xDimNew, xDimNew)), outfn


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_tomo(data, dark, whites, workingdir, thetas,
             shifts=0.0, filterNr=2, doLog=1, extraPad=0,
             autoCentering=1, numCPUs=40, doCleanup=1, ringRemoval=0):
    """Reconstruct from raw projection data.

    Parameters
    ----------
    data : ndarray, shape (nrThetas+2, nrSlices, xDim)
        Raw projections.  The first two frames are tilt-corrected copies
        that the C code skips (nrThetas -= 2 internally).
    dark : ndarray, shape (nrSlices, xDim) or (xDim,)
        Dark-field image.
    whites : ndarray, shape (2, nrSlices, xDim) or (2, xDim)
        Two white-field images.
    workingdir : str
        Directory for temporary files and output.
    thetas : 1D array
        Rotation angles in degrees.
    shifts : float or [start, end, interval]
        Shift values for rotation-axis centering.
    filterNr : int
        0=none, 1=Shepp-Logan, 2=Hann (default), 3=Hamming, 4=Ramp.
    doLog : int
        1 to take -log of transmission, 0 to use intensities directly.
    extraPad : int
        0=half padding, 1=one-half padding.
    autoCentering : int
        1 to auto-center rotation axis.
    numCPUs : int
        Number of parallel threads.
    doCleanup : int
        1 to remove temporary files after reconstruction.
    ringRemoval : float
        Ring-removal coefficient (0 to disable).

    Returns
    -------
    recon : ndarray, shape (nrShifts, nrSlices, xDimNew, xDimNew)
        Reconstructed slices.  xDimNew is the next power of 2 >= xDim
        (doubled if extraPad=1).
    """
    start_time = time.time()
    os.makedirs(workingdir, exist_ok=True)

    nrThetas, nrSlices, xDim = data.shape
    data = data.astype(np.float32)

    # --- Pad odd slices to even (MIDAS_TOMO requirement) ---
    originalNSlices = nrSlices
    if nrSlices % 2 != 0:
        data = np.concatenate([data, data[:, -1:, :]], axis=1)
        nrSlices = data.shape[1]

    # Write binary: dark, whites, data(uint16)
    infn = os.path.join(workingdir, 'input.bin')
    with open(infn, 'wb') as f:
        dark.astype(np.float32).tofile(f)
        whites.astype(np.float32).tofile(f)
        data.astype(np.uint16).tofile(f)

    # The C code subtracts 2 from nrThetas (dark+whites preamble)
    nrThetas -= 2

    outfnstr = os.path.join(workingdir, 'output')
    xDimNew = _next_power_of_2(xDim)
    if extraPad == 1:
        xDimNew *= 2

    # Write thetas
    thetasFN = os.path.join(workingdir, 'midastomo_thetas.txt')
    _write_thetas(thetas, thetasFN)

    # Write config
    shift_str, nrShifts = _parse_shift_arg(shifts)
    configFN = os.path.join(workingdir, 'midastomo.par')
    with open(configFN, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 0\n')
        f.write(f'detXdim {xDim}\n')
        f.write(f'detYdim {nrSlices}\n')
        f.write(f'thetaFileName {thetasFN}\n')
        f.write(f'shiftValues {shift_str}\n')
        f.write(f'ringRemovalCoefficient {ringRemoval}\n')
        f.write(f'doLog {doLog}\n')
        f.write('slicesToProcess -1\n')
        f.write(f'ExtraPad {extraPad}\n')
        f.write(f'AutoCentering {autoCentering}\n')

    print(f'Time elapsed in preprocessing: {time.time() - start_time:.3f}s.')

    # Run MIDAS_TOMO
    tomo_exe = _find_tomo_exe()
    subprocess.run([tomo_exe, configFN, str(numCPUs)], check=True)

    # Read result
    start_time = time.time()
    recon, outfn = _read_recon(outfnstr, nrShifts, nrSlices, xDimNew)

    # Truncate back to original slice count
    recon = recon[:, :originalNSlices, :, :]

    if doCleanup:
        for fn in [outfn, configFN, thetasFN, infn,
                   os.path.join(workingdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'),
                   os.path.join(workingdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt')]:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    print(f'Time elapsed in postprocessing: {time.time() - start_time:.3f}s.')
    return recon


def run_tomo_from_sinos(sinograms, workingdir, thetas,
                        shifts=0.0, filterNr=2, doLog=0, extraPad=0,
                        autoCentering=1, numCPUs=1, doCleanup=1,
                        ringRemoval=0):
    """Reconstruct from pre-formed sinograms (areSinos=1 mode).

    Parameters
    ----------
    sinograms : ndarray
        Shape ``(nThetas, detXdim)`` for a single slice, or
        ``(nSlices, nThetas, detXdim)`` for multiple slices.
        Data is converted to float32 internally.
    workingdir : str
        Directory for temporary files and output.
    thetas : 1D array
        Rotation angles in degrees.
    shifts : float or [start, end, interval]
        Shift values for rotation-axis centering.
    filterNr : int
        0=none, 1=Shepp-Logan, 2=Hann (default), 3=Hamming, 4=Ramp.
    doLog : int
        0 (default for sinogram input) to use intensities directly,
        1 to apply -log.
    extraPad : int
        0=half padding, 1=one-half padding.
    autoCentering : int
        1 to auto-center rotation axis.
    numCPUs : int
        Number of parallel threads.
    doCleanup : int
        1 to remove temporary files after reconstruction.
    ringRemoval : float
        Ring-removal coefficient (0 to disable).

    Returns
    -------
    recon : ndarray, shape (nrShifts, nSlices, xDimNew, xDimNew)
        Reconstructed slices.  xDimNew is the next power of 2 >= detXdim
        (doubled if extraPad=1).
    """
    start_time = time.time()
    os.makedirs(workingdir, exist_ok=True)

    # Normalize to 3D
    sinograms = np.asarray(sinograms, dtype=np.float32)
    if sinograms.ndim == 2:
        sinograms = sinograms[np.newaxis, :, :]  # (1, nThetas, detXdim)

    nSlices, nThetas, detXdim = sinograms.shape

    # --- Pad odd slices to even (MIDAS_TOMO requirement) ---
    originalNSlices = nSlices
    if nSlices % 2 != 0:
        sinograms = np.concatenate([sinograms, sinograms[-1:, :, :]], axis=0)
        nSlices = sinograms.shape[0]

    # Write sinogram binary (flat float32, each slice is nThetas × detXdim)
    infn = os.path.join(workingdir, 'input_sino.bin')
    sinograms.tofile(infn)

    outfnstr = os.path.join(workingdir, 'output')
    xDimNew = _next_power_of_2(detXdim)
    if extraPad == 1:
        xDimNew *= 2

    # Write thetas
    thetasFN = os.path.join(workingdir, 'midastomo_thetas.txt')
    _write_thetas(thetas, thetasFN)

    # Write config
    shift_str, nrShifts = _parse_shift_arg(shifts)
    configFN = os.path.join(workingdir, 'midastomo.par')
    with open(configFN, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 1\n')
        f.write(f'detXdim {detXdim}\n')
        f.write(f'detYdim {nSlices}\n')
        f.write(f'thetaFileName {thetasFN}\n')
        f.write(f'filter {filterNr}\n')
        f.write(f'shiftValues {shift_str}\n')
        f.write(f'ringRemovalCoefficient {ringRemoval}\n')
        f.write(f'doLog {doLog}\n')
        f.write('slicesToProcess -1\n')
        f.write(f'ExtraPad {extraPad}\n')
        f.write(f'AutoCentering {autoCentering}\n')

    print(f'Time elapsed in preprocessing: {time.time() - start_time:.3f}s.')

    # Run MIDAS_TOMO
    tomo_exe = _find_tomo_exe()
    subprocess.run([tomo_exe, configFN, str(numCPUs)], check=True)

    # Read result
    start_time = time.time()
    recon, outfn = _read_recon(outfnstr, nrShifts, nSlices, xDimNew)

    # Truncate back to original slice count
    recon = recon[:, :originalNSlices, :, :]

    if doCleanup:
        for fn in [outfn, configFN, thetasFN, infn,
                   os.path.join(workingdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'),
                   os.path.join(workingdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt')]:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    print(f'Time elapsed in postprocessing: {time.time() - start_time:.3f}s.')
    return recon
