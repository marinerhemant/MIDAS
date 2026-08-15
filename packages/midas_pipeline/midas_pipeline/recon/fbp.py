"""Filtered Back-Projection reconstruction wrapper.

Backed by the ``midas-tomo`` package. The public entry point ``fbp_recon``
matches the contract expected by ``midas_pipeline.stages.reconstruct``.

This used to reach the reconstruction a different way: a ``sys.path`` hop into
``<repo>/TOMO`` to import the in-tree ``midas_tomo_python``, which then shelled
out to the ``MIDAS_TOMO_C`` / ``MIDAS_TOMO_GPU`` binary. That had two defects.

1. **It only worked from a source checkout.** The hop resolved the repo root as
   ``Path(__file__).parents[4]``, which holds for
   ``<repo>/packages/midas_pipeline/midas_pipeline/recon/fbp.py`` and is
   meaningless for an installed ``site-packages/midas_pipeline/recon/fbp.py``.
   Every pip user got an ImportError telling them to check a source tree they
   had no reason to have.
2. **It depended on a separately-built C binary**, which silently rots: the
   tree's ``TOMO/bin/MIDAS_TOMO`` was three months older than its own sources
   and aborted with SIGABRT.

``midas-tomo`` is now a real dependency, so the reconstruction is importable
wherever midas-pipeline is installed, and the FFTW/OpenMP-free build degrades
to the Python path instead of dying.

**Why the subprocess backend by default.**

OBSERVED, reproducibly: run this module's FBP under pytest and a later
``gemmi.read_small_structure`` fails with ``ValueError: vector`` on a CIF it
parses fine in isolation. Order-dependent (reverse the two tests and both
pass), not capture-dependent (``-s`` behaves the same), and it goes away
entirely with ``backend="subprocess"``. That combination is reachable in an
ordinary run -- reconstruct, then read a crystal CIF in calc_radius_v -- so the
risk is wrong results, not just a red test.

NOT PROVEN: the mechanism. The suspicion is a C++ ABI conflict --
``libmidastomo.dylib``, which the ``"library"`` backend dlopens into this
process, links GCC's ``libstdc++`` (Homebrew gcc + libgomp) while gemmi links
Apple's ``libc++``, and two C++ runtimes in one process is undefined behaviour.
The linkage difference is real (``otool -L`` on both), but a standalone script
doing the same two calls does NOT reproduce the failure, so something about the
pytest process is also required and the causal chain is unconfirmed. Treat the
above as the reason to be careful, not as a diagnosis.

Subprocess keeps the engine in its own address space, which empirically avoids
it. It costs roughly 10x per call (measured 0.002s -> 0.022s on a 16x16
phantom), and ``fbp_recon_per_grain`` calls this per grain, so ``backend`` is
exposed for callers who have established their environment is unaffected.
A proper fix -- if the ABI theory holds -- belongs in midas-tomo's build.

Note the keyword spellings differ: the legacy module took ``filterNr`` /
``doLog`` / ``extraPad`` / ``numCPUs``; the package takes ``filter_nr`` /
``do_log`` / ``extra_pad`` / ``n_cpus``. This wrapper's own signature is
unchanged, so callers are unaffected.
"""

from __future__ import annotations

import os
from math import ceil, log2
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


def _load_run_tomo_from_sinos():
    """Return ``midas_tomo.run_tomo_from_sinos``.

    Imported at call time rather than module import so that merely importing
    midas_pipeline does not require midas-tomo to be present.
    """
    try:
        from midas_tomo import run_tomo_from_sinos
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "midas_pipeline.recon.fbp requires the midas-tomo package. "
            "Install it with `pip install midas-tomo` (it is a declared "
            "dependency of midas-pipeline, so this normally means a partial "
            "or --no-deps install)."
        ) from exc
    return run_tomo_from_sinos


def fbp_recon(
    sinograms: np.ndarray,
    thetas: np.ndarray,
    workingdir: Union[str, Path],
    *,
    n_scans: Optional[int] = None,
    filter_nr: int = 2,
    do_log: int = 0,
    extra_pad: int = 0,
    auto_centering: int = 1,
    num_cpus: int = 1,
    do_cleanup: int = 1,
    use_gpu: bool = False,
    backend: str = "subprocess",
) -> np.ndarray:
    """Run FBP on a single sinogram, return a centered nScans×nScans recon.

    Thin wrapper around ``midas_tomo.run_tomo_from_sinos`` that does the
    standard pf-HEDM post-crop: the reconstruction upscales
    ``detXdim`` to the next power of two; we crop the central
    ``nScans × nScans`` region so the recon center aligns with the
    voxel-grid center ``(nScans - 1) / 2``.

    Parameters
    ----------
    sinograms : ndarray, shape (nThetas, detXdim) or (nSlices, nThetas, detXdim)
        Pre-formed sinograms.
    thetas : ndarray, shape (nThetas,)
        Projection angles in degrees.
    workingdir : str | Path
        Temp directory for the reconstruction's inputs / outputs.
    n_scans : int, optional
        If given, crop to (n_scans, n_scans). Default: full upscaled recon.
    filter_nr, do_log, extra_pad, auto_centering, num_cpus, do_cleanup, use_gpu
        Forwarded to ``run_tomo_from_sinos``; see that function's
        docstring for semantics.
    backend : {"subprocess", "library", "auto"}
        How midas-tomo runs its engine. Defaults to ``"subprocess"`` -- see
        the module docstring for why the in-process ``"library"`` backend is
        not the default here. Pass ``"library"`` explicitly to opt back in
        (~10x faster per call), but only in a process that will not also use
        a libc++-linked extension.

    Returns
    -------
    ndarray
        If ``n_scans`` is given: ``(n_scans, n_scans)``, cropped + transposed
        to match the voxel-grid spatial convention. Otherwise the
        upscaled square returned by the reconstruction.
    """
    run_tomo_from_sinos = _load_run_tomo_from_sinos()
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)

    recon_arr = run_tomo_from_sinos(
        np.asarray(sinograms, dtype=np.float32),
        str(workingdir),
        np.asarray(thetas, dtype=np.float64),
        shifts=0.0,
        filter_nr=filter_nr,
        do_log=do_log,
        extra_pad=extra_pad,
        auto_centering=auto_centering,
        n_cpus=num_cpus,
        do_cleanup=do_cleanup,
        use_gpu=use_gpu,
        backend=backend,
    )
    # recon_arr shape: (nrShifts, nSlices, xDimNew, xDimNew)
    full = recon_arr[0, 0]
    if n_scans is None:
        return full
    recon_dim = 1 << int(ceil(log2(n_scans))) if n_scans > 1 else 1
    if extra_pad == 1:
        recon_dim *= 2
    crop_start = recon_dim // 2 - n_scans // 2
    crop_end = crop_start + n_scans
    return full[crop_start:crop_end, crop_start:crop_end]


def fbp_recon_per_grain(
    sinos_by_grain: np.ndarray,
    omegas_by_grain: np.ndarray,
    n_hkls_per_grain: np.ndarray,
    n_scans: int,
    workingdir: Union[str, Path],
    *,
    filter_nr: int = 2,
    do_log: int = 0,
    extra_pad: int = 0,
    auto_centering: int = 1,
    num_cpus: int = 1,
    do_cleanup: int = 1,
    use_gpu: bool = False,
    transpose_to_voxel: bool = True,
) -> np.ndarray:
    """FBP every grain in one call, return the ``(nGrs, nScans, nScans)`` stack.

    Parameters
    ----------
    sinos_by_grain : ndarray, shape (nGrs, maxNHKLs, nScans), float64
    omegas_by_grain : ndarray, shape (nGrs, maxNHKLs), float64
        Per-grain projection angles in degrees.
    n_hkls_per_grain : ndarray, shape (nGrs,), int
        Number of valid HKLs per grain.
    n_scans : int
        Spatial dimension; used to crop the upscaled recon.
    workingdir : str | Path
        Per-grain temp directory.
    transpose_to_voxel : bool
        Apply the historical ``.T`` swap that aligns the recon with
        the voxel-grid spatial convention used everywhere else in
        pf-HEDM. True by default to match legacy.

    Returns
    -------
    ndarray, shape (nGrs, nScans, nScans), float32
    """
    n_grs = int(n_hkls_per_grain.shape[0])
    out = np.zeros((n_grs, n_scans, n_scans), dtype=np.float32)
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)
    for g in range(n_grs):
        n_sp = int(n_hkls_per_grain[g])
        if n_sp <= 0:
            continue
        thetas = np.asarray(omegas_by_grain[g, :n_sp], dtype=np.float64)
        sino = np.asarray(sinos_by_grain[g, :n_sp, :], dtype=np.float32)
        if (sino > 0).sum() == 0:
            continue
        # Drop all-zero rows before back-projecting. They carry no
        # measurement — either an unpopulated (grain, hkl) cell or a row
        # the concentration filter zeroed — and feeding them in as
        # "measured zero at this angle" dilutes the reconstruction.
        # MLEM/OSEM already do this (recon/mlem.py: ``row_has_data``).
        row_has_data = (sino > 0).any(axis=1)
        if not row_has_data.all():
            sino = sino[row_has_data]
            thetas = thetas[row_has_data]
        recon = fbp_recon(
            sino, thetas, workingdir,
            n_scans=n_scans,
            filter_nr=filter_nr, do_log=do_log, extra_pad=extra_pad,
            auto_centering=auto_centering, num_cpus=num_cpus,
            do_cleanup=do_cleanup, use_gpu=use_gpu,
        )
        if transpose_to_voxel:
            recon = recon.T
        out[g] = recon.astype(np.float32, copy=False)
    return out
