"""Tolerance-mode sinogram assembly.

Pure-Python port of :c:func:`generate_sinograms`
(findSingleSolutionPFRefactored.c:1286–1752). For each (grain g, spot s,
scan c) cell, scan ``Spots.bin`` for spots that match the grain's known
spot signature (ring_nr equal, omega/eta within tolerance) and keep the
maximum intensity. Then sort spots within each grain by mean omega,
write the raw + 3 normalized variants + per-cell spotID and metadata.

Output filenames (all under ``output_dir``):

  - ``sinos_<nG>_<maxH>_<nS>.bin``         (main)
  - ``sinos_raw_<nG>_<maxH>_<nS>.bin``
  - ``sinos_norm_<nG>_<maxH>_<nS>.bin``
  - ``sinos_abs_<nG>_<maxH>_<nS>.bin``
  - ``sinos_normabs_<nG>_<maxH>_<nS>.bin``
  - ``omegas_<nG>_<maxH>.bin``
  - ``nrHKLs_<nG>.bin``
  - ``spotMapping_<nG>_<maxH>_<nS>.bin``
  - ``spotMeta_<nG>_<maxH>_<nS>.bin``

All arrays are written contiguously C-order (numpy default). float64 for
sinos/omegas/spotMeta, int32 for spotMapping and nrHKLs.

The torch path is provided for downstream differentiable use; the
intensity-fill is intrinsically a max-pool over irregular spot lookups,
so we run that on host but expose the post-sort-and-normalize stages as
torch-friendly. Returns torch tensors when the caller asks via
``backend="torch"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False


SPOTS_ARRAY_COLS = 10  # match _spot_association


@dataclass
class SinogenOutputs:
    """Path + shape bookkeeping for :func:`generate_sinograms_tolerance`."""

    n_grains: int
    max_n_hkls: int
    n_scans: int
    sino_paths: dict        # variant → path
    omegas_path: str
    nr_hkls_path: str
    spot_map_path: str
    spot_meta_path: str


def _apply_variant(sino: np.ndarray, max_int: np.ndarray, *, normalize: bool, abs_transform: bool) -> np.ndarray:
    """Apply normalize / abs_transform (= exp(-x)) to a sino array.

    Matches the C combo loop at lines 1682–1714. Acts in-place on a
    copy; mutates only cells where ``sino > 0``.
    """
    out = sino.copy()
    n_g, n_h, n_s = out.shape
    for g in range(n_g):
        for h in range(n_h):
            mi = max_int[g, h]
            for c in range(n_s):
                v = out[g, h, c]
                if v > 0:
                    if normalize and mi > 0:
                        v = v / mi
                    if abs_transform:
                        v = float(np.exp(-v))
                    out[g, h, c] = v
    return out


# ---------------------------------------------------------------------------
# Concentration filter — drop contaminated ("vertical stripe") sino rows.
# ---------------------------------------------------------------------------


def sinogram_concentration(
    raw_sino: np.ndarray,
    ome_arr: np.ndarray,
    nr_hkls_per_grain: np.ndarray,
    *,
    scan_positions: Optional[np.ndarray] = None,
    min_band_um: float = 4.0,
) -> np.ndarray:
    """Per-row fraction of intensity lying on the grain's own sinusoid.

    A clean reflection puts all of its intensity in the few scan
    positions where the beam actually crosses the grain, so in the
    ``(scan position, reflection)`` sinogram it is a compact blob riding
    a sinusoid. A *contaminated* row — one that also collected a
    neighbouring grain's spot, or a spot from outside the scanned field
    — smears across all scan positions and reads as a vertical stripe.

    For grain ``g`` this fits the offset-free sinusoid
    ``s(ω) = a·sin ω + b·cos ω`` to the intensity-weighted centroid of
    every populated row (least squares, all rows), estimates the grain
    width ``D`` from the median second moment (``D = sqrt(12·var)``, the
    equivalent width of a uniform chord), and returns

        concentration = (intensity within ±max(D, min_band_um)/2 of the
                         fitted track) / (total row intensity)

    Parameters
    ----------
    raw_sino : ndarray (nG, nH, nS) float64
        The *raw* sino — before normalize / abs transforms.
    ome_arr : ndarray (nG, nH) float64
        Mean omega per row, degrees. Rows at the ``-10000`` sentinel are
        unpopulated and are excluded by the ``total > 0`` test anyway.
    nr_hkls_per_grain : ndarray (nG,) int
    scan_positions : ndarray (nS,) float, optional
        Scan positions in micrometres, in sinogram-column order (i.e.
        :attr:`._geom.ScanGrid.spatial_positions`). Re-centred on the
        midpoint of the scan, because the fit has no ``y0`` term and so
        places the rotation axis at ``s = 0``. If ``None``, centred bin
        indices are used and ``min_band_um`` is read as a bin count.
    min_band_um : float
        Floor on the acceptance band, so that a grain whose width
        estimate collapses (single-voxel grains, or rows with one
        populated cell) still gets a physically sensible window.

    Returns
    -------
    conc : ndarray (nG, nH) float64
        Concentration in ``[0, 1]``; ``NaN`` for rows with no intensity
        and for grains with too few populated rows to fit.

    Notes
    -----
    The track is fitted using *all* populated rows, contaminated ones
    included — measured on 20-ID ``pf_nf709`` set A, one pass is enough
    to separate the stripes (they are 1.7 % of rows). Iterating the fit
    after dropping would change the numbers the threshold was calibrated
    against, so it is deliberately not done here.

    **The scan must be centred on the rotation axis.** The model has no
    ``y0`` term, and ``a·sin ω + b·cos ω`` cannot represent a constant
    over a full rotation, so a mis-centred scan displaces the whole
    fitted track and depresses *every* row's concentration equally —
    clean rows included. Measured on the synthetic clean grain of
    ``tests/unit/find_grains/test_sinogen_concentration.py`` (5 µm blob,
    51 scan positions, 40 rows), sweeping the axis offset:

    ======  ========  =============================
    offset  min conc  clean rows below 0.35 (of 40)
    ======  ========  =============================
    0 µm       0.90    0
    3 µm       0.54    0
    5 µm       0.20   40
    8 µm       0.01   40
    ======  ========  =============================

    So an offset comparable to the grain width discards the entire
    grain, and the filter cannot tell that from real contamination. The
    ``LOG.info`` drop percentage in :func:`write_clean_variant` is the
    check: a rate far above the ~2 % this was calibrated for means a
    centring problem, not dirty data. Adding a ``y0`` column to the
    design matrix would absorb it, at the cost of re-calibrating the
    threshold.
    """
    raw_sino = np.asarray(raw_sino, dtype=np.float64)
    ome_arr = np.asarray(ome_arr, dtype=np.float64)
    n_g, n_h, n_s = raw_sino.shape

    if scan_positions is None:
        s_vals = np.arange(n_s, dtype=np.float64) - (n_s - 1) / 2.0
    else:
        s_vals = np.asarray(scan_positions, dtype=np.float64).ravel()
        if s_vals.size != n_s:
            raise ValueError(
                f"scan_positions has {s_vals.size} entries but the sino has "
                f"{n_s} scan columns"
            )
        s_vals = s_vals - 0.5 * (s_vals.min() + s_vals.max())

    conc = np.full((n_g, n_h), np.nan, dtype=np.float64)
    for g in range(n_g):
        n = int(nr_hkls_per_grain[g])
        if n <= 0:
            continue
        inten = np.clip(raw_sino[g, :n, :], 0.0, None)
        tot = inten.sum(axis=1)
        ok = tot > 0
        if int(ok.sum()) < 5:
            # Too few rows to fit a 2-parameter sinusoid meaningfully.
            continue
        safe = np.maximum(tot, 1e-12)
        centroid = (inten * s_vals).sum(axis=1) / safe
        w = np.radians(ome_arr[g, :n])
        design = np.column_stack([np.sin(w), np.cos(w)])
        sol, *_ = np.linalg.lstsq(design[ok], centroid[ok], rcond=None)
        track = design @ sol
        var = (inten * (s_vals[None, :] - centroid[:, None]) ** 2).sum(axis=1) / safe
        width = float(np.median(np.sqrt(12.0 * np.clip(var[ok], 0.0, None))))
        half = max(width, min_band_um) / 2.0
        band = np.abs(s_vals[None, :] - track[:, None]) <= half
        row_conc = (inten * band).sum(axis=1) / safe
        conc[g, :n] = np.where(ok, row_conc, np.nan)
    return conc


def apply_concentration_filter(
    raw_sino: np.ndarray, conc: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero every sino row whose concentration falls below ``threshold``.

    Returns ``(clean_sino, dropped)`` where ``dropped`` is a (nG, nH)
    boolean mask of the rows that were zeroed. Rows with ``NaN``
    concentration (unpopulated, or an unfittable grain) are kept — the
    filter only ever removes rows it has positive evidence against.
    """
    dropped = np.isfinite(conc) & (conc < float(threshold))
    clean = np.array(raw_sino, dtype=np.float64, copy=True)
    clean[dropped] = 0.0
    return clean, dropped


# ---------------------------------------------------------------------------
# Occupancy — how much of the scan line each grain lights up.
# ---------------------------------------------------------------------------


def sinogram_occupancy(
    raw_sino: np.ndarray,
    nr_hkls_per_grain: np.ndarray,
    *,
    rel_threshold: float = 0.05,
) -> np.ndarray:
    """Median fraction of scan positions a grain's rows actually light up.

    A grain that fits inside the scanned field is crossed by the beam
    over a chord, so each row is populated over part of the scan line
    and dark elsewhere. A grain comparable to or larger than the field
    is crossed almost everywhere, so its rows approach full occupancy.

    Returns ``(n_grains,)`` in ``[0, 1]``, ``NaN`` for grains with no
    populated rows. Cells above ``rel_threshold`` times the row maximum
    count as lit.

    This is a **diagnostic, not a filter**. Two things follow from the
    number and neither is "drop it":

    - Occupancy is what the concentration filter is blind to. The
      concentration band adapts to the grain's *own* width, so a grain
      that fills the field gets a wide band and scores as perfectly
      concentrated. Measured on 20-ID ``pf_nf709`` set A: the two
      out-of-field grains have occupancy 0.84 / 0.78 against ≤ 0.51 for
      the other eight, yet their concentration is 0.94 / 0.93 median
      with nothing below 0.68. No concentration threshold reaches them.
    - Excluding high-occupancy grains from the reconstruction's
      grain-ID competition makes the map **much worse**, not better.
      Preregistered test on the same data: excluding them took
      agreement with the point-by-point map from 47.8 % to 11.0 %,
      because the largest grain is most of the material and its voxels
      then go to whichever small grain wins by default. Fallback-only
      (29.8 %) and volume-weighted (31.2 %) rules also lost. Use this to
      flag which grain *shapes* are untrustworthy, and leave the
      assignment alone.
    """
    raw_sino = np.asarray(raw_sino, dtype=np.float64)
    n_g, n_h, n_s = raw_sino.shape
    occ = np.full(n_g, np.nan, dtype=np.float64)
    for g in range(n_g):
        n = int(nr_hkls_per_grain[g])
        if n <= 0:
            continue
        rows = np.clip(raw_sino[g, :n, :], 0.0, None)
        peak = rows.max(axis=1)
        ok = peak > 0
        if not ok.any():
            continue
        lit = rows[ok] > (rel_threshold * peak[ok][:, None])
        occ[g] = float(np.median(lit.sum(axis=1) / n_s))
    return occ


def write_occupancy(
    out_dir: Path, raw_sino: np.ndarray, nr_hkls_per_grain: np.ndarray,
    *, rel_threshold: float = 0.05,
) -> str:
    """Emit ``sinoOccupancy_<nG>.bin``; return its path."""
    occ = sinogram_occupancy(raw_sino, nr_hkls_per_grain,
                             rel_threshold=rel_threshold)
    fn = out_dir / f"sinoOccupancy_{raw_sino.shape[0]}.bin"
    fn.write_bytes(occ.astype(np.float64, copy=False).tobytes())
    return str(fn)


def write_clean_variant(
    out_dir: Path,
    raw_sino: np.ndarray,
    ome_arr: np.ndarray,
    nr_hkls_per_grain: np.ndarray,
    *,
    conc_threshold: float,
    scan_positions: Optional[np.ndarray] = None,
    min_band_um: float = 4.0,
) -> tuple[str, str]:
    """Emit ``sinos_clean_*.bin`` + ``sinoConc_*.bin``; return their paths.

    The clean variant is the *raw* sino with contaminated rows zeroed —
    no normalize, no abs. Zeroed rows are indistinguishable from
    never-populated ones downstream, which is what the reconstructors
    already expect.
    """
    from .._logging import LOG

    n_g, n_h, n_s = raw_sino.shape
    conc = sinogram_concentration(
        raw_sino, ome_arr, nr_hkls_per_grain,
        scan_positions=scan_positions, min_band_um=min_band_um,
    )
    clean, dropped = apply_concentration_filter(raw_sino, conc, conc_threshold)

    n_rows = int(np.isfinite(conc).sum())
    n_drop = int(dropped.sum())
    LOG.info(
        "sinogen: concentration filter at %.2f dropped %d/%d rows (%.1f %%)",
        conc_threshold, n_drop, n_rows,
        100.0 * n_drop / n_rows if n_rows else 0.0,
    )

    clean_fn = out_dir / f"sinos_clean_{n_g}_{n_h}_{n_s}.bin"
    conc_fn = out_dir / f"sinoConc_{n_g}_{n_h}.bin"
    clean_fn.write_bytes(clean.astype(np.float64, copy=False).tobytes())
    conc_fn.write_bytes(conc.astype(np.float64, copy=False).tobytes())
    return str(clean_fn), str(conc_fn)


def generate_sinograms_tolerance(
    spot_list,
    n_unique: int,
    all_spots: np.ndarray,
    *,
    n_scans: int,
    tol_ome: float,
    tol_eta: float,
    output_dir: str | Path,
    scan_to_spatial: Optional[np.ndarray] = None,
    normalize_sino: bool = False,
    abs_transform: bool = False,
    # --- P7: soft sino assembly ---
    soft_weight_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    emit_softsum: bool = False,
    # --- concentration filter ---
    conc_threshold: float = 0.0,
    conc_min_band_um: float = 4.0,
    scan_positions: Optional[np.ndarray] = None,
) -> SinogenOutputs:
    """Build sinograms in tolerance mode and write all output files.

    Parameters
    ----------
    spot_list : :class:`._spot_association.SpotList`
        Output of :func:`._spot_association.process_spots`.
    n_unique : int
        ``unique_result.n_uniques`` — number of grains.
    all_spots : ndarray (n_spots_all, 10) float64
    n_scans : int
    tol_ome, tol_eta : float (degrees)
    output_dir : path-like — where to write binaries.
    scan_to_spatial : ndarray (n_scans,) int — optional argsort mapping
        from :func:`._geom.read_positions_csv`. If ``None``, identity
        (file-order == spatial-order).
    normalize_sino, abs_transform : bool — applied to the *main*
        ``sinos_<nG>_<maxH>_<nS>.bin`` file. The 4 sinos_{raw,norm,abs,
        normabs}_*.bin files are always written regardless.

    soft_weight_fn : callable(ome_diff_deg, eta_diff_deg) -> weight, optional
        When provided OR ``emit_softsum=True``, build a parallel
        weighted-sum sino (``sinos_softsum_<nG>_<maxH>_<nS>.bin``) whose
        cells are ``Σ_s w(s) · IntegratedIntensity(s)`` rather than the
        ``max(IntegratedIntensity)`` of the standard variants.  This
        preserves contributions from multiple overlapping spots and is
        the input to the per-voxel V-map refinement (P4 / P8).

        If ``soft_weight_fn is None`` and ``emit_softsum=True``, weights
        default to 1.0 (pure sum-pool).  Signature expects vectorised
        numpy inputs and returns a same-shape array of weights in
        ``[0, 1]``.

        For typical use, pair this with
        ``midas_index.compute.soft_attribution.soft_gaussian_fn`` (etc.)
        — wrap it to convert (omega_diff, eta_diff) to a 1-D distance
        first, e.g.::

            from midas_index.compute.soft_attribution import soft_gaussian_fn
            sgn = soft_gaussian_fn(fwhm_um=1.0)        # 1° gaussian
            fn = lambda od, ed: sgn(np.sqrt(od**2 + ed**2))

    emit_softsum : bool
        Force-write the softsum file even when no ``soft_weight_fn`` is
        provided (uniform weights = pure sum-pool).  Default off.

    conc_threshold : float
        When > 0, also emit ``sinos_clean_<nG>_<maxH>_<nS>.bin``: the raw
        sino with every row whose on-sinusoid concentration falls below
        this value zeroed. ``0.0`` (default) disables the filter, so the
        set of emitted files is unchanged. See
        :func:`sinogram_concentration`.
    conc_min_band_um : float
        Floor on the concentration acceptance band, micrometres.
    scan_positions : ndarray (n_scans,) float, optional
        Scan positions in micrometres in *sinogram-column* order
        (``ScanGrid.spatial_positions``). Used only by the concentration
        filter; without it the filter falls back to centred bin indices.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spots = np.ascontiguousarray(all_spots, dtype=np.float64)
    n_spots_all = int(all_spots.shape[0])
    if scan_to_spatial is None:
        scan_to_spatial = np.arange(n_scans, dtype=np.int64)
    else:
        scan_to_spatial = np.ascontiguousarray(scan_to_spatial, dtype=np.int64)

    spots = spot_list.spots
    max_n_hkls = int(spot_list.max_n_hkls)

    # nrHKLsPerGrain: max(spotNr+1) over each grain's spots.
    nr_hkls_per_grain = np.zeros(n_unique, dtype=np.int32)
    for sd in spots:
        if 0 <= sd.grain_nr < n_unique:
            nr_hkls_per_grain[sd.grain_nr] = max(
                nr_hkls_per_grain[sd.grain_nr], sd.spot_nr + 1
            )

    if max_n_hkls == 0:
        raise RuntimeError(
            "generate_sinograms: max_n_hkls is 0 — no spots in spot_list. "
            "Refusing to write empty sinos."
        )

    sz_shape = (n_unique, max_n_hkls, n_scans)
    sino = np.zeros(sz_shape, dtype=np.float64)
    spot_id_arr = np.full(sz_shape, -1, dtype=np.int32)
    spot_meta = np.full(sz_shape + (4,), np.nan, dtype=np.float64)
    max_int = np.zeros((n_unique, max_n_hkls), dtype=np.float64)
    sum_ome = np.zeros((n_unique, max_n_hkls), dtype=np.float64)
    count_ome = np.zeros((n_unique, max_n_hkls), dtype=np.int64)
    ome_arr = np.full((n_unique, max_n_hkls), -10000.0, dtype=np.float64)

    # P7: soft-sum sino accumulator (parallel to the max-pool ``sino`` above).
    # Cells are Σ_s w(s) · I(s) — preserves multi-spot evidence the max-pool drops.
    soft_active = (soft_weight_fn is not None) or emit_softsum
    softsum_sino: Optional[np.ndarray] = (
        np.zeros(sz_shape, dtype=np.float64) if soft_active else None
    )

    # Pre-extract scanNr / ringNr / intensity / omega / eta for fast loop.
    s_scanNr = all_spots[:, 9].astype(np.int64)
    s_ring = all_spots[:, 5].astype(np.int64)
    s_omega = all_spots[:, 2]
    s_eta = all_spots[:, 6]
    s_intensity = all_spots[:, 3]
    s_theta = all_spots[:, 7]
    s_y = all_spots[:, 0]
    s_z = all_spots[:, 1]
    s_id = all_spots[:, 4].astype(np.int64)

    # The C kernel parallelizes over scanNr; we do the same logical loop
    # serially but vectorize the "match against ALL unique spots for this
    # scan's spots" portion below. Iterate one scan at a time.
    # For each unique spot, gather matching spotIdx via boolean masking.
    # n_unique * max_n_hkls is small for typical PF runs (≤ thousands).
    # Pre-group spot_list by grain for faster access.
    spot_by_idx = list(spots)

    for sd in spot_by_idx:
        g = sd.grain_nr
        h = sd.spot_nr
        if g < 0 or g >= n_unique or h < 0 or h >= max_n_hkls:
            continue
        mask = (
            (s_ring == sd.ring_nr)
            & (np.abs(s_omega - sd.omega) < tol_ome)
            & (np.abs(s_eta - sd.eta) < tol_eta)
        )
        if not mask.any():
            continue
        idxs = np.flatnonzero(mask)
        # Pre-compute soft weights vectorised over the matching subset.
        if soft_active:
            ome_diffs = np.abs(s_omega[idxs] - sd.omega)
            eta_diffs = np.abs(s_eta[idxs] - sd.eta)
            if soft_weight_fn is not None:
                weights_arr = np.asarray(
                    soft_weight_fn(ome_diffs, eta_diffs), dtype=np.float64
                )
            else:
                weights_arr = np.ones_like(ome_diffs)
        for j, sidx in enumerate(idxs):
            scan_n = int(s_scanNr[sidx])
            if scan_n < 0 or scan_n >= n_scans:
                continue
            spatial_col = int(scan_to_spatial[scan_n])
            cur_int = float(s_intensity[sidx])
            cur_ome = float(s_omega[sidx])
            # Existing max-pool sino (back-compat — unchanged).
            if cur_int > sino[g, h, spatial_col]:
                sino[g, h, spatial_col] = cur_int
                spot_id_arr[g, h, spatial_col] = int(s_id[sidx])
                spot_meta[g, h, spatial_col, 0] = float(s_eta[sidx])
                spot_meta[g, h, spatial_col, 1] = float(s_theta[sidx] * 2.0)
                spot_meta[g, h, spatial_col, 2] = float(s_y[sidx])
                spot_meta[g, h, spatial_col, 3] = float(s_z[sidx])
            if soft_active:
                softsum_sino[g, h, spatial_col] += cur_int * float(weights_arr[j])
            if max_int[g, h] < cur_int:
                max_int[g, h] = cur_int
            if cur_int > 0:
                sum_ome[g, h] += cur_ome
                count_ome[g, h] += 1

    # Compute average omega for each (g, h).
    for g in range(n_unique):
        for h in range(max_n_hkls):
            if count_ome[g, h] > 0:
                ome_arr[g, h] = sum_ome[g, h] / count_ome[g, h]

    # Sort within each grain by omega.
    for g in range(n_unique):
        valid = np.where(ome_arr[g] > -9999.0)[0]
        if valid.size == 0:
            continue
        order = valid[np.argsort(ome_arr[g, valid], kind="stable")]
        # Build new arrays for this grain.
        new_ome = np.full(max_n_hkls, -10000.0, dtype=np.float64)
        new_sino = np.zeros((max_n_hkls, n_scans), dtype=np.float64)
        new_sid = np.full((max_n_hkls, n_scans), -1, dtype=np.int32)
        new_meta = np.full((max_n_hkls, n_scans, 4), np.nan, dtype=np.float64)
        new_softsum = (
            np.zeros((max_n_hkls, n_scans), dtype=np.float64)
            if soft_active else None
        )
        for k_new, k_old in enumerate(order):
            new_ome[k_new] = ome_arr[g, k_old]
            new_sino[k_new] = sino[g, k_old]
            new_sid[k_new] = spot_id_arr[g, k_old]
            new_meta[k_new] = spot_meta[g, k_old]
            if soft_active:
                new_softsum[k_new] = softsum_sino[g, k_old]
        # max_int needs to be re-sorted in lockstep so the normalize pass
        # uses the right per-spot maximum.
        new_maxI = np.zeros(max_n_hkls, dtype=np.float64)
        for k_new, k_old in enumerate(order):
            new_maxI[k_new] = max_int[g, k_old]
        max_int[g] = new_maxI
        ome_arr[g] = new_ome
        sino[g] = new_sino
        spot_id_arr[g] = new_sid
        spot_meta[g] = new_meta
        if soft_active:
            softsum_sino[g] = new_softsum

    # raw sino BEFORE transforms.
    raw_sino = sino.copy()

    # Main file: apply requested normalize/abs to a working copy.
    main_sino = _apply_variant(raw_sino, max_int, normalize=normalize_sino, abs_transform=abs_transform)

    # Write all output files.
    nG = n_unique
    nH = max_n_hkls
    nS = n_scans
    main_name = f"sinos_{nG}_{nH}_{nS}.bin"
    omegas_name = f"omegas_{nG}_{nH}.bin"
    hkls_name = f"nrHKLs_{nG}.bin"
    spot_map_name = f"spotMapping_{nG}_{nH}_{nS}.bin"
    spot_meta_name = f"spotMeta_{nG}_{nH}_{nS}.bin"

    (out_dir / main_name).write_bytes(main_sino.astype(np.float64, copy=False).tobytes())
    (out_dir / omegas_name).write_bytes(ome_arr.astype(np.float64, copy=False).tobytes())
    (out_dir / hkls_name).write_bytes(nr_hkls_per_grain.astype(np.int32, copy=False).tobytes())
    (out_dir / spot_map_name).write_bytes(spot_id_arr.astype(np.int32, copy=False).tobytes())
    (out_dir / spot_meta_name).write_bytes(spot_meta.astype(np.float64, copy=False).tobytes())

    sino_paths: dict[str, str] = {"main": str(out_dir / main_name)}
    variants = [
        ("raw", False, False),
        ("norm", True, False),
        ("abs", False, True),
        ("normabs", True, True),
    ]
    for label, do_norm, do_abs in variants:
        arr = _apply_variant(raw_sino, max_int, normalize=do_norm, abs_transform=do_abs)
        fn = f"sinos_{label}_{nG}_{nH}_{nS}.bin"
        (out_dir / fn).write_bytes(arr.astype(np.float64, copy=False).tobytes())
        sino_paths[label] = str(out_dir / fn)

    # P7: soft-sum variant — Σ w·I per cell instead of max(I).  Always raw
    # (no normalize/abs); downstream V-map refinement consumes this directly.
    if soft_active:
        fn = f"sinos_softsum_{nG}_{nH}_{nS}.bin"
        (out_dir / fn).write_bytes(softsum_sino.astype(np.float64, copy=False).tobytes())
        sino_paths["softsum"] = str(out_dir / fn)

    # Per-grain occupancy — always written; it is a diagnostic, costs
    # nothing, and is the only signal that flags out-of-field grains.
    sino_paths["occupancy"] = write_occupancy(
        out_dir, raw_sino, nr_hkls_per_grain,
    )

    # Concentration-filtered variant (off unless conc_threshold > 0).
    if conc_threshold and conc_threshold > 0:
        clean_path, conc_path = write_clean_variant(
            out_dir, raw_sino, ome_arr, nr_hkls_per_grain,
            conc_threshold=conc_threshold,
            scan_positions=scan_positions,
            min_band_um=conc_min_band_um,
        )
        sino_paths["clean"] = clean_path
        sino_paths["conc"] = conc_path

    return SinogenOutputs(
        n_grains=nG,
        max_n_hkls=nH,
        n_scans=nS,
        sino_paths=sino_paths,
        omegas_path=str(out_dir / omegas_name),
        nr_hkls_path=str(out_dir / hkls_name),
        spot_map_path=str(out_dir / spot_map_name),
        spot_meta_path=str(out_dir / spot_meta_name),
    )


# ---------------------------------------------------------------------------
# Torch-friendly variant pass — exposed for use in differentiable
# downstream pipelines that consume the raw sino as a tensor.
# ---------------------------------------------------------------------------


def apply_variant_torch(raw_sino, max_int, *, normalize: bool, abs_transform: bool):
    """Torch version of :func:`_apply_variant` — autograd-safe.

    Inputs:
      - ``raw_sino`` : torch.Tensor (nG, nH, nS)
      - ``max_int``  : torch.Tensor (nG, nH)

    Returns a new tensor on the same device / dtype.
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch not available")
    if not isinstance(raw_sino, torch.Tensor):
        raise TypeError("raw_sino must be a torch.Tensor")
    out = raw_sino.clone()
    pos = out > 0
    if normalize:
        # broadcast (nG, nH) → (nG, nH, nS)
        mi3 = max_int.unsqueeze(-1).expand_as(out)
        safe = torch.where(mi3 > 0, mi3, torch.ones_like(mi3))
        normalized = out / safe
        # Only replace cells where pos AND mi3>0; elsewhere leave raw.
        out = torch.where(pos & (mi3 > 0), normalized, out)
    if abs_transform:
        # exp(-out) but only where pos.
        expd = torch.exp(-out)
        out = torch.where(pos, expd, out)
    return out
