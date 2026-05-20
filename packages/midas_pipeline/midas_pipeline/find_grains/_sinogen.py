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
