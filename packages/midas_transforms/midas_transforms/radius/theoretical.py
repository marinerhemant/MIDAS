"""Theoretical per-ring intensity + per-spot V_rel.

Replaces the empirical observed-powder normalization in the legacy
``calc_radius`` workflow with a fully theoretical reference computed from
the structure factor + Lorentz-polarization + multiplicity via
``midas-hkls``.  This is essential for pf-HEDM with only a handful of grains,
where the observed powder reference is biased (a single grain's accidental
strong reflection inflates a ring's reference, suppressing genuine signal).

Pipeline::

    theoretical_intensity_per_ring(crystal, λ, ring_2θ)        -> (R,)
    per_spot_relative_volume(spot_ring_idx, spot_intensity, I_ring) -> (N_spots,)
    aggregate_per_voxel(V_spot, voxel_idx_per_spot, n_voxels)   -> (N_vox,)
    aggregate_per_grain(V_spot, grain_idx_per_spot, n_grains)   -> (N_gr,)

All compute paths are torch-native, autograd-differentiable, and
device-portable (CPU / CUDA / MPS).  I/O wrappers live at module boundaries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from midas_hkls import Crystal
    from midas_hkls.structure_factor import CrystalTensor


__all__ = [
    "load_rings_from_hkls_csv",
    "theoretical_intensity_per_ring",
    "per_spot_relative_volume",
    "aggregate_per_voxel",
    "aggregate_per_grain",
    "refine_K_per_ring_closed_form",
    "RingTable",
    "SpotTensors",
    "load_spots_from_input_extra_info_csvs",
    # FF plumbing
    "FFGrainTensors",
    "FFSpotTensors",
    "load_ff_grains_to_tensors",
    "load_ff_spots_to_tensors",
]


# ---------------------------------------------------------------- I/O helpers


@dataclass
class RingTable:
    """Ring numbering + 2θ as torch tensors on a chosen device.

    ``ring_numbers`` is the MIDAS-assigned integer ring number (1-based in
    legacy data; we preserve that and index downstream tensors by position).
    ``two_theta_deg[i]`` is the 2θ in degrees of ring ``ring_numbers[i]``.
    """
    ring_numbers:   "torch.Tensor"      # (R,) int
    two_theta_deg:  "torch.Tensor"      # (R,) float


def load_rings_from_hkls_csv(
    hkls_csv_path: Union[str, Path],
    *,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> RingTable:
    """Parse a MIDAS ``hkls.csv`` and extract unique (ring_number, 2θ) pairs.

    The CSV layout is::

        h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
    """
    import torch

    arr = np.loadtxt(str(hkls_csv_path), comments="#", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ring_nr_all = arr[:, 4].astype(np.int64)
    ttheta_all = arr[:, 9].astype(np.float64)

    # Unique rings -- preserve the first occurrence's 2θ
    uniq, first_idx = np.unique(ring_nr_all, return_index=True)
    order = np.argsort(first_idx)
    ring_numbers = uniq[order]
    two_theta = ttheta_all[first_idx[order]]

    dt = dtype or torch.float64
    return RingTable(
        ring_numbers=torch.as_tensor(ring_numbers, dtype=torch.int64, device=device),
        two_theta_deg=torch.as_tensor(two_theta, dtype=dt, device=device),
    )


@dataclass
class SpotTensors:
    """Per-spot observable data, packed as torch tensors on one device.

    All tensors share ``device`` and (where applicable) ``dtype``.
    """
    spot_id:     "torch.Tensor"        # (N,) int   — global spotID (1-based in MIDAS)
    scan_nr:     "torch.Tensor"        # (N,) int   — origin scan number (PF only)
    ring_number: "torch.Tensor"        # (N,) int   — MIDAS RingNumber column
    ring_idx:    "torch.Tensor"        # (N,) int   — index into RingTable
    intensity:   "torch.Tensor"        # (N,) float — IntegratedIntensity (col 14)
    omega_deg:   "torch.Tensor"        # (N,) float
    eta_deg:     "torch.Tensor"        # (N,) float


def load_spots_from_input_extra_info_csvs(
    workdir: Union[str, Path],
    *,
    ring_table: RingTable,
    n_scans: Optional[int] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> SpotTensors:
    """Load all ``InputAllExtraInfoFittingAll<scan>.csv`` files in workdir.

    Columns (legacy MIDAS, header line starts with %):
        YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni ...
        ... IntegratedIntensity(count) RawSumIntensity maskTouched FitRMSE

    We extract: SpotID(4), RingNumber(5), Eta(6), Omega(2),
    IntegratedIntensity(14). Per-scan files are concatenated; scan_nr is
    inferred from the filename.
    """
    import torch

    work = Path(workdir)
    # Build ring_number -> ring_idx lookup
    ring_to_idx = {int(rn): i for i, rn in enumerate(ring_table.ring_numbers.tolist())}

    # Collect all per-scan files
    all_rows = []
    scan_pattern = "InputAllExtraInfoFittingAll*.csv"

    def _scan_key(p) -> int:
        # PF flavor: InputAllExtraInfoFittingAll<scan>.csv (numeric suffix)
        # FF flavor: InputAllExtraInfoFittingAll.csv (no suffix) ⇒ scan 0
        suffix = p.stem.replace("InputAllExtraInfoFittingAll", "")
        if suffix == "":
            return 0
        try:
            return int(suffix)
        except ValueError:
            # Skip anything weird (e.g., suffixes with non-numeric chars).
            return 10**9
    files = sorted(work.glob(scan_pattern), key=_scan_key)
    # Filter out the numeric-overflow placeholder (suffix wasn't an int).
    files = [p for p in files if _scan_key(p) < 10**9]
    # De-dup the case where both InputAllExtraInfoFittingAll.csv and
    # InputAllExtraInfoFittingAll0.csv exist for the same FF layer.
    seen_keys: set[int] = set()
    deduped: list = []
    for p in files:
        k = _scan_key(p)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(p)
    files = deduped
    if not files:
        raise FileNotFoundError(f"No {scan_pattern} files in {work}")

    spot_ids = []
    scan_nrs = []
    rings = []
    intensities = []
    omegas = []
    etas = []

    for f in files:
        scan = _scan_key(f)
        arr = np.loadtxt(str(f), comments="%")
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        spot_ids.append(arr[:, 4].astype(np.int64))
        scan_nrs.append(np.full(arr.shape[0], scan, dtype=np.int64))
        rings.append(arr[:, 5].astype(np.int64))
        intensities.append(arr[:, 14].astype(np.float64))
        omegas.append(arr[:, 2].astype(np.float64))
        etas.append(arr[:, 6].astype(np.float64))

    if not spot_ids:
        raise ValueError(f"Loaded no spots from {work}")

    spot_id = np.concatenate(spot_ids)
    scan_nr = np.concatenate(scan_nrs)
    ring = np.concatenate(rings)
    intensity = np.concatenate(intensities)
    omega = np.concatenate(omegas)
    eta = np.concatenate(etas)

    # Map ring number -> ring_idx; spots in unknown rings get idx=-1 (excluded later)
    ring_idx = np.array(
        [ring_to_idx.get(int(r), -1) for r in ring], dtype=np.int64
    )

    dt = dtype or torch.float64
    return SpotTensors(
        spot_id=torch.as_tensor(spot_id, dtype=torch.int64, device=device),
        scan_nr=torch.as_tensor(scan_nr, dtype=torch.int64, device=device),
        ring_number=torch.as_tensor(ring, dtype=torch.int64, device=device),
        ring_idx=torch.as_tensor(ring_idx, dtype=torch.int64, device=device),
        intensity=torch.as_tensor(intensity, dtype=dt, device=device),
        omega_deg=torch.as_tensor(omega, dtype=dt, device=device),
        eta_deg=torch.as_tensor(eta, dtype=dt, device=device),
    )


# ---------------------------------------------------------------- core kernels


def theoretical_intensity_per_ring(
    crystal_t: "CrystalTensor",
    wavelength_A: "torch.Tensor",
    ring_table: RingTable,
    *,
    two_theta_max_deg: Optional[float] = None,
    two_theta_tol_deg: float = 0.05,
    polarization: float = 0.5,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """Theoretical per-ring intensity I_th[ring] = Σ (|F|²·m·Lp) over HKLs
    whose 2θ matches the ring within ``two_theta_tol_deg``.

    Parameters
    ----------
    crystal_t          : ``midas_hkls.CrystalTensor`` (torch crystal w/ symmetry).
    wavelength_A       : 0-d tensor (Å) — autograd target.
    ring_table         : ring numbering + 2θ (typically loaded from MIDAS ``hkls.csv``).
    two_theta_max_deg  : cap for ``generate_hkls``; defaults to
                         ``ring_table.two_theta_deg.max() + 1°``.
    two_theta_tol_deg  : ring assignment tolerance (HKLs farther than this
                         from a ring's 2θ are ignored).
    polarization       : Lp polarization factor (0.5 = unpolarized / standard).

    Returns
    -------
    I_ring_theory : (R,) torch tensor on requested device.  Differentiable
                    w.r.t. ``crystal_t.lattice_params``, ``wavelength_A``,
                    and other crystal_t parameters via ``intensity_from_crystal``.
    """
    import torch
    from midas_hkls import (
        Lattice,
        SpaceGroup,
        generate_hkls,
    )
    from midas_hkls.intensity import intensity_from_crystal

    # Resolve dtype/device from inputs
    out_dtype = dtype or ring_table.two_theta_deg.dtype
    out_device = device or ring_table.two_theta_deg.device

    # Determine 2θ_max for HKL generation
    if two_theta_max_deg is None:
        # Pull as Python float from the (small) ring table
        two_theta_max_deg = float(ring_table.two_theta_deg.max().item()) + 1.0

    # Reconstruct Lattice + SpaceGroup objects (lightweight CPU operation; needed
    # by midas-hkls generate_hkls API).  Detach lattice params for HKL generation
    # (HKL enumeration is non-diff'ble; |F|^2 below still gets gradients via
    # crystal_t.lattice_params).
    lat_params = crystal_t.lattice_params.detach().cpu().numpy().tolist()
    lat = Lattice(*lat_params)
    sg = SpaceGroup.from_number(int(crystal_t.space_group_number))

    wavelength_val = float(
        wavelength_A.detach().cpu().item()
        if torch.is_tensor(wavelength_A)
        else wavelength_A
    )
    refs = generate_hkls(
        sg, lat, wavelength_A=wavelength_val, two_theta_max_deg=two_theta_max_deg
    )
    if not refs:
        return torch.zeros(
            ring_table.two_theta_deg.shape[0], dtype=out_dtype, device=out_device
        )

    # Compute per-HKL theoretical intensity (m·|F|²·Lp) via midas-hkls
    _, I_per_hkl = intensity_from_crystal(
        crystal_t, refs, wavelength_A=wavelength_A, polarization=polarization,
    )                                                     # (n_refs,) — torch
    I_per_hkl = I_per_hkl.to(dtype=out_dtype, device=out_device)

    # Per-HKL 2θ as a tensor
    ref_two_theta = torch.tensor(
        [r.two_theta_deg for r in refs], dtype=out_dtype, device=out_device
    )                                                     # (n_refs,)

    # For each HKL, find the nearest ring; accept if within tolerance
    diff = (ref_two_theta.unsqueeze(1) -
            ring_table.two_theta_deg.unsqueeze(0)).abs()    # (n_refs, R)
    nearest_ring = diff.argmin(dim=1)                       # (n_refs,)
    within_tol = diff.gather(1, nearest_ring.unsqueeze(1)).squeeze(1) < two_theta_tol_deg

    # Sum intensities per ring (scatter_add — torch-native, autograd-safe)
    R = ring_table.two_theta_deg.shape[0]
    I_per_ring = torch.zeros(R, dtype=out_dtype, device=out_device)
    # Filter both I and indices by within_tol mask
    valid_I = torch.where(within_tol, I_per_hkl, torch.zeros_like(I_per_hkl))
    I_per_ring.scatter_add_(0, nearest_ring, valid_I)

    return I_per_ring


def per_spot_relative_volume(
    spot_ring_idx: "torch.Tensor",        # (N_spots,) int
    spot_intensity: "torch.Tensor",       # (N_spots,) float — observed IntegratedIntensity
    theoretical_intensity_per_ring: "torch.Tensor",   # (R,) — from above
    *,
    eps: float = 1e-30,
) -> "torch.Tensor":                       # (N_spots,) — V_rel per spot
    """V_rel[s] = spot_intensity[s] / I_ring_theory[ spot_ring_idx[s] ]

    Spots with ``ring_idx < 0`` (no matching ring) are returned as 0.
    Spots whose ring has zero theoretical intensity get 0.
    Differentiable in both inputs and in the theoretical reference.
    """
    import torch

    valid = spot_ring_idx >= 0
    safe_idx = torch.clamp(spot_ring_idx, min=0)
    I_ref = theoretical_intensity_per_ring.gather(0, safe_idx)
    # Avoid div-by-zero -- where I_ref is zero, return zero (no information)
    safe_ref = torch.where(I_ref > eps, I_ref, torch.ones_like(I_ref))
    V_rel = spot_intensity / safe_ref
    return torch.where(valid & (I_ref > eps), V_rel, torch.zeros_like(V_rel))


def _scatter_reduce_mean(
    values: "torch.Tensor",
    index: "torch.Tensor",
    n: int,
) -> "torch.Tensor":
    """Compute mean of ``values`` grouped by ``index`` (size ``n``).

    Returns a tensor of shape (n,) with the per-group mean, or 0 for empty groups.
    Autograd-safe via scatter_add.
    """
    import torch

    valid = index >= 0
    safe_idx = torch.clamp(index, min=0)
    weights = torch.where(valid,
                          torch.ones_like(values),
                          torch.zeros_like(values))
    out_sum = torch.zeros(n, dtype=values.dtype, device=values.device)
    out_cnt = torch.zeros(n, dtype=values.dtype, device=values.device)
    out_sum.scatter_add_(0, safe_idx, values * weights)
    out_cnt.scatter_add_(0, safe_idx, weights)
    return torch.where(out_cnt > 0, out_sum / out_cnt.clamp(min=1.0), out_sum)


def _scatter_reduce_median(
    values: "torch.Tensor",
    index: "torch.Tensor",
    n: int,
) -> "torch.Tensor":
    """Per-group median.  Not autograd-friendly (hard pick); use only for
    reporting / non-refinement aggregation."""
    import torch

    out = torch.zeros(n, dtype=values.dtype, device=values.device)
    for g in range(n):
        sel = (index == g) & (values > 0)
        if sel.any():
            out[g] = torch.median(values[sel])
    return out


def aggregate_per_voxel(
    spot_V_rel: "torch.Tensor",       # (N_spots,)
    voxel_idx_per_spot: "torch.Tensor",   # (N_spots,) int — voxel index per spot
    n_voxels: int,
    method: str = "median",            # "median" | "mean"
) -> "torch.Tensor":                    # (n_voxels,)
    """Aggregate per-spot V_rel into per-voxel V (PF mode).

    ``method``:
      * ``"median"`` — robust to outlier spots; NOT smoothly differentiable
                       (gradient is the picked-element gradient).  Default
                       for final reporting.
      * ``"mean"``   — autograd-friendly; use for joint refinement initialization.
    """
    if method == "median":
        return _scatter_reduce_median(spot_V_rel, voxel_idx_per_spot, n_voxels)
    if method == "mean":
        return _scatter_reduce_mean(spot_V_rel, voxel_idx_per_spot, n_voxels)
    raise ValueError(f"unknown aggregation method: {method!r}")


def aggregate_per_grain(
    spot_V_rel: "torch.Tensor",
    grain_idx_per_spot: "torch.Tensor",
    n_grains: int,
    method: str = "median",
) -> "torch.Tensor":
    """Aggregate per-spot V_rel into per-grain V (FF mode).  Same interface
    as ``aggregate_per_voxel``; FF currently sums vs median is an open
    question handled at the caller level."""
    if method == "median":
        return _scatter_reduce_median(spot_V_rel, grain_idx_per_spot, n_grains)
    if method == "mean":
        return _scatter_reduce_mean(spot_V_rel, grain_idx_per_spot, n_grains)
    raise ValueError(f"unknown aggregation method: {method!r}")


# ---------------------------------------------------------------- K refinement


# ---------------------------------------------------------------- FF I/O


@dataclass
class FFGrainTensors:
    """Per-grain FF outputs from MIDAS ``Grains.csv``.

    Fields are torch tensors on a single device; integer fields are int64,
    floats follow ``dtype``.
    """
    grain_id:     "torch.Tensor"      # (G,) int — MIDAS GrainID
    position_um:  "torch.Tensor"      # (G, 3) float — X, Y, Z (µm)
    volume_um3:   "torch.Tensor"      # (G,) float — (4/3) π r³ from GrainRadius
    radius_um:    "torch.Tensor"      # (G,) float — MIDAS GrainRadius column
    confidence:   Optional["torch.Tensor"] = None  # (G,) float — Confidence column if present


def load_ff_grains_to_tensors(
    grains_csv_path: Union[str, Path],
    *,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> FFGrainTensors:
    """Load a MIDAS ``Grains.csv`` into a :class:`FFGrainTensors`.

    Delegates parsing to :func:`midas_stress.io.read_grains_csv` (the
    canonical header-driven parser shared across MIDAS).  Volume is
    derived from ``GrainRadius`` as ``(4/3) π r³``.
    """
    import math
    import torch

    from midas_stress.io import read_grains_csv

    dt = dtype or torch.float64
    g = read_grains_csv(str(grains_csv_path))
    if "grain_ids" not in g:
        raise ValueError(f"no GrainID column in {grains_csv_path}")
    if "positions" not in g:
        raise ValueError(f"no X/Y/Z columns in {grains_csv_path}")
    if "radii" not in g:
        raise ValueError(f"no GrainRadius column in {grains_csv_path}")

    gid = torch.as_tensor(np.asarray(g["grain_ids"]), dtype=torch.int64, device=device)
    pos = torch.as_tensor(np.asarray(g["positions"]), dtype=dt, device=device)
    r = torch.as_tensor(np.asarray(g["radii"]), dtype=dt, device=device)
    vol = (4.0 / 3.0) * math.pi * (r ** 3)
    conf = (
        torch.as_tensor(np.asarray(g["confidences"]), dtype=dt, device=device)
        if "confidences" in g else None
    )
    return FFGrainTensors(
        grain_id=gid, position_um=pos, volume_um3=vol, radius_um=r, confidence=conf,
    )


@dataclass
class FFSpotTensors:
    """Per-spot tensors for FF mode.

    Joins MIDAS ``SpotMatrix.csv`` (which carries ``grain_id`` per spot but
    no intensity) with ``InputAllExtraInfoFittingAll<scan>.csv`` (which
    carries ``IntegratedIntensity`` per ``SpotID``) by SpotID.
    """
    spot_id:     "torch.Tensor"       # (N,) int
    grain_id:    "torch.Tensor"       # (N,) int — MIDAS GrainID
    grain_idx:   "torch.Tensor"       # (N,) int — index into FFGrainTensors (0..G-1)
    ring_number: "torch.Tensor"       # (N,) int — MIDAS RingNr from SpotMatrix
    ring_idx:    "torch.Tensor"       # (N,) int — index into RingTable
    intensity:   "torch.Tensor"       # (N,) float — IntegratedIntensity
    omega_deg:   "torch.Tensor"       # (N,) float — Omega column
    eta_deg:     "torch.Tensor"       # (N,) float


def _parse_spotmatrix_csv(path: Path) -> dict:
    """MIDAS SpotMatrix.csv column layout::

        GrainID SpotID Omega DetectorHor DetectorVert OmeRaw Eta RingNr YLab ZLab Theta StrainError

    Header is a single line starting with '%'.
    """
    arr = np.loadtxt(str(path), skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {
        "grain_id":   arr[:, 0].astype(np.int64),
        "spot_id":    arr[:, 1].astype(np.int64),
        "omega_deg":  arr[:, 2],
        "eta_deg":    arr[:, 6],
        "ring_nr":    arr[:, 7].astype(np.int64),
    }


def load_ff_spots_to_tensors(
    spotmatrix_csv_path: Union[str, Path],
    input_extra_info_csv_path: Optional[Union[str, Path]],
    *,
    ring_table: RingTable,
    grain_table: Optional[FFGrainTensors] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> FFSpotTensors:
    """Load an FF spot table.

    The SpotMatrix gives ``GrainID``, ``SpotID``, ``Omega``, ``Eta``,
    ``RingNr`` per spot.  Intensities come from the corresponding
    ``InputAllExtraInfoFittingAll*.csv`` (col 14, ``IntegratedIntensity``)
    via a SpotID join.  Pass ``input_extra_info_csv_path=None`` to skip
    the join (intensities will be zeros — useful for geometry-only tests).

    If ``grain_table`` is provided, the returned ``grain_idx`` maps each
    spot's MIDAS GrainID to its 0-based row in the grain table.  Spots
    whose grain is not in the table get ``grain_idx = -1``.

    Spots with unknown ring number are returned with ``ring_idx = -1``.
    """
    import torch

    dt = dtype or torch.float64
    sm = _parse_spotmatrix_csv(Path(spotmatrix_csv_path))

    # SpotID -> IntegratedIntensity join.
    if input_extra_info_csv_path is None:
        intensity_np = np.zeros(sm["spot_id"].shape[0], dtype=np.float64)
    else:
        arr = np.loadtxt(str(input_extra_info_csv_path), comments="%")
        if arr.size == 0:
            intensity_np = np.zeros(sm["spot_id"].shape[0], dtype=np.float64)
        else:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            id_to_int = {
                int(arr[k, 4]): float(arr[k, 14]) for k in range(arr.shape[0])
            }
            intensity_np = np.array(
                [id_to_int.get(int(sid), 0.0) for sid in sm["spot_id"]],
                dtype=np.float64,
            )

    # Ring number -> ring index.
    ring_to_idx = {int(rn): i for i, rn in enumerate(ring_table.ring_numbers.tolist())}
    ring_idx_np = np.array(
        [ring_to_idx.get(int(r), -1) for r in sm["ring_nr"]], dtype=np.int64
    )

    # Grain id -> grain index.
    if grain_table is None:
        grain_idx_np = np.full_like(sm["grain_id"], -1, dtype=np.int64)
    else:
        gid_to_idx = {
            int(g): i for i, g in enumerate(grain_table.grain_id.tolist())
        }
        grain_idx_np = np.array(
            [gid_to_idx.get(int(g), -1) for g in sm["grain_id"]], dtype=np.int64
        )

    return FFSpotTensors(
        spot_id=torch.as_tensor(sm["spot_id"], dtype=torch.int64, device=device),
        grain_id=torch.as_tensor(sm["grain_id"], dtype=torch.int64, device=device),
        grain_idx=torch.as_tensor(grain_idx_np, dtype=torch.int64, device=device),
        ring_number=torch.as_tensor(sm["ring_nr"], dtype=torch.int64, device=device),
        ring_idx=torch.as_tensor(ring_idx_np, dtype=torch.int64, device=device),
        intensity=torch.as_tensor(intensity_np, dtype=dt, device=device),
        omega_deg=torch.as_tensor(sm["omega_deg"], dtype=dt, device=device),
        eta_deg=torch.as_tensor(sm["eta_deg"], dtype=dt, device=device),
    )


# ---------------------------------------------------------------- K refinement


def refine_K_per_ring_closed_form(
    V_voxel: "torch.Tensor",                            # (Nv,) — FIXED
    theoretical_intensity_per_ring: "torch.Tensor",     # (R,)
    spot_observed_intensity: "torch.Tensor",            # (Ns,)
    spot_ring_idx: "torch.Tensor",                       # (Ns,) int
    spot_grain_idx: "torch.Tensor",                      # (Ns,) int
    spot_scan_pos_um: "torch.Tensor",                    # (Ns,) float
    spot_omega_rad: "torch.Tensor",                      # (Ns,) float
    sample_grid,                                          # SampleGrid
    beam_profile,                                         # BeamProfile
    n_rings: int,
    *,
    scan_axis: str = "pf",
    eps: float = 1e-30,
) -> "torch.Tensor":                                      # (R,)
    """Closed-form per-ring scale via geometric-mean residual.

    Solves the 1-parameter least-squares problem in log-space, one ring
    at a time::

        log K[r] = mean over spots s in ring r of
                   [ log I_obs(s) - log I_pred_unitK(s) ]

    where ``I_pred_unitK`` is the forward-model prediction with ``K = 1``
    everywhere.  This is the maximum-likelihood estimator under
    independent log-normal observation noise — a sensible starting point
    for the joint refinement in P4.

    Spots with non-positive observed or predicted intensity are masked out.
    Rings with zero valid spots get ``K = 1`` (no update).

    Returns
    -------
    K_ring : (R,) tensor — geometric mean per ring.
    """
    import torch

    from .forward_model import predicted_spot_intensities

    K_unit = torch.ones(
        n_rings, dtype=V_voxel.dtype, device=V_voxel.device
    )
    I_pred_unit = predicted_spot_intensities(
        V_voxel, K_unit, theoretical_intensity_per_ring,
        spot_ring_idx, spot_grain_idx,
        spot_scan_pos_um, spot_omega_rad,
        sample_grid, beam_profile,
        scan_axis=scan_axis,
    )                                                   # (Ns,)

    valid = (spot_observed_intensity > eps) & (I_pred_unit > eps) & (spot_ring_idx >= 0)
    safe_obs = torch.where(valid, spot_observed_intensity,
                           torch.ones_like(spot_observed_intensity))
    safe_pred = torch.where(valid, I_pred_unit,
                            torch.ones_like(I_pred_unit))
    log_ratio = torch.log(safe_obs) - torch.log(safe_pred)

    w = valid.to(V_voxel.dtype)
    safe_idx = torch.clamp(spot_ring_idx, min=0)
    sum_logr = torch.zeros(n_rings, dtype=V_voxel.dtype, device=V_voxel.device)
    cnt = torch.zeros_like(sum_logr)
    sum_logr.scatter_add_(0, safe_idx, log_ratio * w)
    cnt.scatter_add_(0, safe_idx, w)

    log_K = torch.where(
        cnt > 0,
        sum_logr / cnt.clamp(min=1.0),
        torch.zeros_like(sum_logr),
    )
    return torch.exp(log_K)
