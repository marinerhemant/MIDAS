"""fit_setup: drop-in replacement for ``FitSetupZarr``.

End-to-end:

1. Read the 24-col ``Radius_*.csv`` and project into the C ``SpotsInfo`` layout.
2. Apply the per-spot omega correction (from t_int / t_gap / OmegaStep / Z_pix).
3. Sort spots within each ring by Omega (stable, secondary key = original index)
   and renumber SpotID consecutively across all rings.
4. Apply tilt + distortion (lab µm) via ``transform.apply_tilt_distortion``.
5. Optional wedge correction (full or no-op based on ``|wedge|<1e-10``).
6. Spot filter: ``MinEta``, ``OmegaRanges``, ``BoxSizes``, ``RingToIndex``,
   ``MaxOmeSpotIDsToIndex`` / ``MinOmeSpotIDsToIndex``.
7. Optional 5-param refine via ``midas_calibrate.refine_geometry``.
8. Write all output files: ``InputAll.csv``, ``InputAllExtraInfoFittingAll.csv``,
   ``IDRings.csv``, ``IDsHash.csv``, ``SpotsToIndex.csv``, ``paramstest.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import csv as csv_io
from ..params import ParamsTest, ZarrParams, write_paramstest
from .transform import (
    apply_tilt_distortion, calc_eta_angle_local,
    correct_wedge_full, correct_wedge_no_op,
)

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass
class FitSetupResult:
    """In-memory result of the fit_setup stage."""

    spots_inputall: torch.Tensor                  # (N, 8) float64 (matches InputAll.csv)
    extra: torch.Tensor                           # (N, 18) float64 (matches InputAllExtraInfoFittingAll.csv)
    spot_ids_to_index: torch.Tensor               # (M,) int64
    paramstest: Optional[ParamsTest] = None
    refine_result: Optional[object] = None        # RefineParams or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_hkls_for_rings(hkls_path: Path, ring_numbers: List[int]):
    """Read hkls.csv → (theta_deg per ring, ring_radius_px per ring, m_hkl per ring)."""
    thetas = [0.0] * len(ring_numbers)
    radii = [0.0] * len(ring_numbers)
    counts = [0] * len(ring_numbers)
    if not hkls_path.exists():
        return thetas, radii, counts
    with open(hkls_path, "r") as f:
        f.readline()
        for line in f:
            tokens = line.split()
            if len(tokens) < 11:
                continue
            try:
                # hkls.csv schema: h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
                #   tokens[3]=D-spacing, [4]=RingNr, [8]=Theta(deg), [9]=2Theta, [10]=Radius.
                rn = int(tokens[4])
                two_theta = float(tokens[9])
                rrad = float(tokens[10])
            except (ValueError, IndexError):
                continue
            for i, target in enumerate(ring_numbers):
                if rn == target:
                    thetas[i] = two_theta / 2.0
                    radii[i] = rrad
                    counts[i] += 1
                    break
    return thetas, radii, counts


def _radius_csv_to_spotsinfo(radius_arr: np.ndarray) -> np.ndarray:
    """Map the 24-col Radius_*.csv to the 10-col C ``SpotsInfo`` layout.

    SpotsInfo columns (per ``FitSetupParamsAllZarr.c:1305-1314``):
        [0] SpotID, [1] Omega, [2] YCen px, [3] ZCen px, [4] RingNumber,
        [5] GrainRadius, [6] IntegratedIntensity, [7] RawSumIntensity,
        [8] maskTouched, [9] FitRMSE.
    """
    n = radius_arr.shape[0]
    out = np.zeros((n, 10), dtype=np.float64)
    out[:, 0] = radius_arr[:, 0]    # SpotID
    out[:, 1] = radius_arr[:, 2]    # Omega
    out[:, 2] = radius_arr[:, 3]    # YCen px
    out[:, 3] = radius_arr[:, 4]    # ZCen px
    out[:, 4] = radius_arr[:, 13]   # RingNumber
    out[:, 5] = radius_arr[:, 15]   # GrainRadius
    out[:, 6] = radius_arr[:, 1]    # IntegratedIntensity
    out[:, 7] = radius_arr[:, 21]   # RawSumIntensity
    out[:, 8] = radius_arr[:, 22]   # maskTouched
    out[:, 9] = radius_arr[:, 23]   # FitRMSE
    return out


def _omega_correction(
    spotsinfo: np.ndarray, t_gap: float, t_int: float, omega_step: float, nr_pixels: int
) -> np.ndarray:
    """Per-spot Omega correction (FitSetupParamsAllZarr.c:1330-1338).

    Adjusts SpotsInfo[:, 1] = Omega by:
        Omega -= (t_gap / (t_gap + t_int)) * OmegaStep * (1 - |2*ZCen - NrPixels| / NrPixels)
    Then wraps to [-180, 180].
    """
    if (t_gap + t_int) == 0:
        return spotsinfo
    factor = t_gap / (t_gap + t_int) * omega_step
    z = spotsinfo[:, 3]
    delta = factor * (1.0 - np.abs(2 * z - nr_pixels) / max(nr_pixels, 1))
    spotsinfo[:, 1] = spotsinfo[:, 1] - delta
    spotsinfo[:, 1] = np.where(spotsinfo[:, 1] < -180, spotsinfo[:, 1] + 360, spotsinfo[:, 1])
    spotsinfo[:, 1] = np.where(spotsinfo[:, 1] > 180, spotsinfo[:, 1] - 360, spotsinfo[:, 1])
    return spotsinfo


def _sort_per_ring_renumber(
    spotsinfo: np.ndarray, ring_numbers: List[int],
):
    """Sort within each ring by Omega (stable, secondary key = original index)
    and renumber SpotID across all rings.

    Returns:
        sorted_info : (n, 10) — same columns, with col[0] now the new SpotID.
        per_ring_count : list[int] — number of spots per ring (in ``ring_numbers`` order).
        id_rings_rows : list[(ring_nr, original_id, new_id)] — for IDRings.csv.
    """
    parts = []
    per_ring_count = []
    id_rings_rows = []
    new_id = 1
    for rn in ring_numbers:
        mask = spotsinfo[:, 4].astype(np.int64) == rn
        sub = spotsinfo[mask].copy()
        if sub.shape[0] == 0:
            per_ring_count.append(0)
            continue
        # lexsort: primary = Omega, secondary = original index for stability
        original_idx = np.flatnonzero(mask)
        order = np.lexsort((original_idx, sub[:, 1]))
        sub = sub[order]
        for j in range(sub.shape[0]):
            id_rings_rows.append((rn, int(sub[j, 0]), new_id + j))
        sub[:, 0] = np.arange(new_id, new_id + sub.shape[0])
        new_id += sub.shape[0]
        per_ring_count.append(sub.shape[0])
        parts.append(sub)
    if not parts:
        return np.empty((0, 10), dtype=np.float64), per_ring_count, id_rings_rows
    return np.concatenate(parts, axis=0), per_ring_count, id_rings_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_setup(
    result_folder: Union[str, Path] = ".",
    *,
    zarr_params: Optional[ZarrParams] = None,
    radius_csv: Optional[Union[str, Path]] = None,
    radius_array: Optional[np.ndarray] = None,
    hkls_path: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    start_nr: int = 1,
    end_nr: Optional[int] = None,
    do_fit: Optional[bool] = None,
    write: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> FitSetupResult:
    """Run the per-spot transform + filter + paramstest stage."""
    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)

    if zarr_params is None:
        raise ValueError(
            "fit_setup requires zarr_params (parsed from the Zarr archive)."
        )

    if end_nr is None:
        end_nr = zarr_params.EndNr if zarr_params.EndNr > 0 else 1

    if radius_array is None:
        if radius_csv is None:
            radius_csv = rf / f"Radius_StartNr_{start_nr}_EndNr_{end_nr}.csv"
        radius_array = csv_io.read_radius_csv(radius_csv)

    if hkls_path is None:
        hkls_path = rf / "hkls.csv"

    # Project into 10-col SpotsInfo.
    spotsinfo = _radius_csv_to_spotsinfo(np.asarray(radius_array, dtype=np.float64))

    # Apply omega correction.
    spotsinfo = _omega_correction(
        spotsinfo,
        t_gap=zarr_params.tGap,
        t_int=zarr_params.tInt,
        omega_step=zarr_params.OmegaStep,
        nr_pixels=zarr_params.NrPixels if zarr_params.NrPixels > 0 else 1,
    )

    # Determine ring iteration order from RingThresh.
    ring_numbers = [int(rn) for (rn, _) in zarr_params.RingThresh]
    # Read hkls.csv for theta + radii.
    thetas_per_ring, radii_per_ring, mhkl_per_ring = _load_hkls_for_rings(
        Path(hkls_path), ring_numbers,
    )

    # Sort per-ring + renumber.
    spotsinfo_sorted, per_ring_count, id_rings_rows = _sort_per_ring_renumber(
        spotsinfo, ring_numbers,
    )
    n = spotsinfo_sorted.shape[0]

    # Build per-spot ring index (0-based into ring_numbers list).
    ring_idx_per_spot = np.zeros(n, dtype=np.int64)
    for i, rn in enumerate(ring_numbers):
        ring_idx_per_spot[spotsinfo_sorted[:, 4].astype(np.int64) == rn] = i

    # Move tensors to device for the differentiable transform.
    Y_pix = torch.from_numpy(spotsinfo_sorted[:, 2]).to(device=dev, dtype=dt)
    Z_pix = torch.from_numpy(spotsinfo_sorted[:, 3]).to(device=dev, dtype=dt)
    Lsd_t = torch.tensor(zarr_params.Lsd, dtype=dt, device=dev)
    BCy_t = torch.tensor(zarr_params.YCen, dtype=dt, device=dev)
    BCz_t = torch.tensor(zarr_params.ZCen, dtype=dt, device=dev)
    tx_t = torch.tensor(zarr_params.tx, dtype=dt, device=dev)
    ty_t = torch.tensor(zarr_params.ty, dtype=dt, device=dev)
    tz_t = torch.tensor(zarr_params.tz, dtype=dt, device=dev)
    p_t = torch.tensor(
        [getattr(zarr_params, f"p{i}") for i in range(15)],
        dtype=dt, device=dev,
    )
    px_t = torch.tensor(zarr_params.PixelSize, dtype=dt, device=dev)
    rho_d_t = torch.tensor(
        zarr_params.RhoD if zarr_params.RhoD > 0 else zarr_params.MaxRingRad,
        dtype=dt, device=dev,
    )

    # Optional refine.
    do_fit_active = do_fit if do_fit is not None else (zarr_params.DoFit == 1)
    refine_out = None
    if do_fit_active:
        from .refine import refine_5param
        # Build ring d-spacings (theta -> 2θ -> d via Bragg).
        # 2θ_deg is what we have per ring; use directly for the LM solve.
        ring_d = np.zeros(len(ring_numbers), dtype=np.float64)
        for i, th in enumerate(thetas_per_ring):
            two_theta = 2 * th
            if zarr_params.Wavelength > 0 and two_theta > 0:
                ring_d[i] = zarr_params.Wavelength / (2 * math.sin(th * _DEG2RAD))
        ring_d_t = torch.from_numpy(ring_d).to(device=dev, dtype=dt)
        ring_2t_t = torch.tensor(
            [2 * t for t in thetas_per_ring], dtype=dt, device=dev,
        )
        refine_out = refine_5param(
            Y_pix, Z_pix,
            ring_idx=torch.from_numpy(ring_idx_per_spot).to(device=dev, dtype=torch.int64),
            ring_d_spacing=ring_d_t,
            ring_two_theta_deg=ring_2t_t,
            Lsd=zarr_params.Lsd, BC_y=zarr_params.YCen, BC_z=zarr_params.ZCen,
            tx=zarr_params.tx, ty=zarr_params.ty, tz=zarr_params.tz,
            p_coeffs=tuple(getattr(zarr_params, f"p{i}") for i in range(15)),
            px=zarr_params.PixelSize, rho_d=float(rho_d_t.item()),
            tol_lsd=zarr_params.tolLsd, tol_bc=zarr_params.tolBC,
            tol_tilts=zarr_params.tolTilts,
            device=dev.type, dtype=dt,
        )
        # Update geometry from refine result for downstream transform.
        Lsd_t = torch.tensor(refine_out.Lsd, dtype=dt, device=dev)
        BCy_t = torch.tensor(refine_out.BC_y, dtype=dt, device=dev)
        BCz_t = torch.tensor(refine_out.BC_z, dtype=dt, device=dev)
        ty_t = torch.tensor(refine_out.ty, dtype=dt, device=dev)
        tz_t = torch.tensor(refine_out.tz, dtype=dt, device=dev)

    # Tilt + distortion.
    Y_lab, Z_lab = apply_tilt_distortion(
        Y_pix, Z_pix,
        Lsd=Lsd_t, BC_y=BCy_t, BC_z=BCz_t,
        tx=tx_t, ty=ty_t, tz=tz_t,
        p_coeffs=p_t, px=px_t, rho_d=rho_d_t,
        residual_corr_map=None,  # TODO: load from zarr_params.ResidualCorrectionMap
    )

    # Wedge correction (per-spot, branchless).
    omega_t = torch.from_numpy(spotsinfo_sorted[:, 1]).to(device=dev, dtype=dt)
    wedge_t = torch.tensor(zarr_params.Wedge, dtype=dt, device=dev)
    wl_t = torch.tensor(zarr_params.Wavelength, dtype=dt, device=dev)
    if abs(zarr_params.Wedge) < 1e-10:
        Y_w, Z_w, omega_w, eta_w, tth_w = correct_wedge_no_op(
            Y_lab, Z_lab, Lsd_t, omega_t,
        )
    else:
        Y_w, Z_w, omega_w, eta_w, tth_w = correct_wedge_full(
            Y_lab, Z_lab, Lsd_t, omega_t, wl_t, wedge_t,
        )

    # Filtering.
    eta_np = eta_w.detach().cpu().numpy()
    omega_np = omega_w.detach().cpu().numpy()
    Y_w_np = Y_w.detach().cpu().numpy()
    Z_w_np = Z_w.detach().cpu().numpy()

    min_eta = zarr_params.MinEta
    in_eta_band = (
        ((eta_np > -180 + min_eta) & (eta_np < -min_eta))
        | ((eta_np > min_eta) & (eta_np < 180 - min_eta))
    )
    keep = in_eta_band.copy()
    if zarr_params.OmegaRanges and zarr_params.BoxSizes:
        in_box = np.zeros(n, dtype=bool)
        for omr, bx in zip(zarr_params.OmegaRanges, zarr_params.BoxSizes):
            mm = (
                (omega_np >= omr[0]) & (omega_np <= omr[1])
                & (Y_w_np > bx[0]) & (Y_w_np < bx[1])
                & (Z_w_np > bx[2]) & (Z_w_np < bx[3])
            )
            in_box |= mm
        keep &= in_box

    # SpotsToIndex (RingToIndex + omega range filter).
    ring_to_index = zarr_params.OverallRingToIndex
    spots_to_index_mask = (
        keep
        & (spotsinfo_sorted[:, 4].astype(np.int64) == int(ring_to_index))
        & (omega_np >= zarr_params.MinOmeSpotIDsToIndex)
        & (omega_np <= zarr_params.MaxOmeSpotIDsToIndex)
    )
    spots_to_index_ids = spotsinfo_sorted[spots_to_index_mask, 0].astype(np.int64)

    # Build the InputAll 8-col layout, with rejected spots zeroed out
    # (matches C: rejected rows still appear with 0s except SpotID).
    inputall = np.zeros((n, 8), dtype=np.float64)
    inputall_extra = np.zeros((n, 18), dtype=np.float64)
    spot_id_col = spotsinfo_sorted[:, 0]
    inputall[:, 4] = spot_id_col
    inputall_extra[:, 4] = spot_id_col

    if keep.any():
        # Cols 0..7 = (YLab, ZLab, Omega, GrainRadius, SpotID, RingNumber, Eta, Ttheta)
        inputall[keep, 0] = Y_w_np[keep]
        inputall[keep, 1] = Z_w_np[keep]
        inputall[keep, 2] = omega_np[keep]
        inputall[keep, 3] = spotsinfo_sorted[keep, 5]   # GrainRadius
        inputall[keep, 5] = spotsinfo_sorted[keep, 4]   # RingNumber
        inputall[keep, 6] = eta_np[keep]
        inputall[keep, 7] = tth_w.detach().cpu().numpy()[keep]

        inputall_extra[keep, :8] = inputall[keep, :]
        # Col 8: OmegaIni (pre-wedge)
        inputall_extra[keep, 8] = spotsinfo_sorted[keep, 1]
        # Col 9, 10: YOrigDetCor, ZOrigDetCor (post-tilt-distortion, pre-wedge)
        inputall_extra[keep, 9] = Y_lab.detach().cpu().numpy()[keep]
        inputall_extra[keep, 10] = Z_lab.detach().cpu().numpy()[keep]
        # Col 11, 12: YOrigNoWedge, ZOrigNoWedge (raw px)
        inputall_extra[keep, 11] = spotsinfo_sorted[keep, 2]
        inputall_extra[keep, 12] = spotsinfo_sorted[keep, 3]
        # Col 13, 14, 15, 16, 17 (per FitSetupParamsAllZarr.c:1551-1559)
        inputall_extra[keep, 13] = spotsinfo_sorted[keep, 1]   # OmegaOrig (DetCor) — same as OmegaIni for now
        inputall_extra[keep, 14] = spotsinfo_sorted[keep, 6]   # IntegratedIntensity
        inputall_extra[keep, 15] = spotsinfo_sorted[keep, 7]   # RawSumIntensity
        inputall_extra[keep, 16] = spotsinfo_sorted[keep, 8]   # maskTouched
        inputall_extra[keep, 17] = spotsinfo_sorted[keep, 9]   # FitRMSE

    # Build the canonical paramstest view ONCE so the in-memory result
    # (returned to ``Pipeline``) and the on-disk paramstest.txt are identical.
    # ``zarr_params.to_paramstest()`` carries the unfiltered RingThresh ring
    # numbers and no RingRadii; the binner requires per-ring RingRadii > 0 to
    # assign spots to bins (else Data.bin is empty). Mirror the disk-write
    # path: restrict RingNumbers to the rings actually present after filtering
    # and populate RingRadii (px) from hkls.csv.
    pt = zarr_params.to_paramstest()
    if refine_out is not None:
        pt.Lsd = refine_out.Lsd
    unique_rings = sorted(set(int(r) for r in spotsinfo_sorted[keep, 4]))
    pt.RingNumbers = list(unique_rings)
    radii_lookup = {rn: rr for rn, rr in zip(ring_numbers, radii_per_ring)}
    pt.RingRadii = [radii_lookup.get(rn, 0.0) for rn in unique_rings]

    # Write outputs.
    if write:
        csv_io.write_inputall_csv(out_dir / "InputAll.csv", inputall)
        csv_io.write_inputall_extra_csv(out_dir / "InputAllExtraInfoFittingAll.csv", inputall_extra)
        # IDRings.csv: header + tuples
        with open(out_dir / "IDRings.csv", "w") as f:
            f.write("RingNumber OriginalID NewID(RingsMerge)\n")
            for (rn, oid, nid) in id_rings_rows:
                f.write(f"{rn} {oid} {nid}\n")
        # IDsHash.csv: per-ring (ring_nr, start_row, end_row, ds).
        # The d-spacing column is read by midas-process-grains for the Kenesei
        # per-spot strain gauge (ds_0). It must be the real ring d-spacing
        # d = λ/(2·sinθ) — not a 0 placeholder, or every spot is dropped and
        # the strain comes out exactly 0.
        lam = float(zarr_params.Wavelength)
        ds_per_ring = [
            (lam / (2.0 * math.sin(t * _DEG2RAD))) if (lam > 0 and t > 0) else 0.0
            for t in thetas_per_ring
        ]
        with open(out_dir / "IDsHash.csv", "w") as f:
            start_row = 1
            for rn, count, dval in zip(ring_numbers, per_ring_count, ds_per_ring):
                f.write(f"{rn} {start_row} {start_row + count + 1} {dval}\n")
                start_row += count
        # SpotsToIndex.csv
        csv_io.write_spots_to_index(out_dir / "SpotsToIndex.csv", spots_to_index_ids.tolist())
        # paramstest.txt — written from the canonical ``pt`` built above
        # (RingNumbers restricted to filtered rings, RingRadii populated).
        write_paramstest(pt, out_dir / "paramstest.txt")

    # Tensor outputs (for Pipeline).
    inputall_t = torch.from_numpy(inputall).to(device=dev, dtype=dt)
    inputall_extra_t = torch.from_numpy(inputall_extra).to(device=dev, dtype=dt)
    spots_to_index_t = torch.from_numpy(spots_to_index_ids).to(device=dev)

    return FitSetupResult(
        spots_inputall=inputall_t,
        extra=inputall_extra_t,
        spot_ids_to_index=spots_to_index_t,
        paramstest=pt,
        refine_result=refine_out,
    )
