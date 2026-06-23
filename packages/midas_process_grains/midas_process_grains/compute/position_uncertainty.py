"""Per-grain position uncertainty via :mod:`midas_propagate`.

For each final grain, we compute the data-driven (σ_X, σ_Y, σ_Z) by
inverting the Hessian of the spot-residual NLL on the 12-parameter
``(euler[3], latc[6], pos[3])`` block. This is the FROZEN-calibration
variant (assumes ``Σ_cc = 0``). The Schur-marginalised variant — which
propagates calibration uncertainty too — requires a Bayesian calibration
fit and lives downstream.

Cost is ~0.5 s/grain (jacfwd autograd for J_g, FD for J_c on the 5-param
calibration). For full datasets we recommend sampling 5-10k grains
unless you really want every grain (multi-day single-thread for 150k+).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch


__all__ = ["PerGrainSigmaResult", "compute_per_grain_position_sigma"]


@dataclass
class PerGrainSigmaResult:
    """Per-grain position covariance + scalar σ.

    All arrays are length ``n_grains`` (matches the v4 leaf row order).
    ``ok`` is False if the grain failed the Hessian computation (too few
    matched spots, ill-conditioned, etc.).
    """

    sigma_X_um: np.ndarray
    sigma_Y_um: np.ndarray
    sigma_Z_um: np.ndarray
    n_spots_matched: np.ndarray
    residual_rms_px: np.ndarray
    ok: np.ndarray


def compute_per_grain_position_sigma(
    *,
    grain_OM: np.ndarray,                  # (N, 3, 3) — consensus OMs (FZ-canonical)
    grain_pos_um: np.ndarray,              # (N, 3)
    rep_cand_idx: np.ndarray,              # (N,) — OPF row index of the grain's rep
    pk_path: Path,                          # ProcessKey.bin
    inputall_df: pd.DataFrame,              # InputAll (SpotID-indexed) with YLab,ZLab,Omega
    hkls,                                   # HklTable
    geometry,                               # midas_diffract.HEDMGeometry
    latc: np.ndarray,                       # (6,) lattice (a,b,c,α,β,γ)
    calibration_names: Sequence[str] = ("Lsd", "BC_y", "BC_z", "ty", "tz"),
    calibration_map: Optional[np.ndarray] = None,    # (n_c,) MAP values
    sigma_obs_px: float = 1.0,
    max_match_dist_px: float = 2.0,
    omega_start_deg: float = -180.0,
    omega_step_deg: float = 0.25,
    method: str = "fisher",
    device: Optional[str] = None,
    log=None,
) -> PerGrainSigmaResult:
    """Compute per-grain (σ_X, σ_Y, σ_Z) by Hessian inversion via midas_propagate.

    Parameters
    ----------
    grain_OM : (N, 3, 3) float64
        Consensus FZ-canonical orientation matrix per grain.
    grain_pos_um : (N, 3) float64
        Grain position in sample frame (µm).
    rep_cand_idx : (N,) int64
        ProcessKey row index for each grain's representative candidate
        (== OPF row index = alive_idx[local_rep] from the v4 pipeline).
    pk_path : Path
        ``Results/ProcessKey.bin``.
    inputall_df : DataFrame
        InputAll table indexed by SpotID with columns YLab, ZLab, Omega.
    hkls : HklTable
        Output of :func:`compute.hkl_ingest.read_hkls_csv`.
    geometry : HEDMGeometry
        Detector + scan geometry (used both to build the forward model
        and to convert observed YLab,ZLab to detector pixel coordinates).
    latc : (6,) float64
        Lattice parameters (a, b, c, α, β, γ) — Å and degrees.
    calibration_names, calibration_map : optional
        Calibration parameter names + MAP values forwarded to
        ``midas_propagate.per_grain_hessian_blocks``. Defaults to the
        five calibration parameters the Schur path expects. Position
        covariance does not require Σ_cc; this argument is only used to
        construct the (g, c) Hessian blocks. Pass calibration_map = MAP
        values from paramstest.
    sigma_obs_px : float
        Per-coordinate spot measurement noise (px / frame). Default 1.0
        (= typical refiner-noise scale).
    max_match_dist_px : float
        Nearest-neighbor association threshold at MAP for spot matching.
        Default 2.0 (= 2 px).
    method : 'fisher' | 'hessian'
        See ``midas_propagate.joint_nll.per_grain_hessian_blocks``.
    log : callable or None
        Verbose progress logger.

    Returns
    -------
    PerGrainSigmaResult
    """
    from midas_diffract.forward import HEDMForwardModel
    from midas_propagate.joint_nll import GrainObs, per_grain_hessian_blocks
    from midas_stress.orientation import orient_mat_to_euler

    if log is None:
        log = lambda *a, **k: None
    N = int(grain_OM.shape[0])

    # Device routing: MPS or CUDA if available, CPU otherwise
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    if device != "cpu":
        log(f"  per-grain σ: using device={device}")

    # Load ProcessKey for matched-spot lookup
    pk_rows = os.path.getsize(pk_path) // (5000 * 4)
    PK = np.memmap(pk_path, dtype=np.int32, mode="r", shape=(pk_rows, 5000))

    hkls_cart = torch.from_numpy(hkls.g_crystal.astype(np.float64))
    thetas = torch.from_numpy(np.deg2rad(hkls.theta_deg.astype(np.float64)))
    hkls_int = torch.from_numpy(np.stack(
        [hkls.h, hkls.k, hkls.l], axis=1,
    ).astype(np.int64))

    # Detector geometry from the HEDMGeometry instance
    y_BC = float(geometry.y_BC); z_BC = float(geometry.z_BC)
    px = float(geometry.px)
    if calibration_map is None:
        calibration_map = np.array([
            float(geometry.Lsd), y_BC, z_BC,
            float(geometry.ty), float(geometry.tz),
        ], dtype=np.float64)
    calibration_map = torch.from_numpy(np.asarray(calibration_map, dtype=np.float64))
    sigma_obs = torch.full((3,), float(sigma_obs_px), dtype=torch.float64)
    latc_t = torch.from_numpy(np.asarray(latc, dtype=np.float64))

    sx = np.full(N, np.nan, dtype=np.float64)
    sy = np.full(N, np.nan, dtype=np.float64)
    sz = np.full(N, np.nan, dtype=np.float64)
    n_match = np.zeros(N, dtype=np.int32)
    resid_rms = np.full(N, np.nan, dtype=np.float64)
    ok_arr = np.zeros(N, dtype=bool)

    # ── Per-grain σ is embarrassingly parallel over i. The dominant cost
    # (per_grain_hessian_blocks: jacfwd autograd + Hessian invert) is torch
    # C++ that releases the GIL, so a thread pool gives near-core-count
    # scaling with NO data duplication (PK memmap / inputall_df / hkls
    # tensors are all read-only shared). Numerics are identical to the
    # former serial loop; only the iteration is concurrent. Worker count
    # via MIDAS_PG_SIGMA_JOBS (default min(96, N)).
    import os as _os
    from concurrent.futures import ThreadPoolExecutor
    _njobs = int(_os.environ.get("MIDAS_PG_SIGMA_JOBS", "0")) or min(96, max(1, N))
    try:
        torch.set_num_threads(1)   # avoid intra-op × thread-pool oversubscription
    except Exception:
        pass

    def _one(i):
        rep = int(rep_cand_idx[i])
        if not (0 <= rep < pk_rows):
            return None
        sids = PK[rep]; sids = sids[sids != 0].astype(np.int64)
        if len(sids) < 4:
            return None
        try:
            ia_g = inputall_df.loc[sids].dropna()
        except KeyError:
            return None
        if len(ia_g) < 4:
            return None
        y_pix = y_BC - ia_g["YLab"].to_numpy(np.float64) / px
        z_pix = z_BC + ia_g["ZLab"].to_numpy(np.float64) / px
        frame = (ia_g["Omega"].to_numpy(np.float64) - omega_start_deg) / omega_step_deg
        obs_det = torch.from_numpy(np.column_stack([y_pix, z_pix, frame]))
        OM = np.asarray(grain_OM[i], dtype=np.float64).reshape(3, 3)
        euler_rad = torch.from_numpy(orient_mat_to_euler(OM).astype(np.float64))
        pos_um = torch.from_numpy(np.asarray(grain_pos_um[i], dtype=np.float64))
        grain_obs = GrainObs(
            spot_id=i, euler_rad=euler_rad, latc=latc_t,
            pos_um=pos_um, observed_detector=obs_det,
        )
        try:
            res = per_grain_hessian_blocks(
                grain_obs,
                hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
                base_geometry=geometry, scan_config=None,
                calibration_names=list(calibration_names),
                calibration_map=calibration_map,
                sigma_obs_detector=sigma_obs,
                method=method,
                max_match_dist=max_match_dist_px,
            )
        except Exception:
            return None
        H_gg = res.H_gg + 1e-9 * torch.eye(res.H_gg.shape[0], dtype=res.H_gg.dtype)
        try:
            Sigma_g = torch.linalg.inv(H_gg)
        except Exception:
            Sigma_g = torch.linalg.pinv(H_gg)
        pos_var = torch.diag(Sigma_g)[9:12].detach().cpu().numpy()
        sp = np.sqrt(np.maximum(pos_var, 0.0))
        return (i, float(sp[0]), float(sp[1]), float(sp[2]),
                int(res.n_spots_matched),
                float(torch.sqrt((res.residual_at_map ** 2).mean()).item()))

    log(f"  per-grain σ: {N:,} grains over {_njobs} threads")
    n_ok = 0; n_fail = 0; _done = 0
    with ThreadPoolExecutor(max_workers=_njobs) as _ex:
        for r in _ex.map(_one, range(N)):
            _done += 1
            if r is None:
                n_fail += 1
            else:
                i, a, b, c, nm, rr = r
                sx[i] = a; sy[i] = b; sz[i] = c
                n_match[i] = nm; resid_rms[i] = rr; ok_arr[i] = True
                n_ok += 1
            if _done % 1000 == 0:
                log(f"    σ progress: {_done:,}/{N:,}  ({n_ok} ok, {n_fail} fail)")

    return PerGrainSigmaResult(
        sigma_X_um=sx, sigma_Y_um=sy, sigma_Z_um=sz,
        n_spots_matched=n_match, residual_rms_px=resid_rms, ok=ok_arr,
    )
