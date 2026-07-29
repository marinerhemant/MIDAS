"""Raw-frame geometry mini-calibration (P1-6).

Empirical-Jacobian damped Gauss-Newton: bump each geometry parameter in
the ACTUAL forward model (convention-proof — no analytic derivative can
disagree with the model), fit the parameter steps to the measured anchor
residuals with a robust (MAD-rejecting) LSQ, iterate, verify collapse.

On the SOH wt316 campaign this collapsed anchor residuals from 7.06 to
1.4 px RMS and cross-validated between independent loads (tx/BC/tz agree;
ty/Lsd/ω₀ scatter with 2-ring data). Any raw-frame consumer of a promoted
``paramstest.txt`` needs it: the powder calibration behind paramstest is
blind to tx (~0.27° ≈ 3-4e3 µε fake strain) and its convention chain
(ideal↔raw, flips, ω origin) is easy to get subtly wrong — this
calibration absorbs all of it into the model actually being used.

Typical use::

    factory = layer_model_factory(layer_dir, ring_numbers=[1, 2, 3, 4],
                                  n_pixels_y=2880, n_pixels_z=2880,
                                  n_frames=1440, apply_distortion=True,
                                  device=dev)
    ds = load_pf_grain(layer_dir, grain_id, ..., model=factory())
    cal = calibrate_raw_frame_geometry(ds, cached_patches,
                                       model_factory=factory)
    model = factory(geom_mod=cal.calibrated, omega_start=cal.calibrated["om0"])
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel

__all__ = [
    "RawFrameCalibration",
    "calibrate_raw_frame_geometry",
    "layer_model_factory",
    "measure_patch_offsets",
]

# Default parameter set + bump sizes (the empirical-derivative step per
# parameter; also the natural scale of one GN step component).
DEFAULT_BUMPS: Dict[str, float] = {
    "y_BC": 1.0,      # px
    "z_BC": 1.0,      # px
    "tx": 0.05,       # deg
    "ty": 0.05,       # deg
    "tz": 0.05,       # deg
    "Lsd": 500.0,     # um
    "om0": 0.05,      # deg
}


@dataclass
class RawFrameCalibration:
    """Result of :func:`calibrate_raw_frame_geometry`."""

    calibrated: Dict[str, float]        # fitted parameter values
    original: Dict[str, float]          # starting values
    rms_before_px: float                # sqrt(mean(dy^2+dz^2)), first iter
    rms_after_px: float                 # same, after the final iteration
    rms_frames_after: float             # RMS omega-frame residual after
    n_spots_used: int                   # spots with a measured centroid
    per_iter_rms_px: list = field(default_factory=list)

    @property
    def delta(self) -> Dict[str, float]:
        return {k: self.calibrated[k] - self.original[k]
                for k in self.calibrated}


def measure_patch_offsets(
    measured: torch.Tensor,
    *,
    signal_threshold_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-spot measured centroid offsets from a cached patch tensor.

    Parameters
    ----------
    measured : (S, Sigma, F, P, P) tensor
        The cached measured patches (``assemble_grain_patch_data`` layout).
    signal_threshold_frac : float
        A (spot, scan) cell counts as signal when its summed intensity
        exceeds this fraction of the median non-zero cell total.

    Returns
    -------
    (dy, dz, df, ok) : per-spot median centroid offsets (px, px, frames)
        relative to the patch centre, NaN where no scan had signal;
        ``ok`` = finite mask.
    """
    meas = measured.detach().to(torch.float64).cpu()
    S, _Sigma, F, P, _P2 = meas.shape
    c = P // 2
    ps = meas.sum(dim=2)                                  # (S, Sigma, P, P)
    tot = ps.sum(dim=(-1, -2))                            # (S, Sigma)
    nz = tot[tot > 0]
    thr = float(nz.median()) * signal_threshold_frac if nz.numel() else 0.0
    yy, zz = torch.meshgrid(
        torch.arange(P, dtype=torch.float64),
        torch.arange(P, dtype=torch.float64), indexing="ij",
    )
    cy = ((ps * yy).sum(dim=(-1, -2)) / (tot + 1e-12)).numpy() - c
    cz = ((ps * zz).sum(dim=(-1, -2)) / (tot + 1e-12)).numpy() - c
    fidx = torch.arange(F, dtype=torch.float64)
    pf = meas.sum(dim=(-1, -2))                           # (S, Sigma, F)
    cf = ((pf * fidx).sum(dim=-1) / (pf.sum(dim=-1) + 1e-12)).numpy() - F // 2
    has = (tot.numpy() > thr)

    dy = np.full(S, np.nan)
    dz = np.full(S, np.nan)
    df = np.full(S, np.nan)
    for s in range(S):
        g = np.nonzero(has[s])[0]
        if g.size >= 1:
            dy[s] = np.median(cy[s, g])
            dz[s] = np.median(cz[s, g])
            df[s] = np.median(cf[s, g])
    ok = np.isfinite(dy)
    return dy, dz, df, ok


def _anchors_with_model(ds, model: HEDMForwardModel, dtype, device):
    from . import io as _io
    d2 = copy.copy(ds)
    d2.model = model
    ay, az, af, _valid, _obs, S = _io._forward_anchors(d2, dtype, device)
    return (ay.detach().cpu().numpy(), az.detach().cpu().numpy(),
            af.detach().cpu().numpy(), S)


def calibrate_raw_frame_geometry(
    ds,
    measured: torch.Tensor,
    *,
    model_factory: Callable[..., HEDMForwardModel],
    anchor_model: Optional[HEDMForwardModel] = None,
    params_to_fit: Sequence[str] = tuple(DEFAULT_BUMPS),
    bumps: Optional[Dict[str, float]] = None,
    n_iters: int = 3,
    frame_weight: float = 4.0,
    damping: float = 1e-2,
    max_step_bumps: float = 40.0,
    mad_reject: float = 6.0,
    signal_threshold_frac: float = 0.2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> RawFrameCalibration:
    """Fit (y_BC, z_BC, tx, ty, tz, Lsd, ω₀) to measured patch centroids.

    Parameters
    ----------
    ds : PFGrainDataset
        The grain whose cached patches anchor the calibration.
    measured : (S, Sigma, F, P, P) tensor
        Cached measured patches, in the anchor layout of ``anchor_model``.
    model_factory : callable
        ``model_factory(geom_mod: dict | None = None, omega_start:
        float | None = None) -> HEDMForwardModel``. ``geom_mod`` maps
        parameter names (see :data:`DEFAULT_BUMPS`, minus ``om0`` which is
        passed as ``omega_start``) to absolute values. Must build the model
        the CALIBRATION should converge (e.g. distortion ON for raw-frame
        work); :func:`layer_model_factory` builds one from a layer dir.
    anchor_model : optional
        The model the cache was ASSEMBLED with (defaults to ``ds.model``).
        Absolute measured positions are ``anchors(anchor_model) + offsets``.

    Notes
    -----
    Derivatives are empirical: each parameter is bumped by ``bumps[name]``
    in a rebuilt model and the anchor shift is the Jacobian column — no
    convention (flips, ideal↔raw, ω origin) can silently disagree with
    the model. The LSQ is Levenberg-damped and MAD-rejecting; steps are
    clipped to ``max_step_bumps`` bump units.
    """
    bumps = dict(DEFAULT_BUMPS if bumps is None else bumps)
    names = [n for n in params_to_fit if n in bumps]
    if not names:
        raise ValueError(f"params_to_fit {params_to_fit!r} matches no known "
                         f"parameter (choose from {sorted(DEFAULT_BUMPS)})")

    base = anchor_model if anchor_model is not None else ds.model
    ay0, az0, af0, S = _anchors_with_model(ds, base, dtype, device)

    dy, dz, df, ok = measure_patch_offsets(
        measured, signal_threshold_frac=signal_threshold_frac)
    if int(ok.sum()) < len(names):
        raise ValueError(
            f"only {int(ok.sum())} spots have a measured centroid — fewer "
            f"than the {len(names)} parameters being fit."
        )
    abs_y = ay0 + dy
    abs_z = az0 + dz
    abs_f = af0 + df

    def _getf(model, name, fallback=0.0):
        v = getattr(model, name, fallback)
        try:
            v = v.detach().cpu().item()
        except AttributeError:
            pass
        if isinstance(v, (list, tuple)):
            v = v[0]
        return float(v)

    fit0 = model_factory()
    state: Dict[str, float] = {}
    for n in names:
        if n == "om0":
            state[n] = _getf(fit0, "omega_start")
        else:
            state[n] = _getf(fit0, n)
    original = dict(state)

    def model_at(st: Dict[str, float]) -> HEDMForwardModel:
        gm = {k: v for k, v in st.items() if k != "om0"}
        om0 = st.get("om0")
        return model_factory(geom_mod=gm, omega_start=om0)

    rms_before = None
    per_iter = []
    for it in range(n_iters):
        m_it = model_at(state)
        ayi, azi, afi, _ = _anchors_with_model(ds, m_it, dtype, device)
        ry = (abs_y - ayi)[ok]
        rz = (abs_z - azi)[ok]
        rf = (abs_f - afi)[ok]
        rms = math.sqrt(np.nanmean(ry ** 2 + rz ** 2))
        per_iter.append(rms)
        if rms_before is None:
            rms_before = rms
        if verbose:
            print(f"[mini-calib iter {it}] RMS(dy,dz)={rms:.2f}px  "
                  f"RMS(df)={math.sqrt(np.nanmean(rf ** 2)):.2f}fr",
                  flush=True)
        Jy, Jz, Jf = [], [], []
        for n in names:
            st2 = dict(state)
            st2[n] = state[n] + bumps[n]
            ayp, azp, afp, _ = _anchors_with_model(
                ds, model_at(st2), dtype, device)
            Jy.append((ayp - ayi)[ok])
            Jz.append((azp - azi)[ok])
            Jf.append((afp - afi)[ok])
        Jy = np.stack(Jy, 1)
        Jz = np.stack(Jz, 1)
        Jf = np.stack(Jf, 1)
        A = np.vstack([Jy, Jz, frame_weight * Jf])
        b = np.concatenate([ry, rz, frame_weight * rf])
        good = np.isfinite(b) & np.all(np.isfinite(A), axis=1)
        med = np.median(b[good])
        mad = np.median(np.abs(b[good] - med)) + 1e-9
        good &= np.abs(b - med) < mad_reject * 1.4826 * mad
        Ag, bg = A[good], b[good]
        ata = Ag.T @ Ag
        ata += damping * np.eye(len(names)) * np.trace(ata) / len(names)
        x = np.linalg.solve(ata, Ag.T @ bg)          # step in bump units
        x = np.clip(x, -max_step_bumps, max_step_bumps)
        for k, n in enumerate(names):
            state[n] += x[k] * bumps[n]
        if verbose:
            print("   step: " + " ".join(
                f"d{n}={x[k] * bumps[n]:+.4f}" for k, n in enumerate(names)),
                flush=True)

    m_fin = model_at(state)
    ayf, azf, aff, _ = _anchors_with_model(ds, m_fin, dtype, device)
    ryf = (abs_y - ayf)[ok]
    rzf = (abs_z - azf)[ok]
    rff = (abs_f - aff)[ok]
    rms_after = math.sqrt(np.nanmean(ryf ** 2 + rzf ** 2))
    per_iter.append(rms_after)
    return RawFrameCalibration(
        calibrated=state,
        original=original,
        rms_before_px=float(rms_before),
        rms_after_px=float(rms_after),
        rms_frames_after=float(math.sqrt(np.nanmean(rff ** 2))),
        n_spots_used=int(ok.sum()),
        per_iter_rms_px=per_iter,
    )


def layer_model_factory(
    layer_dir: str | Path,
    *,
    ring_numbers: Sequence[int],
    n_pixels_y: int,
    n_pixels_z: int,
    n_frames: Optional[int] = None,
    omega_step: Optional[float] = None,
    apply_distortion: bool = True,
    max_two_theta_deg: Optional[float] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable[..., HEDMForwardModel]:
    """Build a ``model_factory`` for :func:`calibrate_raw_frame_geometry`
    from a MIDAS layer directory (paramstest/hkls/positions).

    The returned callable accepts ``geom_mod`` (dict of absolute geometry
    values: y_BC, z_BC, tx, ty, tz, Lsd) and ``omega_start`` overrides.
    """
    from midas_fit_grain.driver import _cartesian_B_matrix, _read_hkls_csv

    from .io import (
        _first_float, geometry_from_paramstest, parse_paramstest,
        scan_config_from_positions,
    )

    layer = Path(layer_dir)
    params = parse_paramstest(layer / "paramstest.txt")
    lat = tuple(float(x) for x in params["LatticeParameter"][0][:6])
    beam_size = _first_float(params, "BeamSize")

    def factory(geom_mod: Optional[Dict[str, float]] = None,
                omega_start: Optional[float] = None) -> HEDMForwardModel:
        geom = geometry_from_paramstest(
            params, n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
            n_frames=n_frames, omega_step=omega_step,
            apply_distortion=apply_distortion,
        )
        if geom_mod:
            for k, v in geom_mod.items():
                setattr(geom, k, float(v))
        if omega_start is not None:
            geom.omega_start = float(omega_start)
        mtth = max_two_theta_deg
        if mtth is None:
            if "MaxRingRad" in params:
                mrr = _first_float(params, "MaxRingRad")
                lsd = geom.Lsd[0] if isinstance(geom.Lsd, list) else geom.Lsd
                mtth = 2.0 * math.degrees(math.atan(mrr / lsd))
            else:
                mtth = 180.0
        hkls_int, thetas_deg, _ring_nr = _read_hkls_csv(
            layer / "hkls.csv", [int(r) for r in ring_numbers], mtth)
        B = _cartesian_B_matrix(lat)
        cart = (B @ hkls_int.astype(np.float64).T).T
        scan_cfg = scan_config_from_positions(
            layer / "positions.csv", beam_size, dtype=dtype)
        model = HEDMForwardModel(
            torch.from_numpy(cart),
            torch.from_numpy(np.asarray(thetas_deg) * math.pi / 180.0),
            geom,
            hkls_int=torch.from_numpy(hkls_int.astype(np.float64)),
            scan_config=scan_cfg,
            device=device,
        )
        # Raw-frame convention: tilts are part of the raw prediction.
        model.apply_tilts = True
        return model

    return factory
