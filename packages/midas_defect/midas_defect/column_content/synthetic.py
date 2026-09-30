"""Synthetic DAC columns with KNOWN orientation content, for validating the column-content fit.

Extends :func:`midas_defect.synthetic.synthetic_position_frames` (same geometry primitives, same frame layout) with what
that generator lacks for this purpose: per-domain orientation SPREAD (point / 3-D cloud / 1-D streak of
sub-orientations), per-reflection brightness scatter (log-normal, shared by a domain's sub-orientations, standing in
for |F|^2 x Lorentz x absorption, which the fit does not model), the measured-type anisotropic kernel, an optional
FOREIGN single crystal (e.g. a diamond anvil) and powder rings (gasket / pressure marker). Truth records each domain's
intensity SHARE = its rendered flux / all rendered single-crystal flux (sample domains + foreign crystal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from midas_defect.geometry import Geometry, detector_angle_maps
from .forward import DT, observable_reflections, predict_torch, rotvec_to_matrix
from .kernel import GaussKernel3D


@dataclass
class DomainSpec:
    U: np.ndarray                 # crystal -> sample
    brightness: float             # relative scale
    spread: str = "point"         # "point" | "cloud" | "streak"
    param_deg: float = 0.0        # cloud sigma or streak half-width (deg)
    n_comp: int = 41


@dataclass
class ColumnTruth:
    U: List[np.ndarray]
    share: np.ndarray
    spread: List[str]
    param_deg: np.ndarray
    comp_rotvecs: List[np.ndarray]          # per domain, (n, 3) sample-frame offsets (radians)
    foreign_share: float = 0.0
    extras: dict = field(default_factory=dict)


def domain_components(spec: DomainSpec, rng: np.random.Generator) -> np.ndarray:
    if spec.spread == "point":
        return np.zeros((1, 3))
    if spec.spread == "cloud":
        return rng.normal(0.0, math.radians(spec.param_deg), size=(spec.n_comp, 3))
    if spec.spread == "streak":
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        return np.outer(np.radians(np.linspace(-spec.param_deg, spec.param_deg, spec.n_comp)), ax)
    raise ValueError(spec.spread)


def _render_domain(img, U, B, hkl_all, geom, kernel, rotvecs, fac_sigma, brightness, rng, omega_sign, stamp_sig=3.0):
    obs = observable_reflections(U, B, hkl_all, geom, omega_sign=omega_sign)
    if not len(obs.hkl):
        return 0.0
    fac = np.exp(rng.normal(0.0, fac_sigma, size=len(obs.hkl))) * brightness
    f, r, c = predict_torch(U, B, obs, torch.as_tensor(rotvecs, dtype=DT), geom, omega_sign=omega_sign)
    f, r, c = f.detach().numpy(), r.detach().numpy(), c.detach().numpy()
    nF, H, W = img.shape
    hf = int(math.ceil(stamp_sig * kernel.sig_frame)) + 1; hp = int(math.ceil(stamp_sig * max(kernel.sig_rad, kernel.sig_tan))) + 1
    flux = 0.0
    for k in range(f.shape[0]):
        for h in range(f.shape[1]):
            fi, ri, ci = int(round(f[k, h])), int(round(r[k, h])), int(round(c[k, h]))
            f0, f1 = max(fi - hf, 0), min(fi + hf + 1, nF); r0, r1 = max(ri - hp, 0), min(ri + hp + 1, H); c0, c1 = max(ci - hp, 0), min(ci + hp + 1, W)
            if f0 >= f1 or r0 >= r1 or c0 >= c1:
                continue
            ff, rr, cc = np.mgrid[f0:f1, r0:r1, c0:c1]
            v = kernel(torch.as_tensor(ff - f[k, h], dtype=DT), torch.as_tensor(rr - r[k, h], dtype=DT),
                       torch.as_tensor(cc - c[k, h], dtype=DT), torch.as_tensor(r[k, h], dtype=DT),
                       torch.as_tensor(c[k, h], dtype=DT)).numpy()
            amp = fac[h] / f.shape[0]
            img[f0:f1, r0:r1, c0:c1] += amp * v
            flux += amp * float(v.sum())
    return flux


def synthetic_column(domains: Sequence[DomainSpec], *, B: np.ndarray, hkl_all: np.ndarray, geom: Geometry,
                     kernel: GaussKernel3D, peak_counts: float = 20000.0, pedestal: float = 150.0,
                     fac_sigma: float = 1.0, foreign: Optional[dict] = None,
                     powder_two_theta_deg: Sequence[float] = (), powder_amplitude: float = 60.0,
                     omega_sign: int = 1, seed: int = 0):
    """Frames (nF, H, W) float32 with Poisson noise, and the ColumnTruth.

    foreign = dict(U=..., B=..., hkl_all=..., brightness=...) adds an unrelated single crystal (not a sample domain).
    The brightest voxel of the single-crystal signal is scaled to ``peak_counts``."""
    rng = np.random.default_rng(seed)
    shape = (geom.n_frames, geom.n_pix_z, geom.n_pix_y)
    sig = np.zeros(shape)
    fluxes, comps = [], []
    for d in domains:
        rv = domain_components(d, rng); comps.append(rv)
        one = np.zeros(shape)
        fluxes.append(_render_domain(one, d.U, B, hkl_all, geom, kernel, rv, fac_sigma, d.brightness, rng, omega_sign))
        sig += one
    ff = 0.0
    if foreign:
        one = np.zeros(shape)
        ff = _render_domain(one, foreign["U"], foreign["B"], foreign["hkl_all"], geom, kernel, np.zeros((1, 3)),
                            fac_sigma, foreign.get("brightness", 1.0), rng, omega_sign)
        sig += one
    scale = peak_counts / max(sig.max(), 1e-300)
    clean = pedestal + sig * scale
    if powder_two_theta_deg:
        tth, _ = detector_angle_maps(geom); tth = np.asarray(tth)
        ring = sum(powder_amplitude * np.exp(-0.5 * ((tth - t) / 0.05) ** 2) for t in powder_two_theta_deg)
        clean = clean + ring[None]
    frames = rng.poisson(np.clip(clean, 0, None)).astype(np.float32)
    tot = sum(fluxes) + ff
    truth = ColumnTruth(U=[np.asarray(d.U, float) for d in domains], share=np.asarray(fluxes) / max(tot, 1e-300),
                        spread=[d.spread for d in domains], param_deg=np.asarray([d.param_deg for d in domains]),
                        comp_rotvecs=comps, foreign_share=ff / max(tot, 1e-300),
                        extras=dict(peak_counts=peak_counts, scale=scale))
    return frames, truth
