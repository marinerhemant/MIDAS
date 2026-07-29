"""Digital twin: turn a ground-truth grain population into a realistic *measured*
spot list, so the indexing / merge / reconstruction chain can be validated
before any beam time.

Realism added on top of the ideal forward model:

* **Intensity + detectability** -- each spot gets a Lorentz-polarisation x
  Debye-Waller intensity proxy (down-weights high-Q reflections); the weakest
  spots fall below the detection floor and are lost.  (A full |F|^2 needs atom
  positions per material -- a Tier-2 upgrade; the proxy captures the high-Q
  fall-off that matters for detectability.)
* **Measurement noise** -- Gaussian scatter on detector centroid (pixels) and
  omega (degrees).
* **Dead regions** -- inter-module gaps / beamstop already removed by the
  forward accessibility gate.
* **Spurious peaks** -- background / cosmic / neighbouring-grain-tail false
  peaks with no true label.
* **Correspondence stripped** -- the returned list is unordered and unlabelled
  (truth kept separately, only for scoring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation
from . import geometry as geo


@dataclass
class MeasuredSpots:
    """Unlabelled measured spots (+ hidden truth for scoring only)."""
    y_pixel: np.ndarray
    z_pixel: np.ndarray
    omega_deg: np.ndarray
    two_theta_deg: np.ndarray
    eta_deg: np.ndarray
    mounting_id: np.ndarray
    intensity: np.ndarray
    # hidden ground truth (do NOT use in indexing; scoring only)
    true_grain_id: np.ndarray      # -1 for spurious
    true_hkl: np.ndarray           # (S,3), 0 for spurious

    def __len__(self):
        return len(self.y_pixel)

    def for_mounting(self, m: int) -> "MeasuredSpots":
        sel = self.mounting_id == m
        return MeasuredSpots(*[getattr(self, f)[sel] for f in
                               ("y_pixel", "z_pixel", "omega_deg", "two_theta_deg",
                                "eta_deg", "mounting_id", "intensity",
                                "true_grain_id", "true_hkl")])


def _lp_dw_intensity(two_theta_rad: np.ndarray, wavelength_A: float,
                     B_iso: float) -> np.ndarray:
    """Lorentz-polarisation x Debye-Waller intensity proxy (arb. units)."""
    tt = np.clip(two_theta_rad, 1e-4, None)
    th = tt / 2.0
    lorentz = 1.0 / np.clip(np.sin(tt), 1e-3, None)          # rotation method
    polar = (1.0 + np.cos(tt) ** 2) / 2.0
    s = np.sin(th) / wavelength_A                            # sin(theta)/lambda
    dw = np.exp(-2.0 * B_iso * s * s)
    return lorentz * polar * dw


def make_measured_spots(
    cfg: XAFConfig,
    grains: GrainPopulation,
    *,
    fwd: Optional[XAFForwardModel] = None,
    detect_frac: float = 0.6,       # fraction of accessible spots above the floor
    pos_noise_px: float = 1.0,
    omega_noise_deg: float = 0.25,
    spurious_frac: float = 0.10,    # spurious peaks as fraction of real spots
    B_iso: float = 0.5,
    seed: int = 0,
) -> Dict[str, object]:
    """Return a realistic measured spot list + a summary dict."""
    fwd = fwd or XAFForwardModel(cfg)
    sim = fwd.simulate(grains)
    t = sim.table
    rng = np.random.default_rng(seed)

    yy = t.y_pixel.cpu().numpy()
    zz = t.z_pixel.cpu().numpy()
    om = np.degrees(t.omega.cpu().numpy())
    tt = t.two_theta.cpu().numpy()
    eta = np.degrees(t.eta.cpu().numpy())
    mid = t.mounting_id.cpu().numpy()
    gid = t.grain_id.cpu().numpy()
    hkl = t.hkl.cpu().numpy()

    # Real |F|^2 * Lorentz-polarisation per spot (falls back to LP*DW proxy for
    # materials without a defined atom structure).
    from . import structure
    inten = structure.reflection_intensities(cfg.material, hkl, tt / 2.0,
                                             cfg.wavelength_A)
    if inten.max() > 0:
        inten = inten / inten.max()
    # detection floor: keep the strongest ``detect_frac`` of spots
    n_real = len(yy)
    keep_n = max(1, int(round(detect_frac * n_real)))
    thresh = np.sort(inten)[max(0, n_real - keep_n)]
    keep = inten >= thresh

    yy, zz, om, tt, eta, mid, gid, hkl, inten = (
        a[keep] for a in (yy, zz, om, tt, eta, mid, gid, hkl, inten))

    # measurement noise
    yy = yy + rng.normal(scale=pos_noise_px, size=yy.shape)
    zz = zz + rng.normal(scale=pos_noise_px, size=zz.shape)
    om = om + rng.normal(scale=omega_noise_deg, size=om.shape)

    # spurious peaks: random detector positions in the accessible 2theta disk,
    # random omega inside a wedge, weak intensity, no true label.
    n_spur = int(round(spurious_frac * len(yy)))
    if n_spur > 0:
        yc, zc = 0.5 * cfg.n_pixels_y, 0.5 * cfg.n_pixels_z
        Lsd = cfg.resolved_Lsd_um()
        r_max = Lsd * np.tan(np.radians(cfg.tth_max_deg)) / cfg.px_um
        rr = r_max * np.sqrt(rng.uniform(0, 1, n_spur))
        ang = rng.uniform(0, 2 * np.pi, n_spur)
        sy = yc + rr * np.cos(ang)
        sz = zc + rr * np.sin(ang)
        centers = np.asarray(cfg.wedge_centers_deg)
        c = rng.choice(centers, n_spur)
        so = c + rng.uniform(-cfg.wedge_half_deg, cfg.wedge_half_deg, n_spur)
        so = (so + 180) % 360 - 180
        smid = rng.integers(0, cfg.n_mountings, n_spur)
        stt = np.degrees(np.arctan(rr * cfg.px_um / Lsd))
        # keep spurious off the dead regions too
        live = geo.detector_live_mask(torch.tensor(sy), torch.tensor(sz), cfg).numpy()
        sy, sz, so, smid, stt = (a[live] for a in (sy, sz, so, smid, stt))
        n_spur = len(sy)
        yy = np.concatenate([yy, sy]); zz = np.concatenate([zz, sz])
        om = np.concatenate([om, so]); tt = np.concatenate([tt, np.radians(stt)])
        eta = np.concatenate([eta, np.zeros(n_spur)])
        mid = np.concatenate([mid, smid])
        inten = np.concatenate([inten, rng.uniform(0, thresh + 1e-6, n_spur)])
        gid = np.concatenate([gid, -np.ones(n_spur, int)])
        hkl = np.concatenate([hkl, np.zeros((n_spur, 3), int)])

    # strip correspondence: shuffle
    perm = rng.permutation(len(yy))
    ms = MeasuredSpots(
        y_pixel=yy[perm], z_pixel=zz[perm], omega_deg=om[perm],
        two_theta_deg=np.degrees(tt[perm]), eta_deg=eta[perm],
        mounting_id=mid[perm], intensity=inten[perm],
        true_grain_id=gid[perm], true_hkl=hkl[perm])
    summary = {
        "n_accessible": n_real,
        "n_detected": int(keep.sum()),
        "n_spurious": int(n_spur),
        "n_total": len(ms),
        "detect_frac": detect_frac,
        "spurious_frac": spurious_frac,
    }
    return {"spots": ms, "summary": summary}
