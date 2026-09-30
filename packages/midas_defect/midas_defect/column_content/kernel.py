"""Instrument kernel for the column-content fit: an anisotropic 3-D Gaussian in (frame, radial, tangential).

The in-plane axes are radial / tangential about the beam centre at each spot's own position, the directions along
which a mono DAC spot is actually elongated. Unit flux over voxels (analytic normalisation; point-sampled, adequate for
widths >= ~0.8 voxel). A single global ``scale`` stretches all three widths (fitted per column, like the Laue kernel
scale) and keeps unit flux.

:func:`estimate_kernel` MEASURES the widths from the data's own isolated spots (the analogue of measuring the Laue kernel
on a calibrant): a spot is used only if exactly one predicted reflection of the supplied orientations lands in its blob.
Do not assume widths -- a Gaussian with guessed widths invents orientation spread (sampleH lesson: ~0.01 deg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

DT = torch.float64


@dataclass
class GaussKernel3D:
    sig_frame: float          # frames
    sig_rad: float            # px, along the radial direction from the beam centre
    sig_tan: float            # px, perpendicular in the detector plane
    bcz: float                # beam centre row (px)
    bcy: float                # beam centre col (px)

    def __call__(self, df, dr, dc, r0, c0, scale=1.0):
        """Unit-flux kernel values at voxel offsets (df, dr, dc) from a spot centred at pixel (r0, c0).
        All tensors broadcast; ``scale`` (float or 0-d tensor) stretches every width."""
        er = r0 - self.bcz; ec = c0 - self.bcy
        nrm = torch.sqrt(er * er + ec * ec).clamp_min(1e-9)
        er, ec = er / nrm, ec / nrm
        drad = dr * er + dc * ec
        dtan = -dr * ec + dc * er
        sf, sr, st = self.sig_frame * scale, self.sig_rad * scale, self.sig_tan * scale
        z = (df / sf) ** 2 + (drad / sr) ** 2 + (dtan / st) ** 2
        norm = (2.0 * math.pi) ** 1.5 * sf * sr * st
        return torch.exp(-0.5 * z) / norm


def _isolated_moments(sub, labels, pred_frc, bcz, bcy, sat=None):
    """Per isolated blob (exactly one prediction inside, no saturated voxel): (sig_frame, sig_rad, sig_tan, peak)."""
    from scipy import ndimage as ndi
    hits = {}
    for f, r, c in np.asarray(pred_frc, float):
        fi, ri, ci = int(round(f)), int(round(r)), int(round(c))
        sl = labels[max(fi - 1, 0):fi + 2, max(ri - 2, 0):ri + 3, max(ci - 2, 0):ci + 3]
        for lab in set(np.unique(sl)) - {0}:
            hits[lab] = hits.get(lab, 0) + 1
    objs = ndi.find_objects(labels)
    out = []
    for lab, n in hits.items():
        if n != 1 or objs[lab - 1] is None:
            continue
        sl = objs[lab - 1]
        m = labels[sl] == lab
        if sat is not None and np.any(sat[sl][m]):
            continue
        out.append(_moments_of(sub, sl, m, bcz, bcy))
    return [o for o in out if o is not None]


def _moments_of(sub, sl, m, bcz, bcy):
    w = np.clip(sub[sl], 0, None) * m
    tot = w.sum()
    if tot <= 0:
        return None
    ff, rr, cc = np.nonzero(m)
    ww = w[ff, rr, cc]
    ff = ff + sl[0].start; rr = rr + sl[1].start; cc = cc + sl[2].start
    mf, mr, mc = (ww * ff).sum() / tot, (ww * rr).sum() / tot, (ww * cc).sum() / tot
    er, ec = mr - bcz, mc - bcy; nn = math.hypot(er, ec) or 1.0; er, ec = er / nn, ec / nn
    drad = (rr - mr) * er + (cc - mc) * ec; dtan = -(rr - mr) * ec + (cc - mc) * er
    return (math.sqrt((ww * (ff - mf) ** 2).sum() / tot), math.sqrt((ww * drad ** 2).sum() / tot),
            math.sqrt((ww * dtan ** 2).sum() / tot), float(np.max(np.asarray(sub[sl])[m])))


def estimate_kernel(sub: np.ndarray, labels: np.ndarray, pred_frc: np.ndarray, bcz: float, bcy: float, *,
                    min_spots: int = 8, sat: np.ndarray | None = None, calibrate: bool = True,
                    threshold: float = 200.0, pedestal: float = 150.0, n_sim: int = 80, iters: int = 4,
                    seed: int = 0):
    """MEASURED kernel from ISOLATED spots (one prediction inside the blob, no saturated voxel).

    Raw second moments of a thresholded blob are biased NARROW (the threshold cuts the tails): on synthetic DAC
    spots the raw estimate read ~0.84x the true widths. With ``calibrate=True`` (default) the bias is removed by
    simulation: isolated spots with the measured peak heights, the same ``pedestal``, Poisson noise, detection at the
    same ``threshold`` and the same moment code are generated for a trial kernel, and the trial widths are rescaled
    until the simulated moments match the measured ones (fixed point, ``iters`` rounds). Pass the ingest threshold
    actually used. Returns (GaussKernel3D, info)."""
    mom = _isolated_moments(sub, labels, pred_frc, bcz, bcy, sat)
    if len(mom) < min_spots:
        raise RuntimeError(f"estimate_kernel: only {len(mom)} isolated unsaturated spots (need {min_spots})")
    M = np.asarray(mom)
    meas = np.median(M[:, :3], axis=0)
    info = dict(n_spots=len(mom), raw_sigmas=meas.tolist(), peaks=M[:, 3].tolist(), calibrated=False)
    if not calibrate:
        return GaussKernel3D(*map(float, meas), bcz, bcy), info
    trial = meas.copy()
    for _ in range(iters):
        sim = _simulate_moments(trial, M[:, 3], threshold=threshold, pedestal=pedestal, n=n_sim, seed=seed)
        trial = trial * meas / np.maximum(sim, 1e-9)
    info.update(calibrated=True, calibrated_sigmas=trial.tolist(),
                final_sim_over_meas=(_simulate_moments(trial, M[:, 3], threshold=threshold, pedestal=pedestal,
                                                       n=n_sim, seed=seed + 1) / meas).tolist())
    return GaussKernel3D(*map(float, trial), bcz, bcy), info


def _simulate_moments(sig, peaks, *, threshold, pedestal, n, seed):
    """Median raw moments of n isolated simulated spots of true widths ``sig`` (frame, radial, tangential), peak
    heights resampled from ``peaks``, detected at ``threshold`` above a Poisson ``pedestal`` exactly as ingest does."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    sf, sr, st = map(float, sig)
    hf, hp = int(math.ceil(5 * sf)) + 2, int(math.ceil(5 * max(sr, st))) + 2
    shape = (2 * hf + 1, 2 * hp + 1, 2 * hp + 1)
    ff, rr, cc = np.meshgrid(np.arange(shape[0]) - hf, np.arange(shape[1]) - hp, np.arange(shape[2]) - hp, indexing="ij")
    res = []
    for k in range(n):
        th = rng.uniform(0, 2 * math.pi)                      # radial direction of this spot on the detector
        er, ec = math.cos(th), math.sin(th)
        df, dr, dc = ff - rng.uniform(-0.5, 0.5), rr - rng.uniform(-0.5, 0.5), cc - rng.uniform(-0.5, 0.5)
        drad = dr * er + dc * ec; dtan = -dr * ec + dc * er
        amp = float(rng.choice(peaks))
        img = pedestal + amp * np.exp(-0.5 * ((df / sf) ** 2 + (drad / sr) ** 2 + (dtan / st) ** 2))
        sub = rng.poisson(img).astype(float) - pedestal
        lab, _ = ndi.label(sub > threshold, structure=np.ones((3, 3, 3)))
        c = lab[hf, hp, hp]
        if c == 0:
            continue
        m = lab == c
        bcz, bcy = hp - 1e6 * er, hp - 1e6 * ec                 # a far beam centre along -radial: fixes the axes
        mo = _moments_of(sub, (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])), m, bcz, bcy)
        if mo is not None:
            res.append(mo[:3])
    if not res:
        return np.asarray(sig, float) * 0 + 1e-9
    return np.median(np.asarray(res), axis=0)
