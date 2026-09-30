"""Per-grain intensity fitting for NF-HEDM with a censored detection model.

The orientation fitter compares against a binary image (``SpotsInfo.bin``).
With ``WriteGreyResidual 1`` the reduction also keeps the grey value of every
lit pixel (``SpotsGrey.npz``), and this module fits those grey values with a
kinematic forward model, orientations and positions held fixed.

The reason for a forward model rather than summing lit pixels per spot is the
detection step: a pixel is only recorded when its filtered value clears the
blanket, so dim spots lose a larger fraction of their light than bright ones
and a naive sum steepens every intensity trend (|F| exponent, grain size).
Here the likelihood is censored (Tobit): a recorded pixel contributes the
density of its value, a predicted-but-unrecorded pixel the probability of
falling below the threshold.

Model, at detector pixel ``(frame, y, z)``::

    mu = sum_rows  s[grain] * |F[refl]|**p * w_geom * PSF(y, z; sigma_psf) * split(frame)

where a *row* is one (voxel, reflection, omega solution) spot from
:class:`midas_diffract.forward.HEDMForwardModel`, ``w_geom`` carries the
rotation Lorentz factor, polarization, the voxel weight and the beam profile
at the voxel's height. A line beam is a profile that is 1 at z = 0. A box beam
would need a wide profile and 3D positions; that use is NOT implemented or
validated (every gate below assumed a thin beam).

The absolute scale of ``s`` is degenerate with flux; compare scales after
normalising (:meth:`IntensityFit.relative_scales`).

Status (EXPERIMENTAL; validated on line-focus data only: AlON, Ti-7Al, LSHR)
- Hold the pixel-noise sigma fixed at the value measured from pure noise passed
  through the same reduction (``fit_sigma_noise=False``, e.g. 0.82 for a 3x3
  median of sigma-2 noise). Fitted freely it collapses to ~0 and is rewarded by
  empty units.
- Measure the vertical PSF width (``fit_sigma_z``) on pure, untwinned grains and
  freeze it; fitted jointly with mixture fractions it rails at both bounds.
- The |F| exponent is ring-dominated (AlON p ~ 1.9 +- 0.3). Do not read a
  per-ring median of log(O/E) with undetected spots: it is exactly 0 (censoring).
- :class:`Mixture` (per-site softmax fractions) cannot leave a saturated start;
  use :mod:`midas_nf_fitorientation.twin_search` for discrete searches.
- ``SpotsGrey.npz`` comes from midas-nf-preprocess with ``WriteGreyResidual 1``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "ContributionTable",
    "lorentz_polarization",
    "build_contributions",
    "RecordedPixels",
    "IntensityModel",
    "IntensityFit",
    "fit_intensity",
    "heldout_nll",
    "SplatBlurModel",
    "spot_ownership",
    "fit_spot_intensity",
    "SpotFit",
]


def lorentz_polarization(two_theta: np.ndarray, eta: np.ndarray,
                         pol_fraction: float = 1.0) -> np.ndarray:
    """Rotation Lorentz factor times polarization, per spot.

    ``L = 1 / (sin 2θ |sin η|)`` for rotation about the vertical axis, with η
    measured from +z (the MIDAS convention). ``P = 1 - f (sin 2θ sin η)^2`` for a
    beam polarized in the horizontal plane (f = 1 fully polarized).
    """
    s2t = np.sin(two_theta)
    se = np.abs(np.sin(eta))
    L = 1.0 / np.maximum(s2t * se, 1e-12)
    P = 1.0 - pol_fraction * (s2t * np.sin(eta)) ** 2
    return L * P


# Widths are stored as unconstrained u and mapped into a box in log space, so an
# optimiser cannot drive them to overflow; a width at its upper bound = a rail.
_LOG_SIG_LO = math.log(0.05)


def _box(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(u)


def _unbox(v: float, lo: float, hi: float) -> float:
    x = min(max((v - lo) / (hi - lo), 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def _psf_hi(radius: int) -> float:
    return math.log(4.0 * max(radius, 1))


@dataclass
class ContributionTable:
    """One row per predicted (voxel, reflection, omega solution) spot."""
    grain: np.ndarray        # (R,) int64, 0..n_grains-1
    refl: np.ndarray         # (R,) int64, index into log_F
    sol: np.ndarray          # (R,) int8, omega solution 0/1
    frame: np.ndarray        # (R,) float, fractional frame
    y: np.ndarray            # (R,) float, fractional pixel (same flipped convention as SpotsInfo.bin)
    z: np.ndarray            # (R,) float
    w_geom: np.ndarray       # (R,) float, L*P*voxel_weight*beam
    log_F: np.ndarray        # (M,) float, log |F| per reflection
    n_grains: int

    def __len__(self) -> int:
        return int(self.grain.size)

    def concat(self, other: "ContributionTable") -> "ContributionTable":
        if not np.array_equal(self.log_F, other.log_F):
            raise ValueError("tables built for different reflection lists")
        return ContributionTable(
            *(np.concatenate([getattr(self, k), getattr(other, k)])
              for k in ("grain", "refl", "sol", "frame", "y", "z", "w_geom")),
            log_F=self.log_F, n_grains=max(self.n_grains, other.n_grains))


def build_contributions(
    model,
    euler: np.ndarray,
    positions: np.ndarray,
    grain: np.ndarray,
    abs_F: np.ndarray,
    *,
    voxel_weight: Optional[np.ndarray] = None,
    beam_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    min_abs_sin_eta: float = math.sin(math.radians(15.0)),
    pol_fraction: float = 1.0,
    distance: int = 0,
    batch: int = 4096,
    edge_length: Optional[np.ndarray] = None,
    ud: Optional[np.ndarray] = None,
    n_sub: int = 4,
    return_voxel: bool = False,
):
    """Run the forward model for fixed voxels and tabulate their spots.

    Parameters
    ----------
    model : HEDMForwardModel
        e.g. from :func:`midas_nf_fitorientation.soft_overlap.build_forward_model`.
    euler : (N, 3) radians.
    positions : (N, 2) or (N, 3) µm. z matters only through ``beam_profile``
        and the projection (box beam).
    grain : (N,) grain label per voxel, 0..G-1.
    abs_F : (M,) |F| per reflection of ``model`` (e.g. from ``midas_hkls``).
    voxel_weight : (N,) optional relative voxel volume (default 1).
    beam_profile : callable z_um -> relative flux (default 1 everywhere).
    min_abs_sin_eta : spots closer to the rotation axis are dropped (Lorentz blows up).
    distance : which detector distance of a layered model to tabulate.
    edge_length, ud : optional per-voxel triangle edge (µm) and up/down flag (the NF
        grid's space-filling triangle, e.g. ``GridSize``). When given, each spot is
        represented by ``n_sub**2`` equal-area sub-points of the voxel's triangle
        instead of its centre. The sample-to-detector map is affine for a flat
        detector, so the sub-points are interpolated between the projected vertices
        (exact) and carry ``1/n_sub**2`` of the weight each.
    return_voxel : also return the (R,) input-voxel index of every row, e.g. to build a :class:`Mixture`.
    """
    euler = np.asarray(euler, np.float64)
    pos = np.asarray(positions, np.float64)
    if pos.shape[1] == 2:
        pos = np.c_[pos, np.zeros(len(pos))]
    grain = np.asarray(grain, np.int64)
    N = len(euler)
    vw = np.ones(N) if voxel_weight is None else np.asarray(voxel_weight, np.float64)
    bw = np.ones(N) if beam_profile is None else np.asarray(beam_profile(pos[:, 2]), np.float64)
    abs_F = np.asarray(abs_F, np.float64)
    if np.any(abs_F <= 0):
        raise ValueError("abs_F must be > 0 for every reflection (drop forbidden ones first)")
    dev = model.hkls.device if hasattr(model, "hkls") else torch.device("cpu")
    dt = torch.float64
    cols = {k: [] for k in ("grain", "refl", "sol", "frame", "y", "z", "w_geom", "voxel")}
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch)); n = sl.stop - sl.start
        with torch.no_grad():
            sp = model(torch.tensor(euler[sl], device=dev, dtype=dt),
                       torch.tensor(pos[sl], device=dev, dtype=dt))
        M = sp.frame_nr.shape[-1]
        fr = sp.frame_nr.reshape(2, n, M).cpu().numpy()
        va = sp.valid.reshape(2, n, M).cpu().numpy() > 0.5
        yp = sp.y_pixel.reshape(-1, 2, n, M)[distance].cpu().numpy()
        zp = sp.z_pixel.reshape(-1, 2, n, M)[distance].cpu().numpy()
        if sp.layer_valid is not None:
            va &= sp.layer_valid.reshape(-1, 2, n, M)[distance].cpu().numpy() > 0.5
        eta = sp.eta.reshape(2, n, M).cpu().numpy()
        tth = sp.two_theta.reshape(2, n, M).cpu().numpy()
        va &= np.abs(np.sin(eta)) >= min_abs_sin_eta
        k, i, m = np.nonzero(va)
        lp = lorentz_polarization(tth[k, i, m], eta[k, i, m], pol_fraction)
        cols["grain"].append(grain[sl][i]); cols["refl"].append(m.astype(np.int64))
        cols["sol"].append(k.astype(np.int8)); cols["frame"].append(fr[k, i, m])
        cols["y"].append(yp[k, i, m]); cols["z"].append(zp[k, i, m])
        cols["w_geom"].append(lp * vw[sl][i] * bw[sl][i])
        cols["voxel"].append(i + b0)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    if edge_length is not None:
        out = _triangle_subpoints(model, euler, pos, out, np.asarray(edge_length, np.float64),
                                  np.asarray(ud, np.float64), n_sub, distance, batch, dev, dt)
    voxel = out.pop("voxel")
    table = ContributionTable(**out, log_F=np.log(abs_F), n_grains=int(grain.max()) + 1)
    return (table, voxel) if return_voxel else table


def _barycentric_subpoints(n: int) -> np.ndarray:
    """Centroids of the n^2 equal sub-triangles, as barycentric weights (K, 3) on (V0, V1, V2)."""
    pts = []
    for i in range(n):
        for j in range(n - i):
            pts.append(((i + 1 / 3) / n, (j + 1 / 3) / n))          # upward sub-triangles
            if i + j <= n - 2:
                pts.append(((i + 2 / 3) / n, (j + 2 / 3) / n))      # downward sub-triangles
    b = np.array(pts)
    return np.c_[1 - b.sum(1), b]


def _triangle_subpoints(model, euler, pos, out, edge, ud, n_sub, distance, batch, dev, dt):
    """Replace each (voxel, refl, sol) row by n_sub^2 sub-rows spread over the voxel triangle's projection."""
    from midas_diffract.forward import HEDMForwardModel
    N = len(euler)
    V = HEDMForwardModel.tri_vertices(torch.tensor(pos, dtype=dt), torch.tensor(edge, dtype=dt),
                                      torch.tensor(ud, dtype=dt)).numpy()          # (N, 3, 3)
    # project the three vertices (same orientation, so same frame and validity as the centre row)
    vy, vz = [], []
    for v in range(3):
        cy, cz = [], []
        for b0 in range(0, N, batch):
            sl = slice(b0, min(N, b0 + batch)); n = sl.stop - sl.start
            with torch.no_grad():
                sp = model(torch.tensor(euler[sl], device=dev, dtype=dt),
                           torch.tensor(V[sl, v, :], device=dev, dtype=dt))
            M = sp.frame_nr.shape[-1]
            cy.append(sp.y_pixel.reshape(-1, 2, n, M)[distance].cpu().numpy())
            cz.append(sp.z_pixel.reshape(-1, 2, n, M)[distance].cpu().numpy())
        vy.append(np.concatenate(cy, 1)); vz.append(np.concatenate(cz, 1))       # (2, N, M) each
    vox, refl, sol = out["voxel"], out["refl"], out["sol"]
    Yv = np.stack([vy[v][sol, vox, refl] for v in range(3)], 1)                  # (R, 3)
    Zv = np.stack([vz[v][sol, vox, refl] for v in range(3)], 1)
    B = _barycentric_subpoints(n_sub); K = len(B)
    rep = lambda a: np.repeat(a, K)
    new = {k: rep(v) for k, v in out.items() if k not in ("y", "z", "w_geom")}
    new["y"] = (Yv @ B.T).ravel(); new["z"] = (Zv @ B.T).ravel()
    new["w_geom"] = rep(out["w_geom"]) / K
    return new


@dataclass
class RecordedPixels:
    """Sparse recorded pixels at one distance: key = (frame*NY + y)*NZ + z, value above blanket."""
    key: np.ndarray          # sorted int64
    value: np.ndarray        # float, as stored (filtered value minus blanket, > 0)

    @classmethod
    def from_arrays(cls, frame, y, z, value, n_y: int, n_z: int) -> "RecordedPixels":
        key = (np.asarray(frame, np.int64) * n_y + np.asarray(y, np.int64)) * n_z + np.asarray(z, np.int64)
        o = np.argsort(key, kind="stable")
        return cls(key=key[o], value=np.asarray(value, np.float64)[o])

    @classmethod
    def from_spots_grey(cls, path, n_y: int, n_z: int, layer: int = 0) -> "RecordedPixels":
        """Read ``SpotsGrey.npz`` (``midas_nf_preprocess`` ``WriteGreyResidual 1``)."""
        g = np.load(path)
        s = g["layer"] == layer
        return cls.from_arrays(g["frame"][s], g["y"][s], g["z"][s], g["value"][s], n_y, n_z)

    def lookup(self, keys: np.ndarray):
        pos = np.clip(np.searchsorted(self.key, keys), 0, max(self.key.size - 1, 0))
        hit = (self.key[pos] == keys) if self.key.size else np.zeros(keys.shape, bool)
        return hit, np.where(hit, self.value[pos] if self.key.size else 0.0, 0.0)


class IntensityModel(torch.nn.Module):
    """Differentiable predicted image on the support of a :class:`ContributionTable`.

    Each row is spread over a ``(2r+1)^2`` pixel neighbourhood with a
    normalised Gaussian PSF and split linearly between the two frames whose
    centres bracket its fractional frame. The support is the union of those
    pixels; ``forward()`` returns the predicted mean at every support pixel.

    ``median_radius > 0`` mirrors the reduction's spatial median filter
    (``MedFiltRadius``): the returned value at each support pixel is the
    same-frame ``(2m+1)^2`` median of the predicted image, pixels outside the
    support counting as zero. Without it, narrow spots (small grains) are
    over-predicted relative to what the filtered image records, which biases
    fitted scales with grain size.
    """

    def __init__(self, table: ContributionTable, n_frames: int, n_y: int, n_z: int,
                 psf_radius: int = 2, sigma_psf=1.0, p: float = 2.0,
                 median_radius: int = 0, device="cpu", dtype=torch.float64):
        super().__init__()
        self.n_y, self.n_z, self.n_frames, self.r = n_y, n_z, n_frames, psf_radius
        dev = torch.device(device)
        yi = np.rint(table.y).astype(np.int64); zi = np.rint(table.z).astype(np.int64)
        fy = table.y - yi; fz = table.z - zi
        # linear frame split between frame centres (n + 0.5)
        f0 = np.floor(table.frame - 0.5).astype(np.int64)
        t = (table.frame - 0.5) - f0
        off = np.arange(-psf_radius, psf_radius + 1)
        oy, oz = np.meshgrid(off, off, indexing="ij"); oy = oy.ravel(); oz = oz.ravel()
        keys = []
        for df in (0, 1):
            fr = f0 + df
            k = ((fr[:, None] * n_y + (yi[:, None] + oy[None])) * n_z + (zi[:, None] + oz[None]))
            bad = (fr[:, None] < 0) | (fr[:, None] >= n_frames) | \
                  ((yi[:, None] + oy[None]) < 0) | ((yi[:, None] + oy[None]) >= n_y) | \
                  ((zi[:, None] + oz[None]) < 0) | ((zi[:, None] + oz[None]) >= n_z)
            keys.append(np.where(bad, -1, k))
        keys = np.concatenate(keys, 1)                                     # (R, 2L)
        valid = keys >= 0
        self.support_key, inv = np.unique(keys[valid], return_inverse=True)
        if self.support_key.size >= 2 ** 31:
            raise ValueError("support too large for int32 indices")
        idx = np.zeros(keys.shape, np.int32); idx[valid] = inv.ravel()
        del keys
        self.register_buffer("idx", torch.tensor(idx, device=dev))
        self.register_buffer("valid", torch.tensor(valid, device=dev))
        self.register_buffer("t", torch.tensor(t, device=dev, dtype=dtype))
        self.register_buffer("fy", torch.tensor(fy, device=dev, dtype=dtype))
        self.register_buffer("fz", torch.tensor(fz, device=dev, dtype=dtype))
        self.register_buffer("oy", torch.tensor(oy, device=dev, dtype=dtype))
        self.register_buffer("oz", torch.tensor(oz, device=dev, dtype=dtype))
        self.register_buffer("grain", torch.tensor(table.grain, device=dev))
        self.register_buffer("refl", torch.tensor(table.refl, device=dev))
        self.register_buffer("w_geom", torch.tensor(table.w_geom, device=dev, dtype=dtype))
        self.register_buffer("log_F", torch.tensor(table.log_F, device=dev, dtype=dtype))
        self.L = oy.size
        self.median_radius = int(median_radius)
        if self.median_radius > 0:
            mo = np.arange(-self.median_radius, self.median_radius + 1)
            my, mz = np.meshgrid(mo, mo, indexing="ij"); my = my.ravel(); mz = mz.ravel()
            sk = self.support_key
            fr_, rem = np.divmod(sk, n_y * n_z); yy, zz = np.divmod(rem, n_z)
            ny_ = yy[:, None] + my[None]; nz_ = zz[:, None] + mz[None]
            inside = (ny_ >= 0) & (ny_ < n_y) & (nz_ >= 0) & (nz_ < n_z)
            nk = (fr_[:, None] * n_y + ny_) * n_z + nz_
            pos = np.clip(np.searchsorted(sk, nk), 0, sk.size - 1)
            found = inside & (sk[pos] == nk)
            # index sk.size points at an appended zero (outside the support / detector)
            self.register_buffer("med_idx", torch.tensor(np.where(found, pos, sk.size), device=dev))
        sy, sz = (sigma_psf, sigma_psf) if np.isscalar(sigma_psf) else tuple(sigma_psf)
        self.log_s = torch.nn.Parameter(torch.zeros(table.n_grains, device=dev, dtype=dtype))
        self.p = torch.nn.Parameter(torch.tensor(float(p), device=dev, dtype=dtype))
        # (sigma_y, sigma_z) in pixels: voxel spacing and vertical beam / scintillator blur differ
        hi = _psf_hi(psf_radius)
        self.log_sigma_psf = torch.nn.Parameter(torch.tensor([_unbox(math.log(float(v)), _LOG_SIG_LO, hi) for v in (sy, sz)],
                                                             device=dev, dtype=dtype))

    def sigma_psf_now(self) -> torch.Tensor:
        return torch.exp(_box(self.log_sigma_psf, _LOG_SIG_LO, _psf_hi(self.r)))

    @property
    def n_support(self) -> int:
        return int(self.support_key.size)

    def psf_weights(self) -> torch.Tensor:
        """(R, 2L) weights: normalised anisotropic Gaussian x linear frame split, 0 off-detector."""
        # cap at 4x the window: beyond it the kernel is flat anyway, and an unbounded sigma overflows float32
        inv2 = self.sigma_psf_now() ** -2
        dy = self.oy[None, :] - self.fy[:, None]; dz = self.oz[None, :] - self.fz[:, None]
        g = torch.exp(-0.5 * (dy * dy * inv2[0] + dz * dz * inv2[1]))
        g = g / g.sum(1, keepdim=True).clamp_min(1e-300)
        w = torch.cat([g * (1 - self.t)[:, None], g * self.t[:, None]], 1)
        return w * self.valid

    def row_amplitude(self) -> torch.Tensor:
        return torch.exp(self.log_s[self.grain] + self.p * self.log_F[self.refl]) * self.w_geom

    def init_row_observed(self, recorded: "RecordedPixels") -> torch.Tensor:
        _, v = recorded.lookup(self.support_key)
        v = torch.tensor(v, device=self.t.device, dtype=self.t.dtype)
        return v[self.idx.long()].mul(self.psf_weights()).sum(1)

    def forward(self) -> torch.Tensor:
        contrib = self.row_amplitude()[:, None] * self.psf_weights()
        mu = torch.zeros(self.n_support, dtype=contrib.dtype, device=contrib.device)
        mu = mu.index_add_(0, self.idx.reshape(-1).long(), contrib.reshape(-1))
        if self.median_radius > 0:
            ext = torch.cat([mu, mu.new_zeros(1)])
            mu = ext[self.med_idx].median(dim=1).values
        return mu


def _median_index(support_key: np.ndarray, n_y: int, n_z: int, radius: int) -> np.ndarray:
    """(n_support, (2r+1)^2) same-frame neighbour indices into the support; n_support = outside (zero)."""
    mo = np.arange(-radius, radius + 1)
    my, mz = np.meshgrid(mo, mo, indexing="ij"); my = my.ravel(); mz = mz.ravel()
    sk = support_key
    fr_, rem = np.divmod(sk, n_y * n_z); yy, zz = np.divmod(rem, n_z)
    ny_ = yy[:, None] + my[None]; nz_ = zz[:, None] + mz[None]
    inside = (ny_ >= 0) & (ny_ < n_y) & (nz_ >= 0) & (nz_ < n_z)
    nk = (fr_[:, None] * n_y + ny_) * n_z + nz_
    pos = np.clip(np.searchsorted(sk, nk), 0, sk.size - 1)
    found = inside & (sk[pos] == nk)
    return np.where(found, pos, sk.size)


class SplatBlurModel(torch.nn.Module):
    """Predicted image as (sharp splat of every row) convolved with a separable Gaussian PSF.

    Stage 1 (linear in the amplitudes): each row is split bilinearly over the 4
    pixels around its fractional (y, z), and over frames by ``frame_model``:

    - ``"gauss"`` (default): the reflection crosses the Bragg condition at its
      fractional frame and lands in the frame containing it (the C code's
      ``floor``), widened by a Gaussian ω-width σ_ω (frames, fitted) integrated
      over the frame bins [j, j+1) for j = floor-1..floor+1, renormalised.
    - ``"linear"``: split between the two frames whose centres bracket it. This
      spreads a sharp reflection over two frames and is kept only for comparison;
      on real NF data it misallocates 20-40 % of many spots (step3b_omega).

    With triangle sub-point rows (:func:`build_contributions` ``edge_length=...``)
    the spatial splat is the voxel's projected footprint. Stage 2: separable
    Gaussian blur (σ_y, σ_z) within each frame, on the sharp pixel set dilated by
    ``blur_radius``. Stage 3 (optional): the reduction's same-frame median.
    """

    def __init__(self, table: ContributionTable, n_frames: int, n_y: int, n_z: int,
                 blur_radius: int = 4, sigma_psf=1.0, p: float = 2.0,
                 median_radius: int = 0, frame_model: str = "gauss", sigma_omega: float = 0.2,
                 device="cpu", dtype=torch.float64):
        super().__init__()
        if frame_model not in ("gauss", "linear"):
            raise ValueError("frame_model must be 'gauss' or 'linear'")
        self.n_y, self.n_z, self.n_frames, self.r = n_y, n_z, n_frames, blur_radius
        self.frame_model = frame_model
        dev = torch.device(device)
        y0 = np.floor(table.y).astype(np.int64); z0 = np.floor(table.z).astype(np.int64)
        ay = table.y - y0; az = table.z - z0
        if frame_model == "gauss":
            fb = np.floor(table.frame).astype(np.int64) - 1                  # slots floor-1, floor, floor+1
        else:
            fb = np.floor(table.frame - 0.5).astype(np.int64)                # slots f0, f0+1 (third unused)
        keys, sw = [], []
        for df in (0, 1, 2):
            for dy, wy in ((0, 1 - ay), (1, ay)):
                for dz, wz in ((0, 1 - az), (1, az)):
                    fr, yy, zz = fb + df, y0 + dy, z0 + dz
                    ok = (fr >= 0) & (fr < n_frames) & (yy >= 0) & (yy < n_y) & (zz >= 0) & (zz < n_z)
                    keys.append(np.where(ok, (fr * n_y + yy) * n_z + zz, -1))
                    if df == 0:
                        sw.append(wy * wz)
        keys = np.stack(keys, 1)                                              # (R, 12): frame-major
        valid = keys >= 0
        self.sharp_key, inv = np.unique(keys[valid], return_inverse=True)
        idx = np.zeros(keys.shape, np.int64); idx[valid] = inv.ravel()
        del keys
        off = np.arange(-blur_radius, blur_radius + 1)
        def shift(k, d, axis):
            f, rem = np.divmod(k, n_y * n_z); yy, zz = np.divmod(rem, n_z)
            if axis == "y":
                yy = yy[:, None] + d[None]; zz = np.broadcast_to(zz[:, None], yy.shape)
            else:
                zz = zz[:, None] + d[None]; yy = np.broadcast_to(yy[:, None], zz.shape)
            ok = (yy >= 0) & (yy < n_y) & (zz >= 0) & (zz < n_z)
            return np.where(ok, (f[:, None] * n_y + yy) * n_z + zz, -1)
        A = np.unique(shift(self.sharp_key, off, "y")); A = A[A >= 0]
        S = np.unique(shift(A, off, "z")); S = S[S >= 0]
        self.support_key = S
        def nbr(targets, source, axis):
            k = shift(targets, -off, axis)
            pos = np.clip(np.searchsorted(source, k), 0, source.size - 1)
            return np.where((k >= 0) & (source[pos] == k), pos, source.size)
        self.register_buffer("idx", torch.tensor(idx.astype(np.int32), device=dev))
        self.register_buffer("valid", torch.tensor(valid, device=dev))
        self.register_buffer("sw", torch.tensor(np.stack(sw, 1), device=dev, dtype=dtype))        # (R, 4)
        self.register_buffer("fpos", torch.tensor(table.frame - fb, device=dev, dtype=dtype))     # frame pos rel. slot 0
        self.register_buffer("nbr_y", torch.tensor(nbr(A, self.sharp_key, "y").astype(np.int64), device=dev))
        self.register_buffer("nbr_z", torch.tensor(nbr(S, A, "z").astype(np.int64), device=dev))
        self.register_buffer("off", torch.tensor(off, device=dev, dtype=dtype))
        self.register_buffer("grain", torch.tensor(table.grain, device=dev))
        self.register_buffer("refl", torch.tensor(table.refl, device=dev))
        self.register_buffer("w_geom", torch.tensor(table.w_geom, device=dev, dtype=dtype))
        self.register_buffer("log_F", torch.tensor(table.log_F, device=dev, dtype=dtype))
        self.n_A = int(A.size)
        self.median_radius = int(median_radius)
        if self.median_radius > 0:
            self.register_buffer("med_idx", torch.tensor(_median_index(S, n_y, n_z, self.median_radius), device=dev))
        sy, sz = (sigma_psf, sigma_psf) if np.isscalar(sigma_psf) else tuple(sigma_psf)
        self.log_s = torch.nn.Parameter(torch.zeros(table.n_grains, device=dev, dtype=dtype))
        self.p = torch.nn.Parameter(torch.tensor(float(p), device=dev, dtype=dtype))
        hi = _psf_hi(blur_radius)
        self.log_sigma_psf = torch.nn.Parameter(torch.tensor([_unbox(math.log(float(v)), _LOG_SIG_LO, hi) for v in (sy, sz)],
                                                             device=dev, dtype=dtype))
        self._om_lo, self._om_hi = math.log(1e-3), math.log(5.0)
        self.log_sigma_omega = torch.nn.Parameter(
            torch.tensor(_unbox(math.log(sigma_omega), self._om_lo, self._om_hi), device=dev, dtype=dtype),
            requires_grad=(frame_model == "gauss"))

    def sigma_psf_now(self) -> torch.Tensor:
        return torch.exp(_box(self.log_sigma_psf, _LOG_SIG_LO, _psf_hi(self.r)))

    def sigma_omega_now(self) -> torch.Tensor:
        return torch.exp(_box(self.log_sigma_omega, self._om_lo, self._om_hi))

    chunk_rows: int = 4_000_000       # rows per chunk in sharp(); bounds peak memory on large maps

    def frame_weights(self, a: int = 0, b: Optional[int] = None) -> torch.Tensor:
        """(rows a:b, 3) weights of frame slots 0..2."""
        fpos = self.fpos[a:b]
        if self.frame_model == "linear":
            t = fpos - 0.5
            return torch.stack([1 - t, t, torch.zeros_like(t)], 1)
        s = self.sigma_omega_now()
        edges = torch.arange(4, device=fpos.device, dtype=fpos.dtype)       # slot j covers [j, j+1)
        cdf = torch.special.ndtr((edges[None, :] - fpos[:, None]) / s)
        w = cdf[:, 1:] - cdf[:, :-1]
        return w / w.sum(1, keepdim=True).clamp_min(1e-30)

    def w_rows(self, a: int = 0, b: Optional[int] = None) -> torch.Tensor:
        """(rows a:b, 12) sharp weights: frame slot x bilinear, zero off-detector."""
        fw = self.frame_weights(a, b)
        return (fw[:, :, None] * self.sw[a:b, None, :]).reshape(fw.shape[0], 12) * self.valid[a:b]

    @property
    def w(self) -> torch.Tensor:
        return self.w_rows()

    @property
    def n_support(self) -> int:
        return int(self.support_key.size)

    def row_amplitude(self) -> torch.Tensor:
        return torch.exp(self.log_s[self.grain] + self.p * self.log_F[self.refl]) * self.w_geom

    def kernels(self):
        g = torch.exp(-0.5 * self.off[None, :] ** 2 * (self.sigma_psf_now() ** -2)[:, None])   # (2, 2r+1)
        return g / g.sum(1, keepdim=True)

    def _sharp_chunk(self, amp_c: torch.Tensor, a: int, b: int) -> torch.Tensor:
        img = torch.zeros(self.sharp_key.size, dtype=amp_c.dtype, device=amp_c.device)
        return img.index_add_(0, self.idx[a:b].reshape(-1).long(), (amp_c[:, None] * self.w_rows(a, b)).reshape(-1))

    def sharp(self) -> torch.Tensor:
        amp = self.row_amplitude()
        R = amp.shape[0]
        if R <= self.chunk_rows:
            return self._sharp_chunk(amp, 0, R)
        from torch.utils.checkpoint import checkpoint
        img = None
        for a in range(0, R, self.chunk_rows):
            b = min(R, a + self.chunk_rows)
            part = checkpoint(self._sharp_chunk, amp[a:b], a, b, use_reentrant=False) if torch.is_grad_enabled() \
                else self._sharp_chunk(amp[a:b], a, b)
            img = part if img is None else img + part
        return img

    def forward(self) -> torch.Tensor:
        gy, gz = self.kernels()
        s = self.sharp(); s = torch.cat([s, s.new_zeros(1)])
        a = (s[self.nbr_y] * gy[None, :]).sum(1)
        a = torch.cat([a, a.new_zeros(1)])
        mu = (a[self.nbr_z] * gz[None, :]).sum(1)
        if self.median_radius > 0:
            ext = torch.cat([mu, mu.new_zeros(1)])
            mu = ext[self.med_idx].median(dim=1).values
        return mu

    def init_row_observed(self, recorded: "RecordedPixels") -> torch.Tensor:
        """Per row: recorded value at its sharp pixels, weighted by its splat weights (for initialisation)."""
        _, v = recorded.lookup(self.sharp_key)
        v = torch.tensor(v, device=self.sw.device, dtype=self.sw.dtype)
        R = self.idx.shape[0]; out = []
        for a in range(0, R, self.chunk_rows):
            b = min(R, a + self.chunk_rows)
            out.append((v[self.idx[a:b].long()] * self.w_rows(a, b)).sum(1))
        return torch.cat(out)


@dataclass
class IntensityFit:
    scales: np.ndarray       # (G,) fitted s_g (absolute, flux-degenerate)
    p: float
    sigma_psf: float         # geometric mean of (sigma_y, sigma_z)
    sigma_noise: float
    nll: float
    n_support: int
    n_recorded_in_support: int
    grains_seen: np.ndarray  # (G,) bool, grain has >= 1 row
    sigma_psf_yz: tuple = (float("nan"), float("nan"))
    noise_gain: float = 0.0  # g in sigma^2 = sigma_noise^2 + g * max(mu, 0); 0 = constant noise
    sigma_omega: float = float("nan")  # frames; SplatBlurModel frame_model="gauss" only
    frame_model: str = "linear"

    def pixel_sigma(self, mu: np.ndarray) -> np.ndarray:
        return np.sqrt(self.sigma_noise ** 2 + self.noise_gain * np.maximum(mu, 0.0))

    def relative_scales(self) -> np.ndarray:
        s = np.where(self.grains_seen, self.scales, np.nan)
        return s / np.exp(np.nanmean(np.log(s)))


def _make_model(kind, table, n_frames, n_y, n_z, radius, sigma, p, median_radius, dev, dt,
                frame_model="gauss", sigma_omega=0.2):
    if kind == "psf":
        return IntensityModel(table, n_frames, n_y, n_z, psf_radius=radius, sigma_psf=sigma, p=p,
                              median_radius=median_radius, device=dev, dtype=dt)
    if kind == "splat":
        return SplatBlurModel(table, n_frames, n_y, n_z, blur_radius=radius, sigma_psf=sigma, p=p,
                              median_radius=median_radius, frame_model=frame_model,
                              sigma_omega=sigma_omega, device=dev, dtype=dt)
    raise ValueError("model_kind must be 'psf' or 'splat'")


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(x)


def fit_intensity(
    table: ContributionTable,
    recorded: RecordedPixels,
    n_frames: int, n_y: int, n_z: int,
    *,
    blanket: float,
    fit_p: bool = True,
    p_init: float = 2.0,
    sigma_psf_init=1.0,
    fit_sigma_psf: bool = True,
    sigma_noise_init: float = 2.0,
    psf_radius: int = 2,
    median_radius: int = 0,
    exclude_keys: Optional[np.ndarray] = None,
    tie_scales: bool = False,
    noise_model: str = "const",
    noise_gain_init: float = 1.0,
    model_kind: str = "psf",
    frame_model: str = "gauss",
    sigma_omega_init: float = 0.2,
    max_iter: int = 300,
    device="cpu",
    dtype=torch.float64,
    verbose: bool = False,
) -> IntensityFit:
    """Censored maximum-likelihood fit of per-grain scales (and optionally p, PSF width).

    ``median_radius`` should match the reduction's ``MedFiltRadius``.
    Recorded values are taken as ``x - blanket`` with ``x ~ N(mu, sigma)``; a pixel
    of the support is recorded iff ``x > blanket``. Recorded pixels outside the
    support are ignored (they belong to grains or noise not in the model).

    ``model_kind="splat"`` uses :class:`SplatBlurModel` (``psf_radius`` = blur radius),
    the right choice for triangle sub-point tables; ``"psf"`` the per-row Gaussian splat.
    ``noise_model="het"`` fits ``sigma^2 = sigma0^2 + g * max(mu, 0)`` (Poisson-like
    growth with signal) instead of a constant sigma.
    ``exclude_keys``: pixel keys dropped from the likelihood (e.g. pixels that
    grains outside the fitted set also light). ``tie_scales``: one scale shared by
    all grains (a baseline model).
    """
    if tie_scales:
        t1 = ContributionTable(np.zeros_like(table.grain), table.refl, table.sol, table.frame, table.y,
                               table.z, table.w_geom, table.log_F, 1)
        f1 = fit_intensity(t1, recorded, n_frames, n_y, n_z, blanket=blanket, fit_p=fit_p, p_init=p_init,
                           sigma_psf_init=sigma_psf_init, fit_sigma_psf=fit_sigma_psf,
                           sigma_noise_init=sigma_noise_init, psf_radius=psf_radius,
                           median_radius=median_radius, exclude_keys=exclude_keys, max_iter=max_iter,
                           noise_model=noise_model, noise_gain_init=noise_gain_init, dtype=dtype,
                           model_kind=model_kind, frame_model=frame_model, sigma_omega_init=sigma_omega_init,
                           device=device, verbose=verbose)
        seen = np.bincount(table.grain, minlength=table.n_grains) > 0
        f1.scales = np.full(table.n_grains, f1.scales[0]); f1.grains_seen = seen
        return f1
    dev = torch.device(device); dt = dtype
    mdl = _make_model(model_kind, table, n_frames, n_y, n_z, psf_radius, sigma_psf_init, p_init,
                      median_radius, dev, dt, frame_model=frame_model, sigma_omega=sigma_omega_init)
    has_omega = isinstance(mdl, SplatBlurModel) and mdl.frame_model == "gauss"
    hit, val = recorded.lookup(mdl.support_key)
    use = np.ones(mdl.n_support, bool) if exclude_keys is None else ~np.isin(mdl.support_key, exclude_keys)
    hit_t = torch.tensor(hit, device=dev); use_t = torch.tensor(use, device=dev)
    x = torch.tensor(val + blanket, device=dev, dtype=dt)
    log_sig = torch.nn.Parameter(torch.tensor(math.log(sigma_noise_init), device=dev, dtype=dt))
    mdl.p.requires_grad_(fit_p); mdl.log_sigma_psf.requires_grad_(fit_sigma_psf)
    # initialise scales so the predicted recorded-pixel sum matches the observed, per grain
    with torch.no_grad():
        amp = mdl.row_amplitude()
        num = torch.zeros(table.n_grains, dtype=dt, device=dev)
        tot = mdl.init_row_observed(recorded)
        den = torch.zeros_like(num).index_add_(0, mdl.grain, amp)
        num.index_add_(0, mdl.grain, tot)
        seen = den > 0
        mdl.log_s[seen] = torch.log((num[seen] / den[seen]).clamp_min(1e-12) * 2.0)
    if noise_model not in ("const", "het"):
        raise ValueError("noise_model must be 'const' or 'het'")
    log_g = torch.nn.Parameter(torch.tensor(math.log(max(noise_gain_init, 1e-6)), device=dev, dtype=dt),
                               requires_grad=(noise_model == "het"))
    extra = (mdl.log_sigma_omega,) if has_omega else ()
    params = [q for q in (mdl.log_s, mdl.p, mdl.log_sigma_psf, log_sig, log_g) + extra if q.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
    n_rec = int((hit & use).sum())
    rec_t, un_t = hit_t & use_t, (~hit_t) & use_t
    n_use = max(int(use.sum()), 1)

    def sigma_of(mu):
        if noise_model == "const":
            return torch.exp(log_sig).expand_as(mu)
        return torch.sqrt(torch.exp(2 * log_sig) + torch.exp(log_g) * torch.clamp(mu, min=0.0))

    def nll():
        mu = mdl()
        sig = sigma_of(mu)
        zr = (x[rec_t] - mu[rec_t]) / sig[rec_t]
        ll_rec = (-0.5 * zr ** 2 - torch.log(sig[rec_t]) - 0.5 * math.log(2 * math.pi)).sum()
        ll_un = _log_ndtr((blanket - mu[un_t]) / sig[un_t]).sum()
        return -(ll_rec + ll_un) / n_use

    def closure():
        opt.zero_grad()
        loss = nll(); loss.backward(); return loss

    for it in range(3):
        loss = opt.step(closure)
        if verbose:
            print(f"[intensity] round {it}: nll/px {float(loss):.6f} p {float(mdl.p.detach()):.4f} "
                  f"sigma_psf {[round(float(v), 3) for v in mdl.sigma_psf_now().detach()]} sigma {float(torch.exp(log_sig.detach())):.3f}"
                  + (f" gain {float(torch.exp(log_g.detach())):.4f}" if noise_model == "het" else "")
                  + (f" sigma_omega {float(mdl.sigma_omega_now().detach()):.3f}" if has_omega else ""))
    with torch.no_grad():
        final = float(nll())
    return IntensityFit(
        scales=torch.exp(mdl.log_s).detach().cpu().numpy(), p=float(mdl.p.detach()),
        sigma_psf=float(torch.exp(torch.log(mdl.sigma_psf_now().detach()).mean())), sigma_noise=float(torch.exp(log_sig.detach())),
        sigma_psf_yz=tuple(float(v) for v in mdl.sigma_psf_now().detach().cpu()),  # at 4*radius = railed
        nll=final, n_support=mdl.n_support, n_recorded_in_support=n_rec,
        grains_seen=seen.cpu().numpy(),
        noise_gain=float(torch.exp(log_g.detach())) if noise_model == "het" else 0.0,
        sigma_omega=float(mdl.sigma_omega_now().detach()) if has_omega else float("nan"),
        frame_model=mdl.frame_model if isinstance(mdl, SplatBlurModel) else "linear")


def heldout_nll(
    table: ContributionTable,
    recorded: RecordedPixels,
    fit: IntensityFit,
    n_frames: int, n_y: int, n_z: int,
    *,
    blanket: float,
    psf_radius: int = 2,
    median_radius: int = 0,
    exclude_keys: Optional[np.ndarray] = None,
    eval_keys: Optional[np.ndarray] = None,
    model_kind: str = "psf",
    device="cpu",
    dtype=torch.float64,
):
    """Score a fitted model on another reflection set (nothing refitted).

    Returns ``(keys, nll, mu, recorded_mask)`` per pixel, excluded pixels removed;
    ``nll`` is the Tobit negative log-likelihood of each pixel. ``eval_keys`` scores a
    given (sorted) pixel set instead of the model's own support, with mu = 0 outside
    it — needed to compare models whose supports differ.
    """
    dev = torch.device(device); dt = dtype
    syz = fit.sigma_psf_yz if np.all(np.isfinite(fit.sigma_psf_yz)) else fit.sigma_psf
    som = fit.sigma_omega if np.isfinite(fit.sigma_omega) else 0.2
    mdl = _make_model(model_kind, table, n_frames, n_y, n_z, psf_radius, syz, fit.p, median_radius, dev, dt,
                      frame_model=fit.frame_model if model_kind == "splat" else "linear", sigma_omega=som)
    with torch.no_grad():
        s = np.where(fit.scales > 0, fit.scales, 1e-300)
        mdl.log_s[:] = torch.tensor(np.log(s), device=dev, dtype=dt)
        mu = mdl().cpu().numpy()
    keys = mdl.support_key
    if eval_keys is not None:
        pos = np.clip(np.searchsorted(keys, eval_keys), 0, keys.size - 1)
        mu = np.where(keys[pos] == eval_keys, mu[pos], 0.0); keys = np.asarray(eval_keys)
    hit, val = recorded.lookup(keys)
    from scipy.special import log_ndtr
    sig = fit.pixel_sigma(mu)
    nll = np.where(hit, 0.5 * ((val + blanket - mu) / sig) ** 2 + np.log(sig) + 0.5 * math.log(2 * math.pi),
                   -log_ndtr((blanket - mu) / sig))
    keep = np.ones(mu.size, bool) if exclude_keys is None else ~np.isin(keys, exclude_keys)
    return keys[keep], nll[keep], mu[keep], hit[keep]


# ---------------------------------------------------------------------------
#  Spot-level censored fit
# ---------------------------------------------------------------------------

def spot_ownership(model: "SplatBlurModel", table: ContributionTable, radius: float = 6.0) -> np.ndarray:
    """Assign every support pixel of ``model`` to one spot id (grain, refl, sol), or -1.

    Each sharp pixel is labelled by the spot contributing the most sharp weight to
    it; every support pixel then takes the label of the nearest labelled sharp
    pixel in the same frame (within ``radius`` px). Spot id = (grain*M + refl)*2 + sol.
    """
    from scipy.spatial import cKDTree
    M = table.log_F.size
    spot = (table.grain.astype(np.int64) * M + table.refl) * 2 + table.sol
    SPK = int(spot.max()) + 1
    R = model.idx.shape[0]; ks, ws = [], []
    with torch.no_grad():
        for a in range(0, R, model.chunk_rows):                            # chunked: (R, 12) never held at once
            b = min(R, a + model.chunk_rows)
            w = (model.w_rows(a, b) * model.w_geom[a:b, None]).cpu().numpy()
            idx = model.idx[a:b].long().cpu().numpy(); ok = w > 0
            k = idx[ok].astype(np.int64) * SPK + np.repeat(spot[a:b], 12).reshape(idx.shape)[ok]
            uk, inv_ = np.unique(k, return_inverse=True); ks.append(uk); ws.append(np.bincount(inv_.ravel(), w[ok]))
    ck, inv = np.unique(np.concatenate(ks), return_inverse=True)
    cs = np.bincount(inv.ravel(), np.concatenate(ws))
    cpx, csp = np.divmod(ck, SPK)
    o = np.lexsort((-cs, cpx)); first = np.r_[True, np.diff(cpx[o]) != 0]
    lab_px, lab_sp = cpx[o][first], csp[o][first]                         # sharp pixel index -> spot
    ny, nz = model.n_y, model.n_z
    def coords(keys):
        f, rem = np.divmod(keys, ny * nz); y, z = np.divmod(rem, nz)
        return np.c_[f * 1e6, y, z].astype(np.float64)                    # frames never neighbour each other
    tree = cKDTree(coords(model.sharp_key[lab_px]))
    d, j = tree.query(coords(model.support_key), distance_upper_bound=radius)
    return np.where(np.isfinite(d), lab_sp[np.minimum(j, lab_sp.size - 1)], -1)


def _censored_mean_var(mu: torch.Tensor, sig: torch.Tensor, b: float):
    """Mean and variance of (x - b)^+ for x ~ N(mu, sig)."""
    d = (mu - b) / sig
    Phi = torch.special.ndtr(d); phi = torch.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    m1 = (mu - b) * Phi + sig * phi
    m2 = ((mu - b) ** 2 + sig ** 2) * Phi + (mu - b) * sig * phi
    return m1, (m2 - m1 ** 2).clamp_min(1e-12)


@dataclass
class Mixture:
    """Per-site orientation mixture for :func:`fit_spot_intensity`.

    A site (voxel) carries ``n_cand`` candidate orientations, each tabulated as its own
    rows with its own grain label (so spot ownership keeps the candidates apart); the rows
    of candidate k at site i are multiplied by the volume fraction f[i, k] = softmax over k.
    Rows with ``row_comp == -1`` are pure (fraction 1), e.g. neighbour grains.
    ``scale_group`` maps each table grain label to a shared scale, so a twin candidate
    carries its parent's per-voxel scattering. ``pairs`` (P, 2) of neighbouring sites add
    ``tv_lambda`` x mean over pairs of the smoothed L1 difference of their fractions.
    """
    row_comp: np.ndarray              # (R,) int64 component index, -1 = pure row
    comp_site: np.ndarray             # (C,) int64 site 0..S-1
    comp_cand: np.ndarray             # (C,) int64 candidate 0..n_cand-1
    n_cand: int
    scale_group: np.ndarray           # (n_grains,) int64
    pairs: Optional[np.ndarray] = None
    tv_lambda: float = 0.0
    tv_eps: float = 1e-3
    init_logits: Optional[np.ndarray] = None     # (S, n_cand); default: candidate 0 at 0, the rest at init_other
    init_other: float = -3.0
    stage1_kappa: float = 0.3        # kappa (and the start noise sigma) held fixed in the first optimisation stage
    logit_bound: Optional[float] = None   # fractions = softmax(B tanh(u/B)): no logit beyond +-B (keeps gradients alive)
    gauge_fix: bool = True                # candidate 0's logit pinned at 0: removes softmax's exact flat direction
    logit_wall: float = 15.0              # soft wall: zero penalty for |logit| < wall (fraction ~3e-7 is already absent),
    logit_wall_weight: float = 1.0        # quadratic beyond it, so no direction is unbounded for the line search.
                                          # (A plain ridge BIASES: it pulls every logit toward the map candidate.)

    @property
    def n_sites(self) -> int:
        return int(self.comp_site.max()) + 1


@dataclass
class SpotFit:
    scales: np.ndarray
    p: float
    sigma_noise: float
    kappa: float
    nll: float
    n_spots: int
    spot_ids: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    spot_nll: np.ndarray = None     # per-spot 0.5*((O-E)^2/V + log V)
    sigma_psf_yz: tuple = None      # PSF widths used (fitted sigma_z if fit_sigma_z)
    fractions: np.ndarray = None    # (S, n_cand) mixture fractions (mixture fits)
    logits: np.ndarray = None       # (S, n_cand) for warm starts
    tv: float = None                # TV penalty value (unweighted)


def fit_spot_intensity(
    table: ContributionTable, recorded: RecordedPixels, n_frames: int, n_y: int, n_z: int, *,
    blanket: float, sigma_psf=(1.0, 2.6), sigma_omega: float = 0.16, blur_radius: int = 5,
    median_radius: int = 0, exclude_keys: Optional[np.ndarray] = None, fit_p: bool = True, p_init: float = 1.0,
    fit_kappa: bool = True, kappa_init: float = 0.1, sigma_noise_init: float = 10.0, tie_scales: bool = False,
    min_pixels: int = 3, max_iter: int = 300, device="cpu", dtype=torch.float64, verbose: bool = False,
    scales: Optional[np.ndarray] = None, likelihood: str = "gauss_kappa", t_dof: float = 4.0,
    unit: str = "spot", fit_sigma_z: bool = False, mixture: Optional[Mixture] = None,
    fit_sigma_noise: bool = True,
) -> SpotFit:
    """Censored fit at the level of whole spots, the spot shape held FIXED.

    Pixel means come from :class:`SplatBlurModel` with the given (fixed) PSF and
    ω-width; each support pixel is owned by one spot (:func:`spot_ownership`).
    Per spot: observed O = sum of recorded values (above blanket) over its pixels,
    expected E = sum of E[(x - b)^+], variance V = sum Var[(x - b)^+] + (kappa E)^2.
    Loss = sum over spots of (O - E)^2 / V + log V. Redistributing light inside a
    spot or across its frames leaves O and (to first order) E unchanged, so the
    fitted |F| exponent and grain scales are insensitive to pixel-level shape error.
    ``scales`` given (with ``max_iter=0``) evaluates a fixed model, e.g. on held-out rings.

    ``unit="column"``: the likelihood unit is (spot, detector column y) instead of the whole spot —
    the spot's ALONG-STREAK PROFILE, summed over rows (z) and frames. The horizontal detector axis maps
    to position along the projection direction, so a profile sees WHERE inside a spot the light is
    (e.g. a twin gap) while summing over z integrates out vertical blooming. With the log-t
    likelihood, near-empty columns (expected and observed both ~0) contribute ~nothing, so no
    column cut is applied. ``spot_ids`` then holds ``spot * n_y + y``.
    ``fit_sigma_z``: fit the vertical PSF width (sigma_y stays fixed). Summing over z removes the vertical
    SHAPE from a profile but not its effect on censoring (a taller spot loses more light below the
    blanket), so sigma_z biases the |F| exponent unless it is measured; this measures it from the data.
    ``likelihood="logt"``: Student-t (``t_dof``) on r = log((O + c)/(E + c)) with fitted scale
    ``kappa`` (c = 1 count), for multiplicative, heavy-tailed per-spot scatter; the pixel noise
    sigma then only enters E through the censoring. ``tie_scales``: one scale shared by all
    grains, spots still owned per grain.
    ``fit_sigma_noise=False`` holds the pixel-noise sigma at ``sigma_noise_init``: measure it from pure noise
    (blank frames through the same reduction, e.g. a 3x3 median), never from the fit's own residuals. Free, it can
    collapse to ~0 (hard censoring), which the empty units reward.
    ``mixture``: fit per-site orientation fractions (see :class:`Mixture`); ``scales``, if given, are then
    per scale group. The objective adds the mixture's TV term; ``nll`` reports the data term only.
    """
    dev = torch.device(device); dt = dtype
    if mixture is not None and tie_scales:
        raise ValueError("mixture and tie_scales are exclusive")
    if likelihood not in ("gauss_kappa", "logt"):
        raise ValueError("likelihood must be 'gauss_kappa' or 'logt'")
    if unit not in ("spot", "column"):
        raise ValueError("unit must be 'spot' or 'column'")
    mdl = SplatBlurModel(table, n_frames, n_y, n_z, blur_radius=blur_radius, sigma_psf=sigma_psf, p=p_init,
                         median_radius=median_radius, frame_model="gauss", sigma_omega=sigma_omega,
                         device=dev, dtype=dt)
    mdl.log_sigma_psf.requires_grad_(fit_sigma_z); mdl.log_sigma_omega.requires_grad_(False)
    if fit_sigma_z:   # only the vertical width is free: zero the horizontal component's gradient
        mdl.log_sigma_psf.register_hook(lambda g: g * torch.tensor([0.0, 1.0], device=g.device, dtype=g.dtype))
    mdl.p.requires_grad_(fit_p)
    own = spot_ownership(mdl, table)
    use = own >= 0
    if exclude_keys is not None:
        use &= ~np.isin(mdl.support_key, exclude_keys)
    hit, val = recorded.lookup(mdl.support_key)
    obs_px = np.where(hit & use, val, 0.0)
    if unit == "column":
        ycol = (mdl.support_key // n_z) % n_y
        key = np.where(use, own.astype(np.int64) * n_y + ycol, -1)
    else:
        key = np.where(use, own, -1)
    us, uinv = np.unique(key, return_inverse=True)
    uinv = uinv.ravel()
    npx = np.bincount(uinv, minlength=us.size)
    keep_spot = (us >= 0) & (npx >= (1 if unit == "column" else min_pixels))
    O = torch.tensor(np.bincount(uinv, obs_px, minlength=us.size)[keep_spot], device=dev, dtype=dt)
    uinv_t = torch.tensor(uinv, device=dev); use_t = torch.tensor(use, device=dev)
    kmask = torch.tensor(keep_spot, device=dev)
    log_sig = torch.nn.Parameter(torch.tensor(math.log(sigma_noise_init), device=dev, dtype=dt),
                                 requires_grad=fit_sigma_noise)
    log_k = torch.nn.Parameter(torch.tensor(math.log(max(kappa_init, 1e-6)), device=dev, dtype=dt),
                               requires_grad=fit_kappa)
    if scales is not None:
        with torch.no_grad():
            mdl.log_s[:] = torch.tensor(np.log(np.maximum(scales, 1e-300)), device=dev, dtype=dt)
    else:
        with torch.no_grad():
            tot_obs = float(obs_px.sum()); mu0 = float(mdl().clamp_min(0).sum())
            mdl.log_s[:] = math.log(max(tot_obs, 1e-9) / max(mu0, 1e-9))
    logits = None
    if mixture is not None:
        mx = mixture
        grp = torch.tensor(np.asarray(mx.scale_group, np.int64), device=dev)
        n_grp = int(grp.max()) + 1
        rc = np.asarray(mx.row_comp, np.int64)
        if rc.size != len(table):
            raise ValueError("mixture.row_comp must have one entry per table row")
        pure_t = torch.tensor(rc < 0, device=dev)
        rsite = torch.tensor(np.where(rc >= 0, np.asarray(mx.comp_site)[np.maximum(rc, 0)], 0), device=dev)
        rcand = torch.tensor(np.where(rc >= 0, np.asarray(mx.comp_cand)[np.maximum(rc, 0)], 0), device=dev)
        S, K = mx.n_sites, int(mx.n_cand)
        if mx.init_logits is not None:
            L0 = torch.tensor(np.asarray(mx.init_logits), device=dev, dtype=dt)
        else:
            L0 = torch.full((S, K), float(mx.init_other), device=dev, dtype=dt); L0[:, 0] = 0.0
        # Free variables are float64 whatever the model dtype: a float32 L-BFGS update overflowed on flat directions
        # (Phase 2 v2: 'value cannot be converted to type float without overflow').
        L0 = L0.to(torch.float64)
        if mx.gauge_fix:
            L0 = L0[:, 1:] - L0[:, :1]                      # logits relative to candidate 0
        B = mx.logit_bound
        if B is not None:                                   # store the free variable u with B tanh(u/B) = L0
            L0 = B * torch.atanh(torch.clamp(L0 / B, -0.999, 0.999))
        logits = torch.nn.Parameter(L0)
        with torch.no_grad():
            s0 = float(mdl.log_s[0]) if scales is None else 0.0
        log_sg = torch.nn.Parameter(torch.full((n_grp,), s0, device=dev, dtype=torch.float64) if scales is None else
                                    torch.tensor(np.log(np.maximum(scales, 1e-300)), device=dev, dtype=torch.float64))
        mdl.log_s.requires_grad_(False)
        pairs_t = None if mx.pairs is None or len(mx.pairs) == 0 else torch.tensor(np.asarray(mx.pairs, np.int64), device=dev)

        def eff_logits():
            u = logits if mx.logit_bound is None else mx.logit_bound * torch.tanh(logits / mx.logit_bound)
            return torch.cat([torch.zeros_like(u[:, :1]), u], 1) if mx.gauge_fix else u

        def fractions():
            return torch.softmax(eff_logits(), 1).to(dt)

        def row_amp():
            f = fractions()[rsite, rcand]
            f = torch.where(pure_t, torch.ones_like(f), f)
            return torch.exp(log_sg[grp[mdl.grain]].to(dt) + mdl.p * mdl.log_F[mdl.refl]) * mdl.w_geom * f
        mdl.row_amplitude = row_amp

        def tv_term():
            if pairs_t is None:
                return torch.zeros((), device=dev, dtype=dt)
            f = fractions(); d = f[pairs_t[:, 0]] - f[pairs_t[:, 1]]
            return (torch.sqrt(d * d + mx.tv_eps ** 2) - mx.tv_eps).sum(1).mean()
    c_tie = None
    if tie_scales and scales is None:
        c_tie = torch.nn.Parameter(mdl.log_s.detach()[0].clone())
        mdl.log_s.requires_grad_(False)
        mdl.row_amplitude = lambda: torch.exp(c_tie + mdl.p * mdl.log_F[mdl.refl]) * mdl.w_geom

    def terms():
        mu = mdl()
        sig = torch.exp(log_sig).expand_as(mu)
        m1, v1 = _censored_mean_var(mu, sig, blanket)
        m1 = torch.where(use_t, m1, torch.zeros_like(m1)); v1 = torch.where(use_t, v1, torch.zeros_like(v1))
        E = torch.zeros(us.size, device=dev, dtype=dt).index_add_(0, uinv_t, m1)[kmask]
        V = torch.zeros(us.size, device=dev, dtype=dt).index_add_(0, uinv_t, v1)[kmask] + (torch.exp(log_k) * E) ** 2
        return E, V

    def per_spot(E, V):
        if likelihood == "gauss_kappa":
            return 0.5 * ((O - E) ** 2 / V + torch.log(V))
        k = torch.exp(log_k)
        r = torch.log((O + 1.0) / (E + 1.0))
        nu = t_dof
        return 0.5 * (nu + 1) * torch.log1p((r / k) ** 2 / nu) + torch.log(k) \
            - (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi))

    def nll():
        E, V = terms()
        return per_spot(E, V).mean()

    def objective():
        if logits is None:
            return nll()
        out = nll().to(torch.float64) + mixture.logit_wall_weight * \
            (torch.relu(logits.abs() - mixture.logit_wall) ** 2).mean()
        return out if mixture.tv_lambda == 0 else out + mixture.tv_lambda * tv_term()

    params = [q for q in (mdl.log_s, mdl.p, log_sig, log_k) if q.requires_grad] + ([c_tie] if c_tie is not None else []) \
        + ([mdl.log_sigma_psf] if fit_sigma_z else [])
    if scales is not None:
        params = [q for q in (log_sig, log_k) if q.requires_grad] if max_iter > 0 else []
    if logits is not None:
        params = params + [logits] + ([log_sg] if scales is None else [])
    if params and max_iter > 0:
        # Mixture fits start far from the answer (wrong fractions -> large residuals); with kappa and the noise
        # sigma free from the start the fit can slide into a dark state (E ~ 0, kappa -> 0, sigma -> 0) that the
        # empty units reward. Stage 1 holds both (kappa at >= stage1_kappa); the staged optimum has the LOWER NLL
        # (checked on the mixture test scene: -0.578 staged vs +0.008 collapsed).
        stages = [params]
        if logits is not None:
            with torch.no_grad():
                log_k.fill_(math.log(max(float(torch.exp(log_k)), mixture.stage1_kappa)))
            held = [q for q in params if q is not log_k and q is not log_sig]
            stages = [held, params] if len(held) < len(params) else [params]   # nothing to release -> one stage
        for st_params in stages:
            opt = torch.optim.LBFGS(st_params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)
            def closure():
                opt.zero_grad(); loss = objective(); loss.backward(); return loss
            for it in range(3):
                loss = opt.step(closure)
                if verbose:
                    print(f"[spot] round {it}: nll/spot {float(loss.detach()):.5f} p {float(mdl.p.detach()):.4f} "
                      f"sigma {float(torch.exp(log_sig.detach())):.3f} kappa {float(torch.exp(log_k.detach())):.4f}")
    with torch.no_grad():
        E, V = terms(); final = float(nll())
        per = per_spot(E, V).cpu().numpy()
        sc = (torch.exp(c_tie).expand(table.n_grains) if c_tie is not None else torch.exp(mdl.log_s)).detach().cpu().numpy()
        mix_out = {}
        if logits is not None:
            sc = torch.exp(log_sg).detach().cpu().numpy()
            mix_out = dict(fractions=fractions().cpu().numpy(), logits=eff_logits().detach().cpu().numpy(),
                           tv=float(tv_term()))
    return SpotFit(**mix_out, spot_nll=per, scales=sc, sigma_psf_yz=tuple(float(v) for v in mdl.sigma_psf_now().detach().cpu()), p=float(mdl.p.detach()),
                   sigma_noise=float(torch.exp(log_sig.detach())), kappa=float(torch.exp(log_k.detach())),
                   nll=final, n_spots=int(keep_spot.sum()), spot_ids=us[keep_spot],
                   observed=O.cpu().numpy(), expected=E.cpu().numpy())
