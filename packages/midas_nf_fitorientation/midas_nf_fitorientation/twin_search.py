"""Discrete twin-lamella search on an NF orientation map.

A hypothesis for grain G and a candidate twin orientation g is a straight strip (in-layer unit normal ``m``, offset
``o``, width ``w``, all um): each voxel's twin fraction ``t_v`` is the share of its triangle inside the strip, and on
that share the voxel's map orientation is replaced by ``g``. For a COHERENT Sigma3 twin the strip direction is fixed
by the twin plane: ``m`` is the in-layer projection of the parent's {111} normal (:func:`coherent_strip_normals`).

Each hypothesis is scored by the exact change of the censored log-t likelihood, computed LOCALLY. The model image is
linear in the row amplitudes up to the reduction's same-frame median, so only pixels within the median radius of the
grain's support change. No gradient is involved, so neither softmax saturation nor an L-BFGS overflow can arise.

STATUS: EXPERIMENTAL, validated on SYNTHETIC line-focus scenes only, and NOT yet on real data. A 0.2 deg error on the
parent orientation makes related Sigma3 candidates absorb the parent's misfit and creates false twins (decoys cannot
mimic it): refine the map orientations against the recorded light and require a twin to beat its own re-oriented
parent (see the Phase 2e/2f registrations of the AlON comparison). Twin acceptance thresholds must come from a null that
gives the parent the same freedom.

Likelihood unit = (frame, detector column y, z bin of ``zbin`` px). The grid does not depend on the model, so a unit
exists wherever light is recorded or predicted; a twin's own spots, absent from the map, still count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .intensity import ContributionTable, RecordedPixels, SplatBlurModel, _barycentric_subpoints


@dataclass
class Nuisance:
    """Detector and noise model, all FROZEN during the search (measure them first, e.g. on the pure map)."""
    sigma_psf: tuple = (1.0, 2.6)     # px (sigma_y, sigma_z)
    sigma_omega: float = 0.16         # frames
    blur_radius: int = 10
    median_radius: int = 1
    blanket: float = 5.0
    sigma_noise: float = 0.82         # post-median pure-noise sigma
    kappa: float = 0.5                # log-t scale of the per-unit multiplicative scatter
    t_dof: float = 4.0
    p: float = 2.0                    # |F| exponent


A111 = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float) / math.sqrt(3.0)


def sigma3_variants(om: np.ndarray) -> list:
    """The four Sigma3 twins of ``om`` (crystal->lab): om @ Rot(60 deg, <111>_k), k = 0..3."""
    from midas_stress.orientation import axis_angle_to_orient_mat
    return [np.asarray(om) @ np.asarray(axis_angle_to_orient_mat(a, 60.0)).reshape(3, 3) for a in A111]


def coherent_strip_normals(om: np.ndarray):
    """(4, 2) in-layer unit normals of a coherent twin strip for each variant, and (4,) |in-layer component|.

    The twin plane of variant k is the parent's (111)_k; its lab normal is om @ a_k. The trace on the layer (normal z)
    is perpendicular to the in-layer projection of that normal, which is therefore the strip normal. A small in-layer
    component (plane nearly parallel to the layer) means an ill-conditioned trace.
    """
    n = (np.asarray(om) @ A111.T).T[:, :2]
    mag = np.linalg.norm(n, axis=1)
    return n / np.maximum(mag, 1e-12)[:, None], mag


def _unit_ids(keys: np.ndarray, n_y: int, n_z: int, zbin: int) -> np.ndarray:
    f, rem = np.divmod(keys, n_y * n_z); y, z = np.divmod(rem, n_z)
    return (f * n_y + y) * (n_z // zbin + 1) + z // zbin


def _neighbour_keys(keys: np.ndarray, n_y: int, n_z: int, r: int):
    """(n, (2r+1)^2) same-frame neighbour keys, and an inside-detector mask."""
    mo = np.arange(-r, r + 1); my, mz = np.meshgrid(mo, mo, indexing="ij"); my = my.ravel(); mz = mz.ravel()
    f, rem = np.divmod(keys, n_y * n_z); yy, zz = np.divmod(rem, n_z)
    ny_ = yy[:, None] + my[None]; nz_ = zz[:, None] + mz[None]
    inside = (ny_ >= 0) & (ny_ < n_y) & (nz_ >= 0) & (nz_ < n_z)
    return (f[:, None] * n_y + ny_) * n_z + nz_, inside


def _positions(sorted_keys: np.ndarray, query: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Index of each query key in ``sorted_keys``; ``len(sorted_keys)`` (a zero pad slot) when absent."""
    if sorted_keys.size == 0:
        return np.full(query.shape, 0, np.int64)
    pos = np.clip(np.searchsorted(sorted_keys, query), 0, sorted_keys.size - 1)
    ok = sorted_keys[pos] == query
    if valid is not None:
        ok &= valid
    return np.where(ok, pos, sorted_keys.size)


def _censored_mean(mu: torch.Tensor, sig: float, b: float) -> torch.Tensor:
    d = (mu - b) / sig
    return (mu - b) * torch.special.ndtr(d) + sig * torch.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)


def _logt_terms(O: torch.Tensor, E: torch.Tensor, kappa: float, nu: float) -> torch.Tensor:
    """Per-unit log-t NLL up to constants (they cancel in differences)."""
    r = torch.log((O + 1.0) / (E + 1.0))
    return 0.5 * (nu + 1) * torch.log1p((r / kappa) ** 2 / nu)


def _median_last(x: torch.Tensor) -> torch.Tensor:
    return x.median(dim=-1).values


class BaseImage:
    """The map's linear (pre-median) model image, its median image, and the unit table (O, E)."""

    def __init__(self, table: ContributionTable, log_scales: np.ndarray, recorded: RecordedPixels, n_frames: int,
                 n_y: int, n_z: int, nuis: Nuisance, zbin: int = 16, device="cpu", dtype=torch.float32):
        self.n_frames, self.n_y, self.n_z, self.nuis, self.zbin = n_frames, n_y, n_z, nuis, zbin
        self.dev, self.dt = torch.device(device), dtype
        m = SplatBlurModel(table, n_frames, n_y, n_z, blur_radius=nuis.blur_radius, sigma_psf=nuis.sigma_psf,
                           p=nuis.p, median_radius=0, frame_model="gauss", sigma_omega=nuis.sigma_omega,
                           device=self.dev, dtype=dtype)
        with torch.no_grad():
            m.log_s[:] = torch.tensor(np.asarray(log_scales), device=self.dev, dtype=dtype)
            mu = m()
        self.key = m.support_key.copy()
        self.mu_lin = mu.detach()
        del m
        rk = recorded.key
        self.rec_unit = _unit_ids(rk, n_y, n_z, zbin); self.rec_val = recorded.value
        self._rebuild()

    def median_at(self, keys: np.ndarray) -> torch.Tensor:
        """Median image at ``keys`` (sorted or not) from the linear image (zero outside its support)."""
        r = self.nuis.median_radius
        if r == 0:
            pos = _positions(self.key, keys)
            return torch.cat([self.mu_lin, self.mu_lin.new_zeros(1)])[torch.tensor(pos, device=self.dev)]
        nk, inside = _neighbour_keys(keys, self.n_y, self.n_z, r)
        pos = _positions(self.key, nk, inside)
        ext = torch.cat([self.mu_lin, self.mu_lin.new_zeros(1)])
        return _median_last(ext[torch.tensor(pos, device=self.dev)])

    def _rebuild(self):
        nz = self.nuis
        # pixels whose median can be non-zero: the support dilated by the median radius
        if nz.median_radius > 0:
            nk, inside = _neighbour_keys(self.key, self.n_y, self.n_z, nz.median_radius)
            self.mkey = np.unique(nk[inside])
        else:
            self.mkey = self.key
        med = torch.cat([self.median_at(self.mkey[i:i + 20_000_000]) for i in range(0, self.mkey.size, 20_000_000)])
        E_pix = _censored_mean(med.double(), nz.sigma_noise, nz.blanket)
        u_pix = _unit_ids(self.mkey, self.n_y, self.n_z, self.zbin)
        self.ukey = np.union1d(np.unique(u_pix), np.unique(self.rec_unit))
        up = torch.tensor(np.searchsorted(self.ukey, u_pix), device=self.dev)
        self.E = torch.zeros(self.ukey.size, dtype=torch.float64, device=self.dev).index_add_(0, up, E_pix)
        ur = np.searchsorted(self.ukey, self.rec_unit)
        self.O = torch.tensor(np.bincount(ur, self.rec_val, minlength=self.ukey.size), dtype=torch.float64,
                              device=self.dev)

    def total_nll(self) -> float:
        n = self.nuis
        return float(_logt_terms(self.O, self.E, n.kappa, n.t_dof).sum())

    def commit(self, keys: np.ndarray, delta: torch.Tensor):
        """Add a linear-image change ``delta`` on ``keys`` (sorted) and rebuild the median image and units."""
        new = np.union1d(self.key, keys)
        mu = torch.zeros(new.size, dtype=self.mu_lin.dtype, device=self.dev)
        mu[torch.tensor(np.searchsorted(new, self.key), device=self.dev)] += self.mu_lin
        mu[torch.tensor(np.searchsorted(new, keys), device=self.dev)] += delta.to(mu.dtype)
        self.key, self.mu_lin = new, mu
        self._rebuild()


class LocalSearch:
    """Batched exact local dNLL for re-weighting a block of rows.

    ``table`` holds every row a hypothesis may re-weight: the MAP rows of the region voxels (block 0, the same rows,
    orientations and scales as in the base table) and one block per candidate orientation (blocks 1..C).
    A hypothesis is a weight vector W (R,) on these rows; its image change is linear in W. Typical use: W = -t_v on the
    map rows and +t_v on one candidate block.
    """

    def __init__(self, base: BaseImage, table: ContributionTable, log_scale_per_row: np.ndarray,
                 row_block: np.ndarray, row_voxel: np.ndarray):
        self.base = base; nz = base.nuis; dev, dt = base.dev, base.dt
        self.row_block = np.asarray(row_block); self.row_voxel = np.asarray(row_voxel)
        m = SplatBlurModel(table, base.n_frames, base.n_y, base.n_z, blur_radius=nz.blur_radius,
                           sigma_psf=nz.sigma_psf, p=nz.p, median_radius=0, frame_model="gauss",
                           sigma_omega=nz.sigma_omega, device=dev, dtype=dt)
        with torch.no_grad():
            self.w12 = m.w_rows().detach()                                   # (R, 12)
            self.idx = m.idx.long()
            self.gy, self.gz = [k.detach() for k in m.kernels()]
            self.amp0 = (torch.exp(torch.tensor(np.asarray(log_scale_per_row), device=dev, dtype=dt)
                                   + nz.p * m.log_F[m.refl]) * m.w_geom).detach()
        self.n_sharp = int(m.sharp_key.size); self.nbr_y = m.nbr_y; self.nbr_z = m.nbr_z; self.n_A = m.n_A
        self.S = m.support_key.copy()
        del m
        r = nz.median_radius
        if r > 0:
            nk, inside = _neighbour_keys(self.S, base.n_y, base.n_z, r)
            self.M = np.unique(nk[inside])
            nb, ins = _neighbour_keys(self.M, base.n_y, base.n_z, r)
        else:
            self.M = self.S; nb = self.M[:, None]; ins = np.ones(nb.shape, bool)
        self.pos_S = torch.tensor(_positions(self.S, nb, ins), device=dev)          # into delta (pad = len S)
        self.pos_B = torch.tensor(_positions(base.key, nb, ins), device=dev)        # into base (pad)
        uM = _unit_ids(self.M, base.n_y, base.n_z, base.zbin)
        ul, inv = np.unique(uM, return_inverse=True)
        self.inv = torch.tensor(inv.ravel(), device=dev); self.n_ul = ul.size
        # units a candidate's NEW spots reach may be absent from the base table (no base light, nothing recorded):
        # there O = 0 and E_base = 0 (refresh_base), and a hypothesis predicting light is penalised
        self.ul = ul
        self.refresh_base()

    def refresh_base(self):
        """Re-read the base image around this region (after a commit elsewhere)."""
        b = self.base; nz = b.nuis
        ext = torch.cat([b.mu_lin, b.mu_lin.new_zeros(1)])
        self.pos_B = torch.tensor(_positions(b.key, *self._nb_keys()), device=b.dev)
        self.base_nb = ext[self.pos_B]                                              # (nM, 9)
        med0 = _median_last(self.base_nb) if nz.median_radius > 0 else self.base_nb[:, 0]
        self.E0_pix = _censored_mean(med0.double(), nz.sigma_noise, nz.blanket)
        upos = np.searchsorted(b.ukey, self.ul)
        known = (upos < b.ukey.size) & (b.ukey[np.minimum(upos, b.ukey.size - 1)] == self.ul)
        up = torch.tensor(np.where(known, upos, 0), device=b.dev); kn = torch.tensor(known, device=b.dev)
        zero = torch.zeros(self.ul.size, dtype=torch.float64, device=b.dev)
        self.O = torch.where(kn, b.O[up], zero); self.E_base = torch.where(kn, b.E[up], zero)
        self.nll0 = _logt_terms(self.O, self.E_base, nz.kappa, nz.t_dof)

    def _nb_keys(self):
        r = self.base.nuis.median_radius
        if r > 0:
            return _neighbour_keys(self.M, self.base.n_y, self.base.n_z, r)
        return self.M[:, None], np.ones((self.M.size, 1), bool)

    def delta_image(self, W: torch.Tensor) -> torch.Tensor:
        """(B, nS) linear-image change for weights W (B, R)."""
        B = W.shape[0]
        amp = self.amp0[None] * W                                                   # (B, R)
        vals = (amp[:, :, None] * self.w12[None]).reshape(B, -1)
        sharp = torch.zeros(B, self.n_sharp + 1, dtype=amp.dtype, device=amp.device)
        sharp[:, :self.n_sharp].index_add_(1, self.idx.reshape(-1), vals)
        a = (sharp[:, self.nbr_y] * self.gy).sum(-1)                               # (B, nA)
        a = torch.cat([a, a.new_zeros(B, 1)], 1)
        return (a[:, self.nbr_z] * self.gz).sum(-1)                                # (B, nS)

    @torch.no_grad()
    def dnll(self, W: torch.Tensor) -> torch.Tensor:
        """(B,) exact change of the total log-t NLL for weights W (B, R)."""
        nz = self.base.nuis
        d = self.delta_image(W)
        d = torch.cat([d, d.new_zeros(d.shape[0], 1)], 1)
        x = self.base_nb[None] + d[:, self.pos_S]                                   # (B, nM, 9)
        med = _median_last(x) if nz.median_radius > 0 else x[..., 0]
        dE_pix = _censored_mean(med.double(), nz.sigma_noise, nz.blanket) - self.E0_pix[None]
        dE = torch.zeros(W.shape[0], self.n_ul, dtype=torch.float64, device=W.device).index_add_(1, self.inv, dE_pix)
        nll = _logt_terms(self.O[None], self.E_base[None] + dE, nz.kappa, nz.t_dof)
        return (nll - self.nll0[None]).sum(1)

    def commit(self, w: torch.Tensor):
        """Fold one hypothesis (weights (R,)) into the base image."""
        with torch.no_grad():
            d = self.delta_image(w[None])[0]
        self.base.commit(self.S, d)
        self.refresh_base()


def triangle_subpoints(xy: np.ndarray, ud: np.ndarray, edge: float, n: int = 4) -> np.ndarray:
    """(N, n^2, 2) centroids of the n^2 equal sub-triangles of each voxel triangle (MIDAS tri_vertices layout)."""
    from midas_diffract.forward import HEDMForwardModel
    V = HEDMForwardModel.tri_vertices(torch.tensor(xy, dtype=torch.float64), torch.full((len(xy),), float(edge),
                                      dtype=torch.float64), torch.tensor(ud, dtype=torch.float64)).numpy()[:, :, :2]
    return np.einsum("kj,njd->nkd", _barycentric_subpoints(n), V)


def strip_fractions(sub: torch.Tensor, m: torch.Tensor, o: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Twin fraction per voxel for B strips: sub (N, K, 2); m (B, 2); o, w (B,) -> (B, N)."""
    d = torch.einsum("nkd,bd->bnk", sub, m.to(sub.dtype))
    o = o.to(sub.dtype); w = w.to(sub.dtype)
    inside = (d >= o[:, None, None]) & (d < (o + w)[:, None, None])
    return inside.to(sub.dtype).mean(-1)


def residual_keys(base: BaseImage, recorded: RecordedPixels, dilate: int = 1) -> np.ndarray:
    """Sorted keys of recorded pixels the base map leaves UNEXPLAINED (base median image below the blanket),
    dilated by ``dilate`` px in y and z (same frame). Light a candidate orientation might explain."""
    med = torch.cat([base.median_at(recorded.key[i:i + 5_000_000]) for i in range(0, recorded.key.size, 5_000_000)])
    k = recorded.key[(med < base.nuis.blanket).cpu().numpy()]
    if dilate == 0:
        return k
    nk, inside = _neighbour_keys(k, base.n_y, base.n_z, dilate)
    return np.unique(nk[inside])


def _small_rotations(half: float, step: float) -> np.ndarray:
    """(n, 3, 3) rotations exp([v]x) for axis-angle vectors v on a cubic grid (degrees) of the given half-width."""
    from scipy.spatial.transform import Rotation
    g = np.arange(-half, half + 1e-9, step)
    v = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return Rotation.from_rotvec(np.radians(v)).as_matrix()


@torch.no_grad()
def refine_candidate(model, om0: np.ndarray, xy_vox: np.ndarray, res_keys: np.ndarray, n_frames: int, n_y: int,
                     n_z: int, half: float = 0.6, step: float = 0.2, fine_half: float = 0.1, fine_step: float = 0.05,
                     batch: int = 8):
    """Refine a candidate orientation near ``om0`` against unexplained recorded light.

    Score of an orientation g = number of (voxel, reflection, omega-solution) spot centres, predicted for every voxel
    in ``xy_vox``, that land on a pixel of ``res_keys`` (floor frame, rounded y and z). A coarse grid of small
    rotations (applied on the crystal side, om0 @ dR) and then a fine grid around the best. Truth-free: the same
    procedure refines decoys, so the decoy null carries the same freedom.
    Returns (om, score, score_at_om0).
    """
    from midas_stress.orientation import orient_mat_to_euler
    dev = model.hkls.device if hasattr(model, "hkls") else torch.device("cpu")
    rk = torch.tensor(res_keys, device=dev)
    pos = torch.tensor(np.c_[xy_vox, np.zeros(len(xy_vox))], device=dev, dtype=torch.float64)
    N = len(xy_vox)

    def scores(oms):
        out = []
        for b0 in range(0, len(oms), batch):
            ob = oms[b0:b0 + batch]; B = len(ob)
            eul = torch.tensor(np.array([np.asarray(orient_mat_to_euler(o)).ravel() for o in ob]), device=dev,
                               dtype=torch.float64)
            sp = model(eul.repeat_interleave(N, 0), pos.repeat(B, 1))
            M = sp.frame_nr.shape[-1]
            fr = torch.floor(sp.frame_nr.reshape(2, B, N, M)).long()
            yy = torch.round(sp.y_pixel.reshape(-1, 2, B, N, M)[0]).long()
            zz = torch.round(sp.z_pixel.reshape(-1, 2, B, N, M)[0]).long()
            va = sp.valid.reshape(2, B, N, M) > 0.5
            if sp.layer_valid is not None:
                va &= sp.layer_valid.reshape(-1, 2, B, N, M)[0] > 0.5
            va &= (fr >= 0) & (fr < n_frames) & (yy >= 0) & (yy < n_y) & (zz >= 0) & (zz < n_z)
            key = (fr * n_y + yy) * n_z + zz
            if rk.numel() == 0:
                out.extend([0] * B); continue
            p = torch.clamp(torch.searchsorted(rk, key.reshape(-1)), max=rk.numel() - 1).reshape(key.shape)
            hit = (rk[p] == key) & va
            out.extend(hit.sum(dim=(0, 2, 3)).cpu().tolist())
        return np.array(out)

    om0 = np.asarray(om0)
    s0 = int(scores([om0])[0])
    coarse = [om0 @ d for d in _small_rotations(half, step)]
    sc = scores(coarse); j = int(np.argmax(sc))
    fine = [coarse[j] @ d for d in _small_rotations(fine_half, fine_step)]
    sf = scores(fine); jf = int(np.argmax(sf))
    return (fine[jf], int(sf[jf]), s0) if sf[jf] >= sc[j] else (coarse[j], int(sc[j]), s0)
