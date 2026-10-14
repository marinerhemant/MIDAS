"""Fit one box-beam FF grain directly to its raw pixels.

STATUS: EXPERIMENTAL. Do not use it to replace centroid (c-omp refiner) results.
* Synthetic data: five pre-registered tests; the final one read CONFIRMED but
  /verify left it PROVISIONAL.
* Real data (a 20-ID Varex calibration sample, 2026-10-13): the registered split-half reproducibility test read
  REFUTED. Bright-grain positions were LESS reproducible than the centroid refiner (median ratio 1.63, 95 % CI 1.16-2.71),
  and 18 % of grains were more than 3x worse. Weak and edge grains pointed the right way, but not significantly. Cause not
  yet diagnosed.
* ``holdout_guard`` / ``heldout_cost``: the registered guard check passed, but /verify found the held-out score does not
  separate good from bad fits (accepting every fit beat it). Do not use it as an acceptance rule.
* The Anscombe weighting assumes photon counts. Detector ADU without a gain correction mis-weights pixels.

The standard FF chain reduces every spot to a centroid (per-frame pseudo-Voigt,
merged across frames) and then fits the grain to the centroids with every spot
weighted equally. Weak spots, spots smeared by orientation spread, and spots
that sit in one frame (no sub-frame omega) then pull the grain as hard as sharp
bright ones. Here the grain is fitted to the pixels of its spots instead:

    model(s, f, y, z) = a_s * T_s(f, y, z; theta) + b_s

``T_s`` is the unit-amplitude template of reflection ``s`` rendered by the
differentiable forward model (``midas_diffract.HEDMForwardModel``) for

    theta = [dv (3, rad), pos (3, um), eps_sample (6, plain Voigt),
             log sigma_theta (1, rad), log psf (1, px)]

* orientation ``R = R0 @ expm(skew(dv))`` (smooth at dv = 0);
* sample-frame strain, rotated to the crystal frame for ``correct_hkls_latc``;
* an isotropic orientation spread represented by ``K`` FIXED orientation
  particles ``R exp(skew(sigma_theta * u_k))`` (``u_k`` standard normal,
  quasi-random, drawn once). Each particle goes through the exact geometry, so
  the shape a spread produces on the detector and in omega comes from the
  forward model, not from a hand-set width mapping, and ``sigma_theta = 0``
  collapses every particle onto the mean;
* an instrumental kernel (psf px on the detector, ``sigma_f`` frames in omega),
  integrated over each frame's exposure (``splat_spots_to_patches_sparse``
  ``frame_integrate=True``).

``a_s, b_s`` are profiled out per spot by weighted least squares at every
evaluation. The residual is the Anscombe form ``2 (sqrt(D + 3/8) -
sqrt(M + 3/8))``, approximately unit-variance Poisson, so a damped Gauss-Newton
(Levenberg-Marquardt) fit is close to maximum likelihood and each pixel counts
by its photon statistics.

Fit in stages, because strain and spread trade off (pf_odf
STAGE1_RUNAWAY_NOTE): orientation + position, then + strain, then + spread and
psf.

Frame convention: the MIDAS pipeline labels zarr frame k with omega
``OmegaStart + k * OmegaStep`` (``midas_peakfit.zarr_io.frame_omega``), and a
single-frame spot's centroid is that label, so in pipeline terms the label is
the frame's CENTRE. ``midas_diffract`` gives ``frame_nr = (omega -
omega_start) / omega_step``; with ``omega_start = OmegaStart`` a spot at frame
k's label has ``frame_nr = k``, which is the centre of the splatter's cell k.
Hence ``spot_f = frame_nr - frame_offset`` with ``frame_offset = 0`` (default).
Use 0.5 only for a geometry whose omega_start is the START of frame 0's
exposure. A half-frame mismatch is a rigid rotation of every grain about the
rotation axis, so it would not show in strain, but it must match the
reconstruction the fit is seeded from.
Pixel cells are centred on integer pixel coordinates, the MIDAS convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .spot_extract import SpotPatchSpec, splat_spots_to_patches_sparse

N_THETA = 15
IDX_DV = slice(0, 3)
IDX_POS = slice(3, 6)
IDX_EPS = slice(6, 12)
IDX_SPREAD = 12
IDX_PSF = 13
IDX_SF = 14   # log rocking width (frames); free only with PixelFitConfig.fit_sigma_f
STAGES = {
    "pose": list(range(0, 6)),
    "strain": list(range(0, 12)),
    "spread": list(range(0, 14)),          # + IDX_SF when fit_sigma_f
}


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def rot_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation matrix exp(skew(v)); smooth (and exact) at v = 0."""
    return torch.linalg.matrix_exp(skew(v))


def orientation_particles(k: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """K fixed standard-normal 3-vectors (scrambled Sobol -> inverse normal),
    re-centred to zero mean so the particle cloud does not shift the grain."""
    eng = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    u = eng.draw(k).to(dtype).clamp(1e-6, 1 - 1e-6)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return z - z.mean(0, keepdim=True)


@dataclass
class GrainPixelData:
    """Measured patches of one grain.

    ``measured`` / ``weight``: (S, F, P, P), indexed [frame, y, z] in the
    forward model's detector pixel frame (``y_pixel``, ``z_pixel``). ``weight``
    is 0 for saturated, masked or other-grain pixels. ``anchor_*``: patch
    centres (integer pixel / data-frame index). ``refl``: index of each patch's
    reflection in the forward model's flat 2M spot list.
    """
    measured: torch.Tensor
    weight: torch.Tensor
    anchor_y: torch.Tensor
    anchor_z: torch.Tensor
    anchor_f: torch.Tensor
    refl: torch.Tensor
    patch_F: int
    patch_P: int


@dataclass
class PixelFitConfig:
    n_particles: int = 48
    particle_seed: int = 0
    sigma_f: float = 0.3            # instrumental omega width, frames (start value when fitted)
    # Fit the rocking width per grain (spread stage) instead of fixing it. A fixed width that is too WIDE
    # wrecks sharp grains (s17 /verify: 0.5 vs true 0.24 frame -> 25-93 ue); too narrow is absorbed.
    fit_sigma_f: bool = False
    psf0: float = 1.0               # instrumental detector width, px (initial)
    spread0_deg: float = 0.02       # initial orientation spread
    frame_offset: float = 0.0
    radius_yz: int = 6
    radius_f: int = 2
    max_iter: Dict[str, int] = field(default_factory=lambda: {"pose": 80, "strain": 40, "spread": 40})
    # Per-spot omega offset nuisance (frames) with a Gaussian prior of this sigma, profiled out per spot in
    # the strain and spread stages; 0 disables. Real data carry omega errors far above Poisson (readout
    # timing, frame jitter); a fit that trusts omega to the Poisson level bends strain to follow them
    # (s17 Phase 0: a 0.15-frame jitter -> 31-59 ue on bright sharp grains).
    omega_floor_frames: float = 0.05
    omega_profile_iter: int = 4
    pixel_integrate: bool = True
    verbose: bool = False           # print per-iteration cost / damping / time from the LM
    # Detector-readout omega timing (midas_transforms OmegaReadout*): the pipeline CORRECTS observed omega by
    # omega -= shift * shape(pixel); a forward model must put the intensity where the detector saw it, so
    # frame_nr += shift * shape / omega_step. axis 1 = columns (y_pixel), 0 = rows; shape 0 = centre-out V
    # 1 - |X - C| / (N/2), 1 = sign(X - C); centre <= 0 -> N/2. omega_step_deg must be the scan step (signed).
    readout_shift_deg: float = 0.0
    readout_axis: int = 1
    readout_shape: int = 0
    readout_centre: float = 0.0
    readout_npx: int = 0
    omega_step_deg: float = 0.0
    stages: Sequence[str] = ("pose", "strain", "spread")
    lm_lambda0: float = 1e-3
    tol_rel: float = 1e-7
    eps_bound: float = 0.01         # |strain| sanity bound (reported, not enforced)


@dataclass
class PixelFitResult:
    theta: torch.Tensor
    R: torch.Tensor
    pos: torch.Tensor
    eps_sample: torch.Tensor
    spread_deg: float
    psf_px: float
    sigma_f_frames: float
    scale: torch.Tensor
    background: torch.Tensor
    cost: float
    cost_init: float
    n_iter: Dict[str, int]
    converged: Dict[str, bool]
    theta_sigma: Optional[torch.Tensor]
    n_pixels: int
    omega_offsets: Optional[torch.Tensor] = None


def readout_frame_offset(cfg: PixelFitConfig, y_pixel: torch.Tensor, z_pixel: torch.Tensor):
    """Frames to ADD to a spot's frame coordinate for the detector-readout omega timing (0 when off)."""
    if not cfg.readout_shift_deg:
        return torch.zeros_like(y_pixel)
    if not cfg.omega_step_deg or cfg.readout_npx <= 0:
        raise ValueError("readout_shift_deg needs omega_step_deg and readout_npx")
    x = y_pixel if cfg.readout_axis == 1 else z_pixel
    n = float(cfg.readout_npx)
    c = cfg.readout_centre if cfg.readout_centre > 0 else n / 2.0
    g = 1.0 - (x - c).abs() / (n / 2.0) if cfg.readout_shape == 0 else torch.sign(x - c)
    return cfg.readout_shift_deg * g / cfg.omega_step_deg


def default_theta(cfg: PixelFitConfig, device=None, dtype=torch.float64) -> torch.Tensor:
    """Starting parameter vector: zero pose/strain offsets, the configured spread, psf and rocking width.
    Build seeds from this rather than by hand: a hand-built vector that leaves IDX_SF at 0 means a
    rocking width of exp(0) = 1 frame."""
    t = torch.zeros(N_THETA, dtype=dtype, device=device)
    t[IDX_SPREAD] = math.log(math.radians(cfg.spread0_deg))
    t[IDX_PSF] = math.log(cfg.psf0)
    t[IDX_SF] = math.log(cfg.sigma_f)
    return t


class GrainPixelModel:
    """Renders one grain's patches and its Anscombe residual."""

    def __init__(self, model, data: GrainPixelData, R0: torch.Tensor, pos0: torch.Tensor,
                 lattice: torch.Tensor, cfg: PixelFitConfig, dtype=torch.float64):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.dtype = dtype
        dev = data.measured.device
        self.device = dev
        self.R0 = R0.to(dev, dtype)
        self.pos0 = pos0.to(dev, dtype)
        self.lattice = lattice.to(dev, dtype)
        self.u = orientation_particles(cfg.n_particles, cfg.particle_seed, dtype).to(dev)
        self.D = data.measured.to(dev, dtype)
        self.W = data.weight.to(dev, dtype)
        self.sqrtD = torch.sqrt(self.D.clamp_min(0) + 0.375)
        S = data.refl.numel()
        self.delta = torch.zeros(S, dtype=dtype, device=dev)    # per-spot omega offsets (frames), profiled
        self.profile_omega = False
        self.spec = SpotPatchSpec(n_spots=S, patch_F=data.patch_F, patch_P=data.patch_P,
                                  sigma_yz=cfg.psf0, sigma_f=cfg.sigma_f,
                                  anchor_y=data.anchor_y.to(dev, dtype),
                                  anchor_z=data.anchor_z.to(dev, dtype),
                                  anchor_f=data.anchor_f.to(dev, dtype))

    def theta0(self) -> torch.Tensor:
        return default_theta(self.cfg, self.device, self.dtype)

    def spots(self, theta: torch.Tensor, delta: Optional[torch.Tensor] = None):
        """Particle spot coordinates for the patch reflections: (K, S) each."""
        K = self.u.shape[0]
        R = self.R0 @ rot_exp(theta[IDX_DV])
        sig = torch.exp(theta[IDX_SPREAD])
        Rk = R.unsqueeze(0) @ rot_exp(sig * self.u)                   # (K, 3, 3)
        eps_c = self.model.rotate_strain_sample_to_crystal(Rk, theta[IDX_EPS].expand(K, 6))
        hk, th = self.model.correct_hkls_latc(self.lattice.expand(K, 6), strain=eps_c)
        om, eta, tt, valid = self.model.calc_bragg_geometry(Rk.unsqueeze(1), hk, th)
        pos = (self.pos0 + theta[IDX_POS]).expand(K, 1, 3)
        sp = self.model.project_to_detector(om, eta, tt, pos, valid)
        M2 = sp.y_pixel.shape[-1] * sp.y_pixel.shape[-2]
        flat = lambda t: t.reshape(K, M2)[:, self.data.refl]
        f = flat(sp.frame_nr) - self.cfg.frame_offset
        f = f + readout_frame_offset(self.cfg, flat(sp.y_pixel), flat(sp.z_pixel))
        if delta is not None:
            f = f + delta.reshape(1, -1)
        return flat(sp.y_pixel), flat(sp.z_pixel), f, flat(sp.valid.to(self.dtype))

    def template(self, theta: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        sy, sz, sf, sv = self.spots(theta, delta)
        K = sy.shape[0]
        psf = torch.exp(theta[IDX_PSF])
        w = torch.full((K,), 1.0 / K, dtype=self.dtype, device=self.device)
        return splat_spots_to_patches_sparse(
            self.spec, sy, sz, sf, w, sv, radius_yz=self.cfg.radius_yz, radius_f=self.cfg.radius_f,
            sigma_yz_row=psf.expand(K), sigma_f_row=torch.exp(theta[IDX_SF]).expand(K), frame_integrate=True,
            pixel_integrate=self.cfg.pixel_integrate)

    def profile(self, T: torch.Tensor):
        """Per-spot (a_s, b_s) by weighted LS, weights W / (D + 1)."""
        w = self.W / (self.D.clamp_min(0) + 1.0)
        dims = (1, 2, 3)
        sw, st, sd = w.sum(dims), (w * T).sum(dims), (w * self.D).sum(dims)
        stt, std = (w * T * T).sum(dims), (w * T * self.D).sum(dims)
        det = (sw * stt - st * st).clamp_min(1e-30)
        a = (sw * std - st * sd) / det
        b = (sd - a * st) / sw.clamp_min(1e-30)
        return a.clamp_min(0.0), b

    def _pixel_residual(self, theta, delta):
        T = self.template(theta, delta)
        a, b = self.profile(T)
        M = a.reshape(-1, 1, 1, 1) * T + b.reshape(-1, 1, 1, 1)
        r = 2.0 * (self.sqrtD - torch.sqrt(M.clamp_min(0) + 0.375)) * torch.sqrt(self.W)
        return r, a, b

    def profile_delta(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-spot omega offsets minimising |r_s|^2 + (d_s/sigma)^2, each spot independently (its offset
        moves only its own patch): vectorised 1-D Gauss-Newton on detached inputs.

        * Started from ZERO every call with a fixed number of steps, so the profiled objective is a
          deterministic function of theta (a warm start made it drift between calls and the LM never
          met its tolerance).
        * Projected to ZERO MEAN after every step: a common omega shift of all spots is a rotation of the
          grain about the rotation axis, which belongs to the orientation; left free, the offsets
          absorbed it (same-renderer test: 0.00025 -> 0.0063 deg misorientation).
        """
        sig = self.cfg.omega_floor_frames
        th = theta.detach()
        S = self.delta.numel()
        d = torch.zeros(S, dtype=self.dtype, device=self.device)
        h = 1e-3
        with torch.no_grad():
            for _ in range(self.cfg.omega_profile_iter):
                r0 = self._pixel_residual(th, d)[0].reshape(S, -1)
                r1 = self._pixel_residual(th, d + h)[0].reshape(S, -1)
                J = (r1 - r0) / h
                g = (J * r0).sum(1) + d / sig ** 2
                H = (J * J).sum(1) + 1.0 / sig ** 2
                d = (d - g / H).clamp(-6 * sig, 6 * sig)
                d = d - d.mean()
        self.delta = d
        return d

    def residual(self, theta: torch.Tensor):
        if self.profile_omega and self.cfg.omega_floor_frames > 0:
            d = self.profile_delta(theta)
            r, a, b = self._pixel_residual(theta, d)
            prior = d / self.cfg.omega_floor_frames
            return torch.cat([r.reshape(-1), prior]), a, b
        r, a, b = self._pixel_residual(theta, None)
        return r.reshape(-1), a, b


def _gn_step_small(J: torch.Tensor, r: torch.Tensor, frac: float = 0.01) -> bool:
    """Scale-free convergence test: the full Gauss-Newton step is below ``frac`` of every free parameter's
    own 1-sigma uncertainty (sigma^2 = diag((J^T J)^-1) * cost / dof). A gradient threshold against the cost
    (the old test) sat at floating-point noise level with ~1e6 pixel residuals and flipped between runs
    from identical states (s17 /verify of the held-out guard)."""
    H = J.T @ J
    try:
        step = torch.linalg.solve(H + 1e-12 * torch.diag(torch.diag(H)).clamp_min(1e-30), -(J.T @ r))
        cov = torch.linalg.inv(H) * (float(r @ r) / max(r.numel() - J.shape[1], 1))
    except RuntimeError:
        return False
    sig = torch.sqrt(torch.diag(cov).clamp_min(1e-300))
    return bool(torch.all(step.abs() <= frac * sig))


def _lm(fun: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor, free: List[int],
        max_iter: int, lam: float, tol: float, verbose: bool = False, tag: str = ""):
    """Levenberg-Marquardt over the ``free`` entries; forward-mode Jacobian."""
    th = theta.clone()
    idx = torch.tensor(free, device=th.device)

    def f_free(x):
        t = th.clone()
        t = t.index_put((idx,), x)
        return fun(t)

    x = th[idx].clone()
    r = f_free(x)
    cost = float(r @ r)
    cost0 = cost
    it = 0
    converged = False
    J = None
    import time as _time
    for it in range(1, max_iter + 1):
        _t0 = _time.time()
        J = torch.func.jacfwd(f_free)(x)                              # (N, n)
        if _gn_step_small(J, r):
            converged = True
            break
        g = J.T @ r
        H = J.T @ J
        dH = torch.diag(H).clamp_min(1e-12)
        improved = False
        rel = 0.0
        for _ in range(30):
            step = torch.linalg.solve(H + lam * torch.diag(dH), -g)
            r_new = f_free(x + step)
            c_new = float(r_new @ r_new)
            if math.isfinite(c_new) and c_new < cost:
                rel = (cost - c_new) / max(cost, 1e-30)
                x, r, cost = x + step, r_new, c_new
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 4.0
        if verbose:
            print(f"    [lm {tag}] it {it} cost {cost:.6g} lam {lam:.2e} improved {improved} "
                  f"{_time.time() - _t0:.1f}s", flush=True)
        if not improved:
            # no step lowers the cost even at heavy damping: at a minimum to
            # numerical precision only if the gradient is ~0; report honestly
            converged = _gn_step_small(J, r)
            break
        if rel < tol:
            converged = True
            break
    th = th.index_put((idx,), x)
    sigma = None
    if J is not None:
        n_dof = max(r.numel() - len(free), 1)
        try:
            cov = torch.linalg.inv(J.T @ J) * (cost / n_dof)
            sigma = torch.sqrt(torch.diag(cov).clamp_min(0))
        except RuntimeError:
            sigma = None
    return th, cost0, cost, it, converged, lam, sigma


def fit_grain_pixels(model, data: GrainPixelData, R0: torch.Tensor, pos0: torch.Tensor,
                     lattice: torch.Tensor, cfg: Optional[PixelFitConfig] = None,
                     theta_init: Optional[torch.Tensor] = None) -> PixelFitResult:
    """Staged LM fit of one grain to its pixel patches (see module doc)."""
    cfg = cfg or PixelFitConfig()
    gm = GrainPixelModel(model, data, R0, pos0, lattice, cfg)
    theta = gm.theta0() if theta_init is None else theta_init.to(gm.device, gm.dtype).clone()
    fun = lambda t: gm.residual(t)[0]
    with torch.no_grad():
        r0 = fun(theta)
    cost_init = float(r0 @ r0)
    n_iter, conv = {}, {}
    sigma_free = None
    for st in cfg.stages:
        free = STAGES[st] + ([IDX_SF] if (st == "spread" and cfg.fit_sigma_f) else [])
        gm.profile_omega = st in ("strain", "spread")
        # fresh damping per stage: newly freed parameters start far from their optimum
        theta, _, cost, it, ok, _, sig = _lm(fun, theta, free, cfg.max_iter[st], cfg.lm_lambda0, cfg.tol_rel,
                                             verbose=cfg.verbose, tag=st)
        n_iter[st], conv[st] = it, ok
        if sig is not None:
            sigma_free = torch.full((N_THETA,), float("nan"), dtype=gm.dtype, device=gm.device)
            sigma_free[torch.tensor(free, device=gm.device)] = sig
    with torch.no_grad():
        r, a, b = gm.residual(theta)
    gm.profile_omega = False
    return PixelFitResult(
        theta=theta.detach(), R=(gm.R0 @ rot_exp(theta[IDX_DV])).detach(),
        pos=(gm.pos0 + theta[IDX_POS]).detach(), eps_sample=theta[IDX_EPS].detach(),
        spread_deg=math.degrees(float(torch.exp(theta[IDX_SPREAD]))),
        psf_px=float(torch.exp(theta[IDX_PSF])), sigma_f_frames=float(torch.exp(theta[IDX_SF])),
        scale=a.detach(), background=b.detach(),
        cost=float(r @ r), cost_init=cost_init, n_iter=n_iter, converged=conv,
        theta_sigma=sigma_free, n_pixels=int((data.weight > 0).sum()),
        omega_offsets=gm.delta.detach().clone())


def subset_data(data: GrainPixelData, idx) -> GrainPixelData:
    """The same grain's data restricted to spots ``idx``."""
    idx = torch.as_tensor(idx, dtype=torch.long, device=data.measured.device)
    return GrainPixelData(data.measured[idx], data.weight[idx], data.anchor_y[idx], data.anchor_z[idx],
                          data.anchor_f[idx], data.refl[idx], data.patch_F, data.patch_P)


def heldout_cost(model, data: GrainPixelData, R: torch.Tensor, pos: torch.Tensor, eps_sample: torch.Tensor,
                 lattice: torch.Tensor, cfg: PixelFitConfig, nuisance: torch.Tensor) -> float:
    """Profiled cost of a FIXED grain state (R, pos, sample strain) on ``data``: per-spot scale/background and,
    when ``cfg.omega_floor_frames > 0``, per-spot omega offsets are profiled; spread / psf / rocking width are
    taken from ``nuisance`` (a theta vector, e.g. a fit's). Lower = the state predicts these pixels better."""
    gm = GrainPixelModel(model, data, R, pos, lattice, cfg)
    th = default_theta(cfg, gm.device, gm.dtype)
    th[IDX_EPS] = eps_sample.to(gm.device, gm.dtype)
    nz = nuisance.to(gm.device, gm.dtype)
    th[IDX_SPREAD], th[IDX_PSF], th[IDX_SF] = nz[IDX_SPREAD], nz[IDX_PSF], nz[IDX_SF]
    gm.profile_omega = cfg.omega_floor_frames > 0
    with torch.no_grad():
        r = gm.residual(th)[0]
    return float(r @ r)


def holdout_guard(model, data: GrainPixelData, R_seed: torch.Tensor, pos_seed: torch.Tensor,
                  eps_seed: torch.Tensor, lattice: torch.Tensor, cfg: PixelFitConfig, *,
                  frac_heldout: float = 0.2, seed: int = 0) -> dict:
    """Accept or reject a pixel fit WITHOUT truth: fit on ~80 % of the spots, then ask whether the fitted state
    predicts the held-out 20 % better than the seed (centroid) state does, both scored with the same profiled
    cost and the fit's own spread / psf / rocking width. If accepted, refit on all spots.

    Why: a distance cap ("moved > 10 sigma from the seed") rejected fits that had moved TO the truth (s17 final
    registration: 21.4 -> 1.0 um discarded); held-out prediction tests what matters and works on real data.
    Returns dict(accepted, cost_fit_heldout, cost_seed_heldout, fit80 (PixelFitResult), final (PixelFitResult or None)).
    """
    S = data.refl.numel()
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(S, generator=g)
    n_ho = max(1, int(round(frac_heldout * S)))
    ho, fit = perm[:n_ho], perm[n_ho:]
    th0 = default_theta(cfg, data.measured.device, data.measured.dtype)
    th0[IDX_EPS] = eps_seed.to(th0)
    res80 = fit_grain_pixels(model, subset_data(data, fit), R_seed, pos_seed, lattice, cfg, theta_init=th0)
    d_ho = subset_data(data, ho)
    c_fit = heldout_cost(model, d_ho, res80.R, res80.pos, res80.eps_sample, lattice, cfg, res80.theta)
    c_seed = heldout_cost(model, d_ho, R_seed, pos_seed, eps_seed, lattice, cfg, res80.theta)
    accepted = all(res80.converged.values()) and c_fit < c_seed
    final = fit_grain_pixels(model, data, R_seed, pos_seed, lattice, cfg, theta_init=th0) if accepted else None
    return dict(accepted=bool(accepted), cost_fit_heldout=c_fit, cost_seed_heldout=c_seed, n_heldout=int(n_ho),
                fit80=res80, final=final)


def select_reflections(model, R: torch.Tensor, pos: torch.Tensor, lattice: torch.Tensor,
                       eps_sample: Optional[torch.Tensor] = None, *, n_frames: int,
                       n_pixels_y: int, n_pixels_z: int, patch_F: int, patch_P: int,
                       frame_offset: float = 0.0, dtype=torch.float64, cfg: Optional[PixelFitConfig] = None):
    """Predicted reflections of a grain whose whole patch lies on the detector
    and inside the frame range. Returns (refl, anchor_y, anchor_z, anchor_f)
    with integer anchors (pixel / data-frame index)."""
    eps = (torch.zeros(6, dtype=dtype, device=R.device) if eps_sample is None
           else eps_sample.to(R.device, dtype))   # None used to build it on the CPU: a CUDA model crashed
    R = R.to(dtype).reshape(1, 3, 3)
    eps_c = model.rotate_strain_sample_to_crystal(R, eps.reshape(1, 6))
    hk, th = model.correct_hkls_latc(lattice.to(dtype).reshape(1, 6), strain=eps_c)
    om, eta, tt, valid = model.calc_bragg_geometry(R.unsqueeze(1), hk, th)
    sp = model.project_to_detector(om, eta, tt, pos.to(dtype).reshape(1, 1, 3), valid)
    y, z = sp.y_pixel.reshape(-1), sp.z_pixel.reshape(-1)
    f = sp.frame_nr.reshape(-1) - frame_offset
    if cfg is not None:   # readout timing: anchor the patch where the detector recorded the spot
        f = f + readout_frame_offset(cfg, sp.y_pixel.reshape(-1), sp.z_pixel.reshape(-1))
    v = sp.valid.reshape(-1) > 0.5
    hp, hf = patch_P // 2, patch_F // 2
    ay, az, af = torch.round(y), torch.round(z), torch.round(f)
    ok = (v & (ay - hp >= 0) & (ay + hp < n_pixels_y) & (az - hp >= 0) & (az + hp < n_pixels_z)
          & (af - hf >= 0) & (af + hf < n_frames))
    refl = torch.nonzero(ok).reshape(-1)
    return refl, ay[refl].long(), az[refl].long(), af[refl].long()


def crop_patches(frames_fyz: np.ndarray, anchor_y, anchor_z, anchor_f, patch_F: int, patch_P: int):
    """Crop (S, F, P, P) patches from an in-memory [frame, y, z] stack."""
    hp, hf = patch_P // 2, patch_F // 2
    out = np.empty((len(anchor_y), patch_F, patch_P, patch_P), dtype=np.float64)
    for s, (y, z, f) in enumerate(zip(np.asarray(anchor_y), np.asarray(anchor_z), np.asarray(anchor_f))):
        out[s] = frames_fyz[f - hf:f + hf + 1, y - hp:y + hp + 1, z - hp:z + hp + 1]
    return out


__all__ = ["readout_frame_offset", "subset_data", "heldout_cost", "holdout_guard", "default_theta", "GrainPixelData", "PixelFitConfig", "PixelFitResult", "GrainPixelModel",
           "fit_grain_pixels", "select_reflections", "crop_patches", "orientation_particles", "rot_exp"]
