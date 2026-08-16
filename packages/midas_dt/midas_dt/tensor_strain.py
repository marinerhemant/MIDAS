"""Per-voxel **deviatoric** strain from multi-ring XRD-CT, by direct inversion.

Branch C solves a free peak centre per voxel. This replaces that centre with
physics: the peak position at every (hkl, eta, omega) follows from one strain
tensor per voxel, so every measurement of every ring constrains the same five
numbers instead of each lineout being fitted alone.

Registered in ``dev/paper/PREREGISTER_tensor_null.md`` before any of it was
written. Two things from that registration are structural here rather than
optional:

**Five components, not six.** ``q_hat`` is a unit vector, so
``q_x^2 + q_y^2 + q_z^2 == 1`` identically and ``tr(eps)`` multiplies the same
constant as the reference lattice parameter. The two are EXACTLY degenerate --
not weakly, not geometry-dependent. Fitting six components plus ``d0`` gives a
design matrix that is rank-deficient by construction (measured: condition
number 5.4e14, answers of order 1e12 ue). So the model carries the deviatoric
part plus a per-voxel mean spacing, and the mean spacing is reported as an
apparent d-spacing, never decomposed into ``d0`` and a dilatation.

**The projection is of PATTERNS, not of peak positions.** A ray's observed peak
sits near the intensity-weighted mean of the voxel spacings along it, and a
weighted mean does not add -- back-projecting fitted positions is the exact
error ``conventions`` warns about, measured at correlation 0.03 between branches
on real data. So each voxel renders its own pattern, the patterns are summed
along the ray by the same sparse Radon operator Branch C uses, and the residual
is taken in projection space where the measurement actually lives.

No performance claim is made against Branch A or B: neither can produce a
strain tensor at all, so there is nothing to compare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .direct import _profile, _torch, projection_matrix

__all__ = [
    "DeviatoricStrain",
    "q_hat_sample_frame",
    "deviatoric_design",
    "strain_to_radius",
    "TensorResult",
    "fit_tensor_strain",
]

#: Order of the five deviatoric degrees of freedom everywhere in this module.
#: ``dev_zz`` is not free: it is ``-(dev_xx + dev_yy)``.
COMPONENT_NAMES = ("dev_xx", "dev_yy", "eps_yz", "eps_xz", "eps_xy")


@dataclass(frozen=True)
class DeviatoricStrain:
    """A deviatoric strain tensor, as the five free components."""

    dev_xx: float
    dev_yy: float
    eps_yz: float
    eps_xz: float
    eps_xy: float

    def as_matrix(self) -> np.ndarray:
        zz = -(self.dev_xx + self.dev_yy)          # traceless by construction
        return np.array([
            [self.dev_xx, self.eps_xy, self.eps_xz],
            [self.eps_xy, self.dev_yy, self.eps_yz],
            [self.eps_xz, self.eps_yz, zz],
        ], dtype=np.float64)

    def as_vector(self) -> np.ndarray:
        return np.array([self.dev_xx, self.dev_yy, self.eps_yz,
                         self.eps_xz, self.eps_xy], dtype=np.float64)


def q_hat_sample_frame(two_theta_deg, eta_deg, omega_deg):
    """Scattering direction in the **sample** frame, shape ``(..., 3)``.

    Lab convention, matching the scan: beam along x, translations along y,
    rotation about the vertical z. In the lab

        q_lab = (-sin(th), cos(th) cos(eta), cos(th) sin(eta))

    and a vector fixed in the sample appears in the lab as ``v_lab = R v_s``,
    so ``q_s = R_z(omega)^T q_lab``. Getting that transpose backwards rotates
    every recovered tensor by ``-2*omega`` and still produces a smooth,
    plausible map.
    """
    th = np.radians(np.asarray(two_theta_deg, dtype=np.float64) / 2.0)
    e = np.radians(np.asarray(eta_deg, dtype=np.float64))
    w = np.radians(np.asarray(omega_deg, dtype=np.float64))
    th, e, w = np.broadcast_arrays(th, e, w)

    qx = -np.sin(th)
    qy = np.cos(th) * np.cos(e)
    qz = np.cos(th) * np.sin(e)

    c, s = np.cos(w), np.sin(w)
    return np.stack([c * qx + s * qy, -s * qx + c * qy, qz], axis=-1)


def deviatoric_design(q):
    """Rows mapping the five components to ``eps_qq``, shape ``(..., 5)``.

    With ``dev_zz = -(dev_xx + dev_yy)``::

        eps_qq = (qx^2 - qz^2) dev_xx + (qy^2 - qz^2) dev_yy
                 + 2 qy qz eps_yz + 2 qx qz eps_xz + 2 qx qy eps_xy

    The hydrostatic part is absent by construction -- see the module docstring.
    """
    qx, qy, qz = q[..., 0], q[..., 1], q[..., 2]
    return np.stack([qx * qx - qz * qz, qy * qy - qz * qz,
                     2 * qy * qz, 2 * qx * qz, 2 * qx * qy], axis=-1)


def strain_to_radius(d0_a, eps_qq, wavelength_a, lsd_um, px_um):
    """Ring radius in pixels for a spacing strained by ``eps_qq``.

    Exact rather than the small-angle ``R ≈ R0 (1 - eps)``: the closed form is
    no more expensive and does not need defending at a particular 2theta.
    """
    xp = np.ndarray  # documentation only; works for numpy or torch inputs
    d = d0_a * (1.0 + eps_qq)
    s = wavelength_a / (2.0 * d)
    if hasattr(s, "clamp"):
        s = s.clamp(-1.0 + 1e-12, 1.0 - 1e-12)
        theta = s.asin()
        return (lsd_um / px_um) * (2.0 * theta).tan()
    s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)
    theta = np.arcsin(s)
    return (lsd_um / px_um) * np.tan(2.0 * theta)


@dataclass
class TensorResult:
    """Recovered per-voxel deviatoric strain, plus what it took to get it."""

    strain: np.ndarray                 # (n_voxels, 5), in strain units
    d_mean_a: np.ndarray               # (n_voxels,) apparent spacing, NOT d0
    intensity: np.ndarray              # (n_voxels,)
    active: np.ndarray                 # (n_voxels,) bool, voxels actually fitted
    size: int
    loss: float
    n_steps: int
    converged: bool
    residual_rel: float
    sigma_px: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    r_bin_px: float = 1.0
    component_names: tuple = COMPONENT_NAMES

    @property
    def valid(self) -> bool:
        """Is this run interpretable at all?

        Registered after run 1 of the null test printed a full verdict from a
        fit that had collapsed. A width far from the sampling, or a residual
        larger than the data, means the model never described the measurement --
        and no threshold comparison applies to it.
        """
        s = np.asarray(self.sigma_px, dtype=float)
        return bool(s.size and np.all(s > 0.3 * self.r_bin_px)
                    and np.all(s < 5.0 * self.r_bin_px)
                    and np.isfinite(self.residual_rel)
                    and self.residual_rel < 1.0)

    def why_invalid(self) -> str:
        s = np.asarray(self.sigma_px, dtype=float)
        bad = []
        if s.size and (np.any(s <= 0.3 * self.r_bin_px)
                       or np.any(s >= 5.0 * self.r_bin_px)):
            bad.append(f"sigma {np.array2string(s, precision=3)} px outside "
                       f"0.3-5x the {self.r_bin_px:.2f} px bin")
        if not np.isfinite(self.residual_rel) or self.residual_rel >= 1.0:
            bad.append(f"relative residual {self.residual_rel:.3f} >= 1")
        return "; ".join(bad) or "valid"

    def maps(self) -> dict:
        """Each component as a ``(size, size)`` map, NaN outside the mask."""
        out = {}
        for i, name in enumerate(self.component_names):
            m = np.full(self.size * self.size, np.nan)
            m[self.active] = self.strain[:, i]
            out[name] = m.reshape(self.size, self.size)
        m = np.full(self.size * self.size, np.nan)
        m[self.active] = self.d_mean_a
        out["d_apparent_A"] = m.reshape(self.size, self.size)
        return out

    def describe(self) -> str:
        n = int(self.active.sum())
        rms = np.sqrt((self.strain ** 2).mean(axis=0)) * 1e6
        worst = self.component_names[int(np.argmax(rms))]
        return (f"deviatoric strain: {n} voxels, {self.n_steps} steps, "
                f"loss {self.loss:.6g}, relative residual {self.residual_rel:.4f}, "
                f"rms {rms.max():.1f} ue (worst {worst}), "
                f"sigma {np.array2string(self.sigma_px, precision=3)} px"
                f"{'' if self.converged else '  [NOT CONVERGED]'}")


def fit_tensor_strain(
    lineouts,
    *,
    radii_px,
    rings_d0_a: Sequence[float],
    two_theta_deg: Sequence[float],
    omega_deg,
    wavelength_a: float,
    lsd_um: float,
    px_um: float,
    eta_deg,
    intensity_map=None,
    size: int | None = None,
    shift: float = 0.0,
    sigma_px: float | str = "fit",
    steps: int = 400,
    lr: float = 0.02,
    mask_threshold: float | None = None,
    omega_chunk: int = 24,
    device=None,
) -> TensorResult:
    """Fit five deviatoric components per voxel to multi-ring lineouts.

    Parameters
    ----------
    lineouts : ndarray
        ``(n_rings, n_omega, n_eta, n_translations, n_r)`` measured radial
        lineouts. This is the cake, sliced to each ring's window.
    radii_px : ndarray
        ``(n_rings, n_r)`` radial bin centres for each ring's window.
    rings_d0_a, two_theta_deg :
        Unstrained spacing and Bragg angle per ring.
    intensity_map : ndarray, optional
        ``(size, size)`` per-voxel intensity, e.g. from Branch B. Supplying it
        fixes the intensities and leaves only the strain free, which is what
        the null test wants: an intensity that is already known should not be
        re-fitted, and letting it float lets the optimiser trade intensity
        against strain.
    mask_threshold : float, optional
        Fraction of the peak intensity below which a voxel is not fitted.
        Voxels with no sample in them have no spacing to measure, and fitting
        them produces impressive-looking noise.
    """
    torch = _torch()
    from midas_invert import fit

    li = np.asarray(lineouts, dtype=np.float64)
    n_rings, n_w, n_eta, n_t, n_r = li.shape
    size = int(size or n_t)
    dev = device or "cpu"
    dt = torch.float64

    # Dense projector, one slab per rotation: (n_omega, n_translations, n_vox).
    # Sparse would save nothing at this size (14 x 256 per omega) and the dense
    # form lets the whole rotation axis go through one batched mat-mul.
    A_sp = projection_matrix(size, np.asarray(omega_deg, dtype=np.float64), n_t,
                             dtype=dt, device=dev)
    A = A_sp.to_dense().reshape(n_w, n_t, size * size)

    # ---- which voxels are worth fitting -------------------------------------
    if intensity_map is None:
        raise ValueError(
            "intensity_map is required: strain is measured from where the peak "
            "sits, and a voxel with no intensity has no peak. Reconstruct it "
            "first (Branch B) and pass it in.")
    imap = np.asarray(intensity_map, dtype=np.float64).reshape(-1)
    thr = (mask_threshold if mask_threshold is not None else 0.1) * np.nanmax(imap)
    active = np.isfinite(imap) & (imap > thr)
    n_act = int(active.sum())
    if n_act < 4:
        raise ValueError(f"only {n_act} voxels above the mask threshold")

    act_t = torch.as_tensor(np.where(active)[0], device=dev)
    I_v = torch.as_tensor(np.clip(imap[active], 0.0, None), dtype=dt, device=dev)
    # Only the active columns of the projector are ever multiplied by anything.
    A_act = A[:, :, act_t]                       # (n_w, n_t, n_act)

    # ---- design: eps_qq = D @ eps, per (ring, omega, eta) --------------------
    tt = np.asarray(two_theta_deg, dtype=np.float64)[:, None, None]
    et = np.asarray(eta_deg, dtype=np.float64)[None, None, :]
    om = np.asarray(omega_deg, dtype=np.float64)[None, :, None]
    D = deviatoric_design(q_hat_sample_frame(tt, et, om))       # (nr, nw, ne, 5)
    D_t = torch.as_tensor(D, dtype=dt, device=dev)

    R_t = torch.as_tensor(np.asarray(radii_px, dtype=np.float64),
                          dtype=dt, device=dev)                  # (n_rings, n_r)
    obs = torch.as_tensor(li, dtype=dt, device=dev)
    # Weight by 1/sqrt(I): Poisson, and it stops the weak outer rings dominating
    # the residual purely by being noisy.
    w = 1.0 / torch.sqrt(torch.clamp(obs, min=1.0))

    d0_t = torch.as_tensor(np.asarray(rings_d0_a, dtype=np.float64),
                           dtype=dt, device=dev)

    # The peak width is NOT a free choice. Measured on synthetic data, a 6%
    # error in an assumed width manufactures up to 49 ue of strain -- the same
    # order as the whole instrumental floor, and indistinguishable from it.
    # See PREREGISTER_tensor_null.md section 9a.
    #
    # Fitted per ring rather than per voxel: instrumental broadening varies with
    # Bragg angle, but a uniform powder has no reason to vary voxel to voxel, and
    # a per-voxel width would trade against strain. A single width scales all
    # peaks equally while strain shifts them differentially with eta, so the two
    # are largely orthogonal.
    fit_sigma = isinstance(sigma_px, str)
    if fit_sigma and sigma_px != "fit":
        raise ValueError(f"sigma_px must be a number or 'fit', got {sigma_px!r}")

    # sigma is BOUNDED, not merely free. Run 1 of the null test drove it to
    # 0.02 px against a 1.0 px bin -- fifty times narrower than the sampling --
    # because it was the only parameter that could shed an over-bright model.
    # A runaway must be impossible, not merely detectable afterwards.
    r_bin = float(np.median(np.diff(np.asarray(radii_px, dtype=np.float64)[0])))
    sig_lo, sig_hi = 0.3 * r_bin, 5.0 * r_bin
    # Per ring, not one value for all: instrumental broadening rises with Bragg
    # angle, measured on this instrument as 0.98 -> 1.36 px across nine rings, a
    # 39% spread. Collapsing that to a median mis-sets the inner and outer rings
    # by ~18% each, and section 9a measured a 6% width error at 49 ue of false
    # strain. A sequence is therefore the expected input, not a convenience.
    if fit_sigma:
        sig_arr = np.full(n_rings, 1.5)
    else:
        sig_arr = np.broadcast_to(np.asarray(sigma_px, dtype=np.float64),
                                  (n_rings,)).copy()
    sig_arr = np.clip(sig_arr, sig_lo * 1.05, sig_hi * 0.95)
    sig0 = float(np.median(sig_arr))
    u0 = np.log((sig_arr - sig_lo) / (sig_hi - sig_arr))
    sig_raw = torch.as_tensor(u0, dtype=dt, device=dev)
    sig_raw.requires_grad_(fit_sigma)

    def sigma_of(k):
        return sig_lo + (sig_hi - sig_lo) * torch.sigmoid(sig_raw[k])

    # The intensity SHAPE is seeded (Branch B knows it); its absolute SCALE is
    # not meaningful -- Branch B reports integrated intensity, the cake holds
    # counts, and on real data the two differed by ~8x per voxel and ~81x in
    # projection. One free amplitude per ring, since structure factors differ.
    # Amplitude and background are solved in CLOSED FORM at initialisation,
    # not left for the optimiser to find.
    #
    # At zero strain and a given width the model is exactly LINEAR in the two of
    # them: ``obs ~ a * M0 + b``. So a weighted 2x2 normal equation gives both
    # exactly, and the optimiser starts at a ~ 1 instead of having to travel
    # ln(80) while the strain absorbs the mismatch in the meantime.
    #
    # This matters because they are otherwise degenerate with the width -- a
    # peak's area is amplitude x width and a background absorbs either. Measured
    # without this: an 80x seed mismatch left the amplitude 2.6x short after
    # 3000 steps and 137 ue of false strain, with the width pinned to its bound.
    log_amp = torch.zeros((n_rings,), dtype=dt, device=dev)
    bg_init = torch.zeros((n_rings, n_t), dtype=dt, device=dev)
    with torch.no_grad():
        for k in range(n_rings):
            cen0 = strain_to_radius(d0_t[k], torch.zeros((), dtype=dt,
                                                         device=dev),
                                    wavelength_a, lsd_um, px_um)
            prof0 = _profile(R_t[k], cen0.reshape(1),
                             sigma_of(k).reshape(1))[0]          # (n_r,)
            # Rendered pattern at zero strain: intensity along each ray.
            ray = torch.einsum("wtv,v->wt", A_act, I_v)          # (n_w, n_t)
            M0 = ray[:, :, None, None] * prof0[None, None, None, :]
            M0 = M0.expand(n_w, n_t, n_eta, n_r).permute(0, 2, 1, 3)
            y, ww = obs[k], w[k] ** 2
            # Weighted normal equations for [a, b]. Both reductions are FULL
            # sums, so both solutions are one scalar per ring -- the background
            # init is a single value shared across translations, not solved per
            # translation. That is the starting point only: bg_raw is free per
            # (ring, translation) and separates them from here. Solving b per
            # translation would be a behaviour change, not a tidy-up.
            Smm = (ww * M0 * M0).sum()
            Sm = (ww * M0).sum()
            Sy = (ww * y).sum()
            Smy = (ww * M0 * y).sum()
            S1 = ww.sum()
            det = Smm * S1 - Sm * Sm
            if abs(float(det)) > 1e-30:
                a = float((Smy * S1 - Sm * Sy) / det)
                b = float((Smm * Sy - Sm * Smy) / det)
            else:
                a, b = 1.0, 0.0
            log_amp[k] = float(np.log(max(a, 1e-12)))
            bg_init[k] = b
    _data_scale = float(obs.mean())
    log_amp = log_amp.clone().requires_grad_(True)

    # Background carried as a FRACTION of the data scale: a parameter that must
    # travel thousands of counts cannot share a learning rate with one that must
    # travel 1e-4 of strain.
    bg_scale = max(_data_scale, 1.0)
    bg_raw = (bg_init / bg_scale).clone().requires_grad_(True)

    scale = 1.0e-4          # so raw strain parameters are order 1
    eps_raw = torch.zeros((n_act, 5), dtype=dt, device=dev, requires_grad=True)
    dlog_raw = torch.zeros((n_act,), dtype=dt, device=dev, requires_grad=True)

    # midas_invert.fit takes a SEQUENCE of leaf tensors and a zero-argument
    # closure, and updates the tensors in place.
    raw = ([eps_raw, dlog_raw, log_amp, bg_raw]
           + ([sig_raw] if fit_sigma else []))

    # Rotations are processed in chunks: the per-voxel patterns are
    # (n_act, n_omega, n_eta, n_r), which at a full scan is gigabytes if
    # materialised at once. Chunking costs nothing and bounds the memory.
    chunk = max(1, int(omega_chunk))

    def loss_fn():
        eps = eps_raw * scale                                # (n_act, 5)
        dfac = (dlog_raw * scale)[:, None, None]
        total = torch.zeros((), dtype=dt, device=dev)
        for k in range(n_rings):
            for w0 in range(0, n_w, chunk):
                w1 = min(w0 + chunk, n_w)
                # eps_qq for each active voxel at each (omega, eta) in the chunk
                eqq = torch.einsum("wec,vc->vwe", D_t[k, w0:w1], eps) + dfac
                cen = strain_to_radius(d0_t[k], eqq, wavelength_a, lsd_um, px_um)
                prof = _profile(R_t[k], cen.reshape(-1),
                                sigma_of(k).expand(cen.numel()))
                # (n_act, nw_c, n_eta, n_r), scaled by the voxel's intensity
                prof = prof.reshape(n_act, w1 - w0, n_eta, n_r) * \
                    I_v[:, None, None, None]
                # Project along the rays of each rotation in the chunk:
                # (nw_c, n_t, n_act) @ (nw_c, n_act, n_eta*n_r)
                model = torch.bmm(
                    A_act[w0:w1],
                    prof.permute(1, 0, 2, 3).reshape(w1 - w0, n_act, -1),
                ).reshape(w1 - w0, n_t, n_eta, n_r)
                model = (model * log_amp[k].exp()
                         + (bg_raw[k] * bg_scale)[None, :, None, None])
                r = (model.permute(0, 2, 1, 3) - obs[k, w0:w1]) * w[k, w0:w1]
                total = total + (r * r).sum()
        return total / obs.numel()

    info = fit(raw, loss_fn, steps=steps, lr=lr, optimizer="adam")
    with torch.no_grad():
        final = float(loss_fn())
        eps_out = (eps_raw * scale).detach().cpu().numpy()
        d_out = (1.0 + dlog_raw * scale).detach().cpu().numpy()
        sig_out = np.array([float(sigma_of(k)) for k in range(n_rings)])
        amp_out = log_amp.exp().detach().cpu().numpy()

    denom = float((obs * w).pow(2).mean())
    return TensorResult(
        strain=eps_out,
        d_mean_a=np.asarray(rings_d0_a, dtype=np.float64)[0] * d_out,
        intensity=imap[active],
        active=active, size=size, loss=final,
        n_steps=int((info or {}).get("steps", steps)) if isinstance(info, dict)
        else steps,
        converged=bool((info or {}).get("converged", True))
        if isinstance(info, dict) else True,
        residual_rel=float(np.sqrt(final / denom)) if denom else float("nan"),
        sigma_px=sig_out, amplitude=amp_out, r_bin_px=r_bin,
    )
