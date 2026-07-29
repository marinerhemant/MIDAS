"""Pink-beam inverse problem: differentiable per-spot ROI image-space loss.

For every reflection (K_idx, M_idx) we keep a small (R x R) region of the
detector centred on the spot's mono-equivalent location. The per-energy
forward model deposits a Gaussian PSF at each spot's (y_k, z_k) inside
the ROI weighted by S(E_k); summing across N_E gives the predicted ROI
image. The loss is L2 between predicted and observed ROIs.

This is fully differentiable in:
  - euler_angles, position, lattice_params (via per-energy spot positions)
  - spectrum weights (when the spectrum is parameterised)

Why ROIs and not a full detector image:
  - Only ~K*M small windows are non-zero per grain (sparse)
  - Vectorised splat across (n_rois, n_energies, R, R) is one einsum
  - Memory and compute scale with the actual signal, not the panel size
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import midas_diffract as md

from .spectrum import ParameterisedSpectrum

H_C_KEV_A = 12.398419739


# ---------------------------------------------------------------------------
#  Per-energy mono model bank
# ---------------------------------------------------------------------------

@dataclass
class PinkBank:
    """Per-energy mono model bank built from a ParameterisedSpectrum.

    The energies and lambdas come from the spectrum's fixed grid; only
    the *weights* are learnable. Each per-energy ``HEDMForwardModel`` is
    constructed once at bank-build time.
    """
    spectrum: ParameterisedSpectrum
    models: List[md.HEDMForwardModel]
    hkls_int: torch.Tensor
    hkls_cart: torch.Tensor
    g_mag: torch.Tensor


def build_pink_bank(
    spectrum: ParameterisedSpectrum,
    *,
    space_group,
    lattice,
    geom_factory: Callable[[float], md.HEDMGeometry],
    two_theta_max_deg: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    scan_config: Optional["md.ScanConfig"] = None,
) -> PinkBank:
    Es = spectrum.energies_keV.detach().cpu().numpy()
    lams = spectrum.lambdas_A.detach().cpu().numpy()
    lam_ref = float(lams.max())
    hkls_cart, _, hkls_int = md.hkls_for_forward_model(
        space_group, lattice,
        wavelength_A=lam_ref,
        two_theta_max_deg=two_theta_max_deg,
        dtype=dtype,
    )
    hkls_cart = hkls_cart.to(device=device, dtype=dtype)
    hkls_int = hkls_int.to(device=device, dtype=dtype)
    g_mag = torch.norm(hkls_cart, dim=-1)

    models = []
    for lam in lams:
        sin_theta = (g_mag * float(lam) / 2.0).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
        thetas_k = torch.asin(sin_theta).to(device=device, dtype=dtype)
        geom = geom_factory(float(lam))
        m = md.HEDMForwardModel(
            hkls=hkls_cart,
            thetas=thetas_k,
            geometry=geom,
            hkls_int=hkls_int,
            scan_config=scan_config,
        ).to(device=device, dtype=dtype)
        models.append(m)
    return PinkBank(spectrum=spectrum, models=models,
                    hkls_int=hkls_int, hkls_cart=hkls_cart, g_mag=g_mag)


# ---------------------------------------------------------------------------
#  Forward evaluation -> stacked per-energy spots
# ---------------------------------------------------------------------------

def pink_forward_stacked(
    bank: PinkBank,
    euler_angles: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor] = None,
    strain: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run all per-energy models. Returns stacked tensors (N_E, K, M).

    Supports two input conventions:

      Single grain (or shared lattice across N grains):
        ``euler_angles`` shape ``(1, 3)``, output per energy ``(K, M)``.

      Per-grain lattice / strain (multi-grain, anisotropic strain):
        ``euler_angles`` shape ``(N_grains, 1, 3)``,
        ``positions`` ``(N_grains, 1, 3)``,
        ``lattice_params`` ``(N_grains, 6)``.
        Per-energy forward returns ``(N_grains, 2, M)``; this helper
        flattens into ``(K = 2*N_grains, M)``. Grain index for spot at
        flat ``K_idx`` is ``K_idx // 2``.
    """
    omegas, etas, ttheta, ys, zs, frames, valids = [], [], [], [], [], [], []
    for m in bank.models:
        sp = m(euler_angles, positions,
               lattice_params=lattice_params, strain=strain)
        # Detect multi-grain output (..., N_grains, K_per_grain=2, M) and
        # flatten to (..., K_total = 2*N_grains, M).
        def _flat(t):
            if t.dim() == 3:
                # (N_grains, K_per_grain, M) -> (N_grains*K_per_grain, M)
                return t.reshape(-1, t.shape[-1])
            return t
        omegas.append(_flat(sp.omega)); etas.append(_flat(sp.eta))
        ttheta.append(_flat(sp.two_theta))
        ys.append(_flat(sp.y_pixel)); zs.append(_flat(sp.z_pixel))
        frames.append(_flat(sp.frame_nr)); valids.append(_flat(sp.valid))
    return (torch.stack(omegas, 0), torch.stack(etas, 0),
            torch.stack(ttheta, 0), torch.stack(ys, 0),
            torch.stack(zs, 0), torch.stack(frames, 0),
            torch.stack(valids, 0))


# ---------------------------------------------------------------------------
#  ROI selection from a "ground-truth" forward call
# ---------------------------------------------------------------------------

@dataclass
class RoiPlan:
    """Pre-computed ROI plan for image-space inversion.

    Each kept reflection (n_kept) has:
      roi_centre_y, roi_centre_z : integer pixel of the ROI corner
      spot_idx : (K_idx, M_idx) into the (K, M) layout
    All ROIs are size (roi_h, roi_w) and lie fully inside the panel.
    """
    centres_y: torch.Tensor   # (n_kept,) integer
    centres_z: torch.Tensor   # (n_kept,) integer
    spot_kidx: torch.Tensor   # (n_kept,)
    spot_midx: torch.Tensor   # (n_kept,)
    roi_h: int
    roi_w: int
    n_pixels_y: int
    n_pixels_z: int

    @property
    def n_kept(self) -> int:
        return int(self.centres_y.shape[0])


def plan_rois_from_state(
    bank: PinkBank,
    euler_angles: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
    *,
    roi_h: int,
    roi_w: int,
    require_all_energies_valid: bool = False,
) -> RoiPlan:
    """Pick ROIs centred on each (K, M) spot's centre-energy position.

    Drops spots where the ROI would fall off the panel, or (optionally)
    where any energy sample produced an invalid prediction.

    Single-grain forward output is ``(N_E, K, M)``; this helper assumes
    that layout (i.e., euler is ``(1, 3)`` -> N=1, no batch).
    """
    with torch.no_grad():
        omegas, etas, tt, ys, zs, frames, valids = pink_forward_stacked(
            bank, euler_angles, positions, lattice_params=lattice_params,
        )
    if ys.dim() != 3:
        raise ValueError(
            f"plan_rois_from_state expects (N_E, K, M); got {tuple(ys.shape)}"
        )
    N_E, K, M = ys.shape
    centre_idx = N_E // 2

    n_pixels_y = bank.models[0].n_pixels_y
    n_pixels_z = bank.models[0].n_pixels_z
    half_h = roi_h // 2; half_w = roi_w // 2

    centres_y = torch.round(ys[centre_idx]).to(torch.long)   # (K, M)
    centres_z = torch.round(zs[centre_idx]).to(torch.long)

    if require_all_energies_valid:
        valid_per_spot = (valids > 0).all(dim=0)   # (K, M)
    else:
        valid_per_spot = (valids[centre_idx] > 0)

    in_panel = ((centres_y - half_h >= 0) & (centres_y + half_h < n_pixels_y) &
                (centres_z - half_w >= 0) & (centres_z + half_w < n_pixels_z))
    keep = (valid_per_spot & in_panel)
    kept = torch.nonzero(keep, as_tuple=False)
    spot_kidx = kept[:, 0]
    spot_midx = kept[:, 1]
    cy = centres_y[spot_kidx, spot_midx]
    cz = centres_z[spot_kidx, spot_midx]
    return RoiPlan(
        centres_y=cy, centres_z=cz,
        spot_kidx=spot_kidx, spot_midx=spot_midx,
        roi_h=roi_h, roi_w=roi_w,
        n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
    )


# ---------------------------------------------------------------------------
#  Differentiable per-ROI splat
# ---------------------------------------------------------------------------

def _per_spot_intensity(
    bank: PinkBank,
    plan: RoiPlan,
    two_theta: torch.Tensor,
    eta: torch.Tensor,
    *,
    element: Optional[str] = None,
    polarization_horiz_fraction: float = 1.0,
) -> torch.Tensor:
    """Per-(K_idx, M_idx) per-energy intensity weight from form factor + LP.

    Returns shape ``(N_E, n_kept)`` multiplier to apply to each spot's
    splat contribution.

    Form factor: ``|f(s^2, element)|^2`` where ``s = |G|/2`` is energy-
    independent, so the form-factor multiplier is per-(K_idx, M_idx) and
    constant across N_E. Differentiable through s^2.

    Lorentz-polarization: ``LP = (1 + p*cos(2eta)*cos(2theta)^2 +
    (1-p)*sin(2eta)*cos(2theta)^2) / sin(2theta)`` is energy-dependent
    via 2theta_k. Here ``p`` is the horizontal polarization fraction
    (typical synchrotron: ~0.95). The Lorentz factor 1/sin(2theta) is
    included; the standard HEDM rotation-Lorentz factor for grain-
    averaged intensity is 1/(sin(2theta)*sin(eta)).

    For the synthetic-test framework this is a per-spot dimensionless
    weight; absolute intensity scale is absorbed into ``intensity_per_spot``.
    """
    dtype = two_theta.dtype; device = two_theta.device
    N_E = two_theta.shape[0]
    n_kept = plan.n_kept

    # Slice to kept spots: shape (N_E, n_kept)
    tt = two_theta[:, plan.spot_kidx, plan.spot_midx]
    et = eta[:, plan.spot_kidx, plan.spot_midx]

    # Form factor (per-(hkl), constant across energies). Use centre energy's
    # g_mag for the (K_idx, M_idx) -> |G| mapping. The g_mag per (M_idx) is
    # bank.g_mag (M,); spread to (n_kept,) via spot_midx.
    if element is not None:
        try:
            from midas_hkls import form_factor as _ff
        except ImportError:
            _ff = None
        if _ff is not None:
            g_kept = bank.g_mag.to(dtype=dtype, device=device)[plan.spot_midx]
            s2 = (g_kept / 2.0) ** 2                     # (n_kept,)
            f = _ff(s2, element)                          # (n_kept,)
            f_factor = (f * f).reshape(1, n_kept)         # |f|^2, broadcast to N_E
        else:
            f_factor = torch.ones((1, n_kept), dtype=dtype, device=device)
    else:
        f_factor = torch.ones((1, n_kept), dtype=dtype, device=device)

    # LP factor: depends on (tt, et) which vary per (energy, spot).
    # Polarization: P_horiz * (1 - sin^2(2theta) * sin^2(eta))
    #             + P_vert  * (1 - sin^2(2theta) * cos^2(eta))
    p = polarization_horiz_fraction
    sin2t = torch.sin(tt); sin_2t = torch.sin(2.0 * tt)
    polarization = (
        p * (1.0 - sin_2t * sin_2t * torch.sin(et) ** 2) +
        (1.0 - p) * (1.0 - sin_2t * sin_2t * torch.cos(et) ** 2)
    )
    lp = polarization / (sin_2t.abs() + 1e-12)
    return f_factor * lp                                  # (N_E, n_kept)


def splat_rois(
    bank: PinkBank,
    plan: RoiPlan,
    euler_angles: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
    *,
    sigma_psf_px: float,
    intensity_per_spot: float = 1.0,
    use_panel_intensity_clip: bool = True,
    element: Optional[str] = None,
    polarization_horiz_fraction: float = 1.0,
    mosaicity_rad: float = 0.0,
) -> torch.Tensor:
    """Differentiable per-ROI image splat with optional form factor, LP,
    and mosaicity.

    ``element``: atomic symbol used to weight each spot by |f(s^2)|^2 via
        midas_hkls.form_factor. If None, all spots have unit form-factor
        weight (the pre-existing behaviour).
    ``polarization_horiz_fraction``: fraction of horizontal polarization
        for the LP factor (default 1.0 = pure horizontal; typical
        synchrotron ~ 0.95). LP is always applied; for backward-compatible
        unit-weight behaviour, the LP factor effectively cancels into the
        per-spot scale ``intensity_per_spot``.
    ``mosaicity_rad``: scalar isotropic mosaicity (std of grain
        orientation distribution, rad). Adds a Gaussian broadening to
        each spot in (eta, omega) which projects into a (y, z) blur with
        effective sigma sqrt(sigma_psf^2 + (L_sd * mosaicity / px)^2).
        Differentiable through ``mosaicity_rad``.

    Returns
    -------
    rois : Tensor (n_kept, roi_h, roi_w)
    """
    omegas, etas, tt, ys, zs, frames, valids = pink_forward_stacked(
        bank, euler_angles, positions, lattice_params=lattice_params,
    )
    if ys.dim() != 3:
        raise ValueError(
            f"splat_rois expects (N_E, K, M); got {tuple(ys.shape)}"
        )
    weights = bank.spectrum.weights().to(ys.dtype)   # (N_E,)
    N_E = ys.shape[0]
    n_kept = plan.n_kept
    if n_kept == 0:
        return torch.zeros((0, plan.roi_h, plan.roi_w),
                           dtype=ys.dtype, device=ys.device)

    # Grab per-spot per-energy positions: shape (n_kept, N_E)
    ys_kept = ys[:, plan.spot_kidx, plan.spot_midx].T.contiguous()
    zs_kept = zs[:, plan.spot_kidx, plan.spot_midx].T.contiguous()
    valid_kept = valids[:, plan.spot_kidx, plan.spot_midx].T.contiguous()  # (n_kept, N_E)

    # Mosaicity contribution to effective PSF sigma (in pixels). Accepts
    # mosaicity_rad as either a Python float (frozen) or a torch tensor
    # (learnable / differentiable). Lateral spot displacement from a
    # small grain-orientation tilt is Lsd * mosaicity at the detector.
    Lsd_um = float(bank.models[0].Lsd)
    px_um = float(bank.models[0].px)
    if isinstance(mosaicity_rad, torch.Tensor):
        sigma_mos_px = mosaicity_rad.abs() * (Lsd_um / px_um)
        sigma_eff = torch.sqrt(torch.tensor(sigma_psf_px ** 2, dtype=ys.dtype,
                                            device=ys.device) + sigma_mos_px ** 2)
        inv_two_sig2 = 1.0 / (2.0 * sigma_eff * sigma_eff)
    else:
        sigma_mos_px = float(mosaicity_rad) * Lsd_um / px_um if mosaicity_rad > 0 else 0.0
        sigma_eff = math.sqrt(sigma_psf_px ** 2 + sigma_mos_px ** 2)
        inv_two_sig2 = 1.0 / (2.0 * sigma_eff * sigma_eff)

    # ROI pixel grids: (n_kept, roi_h, roi_w)
    half_h = plan.roi_h // 2; half_w = plan.roi_w // 2
    yi = torch.arange(-half_h, half_h + 1, device=ys.device, dtype=ys.dtype)
    zi = torch.arange(-half_w, half_w + 1, device=ys.device, dtype=ys.dtype)
    Yg, Zg = torch.meshgrid(yi, zi, indexing="ij")
    # Anchor each grid at its ROI integer centre
    cy = plan.centres_y.to(ys.dtype).view(-1, 1, 1)   # (n_kept, 1, 1)
    cz = plan.centres_z.to(ys.dtype).view(-1, 1, 1)
    Y_abs = cy + Yg                                    # (n_kept, R, R)
    Z_abs = cz + Zg

    # diff per (n_kept, N_E, R, R); inv_two_sig2 was computed above
    dy = Y_abs.unsqueeze(1) - ys_kept.view(n_kept, N_E, 1, 1)
    dz = Z_abs.unsqueeze(1) - zs_kept.view(n_kept, N_E, 1, 1)
    psf = torch.exp(-(dy * dy + dz * dz) * inv_two_sig2)

    # Mask invalid energies for each spot (so they contribute zero)
    psf = psf * valid_kept.view(n_kept, N_E, 1, 1)

    # Per-spot form factor + LP weighting (N_E, n_kept). Apply ONLY when
    # the caller opts in via element=... or polarization_horiz_fraction<1.
    # Backward-compat: with defaults (element=None, polarization=1.0),
    # this matches the original "uniform per-spot weight" behaviour.
    use_extras = (element is not None) or (polarization_horiz_fraction != 1.0)
    if use_extras:
        fflp = _per_spot_intensity(
            bank, plan, tt, etas,
            element=element,
            polarization_horiz_fraction=polarization_horiz_fraction,
        )
        fflp = fflp / fflp.max().clamp_min(1e-30)
        # psf: (n_kept, N_E, R, R); permute to (N_E, n_kept, R, R) for broadcast.
        psf_p = psf.permute(1, 0, 2, 3)
        total_w = weights.view(N_E, 1) * fflp              # (N_E, n_kept)
        rois = (total_w.view(N_E, n_kept, 1, 1) * psf_p).sum(dim=0) * intensity_per_spot
    else:
        w = weights.view(1, N_E, 1, 1)
        rois = (w * psf).sum(dim=1) * intensity_per_spot
    return rois


# ---------------------------------------------------------------------------
#  Loss + optimisation
# ---------------------------------------------------------------------------

def splat_rois_3d(
    bank: PinkBank,
    plan: RoiPlan,
    euler_angles: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
    *,
    sigma_psf_px: float,
    sigma_frame: float = 0.5,
    roi_frames: int = 7,
    intensity_per_spot: float = 1.0,
    element: Optional[str] = None,
    polarization_horiz_fraction: float = 1.0,
    mosaicity_rad: float = 0.0,
) -> torch.Tensor:
    """3D differentiable per-ROI splat in (frame, y, z) space.

    Extends splat_rois to include the frame_nr axis. Each spot contributes
    a 3D Gaussian centred at (frame_k, y_k, z_k) with PSF
    (sigma_frame, sigma_psf_px, sigma_psf_px). The per-spot ROI is an
    anchored (roi_frames, roi_h, roi_w) volume centred on the spot's
    centre-energy frame_nr and (y, z).

    This is the right loss for testing acquisition-speed claims because
    fewer omega frames -> coarser frame_nr quantization, which a pure
    (y, z) loss does not see.

    Returns
    -------
    rois3d : Tensor (n_kept, roi_frames, roi_h, roi_w)
    """
    omegas, etas, tt, ys, zs, frames, valids = pink_forward_stacked(
        bank, euler_angles, positions, lattice_params=lattice_params,
    )
    if ys.dim() != 3:
        raise ValueError(f"splat_rois_3d expects (N_E, K, M); got {tuple(ys.shape)}")
    weights = bank.spectrum.weights().to(ys.dtype)
    N_E = ys.shape[0]
    n_kept = plan.n_kept
    if n_kept == 0:
        return torch.zeros((0, roi_frames, plan.roi_h, plan.roi_w),
                           dtype=ys.dtype, device=ys.device)

    centre_idx = N_E // 2
    # Centre frame from the centre-energy spot
    frames_kept = frames[:, plan.spot_kidx, plan.spot_midx].T.contiguous()  # (n_kept, N_E)
    ys_kept = ys[:, plan.spot_kidx, plan.spot_midx].T.contiguous()
    zs_kept = zs[:, plan.spot_kidx, plan.spot_midx].T.contiguous()
    valid_kept = valids[:, plan.spot_kidx, plan.spot_midx].T.contiguous()

    centres_f = torch.round(frames_kept[:, centre_idx]).to(torch.long)  # (n_kept,)
    half_f = roi_frames // 2
    half_h = plan.roi_h // 2; half_w = plan.roi_w // 2
    fi = torch.arange(-half_f, half_f + 1, device=ys.device, dtype=ys.dtype)
    yi = torch.arange(-half_h, half_h + 1, device=ys.device, dtype=ys.dtype)
    zi = torch.arange(-half_w, half_w + 1, device=ys.device, dtype=ys.dtype)
    Fg, Yg, Zg = torch.meshgrid(fi, yi, zi, indexing="ij")
    cf = centres_f.to(ys.dtype).view(-1, 1, 1, 1)
    cy = plan.centres_y.to(ys.dtype).view(-1, 1, 1, 1)
    cz = plan.centres_z.to(ys.dtype).view(-1, 1, 1, 1)
    F_abs = cf + Fg                                       # (n_kept, F, R, R)
    Y_abs = cy + Yg
    Z_abs = cz + Zg

    # Mosaicity contributes to (y, z) blur and to omega-axis blur (omega
    # spread from grain orientation tilt is ~ mosaicity / sin(theta), so
    # the frame-axis blur depends on theta per spot; we use a single
    # average value here for simplicity).
    Lsd_um = float(bank.models[0].Lsd)
    px_um = float(bank.models[0].px)
    sigma_mos_px = float(mosaicity_rad) * Lsd_um / px_um if mosaicity_rad > 0 else 0.0
    sigma_eff_yz = math.sqrt(sigma_psf_px ** 2 + sigma_mos_px ** 2)
    # Approximate omega broadening from mosaicity at 2theta ~ 4 deg (typical)
    omega_step_deg = abs(bank.models[0].omega_step) if hasattr(bank.models[0], "omega_step") else 0.25
    sigma_mos_omega_deg = float(mosaicity_rad) * 180.0 / math.pi / max(math.sin(4.0 * math.pi / 180.0), 1e-3) if mosaicity_rad > 0 else 0.0
    sigma_mos_frame = sigma_mos_omega_deg / omega_step_deg if mosaicity_rad > 0 else 0.0
    sigma_eff_frame = math.sqrt(sigma_frame ** 2 + sigma_mos_frame ** 2)

    inv_two_sig2_yz = 1.0 / (2.0 * sigma_eff_yz * sigma_eff_yz)
    inv_two_sig2_f = 1.0 / (2.0 * sigma_eff_frame * sigma_eff_frame)

    # Per (n_kept, N_E, F, R, R)
    df = F_abs.unsqueeze(1) - frames_kept.view(n_kept, N_E, 1, 1, 1)
    dy = Y_abs.unsqueeze(1) - ys_kept.view(n_kept, N_E, 1, 1, 1)
    dz = Z_abs.unsqueeze(1) - zs_kept.view(n_kept, N_E, 1, 1, 1)
    psf = torch.exp(
        -(dy * dy + dz * dz) * inv_two_sig2_yz
        - df * df * inv_two_sig2_f
    )
    psf = psf * valid_kept.view(n_kept, N_E, 1, 1, 1)
    use_extras = (element is not None) or (polarization_horiz_fraction != 1.0)
    if use_extras:
        fflp = _per_spot_intensity(
            bank, plan, tt, etas,
            element=element,
            polarization_horiz_fraction=polarization_horiz_fraction,
        )
        fflp = fflp / fflp.max().clamp_min(1e-30)
        psf_p = psf.permute(1, 0, 2, 3, 4)                # (N_E, n_kept, F, R, R)
        total_w = weights.view(N_E, 1) * fflp             # (N_E, n_kept)
        return (total_w.view(N_E, n_kept, 1, 1, 1) * psf_p).sum(dim=0) * intensity_per_spot
    else:
        w = weights.view(1, N_E, 1, 1, 1)
        return (w * psf).sum(dim=1) * intensity_per_spot


def roi_l2_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Mean L2 loss between predicted and observed ROI stacks.

    Accepts 2D (n_kept, R, R) or 3D (n_kept, F, R, R) shapes — same formula.
    Returns a scalar.
    """
    if predicted.shape != observed.shape:
        raise ValueError(f"shape mismatch: pred {predicted.shape}, obs {observed.shape}")
    return ((predicted - observed) ** 2).mean()


def compute_centroids(
    bank: PinkBank,
    plan: RoiPlan,
    euler_angles: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-kept-spot intensity-weighted centroid (y_px, z_px, frame_nr).

    Returns (n_kept, 3). Differentiable in euler / position / lattice.
    Centroid is computed as the spectrum-weight-weighted average of the
    per-energy spot positions, so the result is what a real peak finder
    would report (intensity-weighted centroid across the pink spectrum).
    """
    omegas, etas, tt, ys, zs, frames, valids = pink_forward_stacked(
        bank, euler_angles, positions, lattice_params=lattice_params,
    )
    if ys.dim() != 3:
        raise ValueError(f"compute_centroids expects (N_E, K, M); got {tuple(ys.shape)}")
    weights = bank.spectrum.weights().to(ys.dtype)             # (N_E,)
    # (N_E, n_kept)
    y_pe = ys[:, plan.spot_kidx, plan.spot_midx]
    z_pe = zs[:, plan.spot_kidx, plan.spot_midx]
    f_pe = frames[:, plan.spot_kidx, plan.spot_midx]
    v_pe = valids[:, plan.spot_kidx, plan.spot_midx].to(ys.dtype)
    w = weights.view(-1, 1) * v_pe                              # (N_E, n_kept)
    w_sum = w.sum(dim=0).clamp_min(1e-30)                       # (n_kept,)
    y_c = (w * y_pe).sum(dim=0) / w_sum
    z_c = (w * z_pe).sum(dim=0) / w_sum
    f_c = (w * f_pe).sum(dim=0) / w_sum
    return torch.stack([y_c, z_c, f_c], dim=-1)                 # (n_kept, 3)


def centroid_loss(
    predicted_centroids: torch.Tensor,
    observed_centroids: torch.Tensor,
    *,
    frame_to_px: float = 1.0,
) -> torch.Tensor:
    """Mean per-spot squared distance in (y, z, frame).

    The frame axis is scaled by ``frame_to_px`` so a 1-frame error
    contributes the same as a ``frame_to_px``-pixel (y, z) error. A
    physically motivated default is ``frame_to_px = Lsd * 2 * sin(theta) *
    omega_step_rad / px``, which equates orientation-induced (y, z) and
    omega shifts. For a typical FF cubic at 2theta ~ 4 deg, this is ~3.
    Set to 1.0 for unit equivalence, or pass a per-experiment value.
    """
    dy = predicted_centroids[:, 0] - observed_centroids[:, 0]
    dz = predicted_centroids[:, 1] - observed_centroids[:, 1]
    df = predicted_centroids[:, 2] - observed_centroids[:, 2]
    return (dy * dy + dz * dz + (frame_to_px * df) ** 2).mean()


@dataclass
class RecoveryConfig:
    sigma_psf_px: float = 1.5
    intensity_per_spot: float = 1.0
    phase1_steps: int = 30
    phase2_steps: int = 30
    phase3_steps: int = 30
    lbfgs_max_iter: int = 30
    lbfgs_lr_orient: float = 1e-2
    lbfgs_lr_lattice: float = 1e-3
    lbfgs_lr_pos: float = 1e1
    fit_position: bool = True
    fit_lattice: bool = True
    fit_orientation: bool = True
    fit_spectrum: bool = False
    verbose: bool = False
    # 3D (frame, y, z) loss instead of 2D (y, z) — needed when orientation
    # has DOF that shift spots primarily in omega. With 2D-only, rotations
    # about the diffraction vector g are nearly invisible to the loss.
    use_3d_loss: bool = False
    sigma_frame: float = 0.5
    roi_frames: int = 7
    # Centroid-distance loss. The image-space ROI loss has a basin of
    # attraction set by the ROI window size (≤ ~10 px shift typical),
    # so large orientation perturbations push predicted spots OUT of
    # their GT-anchored ROIs entirely → zero gradient. The centroid loss
    # ||pred_centroid - obs_centroid||² has gradient everywhere on the
    # panel, so the basin is the whole detector. Combined with image
    # loss for sub-pixel sharpening.
    centroid_loss_weight: float = 0.0
    centroid_frame_to_px: float = 3.0  # physical equivalence factor; see centroid_loss
    # Image-loss weight. Set to 0 in stage 1 of two-stage recovery
    # (centroid-only alignment, skips the expensive splat) and to 1 in
    # stage 2 (image-only refinement, which extracts strain + spectrum
    # info from spot shapes that centroid loss is blind to).
    image_loss_weight: float = 1.0


def recover_grain_state(
    bank: PinkBank,
    plan: RoiPlan,
    observed_rois: torch.Tensor,
    init_euler: torch.Tensor,
    init_position: torch.Tensor,
    init_lattice: torch.Tensor,
    cfg: RecoveryConfig,
    observed_centroids: Optional[torch.Tensor] = None,
) -> dict:
    """L-BFGS recovery of (orientation [, position] [, lattice] [, spectrum]).

    The phase schedule mirrors ``midas_diffract.optimize_single_grain``:
        Phase 1: orientation only
        Phase 2: lattice (and position if requested) only
        Phase 3: joint refinement of all enabled parameters

    If ``cfg.fit_spectrum`` is True and the bank's spectrum is not fixed,
    its ``logits`` are added to the joint phase 3 parameter list.
    """
    dtype = init_euler.dtype; device = init_euler.device
    opt_euler = init_euler.clone().detach().requires_grad_(False)
    opt_pos = init_position.clone().detach().requires_grad_(False)
    opt_latc = init_lattice.clone().detach().requires_grad_(False)

    history = {"phase1": [], "phase2": [], "phase3": []}

    def predict_rois():
        if cfg.use_3d_loss:
            return splat_rois_3d(
                bank, plan,
                opt_euler.unsqueeze(0), opt_pos.unsqueeze(0),
                opt_latc, sigma_psf_px=cfg.sigma_psf_px,
                sigma_frame=cfg.sigma_frame, roi_frames=cfg.roi_frames,
                intensity_per_spot=cfg.intensity_per_spot,
            )
        return splat_rois(
            bank, plan,
            opt_euler.unsqueeze(0), opt_pos.unsqueeze(0),
            opt_latc, sigma_psf_px=cfg.sigma_psf_px,
            intensity_per_spot=cfg.intensity_per_spot,
        )

    use_centroid = (cfg.centroid_loss_weight > 0.0 and observed_centroids is not None)
    use_image = (cfg.image_loss_weight > 0.0)

    def make_closure(params, target_history):
        def closure():
            for p in params:
                if p.grad is not None: p.grad.zero_()
            loss = torch.zeros((), dtype=opt_euler.dtype, device=opt_euler.device)
            if use_image:
                pred = predict_rois()
                loss = loss + cfg.image_loss_weight * roi_l2_loss(pred, observed_rois)
            if use_centroid:
                pred_c = compute_centroids(
                    bank, plan,
                    opt_euler.unsqueeze(0), opt_pos.unsqueeze(0), opt_latc,
                )
                loss = loss + cfg.centroid_loss_weight * centroid_loss(
                    pred_c, observed_centroids,
                    frame_to_px=cfg.centroid_frame_to_px,
                )
            loss.backward()
            target_history.append(float(loss.detach()))
            return loss
        return closure

    # -------- Phase 1: orientation --------
    if cfg.fit_orientation:
        opt_euler.requires_grad_(True)
        optimiser = torch.optim.LBFGS(
            [opt_euler], lr=cfg.lbfgs_lr_orient, max_iter=cfg.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
        for _ in range(cfg.phase1_steps):
            try:
                optimiser.step(make_closure([opt_euler], history["phase1"]))
            except Exception as e:
                if cfg.verbose: print(f"phase1 lbfgs error: {e}")
                break
        opt_euler.requires_grad_(False)

    # -------- Phase 2: position + lattice --------
    p2_params = []
    if cfg.fit_position:
        opt_pos.requires_grad_(True); p2_params.append(opt_pos)
    if cfg.fit_lattice:
        opt_latc.requires_grad_(True); p2_params.append(opt_latc)
    if p2_params:
        optimiser = torch.optim.LBFGS(
            p2_params, lr=cfg.lbfgs_lr_lattice, max_iter=cfg.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
        for _ in range(cfg.phase2_steps):
            try:
                optimiser.step(make_closure(p2_params, history["phase2"]))
            except Exception as e:
                if cfg.verbose: print(f"phase2 lbfgs error: {e}")
                break
        for p in p2_params: p.requires_grad_(False)

    # -------- Phase 3: joint --------
    p3_params = []
    if cfg.fit_orientation:
        opt_euler.requires_grad_(True); p3_params.append(opt_euler)
    if cfg.fit_position:
        opt_pos.requires_grad_(True); p3_params.append(opt_pos)
    if cfg.fit_lattice:
        opt_latc.requires_grad_(True); p3_params.append(opt_latc)
    if cfg.fit_spectrum and not bank.spectrum.fixed:
        bank.spectrum.logits.requires_grad_(True)
        p3_params.append(bank.spectrum.logits)
    if p3_params:
        optimiser = torch.optim.LBFGS(
            p3_params, lr=cfg.lbfgs_lr_orient * 0.5,
            max_iter=cfg.lbfgs_max_iter, line_search_fn="strong_wolfe",
        )
        for _ in range(cfg.phase3_steps):
            try:
                optimiser.step(make_closure(p3_params, history["phase3"]))
            except Exception as e:
                if cfg.verbose: print(f"phase3 lbfgs error: {e}")
                break
        for p in p3_params: p.requires_grad_(False)

    # Final-loss reporting. Skip the image splat entirely when image
    # loss is off (stage 1 of two-stage recovery) — the splat dim must
    # match observed_rois and a centroid-only stage may have use_3d_loss
    # disabled for speed. Report the centroid loss (or float('nan'))
    # instead in that case.
    if use_image:
        final_pred = predict_rois().detach()
        final_loss = float(roi_l2_loss(final_pred, observed_rois))
    elif use_centroid:
        with torch.no_grad():
            pred_c = compute_centroids(
                bank, plan,
                opt_euler.unsqueeze(0), opt_pos.unsqueeze(0), opt_latc,
            )
            final_loss = float(centroid_loss(
                pred_c, observed_centroids,
                frame_to_px=cfg.centroid_frame_to_px,
            ))
    else:
        final_loss = float("nan")

    return {
        "euler": opt_euler.detach().clone(),
        "position": opt_pos.detach().clone(),
        "lattice": opt_latc.detach().clone(),
        "spectrum_weights": bank.spectrum.weights().detach().clone(),
        "final_loss": final_loss,
        "history": history,
    }


def recover_two_stage(
    bank: PinkBank,
    plan: RoiPlan,
    observed_rois: torch.Tensor,
    init_euler: torch.Tensor,
    init_position: torch.Tensor,
    init_lattice: torch.Tensor,
    observed_centroids: torch.Tensor,
    cfg_centroid: RecoveryConfig,
    cfg_image: RecoveryConfig,
) -> dict:
    """Two-stage pose-then-profile recovery.

    Stage 1 (cfg_centroid) — centroid-only alignment. Loss is the
    per-spot ||pred_centroid - obs_centroid||² distance, which has
    gradient everywhere on the detector panel (basin = whole panel).
    Skip the expensive 3D ROI splat. Typically fit_orientation only,
    fit_lattice=False, fit_position optional.

    Stage 2 (cfg_image) — profile refinement. Loss is the image-space
    L2 between predicted and observed ROIs (2D or 3D). Carries
    sub-pixel orientation precision plus strain (lattice) and spectrum
    information that the centroid term is blind to. Typically
    fit_orientation + fit_lattice [+ fit_spectrum] jointly.

    The cfgs must satisfy:
      cfg_centroid.centroid_loss_weight > 0
      cfg_centroid.image_loss_weight == 0       (or just very small)
      cfg_image.centroid_loss_weight == 0       (so centroid stops adding noise)
      cfg_image.image_loss_weight > 0
    """
    # Stage 1 — centroid alignment
    stage1 = recover_grain_state(
        bank, plan, observed_rois,
        init_euler=init_euler, init_position=init_position,
        init_lattice=init_lattice, cfg=cfg_centroid,
        observed_centroids=observed_centroids,
    )

    # Stage 2 — profile refinement, warm-started from stage 1
    stage2 = recover_grain_state(
        bank, plan, observed_rois,
        init_euler=stage1["euler"], init_position=stage1["position"],
        init_lattice=stage1["lattice"], cfg=cfg_image,
        observed_centroids=None,
    )
    return {
        "euler": stage2["euler"],
        "position": stage2["position"],
        "lattice": stage2["lattice"],
        "spectrum_weights": stage2["spectrum_weights"],
        "final_loss": stage2["final_loss"],
        "stage1_loss": stage1["final_loss"],
        "stage1_history": stage1["history"],
        "stage2_history": stage2["history"],
    }


# ---------------------------------------------------------------------------
#  Spectrum-only fit (for proto3 calibrant stage)
# ---------------------------------------------------------------------------

def recover_joint(
    bank: PinkBank,
    plan: RoiPlan,
    observed_rois: torch.Tensor,
    init_euler: torch.Tensor,
    init_position: torch.Tensor,
    init_lattice: torch.Tensor,
    *,
    sigma_psf_px: float,
    intensity_per_spot: float = 1.0,
    n_outer_steps: int = 8,
    spec_steps_per_outer: int = 50,
    grain_steps_per_outer: int = 8,
    spec_lr: float = 0.05,
    grain_lbfgs_max_iter: int = 25,
    grain_lr: float = 5e-3,
    fit_lattice: bool = True,
    fit_position: bool = False,
    centroid_E0_penalty: float = 0.0,
) -> dict:
    """Joint S(E) + grain-state recovery via alternation.

    Each outer iteration:
      a) Adam-fit ``bank.spectrum.logits`` for ``spec_steps_per_outer``
         steps with the current grain state pinned.
      b) L-BFGS-fit ``(euler [, lattice])`` for ``grain_steps_per_outer``
         steps with the current S(E) pinned.

    Initialised by an "outer 0" spectrum-only fit on the wrong-spectrum
    init (gets S(E) close before grain refinement starts).

    Energy-lattice degeneracy
    -------------------------
    From a single grain's spots, an upward shift of the S(E) centroid by
    ``delta_E/E0`` is exactly cancelled by a uniform lattice contraction
    of the same fractional magnitude (Bragg's ``a*lambda = const`` for
    fixed centroid 2theta). To break this, set ``centroid_E0_penalty > 0``
    and the loss adds ``lambda * ((E_centroid - E0) / E0)**2``.
    Recommended values: 1e2-1e4 depending on data magnitude.
    """
    if bank.spectrum.fixed:
        raise ValueError("recover_joint requires a learnable spectrum")
    dtype = init_euler.dtype; device = init_euler.device
    opt_euler = init_euler.clone().detach()
    opt_pos = init_position.clone().detach()
    opt_latc = init_lattice.clone().detach()

    history = {"spec_loss": [], "grain_loss": [], "outer_state_loss": []}

    E0 = bank.spectrum.E0_keV
    Es_buf = bank.spectrum.energies_keV

    def _bd(t, expect_3d=False):
        # ensure forward-compatible shape: scalar (3,) -> (1, 3); already-batched left alone
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t

    def predict():
        return splat_rois(
            bank, plan,
            _bd(opt_euler), _bd(opt_pos), opt_latc,
            sigma_psf_px=sigma_psf_px,
            intensity_per_spot=intensity_per_spot,
        )

    def total_loss():
        l = roi_l2_loss(predict(), observed_rois)
        if centroid_E0_penalty > 0.0:
            w = bank.spectrum.weights().to(Es_buf.dtype)
            E_centre = (w * Es_buf).sum()
            shift_rel = (E_centre - E0) / E0
            l = l + centroid_E0_penalty * shift_rel * shift_rel
        return l

    def loss_now():
        with torch.no_grad():
            return float(roi_l2_loss(predict(), observed_rois))

    # ---- "outer 0" spectrum-only warmup -----------------------------
    bank.spectrum.logits.requires_grad_(True)
    opt = torch.optim.Adam([bank.spectrum.logits], lr=spec_lr)
    for _ in range(spec_steps_per_outer * 2):     # twice as long for warmup
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        opt.step()
        history["spec_loss"].append(float(loss.detach()))
    bank.spectrum.logits.requires_grad_(False)
    history["outer_state_loss"].append(loss_now())

    # ---- alternating outer loop -------------------------------------
    for outer in range(n_outer_steps):
        # (b) grain-state L-BFGS
        params = [opt_euler]; opt_euler.requires_grad_(True)
        if fit_lattice:
            opt_latc.requires_grad_(True); params.append(opt_latc)
        if fit_position:
            opt_pos.requires_grad_(True); params.append(opt_pos)
        lbfgs = torch.optim.LBFGS(
            params, lr=grain_lr, max_iter=grain_lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
        def closure_grain():
            for p in params:
                if p.grad is not None: p.grad.zero_()
            l = total_loss()
            l.backward()
            history["grain_loss"].append(float(l.detach()))
            return l
        for _ in range(grain_steps_per_outer):
            try:
                lbfgs.step(closure_grain)
            except Exception:
                break
        for p in params: p.requires_grad_(False)

        # (a) spectrum Adam
        bank.spectrum.logits.requires_grad_(True)
        opt = torch.optim.Adam([bank.spectrum.logits], lr=spec_lr)
        for _ in range(spec_steps_per_outer):
            opt.zero_grad()
            loss = total_loss()
            loss.backward()
            opt.step()
            history["spec_loss"].append(float(loss.detach()))
        bank.spectrum.logits.requires_grad_(False)
        history["outer_state_loss"].append(loss_now())

    final_loss = loss_now()
    return {
        "euler": opt_euler.detach().clone(),
        "position": opt_pos.detach().clone(),
        "lattice": opt_latc.detach().clone(),
        "spectrum_weights": bank.spectrum.weights().detach().clone(),
        "spectrum_logits": bank.spectrum.logits.detach().clone(),
        "final_loss": final_loss,
        "history": history,
    }


def fit_spectrum_to_rois(
    bank: PinkBank,
    plan: RoiPlan,
    observed_rois: torch.Tensor,
    euler: torch.Tensor,
    position: torch.Tensor,
    lattice: torch.Tensor,
    cfg: RecoveryConfig,
    *,
    n_steps: int = 100,
    lr: float = 0.05,
) -> dict:
    """Adam fit of ``bank.spectrum.logits`` only, with grain state fixed.

    Adam is preferred over L-BFGS for the unconstrained logit space because
    the softmax + image loss landscape can be ill-conditioned.
    """
    if bank.spectrum.fixed:
        raise ValueError("spectrum is fixed; cannot fit")
    bank.spectrum.logits.requires_grad_(True)
    opt = torch.optim.Adam([bank.spectrum.logits], lr=lr)
    history = []
    for _ in range(n_steps):
        opt.zero_grad()
        pred = splat_rois(
            bank, plan,
            euler.unsqueeze(0), position.unsqueeze(0), lattice,
            sigma_psf_px=cfg.sigma_psf_px,
            intensity_per_spot=cfg.intensity_per_spot,
        )
        loss = roi_l2_loss(pred, observed_rois)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    bank.spectrum.logits.requires_grad_(False)
    return {
        "weights": bank.spectrum.weights().detach().clone(),
        "logits": bank.spectrum.logits.detach().clone(),
        "history": history,
    }
