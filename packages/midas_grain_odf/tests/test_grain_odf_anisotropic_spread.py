"""4-scenario joint σ_θ + σ_ε identifiability test for grain_odf.

Mirrors pf_odf's ``test_anisotropic_spread.py`` adapted to the single-grain
grain_odf layout. The forward simulates a phantom with controlled radial
broadening (σ_ε via per-spot c_s) and orientation-direction broadening
(σ_θ via sigma_eta + sigma_f), then runs ``fit_grain_odf`` with BOTH
``refine_strain_spread`` and ``refine_orientation_spread`` enabled and
checks recovery + crosstalk.

Scenarios:
  (a) σ_θ = 0,  σ_ε = 0     — recovery floor (both stay near init)
  (b) σ_θ = 1px, σ_ε = 0    — orientation-only: σ_θ recovered, σ_ε small
  (c) σ_θ = 0,  σ_ε = 1e-3  — strain-only: σ_ε recovered, σ_θ small
  (d) σ_θ = 1px, σ_ε = 1e-3 — both planted: both recovered, crosstalk small
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import make_model, random_orientation  # noqa: E402

from midas_grain_odf.forward_helpers import forward_orientations  # noqa: E402
from midas_grain_odf.inversion import fit_grain_odf  # noqa: E402
from midas_grain_odf.odf import ParticleODF  # noqa: E402
from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches  # noqa: E402


def _build_synth(model, position, R_avg, sigma_yz, sigma_f,
                  patch_F, patch_P,
                  sigma_theta_px, sigma_eps_strain):
    """Forward-simulate a phantom with planted (σ_θ, σ_ε) baked into the
    per-spot kernel widths. Single particle at R_avg → ODF is a delta and
    the only DOFs that affect loss are σ_θ and σ_ε."""
    dtype = R_avg.dtype
    device = R_avg.device

    R_planted = R_avg.unsqueeze(0)
    w_planted = torch.ones(1, dtype=dtype, device=device)

    spots = forward_orientations(model, R_planted, position)
    sy = spots.y_pixel.reshape(1, -1)
    sz = spots.z_pixel.reshape(1, -1)
    sf = spots.frame_nr.reshape(1, -1)
    sv = spots.valid.reshape(1, -1)
    valid = (sv[0] > 0.5)
    spot_indexer = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    sy_sel = sy[:, spot_indexer]; sz_sel = sz[:, spot_indexer]
    sf_sel = sf[:, spot_indexer]; sv_sel = sv[:, spot_indexer]
    anchor_y = sy_sel.squeeze(0).detach()
    anchor_z = sz_sel.squeeze(0).detach()
    anchor_f = sf_sel.squeeze(0).detach()

    # Per-spot c_s = 2·Lsd·tan(θ_s)/px_um for σ_ε → pixel-width coupling.
    thetas = model.thetas.to(dtype).to(device)
    c_s_full = (2.0 * float(model.Lsd) * torch.tan(thetas).repeat(2)
                / float(model.px))
    c_s = c_s_full[spot_indexer].detach()

    sigma_radial = torch.sqrt(
        torch.tensor(sigma_yz * sigma_yz, dtype=dtype, device=device)
        + (sigma_eps_strain * c_s) ** 2
    )
    sigma_eta = torch.full_like(c_s, math.sqrt(sigma_yz ** 2 + sigma_theta_px ** 2))
    sigma_f_per_spot = torch.full_like(c_s, math.sqrt(sigma_f ** 2 + sigma_theta_px ** 2))

    y_BC = float(model.y_BC); z_BC = float(model.z_BC)
    dyR = anchor_y - y_BC; dzR = anchor_z - z_BC
    rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
    radial_y = (dyR / rR).detach(); radial_z = (dzR / rR).detach()

    spec = SpotPatchSpec(
        n_spots=int(spot_indexer.numel()),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=anchor_y.clone(), anchor_z=anchor_z.clone(),
        anchor_f=anchor_f.clone(),
        sigma_radial=sigma_radial, sigma_eta=sigma_eta,
        sigma_f_per_spot=sigma_f_per_spot,
        radial_y=radial_y, radial_z=radial_z,
    )
    measured_patches = splat_spots_to_patches(
        spec, sy_sel, sz_sel, sf_sel, w_planted, sv_sel,
    )
    return dict(
        measured_y=anchor_y.clone(), measured_z=anchor_z.clone(),
        measured_f=anchor_f.clone(),
        measured_patches=measured_patches,
        spot_indexer=spot_indexer,
    )


@pytest.mark.parametrize("planted_theta,planted_eps", [
    (0.0, 0.0),       # (a) recovery floor
    (1.0, 0.0),       # (b) orientation only
    (0.0, 1e-3),      # (c) strain only
    (1.0, 1e-3),      # (d) both
])
def test_aniso_identifiability(planted_theta, planted_eps):
    """Plant (σ_θ, σ_ε), fit jointly with ODF frozen, check recovery +
    crosstalk."""
    torch.manual_seed(0)
    model = make_model()
    position = torch.zeros(3, dtype=torch.float64)
    R_avg = random_orientation(seed=7).to(torch.float64)

    sigma_yz = 0.6; sigma_f = 0.4
    patch_F = 5; patch_P = 21

    synth = _build_synth(
        model, position, R_avg,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        patch_F=patch_F, patch_P=patch_P,
        sigma_theta_px=planted_theta,
        sigma_eps_strain=planted_eps,
    )

    odf = ParticleODF(R_avg=R_avg, K=1, theta_max=math.radians(0.1), seed=0,
                       init_axis_angle=torch.zeros(1, 3, dtype=torch.float64))

    # Inits well off truth in both directions so the test exercises recovery.
    eps_init = 5e-4; theta_init = 0.3   # pixels

    result = fit_grain_odf(
        odf, model, position,
        synth["measured_y"], synth["measured_z"], synth["measured_f"],
        synth["measured_patches"], synth["spot_indexer"],
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        delta_iters=1, inner_steps=500,
        lr_axis_angle=0.0, lr_logits=0.0,
        loss_norm="mean",
        refine_strain_spread=True,
        strain_spread_init=eps_init,
        strain_spread_microstrain_units=True,
        lr_strain_spread=5e-4,
        refine_orientation_spread=True,
        orientation_spread_init=theta_init,
        lr_orientation_spread=50.0,
    )

    eps_fit = float(result.strain_spread_fit.item())
    th_fit = float(result.orientation_spread_fit.item())
    print(f"\nplanted (σ_θ,σ_ε)=({planted_theta:.2f}px,{planted_eps:.2e})  "
          f"recovered ({th_fit:.3f}px,{eps_fit:.3e})  "
          f"final_loss={result.losses[-1]:.3e}")

    # Joint scenario (d) has reduced precision: with both DOFs active on a
    # single grain (only 2 free params constrained by ~hundred spots), σ_θ
    # and σ_ε can trade off — both can compensate for each other within a
    # ~30-50% band. The unambiguous claims are scenarios (b) and (c) where
    # one of the DOFs is zero — those decouple cleanly and recover to 30%.
    joint_case = (planted_theta > 0 and planted_eps > 0)
    bound = 0.50 if joint_case else 0.30
    if planted_eps > 0:
        rel = abs(eps_fit - planted_eps) / planted_eps
        assert rel < bound, (
            f"σ_ε recovery off: planted {planted_eps:.2e} -> {eps_fit:.3e} "
            f"(rel err {rel:.2%}, bound {bound:.0%})"
        )
    else:
        assert eps_fit < 2.0 * eps_init, (
            f"σ_ε runaway with no planted signal: recovered {eps_fit:.3e}"
        )
    if planted_theta > 0:
        rel = abs(th_fit - planted_theta) / planted_theta
        assert rel < bound, (
            f"σ_θ recovery off: planted {planted_theta:.3f}px -> {th_fit:.3f}px "
            f"(rel err {rel:.2%}, bound {bound:.0%})"
        )
    else:
        assert th_fit < 2.0 * theta_init, (
            f"σ_θ runaway with no planted signal: recovered {th_fit:.3f}"
        )

    # Crosstalk: when one is planted ZERO the recovered value should be
    # well below the orthogonal planted scale (the observables are
    # orthogonal, so crosstalk should be small).
    if planted_theta == 0 and planted_eps > 0:
        # σ_θ shouldn't be inflated by σ_ε signal — limit at init scale.
        assert th_fit < theta_init * 3.0, (
            f"crosstalk: σ_θ={th_fit:.3f} too large with σ_ε={planted_eps:.2e}"
        )
    if planted_eps == 0 and planted_theta > 0:
        assert eps_fit < eps_init * 3.0, (
            f"crosstalk: σ_ε={eps_fit:.3e} too large with σ_θ={planted_theta:.2f}px"
        )
