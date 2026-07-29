"""Per-hkl ⟨C̄⟩ weighting (mWH) identifiability test.

Plants a single-grain phantom whose per-spot radial broadening uses
σ_radial² = σ_yz² + (σ_ε · c_s · sqrt(⟨C̄⟩_hkl))² with non-uniform
⟨C̄⟩_hkl across reflections (computed via midas_defect's
``average_contrast_factor`` for FCC). The fit MUST be supplied the same
⟨C̄⟩ via ``contrast_factor_per_spot`` to recover the planted scalar; if
the fit ignores the contrast factor, it should land on the
intensity-weighted average — verifiably different from truth.
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


def _build_synth_with_contrast(model, position, R_avg, sigma_yz, sigma_f,
                                 patch_F, patch_P, sigma_eps_strain, cbar_per_spot):
    """Plant patches where per-spot radial width depends on ⟨C̄⟩_hkl."""
    dtype = R_avg.dtype; device = R_avg.device
    R_planted = R_avg.unsqueeze(0)
    w = torch.ones(1, dtype=dtype, device=device)
    spots = forward_orientations(model, R_planted, position)
    sy = spots.y_pixel.reshape(1, -1); sz = spots.z_pixel.reshape(1, -1)
    sf = spots.frame_nr.reshape(1, -1); sv = spots.valid.reshape(1, -1)
    idx = torch.nonzero(sv[0] > 0.5, as_tuple=False).squeeze(-1)
    thetas = model.thetas.to(dtype).to(device)
    c_s_full = (2.0 * float(model.Lsd) * torch.tan(thetas).repeat(2) / float(model.px))
    c_s = c_s_full[idx].detach()
    cbar = cbar_per_spot[idx].detach()

    sigma_radial = torch.sqrt(
        torch.tensor(sigma_yz * sigma_yz, dtype=dtype, device=device)
        + (sigma_eps_strain * c_s * torch.sqrt(cbar)) ** 2
    )
    sigma_eta = torch.full_like(c_s, sigma_yz)
    sigma_f_const = torch.full_like(c_s, sigma_f)

    anchor_y = sy[0, idx].detach(); anchor_z = sz[0, idx].detach()
    anchor_f = sf[0, idx].detach()
    y_BC, z_BC = float(model.y_BC), float(model.z_BC)
    dyR = anchor_y - y_BC; dzR = anchor_z - z_BC
    rR = torch.sqrt(dyR*dyR + dzR*dzR).clamp(min=1.0)
    ry = (dyR/rR).detach(); rz = (dzR/rR).detach()

    spec = SpotPatchSpec(
        n_spots=int(idx.numel()), patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=anchor_y.clone(), anchor_z=anchor_z.clone(), anchor_f=anchor_f.clone(),
        sigma_radial=sigma_radial, sigma_eta=sigma_eta,
        sigma_f_per_spot=sigma_f_const, radial_y=ry, radial_z=rz,
    )
    meas = splat_spots_to_patches(spec, sy[:, idx], sz[:, idx], sf[:, idx],
                                    w, sv[:, idx])
    return dict(
        measured_y=anchor_y.clone(), measured_z=anchor_z.clone(),
        measured_f=anchor_f.clone(), measured_patches=meas,
        spot_indexer=idx, c_s=c_s, cbar=cbar,
    )


def _cbar_per_unique_hkl(model):
    """Compute ⟨C̄⟩_hkl per unique reflection via midas_defect (FCC, screw).

    Uses 304L SS-like stiffness (c11=209, c12=133, c44=121 GPa).
    """
    from midas_defect.contrast_factor import cubic_stiffness, average_contrast_factor
    C6 = cubic_stiffness(209.0, 133.0, 121.0, dtype=torch.float64)
    hkls_int = model.hkls_int.detach().cpu().numpy()
    M = hkls_int.shape[0]
    cbar = torch.zeros(M, dtype=torch.float64)
    for m in range(M):
        cbar[m] = average_contrast_factor(
            C6, hkls_int[m], family="fcc", character="screw", n_phi=180,
        )
    # Branch-major repeat(2) to match the flat (2M,) spot layout.
    return cbar.repeat(2)


@pytest.mark.parametrize("planted_strain", [1e-3, 2e-3])
def test_contrast_factor_recovery(planted_strain):
    """Plant with non-uniform ⟨C̄⟩_hkl; assert σ_ε recovers truth when
    ⟨C̄⟩ is provided to the fit."""
    torch.manual_seed(0)
    model = make_model()
    position = torch.zeros(3, dtype=torch.float64)
    R_avg = random_orientation(seed=7).to(torch.float64)
    sigma_yz, sigma_f = 0.6, 0.4
    patch_F, patch_P = 5, 21

    cbar_per_spot_full = _cbar_per_unique_hkl(model)        # (2M,)
    # Sanity: there IS variation across reflections.
    cv = cbar_per_spot_full.std() / cbar_per_spot_full.mean()
    assert float(cv) > 0.01, "⟨C̄⟩ should vary across reflections"

    synth = _build_synth_with_contrast(
        model, position, R_avg, sigma_yz, sigma_f, patch_F, patch_P,
        planted_strain, cbar_per_spot_full,
    )
    cbar_matched = cbar_per_spot_full[synth["spot_indexer"]]
    odf = ParticleODF(R_avg=R_avg, K=1, theta_max=math.radians(0.1), seed=0,
                       init_axis_angle=torch.zeros(1, 3, dtype=torch.float64))

    result = fit_grain_odf(
        odf, model, position,
        synth["measured_y"], synth["measured_z"], synth["measured_f"],
        synth["measured_patches"], synth["spot_indexer"],
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        delta_iters=1, inner_steps=200,
        lr_axis_angle=0.0, lr_logits=0.0,
        loss_norm="mean",
        refine_strain_spread=True,
        strain_spread_init=5e-4, lr_strain_spread=5e-4,
        strain_spread_microstrain_units=True,
        contrast_factor_per_spot=cbar_matched,
    )

    fit = float(result.strain_spread_fit.item())
    print(f"\nplanted σ_ε={planted_strain:.2e} (with non-uniform ⟨C̄⟩, "
          f"mean={float(cbar_matched.mean()):.3f}, cv={float(cv):.3f})  "
          f"recovered={fit:.3e}  final_loss={result.losses[-1]:.3e}")
    rel = abs(fit - planted_strain) / planted_strain
    assert rel < 0.30, (
        f"σ_ε with ⟨C̄⟩ recovery off: planted {planted_strain:.2e} -> "
        f"recovered {fit:.3e}  (rel err {rel:.2%})"
    )
