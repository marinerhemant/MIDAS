"""Synthetic identifiability test for the per-grain σ_ε microstrain DOF.

Mirrors the pf_odf ``test_anisotropic_spread.py`` pattern, adapted to the
per-grain (single-Σ) layout of grain_odf:

Plants a single-grain phantom with controlled per-spot radial broadening
sigma_radial = sqrt(sigma_yz² + (σ_ε · c_s)²) where c_s = 2·Lsd·tan(θ_s)/px,
then runs ``fit_grain_odf`` with ``refine_strain_spread=True`` and checks
that the recovered scalar σ_ε matches the planted value.

Scenarios:
  (a) σ_ε = 0       — recovery floor (stays near init floor)
  (b) σ_ε = 1e-3   — typical FF microstrain (broadens by ~few px radially)
  (c) σ_ε = 2e-3   — larger
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


# --- Synth helpers ----------------------------------------------------------


def _build_synth_strain(model, position, R_avg, sigma_yz, sigma_f,
                         patch_F, patch_P, strain_spread_planted):
    """Forward-simulate a single-grain phantom with planted per-spot σ_ε.

    Returns the kwargs needed by ``fit_grain_odf`` plus the per-spot c_s
    that was used to plant the data (for verification).
    """
    dtype = R_avg.dtype
    device = R_avg.device

    # Single planted orientation at R_avg (ODF is a delta). With K=1 the
    # planted "ODF" is fully described by the mean orientation; we won't
    # need to fit the ODF body — only σ_ε — so the K-particle structure
    # exists only to give the forward something to call.
    R_planted = R_avg.unsqueeze(0)                                  # (1, 3, 3)
    w_planted = torch.ones(1, dtype=dtype, device=device)

    spots = forward_orientations(model, R_planted, position)
    sy = spots.y_pixel.reshape(1, -1)                               # (1, 2M)
    sz = spots.z_pixel.reshape(1, -1)
    sf = spots.frame_nr.reshape(1, -1)
    sv = spots.valid.reshape(1, -1)

    # Select valid spots only.
    valid_global = (sv[0] > 0.5)
    spot_indexer = torch.nonzero(valid_global, as_tuple=False).squeeze(-1)
    sy_sel = sy[:, spot_indexer]
    sz_sel = sz[:, spot_indexer]
    sf_sel = sf[:, spot_indexer]
    sv_sel = sv[:, spot_indexer]
    anchor_y = sy_sel.squeeze(0).detach()
    anchor_z = sz_sel.squeeze(0).detach()
    anchor_f = sf_sel.squeeze(0).detach()

    # Per-spot Bragg-law scale c_s = 2·Lsd·tan(θ_s)/px_um, branch-major
    # ordering then index by spot_indexer.
    thetas = model.thetas.to(dtype).to(device)
    c_s_full = (2.0 * float(model.Lsd) * torch.tan(thetas).repeat(2)
                / float(model.px))                                  # (2M,)
    c_s = c_s_full[spot_indexer].detach()                           # (S,)

    # Per-spot anisotropic kernel widths reflecting the planted σ_ε.
    sigma_radial = torch.sqrt(
        torch.tensor(sigma_yz * sigma_yz, dtype=dtype, device=device)
        + (strain_spread_planted * c_s) ** 2
    )
    sigma_eta = torch.full_like(c_s, sigma_yz)

    y_BC = float(model.y_BC)
    z_BC = float(model.z_BC)
    dyR = anchor_y - y_BC
    dzR = anchor_z - z_BC
    rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
    radial_y = (dyR / rR).detach()
    radial_z = (dzR / rR).detach()

    spec = SpotPatchSpec(
        n_spots=int(spot_indexer.numel()),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=anchor_y.clone(), anchor_z=anchor_z.clone(),
        anchor_f=anchor_f.clone(),
        sigma_radial=sigma_radial, sigma_eta=sigma_eta,
        radial_y=radial_y, radial_z=radial_z,
    )
    measured_patches = splat_spots_to_patches(
        spec, sy_sel, sz_sel, sf_sel, w_planted, sv_sel,
    )

    return {
        "measured_y": anchor_y.clone(),
        "measured_z": anchor_z.clone(),
        "measured_f": anchor_f.clone(),
        "measured_patches": measured_patches,
        "spot_indexer": spot_indexer,
        "c_s": c_s,
    }


@pytest.mark.parametrize("planted_strain", [0.0, 1e-3, 2e-3])
def test_microstrain_identifiability(planted_strain):
    """Plant a per-grain σ_ε, freeze ODF, fit, assert recovery within 30%."""
    torch.manual_seed(0)
    model = make_model()
    position = torch.zeros(3, dtype=torch.float64)
    R_avg = random_orientation(seed=7).to(torch.float64)

    sigma_yz = 0.6
    sigma_f = 0.4
    patch_F = 5
    patch_P = 21       # generous; σ_ε=2e-3 with c_s~2000 needs ~±4 px room

    synth = _build_synth_strain(
        model, position, R_avg,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        patch_F=patch_F, patch_P=patch_P,
        strain_spread_planted=planted_strain,
    )

    # ODF frozen: K=1 particle at R_avg, lr=0 on both ODF parameter groups.
    odf = ParticleODF(R_avg=R_avg, K=1, theta_max=math.radians(0.1),
                       seed=0,
                       init_axis_angle=torch.zeros(1, 3, dtype=torch.float64))

    # Init σ_ε well off the truth so the test exercises recovery rather than
    # just confirming a good init.
    init_strain = 5e-4

    result = fit_grain_odf(
        odf, model, position,
        synth["measured_y"], synth["measured_z"], synth["measured_f"],
        synth["measured_patches"], synth["spot_indexer"],
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        delta_iters=1,           # σ_ε is the focus; one Delta pass is enough
        inner_steps=200,
        lr_axis_angle=0.0,       # freeze ODF axis-angle
        lr_logits=0.0,           # freeze ODF weights
        loss_norm="mean",
        refine_strain_spread=True,
        strain_spread_init=init_strain,
        strain_spread_microstrain_units=True,
        lr_strain_spread=5e-4,
    )

    assert result.strain_spread_fit is not None
    fit = float(result.strain_spread_fit.item())
    print(f"\nplanted σ_ε={planted_strain:.2e}  init={init_strain:.2e}  "
          f"recovered={fit:.3e}  final_loss={result.losses[-1]:.3e}")

    if planted_strain > 0:
        rel_err = abs(fit - planted_strain) / planted_strain
        assert rel_err < 0.30, (
            f"σ_ε recovery off: planted {planted_strain:.2e} -> "
            f"recovered {fit:.3e}  (rel err {rel_err:.2%})"
        )
    else:
        # Recovery floor: with σ_ε planted at 0, recovered value should
        # not run away. Use a generous bound — init is 5e-4, so anything
        # not larger than init * 2 is "stayed near floor".
        assert fit < 2.0 * init_strain, (
            f"σ_ε runaway with no planted signal: recovered {fit:.3e}"
        )
