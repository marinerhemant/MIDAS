"""Phase 2 anchor — DFXM field forward on a realistic bent / sub-grain crystal.

Item #5 of the post-Phase-5 roadmap: the "does this look like real DFXM" gut check,
and the drop-in shape for an external collaborator field (swap the planted field for
generators.field_from_deformation_gradient(F_external, positions)).

Produces:
  (a) a mosaicity map — local lattice rotation from the center-of-mass of a (chi,phi)
      rocking scan (the standard DFXM orientation product), vs the planted ground truth;
  (b) DFXM images at three rocking angles showing the diffracting sub-region sweep
      across a bent crystal (the qualitative DFXM signature).

Saves field_forward.png under the directory chosen by ``figure_dir``.
Run:  export KMP_DUPLICATE_LIB_OK=TRUE; python -m midas_dfxm.examples.tutorial_field_forward
"""
from __future__ import annotations

import os

import torch

from midas_dfxm import (
    GoniometerSetting,
    ObjectiveOptics,
    aligned_resolution,
    bragg_two_theta_deg,
    dfxm_image,
    make_uniform_field,
    mosaicity_scan,
    reference_q_nom,
    voxel_intensity,
    with_orientation_gradient,
)
from midas_dfxm.examples._figures import figure_dir

DT = torch.float64
WAVELENGTH = 0.172979


def build_bent_crystal(n=64, spacing=0.5):
    """A smoothly bent crystal: lattice curvature about z, growing along x."""
    field = make_uniform_field(shape=(n, n, 1), spacing_um=spacing, dtype=DT)
    return with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)


def mosaicity_com_map(field, hkl, settings, res):
    """Per-voxel center-of-mass tilt (chi, phi) — the standard DFXM orientation map."""
    stack = torch.stack([voxel_intensity(field, hkl, s, res) for s in settings], dim=0)  # (S, N)
    chi = torch.tensor([s.chi for s in settings], dtype=DT)
    phi = torch.tensor([s.phi for s in settings], dtype=DT)
    w = stack.clamp_min(0) + 1e-12
    wsum = w.sum(0)
    chi_com = (w * chi[:, None]).sum(0) / wsum
    phi_com = (w * phi[:, None]).sum(0) / wsum
    return chi_com, phi_com


def main():
    outdir = figure_dir()

    field = build_bent_crystal()
    nx, ny, _ = field.shape
    hkl, center = (1, 1, 1), GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=4e-3)

    # (a) mosaicity map from a (chi, phi) scan.
    settings = mosaicity_scan(center, chi_range=(-0.5, 0.5), phi_range=(-0.1, 0.1),
                              n_chi=21, n_phi=5)
    chi_com, phi_com = mosaicity_com_map(field, hkl, settings, res)
    chi_map = chi_com.reshape(nx, ny)
    # Ground-truth local rotation about z, projected onto the chi axis.
    R, _ = field.local_rotation_stretch()
    import math
    truth = torch.tensor([math.degrees(math.atan2(R[i, 1, 0].item(), R[i, 0, 0].item()))
                          for i in range(field.n_voxels)], dtype=DT).reshape(nx, ny)
    corr = torch.corrcoef(torch.stack([chi_map.reshape(-1), truth.reshape(-1)]))[0, 1]
    print(f"(a) mosaicity map vs planted tilt: correlation = {float(corr):.4f}")

    # (b) DFXM images at three rocking angles.
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), WAVELENGTH)
    optics = ObjectiveOptics(two_theta_deg=tt, magnification=4.0, pixel_um=1.0,
                             detector_shape=(200, 200))
    imgs = {}
    for chi in (-0.3, 0.0, 0.3):
        imgs[chi] = dfxm_image(field, hkl, GoniometerSetting(chi=chi), res, optics).detach()
        print(f"(b) rocking chi={chi:+.1f} deg: image sum={float(imgs[chi].sum()):.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        im0 = ax[0].imshow(chi_map.T, origin="lower", cmap="twilight")
        ax[0].set_title(f"(a) mosaicity map  (corr={float(corr):.3f})")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)
        for j, chi in enumerate((-0.3, 0.0, 0.3)):
            ax[j + 1].imshow(imgs[chi].T, origin="lower", cmap="magma")
            ax[j + 1].set_title(f"(b) DFXM image  chi={chi:+.1f} deg")
        fig.tight_layout()
        out = os.path.join(outdir, "field_forward.png")
        fig.savefig(out, dpi=140)
        print(f"saved figure -> {os.path.normpath(out)}")
    except ImportError:
        print("matplotlib unavailable; numbers printed above")


if __name__ == "__main__":
    main()
