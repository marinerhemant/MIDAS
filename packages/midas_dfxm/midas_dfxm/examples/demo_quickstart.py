"""midas-dfxm 60-second quickstart demo (CPU, no GPU, no data files needed).

Runs the whole differentiable DFXM twin end to end on a synthetic curved+strained
crystal grain:
  1. build a deformation field (smooth lattice rotation + a strain gradient),
  2. render a realistic DFXM image and the standard center-of-mass mosaicity map
     (the product current DFXM analysis stops at),
  3. recover the FULL nine-component deformation gradient F per voxel from a few
     reflections with the differentiable inverse, and report the round-trip error
     (this is the extra information the twin adds on top of the mosaicity map).

Run:
  set KMP_DUPLICATE_LIB_OK=TRUE        (Windows: setx, or in the shell)
  python examples/demo_quickstart.py

Saves demo_quickstart.png next to this script. Pure Python + torch (CPU); on
Windows nothing else is needed.
"""
from __future__ import annotations
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from midas_dfxm import (
    make_uniform_field, with_orientation_gradient,
    GoniometerSetting, reference_q_nom, aligned_resolution,
    ObjectiveOptics, bragg_two_theta_deg, dfxm_image,
)
from midas_dfxm.field_inverse import deformation_observable, recover_deformation_direct
from midas_dfxm.examples._figures import figure_dir

DT = torch.float64


def main():
    # 1. a curved crystal grain: smooth lattice rotation across x, plus a small
    #    axial strain gradient injected into F_xx so the field is not pure rotation.
    field = make_uniform_field(shape=(48, 48, 1), spacing_um=0.5)
    field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
    F = field.F.clone()
    nx, ny = 48, 48
    xfrac = torch.linspace(-1, 1, nx, dtype=F.dtype).repeat_interleave(ny)
    F[:, 0, 0] = F[:, 0, 0] + 1.0e-3 * xfrac              # e_xx strain gradient
    import dataclasses
    field = dataclasses.replace(field, F=F.to(DT))

    # 2. one realistic DFXM image
    hkl, center = (1, 1, 1), GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), wavelength_A=0.172979)
    optics = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, detector_shape=(200, 200))
    image = dfxm_image(field, hkl, center, res, optics)

    # 3. FULL nine-component F recovery from 4 non-coplanar reflections (+ noise)
    refls = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3)]
    meas = deformation_observable(field, refls)
    meas = meas + 1e-3 * meas.abs().mean() * torch.randn_like(meas)
    F_rec = recover_deformation_direct(meas, refls, field=field)
    err = (F_rec - field.F).abs()
    # orientation channel (what a COM mosaicity scan gives) = lattice rotation about z
    rot_rec = (0.5 * (F_rec[:, 1, 0] - F_rec[:, 0, 1])).reshape(nx, ny) * (180.0 / torch.pi) * 1e3
    exx_true = (field.F[:, 0, 0] - 1.0).reshape(nx, ny)
    exx_rec = (F_rec[:, 0, 0] - 1.0).reshape(nx, ny)
    print(f"full-F round-trip: max |dF| = {float(err.max()):.2e}, "
          f"mean = {float(err.mean()):.2e}")
    print(f"recovered e_xx range [{float(exx_rec.min()):.2e}, {float(exx_rec.max()):.2e}] "
          f"vs truth [{float(exx_true.min()):.2e}, {float(exx_true.max()):.2e}]")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))
    im0 = ax[0].imshow(image.detach().T, origin="lower", cmap="magma")
    ax[0].set_title("(a) realistic DFXM image"); fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(rot_rec.detach().T, origin="lower", cmap="twilight")
    ax[1].set_title("(b) lattice rotation (mdeg)\norientation channel (= COM product)")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    im2 = ax[2].imshow(exx_true.detach().T * 1e6, origin="lower", cmap="RdBu_r")
    ax[2].set_title(r"(c) planted $\epsilon_{xx}$ ($\mu\epsilon$)"); fig.colorbar(im2, ax=ax[2], shrink=0.8)
    im3 = ax[3].imshow(exx_rec.detach().T * 1e6, origin="lower", cmap="RdBu_r")
    ax[3].set_title(r"(d) recovered $\epsilon_{xx}$ ($\mu\epsilon$)" + "\nfull-F inverse (extra info)")
    fig.colorbar(im3, ax=ax[3], shrink=0.8)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("midas-dfxm quickstart: DFXM forward -> orientation (COM) + full-F strain inverse", y=1.02)
    fig.tight_layout()
    out = os.path.join(figure_dir(), "demo_quickstart.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("figure ->", out)


if __name__ == "__main__":
    main()
