#!/usr/bin/env python
"""ExaDiS -> DFXM: virtual dark-field images of a discrete-dislocation network.

Generates a microstructure with ExaDiS -- irradiation-type prismatic loops plus
deformation-type lines -- converts it through :mod:`midas_ddd`, and renders DFXM
images from the resulting elastic distortion field.

    ExaDiS generators  ->  midas_ddd.DislocationNetwork
                       ->  midas_dfxm.network_deformation_field   (elastic beta)
                       ->  midas_dfxm.dfxm_image                  (per setting)

Runs with or without ExaDiS. With `pyexadis` importable it uses the real
generators; without it, `midas_ddd.generate` builds an equivalent configuration
so the example still demonstrates the chain. The `--source` line in the output
says which was used.

Prior art, and it is close: Wang, Bertin, Pal, Irvine, Katagiri, Rudd &
Dresselhaus-Marais, arXiv:2409.01439, compute virtual DFXM images from discrete
dislocation structures with a non-singular formulation and geometrical optics.
That is this capability. Nothing here is claimed as new -- what MIDAS adds is
that the same network also drives the small-angle and near-Bragg forwards
through one kernel.

Why the ELASTIC distortion: DFXM images the lattice, which is continuous across
a dislocation's cut surface for a perfect Burgers vector. The plastic
eigendistortion on that cut is bookkeeping and must not enter the contrast. See
`midas_ddd.realspace`.

Run:
    python exadis_to_dfxm.py [--out DIR] [--grid 61] [--half-um 5.0]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from midas_ddd import (
    combine,
    cubic_stiffness,
    find_loops,
    prismatic_loop,
    relaxation_volumes_um3,
    straight_line,
    validate_network,
)
from midas_dfxm.dislocation import network_deformation_field
from midas_dfxm.forward import dfxm_image
from midas_dfxm.conventions import GoniometerSetting
from midas_dfxm.optics import ObjectiveOptics
from midas_dfxm.resolution import aligned_resolution
from midas_dfxm.scan import bragg_two_theta_deg, reference_q_nom

DT = torch.float64
WAVELENGTH = 0.172979                 # Angstrom, ~71.7 keV
B_CU_A = 2.556
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)     # Cu, GPa
HKL = (2, -2, 0)


def build_network(box_um=10.0, n_loops=8, n_lines=6, seed=7):
    """ExaDiS if available, otherwise the pure-python equivalent."""
    try:
        from midas_ddd.exadis import (
            generate_line_config,
            generate_prismatic_config,
            have_pyexadis,
        )
        if not have_pyexadis():
            raise ImportError
        box_b = box_um / (B_CU_A * 1e-4)
        loops = generate_prismatic_config(
            crystal="fcc", box_size_b=box_b, num_loops=n_loops,
            radius_b=[0.15 * box_b / 4, 0.30 * box_b / 4],
            b_magnitude_A=B_CU_A, maxseg_b=box_b / 100, seed=seed)
        lines = generate_line_config(
            crystal="fcc", box_size_b=box_b, num_lines=n_lines,
            b_magnitude_A=B_CU_A, maxseg_b=box_b / 40, seed=seed + 1)
        return (combine([loops, lines], cell_size_um=box_um),
                "ExaDiS (pyexadis, in process)")
    except ImportError:
        g = torch.Generator().manual_seed(seed)
        parts = []
        for _ in range(n_loops):
            c = (torch.rand(3, generator=g, dtype=DT) * 2 - 1) * (0.3 * box_um)
            R = float(0.2 + 0.3 * torch.rand(1, generator=g, dtype=DT))
            parts.append(prismatic_loop(radius_um=R, burgers=(0, 0, 1.0),
                                        n_segments=24, center_um=tuple(c.tolist()),
                                        cell_size_um=box_um))
        for _ in range(n_lines):
            c = (torch.rand(3, generator=g, dtype=DT) * 2 - 1) * (0.3 * box_um)
            parts.append(straight_line(length_um=0.4 * box_um, n_segments=10,
                                       center_um=tuple(c.tolist()),
                                       cell_size_um=box_um))
        return (combine(parts, cell_size_um=box_um),
                "midas_ddd.generate (ExaDiS not installed)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--half-um", type=float, default=5.0)
    ap.add_argument("--box-um", type=float, default=10.0)
    ap.add_argument("--slab-um", type=float, default=1.0,
                    help="beam thickness sampled along z")
    ap.add_argument("--slab-voxels", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=None,
                    help="resolution width (rad); default: matched to the field")
    args = ap.parse_args()

    net, source = build_network(box_um=args.box_um)
    print(f"--source: {source}")
    print(f"network : {net.n_nodes} nodes, {net.n_segments} segments, "
          f"cell {float(net.cell_size_um.max()):.2f} um")
    summary = validate_network(net)
    print(f"          Burgers conserved: {summary['burgers_conserved']}, "
          f"closed loops: {summary['n_loops']}, "
          f"density {summary['dislocation_density_um2']:.3g} um^-2")
    loops = find_loops(net)
    if loops:
        dV = relaxation_volumes_um3(net, loops)
        print(f"          relaxation volumes: {float(dV.min()):.2e} .. "
              f"{float(dV.max()):.2e} um^3")
    print("          NOTE: the open lines carry no relaxation volume and are "
          "invisible to the\n                small-angle forward -- but they image "
          "fine here, which is the point.")

    # A SLAB, not a sheet. DFXM integrates through the thickness of the
    # line-focused beam, and a zero-thickness plane through a sparse 3-D network
    # mostly misses the dislocations: the first attempt gave max|F-I| = 2e-6
    # against a resolution width of 8e-3, i.e. every voxel at the same point in
    # reciprocal space and a uniformly lit square with no contrast. Sampling the
    # beam thickness puts voxels near the cores, which is where the contrast is.
    n, half = args.grid, args.half_um
    nz = args.slab_voxels
    xs = torch.linspace(-half, half, n, dtype=DT)
    zs = (torch.zeros(1, dtype=DT) if nz == 1
          else torch.linspace(-args.slab_um / 2, args.slab_um / 2, nz, dtype=DT))
    gx, gy, gz = torch.meshgrid(xs, xs, zs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    t0 = time.time()
    field = network_deformation_field(pts, net, CU, shape=(n, n, nz))
    print(f"elastic distortion on {n}x{n} voxels in {time.time() - t0:.1f} s")
    dev = (field.F - torch.eye(3, dtype=DT)).abs().max()
    print(f"max |F - I| = {float(dev):.3e}")

    # Rotation statistics. Report the MEDIAN, not the max.
    #
    # The distortion diverges as 1/r at a dislocation core, so max, std and RMS
    # of the rotation field do not converge -- they grow with how many voxels
    # happen to land near a core. Measured on this configuration: changing only
    # the number of slab planes (1 -> 31) swings the max by 7.6x, and shifting
    # the grid origin by a fraction of a voxel moves the std by 30 %. An earlier
    # version of this script quoted max|w| = 6.7e-6 rad and built a detectability
    # argument on it; that number was a property of the grid, not of the network,
    # and independent resampling at DFXM-scale voxels gave 550-2000x more.
    # Only the median is resolution-stable.
    from midas_dfxm.field_inverse import decompose_deformation
    w = decompose_deformation(field.F)["rotation_vector"].abs()
    wmax_per_voxel = w.max(dim=-1).values
    qs = torch.quantile(wmax_per_voxel,
                        torch.tensor([0.5, 0.9, 0.99], dtype=DT))
    med, p90, p99 = (float(x) for x in qs)
    print(f"rotation |w| per voxel:  median {med:.2e}  p90 {p90:.2e}  "
          f"p99 {p99:.2e}  max {float(w.max()):.2e} rad")
    print(f"  voxel size {2 * args.half_um / (args.grid - 1) * 1e3:.0f} nm; "
          f"median is the only resolution-stable one of these.")
    print("  CAVEAT: these are SLAB statistics. The slab is the right sampling "
          "for the DFXM image\n"
          "          (it is the beam-illuminated sheet) but the wrong sampling "
          "for characterising\n"
          "          the field: measured against a full 3-D cube of the same "
          "network, a 7-plane\n"
          "          slab reports 2.2 % of voxels above 1e-4 rad where the cube "
          "reports 13.9 %,\n"
          "          under-reporting the affected volume by ~6.4x. Do not read "
          "the numbers above\n"
          "          as a description of the microstructure.")

    # For b || n prismatic loops the rotation is essentially all in-plane, so a
    # single scalar compared against a single sigma is the wrong comparison --
    # DFXM sensitivity is anisotropic too (sigma_rock ~ 1e-3 vs sigma_roll ~ 9e-3),
    # and `aligned_resolution` below is transverse-isotropic, a known package
    # simplification. No detectability claim is made here; see
    # manuals/dfxm/LAB_NOTEBOOK.md ledger #4 (per-pixel orientation precision
    # 2 mdeg = 3.5e-5 rad, real data) if you need a floor, and note that single
    # dislocations are routinely imaged well below every rocking width
    # (Jakobsen 2019; Borgi 2024, 2025) because contrast comes from the near-core
    # tail, not the mean amplitude.
    sigma = args.sigma if args.sigma is not None else max(p90, 1e-9)
    print(f"resolution sigma set to the p90 of |w| ({sigma:.2e} rad) so the "
          f"contrast is visible; this is a demonstration of the chain, not an "
          f"instrument prediction.")

    centre = GoniometerSetting()
    q_nom = reference_q_nom(field, HKL, centre)
    res = aligned_resolution(q_nom, sigma_par=sigma, sigma_perp=sigma)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), WAVELENGTH)

    # DFXM is ANAMORPHIC, and strongly so at high energy. The objective looks
    # along k_out, only 2theta = {tt:.1f} deg off the beam, so an observation
    # plane containing the beam is imaged nearly edge-on and the along-beam
    # extent is compressed by sin(2theta). At Cu 220 / 71.7 keV that is a factor
    # ~7, which turns a square field of view into a stripe -- the shipped
    # tutorials show it too (122 x 18 px), so it is the geometry, not a bug.
    #
    # Real instruments recover it with a GRAZING detector, which stretches the
    # compressed axis by 1/cos(tilt). Choosing cos(tilt) = sin(2theta) undoes the
    # foreshortening exactly. This is the physical fix, not a cosmetic one.
    # tilt_axis="u" is the one that stretches the COMPRESSED axis: measured on an
    # 8 um sheet, (tilt=0) gives 48.0 x 6.4 px, ("u", 82.3 deg) gives 48.0 x 48.1,
    # and ("v", 82.3 deg) gives 358 x 6.4 -- i.e. "v" stretches the wrong one.
    grazing_deg = math.degrees(math.acos(
        max(min(math.sin(math.radians(tt)), 1.0 - 1e-9), 1e-9)))
    optics = ObjectiveOptics(two_theta_deg=tt, magnification=6.0, pixel_um=1.0,
                             detector_shape=(180, 180),
                             detector_tilt_deg=grazing_deg, tilt_axis="u")
    print(f"2theta = {tt:.2f} deg -> along-beam compression sin(2theta) = "
          f"{math.sin(math.radians(tt)):.3f}; grazing detector tilt "
          f"{grazing_deg:.1f} deg restores it")

    sd = math.degrees(sigma)
    chis = (0.0, 0.7 * sd, 1.5 * sd)
    imgs = {}
    for chi in chis:
        t0 = time.time()
        imgs[chi] = dfxm_image(field, HKL, GoniometerSetting(chi=chi),
                               res, optics).detach()
        print(f"  chi={chi:+.3f}  image in {time.time() - t0:.1f} s   "
              f"max={float(imgs[chi].max()):.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent; skipping the figure")
        return

    fig, axes = plt.subplots(1, len(chis) + 1, figsize=(4.1 * (len(chis) + 1), 4.0))
    for ax, chi in zip(axes[:-1], chis):
        im = ax.imshow(imgs[chi].numpy(), origin="lower", cmap="magma")
        ax.set_title(rf"DFXM  $\chi$ = {chi:+.2e}$^\circ$", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    beta_xy = (field.F - torch.eye(3, dtype=DT))[:, 0, 1].reshape(n, n, nz)
    rv = beta_xy[:, :, nz // 2].numpy()
    im = axes[-1].imshow(rv, origin="lower", cmap="RdBu_r",
                         extent=[-half, half, -half, half])
    axes[-1].set_title(r"elastic $\beta_{xy}$ (the field being imaged)", fontsize=10)
    axes[-1].set_xlabel("x (µm)"); axes[-1].set_ylabel("y (µm)")
    plt.colorbar(im, ax=axes[-1], fraction=0.046)

    fig.suptitle(f"ExaDiS $\\rightarrow$ midas-ddd $\\rightarrow$ DFXM   "
                 f"[{net.n_segments} segments, {summary['n_loops']} loops]   "
                 f"source: {source}", fontsize=11)
    out = os.path.join(args.out, "exadis_to_dfxm.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
