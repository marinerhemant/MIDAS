"""Tutorial: single-dislocation DFXM contrast, g.b invisibility, and typing.

Runs entirely synthetically (no external data). Produces a 4-panel figure:
  (a) edge-dislocation DFXM image
  (b) screw-dislocation DFXM image
  (c) g.b invisibility series (contrast vs reflection; extinction at g.b = 0)
  (d) edge vs screw dilatation-ratio (character discriminant)

Saves dislocation_typing.png under the directory chosen by ``figure_dir`` (the
dev/paper tree in a clone, ./figures otherwise; never /tmp). Run:

    export KMP_DUPLICATE_LIB_OK=TRUE            # macOS OpenMP workaround
    python -m midas_dfxm.examples.tutorial_dislocation_typing
"""
from __future__ import annotations

import os

import torch

from midas_dfxm import (
    GoniometerSetting,
    ObjectiveOptics,
    aligned_resolution,
    bragg_two_theta_deg,
    cubic_stiffness,
    classify_character,
    dfxm_image,
    dilatation_ratio,
    dislocation_deformation_field,
    g_dot_b,
    recover_burgers,
    reference_q_nom,
    stroh_dislocation,
    visibility_series,
)
from midas_dfxm.examples._figures import figure_dir, paper_style

DT = torch.float64
WAVELENGTH = 0.172979  # Angstrom
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)  # Cu (GPa), Zener ~3.2
PLANE, BURGERS = (1, 1, 1), (1, -1, 0)              # FCC {111}<110>


def _field_grid(n=81, half=10.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3), (n, n, 1)


def dislocation_image(character, hkl=(2, -2, 0)):
    pts, shape = _field_grid()
    disl = stroh_dislocation(CU, burgers=BURGERS, slip_normal=PLANE,
                             character=character, core_radius_um=0.3)
    field = dislocation_deformation_field(pts, disl, shape=shape)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    # Weak-beam: sit on the resolution flank so the strain field lights up.
    res = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=8e-3)
    setting = GoniometerSetting(chi=0.03)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), WAVELENGTH)
    optics = ObjectiveOptics(two_theta_deg=tt, magnification=6.0, pixel_um=1.0,
                             detector_shape=(160, 160))
    return dfxm_image(field, hkl, setting, res, optics).detach()


def main():
    outdir = figure_dir()

    # Panels (a),(b): edge and screw images.
    img_edge = dislocation_image("edge")
    img_screw = dislocation_image("screw")

    # Panel (c): g.b invisibility series for the screw.
    pts, _ = _field_grid(n=41, half=10.0)
    screw = stroh_dislocation(CU, burgers=BURGERS, slip_normal=PLANE,
                              character="screw", core_radius_um=0.3)
    refl = [(2, 2, 0), (0, 0, 2), (1, 1, 1),        # g.b = 0  invisible
            (2, -2, 0), (2, 0, 0), (0, 2, 0)]        # g.b != 0 visible
    vis = visibility_series(screw, pts, refl, mode="intrinsic")
    ranked = recover_burgers(vis, [(1, -1, 0), (1, 1, 0), (1, 0, -1), (0, 1, -1)])

    # Panel (d): character discriminant.
    edge = stroh_dislocation(CU, burgers=BURGERS, slip_normal=PLANE, character="edge")
    r_edge = float(dilatation_ratio(edge, pts))
    r_screw = float(dilatation_ratio(screw, pts))

    print("g.b invisibility series (intrinsic signal):")
    for r in refl:
        print(f"  {r}  g.b={int(g_dot_b(r, BURGERS)):+d}  signal={vis[r]:.3e}")
    print(f"recovered Burgers (best-first): {ranked}")
    print(f"dilatation ratio  edge={r_edge:.3f} ({classify_character(edge, pts)})  "
          f"screw={r_screw:.3e} ({classify_character(screw, pts)})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = paper_style()
        if style is not None:
            plt.style.use(style)

        fig, axg = plt.subplots(2, 2, figsize=(9.6, 8.8))
        ax = axg
        ax[0, 0].imshow(img_edge.T, origin="lower", cmap="magma")
        ax[0, 0].set_title("(a) edge dislocation — DFXM (2$\\bar2$0)")
        ax[0, 1].imshow(img_screw.T, origin="lower", cmap="magma")
        ax[0, 1].set_title("(b) screw dislocation — DFXM (2$\\bar2$0)")

        labels = [f"{r}\n g·b={int(g_dot_b(r, BURGERS)):+d}" for r in refl]
        vals = [vis[r] for r in refl]
        colors = ["tab:blue" if abs(g_dot_b(r, BURGERS)) < 1e-9 else "tab:red" for r in refl]
        ax[1, 0].bar(range(len(refl)), vals, color=colors)
        ax[1, 0].set_xticks(range(len(refl)))
        ax[1, 0].set_xticklabels(labels, fontsize=8)
        ax[1, 0].set_ylabel("intrinsic visibility  std(|ΔQ|)/|G|")
        ax[1, 0].set_title("(c) g·b invisibility (blue = g·b=0, invisible)")

        ax[1, 1].bar(["edge", "screw"], [r_edge, max(r_screw, 1e-6)],
                     color=["tab:orange", "tab:green"])
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_ylabel("dilatation / total distortion")
        ax[1, 1].set_title("(d) edge vs screw character")

        out = os.path.join(outdir, "dislocation_typing.png")
        fig.savefig(out)
        print(f"saved figure -> {os.path.normpath(out)}")
    except ImportError:
        print("matplotlib not available; skipped figure (numbers printed above)")


if __name__ == "__main__":
    main()
