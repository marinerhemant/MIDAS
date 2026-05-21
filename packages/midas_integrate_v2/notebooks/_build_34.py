"""Builds notebook 34: HYDRA multi-detector integration on real 4-panel data."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
("md", """\
# 34 — Multi-Detector Integration on Real HYDRA Data

**Time: ~10 minutes.** Pairs with `midas_calibrate_v2` notebook 23
(joint multi-detector refinement).

High-energy beamlines (e.g. APS 1-ID) tile several flat panels
*azimuthally* around the beam — the "HYDRA" arrangement — so that four
detectors at one large sample-to-detector distance together cover a
full powder ring. This notebook integrates a real 4-panel CeO₂ dataset
onto a single shared 2θ axis with `MILKMultiGeometryAdapter`, which
accumulates every panel's per-pixel contributions into common bins
(area-weighted) rather than averaging pre-binned 1-D lineouts.

Each panel carries its own beam-centre, tilts, distortion and an
azimuthal mounting angle `tx`; the merge absorbs the mount offset
through `tx` exactly, so no after-the-fact lineout stitching or
inter-panel scaling is needed. The geometry comes from the
multi-detector calibration (calibrate-v2 NB 23): a single shared
Lsd ≈ 2455 mm with per-panel beam centres.
"""),

("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, h5py
import matplotlib.pyplot as plt

# Real 4-panel HYDRA CeO2 dataset. Set $V2_HYDRA_BASE to a directory with
#   data/ge{1..4}/CeO2_7s.../CeO2_*.geN.h5 (+ dark_before_*.geN.h5)
#   refined_MIDAS_params_geN_Tx_cake_MULTI.txt   (corrected joint calibration)
BASE = Path(os.environ.get("V2_HYDRA_BASE",
                           os.path.expanduser("~/Desktop/analysis/hydra")))
PANELS = (1, 2, 3, 4)

def panel_dir(n):
    return BASE / f"data/ge{n}/CeO2_7s_2700mm_20x20um_slitted"
def param_path(n):
    return BASE / f"refined_MIDAS_params_ge{n}_Tx_cake_MULTI.txt"

HAVE = BASE.exists() and all(panel_dir(n).exists() and param_path(n).exists()
                             for n in PANELS)
if not HAVE:
    print(f"HYDRA dataset / corrected params not found under {BASE}.")
    print("Run midas_calibrate_v2 NB 23 first to produce the joint calibration,")
    print("then set $V2_HYDRA_BASE. Skipping execution below.")
else:
    def summed(n):
        D = panel_dir(n)
        fr = sorted(D.glob(f"CeO2_7s_2700mm_20x20um_slitted_0020*.ge{n}.h5"))
        acc = None
        for fp in fr:
            with h5py.File(fp, "r") as f:
                a = np.asarray(f["exchange/data"][0], dtype=np.float64)
            acc = a if acc is None else acc + a
        with h5py.File(D / f"dark_before_002019.ge{n}.h5", "r") as f:
            dk = np.asarray(f["exchange/data"][0], dtype=np.float64)
        return np.clip(acc / len(fr) - dk, 0.0, None)
    images = [summed(n) for n in PANELS]
    print(f"loaded {len(PANELS)} dark-subtracted panels, "
          f"each {images[0].shape}")
"""),

("md", """\
## Combine the four panels onto one 2θ axis

`MILKMultiGeometryAdapter` takes a list of per-panel
`IntegrationSpec`s (built from each panel's calibration paramstest)
and integrates a list of panel images onto a shared axis. With
`method="polygon"` it uses the exact fractional-pixel-area weights,
and the per-panel variance is propagated to the merged bins.
"""),

("code", """\
if HAVE:
    from midas_integrate_v2 import spec_from_v1_paramstest
    from midas_integrate_v2.io import MILKMultiGeometryAdapter

    specs = [spec_from_v1_paramstest(param_path(n), requires_grad=False)
             for n in PANELS]
    for n, s in zip(PANELS, specs):
        print(f"ge{n}: Lsd={float(s.Lsd)/1000:.2f}mm "
              f"BC=({float(s.BC_y):.1f},{float(s.BC_z):.1f}) tx={float(s.tx):.1f}")

    adapter = MILKMultiGeometryAdapter(specs, unit="2th_deg")
    combined = adapter.integrate1d(images, npt=2048, method="polygon")
    print(f"\\ncombined 2θ range: {combined.radial[0]:.3f} → "
          f"{combined.radial[-1]:.3f}°  ({len(combined.radial)} bins)")
"""),

("code", """\
if HAVE:
    # Per-panel single-detector integration, for the overlay.
    per_panel = []
    for k, s in enumerate(specs):
        single = MILKMultiGeometryAdapter([s], unit="2th_deg")
        per_panel.append(single.integrate1d([images[k]], npt=2048,
                                            method="polygon"))

    # CeO2 ring positions for reference.
    from midas_hkls import SpaceGroup, Lattice, generate_hkls
    refs = generate_hkls(SpaceGroup.from_number(225),
                         Lattice(a=5.4116, b=5.4116, c=5.4116,
                                 alpha=90, beta=90, gamma=90),
                         wavelength_A=float(specs[0].Wavelength),
                         two_theta_max_deg=8.0)
    rings = sorted({round(r.two_theta_deg, 3): f"{r.h}{r.k}{r.l}"
                    for r in refs}.items())

    fig, (axc, axp) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for tt0, hkl in rings:
        for ax in (axc, axp):
            ax.axvline(tt0, color="0.8", lw=0.5, ls="--", zorder=0)
        axc.text(tt0, axc.get_ylim()[1] if False else 1.0, hkl, fontsize=7,
                 rotation=90, ha="center", va="bottom", color="0.4")
    c = combined
    m = np.isfinite(c.intensity)
    axc.plot(c.radial[m], c.intensity[m], "k-", lw=0.9, label="combined (4 panels)")
    axc.set_ylabel("Intensity (a.u.)"); axc.legend(loc="upper right")
    axc.set_title("HYDRA: 4 azimuthal panels merged onto one 2θ axis")
    for n, pp, col in zip(PANELS, per_panel,
                          ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
        mm = np.isfinite(pp.intensity) & (pp.intensity > 0)
        axp.plot(pp.radial[mm], pp.intensity[mm], "-", lw=0.8, color=col,
                 label=f"ge{n}")
    axp.set_xlabel("2θ (deg)"); axp.set_ylabel("Intensity (a.u.)")
    axp.set_title("Per-panel profiles (rings coincide despite different mounts)")
    axp.legend(ncol=4, fontsize=8); axp.set_xlim(2.5, 7.5)
    plt.tight_layout(); plt.show()
"""),

("md", """\
## What to look for

- The **combined** profile is a clean CeO₂ powder pattern — the
  (111), (200), (220), (311), (222), (400) reflections all land on
  their nominal 2θ (dashed lines).
- The **per-panel** profiles overlay: each of the four panels, despite
  its different azimuthal mount, places its rings at the *same* 2θ.
  That coincidence is the proof the joint calibration (shared Lsd +
  per-panel beam-centre + `tx`) is correct.
- The merged σ (in `combined.sigma`) is propagated per-pixel through
  the correction chain, not a `sqrt(I)` afterthought — write it out
  for Rietveld/PDF with `write_dat` / `write_esg`.

Next: **NB 35** fits the rings *per azimuthal bin* (η) instead of
collapsing to a single lineout — the basis for strain/texture maps.
"""),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("34_hydra_multi_detector_real_data.ipynb")
    write_notebook(out, CELLS)
    print(f"wrote {out}")
