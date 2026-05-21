"""Builds notebook 35: azimuthally-resolved (per-eta-bin) peak fitting."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
("md", """\
# 35 — Azimuthal Peak Fitting: One Fit per η-Bin (Full Ring)

**Time: ~15 minutes.** Pairs with NB 34 (multi-detector integration).

Collapsing a frame to a single 1-D lineout averages over the whole
azimuth and throws away the η-dependence that carries **strain** (ring
radius vs η — the cos 2η ellipse) and **texture** (intensity vs η).
This notebook keeps the second dimension: it integrates to the full
2-D cake `(n_eta, n_r)` and **auto-fits the diffraction peak in every
η-bin across the whole ring**, then plots the fitted peak position,
intensity and width as a function of azimuth.

Workflow:

1. `build_geometry(spec)` with a fine `EtaBinSize` → a cake with many
   η rows.
2. `integrate(image, geom)` → `(n_eta, n_r)` array.
3. For each η row with signal, fit a Gaussian + linear background to
   the target ring and record centre (→ 2θ), amplitude and FWHM.
4. Plot the three vs η.
"""),

("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from midas_integrate_v2 import spec_from_v1_paramstest, build_geometry, integrate

# One real HYDRA panel (ge1 is the brightest). Set $V2_HYDRA_BASE as in NB 34.
BASE = Path(os.environ.get("V2_HYDRA_BASE",
                           os.path.expanduser("~/Desktop/analysis/hydra")))
PANEL = 1
pdir = BASE / f"data/ge{PANEL}/CeO2_7s_2700mm_20x20um_slitted"
pparam = BASE / f"refined_MIDAS_params_ge{PANEL}_Tx_cake_MULTI.txt"
HAVE = pdir.exists() and pparam.exists()
if not HAVE:
    print(f"HYDRA panel/params not found under {BASE}. Run calibrate-v2 NB 23 "
          f"and set $V2_HYDRA_BASE. Skipping.")
else:
    fr = sorted(pdir.glob(f"CeO2_7s_2700mm_20x20um_slitted_0020*.ge{PANEL}.h5"))
    acc = None
    for fp in fr:
        with h5py.File(fp, "r") as f:
            a = np.asarray(f["exchange/data"][0], dtype=np.float64)
        acc = a if acc is None else acc + a
    with h5py.File(pdir / f"dark_before_002019.ge{PANEL}.h5", "r") as f:
        dk = np.asarray(f["exchange/data"][0], dtype=np.float64)
    img = np.clip(acc / len(fr) - dk, 0.0, None)
    print(f"panel ge{PANEL}: {img.shape}, max={img.max():.0f}")
"""),

("md", """\
## Build the cake and integrate

A fine `EtaBinSize` gives many azimuthal rows. The radial axis is in
pixels; convert to 2θ with the panel's `Lsd` and pixel pitch.
"""),

("code", """\
if HAVE:
    spec = spec_from_v1_paramstest(pparam, requires_grad=False)
    spec.RMin, spec.RMax = 400.0, 1300.0   # inner rings (111, 200, 220)
    spec.RBinSize = 1.0
    spec.EtaMin, spec.EtaMax = -180.0, 180.0
    spec.EtaBinSize = 2.0                  # 180 azimuthal bins
    geom = build_geometry(spec)
    # The cake is (n_eta, n_r) — azimuth along axis 0, radius along axis 1
    # (the package-wide convention). Index a row, cake[i, :], for the radial
    # profile at η-bin i.
    cake = integrate(torch.as_tensor(img), geom).cpu().numpy()
    n_eta, n_r = cake.shape
    print(f"cake shape: {cake.shape}  (η bins × R bins)")

    Lsd_um, px_um = float(spec.Lsd), float(spec.pxY)
    r_px = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    tth = np.degrees(np.arctan(r_px * px_um / Lsd_um))            # 2θ per R bin
    eta = spec.EtaMin + spec.EtaBinSize * (np.arange(n_eta) + 0.5)
    # CeO2 (111) target 2θ.
    lam = float(spec.Wavelength); d111 = 5.4116 / np.sqrt(3)
    tth_111 = np.degrees(2 * np.arcsin(lam / (2 * d111)))
    print(f"target ring CeO2(111) at 2θ = {tth_111:.3f}°")
"""),

("md", """\
## Fit the (111) ring in every η-bin

For each azimuthal row we take a narrow 2θ window around the (111)
ring and fit `Gaussian + linear background`. Rows where the panel has
no coverage (this is an off-panel arc, so only part of the ring is
seen) are skipped automatically when the peak amplitude is not
significant.
"""),

("code", """\
if HAVE:
    def gauss_lin(x, amp, cen, sig, b0, b1):
        return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + b0 + b1 * x

    win = np.abs(tth - tth_111) < 0.25          # ±0.25° fitting window
    xw = tth[win]
    cen_eta, amp_eta, fwhm_eta, ok_eta = [], [], [], []
    for i in range(n_eta):
        yw = cake[i, win]                       # R-window at azimuth i
        if yw.size == 0 or not np.isfinite(yw).all() or yw.max() <= 0:
            cen_eta.append(np.nan); amp_eta.append(np.nan)
            fwhm_eta.append(np.nan); ok_eta.append(False); continue
        a0 = yw.max() - np.median(yw)
        p0 = [max(a0, 1.0), tth_111, 0.05, np.median(yw), 0.0]
        try:
            popt, _ = curve_fit(gauss_lin, xw, yw, p0=p0, maxfev=5000)
            amp, cen, sig = popt[0], popt[1], abs(popt[2])
            good = (amp > 3 * np.std(yw - gauss_lin(xw, *popt))) and \\
                   (abs(cen - tth_111) < 0.2) and (0.005 < sig < 0.2)
            cen_eta.append(cen if good else np.nan)
            amp_eta.append(amp if good else np.nan)
            fwhm_eta.append(2.3548 * sig if good else np.nan)
            ok_eta.append(bool(good))
        except Exception:
            cen_eta.append(np.nan); amp_eta.append(np.nan)
            fwhm_eta.append(np.nan); ok_eta.append(False)
    cen_eta = np.array(cen_eta); amp_eta = np.array(amp_eta)
    fwhm_eta = np.array(fwhm_eta); ok_eta = np.array(ok_eta)
    print(f"fitted {ok_eta.sum()} / {n_eta} η-bins "
          f"(rest have no panel coverage at this ring)")
"""),

("code", """\
if HAVE:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (a) the cake itself
    ax = axes[0, 0]
    im = ax.imshow(cake, aspect="auto", origin="lower", cmap="viridis",
                   extent=[tth[0], tth[-1], eta[0], eta[-1]],
                   vmax=np.nanpercentile(cake[cake > 0], 99))
    ax.axvline(tth_111, color="w", lw=0.6, ls="--")
    ax.set_xlabel("2θ (deg)"); ax.set_ylabel("η (deg)")
    ax.set_title("(a) 2-D cake (η × 2θ)"); fig.colorbar(im, ax=ax)
    # (b) fitted peak 2θ vs η  (strain signature)
    ax = axes[0, 1]
    ax.plot(eta[ok_eta], cen_eta[ok_eta], ".", ms=4, color="tab:red")
    ax.axhline(tth_111, color="0.5", lw=0.8, ls="--", label="nominal (111)")
    ax.set_xlabel("η (deg)"); ax.set_ylabel("fitted (111) 2θ (deg)")
    ax.set_title("(b) peak position vs η  →  strain"); ax.legend(fontsize=8)
    # (c) intensity vs η  (texture)
    ax = axes[1, 0]
    ax.plot(eta[ok_eta], amp_eta[ok_eta], ".", ms=4, color="tab:blue")
    ax.set_xlabel("η (deg)"); ax.set_ylabel("fitted amplitude (a.u.)")
    ax.set_title("(c) intensity vs η  →  texture / coverage")
    # (d) FWHM vs η
    ax = axes[1, 1]
    ax.plot(eta[ok_eta], fwhm_eta[ok_eta], ".", ms=4, color="tab:green")
    ax.set_xlabel("η (deg)"); ax.set_ylabel("FWHM (deg 2θ)")
    ax.set_title("(d) peak width vs η")
    plt.tight_layout(); plt.show()
"""),

("md", """\
## Reading the result

- **(b) peak 2θ vs η** is the raw material for **strain**: a uniform
  strain shifts the ring radius as `cos 2(η − η₀)`, so fitting that
  sinusoid to the per-η centres gives the in-plane strain tensor. A
  flat line (as for an unstrained calibrant) confirms the geometry is
  right and there is no residual ellipticity.
- **(c) amplitude vs η** carries **texture** (and, for a single
  off-panel HYDRA panel, the azimuthal *coverage* — only the η arc the
  panel subtends is populated; combine panels as in NB 34 for full η).
- **(d) FWHM vs η** flags azimuthal broadening (microstrain, grain
  size, or geometric defocus).

This per-η fit is the standard front end for strain/texture analysis
and for QC of a multi-panel calibration: run it on each panel, or on
the combined cake, and check that the (111) centre is flat in η.
"""),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("35_azimuthal_peak_fitting_per_eta.ipynb")
    write_notebook(out, CELLS)
    print(f"wrote {out}")
