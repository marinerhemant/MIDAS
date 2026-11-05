#!/usr/bin/env python3
"""Which rings does the sample show, and which candidate phases have lines there?  Descriptive, frames only.

Usage:  python ring_survey.py <bgdark.h5 | image.h5:dataset> <out_prefix> <lsd_mm> <wavelength_A> <px_um> \
            <name:SG:a,b,c,alpha,beta,gamma> [more candidates ...]  [--bc row,col]

* The image is the temporal-median pseudo-dark P (exchange/dark[0]) built by ff/make_bg_dark.py: the static
  powder-like component of the sweep, which is where rings are best measured.
* The beam centre is found from the data by maximising the sharpness of the brightest ring over a small grid
  (both axis orderings are tried; the pair with the sharper ring wins). Distortion and tilts are NOT applied, so
  d-spacings are good to ~0.1-0.3 %: this is a SURVEY, not a lattice fit.
* For every candidate the report gives, next to the matches, the candidate's LINE COUNT in the observed d window and the
  chance rate: the fraction of random d values in that window that land within the tolerance of a line.
  A match count means nothing without them (known-limits KL-001). Lattice values passed in are NOMINAL and are
  conditional on the ASSERTED wavelength (20-ID files do not record the energy).
"""
import sys

import h5py
import numpy as np
from scipy.signal import find_peaks

TOL = 0.004          # 0.4 % in d


def radial(img, bc, rmax=1500, mask_r=None):
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - bc[0], xx - bc[1])
    rb = np.minimum(r.astype(int), rmax)
    s = np.bincount(rb.ravel(), weights=img.ravel(), minlength=rmax + 1)
    n = np.bincount(rb.ravel(), minlength=rmax + 1)
    return s / np.maximum(n, 1)


def find_bc(img, seed, ring_r):
    """Beam centre from the data: maximise the contrast of the brightest ring (radius ~ring_r px) over a wide grid
    around `seed` (row, col), coarse on a 4x-subsampled image then refined at full resolution. Both axis orderings
    of the seed are tried; the sharper ring wins. Returns (bc, order)."""
    sub = img[::4, ::4]
    yy, xx = np.indices(sub.shape)
    yy, xx = yy * 4.0, xx * 4.0

    def score(bc, im_y, im_x, im, lo, hi, nb=60):
        r = np.hypot(im_y - bc[0], im_x - bc[1])
        m = (r > lo) & (r < hi)
        h, _ = np.histogram(r[m], bins=nb, range=(lo, hi), weights=im[m])
        c, _ = np.histogram(r[m], bins=nb, range=(lo, hi))
        p = h / np.maximum(c, 1)
        return p.max() - np.median(p)

    best = None
    for order in (0, 1):
        s0 = (seed[0], seed[1]) if order == 0 else (seed[1], seed[0])
        for dy in np.arange(-90, 90.1, 6.0):
            for dx in np.arange(-90, 90.1, 6.0):
                bc = (s0[0] + dy, s0[1] + dx)
                sc = score(bc, yy, xx, sub, ring_r - 60, ring_r + 60)
                if best is None or sc > best[0]:
                    best = (sc, bc, order)
    sc0, bc, order = best
    fy, fx = np.indices(img.shape)
    for step, span in ((2.0, 8.0), (0.5, 2.0)):
        b0 = bc
        for dy in np.arange(-span, span + 1e-9, step):
            for dx in np.arange(-span, span + 1e-9, step):
                b2 = (b0[0] + dy, b0[1] + dx)
                # restrict to a thin window of pixels for speed
                r0 = np.hypot(fy[::2, ::2] - b2[0], fx[::2, ::2] - b2[1])
                m = (r0 > ring_r - 25) & (r0 < ring_r + 25)
                h, _ = np.histogram(r0[m], bins=100, range=(ring_r - 25, ring_r + 25), weights=img[::2, ::2][m])
                c, _ = np.histogram(r0[m], bins=100, range=(ring_r - 25, ring_r + 25))
                p = h / np.maximum(c, 1)
                sc = p.max() - np.median(p)
                if sc > best[0] or (step == 0.5 and False):
                    best = (sc, b2, order)
                    bc = b2
    return best[1], best[2]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bcarg = None
    if "--bc" in sys.argv:
        i = sys.argv.index("--bc")
        bcarg = tuple(float(x) for x in sys.argv[i + 1].split(","))
        args = [a for a in args if a != sys.argv[i + 1]]
    src, pre, lsd, lam, px = args[:5]
    lsd, lam, px = float(lsd), float(lam), float(px)
    cands = args[5:]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import midas_hkls as mh

    if ":" in src and not src.endswith(".h5"):
        fn, ds = src.rsplit(":", 1)
        with h5py.File(fn, "r") as f:
            img = np.asarray(f[ds][0] if f[ds].ndim == 3 else f[ds][()], dtype=np.float64)
    else:
        with h5py.File(src, "r") as f:
            img = np.asarray(f["exchange/dark"][0], dtype=np.float64)
    seed = (1440.0, 1440.0)
    prof0 = radial(img, seed, 1500)
    ring0 = int(np.argmax(prof0[200:600])) + 200
    order = None
    if bcarg:
        bc = bcarg
    else:
        bc, order = find_bc(img, seed, ring0)
    prof = radial(img, bc, 1500)
    base = np.convolve(prof, np.ones(61) / 61, mode="same")
    pk, props = find_peaks(prof[150:1450] - base[150:1450], prominence=max(20, 0.02 * np.ptp(prof[150:1450])))
    pk = pk + 150
    tth = np.arctan(pk * px / (lsd * 1000.0))
    d_obs = lam / (2 * np.sin(tth / 2))
    print(f"BC (row,col) = ({bc[0]:.2f}, {bc[1]:.2f}) [axis order {order}]; {len(pk)} rings; Lsd {lsd} mm, lambda {lam} A, px {px} um")
    dmin, dmax = d_obs.min() * (1 - TOL), d_obs.max() * (1 + TOL)
    out = [("R_px", "d_A", "peak_height")]
    rows = []
    cand_lines = {}
    for c in cands:
        name, sg, lat = c.split(":")
        a, b, cc, al, be, ga = [float(x) for x in lat.split(",")]
        cr = mh.Crystal if hasattr(mh, "Crystal") else None
        refl = mh.generate_hkls(mh.SpaceGroup.from_number(int(sg)), mh.Lattice(a, b, cc, al, be, ga), d_min=dmin * 0.98, d_max=dmax * 1.02)
        cand_lines[name] = sorted({(round(r.d_spacing, 4), (r.h, r.k, r.l)) for r in refl})
    print(f"{'R_px':>7} {'d_A':>8} {'height':>8}  matches (within {100*TOL:.1f} % of d)")
    for r_, d_ in zip(pk, d_obs):
        m = []
        for name, lines in cand_lines.items():
            hit = [h for dd, h in lines if abs(dd - d_) / d_ < TOL]
            if hit:
                m.append(f"{name}{hit[0]}" + (f"+{len(hit)-1}" if len(hit) > 1 else ""))
        print(f"{r_:7d} {d_:8.4f} {prof[r_] - base[r_]:8.1f}  {' '.join(m)}")
    print("\ncandidate | distinct d lines in observed window | chance rate (random d within tol of a line) | observed rings matched")
    rng = np.random.default_rng(0)
    for name, lines in cand_lines.items():
        ds_ = np.array(sorted({dd for dd, _ in lines}))
        rand = rng.uniform(dmin, dmax, 20000)
        near = np.abs(rand[:, None] - ds_[None, :]).min(1) / rand < TOL if len(ds_) else np.zeros(len(rand), bool)
        matched = sum(1 for d_ in d_obs if len(ds_) and (np.abs(ds_ - d_) / d_).min() < TOL)
        print(f"{name:10s} | {len(ds_):4d} | {near.mean():.2f} | {matched}/{len(d_obs)}")
    # sub-pixel ring centres: baseline-subtracted centroid within +-5 px of each peak
    import json
    rings = []
    for r_, d_ in zip(pk, d_obs):
        sl = slice(r_ - 5, r_ + 6)
        w = np.clip(prof[sl] - base[sl], 0, None)
        rc = float((np.arange(r_ - 5, r_ + 6) * w).sum() / w.sum()) + 0.5 if w.sum() > 0 else float(r_)
        t2 = np.arctan(rc * px / (lsd * 1000.0))
        rings.append({"R_px": rc, "d_A": float(lam / (2 * np.sin(t2 / 2))), "height": float(prof[r_] - base[r_])})
    json.dump({"bc_row_col": [float(bc[0]), float(bc[1])], "lsd_mm": lsd, "lambda_A": lam, "px_um": px, "rings": rings},
              open(pre + "_rings.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(np.arange(len(prof)), prof, lw=0.8)
    ax.set_yscale("log")
    for k, (name, lines) in enumerate(cand_lines.items()):
        for dd, h in lines:
            R = lsd * 1000 / px * np.tan(2 * np.arcsin(lam / (2 * dd)))
            if 150 < R < 1450:
                ax.axvline(R, ymin=0.0 + 0.04 * k, ymax=0.04 * k + 0.035, color=f"C{k+1}", lw=1)
        ax.plot([], [], color=f"C{k+1}", label=name)
    ax.legend()
    ax.set_xlabel("radius (px)")
    ax.set_title(f"{src.split('/')[-1]}: radial profile of the static median image, candidate lines as ticks (nominal lattices, asserted lambda)")
    fig.tight_layout()
    fig.savefig(pre + "_rings.png", dpi=70)


if __name__ == "__main__":
    main()
