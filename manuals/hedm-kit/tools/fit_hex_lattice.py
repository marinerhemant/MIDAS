#!/usr/bin/env python3
"""Fit a hexagonal/rhombohedral (hex-axes) lattice a, c to powder ring d-spacings from ring_survey.py's JSON.

Usage:  python fit_hex_lattice.py <rings.json> <SG> <a0> <c0> [tol=0.004]

Each observed ring is assigned to the nearest allowed (h k l) line of the nominal cell (within tol of d); then
1/d^2 = 4/3 (h^2+hk+k^2)/a^2 + l^2/c^2 is solved by least squares. Rings that match several hkl of the nominal
cell within tol are dropped (ambiguous). Reports the per-ring residual in ppm of d and the number of rings used.
CONDITIONAL on the Lsd and lambda used to get d from R (they are tied: Lsd*lambda is what a powder pattern fixes).
"""
import json
import sys

import numpy as np
import midas_hkls as mh


def main():
    fn, sg, a0, c0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.004
    J = json.load(open(fn))
    rings = J["rings"]
    d = np.array([r["d_A"] for r in rings])
    refl = mh.generate_hkls(mh.SpaceGroup.from_number(sg), mh.Lattice(a0, a0, c0, 90, 90, 120), d_min=d.min() * 0.97, d_max=d.max() * 1.03)
    lines = [(r.d_spacing, (r.h, r.k, r.l)) for r in refl]
    rows, hk = [], []
    for r, di in zip(rings, d):
        near = [(dd, h) for dd, h in lines if abs(dd - di) / di < tol]
        ds = sorted({round(dd, 4) for dd, _ in near})
        if len(ds) != 1:
            print(f"  skip ring R={r['R_px']:.2f} d={di:.4f}: {'no line' if not ds else 'ambiguous ' + str(ds)}")
            continue
        h, k, l = near[0][1]
        rows.append([4.0 / 3.0 * (h * h + h * k + k * k), l * l])
        hk.append((r, di, (h, k, l)))
    A = np.array(rows)
    y = np.array([1.0 / x[1] ** 2 for x in hk])
    # weight by d^4 so that equal fractional errors in d count equally
    wgt = np.array([x[1] ** 4 for x in hk])
    sol, *_ = np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)
    a, c = 1 / np.sqrt(sol[0]), 1 / np.sqrt(sol[1])
    res = []
    for (r, di, hkl), row in zip(hk, A):
        dc = 1 / np.sqrt(row @ sol)
        res.append((di - dc) / dc * 1e6)
        print(f"  {str(hkl):14s} R={r['R_px']:8.2f} d_obs={di:.5f} d_calc={dc:.5f} resid={res[-1]:8.0f} ppm")
    print(f"a = {a:.5f} A   c = {c:.5f} A   (n rings {len(hk)}, rms resid {np.sqrt(np.mean(np.square(res))):.0f} ppm of d)")
    json.dump({"a": a, "c": c, "n_rings": len(hk), "rms_ppm": float(np.sqrt(np.mean(np.square(res)))), "Lsd_mm": J["lsd_mm"], "lambda_A": J["lambda_A"]},
              open(fn.replace("_rings.json", "_lattice.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
