#!/usr/bin/env python3
"""Per-ring structure factor of a phase, from a CIF: which rings are intrinsically weak?

Usage:  ring_f2.py <phase.cif> --lsd <mm> --lambda <A> [--a <A>] [--px 150] [--rmin 150] [--rmax 1440]

Prints, for every ring in the detector window, d, radius, multiplicity, |F|^2 per spot relative to the strongest ring (no Lorentz term:
for the rotation method the Lorentz factor cancels in per-frame peak height, measured on NF data, see midas_hkls.f2_normalised), and the
spread of |F|^2 inside the ring (a ring that mixes two hkl families has a spread > 0). --a overrides the cubic cell edge with the
calibrated value, as the pipeline does (the CIF supplies only the atom basis). In a run, declare the same CIF as `PhaseCIF <path>` in
Parameters.txt: hkls.csv then carries an F2 column and midas-pipeline's binning stage reports a weak seed ring (seed_ring.json).
A CIF with charged labels (P5+, S2-) falls back to NEUTRAL form factors with a warning: fine for weak-versus-strong, not for absolute F.
Licence: ICSD CIFs are licensed. Do not copy one into a shared folder; record the entry number and where to get it.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main():
    a = sys.argv[1:]
    if len(a) < 5 or a[0] in ("-h", "--help"):
        sys.exit(__doc__)
    cif = a[0]
    opt = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
    lsd, lam = float(opt["--lsd"]), float(opt["--lambda"])
    px, rmin, rmax = float(opt.get("--px", 150)), float(opt.get("--rmin", 150)), float(opt.get("--rmax", 1440))
    import midas_hkls as mh
    from midas_hkls.io.cif import read_cif
    from midas_hkls.structure_factor import structure_factors
    cry = read_cif(cif)
    L = cry.lattice
    lat = mh.Lattice(float(opt["--a"]), float(opt["--a"]), float(opt["--a"]), 90, 90, 90) if "--a" in opt else L
    dmin = lam / (2 * np.sin(np.arctan(rmax * px / (lsd * 1000)) / 2))
    dmax = lam / (2 * np.sin(np.arctan(rmin * px / (lsd * 1000)) / 2))
    refs = mh.generate_hkls(cry.space_group, lat, d_min=dmin * 0.99)            # no d_max: ring numbers must start at the largest d, as in hkls.csv
    ct = cry.to_torch()
    hk = [(r.h, r.k, r.l) for r in refs]
    F2 = np.abs(structure_factors(ct, hk).detach().numpy()) ** 2
    d = np.array([r.d_spacing for r in refs])
    mult = np.array([r.multiplicity for r in refs])
    order = np.argsort(-d)
    rings, anchor, nr = {}, None, 0
    for i in order:                                  # the pipeline's grouping: |d - anchor| < 1e-4 A (GetHKLList Epsilon)
        if anchor is None or abs(d[i] - anchor) >= 1e-4:
            nr += 1
            anchor = d[i]
        rings.setdefault(nr, []).append(i)
    inwin = {k: dmin <= d[ix[0]] <= dmax for k, ix in rings.items()}
    mx = max(F2[ix].mean() for k, ix in rings.items() if inwin[k])
    print(f"space group {cry.space_group.number}, a = {lat.a:.4f}; window d {dmin:.3f}..{dmax:.3f} A, R {rmin:.0f}..{rmax:.0f} px")
    print("ring  d(A)   R(px)  hkl           mult  |F|^2 per spot (strongest in window = 100)  within-ring spread")
    for k, ix in rings.items():
        if not inwin[k]:
            continue
        dd = d[ix[0]]
        R = lsd * 1000 / px * np.tan(2 * np.arcsin(lam / (2 * dd)))
        v = F2[ix]
        print(f"{k:3d} {dd:7.3f} {R:7.1f}  {str(hk[ix[0]]):12s} {int(mult[ix].sum()):5d}   {100 * v.mean() / mx:10.2f}                          {v.std() / max(v.mean(), 1e-12):.3f}")
    print("ring numbers follow hkls.csv (largest d = ring 1); still read the run's own hkls.csv before setting RingThresh / OverAllRingToIndex")


if __name__ == "__main__":
    main()
