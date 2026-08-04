"""Null model for the Friedel test + a bias-free completeness estimate.

The first audit found unindexed spots are overwhelmingly weak, single-frame
and few-pixel -- but ALSO that 77.8% of them have a "Friedel partner". Those
point opposite ways, and the second is the one that needs a null: with +-1 deg
tolerance on a ring carrying 637 spots, pairing by CHANCE is likely.

NULL: destroy the physical pairing while preserving every marginal
distribution, by shuffling Eta within each ring (and, separately, Omega).
If the shuffled data pairs at the same rate, the Friedel test carries no
information at this tolerance and must not be used as evidence either way.

Then: estimate completeness over spots that are PLAUSIBLY REAL on
physically-motivated grounds (ring assigned, non-zero intensity, seen on >=2
omega frames) rather than on a cut derived from the indexed set, which would
be circular.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
AUD = R / "spot_noise_audit/spot_audit.csv"
RNG = np.random.default_rng(12345)
ETA_TOL = OME_TOL = 1.0


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def pair_hits(eta, ome, ring):
    """Friedel rule calibrated earlier: eta -> 180-eta, omega -> omega+180."""
    e_t, o_t = wrap(180.0 - eta), wrap(ome + 180.0)
    hit = np.zeros(len(eta), bool)
    for r in np.unique(ring):
        sel = np.where(ring == r)[0]
        if len(sel) < 2:
            continue
        de = np.abs(wrap(e_t[sel][:, None] - eta[sel][None, :]))
        do = np.abs(wrap(o_t[sel][:, None] - ome[sel][None, :]))
        ok = (de < ETA_TOL) & (do < OME_TOL)
        np.fill_diagonal(ok, False)
        hit[sel] = ok.any(axis=1)
    return hit


def main():
    m = pd.read_csv(AUD)
    eta = m["Eta"].to_numpy()
    ome = m["Omega"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)
    idx = m["indexed"].to_numpy().astype(bool)

    obs = pair_hits(eta, ome, ring)
    print("OBSERVED Friedel pairing")
    print(f"  indexed   {obs[idx].mean():6.1%}   unindexed {obs[~idx].mean():6.1%}")

    print("\nNULL (shuffle within ring, 20 draws) -- destroys pairing, "
          "keeps marginals")
    for what in ("eta", "omega"):
        rates_i, rates_u = [], []
        for _ in range(20):
            e2, o2 = eta.copy(), ome.copy()
            for r in np.unique(ring):
                sel = np.where(ring == r)[0]
                if what == "eta":
                    e2[sel] = RNG.permutation(e2[sel])
                else:
                    o2[sel] = RNG.permutation(o2[sel])
            h = pair_hits(e2, o2, ring)
            rates_i.append(h[idx].mean())
            rates_u.append(h[~idx].mean())
        print(f"  shuffle {what:5s}: indexed {np.mean(rates_i):6.1%}"
              f" +-{np.std(rates_i):.1%}   "
              f"unindexed {np.mean(rates_u):6.1%} +-{np.std(rates_u):.1%}")

    # ---- how separable are the two populations, really? -------------------
    print("\n--- overlap of the two populations ---")
    A, B = m[idx], m[~idx]
    for col in ("IMax", "IntegratedIntensity", "NImgs", "NrPx"):
        lo = np.percentile(A[col], 5)
        n = int((B[col] >= lo).sum())
        print(f"  unindexed with {col:20s} >= indexed 5th pct ({lo:10.4g}): "
              f"{n:5d} / {len(B)}  ({n/len(B):5.1%})")

    # ---- bias-free plausibility cut ---------------------------------------
    print("\n--- completeness over PLAUSIBLY REAL spots ---")
    print("    (physically motivated, not derived from the indexed set)")
    ring_ok = m["RingNumber"] > 0
    inten_ok = m["IntegratedIntensity"] > 0
    multi = m["NImgs"] >= 2
    for name, sel in (
        ("all spots", np.ones(len(m), bool)),
        ("ring assigned", ring_ok.to_numpy()),
        ("ring + nonzero intensity", (ring_ok & inten_ok).to_numpy()),
        ("ring + nonzero + NImgs>=2", (ring_ok & inten_ok & multi).to_numpy()),
    ):
        n, ni = int(sel.sum()), int((sel & idx).sum())
        print(f"  {name:28s} n={n:5d}   indexed {ni:4d}  "
              f"completeness {ni/max(n,1):6.1%}")

    # among the plausible ones, are the unindexed still weak?
    sel = (ring_ok & inten_ok & multi).to_numpy()
    P = m[sel]
    pa, pb = P[P["indexed"]], P[~P["indexed"]]
    print(f"\n  within 'ring+nonzero+NImgs>=2' ({len(P)} spots):")
    for col in ("IMax", "IntegratedIntensity", "NImgs", "NrPx"):
        print(f"    {col:20s} indexed median {np.median(pa[col]):10.4g}   "
              f"unindexed median {np.median(pb[col]):10.4g}")

    out = R / "spot_noise_audit/plausible_unindexed.csv"
    pb.sort_values("IMax", ascending=False).to_csv(out, index=False)
    print(f"\n  wrote {out}  ({len(pb)} rows)")


if __name__ == "__main__":
    main()
