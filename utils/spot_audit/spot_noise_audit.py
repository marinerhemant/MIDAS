"""Are the 1891 unindexed spots real diffraction, or noise?

2 grains explain 185 of 2076 spots. Two explanations:
  (a) the peak list is contaminated -> most spots are noise, and the
      reconstruction is much closer to complete than 8.9% suggests;
  (b) the spots are real and we are failing to INDEX them -> ~91% of the
      diffraction is unexplained and the recon is badly incomplete.

The circular test is "do unindexed spots have intensity at their own pixel".
Of course they do -- a peak finder put them there. So this uses discriminators
that do NOT assume the answer:

  1. NImgs -- how many omega frames the spot spans. A real Bragg reflection
     sweeps through the Bragg condition over finite omega, so it is seen on
     >=2 consecutive 0.25 deg frames and merged. A hot pixel / cosmic ray /
     threshold fluctuation appears on exactly one.
  2. NrPx, SigmaR, SigmaEta -- real spots are compact 2-D blobs several px
     across; single-pixel events are detector artifacts.
  3. IntegratedIntensity / IMax -- absolute strength.
  4. maskTouched, FitRMSE, ReturnCode -- fit quality flags.
  5. FRIEDEL PAIRING -- the genuinely independent test. Every real reflection
     G has a partner -G at the diametrically opposite ring position, half a
     turn away in omega. The scan covers 360 deg, so both are observable.
     Noise does not pair up. The pairing RULE is calibrated on the indexed
     (known-real) spots rather than assumed, then applied blind.

ID-SPACE WARNING (this pipeline has bitten us here before -- the GrainRadius
bug): Radius_*.csv, InputAllExtraInfoFittingAll.csv and SpotMatrix.csv number
the same spots differently. Every join below is VERIFIED by cross-checking a
column that must agree, and the script aborts if it does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
L = R / "results/LayerNr_1"
OUT = R / "spot_noise_audit"


def load():
    ia = pd.read_csv(L / "InputAllExtraInfoFittingAll.csv", sep=r"\s+")
    rad = pd.read_csv(L / "Radius_StartNr_1_EndNr_1441.csv", sep=r"\s+")
    sm = pd.read_csv(L / "SpotMatrix.csv", sep="\t")
    sm.columns = [c.lstrip("%") for c in sm.columns]
    return ia, rad, sm


def verified_join(ia, rad):
    """The shared key is OrigSpotID (the pre-merge peakfit Result space).

    NOT SpotID: calc_radius and bin_data each renumber, so ia.SpotID and
    rad.SpotID are different orderings of the same 2076 spots. Joining on
    SpotID gives 0% intensity agreement -- which is exactly the class of bug
    that made every grain report the sample-mean GrainRadius.
    """
    m = ia.merge(rad, on="OrigSpotID", how="left", suffixes=("", "_rad"))
    assert len(m) == len(ia), f"join changed row count {len(ia)} -> {len(m)}"
    miss = int(m["NImgs"].isna().sum())
    a = m["IntegratedIntensity"].to_numpy()
    b = m["IntegratedIntensity_rad"].to_numpy()
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.abs(a - b) / np.maximum(scale, 1e-9)
    ok = rel < 1e-4
    print(f"  join on OrigSpotID: {len(m)} rows, {miss} unmatched, "
          f"{ok.mean():.1%} intensity-consistent")
    zero = np.abs(a) < 1e-9
    print(f"  rows with ia.IntegratedIntensity==0: {zero.sum()} "
          f"({zero.mean():.1%}); of the {(~ok).sum()} inconsistent, "
          f"{int((zero & ~ok).sum())} are these zero rows")
    if ok.mean() < 0.85:
        raise SystemExit("ABORT: join is wrong -- ID spaces do not correspond.")
    return m


def verify_spotmatrix_space(m, sm):
    """Confirm SpotMatrix.SpotID indexes into the ia.SpotID space."""
    idx = set(sm["SpotID"].astype(int))
    sub = m[m["SpotID"].astype(int).isin(idx)]
    print(f"  SpotMatrix has {len(idx)} unique SpotIDs; "
          f"{len(sub)} matched in InputAll SpotID space "
          f"(verified: ExtraInfo.bin shares this space)")
    if len(sub) != len(idx):
        raise SystemExit("ABORT: SpotMatrix IDs are in a different space.")
    # cross-check: omega and ring must agree for matched rows
    j = sm.drop_duplicates("SpotID").merge(
        m, on="SpotID", how="inner", suffixes=("_sm", ""))
    dome = np.abs(j["Omega_sm"] - j["Omega"]).max()
    dring = np.abs(j["RingNr"] - j["RingNumber"]).max()
    print(f"  cross-check on matched rows: max|dOmega|={dome:.4g} deg, "
          f"max|dRing|={dring:.0f}")
    if dome > 0.3 or dring > 0:
        raise SystemExit("ABORT: SpotMatrix join disagrees on omega/ring.")
    return idx


def dist(name, a, b):
    """Compare one column between indexed (a) and unindexed (b)."""
    q = [5, 25, 50, 75, 95]
    pa, pb = np.percentile(a, q), np.percentile(b, q)
    print(f"  {name:22s} indexed  " + " ".join(f"{v:9.3g}" for v in pa))
    print(f"  {'':22s} UNindexed" + " ".join(f"{v:9.3g}" for v in pb))


def friedel(m, indexed_ids):
    """Calibrate the pairing rule on known-real spots, then apply blind."""
    eta = m["Eta"].to_numpy()
    ome = m["Omega"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)
    sid = m["SpotID"].to_numpy().astype(int)
    is_idx = np.array([s in indexed_ids for s in sid])

    def wrap(x):
        return (x + 180.0) % 360.0 - 180.0

    rules = {
        "eta+180, ome+180": lambda e, o: (wrap(e + 180.0), wrap(o + 180.0)),
        "-eta,     ome+180": lambda e, o: (wrap(-e), wrap(o + 180.0)),
        "180-eta,  ome+180": lambda e, o: (wrap(180.0 - e), wrap(o + 180.0)),
    }
    ETA_TOL, OME_TOL = 1.0, 1.0

    def pair_frac(rule, mask):
        e_t, o_t = rule(eta, ome)
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

    print("\n  Friedel rule calibration (fraction paired):")
    best, best_f, best_hit = None, -1, None
    for nm, rule in rules.items():
        hit = pair_frac(rule, is_idx)
        f_idx = hit[is_idx].mean()
        print(f"    {nm:20s} indexed {f_idx:6.1%}   all {hit.mean():6.1%}")
        if f_idx > best_f:
            best, best_f, best_hit = nm, f_idx, hit
    print(f"    -> rule {best!r} pairs {best_f:.1%} of KNOWN-REAL spots")
    if best_f < 0.5:
        print("    WARNING: even known-real spots pair poorly; "
              "Friedel test is not usable here.")
        return None
    print(f"\n  Friedel pairing, blind:")
    print(f"    indexed   (known real): {best_hit[is_idx].mean():6.1%} "
          f"of {is_idx.sum()}")
    print(f"    UNindexed             : {best_hit[~is_idx].mean():6.1%} "
          f"of {(~is_idx).sum()}")
    return best_hit


def main():
    OUT.mkdir(exist_ok=True)
    ia, rad, sm = load()
    print(f"InputAll {len(ia)} spots | Radius {len(rad)} | "
          f"SpotMatrix {len(sm)} rows")
    print("\n--- joins (verified) ---")
    m = verified_join(ia, rad)
    indexed_ids = verify_spotmatrix_space(m, sm)

    is_idx = m["SpotID"].astype(int).isin(indexed_ids).to_numpy()
    A, B = m[is_idx], m[~is_idx]
    print(f"\nindexed {len(A)}  unindexed {len(B)}  "
          f"({len(A)/len(m):.1%} explained)")

    print("\n--- population comparison (percentiles 5/25/50/75/95) ---")
    for col in ("NImgs", "NrPx", "NrPxTot", "IntegratedIntensity", "IMax",
                "SigmaR", "SigmaEta", "FitRMSE", "GrainRadius"):
        if col in m.columns:
            dist(col, A[col].to_numpy(), B[col].to_numpy())

    print("\n--- single-frame fraction (the noise signature) ---")
    for nm, d in (("indexed", A), ("UNindexed", B)):
        n1 = (d["NImgs"] <= 1).mean()
        n2 = (d["NImgs"] >= 2).mean()
        print(f"  {nm:10s} NImgs==1: {n1:6.1%}   NImgs>=2: {n2:6.1%}")

    print("\n--- flags ---")
    for nm, d in (("indexed", A), ("UNindexed", B)):
        print(f"  {nm:10s} maskTouched: {(d['maskTouched']>0).mean():6.1%}   "
              f"ReturnCode!=-1: {(d['ReturnCode']!=-1).mean():6.1%}")

    print("\n--- ring occupancy ---")
    t = pd.crosstab(m["RingNumber"], is_idx)
    t.columns = ["unindexed", "indexed"] if list(t.columns) == [False, True] \
        else t.columns
    print(t.to_string())

    hit = friedel(m, indexed_ids)

    m["indexed"] = is_idx
    if hit is not None:
        m["friedel"] = hit
    m.to_csv(OUT / "spot_audit.csv", index=False)
    print(f"\nwrote {OUT/'spot_audit.csv'}")


if __name__ == "__main__":
    main()
