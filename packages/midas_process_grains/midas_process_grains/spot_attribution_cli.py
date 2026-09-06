"""``midas-spot-attribution`` — per-claim soft attribution from a SpotMatrix.

Pass 1 of the pipeline gives ``Grains.csv`` + ``SpotMatrix.csv``. This decides,
for each ``(grain, spot)`` claim, how consistent that spot is with that grain,
and emits either the weights or a filtered spot set for a **second refinement
pass**.

Typical use::

    # look first: what would be dropped, and does it match the twin structure?
    midas-spot-attribution SpotMatrix.csv --grains Grains.csv --twin-check

    # then emit the filtered spot sets for pass 2
    midas-spot-attribution SpotMatrix.csv --emit-spot-sets pass2_spots.csv

``--twin-check`` is the honest self-test: it computes Σ3 twin pairs and reports
the keep rate for twin-shared vs accidental claims *separately*. The labels are
used only to score — never as input to the filter. On 1-ID LSHR layer 6 the
label-free rule kept 89.7 % of twin-shared claims against 47.7 % of accidental
ones, a 42 pp separation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .spot_attribution import (
    DEFAULT_MIN_SPOTS,
    DEFAULT_REL_THRESHOLD,
    attribute_from_spot_matrix,
    twin_agreement,
)


def _twin_map(grains_path: Path, *, max_distance_um: float, tol_deg: float):
    """Σ3 twin partners per GrainID, via the post-hoc twin analysis."""
    from .io.read import read_grains_csv
    from .twins_posthoc import find_twins

    g = read_grains_csv(grains_path)
    sg = 225
    meta_sg = getattr(g, "space_group", None)
    if meta_sg:
        sg = int(meta_sg)
    res = find_twins(g.orient_mat, g.positions, sg,
                     max_distance_um=max_distance_um, tol_deg=tol_deg)
    ids = np.asarray(g.ids, dtype=np.int64)
    twin_of: dict[int, set] = {}
    for i, j in res.pairs:
        twin_of.setdefault(int(ids[i]), set()).add(int(ids[j]))
        twin_of.setdefault(int(ids[j]), set()).add(int(ids[i]))
    return twin_of, res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-spot-attribution",
        description="Per-claim soft attribution of contested reflections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A contested spot is not automatically a bad spot: twin variants "
               "legitimately share reflections and those are the BEST-fitting "
               "spots present. Use --twin-check before trusting a filter.")
    p.add_argument("spot_matrix", help="path to SpotMatrix.csv")
    p.add_argument("--rel-threshold", type=float, default=DEFAULT_REL_THRESHOLD,
                   help="keep a claim whose consistency is at least this "
                        f"fraction of the best claimant's (default "
                        f"{DEFAULT_REL_THRESHOLD})")
    p.add_argument("--min-spots", type=int, default=DEFAULT_MIN_SPOTS,
                   help=f"never strip a grain below this many spots (default "
                        f"{DEFAULT_MIN_SPOTS}; 0 disables)")
    p.add_argument("--sigma", type=float, default=None,
                   help="residual scale in um (default: median uncontested "
                        "residual, i.e. derived from spots nobody disputes)")
    p.add_argument("--residual-column", default="DiffLenPost",
                   help="SpotMatrix column to judge claims on (default "
                        "DiffLenPost; falls back to DiffLen)")
    p.add_argument("--out", help="write the full per-claim table here (CSV)")
    p.add_argument("--emit-spot-sets",
                   help="write 'GrainID,SpotID' for KEPT claims only — the "
                        "input for a second refinement pass")
    p.add_argument("--grains", help="Grains.csv, required by --twin-check")
    p.add_argument("--twin-check", action="store_true",
                   help="score the filter against Sigma3 twin labels it never "
                        "saw (validation, not a dependency)")
    p.add_argument("--twin-distance-um", type=float, default=45.0)
    p.add_argument("--twin-tol-deg", type=float, default=1.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sm = Path(args.spot_matrix)
    if not sm.exists():
        print(f"error: no such file: {sm}", file=sys.stderr)
        return 2

    att = attribute_from_spot_matrix(
        sm, residual_column=args.residual_column, sigma=args.sigma,
        rel_threshold=args.rel_threshold, min_spots=args.min_spots)
    print(att.summary())

    if args.twin_check:
        if not args.grains:
            print("error: --twin-check needs --grains Grains.csv", file=sys.stderr)
            return 2
        twin_of, res = _twin_map(Path(args.grains),
                                 max_distance_um=args.twin_distance_um,
                                 tol_deg=args.twin_tol_deg)
        ta = twin_agreement(att, twin_of)
        print(f"\ntwin check ({len(res.pairs)} adjacent Sigma3 pairs; labels used "
              f"only to score):")
        print(f"  twin-shared claims {ta['n_twin_claims']:9.0f}   kept "
              f"{100*ta['keep_rate_twin']:6.2f}%")
        print(f"  accidental claims  {ta['n_accidental_claims']:9.0f}   kept "
              f"{100*ta['keep_rate_accidental']:6.2f}%")
        sep = ta["keep_rate_twin"] - ta["keep_rate_accidental"]
        if np.isnan(sep):
            print("  no contested claims to separate")
        elif sep > 0.30:
            print(f"  separation {100*sep:.1f} pp — the residual-only rule "
                  f"recovers the split")
        else:
            print(f"  separation {100*sep:.1f} pp — WEAK. The filter is not "
                  f"distinguishing legitimate twin sharing from accidental "
                  f"overlap on this data; do not use it to drop spots.")

    if args.out:
        hdr = ("GrainID,SpotID,Residual,NClaimants,Consistency,Responsibility,"
               "RelLikelihood,Keep")
        arr = np.column_stack([att.grain_id, att.spot_id, att.residual,
                               att.n_claimants, att.consistency,
                               att.responsibility, att.rel_likelihood,
                               att.keep.astype(int)])
        np.savetxt(args.out, arr, delimiter=",", header=hdr, comments="",
                   fmt=["%d", "%d", "%.6f", "%d", "%.6e", "%.6f", "%.6e", "%d"])
        print(f"\nwrote {args.out} ({len(arr)} claims)")

    if args.emit_spot_sets:
        sets = att.kept_spot_sets()
        with open(args.emit_spot_sets, "w") as fh:
            fh.write("GrainID,SpotID\n")
            for g in sorted(sets):
                for s in sets[g]:
                    fh.write(f"{g},{s}\n")
        n = sum(len(v) for v in sets.values())
        print(f"wrote {args.emit_spot_sets}: {n} kept claims over "
              f"{len(sets)} grains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
