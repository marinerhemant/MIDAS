"""Seed ring: is the ring the indexer is seeded from a good one?

The FF indexer tries each spot of ``OverAllRingToIndex`` (within the omega range) as a seed, so a grain can only be found if
it has a detectable spot on that ring. Picking the ring by hand or by name copies a calibrant's choice onto a sample whose
structure factors are different. On Li6PS5Cl (S2 70wt cathode, 2026-10-29) the run was seeded from ring 8, a (420) reflection
with |F|^2 = 1 % of the strongest ring; seeding the same layer from ring 7 (4.6 %) or ring 2 (9.9 %) gave 2.6x and 14x more
real grains above an omega-scrambled null (raw ConfidenceMetric, lpsc/f2_readout.txt, ONE sample).

What this module does: tabulate, per ring present in ``InputAll.csv``, the mean ``F2`` of its reflections (the 12th column of
``hkls.csv``, present only when a PhaseAtom/PhaseCIF basis is declared, relative to the strongest ring that has spots), the
number of spots, and the estimated number of seeds; then recommend the ring with the largest F2 whose seed list fits a
budget. It only REPORTS (a log warning and ``seed_ring.json``): changing the seed ring changes the seed list and so the run,
which is the user's decision, and the rule is a heuristic from the one sample above.

Without an F2 column there is nothing to say about strength: the table is still written, no ring is recommended, and the
log says how to get one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["ring_seed_table", "advise_seed_ring", "report_seed_ring"]

MIN_REL_F2 = 0.05        # a seed ring weaker than 5 % of the strongest searched ring is flagged
MAX_SEEDS = 20000        # compute budget: seeds attempted per layer (LPSC: 10.6k seeds = minutes on 56 cores)
MIN_SEEDS = 50           # fewer than this cannot sample the grains


def ring_seed_table(hkls_csv: Union[str, Path], input_all_csv: Union[str, Path],
                    seed_fraction: float = 0.5) -> List[Dict]:
    """One row per ring that has spots: ``ring, n_reflections, f2_rel (None without F2), n_spots, n_seeds_est``.

    ``f2_rel`` is the mean F2 over the ring's rows divided by the largest such mean over rings that have spots.
    ``n_seeds_est = n_spots * seed_fraction`` (the omega-range filter keeps about half of a full-rotation ring).
    """
    with open(hkls_csv) as fh:
        head = fh.readline().split()
    if "RingNr" not in head:
        raise ValueError(f"{hkls_csv}: no RingNr column")
    ir = head.index("RingNr")
    has_f2 = "F2" in head
    cols = (ir, head.index("F2")) if has_f2 else (ir,)
    h = np.loadtxt(hkls_csv, skiprows=1, usecols=cols, ndmin=2)
    rings_h = h[:, 0].astype(int)
    f2_by_ring = {int(r): float(h[rings_h == r, 1].mean()) for r in np.unique(rings_h)} if has_f2 else {}
    mult = {int(r): int((rings_h == r).sum()) for r in np.unique(rings_h)}

    import pandas as pd
    ia = pd.read_csv(input_all_csv, sep=r"\s+", usecols=["RingNumber"])
    counts = ia["RingNumber"].astype(int).value_counts()
    counts = counts[counts.index > 0]                      # ring 0 = placeholder rows
    present = {int(r): int(n) for r, n in counts.items()}
    fmax = max((f2_by_ring.get(r, 0.0) for r in present), default=0.0) if has_f2 else 0.0
    rows = []
    for r in sorted(present):
        rel = (f2_by_ring.get(r, 0.0) / fmax) if (has_f2 and fmax > 0) else None
        rows.append({"ring": r, "n_reflections": mult.get(r, 0), "f2_rel": rel, "n_spots": present[r],
                     "n_seeds_est": int(round(present[r] * seed_fraction))})
    return rows


def advise_seed_ring(table: List[Dict], current: Optional[int], *, min_rel_f2: float = MIN_REL_F2,
                     max_seeds: int = MAX_SEEDS, min_seeds: int = MIN_SEEDS) -> Dict:
    """Recommendation from a :func:`ring_seed_table` result: the ring with the largest F2 among those with
    ``min_seeds <= n_seeds_est <= max_seeds`` and ``f2_rel >= min_rel_f2`` (ties: fewer seeds)."""
    cur = next((r for r in table if r["ring"] == current), None)
    out = {"current": current, "current_f2_rel": cur["f2_rel"] if cur else None, "recommended": None,
           "warn": False, "message": ""}
    if not table or all(r["f2_rel"] is None for r in table):
        out["message"] = ("no F2 column in hkls.csv: declare PhaseCIF or PhaseAtom so the seed ring can be judged "
                          "by structure factor")
        return out
    cands = [r for r in table if r["f2_rel"] is not None and r["f2_rel"] >= min_rel_f2
             and min_seeds <= r["n_seeds_est"] <= max_seeds]
    best = max(cands, key=lambda r: (r["f2_rel"], -r["n_seeds_est"])) if cands else None
    out["recommended"] = best["ring"] if best else None
    weak = cur is not None and cur["f2_rel"] is not None and cur["f2_rel"] < min_rel_f2
    if weak:
        out["warn"] = True
        out["message"] = (f"seed ring {current} is a weak reflection (|F|^2 = {100 * cur['f2_rel']:.1f} % of the strongest "
                          f"searched ring; flag below {100 * min_rel_f2:.0f} %): grains are only found if they show a spot there")
        if best is not None and best["ring"] != current:
            out["message"] += (f". Ring {best['ring']} ({100 * best['f2_rel']:.1f} %, about {best['n_seeds_est']} seeds) "
                               f"fits the budget of {max_seeds} seeds; set OverAllRingToIndex {best['ring']} to seed from it "
                               "(heuristic, measured on one sample)")
        else:
            out["message"] += f"; no ring has |F|^2 >= {100 * min_rel_f2:.0f} % with {min_seeds}-{max_seeds} seeds"
    return out


def report_seed_ring(layer_dir: Union[str, Path], current: Optional[int]) -> Dict:
    """Write ``seed_ring.json`` and log the advice. Diagnostic only: never raises."""
    layer_dir = Path(layer_dir)
    try:
        hk, ia = layer_dir / "hkls.csv", layer_dir / "InputAll.csv"
        frac = 0.5
        sti = layer_dir / "SpotsToIndex.csv"
        tab = ring_seed_table(hk, ia, frac)
        if current is not None and sti.exists():
            n_seed = sum(1 for ln in sti.read_text().splitlines() if ln.strip())
            cur = next((r for r in tab if r["ring"] == current), None)
            if cur and cur["n_spots"] > 0 and n_seed > 0:           # calibrate the seed fraction on the ring actually used
                frac = n_seed / cur["n_spots"]
                tab = ring_seed_table(hk, ia, frac)
        adv = advise_seed_ring(tab, current)
        adv["table"] = tab
        (layer_dir / "seed_ring.json").write_text(json.dumps(adv, indent=1))
        if adv["warn"]:
            LOG.warning("seed ring: %s", adv["message"])
        elif adv["message"]:
            LOG.info("seed ring: %s", adv["message"])
        return adv
    except Exception as e:                                           # noqa: BLE001 - diagnostic only
        LOG.warning("seed ring check failed (%s)", e)
        return {"error": str(e)}
