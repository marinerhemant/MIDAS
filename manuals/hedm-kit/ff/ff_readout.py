#!/usr/bin/env python3
"""Do the real grains separate from the omega-scrambled null arms?   Usage: ff_readout.py <run_dir>   (real/, null1/, null2/, null3/ under it)

Prints grains per arm, the highest null Confidence, N_above (real grains above EVERY null grain), the Mann-Whitney AUC of real versus
pooled null Confidences, and the gap between the real 5th and null 95th percentile. A null arm with no Grains.csv has 0 grains.
Confidence of two different metrics (raw vs weighted) is not comparable; compare N_above and AUC within one setting.
Read it against what you registered BEFORE you saw it. Status of this method: PROVISIONAL (reference beamtime, one layer).
"""
import os
import sys

import numpy as np


def conf(path):
    if not os.path.exists(path):
        return np.zeros(0)
    g = np.genfromtxt(path, comments="%")
    return np.atleast_2d(g)[:, 23] if g.size else np.zeros(0)       # 0-based column 23 = Confidence


def auc(a, b):
    from scipy.stats import rankdata
    if not len(a) or not len(b):
        return float("nan")
    r = rankdata(np.concatenate([a, b]))
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    D = sys.argv[1]
    real = conf(f"{D}/real/results/LayerNr_1/Grains.csv")
    nulls = [conf(f"{D}/null{k}/results/LayerNr_1/Grains.csv") for k in (1, 2, 3)]
    pooled = np.concatenate(nulls) if any(len(n) for n in nulls) else np.zeros(0)
    mx = pooled.max() if len(pooled) else -1.0
    print(f"real grains {len(real)} (median Confidence {np.median(real) if len(real) else float('nan'):.3f}); null grains {[len(n) for n in nulls]}, highest null Confidence {mx:.3f}")
    print(f"N_above (real grains above every null grain): {int((real > mx).sum())}")
    print(f"AUC real vs pooled null: {auc(real, pooled):.3f}")
    if len(real) and len(pooled):
        print(f"real p5 - null p95 Confidence: {np.percentile(real, 5) - np.percentile(pooled, 95):.3f}")


if __name__ == "__main__":
    main()
