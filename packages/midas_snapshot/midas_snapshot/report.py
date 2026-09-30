"""Summary report: one JSON that cites the files every number came from, plus
figures (traces; off-matrix feature d-histogram; injection curve)."""
from __future__ import annotations

import json
import os

import numpy as np

from .config import SnapshotConfig


def write_report(cfg: SnapshotConfig) -> str:
    out = cfg.out
    ana_path = os.path.join(out, "analysis.json")
    ana = json.load(open(ana_path)) if os.path.exists(ana_path) else {}
    rep = dict(config=os.path.join(out, "snapshot_config.json"), windows={}, analysis=ana_path)
    for W in cfg.window_sizes:
        wp = os.path.join(out, f"windows_W{W}.csv")
        if not os.path.exists(wp):
            continue
        win = np.atleast_1d(np.genfromtxt(wp, delimiter=",", names=True))   # one-window series
        meta = json.load(open(os.path.join(out, f"meta_W{W}.json")))
        rep["windows"][f"W{W}"] = dict(file=wp, meta=os.path.join(out, f"meta_W{W}.json"),
                                       n_windows=int(len(win)), threshold=meta["threshold"],
                                       wall_s=meta["wall_s"],
                                       scale_valid=int(np.isfinite(win["scale"]).sum()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        W = cfg.window_sizes[-1]
        win = np.atleast_1d(np.genfromtxt(os.path.join(out, f"windows_W{W}.csv"), delimiter=",", names=True))
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(win["first_frame"], win["scale"], ".", ms=2); ax[0].set_ylabel("matrix scale")
        ax[1].plot(win["first_frame"], win["halo"], lw=0.8); ax[1].set_ylabel("halo index")
        ax[2].plot(win["first_frame"], win["n_spots"], lw=0.8); ax[2].set_ylabel("spots"); ax[2].set_yscale("log")
        ax[2].set_xlabel("frame")
        a = ana.get(f"W{W}", {}).get("windows", {})
        for key, col in (("before", "tab:blue"), ("after", "tab:orange")):
            if a.get(key):
                for x in ax:
                    x.axvspan(a[key][0], a[key][1], color=col, alpha=0.1)
        fig.tight_layout(); p = os.path.join(out, f"traces_W{W}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        rep["figures"] = [p]
    except ImportError:
        rep["figures"] = []
    path = os.path.join(out, "report.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    return path
