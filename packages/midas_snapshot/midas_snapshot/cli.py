"""midas-snapshot command line.

    midas-snapshot init    --frames DIR --geometry PS --out DIR [...]   write a config
    midas-snapshot run     CONFIG                                        per-window pass (all window sizes)
    midas-snapshot analyse CONFIG [--candidates CIF ...] [--controls]    windows, features, phase test
    midas-snapshot report  CONFIG                                        summary JSON + figures
    midas-snapshot select  --frames DIR --geometries A=a.txt B=b.txt --matrices X=x.cif ...
                           pick the (geometry, matrix) pair that fits the series' first frames

Every stage writes its numbers to files in the output directory; the report cites them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .config import SnapshotConfig


def _init(a):
    cfg = SnapshotConfig(frames=a.frames, geometry=a.geometry, out=a.out, flip=a.flip, mask=a.mask,
                         matrix_cif=a.matrix_cif,
                         halo_band=a.halo_band, base_band=a.base_band, tth_max=a.tth_max)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "snapshot_config.json")
    cfg.save(path)
    print(path)


def _run(a):
    from .pipeline import run
    cfg = SnapshotConfig.load(a.config)
    for W in cfg.window_sizes:
        m = run(cfg, W)
        t = m["threshold"]
        tw = m.get("thresholds_per_window") or []
        ts = (f"{min(t['T']):.3g}-{max(t['T']):.3g} (by background level)" if isinstance(t, dict) else
              (f"{min(tw):.3g}-{max(tw):.3g} (per window)" if t == "per_window" and tw else f"{t}"))
        print(f"W{W}: {m['n_windows']} windows, threshold {ts}, {m['wall_s']:.1f} s")


def _analyse(a):
    from midas_hkls.io.cif import read_cif
    from .analysis import choose_windows, classify, features, load_run, phase_test
    from .controls import injection_curve
    from .pipeline import setup, threshold_for
    from .io import list_frames
    cfg = SnapshotConfig.load(a.config)
    maps, lines, fit_lines = setup(cfg)
    from .pipeline import extra_phases
    extra = extra_phases(cfg, maps)
    cands = {os.path.splitext(os.path.basename(p))[0]: read_cif(p) for p in (a.candidates or [])}
    from .io import load_mask, read_frame
    valid = read_frame(list_frames(cfg.frames)[0], cfg.flip, load_mask(cfg.mask, cfg.flip),
                       cfg.invalid_below) >= 0
    result = {}
    for W in cfg.window_sizes:
        win, spots, meta = load_run(cfg.out, W)
        cfg.sigma_px = meta.get("sigma_px_used", cfg.sigma_px)     # the width the run measured and used
        ex_scales = []
        psf = os.path.join(cfg.out, f"phase_scales_W{W}.csv")
        if extra and os.path.exists(psf):
            ps = np.atleast_2d(np.genfromtxt(psf, delimiter=",", skip_header=1))
            ex_scales = [float(np.nanmedian(ps[:, k + 1])) if np.isfinite(ps[:, k + 1]).any() else np.nan
                         for k in range(len(extra))]
        extra_scaled = [(el, es) for (el, _), es in zip(extra, ex_scales)]
        if a.map:
            from .analysis import map_features
            allw = {"before": (0, meta["n_frames"] - 1)}
            cls = classify(spots, W, allw, cfg)
            feats = map_features(cfg, spots, cls)
            r = dict(mode="map", sigma_matrix=cls["sigma_matrix"], n_core=cls["n_core"],
                     n_off=int(cls["off"].sum()), features=feats)
            if cands:
                scales = win["scale"][np.isfinite(win["scale"])]
                mscale = float(np.median(scales)) if scales.size else float("nan")
                ff = dict(n_features=feats["n_features"], d=feats["d"])
                r["phase"] = phase_test(cfg, ff, cls, maps, lines, mscale, cands, n_null=a.n_null,
                                        valid=valid, require_vanishing=False, extra=extra_scaled)
            result[f"W{W}"] = r
            continue
        if a.static:
            windows = dict(before=(0, meta["n_frames"] - 1), after=None,
                           rule="static series: one window, no event")
        else:
            windows = choose_windows(cfg, win, meta["n_frames"])
        r = dict(windows={k: (list(v) if isinstance(v, tuple) else v) for k, v in windows.items()})
        if windows.get("before") is None:
            result[f"W{W}"] = r
            continue
        cls = classify(spots, W, windows, cfg)
        r.update(sigma_matrix=cls["sigma_matrix"], n_core=cls["n_core"],
                 n_analysed=int(cls["analysed"].sum()), n_off=int((cls["off"] & cls["analysed"]).sum()))
        scales = win["scale"][np.isfinite(win["scale"])]
        mscale = float(np.median(scales)) if scales.size else float("nan")
        feats = features(cfg, spots, W, cls, windows, maps, stride=a.stride)
        r["features"] = feats
        if cands:
            r["phase"] = phase_test(cfg, feats, cls, maps, lines, mscale, cands, n_null=a.n_null,
                                    valid=valid, require_vanishing=not a.static, extra=extra_scaled)
        if a.controls and cands and windows.get("after"):
            T = meta["threshold"]
            from midas_hkls.feature_phase import allowed_d_lines
            first = next(iter(cands.values()))
            d_lines = allowed_d_lines(first, 0.3, 50.0)
            r["injection"] = injection_curve(cfg, maps, lines, fit_lines, T, d_lines, windows["after"],
                                             W, cls["sigma_matrix"], extra=extra)
        result[f"W{W}"] = r
    path = os.path.join(cfg.out, "analysis.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(path)


def _select(a):
    from .setup_select import select_setup
    kv = lambda items: dict(x.split("=", 1) for x in items)
    res = select_setup(a.frames, kv(a.geometries), kv(a.matrices), flip=a.flip, mask=a.mask,
                       n_frames=a.n_frames, blocks=(0.0, 0.5, 0.95))
    out = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
    print(out)


def _report(a):
    from .report import write_report
    print(write_report(SnapshotConfig.load(a.config)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="midas-snapshot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--frames", required=True); p.add_argument("--geometry", required=True)
    p.add_argument("--out", required=True); p.add_argument("--flip", default="ud")
    p.add_argument("--mask"); p.add_argument("--matrix-cif", dest="matrix_cif")
    p.add_argument("--halo-band", dest="halo_band", type=float, nargs=2)
    p.add_argument("--base-band", dest="base_band", type=float, nargs=2)
    p.add_argument("--tth-max", dest="tth_max", type=float)
    p.set_defaults(fn=_init)
    p = sub.add_parser("run"); p.add_argument("config"); p.set_defaults(fn=_run)
    p = sub.add_parser("analyse"); p.add_argument("config")
    p.add_argument("--candidates", nargs="*"); p.add_argument("--controls", action="store_true")
    p.add_argument("--stride", type=int, default=1); p.add_argument("--n-null", dest="n_null", type=int, default=2000)
    p.add_argument("--static", action="store_true",
                   help="static series at one position: one window, no before/after test")
    p.add_argument("--map", action="store_true",
                   help="position map: frames are sample positions (no windows, no cross-frame merging)")
    p.set_defaults(fn=_analyse)
    p = sub.add_parser("report"); p.add_argument("config"); p.set_defaults(fn=_report)
    p = sub.add_parser("select")
    p.add_argument("--frames", required=True)
    p.add_argument("--geometries", nargs="+", required=True, help="NAME=params.txt ...")
    p.add_argument("--matrices", nargs="+", required=True, help="NAME=matrix.cif ...")
    p.add_argument("--flip", default="ud"); p.add_argument("--mask")
    p.add_argument("--n-frames", dest="n_frames", type=int, default=25); p.add_argument("--out")
    p.set_defaults(fn=_select)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
