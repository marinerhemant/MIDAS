"""``midas-plot`` — one-shot reconstruction figures from the shell."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="midas-plot",
        description="Standard MIDAS reconstruction maps (orientation, "
                    "confidence, grains).")
    ap.add_argument("mics", nargs="+", help="text .mic file(s)")
    ap.add_argument("--kind", default="orientation",
                    choices=["orientation", "confidence", "grain"])
    ap.add_argument("--sg", default="225",
                    help="space group; a single value, or one per .mic "
                         "comma-separated when comparing PHASES (colouring a "
                         "cubic map with hexagonal symmetry silently produces "
                         "a meaningless figure)")
    ap.add_argument("--cmin", type=float, default=0.3,
                    help="confidence cut (default 0.3, the trust floor)")
    ap.add_argument("--axis", default="0,0,1", help="IPF sample axis")
    ap.add_argument("--titles", default=None, help="'|'-separated")
    ap.add_argument("--suptitle", default=None)
    ap.add_argument("-o", "--out", default="midas_plot.png")
    ap.add_argument("--dpi", type=int, default=145)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    from .maps import compare_maps
    from .mic import read_mic

    mics = [read_mic(m) for m in a.mics]
    for m in mics:
        print(f"{m.path.name}: {m.summary()}")

    sgs = [int(v) for v in str(a.sg).split(",")]
    if len(sgs) == 1:
        sgs *= len(mics)
    elif len(sgs) != len(mics):
        ap.error(f"--sg has {len(sgs)} values for {len(mics)} .mic files; "
                 "give one value or one per file")

    titles = a.titles.split("|") if a.titles else [None] * len(mics)
    axis = tuple(float(v) for v in a.axis.split(","))

    import matplotlib.pyplot as plt
    from .maps import confidence_map, grain_map, orientation_map
    fn = {"orientation": orientation_map, "confidence": confidence_map,
          "grain": grain_map}[a.kind]
    fig, axes = plt.subplots(1, len(mics), figsize=(6.2 * len(mics), 6.4),
                             squeeze=False)
    for ax, m, t, sg in zip(axes[0], mics, titles, sgs):
        kw = {}
        if a.kind == "orientation":
            kw = dict(space_group=sg, cmin=a.cmin, axis=axis)
        elif a.kind == "grain":
            kw = dict(space_group=sg, cmin=a.cmin)
        fn(m, ax=ax, title=t, **kw)
    if a.suptitle:
        fig.suptitle(a.suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
