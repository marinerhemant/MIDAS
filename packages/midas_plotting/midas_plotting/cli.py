"""``midas-plot`` — one-shot reconstruction figures from the shell."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="midas-plot",
        description="Standard MIDAS reconstruction maps (orientation, "
                    "confidence, grains).")
    ap.add_argument("mics", nargs="+",
                    help="near-field .mic file(s), or a far-field Grains.csv")
    ap.add_argument("--kind", default=None,
                    help="NF: orientation | confidence | grain. "
                         "FF: summary | orientation | pole | strain | size | "
                         "completeness | 3d. "
                         "Default: orientation for NF, summary for FF.")
    ap.add_argument("--plane", default="xy",
                    help="FF only: projection plane (xy, xz, yz)")
    ap.add_argument("--hkl", default="0,0,1",
                    help="FF pole figure: crystal direction")
    ap.add_argument("--strain-kind", default="hydrostatic",
                    help="FF strain: hydrostatic | vonmises | 11 | 33 | ...")
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

    if all(_looks_like_ff(m) for m in a.mics):
        return _run_ff(a, ap)
    if a.kind is None:
        a.kind = "orientation"
    if a.kind not in ("orientation", "confidence", "grain"):
        ap.error(f"--kind {a.kind!r} is not valid for near-field .mic input")
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
    if kind != "summary":          # summary() lays itself out with gridspec
        fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {Path(a.out).resolve()}")
    return 0


def _looks_like_ff(path) -> bool:
    """Far-field Grains.csv, by content not by filename.

    Users rename these constantly (Grains_layer1.csv, au3_grains.csv), so sniff
    for the header MIDAS actually writes instead of matching a name.
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with p.open() as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("%NumGrains") or "\tO11\t" in line:
                    return True
    except OSError:
        return False
    return False


def _run_ff(a, ap) -> int:
    """Far-field plotting branch."""
    import matplotlib.pyplot as plt

    from . import ff
    from .grains import read_grains

    kind = a.kind or "summary"
    axis = tuple(float(v) for v in a.axis.split(","))
    sgs = [int(v) for v in str(a.sg).split(",")] if a.sg else [None]
    sg = None if a.sg in (None, "", "auto") else sgs[0]
    # --sg defaults to "225" for the NF path; on FF prefer the file's own
    # header unless the user actually asked for something else.
    if str(a.sg) == "225":
        sg = None

    grains = [read_grains(m) for m in a.mics]
    for g in grains:
        print(f"{g.path.name}: {len(g)} grains, space group "
              f"{g.space_group if sg is None else sg}")

    if kind == "summary":
        if len(grains) != 1:
            ap.error("--kind summary takes exactly one Grains.csv")
        fig = ff.summary(grains[0], space_group=sg, cmin=a.cmin, axis=axis)
    else:
        fns = {
            "orientation": lambda g, ax: ff.grain_map(
                g, ax, plane=a.plane, space_group=sg, axis=axis, cmin=a.cmin),
            "pole": lambda g, ax: ff.pole_figure(
                g, ax, hkl=tuple(float(v) for v in a.hkl.split(",")),
                space_group=sg, cmin=a.cmin, axis=axis),
            "strain": lambda g, ax: ff.strain_map(
                g, ax, kind=a.strain_kind, plane=a.plane, cmin=a.cmin),
            "size": lambda g, ax: ff.grain_size_distribution(g, ax, cmin=a.cmin),
            "completeness": lambda g, ax: ff.completeness_hist(g, ax),
        }
        if kind == "3d":
            fig = plt.figure(figsize=(6.6 * len(grains), 6.0))
            for k, g in enumerate(grains):
                ax = fig.add_subplot(1, len(grains), k + 1, projection="3d")
                ff.grain_map_3d(g, ax, space_group=sg, axis=axis, cmin=a.cmin)
        elif kind in fns:
            fig, axes = plt.subplots(1, len(grains),
                                     figsize=(6.2 * len(grains), 5.6),
                                     squeeze=False)
            for ax, g in zip(axes[0], grains):
                fns[kind](g, ax)
        else:
            ap.error(f"--kind {kind!r} is not valid for far-field input; use "
                     "summary, orientation, pole, strain, size, completeness "
                     "or 3d")
    if a.suptitle:
        fig.suptitle(a.suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
