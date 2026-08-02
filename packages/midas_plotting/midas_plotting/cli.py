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
                    help="near-field .mic file(s), a far-field Grains.csv, or a "
                         "Laue solutions.txt / validated .npz")
    ap.add_argument("--kind", default=None,
                    help="NF: orientation | confidence | grain. "
                         "FF: summary | orientation | pole | strain | size | "
                         "completeness | 3d. "
                         "Laue: summary | orientation | pole | tilt | size | "
                         "sweep. "
                         "Default: orientation for NF, summary for FF/Laue.")
    ap.add_argument("--plane", default="xy",
                    help="FF only: projection plane (xy, xz, yz)")
    ap.add_argument("--hkl", default="0,0,1",
                    help="FF pole figure: crystal direction")
    ap.add_argument("--strain-kind", default="hydrostatic",
                    help="FF strain: hydrostatic | vonmises | 11 | 33 | ...")
    ap.add_argument("--sg", default=None,
                    help="space group; a single value, or one per .mic "
                         "comma-separated when comparing PHASES (colouring a "
                         "cubic map with hexagonal symmetry silently produces "
                         "a meaningless figure). Unset: 225 for near-field, "
                         "the file's own header for far-field, 194 for Laue.")
    ap.add_argument("--cmin", type=float, default=0.3,
                    help="confidence cut (default 0.3, the trust floor)")
    ap.add_argument("--axis", default="0,0,1", help="IPF sample axis")
    ap.add_argument("--titles", default=None, help="'|'-separated")
    ap.add_argument("--suptitle", default=None)
    ap.add_argument("-o", "--out", default="midas_plot.png")
    ap.add_argument("--dpi", type=int, default=145)
    ap.add_argument("--gate", type=int, default=None,
                    help="Laue: keep solutions matching MORE than this many "
                         "reflections. No default -- it is the measured "
                         "random-orientation null for that scan, not a "
                         "universal constant.")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="Laue: grain clustering tolerance in degrees")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    if all(_looks_like_laue(m) for m in a.mics):
        return _run_laue(a, ap)
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

    sgs = [int(v) for v in str(a.sg if a.sg is not None else "225").split(",")]
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
    # `kind` (bare) used to be referenced here; it is only ever bound in the FF
    # branch, so every near-field CLI run raised NameError after doing all the
    # work and before writing the file. No test covered the CLI path.
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


def _looks_like_laue(path) -> bool:
    """Laue solutions.txt or a validated .npz, by content not by filename."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix == ".npz":
        try:
            import numpy as np
            with np.load(p, allow_pickle=True) as d:
                return {"oms", "X", "Z", "nhit"} <= set(d.files)
        except Exception:
            return False
    try:
        with p.open() as fh:
            head = fh.readline()
    except OSError:
        return False
    return head.startswith("%ImageNr") and "OrientMatrix0" in head


def _run_laue(a, ap) -> int:
    """Laue plotting branch."""
    import matplotlib.pyplot as plt

    from . import laue
    from .solutions import read_solutions, read_validated

    kind = a.kind or "summary"
    if a.sg is None:
        sg = 194
        print("note: --sg not given, using 194 (hexagonal). The wrong symmetry "
              "silently changes grain counts, so pass it for another phase.")
    else:
        sg = int(str(a.sg).split(",")[0])
    hkl = tuple(float(v) for v in a.hkl.split(","))

    sols = [read_validated(m) if str(m).endswith(".npz") else read_solutions(m)
            for m in a.mics]
    for s in sols:
        print(f"{Path(s.path).name}: {s.summary()}")
    if a.gate is not None:
        sols = [s.gate(a.gate) for s in sols]
        for s in sols:
            print(f"  after gate >{a.gate}: {len(s)} solutions")
    else:
        print("note: no --gate given, so every solution is plotted including "
              "ones a randomly oriented crystal could produce.")

    if kind == "summary":
        if len(sols) != 1:
            ap.error("--kind summary takes exactly one Laue input")
        fig = laue.summary(sols[0], tolerance=a.tol, space_group=sg, hkl=hkl)
    else:
        fig, axes = plt.subplots(1, len(sols),
                                 figsize=(6.2 * len(sols), 5.4), squeeze=False)
        for ax, s in zip(axes[0], sols):
            if kind == "orientation":
                laue.orientation_map(s, ax, hkl=hkl)
            elif kind in ("pole", "tilt", "size"):
                c = laue.cluster(s, a.tol, space_group=sg)
                reps = c.representatives(s.orient_mat)
                if kind == "pole":
                    laue.pole_figure(reps, ax, hkl=hkl)
                elif kind == "tilt":
                    laue.tilt_histogram(reps, ax, hkl=hkl)
                else:
                    laue.grain_size_distribution(c, ax)
            elif kind == "sweep":
                laue.tolerance_sweep(s, ax, space_group=sg)
            else:
                ap.error(f"--kind {kind!r} is not valid for Laue input; use "
                         "summary, orientation, pole, tilt, size or sweep")
        fig.tight_layout()
    if a.suptitle:
        fig.suptitle(a.suptitle, fontsize=12)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {Path(a.out).resolve()}")
    return 0


def _run_ff(a, ap) -> int:
    """Far-field plotting branch."""
    import matplotlib.pyplot as plt

    from . import ff
    from .grains import read_grains

    kind = a.kind or "summary"
    axis = tuple(float(v) for v in a.axis.split(","))
    # Unset means "use the file's own header" -- Grains.csv states its space
    # group, and overriding it with a default would colour a hexagonal sample
    # through the cubic triangle and produce a plausible, wrong figure.
    sg = None if a.sg in (None, "", "auto") else int(str(a.sg).split(",")[0])

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
