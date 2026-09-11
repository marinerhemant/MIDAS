"""Copy the shipped tutorials out of site-packages, into a folder you can edit.

The tutorials install inside the package, where editing them is awkward and a reinstall
would overwrite your changes. Copy them out first::

    python -m midas_dfxm.examples.get_notebooks                 # -> ./midas_dfxm_tutorials
    python -m midas_dfxm.examples.get_notebooks my_folder
    python -m midas_dfxm.examples.get_notebooks my_folder --force   # overwrite existing copies

Existing files are never overwritten without ``--force``, so re-running cannot erase a
notebook you have been working in.
"""
from __future__ import annotations

import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIALS = {
    "rocking_curve_reconstruction_tutorial.ipynb":
        "the school notebook: Part A simulates rocking scans, Part B reduces a scan you "
        "measured (APS 6-ID-C)",
    "reduce_6idc_scan.ipynb":
        "Part B on its own, for a measured theta or theta-2theta scan, with a cell to pick the "
        "ROI from the frames",
    "tutorial_school_dfxm.py":
        "the differentiable forward model and full-F inverse on synthetic data (Run Cell in "
        "VS Code, or run it as a script)",
}


def copy_tutorials(dest: str = "midas_dfxm_tutorials", *, force: bool = False) -> list:
    """Copy the tutorials into ``dest`` and return the paths written or kept."""
    os.makedirs(dest, exist_ok=True)
    out = []
    for name in TUTORIALS:
        src = os.path.join(_HERE, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{name} is not in this installation ({_HERE}); "
                                    "reinstall a midas-dfxm release that ships the tutorials")
        dst = os.path.join(dest, name)
        if os.path.exists(dst) and not force:
            print(f"kept existing {dst} (use --force to overwrite)")
        else:
            shutil.copy2(src, dst)
            print(f"copied {dst}")
        print(f"    {TUTORIALS[name]}")
        out.append(os.path.abspath(dst))
    return out


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if any(a in ("-h", "--help") for a in args) or len(args) > 1:
        print(__doc__)
        return 0 if args and args[0] in ("-h", "--help") else 2
    copy_tutorials(args[0] if args else "midas_dfxm_tutorials", force=force)
    print("open the notebook with:  jupyter lab   (pip install jupyterlab, if needed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
