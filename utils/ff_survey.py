#!/usr/bin/env python3
"""Survey a far-field data folder. Reads metadata only; cheap on a full beamtime.

Usage:  python ff_survey.py <data-dir> [<metadata-dir>]

Writes nothing. Paste the output into SURVEY.md and fill in the CHECKS.
"""
import re
import sys
from pathlib import Path

import h5py

CALIBRANTS = ("ceo2", "lab6", "si_", "al2o3", "ni_powder", "au_powder")


def scalar(f, path):
    try:
        v = f[path][()]
        return float(v.ravel()[0] if hasattr(v, "ravel") else v)
    except Exception:
        return None


def classify(name):
    low = name.lower()
    if "dark" in low:
        return "DARK"
    if any(c in low for c in CALIBRANTS):
        return "CALIBRANT"
    return "sweep"


def acq_number(name):
    m = re.search(r"_(\d{6})\.", name)
    return int(m.group(1)) if m else None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = Path(sys.argv[1])
    files = sorted(p for p in data.rglob("*.h5") if p.is_file())
    if not files:
        sys.exit(f"no .h5 under {data} — is this the image tree? (§3a)")

    # 1. The ACTUAL layout of one file. Do not assume the paths in §3c.
    print(f"=== HDF5 layout of {files[0].name} — verify against §3c, do not assume ===")
    with h5py.File(files[0], "r") as f:
        f.visititems(lambda n, o: print(f"  {n:52s} {str(o.shape):18s} {o.dtype}")
                     if isinstance(o, h5py.Dataset) else None)

    # 2. Classify every file.
    print(f"\n=== {len(files)} file(s) ===")
    print(f"{'file':<46}{'kind':<11}{'frames':>8}{'energy':>9}{'DetZ':>10}")
    print("-" * 84)
    rows = []
    for p in files:
        kind, n, e, z = classify(p.name), None, None, None
        try:
            with h5py.File(p, "r") as f:
                if "exchange/data" in f:
                    n = f["exchange/data"].shape[0]
                e = scalar(f, "instrument/HEM/Energy")
                z = scalar(f, "instrument/DMS/DetZ")
        except Exception as exc:
            kind = f"UNREADABLE({exc.__class__.__name__})"
        rows.append((p, kind))
        fs = "" if n is None else str(n)
        es = "" if e is None else f"{e:.3f}"
        zs = "" if z is None else f"{z:.2f}"
        print(f"{p.name:<46}{kind:<11}{fs:>8}{es:>9}{zs:>10}")

    # 3. dark_before_<N-1> pairs with data <N>  (§3d)
    darks = {acq_number(p.name): p for p, k in rows if k == "DARK"}
    print("\n=== proposed dark pairing (dark_before_<N-1> -> data <N>, §3d) ===")
    for p, k in rows:
        if k == "DARK":
            continue
        n = acq_number(p.name)
        if n is None:
            continue
        d = darks.get(n - 1)
        miss = "*** none by the N-1 rule — find its dark by name before proceeding ***"
        print(f"  {p.name:<46} <- {d.name if d else miss}")

    kinds = [k for _, k in rows]
    print("\n=== CHECKS — answer these in SURVEY.md before writing a paramfile ===")
    if "CALIBRANT" not in kinds:
        print("  *** no calibrant file matched. STOP: no geometry without one (§5) ***")
    print("  [ ] frame counts match the par file image range (§3b)")
    print("  [ ] energy from instrument/HEM/Energy, cross-checked vs Emon f6 + spec log —")
    print("      NEVER from the filename (§4a)")
    print("  [ ] DetZ recorded as an Lsd SEED only (§4b)")
    print("  [ ] every sweep has a dark, and the dark is non-zero after zipping (§3d)")
    print("  [ ] SkipFrame 1 for GE/far-field (§3e)")

    if len(sys.argv) > 2:
        md = Path(sys.argv[2])
        print(f"\n=== metadata dir: {md} ===")
        par = sorted(md.glob("*_FF.par"))
        print(f"  *_FF.par        : {[q.name for q in par] or '*** MISSING — cannot write a paramfile (§3a) ***'}")
        for extra in ("fastsweep_Emon.txt", "FullLog.log"):
            print(f"  {extra:<16}: {'present' if (md / extra).exists() else '*** missing ***'}")
    else:
        print("\n=== metadata dir not given — find it before writing a paramfile (§3a) ===")
        print("  find /home/beams /gdata -maxdepth 6 -name '*_FF.par' 2>/dev/null")


if __name__ == "__main__":
    main()
