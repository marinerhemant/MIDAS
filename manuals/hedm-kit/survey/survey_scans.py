#!/usr/bin/env python3
"""Survey a 20-ID-D style Varex data folder, one row per SCAN FOLDER (layers = data files).

Usage:  python survey_scans.py <varexD-dir> <out.csv> [name-filter-regex]

Reads HDF5 metadata and a few small datasets only (dark/bright frames, one data frame); never a
whole sweep. Every number is read from the files. The energy is NOT in these files (the Varex file does not record it),
so the column `energy_from_name` is an ASSERTION taken from the folder or file name and is flagged.

Columns that matter downstream:
  n_layers      data files, excluding dark_before/dark_after
  darkLoc       which of exchange/dark | exchange/bright holds the dark in the dark_before file (per scan!)
  dark_med_*    median of that dark frame(s); data_med_* median of one data frame, to compare pedestal
  layer_pitch_um  mean |delta samY| between consecutive data files, from the files not from the name
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

NAME_ENERGY = re.compile(r"(\d+(?:p\d+)?)\s*keV", re.I)
NAME_LSD = re.compile(r"(\d{3,4})\s*mm")
FILE_RE = re.compile(r"^(?P<stem>.+?)(?:_(?P<kind>dark_before|dark_after))?_(?P<nr>\d{6})\.(?:vrx\.)?h5$")


def scalar(f, path):
    try:
        v = f[path][()]
        return float(np.asarray(v).ravel()[0])
    except Exception:
        return None


def med_of(f, path, n=1):
    """MEAN of up to n frames of a (k,Y,X) dataset, without reading more than that (a median of a pedestal-free
    dark is exactly 0 and reads as 'all zero', so the mean is used for the dark-group test)"""
    try:
        d = f[path]
        k = d.shape[0]
        idx = sorted({0, k // 2, k - 1})[:n] if k > 1 else [0]
        return float(np.mean(np.concatenate([np.asarray(d[i]).ravel() for i in idx])))
    except Exception:
        return None


def survey_folder(folder: Path, label=None):
    groups = defaultdict(lambda: {"data": [], "dark_before": [], "dark_after": []})
    for p in sorted(folder.glob("*.h5")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        groups[m["stem"]][m["kind"] or "data"].append((int(m["nr"]), p))
    rows = []
    for stem, g in groups.items():
        data = sorted(g["data"])
        if not data:
            continue
        r = {"folder": label or folder.name, "stem": stem, "n_layers": len(data),
             "first_nr": data[0][0], "last_nr": data[-1][0],
             "contiguous": data[-1][0] - data[0][0] + 1 == len(data),
             "dark_before": len(g["dark_before"]), "dark_after": len(g["dark_after"])}
        with h5py.File(data[0][1], "r") as f:
            r["frames"] = f["exchange/data"].shape[0]
            r["shape"] = "x".join(map(str, f["exchange/data"].shape[1:]))
            r["size_GB"] = round(data[0][1].stat().st_size / 1e9, 2)
            for k in ("start", "end", "step", "steptime"):
                r[f"scan_{k}"] = scalar(f, f"measurement/process/scan_parameters/{k}")   # as COMMANDED; can be stale
            try:   # the per-frame readback is the real sweep (jittery step, but first/last/span are reliable)
                a = np.asarray(f["measurement/instrument/SMS/aero"][()]).ravel()
                r["aero_first"], r["aero_last"], r["aero_span"] = float(a[0]), float(a[-1]), float(a.max() - a.min())
            except Exception:
                pass
            try:
                r["stage"] = f["measurement/process/scan_parameters/stage"][()].ravel()[0].decode()
            except Exception:
                r["stage"] = None
            r["detY_mm"] = scalar(f, "measurement/instrument/Detector/detY")
            for s in ("DS", "US"):
                for ax in ("H", "V"):
                    r[f"slit_{s}_{ax}"] = scalar(f, f"measurement/instrument/BCS/slits_{s}/{ax}_size")
            r["acq_time_s"] = scalar(f, "WM/AcqTime")
            r["data_med_f1"] = None
            if r["frames"] > 2:
                d = f["exchange/data"]
                r["data_med_f1"] = float(np.median(np.asarray(d[1])))
                r["data_med_mid"] = float(np.median(np.asarray(d[r["frames"] // 2])))
            try:
                r["sample_name"] = f["WM/SampleName"][()].ravel()[0].decode()
            except Exception:
                r["sample_name"] = None
        # sample translations from first/last data file (layer pitch from the files, not the name)
        ys = []
        for _, p in data:
            with h5py.File(p, "r") as f:
                ys.append((scalar(f, "measurement/instrument/SMS/samY"),
                           scalar(f, "measurement/instrument/SMS/samX"),
                           scalar(f, "measurement/instrument/SMS/samZ")))
        y = [v[0] for v in ys if v[0] is not None]
        r["samY_first"], r["samY_last"] = (y[0], y[-1]) if y else (None, None)
        r["layer_pitch_um"] = (round(1000 * float(np.median(np.abs(np.diff(y)))), 3) if len(y) > 1 else None)   # median: one aborted file must not inflate it
        r["samX"], r["samZ"] = ys[0][1], ys[0][2]
        # dark: which group holds it, in the dark_before file (the pipeline needs darkLoc per scan)
        if g["dark_before"]:
            with h5py.File(g["dark_before"][0][1], "r") as f:
                cand = {k: med_of(f, f"exchange/{k}", 3) for k in ("dark", "bright", "data")}
            r["dark_med_dark"], r["dark_med_bright"], r["dark_med_data"] = (cand["dark"], cand["bright"], cand["data"])
            nz = [k for k, v in cand.items() if v not in (None, 0.0)]
            r["darkLoc"] = "/exchange/" + nz[0] if len(nz) == 1 else ("AMBIGUOUS:" + ",".join(nz) if nz else "ALL-ZERO")
        # energy and distance as ASSERTED by names
        txt = " ".join([folder.name, stem])
        e = NAME_ENERGY.search(txt)
        r["energy_from_name_keV(ASSERTED)"] = e.group(1).replace("p", ".") if e else None
        d = NAME_LSD.search(txt)
        r["lsd_from_name_mm(ASSERTED)"] = d.group(1) if d else None
        rows.append(r)
    return rows


def per_file_rows(folder: Path, label=None):
    """One row per h5 file: the layer table. The aero readback span is the REAL sweep (scan_parameters can be stale)."""
    rows = []
    for p in sorted(folder.glob("*.h5")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        r = {"folder": label or folder.name, "file": p.name, "kind": m["kind"] or "data", "nr": int(m["nr"]),
             "size_GB": round(p.stat().st_size / 1e9, 3)}
        try:
            with h5py.File(p, "r") as f:
                r["frames"] = f["exchange/data"].shape[0]
                a = np.asarray(f["measurement/instrument/SMS/aero"][()]).ravel()
                r["aero_first"], r["aero_last"] = float(a[0]), float(a[-1])
                r["aero_span"] = float(a.max() - a.min())
                for k in ("start", "end", "step"):
                    r[f"scan_{k}"] = scalar(f, f"measurement/process/scan_parameters/{k}")
                for k, path in (("samY", "SMS/samY"), ("samX", "SMS/samX"), ("samZ", "SMS/samZ")):
                    r[k] = scalar(f, f"measurement/instrument/{path}")
                r["acq_time_s"] = scalar(f, "WM/AcqTime")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
        rows.append(r)
    return rows


def write_csv(rows, out):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    rows, frows = [], []
    for d in sorted([root] + [p for p in root.iterdir() if p.is_dir()]):
        if pat and d != root and not pat.search(d.name):
            continue
        try:
            lab = "(loose files in the root folder)" if d == root else None
            rows += survey_folder(d, lab)
            frows += per_file_rows(d, lab)
        except Exception as e:
            print(f"SKIP {d.name}: {type(e).__name__}: {e}", file=sys.stderr)
    write_csv(rows, out)
    perfile = out.with_name(out.stem + "_files.csv")
    write_csv(frows, perfile)
    print(f"{len(rows)} scans -> {out}\n{len(frows)} files -> {perfile}")
    keep = ("folder", "n_layers", "frames", "aero_first", "aero_last", "scan_step", "detY_mm", "slit_DS_H", "slit_DS_V",
            "layer_pitch_um", "acq_time_s", "darkLoc", "dark_med_bright", "data_med_mid", "energy_from_name_keV(ASSERTED)")
    print("\t".join(k.replace("(ASSERTED)", "*") for k in keep))
    for r in rows:
        print("\t".join("" if r.get(k) is None else (f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])) for k in keep))
    print("* = taken from a name, NOT recorded in the file (the Varex file does not record it)")


if __name__ == "__main__":
    main()
