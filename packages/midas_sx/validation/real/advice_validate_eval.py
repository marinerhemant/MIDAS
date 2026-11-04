"""Apply the registered pass conditions (PREREG_advice.md) to the scenario reports written by advice_validate.py. Imported by `advice_validate.py eval`."""
import glob, json, os, pickle

SEV = {"note": 0, "limiting": 1, "blocking": 2}
COVERAGE_IDS = {"completeness", "redundancy", "next_setting", "direction_holes"}
ALLOWED = {"C": COVERAGE_IDS, "C6": set(), "C6_dmin": set(), "C6hi": set(), "C6hi_dmin": set(), "D1": COVERAGE_IDS, "D2": COVERAGE_IDS, "D3": COVERAGE_IDS, "D4_unset": COVERAGE_IDS | {"resolution_detector_limited", "resolution_signal_limited"},
           "D4_set": COVERAGE_IDS | {"resolution_detector_limited", "resolution_signal_limited"}, "D4_set40": COVERAGE_IDS | {"resolution_detector_limited", "resolution_signal_limited"},
           "D5": COVERAGE_IDS | {"scan_range"}, "D6": COVERAGE_IDS, "D7": COVERAGE_IDS, "D8": COVERAGE_IDS, "D8b": COVERAGE_IDS}


def _load(rep_dir, name):
    p = os.path.join(rep_dir, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _f(rep, fid):
    for f in rep["findings"]:
        if f["id"] == fid:
            return f
    return None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _false_positives(rep, name, named):
    ok_ids = set(named) | ALLOWED.get(name, set())
    return [f["id"] + ":" + f["severity"] for f in rep["findings"] if SEV[f["severity"]] >= 1 and f["id"] not in ok_ids]


def _wedges(work, name):
    from advice_validate import SEEDS  # noqa: F401  (same directory)
    out = {}
    for p in sorted(glob.glob(os.path.join(work, "red", f"{name}_*.pkl"))):
        red = pickle.load(open(p, "rb"))
        out[os.path.basename(p)[:-4]] = float(red.geom.wedge_deg)
    return out


def evaluate(rep_dir):
    work = os.path.dirname(rep_dir)
    rows = []
    P = "H_" if os.environ.get("ADVICE_HI") else ""

    def add(scn, check, expected, fired, ok, note=""):
        rows.append((scn, check, expected, fired, "PASS" if ok else "FAIL", note))

    r = _load(rep_dir, P + "C")
    if r is not None:
        fp = _false_positives(r, "C", [])
        add("C", "no blocking/limiting (quality, scan_range, resolution)", "none", ", ".join(fp) or "none", not fp, "coverage ids (completeness, ...) reported not judged: " + ", ".join(f["id"] + ":" + f["severity"] for f in r["findings"] if f["id"] in COVERAGE_IDS))
    w = _wedges(work, P + "Cw")
    if w:
        add("Cw", "refined wedge on clean frames < 0.1 deg", "< 0.1", str({k: round(v, 3) for k, v in w.items()}), all(abs(v) < 0.1 for v in w.values()))
    r = _load(rep_dir, P + "Cw")
    if r is not None:
        g = _f(r, "geometry_consistency")
        add("Cw", "geometry_consistency not limiting", "note or none", "none" if g is None else g["severity"], g is None or g["severity"] == "note")
    r = _load(rep_dir, P + "C6")
    if r is not None:
        fp = _false_positives(r, "C6", [])
        add("C6", "no blocking/limiting finding from ANY check", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "C6hi")
    if r is not None:
        fp = _false_positives(r, "C6hi", [])
        add("C6hi", "no blocking/limiting finding from ANY check (6 runs, 100 x counts)", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "C6_dmin")
    if r is not None:
        lim = [f["id"] for f in r["findings"] if f["severity"] in ("limiting", "blocking") and f["id"] in COVERAGE_IDS | {"resolution_signal_limited"}]
        add("C6_dmin", "NEGATIVE control (low counts, target d_min 0.66): coverage/resolution still limiting", "limiting", ", ".join(lim) or "none", bool(lim), "the low-count scene is signal-limited")
    r = _load(rep_dir, P + "C6hi_dmin")
    if r is not None:
        fp = _false_positives(r, "C6hi_dmin", [])
        add("C6hi_dmin", "no blocking/limiting finding from ANY check (6 runs, 100 x counts, target d_min 0.66)", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D1")
    if r is not None:
        d = _f(r, "intensity_drift")
        ok = d is not None and d["severity"] == "limiting"
        pl = _num(d["measured"].get("placebo_gain")) if d else float("nan")
        rg = _num(d["measured"].get("rint_gain")) if d else float("nan")
        add("D1", "intensity_drift limiting", "limiting", "absent" if d is None else d["severity"], ok)
        add("D1", "placebo gain < half the real gain", "< 0.5 x gain", f"placebo {pl:.3f} real {rg:.3f}", ok and pl < 0.5 * rg)
        fp = _false_positives(r, "D1", ["intensity_drift"])
        add("D1", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D2")
    if r is not None:
        sat = [f for f in r["findings"] if f["id"] == "dynamic_range" and "saturated" in f["title"]]       # Amendment 7: the clipping test, not the 25 %-of-cutoff note
        add("D2", "dynamic_range 'saturated' finding, limiting (Amendment 7)", "limiting", ", ".join(f["severity"] for f in sat) or "absent", bool(sat) and sat[0]["severity"] == "limiting")
        fp = _false_positives(r, "D2", ["dynamic_range"])
        add("D2", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D3")
    if r is not None:
        d = _f(r, "omega_sampling")
        sf = _num(d["measured"].get("single_frame_fraction")) if d else float("nan")
        add("D3", "omega_sampling present with single_frame_fraction >= 0.25 (Amendment 7)", ">= 0.25", "absent" if d is None else f"{d['severity']}, single-frame fraction {sf:.3f}", d is not None and sf >= 0.25)
        fp = _false_positives(r, "D3", ["omega_sampling"])
        add("D3", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "C")
    if r is not None:
        bad = [f["id"] for f in r["findings"] if (f["id"] == "dynamic_range" and "saturated" in f["title"]) or f["id"] == "omega_sampling"]
        add("C", "clean control: no 'saturated' and no omega_sampling finding (Amendment 7)", "none", ", ".join(bad) or "none", not bad)
    for nm, want_sev in (("D4_unset", "limiting"), ("D4_set", "note"), ("D4_set40", "note")):
        r = _load(rep_dir, P + nm)
        if r is None:
            continue
        d = _f(r, "scan_range")
        rngs = [_num(v["range_deg"]) for v in d["measured"]["per_run"].values()] if d else []
        add(nm, f"scan_range {want_sev}, range 40 +- 1 deg", want_sev, "absent" if d is None else f"{d['severity']}, ranges {rngs}", d is not None and d["severity"] == want_sev and all(abs(x - 40) <= 1 for x in rngs))
        fp = _false_positives(r, nm, ["scan_range"])
        add(nm, "no unexpected limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D5")
    if r is not None:
        cr = [f for f in r["findings"] if f["id"] in ("completeness", "redundancy") and f["severity"] == "limiting"]
        add("D5", "completeness or redundancy limiting", "limiting", ", ".join(f["id"] for f in cr) or "none", bool(cr))
        fp = _false_positives(r, "D5", ["completeness", "redundancy"])
        add("D5", "no unexpected limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D6")
    w = _wedges(work, P + "D6")
    if r is not None:
        g = _f(r, "geometry_consistency")
        add("D6", "geometry_consistency present", "note or limiting", "absent" if g is None else g["severity"], g is not None)
        add("D6", "refined wedge in 0.3..0.5 deg for every run", "0.3-0.5", str({k: round(v, 3) for k, v in w.items()}), bool(w) and all(0.3 <= abs(v) <= 0.5 for v in w.values()))
        fp = _false_positives(r, "D6", ["geometry_consistency"])
        add("D6", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D7")
    if r is not None:
        u = _f(r, "unexplained_strong_blobs")
        rings = [_num(x["tth_deg"]) for x in u["measured"].get("rings", [])] if u else []
        near = [x for x in rings if abs(x - 12.0) <= 0.3]
        add("D7", "contamination names a ring within 0.3 deg of 2theta = 12.0", "ring at 12.0", f"rings {rings}", bool(near))
        fp = _false_positives(r, "D7", ["unexplained_strong_blobs"])
        add("D7", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D8")
    if r is not None:
        s = _f(r, "systematic_floor"); d = _f(r, "intensity_drift")
        add("D8", "systematic_floor limiting", "limiting", "absent" if s is None else s["severity"], s is not None and s["severity"] == "limiting")
        add("D8", "intensity_drift NOT limiting", "not limiting", "absent" if d is None else d["severity"], d is None or d["severity"] != "limiting")
        fp = _false_positives(r, "D8", ["systematic_floor"])
        add("D8", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    r = _load(rep_dir, P + "D8b")
    if r is not None:
        s_ = _f(r, "systematic_floor"); d = _f(r, "intensity_drift")
        add("D8b", "systematic_floor limiting (per-frame sigma 0.25)", "limiting", "absent" if s_ is None else s_["severity"] + f", floor {_num(s_['measured'].get('floor_f')):.3f}", s_ is not None and s_["severity"] == "limiting")
        add("D8b", "intensity_drift NOT limiting", "not limiting", "absent" if d is None else d["severity"], d is None or d["severity"] != "limiting")
        fp = _false_positives(r, "D8b", ["systematic_floor"])
        add("D8b", "no other limiting finding", "none", ", ".join(fp) or "none", not fp)
    lines = ["| scenario | condition | expected | fired | result | note |", "|---|---|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} | **{e}** | {n} |" for a, b, c, d, e, n in rows]
    txt = "\n".join(lines)
    open(os.path.join(rep_dir, "table.md"), "w").write(txt + "\n")
    print(txt)
    by = {}
    for a, b, c, d, e, n in rows:
        by.setdefault(a.split("_")[0], []).append(e == "PASS")
    print("\nPER SCENARIO:", {k: ("CONFIRMED" if all(v) else "REFUTED") for k, v in by.items()})
