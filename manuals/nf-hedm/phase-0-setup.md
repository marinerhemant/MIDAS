# Phase 0 — Environment

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 1. Environment

**First, work out which situation you are in:**

```bash
if [ -d /home/beams12/S1IDUSER/opt/envs/midas ]; then
  echo "APS beamline host — use §1a"
else
  echo "your own machine or cluster — use §1b"
fi
```

### 1a. On an APS beamline host

**On the APS hosts** (chiltepin, copland, alleppey, sentosa, chutoro — all share
`/home/beams*`), call by full path; conda is not on the non-interactive ssh PATH:

```bash
/home/beams12/S1IDUSER/opt/envs/midas/bin/python
```

Contents re-checked 2026-08-01 with `importlib.metadata` on chiltepin: `midas-nf-pipeline
0.1.1`, `midas-nf-preprocess 0.1.2`, `midas-nf-fitorientation 0.3.2`, `midas-hkls 0.5.0`,
`numpy 2.4.6`, `tifffile 2026.3.3`, `h5py 3.16.0`, `scipy 1.17.1`, `torch 2.11.0+cu128`.

> **This paragraph is a worked example of the check, and its verdict is out of date.**
> Measured **2026-08-28**, the shared env is **current** — all four packages match the
> tree, `importlib.metadata` and `__version__` agree, and the HDF5 capability is present.
> `RUNBOOK.md` §R1a has that measurement. What survives here is the *method*, because the
> failure it describes is real and will recur. Read on for that, not for the versions.
>
> **[Historical, 2026-08-07] The shared env was BEHIND this repo tree, and still carried
> the `GridPoints` off-by-one.** As of **2026-08-07** `packages/*/pyproject.toml` read
> `midas_nf_pipeline 0.6.0`, `midas_nf_preprocess 0.6.0`, `midas_nf_fitorientation 0.8.0`,
> `midas_hkls 0.7.0` — several releases past the env contents quoted above, which were
> last read on 2026-08-01. **Treat every version number in this section as a worked
> example of the check, not as current state: re-run the command below.** The gap was
> last measured functionally on
> 2026-08-01 — at that time the installed `midas_nf_fitorientation/params.py` read
> `args[4,5,7,8,9,10]`, the **broken** indices (lab notebook defect 10), against the fixed
> tree's `args[3,4,6,7,8,9]`.
>
> **That is no longer true, and how it stopped being true is the point.** Re-checked
> 2026-08-12: the installed copy already has the fixed indices, `--fit-gpus`, the scipy
> labeller and the `hard` objective — while `importlib.metadata` and `__version__` both
> still report the old versions, and agree with each other. The files were patched in
> place on an **ordinary, non-editable** install and the version strings were never
> bumped. **Version numbers on that env are therefore meaningless in both directions**,
> and no functional claim in this paragraph should be believed without re-checking.
> Before any remote run, either re-check the env
>
> ```bash
> ssh chiltepin '/home/beams12/S1IDUSER/opt/envs/midas/bin/python -c "
> import importlib.metadata as m
> [print(p, m.version(p)) for p in [\"midas-nf-pipeline\",\"midas-nf-preprocess\",
>                                   \"midas-nf-fitorientation\",\"midas-hkls\"]]"'
> ```
>
> or overlay the tree with `PYTHONPATH` and say in the write-up which you used. Install on
> chiltepin (only host with internet); the shared home makes it visible everywhere.

**Do not trust the version numbers in this document. Trust the tree.** Run this from the
repo root; it takes the **strictest** floor any package declares, so one stale dependency
list cannot weaken the check:

```bash
python - <<'PY'
import importlib, importlib.metadata as m, re, pathlib
def vt(s): return tuple(int(x) for x in re.findall(r'\d+', s)[:3])
floors = {}
for p in pathlib.Path("packages").glob("*/pyproject.toml"):
    for pkg, need in re.findall(r'"(midas-[a-z0-9-]+)(?:\[[\w,]+\])?>=([0-9][0-9.]*)"',
                                p.read_text()):
        if vt(need) > vt(floors.get(pkg, "0")): floors[pkg] = need
bad, drift = [], []
for pkg, need in sorted(floors.items()):
    try: meta = m.version(pkg)                       # what pip recorded
    except m.PackageNotFoundError: continue
    try: code = getattr(importlib.import_module(pkg.replace("-", "_")),
                        "__version__", None)         # what actually imports
    except Exception: code = None
    eff = max([v for v in (meta, code) if v], key=vt)
    if code and vt(code) != vt(meta):
        drift.append(f"{pkg:26} dist-info {meta:8} code {code:8} <- editable, stale metadata")
    if vt(eff) < vt(need): bad.append(f"{pkg:26} running {eff:8} need >={need}")
print(f"scanned {len(floors)} floors across the tree")
if drift:
    print("\nMETADATA DRIFT (the code is what runs; `pip install -e` refreshes it):")
    [print(" ", d) for d in drift]
print("\n*** BELOW FLOOR ***" if bad else "\nall installed midas packages satisfy the tree")
[print(" ", b) for b in bad]
PY
```

**Then check the capability, not the number.** A version gate cannot tell you whether the
code you need is present, and on this env it has been wrong in both directions. For 20-ID,
the load-bearing capability is the HDF5 frame source — test for it directly:

```python
from midas_nf_preprocess.process_images import io, params, median
import inspect
print([hasattr(io, n) for n in ("is_hdf5", "Hdf5FrameSource",
                               "check_pixel_scale", "open_source")])   # all True
print(hasattr(median, "streaming_temporal_median"))                     # True
src = inspect.getsource(params)
print([k for k in ("extOrig", "DataLoc", "PixelScale", "StreamFrames",
                   "MedianFrames", "MedianRowBlock") if k not in src])  # []
```

> Test for those **names**. A grep for `"h5"` over `dir(io)` returns nothing and means
> nothing — the symbols are spelled `is_hdf5` and `Hdf5FrameSource`, neither of which
> contains that substring. That false negative has already been reported once.

**The gate has a blind spot. Read this before trusting a pass.** The check below compares
metadata against the imported `__version__` and reports disagreement. That catches an
*editable* install, where code runs ahead of stale metadata. It cannot catch the opposite,
which is what was actually found on `copland` on 2026-08-12: an ordinary `site-packages`
install whose **files were hand-patched while the version string stayed put**. Metadata and
`__version__` agree, so by this check's own definition there is no drift, and a below-floor
verdict is returned for code that is already fixed — or, far worse on another day, an
above-floor pass for code that is not. When the verdict matters, diff the installed file
against the commit that fixed the defect; the version string is not evidence.

**Check the code, not just the metadata — an editable install lies about its version.**
`pip install -e` records the version *at install time* and never updates it, so on a
development checkout `importlib.metadata.version()` can report 0.3.0 while the code that
imports is 0.6.0. A gate reading metadata alone fails that tree at step one, for a problem
that does not exist. The reverse is also possible on a stale wheel, which is why the check
takes the **higher** of the two and reports any disagreement rather than silently picking.

**Strictest, not nearest.** The declarations have disagreed before. `midas_suite` floored
`midas-nf-preprocess` at 0.6.0 and `midas-nf-fitorientation` at 0.8.0 while
`midas_nf_pipeline` still admitted 0.4.0 and 0.6.0 — the versions *before* the `SumFrames`
change. `9450901d`'s own message warns that a floor left behind "would let a resolve mix a
package that reads the keys as raw with one that reads them as post-sum", and its own
dependency list permitted exactly that until `midas-nf-pipeline` 0.6.1. Closed now — but
the failure mode is a per-package check against whichever list happens to be weakest, so
scan them all rather than trusting the one nearest to hand.

**This matters more here than the version strings suggest.** In the six days after this
document was written, `SumFrames` **inverted** its unit convention (§8j) and a new
threshold key `BlanketSigma` appeared (§8k). Neither change raises an error if you follow
the old instructions. **When this document and the tree disagree, the tree is right** —
record the discrepancy as a finding about this document rather than working around it.

**`matplotlib` is NOT installed in that env.** Therefore: **reduce remotely, plot
locally.** Write an `.npz` of the reductions on the host, `scp` it to the Mac, plot there.
See §5a for the pattern actually used.

GPU prefix on any of those hosts: `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE`.
Pick a GPU by *utilization*, not free memory. Long jobs: `setsid`/`nohup` + redirect to a
log, or SIGHUP kills them.

### 1b. On your own machine or cluster

```bash
pip install "midas-nf-pipeline>=0.6.6" "midas-nf-preprocess>=0.7.0" \
            "midas-hkls>=0.6.0" matplotlib
```

`midas-nf-pipeline >= 0.6.1` pulls `midas-nf-preprocess` and
`midas-nf-fitorientation>=0.8.0` transitively, which is what keeps `SumFrames` consistent
(§8j). **Below 0.6.1 you must pin those two by hand:** 0.6.0's metadata floored them at
0.4.0 and 0.6.0, so a plain `pip install midas-nf-pipeline` could resolve a mix where one
package reads `NrFilesPerDistance`/`OmegaStep` as raw and another as post-sum — the resolve
`9450901d` was written to prevent, which its own dependency list did not yet enforce.

> **`midas-nf-preprocess>=0.7.0` is pinned explicitly on purpose, and for 20-ID it is not
> optional.** The HDF5 frame source and the row-blocked streaming median first shipped in
> **0.7.0**; below it `extOrig h5` cannot work at all, so the entire 20-ID route (§3h) is
> absent. `midas-nf-pipeline` floored it at `>=0.6.0` until **0.6.6**, which means an
> install of any earlier pipeline release can resolve an env with no HDF5 reader — **and
> the floor gate below will pass it**, because the gate reads the floors out of the tree
> and a floor that is too low makes the gate blind. Drop the explicit pin only once
> `midas-nf-pipeline >= 0.6.6` is what you are installing.

Then **run the floor gate above** and read its output. It is the check that catches this;
do not infer a good install from `pip install` exiting 0.

**Seed cache.** The orchestrator re-derives the cache path from the *install* directory
(`from_cache.py:106`), which in a conda env resolves to a `NF_HEDM/seedOrientations` that
does not exist — it then dies with `SeedCacheNotFound` **after** writing `hkls.csv`, so the
run looks like it started fine (§8a). Set it explicitly:

```bash
export MIDAS_NF_SEED_DIR=<path-to-MIDAS-checkout>/NF_HEDM/seedOrientations
```

**No GPU?** Pass `--device cpu`. Note that `--device` was silently dropped by two of three
call sites before `2719f322` — on an install below that fix, a CPU request still ran the
reduction on CUDA and surfaced as an OOM from a stage with no reason to be on the GPU. The
floor gate catches this.

**Working directory.** Results go in a project directory you own — **never `/tmp`**.

---
