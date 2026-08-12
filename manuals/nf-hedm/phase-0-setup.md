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

> **The shared env is BEHIND this repo tree, and it still carries the `GridPoints`
> off-by-one.** As of **2026-08-07** `packages/*/pyproject.toml` reads
> `midas_nf_pipeline 0.6.0`, `midas_nf_preprocess 0.6.0`, `midas_nf_fitorientation 0.8.0`,
> `midas_hkls 0.7.0` — several releases past the env contents quoted above, which were
> last read on 2026-08-01. **Treat every version number in this section as a worked
> example of the check, not as current state: re-run the command below.** The gap was
> last measured functionally on
> 2026-08-01, not inferred from version strings — the installed
> `midas_nf_fitorientation/params.py` reads `args[4,5,7,8,9,10]`, i.e. the **broken**
> indices (lab notebook defect 10); the fixed tree reads `args[3,4,6,7,8,9]`. Also absent
> from the installed copies: the scipy labeller in `process_images`, `--fit-gpus`, and the
> `hard` multipoint objective. The packages are ordinary copies under `site-packages`, not
> editable installs, and `~s1iduser/opt/MIDAS_canonical` sits on an unrelated HEAD
> (`d3a55ca`) that does not contain `b95c38c0`/`d231fdf3` — so nothing on that host picks
> these fixes up implicitly. **Any multipoint refinement run in that env is invalid.**
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
pip install "midas-nf-pipeline>=0.6.1" "midas-hkls>=0.6.0" matplotlib
```

`midas-nf-pipeline >= 0.6.1` pulls `midas-nf-preprocess>=0.6.0` and
`midas-nf-fitorientation>=0.8.0` transitively, which is what keeps `SumFrames` consistent
(§8j). **Below 0.6.1 you must pin those two by hand:** 0.6.0's metadata floored them at
0.4.0 and 0.6.0, so a plain `pip install midas-nf-pipeline` could resolve a mix where one
package reads `NrFilesPerDistance`/`OmegaStep` as raw and another as post-sum — the resolve
`9450901d` was written to prevent, which its own dependency list did not yet enforce.

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
