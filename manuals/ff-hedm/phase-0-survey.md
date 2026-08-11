# Phase 0 — Environment and survey

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 0b. Survey the data folder — write `SURVEY.md` before promising anything

**Goal: a written `SURVEY.md` in your work directory answering *what is actually here*,
with every number read from the files, never from a folder or file name.** §4a exists
because a file named `..._96keV_...` held a 95.0 keV scan.

The script below reads metadata only, so it is cheap on a full beamtime. It dumps the
**actual** HDF5 layout first — §3c documents what that layout usually is, but confirm it
rather than assuming it:

```bash
python utils/ff_survey.py <data-dir> [<metadata-dir>]
```

Record, per file:

| field | how to get it | why it matters |
|---|---|---|
| kind (sweep / dark / calibrant) | name + frame count, confirmed against the par file | decides what each file is *for*; nothing downstream works if this is wrong |
| frame count | `exchange/data.shape[0]` | must match the par image range (§3b); off by one means `SkipFrame` (§3e) |
| image dataset path | `visititems` dump, **not** an assumption | goes into `dataLoc`/`darkLoc` (§3d) |
| its dark | `dark_before_<N-1>` for data `<N>` (§3d) | the single highest-cost trap in this path |
| energy | `instrument/HEM/Energy`, cross-checked twice | **never the filename** (§4a) |
| `DetZ` | `instrument/DMS/DetZ` | an `Lsd` **seed** only; was +181 mm off here (§4b) |
| ω sweep bounds and step | par fields 10, 11, 17 | negate for `aero` (§2) |
| is a calibrant present? | classification above | **if not, stop** — there is no geometry without one (§5) |

**Do not derive anything from a folder name.** A companion pipeline lost a factor of 2 in
area this way: a folder called `10x10um_0p25umStepSize` was measured from the stage
coordinates as 20.000 µm × 14.142 µm, because the sample sat at 45° to the beam
(`LaueMatching/scripts/pipeline/Laue_Handbook.md`, Phase 0). The same discipline applies
here to energy, distance, frame count and step.

**Is the scan still being written?** Count the files twice, 120 s apart. Never reconstruct
a sweep that is still growing.

---

## 0c. Already processed? Check before recomputing

A previous run leaves these flat in the result directory. Their presence means a stage
already ran — and §7 will silently resume off them:

| artifact | means |
|---|---|
| `<stem>.MIDAS.zip` | `zip_convert` ran. **Check the dark is non-zero (§3d)** before reusing it |
| `Temp/AllPeaks_PS.bin` | the peak search ran — at *some* threshold, not necessarily yours |
| `InputAll.csv`, `Spots.bin`, `Data.bin`, `nData.bin` | transforms + binning ran |
| `Output/IndexBest_all.bin` + `IndexKey_all.bin` | indexing ran — the **consolidated** family, written by both the python and c-omp backends |
| `Output/IndexBest.bin` + `IndexBestFull.bin` | indexing ran — the **legacy** pair, written only by the classical C `IndexerOMP`. Both families are recognised since `8a594ea5` (§11) |
| `Grains.csv`, `SpotMatrix.csv` | a full reconstruction exists |
| `midas_state.h5` | per-stage provenance — **read this** rather than guessing which stages ran |

**After changing any peak-search or dark parameter, delete `results/` entirely** (§7).
Resume is silent and costs 0.3 s where a real run costs 55 s, so an inherited result is
easy to mistake for a fast one.

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

All APS hosts share `/home/beams*`. conda is **not** on the non-interactive ssh PATH, so
call the shared env by full path:

```bash
/home/beams12/S1IDUSER/opt/envs/midas/bin/python
```

GPU prefix: `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE`. Pick a GPU by
**utilisation**, not free memory.

| Host | GPU | Note |
|---|---|---|
| chiltepin | driver dead | **only host with internet — install here** |
| copland | 2× A6000, 96 cores | general workhorse; jump host for toro/shannon |
| alleppey | 4× H100 | |
| sentosa | 2× H200 + 2× RTX PRO 6000 | most GPU memory |
| chutoro | 2× A6000, 64 cores | no internet |

**The shared env is not complete.** Verified 2026-07-30: `matplotlib` and `scikit-image`
were both absent, and `scikit-image` is a hard requirement of the v2 auto-seeder
(`midas_calibrate_v2/seed/auto_seed.py:523`). Install from chiltepin:

```bash
ssh chiltepin '/home/beams12/S1IDUSER/opt/envs/midas/bin/pip install matplotlib scikit-image'
```

Long jobs need `setsid`/`nohup` + a redirect or they die on SSH hangup. Write scripts to a
file and `scp` them; do not inline `cat > file && python &`.

Outputs go under the beamtime's own `analysis/` tree, e.g.
`/gdata/dm/1ID/<year>/<beamtime>/analysis/<task>/`. **Never leave results in `/tmp`.**

### 1b. On your own machine or cluster

**Do not `pip install midas-suite[ff]`.** That extra is wrong for this runbook in two ways
at once, both silent:

1. It pulls **`midas-ff-pipeline`**, which this document deprecates (§7), and **not**
   `midas-pipeline` — so `midas-pipeline run --scan-mode ff`, the command §7 tells you to
   run, is not installed at all.
2. `midas-ff-pipeline`'s own dependency list floors `midas-fit-grain` at **0.6.0** and
   `midas-process-grains` at **0.6.1** — the versions *below* the silent-wrong-answer
   fixes (§0). A clean install from that extra reproduces both bugs.

Install the orchestrator this runbook actually uses, which carries the correct floors:

```bash
pip install "midas-pipeline>=0.8.2" \
            "midas-calibrate-v2>=0.5.3" \
            matplotlib scikit-image
```

`midas-pipeline >= 0.8.2` pulls `midas-peakfit`, `midas-transforms`, `midas-index`,
`midas-fit-grain>=0.7.0`, `midas-process-grains>=0.7.0`, `midas-zipper>=0.1.5`,
`midas-hkls`, `midas-stress` and `midas-diffract` transitively. **0.8.2 is the floor that
matters** — it is the first version declaring the zipper floor, without which the peak
search silently runs at the defaults (§0). `midas-calibrate-v2` is **not** pulled by the
orchestrator and is needed for §5.
`scikit-image` is a hard requirement of the v2 auto-seeder
(`midas_calibrate_v2/seed/auto_seed.py:523`); `matplotlib` is needed to produce the
mandatory ring overlay (§5d).

**Then run two checks and read their output.** `pip install` exiting 0 tells you nothing.

```bash
git clone https://github.com/marinerhemant/MIDAS.git   # for the pyproject floors + utils/
cd MIDAS
# 1. version floors — the §0 script
# 2. the bundled c-omp indexer actually shipped:
python -c "
from midas_index import backend_c as b
print('c indexer available:', b.available())
print('binary:', b.binary_path())"
```

`available()` must print `True`. It resolves `midas_index/bin/midas_indexer` **inside the
installed package** (`backend_c.py:47-71`), so it is present in a wheel and **absent in a
plain source checkout** — where the binary is built under `build/<platform>/` instead. If
you are running from a clone rather than an install, you will fall back to the slow Python
indexer without being told.

**No GPU?** Every stage runs on CPU: pass `--device cpu` instead of `--device cuda` in §7,
and expect the peak search to dominate. Nothing in this runbook requires CUDA — the c-omp
indexer and refiner are OpenMP, and they are the **preferred** fast path (header).

**Working directory.** Put results next to the data, or in a project directory you own.
**Never leave results in `/tmp`** — on a shared cluster they are also visible to others.

---
