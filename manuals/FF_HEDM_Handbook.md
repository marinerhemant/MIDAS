# FF-HEDM Reconstruction Runbook — survey → calibrate → reconstruct → report

**Use this doc to start a fresh session on a far-field dataset this pipeline has never
seen.** Paste it in together with `FF_HEDM_Lab_Notebook.md`, then give three lines:

```
Data folder:     <ABSOLUTE PATH>     # the image tree, e.g. /gdata/dm/1ID/<year>/<bt>/data/ge5/
Metadata folder: <ABSOLUTE PATH>     # or "find it" — see §0b
Sample material: <e.g. gold cubes / Ti-6Al-4V / unknown, tell me from the data>
```

Everything else the agent works out or asks for. **The order in §0a is not optional** —
each step produces an input the next one needs, and two of them (§2, §3e) cannot be
checked after the fact.

**Scope.** Every recipe here is **1-ID, a single monolithic GE panel, DM-converted
`.ge5.h5`, one layer.** Multi-panel (GE1–4) and multi-layer scans are *not* covered:
`cross_det_merge` appears in this document only as a no-op. If your data differs in
detector count, file format, or layer count, **stop and ask** rather than adapting a
recipe — the §3 field maps and §5 calibration assume this configuration throughout.

Not a tutorial. Follow the steps in order; each one names the file to read, the command to
run, the field to look at, and the branch to take.

Citations are `path:line` relative to `$MIDAS = /Users/hsharma/opt/MIDAS`. Read them with
absolute paths. Every non-obvious claim carries one. Claims that are convention, or that
could not be verified, are flagged inline and summarised in §11. **Do not promote a §11
item to a fact.**

**`FF_HEDM_Lab_Notebook.md` is the companion to this file.** This document is the
procedure — what to do, in order. The notebook is the evidence: how each defect was found,
the measurements behind every number here, and the claims that had to be *retracted*.
Read the notebook before re-investigating anything in this pipeline; several attractive
hypotheses are recorded there as refuted, and one of them (Lab Notebook §4c) will actively
damage a reconstruction if it comes back — it invites the change hard rule 9 forbids.

Sibling document: `NF_HEDM_Handbook.md`. The two share §2 (ω sign), §3 (metadata) and §4
(energy/distance) almost verbatim, because those are properties of the *beamline*, not of
the modality. Where FF differs, it is called out.

Maintained code = `midas_zipper`, `midas_calibrate_v2`, `midas_peakfit`,
`midas_transforms`, `midas_index`, `midas_fit_grain`, `midas_process_grains`, and the
orchestrator `midas_pipeline`. **The version floors are not cosmetic** — below them the
pipeline is not reproducible, `GrainRadius` is ~5× wrong, the refiner silently returns
its input, and the grain-selection keys you wrote are discarded (§0). `FF_HEDM/` is
soft-deprecated C; only its example parameter files are used here.
**The bundled c-omp binaries in `midas_index/bin` and `midas_fit_grain/bin` are the
preferred fast path** — "deprecated C source" does not mean "don't use the c-omp indexer."

---

## STOP — read this before touching anything

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** The failures in this pipeline
do not feel like being stuck — a mirrored reconstruction from a wrong ω sign produces a
clean grain list with normal completeness; a wrong ring assignment converges beautifully;
`DetZ`-as-`Lsd` gives an 11 % geometry error that still fits; the refiner returns its seed
positions and reports success. In each case the run finishes and looks right.

So the trigger is not confusion. **Halt on these named conditions, whether or not anything
seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| par field 9 is **not** `aero` | no other value's ω sign has ever been established here (§2, §11) |
| the data is not 1-ID / single GE panel / DM-`.ge5.h5` / one layer | every field map and geometry recipe below assumes it (header, §3) |
| **no calibrant** file in the folder | there is no geometry without one, and `DetZ` is not a substitute (§0b, §4b) |
| any package **below floor** after §0 | three of them produce plausible wrong answers, not errors (§0) |
| calibrant strain **> 100 µε** after §5 | hard gate; a converged fit above it is not usable (rule 6) |
| the ring overlay does not match the frame | the fit is on the wrong rings; nothing downstream can detect it (§5d) |
| the dark reads **all zero** in the zarr | every threshold returns 0 peaks; tuning `RingThresh` cannot fix it (§3d) |
| `nFrames` ≠ logged frames − `SkipFrame` | something is skipping twice or not at all; ω is shifted either way (§3e) |
| grain positions **pile up** at ±`Rsample` or ±`Hbeam`/2 | the envelope is binding — and the fix is forbidden to you by rule 9 (§6, §8b) |
| this document and the tree **disagree** | report it; do not work around it (§0) |

When you halt, say which row fired, what you measured, and what you would need in order to
proceed. Everything not blocked by it should still be finished first.

### Hard rules

1. **Determine the ω sign convention first (§2).** Field 9 of `<beamtime>_FF.par`. If it
   reads `aero`, the stage turns **clockwise** and **ω_MIDAS = −ω_logged**: negate
   `OmegaStart` *and* `OmegaStep`. Get this wrong and the reconstruction is **mirrored**,
   which is **not detectable from the grain list**. Step 1 of every new dataset, no
   exceptions.
2. **On the 1-ID GE detector, frame 0 of every acquisition is a throwaway — set
   `SkipFrame 1` (§3e).** This applies to the FF data sweep, the dark, and the calibrant.
   It is required for *correctness*, not tidiness: the frame is real data taken before the
   stage settled, and if the sweep is defined off the raw frame count it also shifts ω by
   one step for every subsequent frame. **Scope: GE / far-field only. Do NOT carry this to
   the near-field detector** — on an NF `DoVolume`/`DoLayer` scan the spare file is a
   *trailing* ω-wrap frame, not a leading throwaway, and `StartNr` is the first image
   (`NF_HEDM_Handbook.md` §3g).
3. **`DetZ` is not `Lsd` (§4b).** The detector-stage readback carries an arbitrary zero
   offset. On `bt_1id_jul26` the offset was **+181 mm** on a 1666 mm distance — 11 %.
   Only *differences* between `DetZ` readbacks are trustworthy. Lsd comes from the
   calibrant, always.
4. **The filename is not the energy (§4a).** `bt_1id_jul26` wrote
   `CeO2_..._96keV_000001.ge5.h5` for a scan taken at **95.0 keV**. Three instrument
   records agreed on 95 and the string was stale. Read `instrument/HEM/Energy` from the
   HDF5, cross-check `fastsweep_Emon.txt` field 6 and the spec log's `Energy (keV):`.
5. **Never trust a calibration you have not overlaid on the image (§5d).** A converged fit
   with a good strain number can still sit on the wrong ring assignment. Overlay predicted
   rings on the measured frame and look at it.
6. **Report calibrant strain as median and 5 %-trimmed, not just mean (§5c).** The mean is
   dominated by a handful of bad fits. **Hard gate: no calibrant geometry above 100 µε
   goes downstream.**
7. **Units: µm, degrees, Å** (Å for wavelength and lattice parameters only). Output Euler
   angles are **radians**.
8. **`Lsd` and `Wavelength` are a *pair*.** They are near-degenerate against a single
   powder pattern (§5e). Whatever λ you fit at, downstream must use the *same* λ. A 1 %
   λ error mostly cancels in relative strain but scales absolute lattice parameters by 1 %.
9. **`Rsample` and `Hbeam` are a SEARCH BOUND, never the sample size (§6).** Setting them
   to the true dimensions plops grains onto the bounding-box edges, manufacturing a
   pile-up of positions at ±`Rsample` and ±`Hbeam`/2. Keep the generous defaults. If
   indexing is slow, suspect the binned-file format (Lab Notebook §3b), never the envelope.
10. **Check what the pipeline actually skipped.** `midas-ff-pipeline` no-ops stages that
   don't apply; a silent no-op and a silent failure look identical in the log tail. Read
   the per-stage provenance in `<result>/LayerNr_N/midas_state.h5`.

The first ten rules are about distrusting the *data*. These four are about distrusting
your own run, and they are the ones a context-free session skips:

11. **Suspect success.** Almost every defect found in this pipeline **reported success**:
    the refiner returned its input and converged; the calibration loop shipped its worst
    iterate; `process-grains` discarded your `Completeness` and produced 4× the grains;
    the peak fit was non-deterministic and every run looked fine. "It ran" is not
    evidence. Ask what the stage would look like if it had silently no-opped, then check
    that specific thing (§0c, §7).
12. **Debug your own configuration before the data, the indexer, or the physics.** When a
    result looks wrong, the order is: a version below floor (§0) → a key that was dropped
    or misspelled (§0c, §6c) → a sign or unit convention (§2, §7) → a stale resumed stage
    (§7) → only then the sample. Lab Notebook §4c records three attractive physical
    hypotheses that were **refuted** after the mundane cause was found.
13. **Never take a number from a name.** Not the energy from a filename (§4a), not the
    distance from a folder, not the frame count from a `DoVolume` argument. Read it from
    the file, and say which file in the report.
14. **Do not reimplement what a `midas_*` package already does.** §6b carries a worked
    example of the cost: a hand-rolled ring-band mask shared **13.4 %** of its pixels with
    the production band and manufactured a "background varies by 20σ" result that had to
    be retracted. Ring overlay → `midas_integrate.geometry`; orientations and
    misorientation → `midas_stress`; structure factors → `midas_hkls`; image reading →
    `midas_calibrate_v2.io.readers.read_image`.

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `aero` ω sign | mirrored microstructure, plausible completeness | §2 |
| first frame kept | one bad frame + every ω off by one step | §3e |
| "fixing" the zipper to skip the first file too | double skip — 1440 frames become 1439 | §3e |
| `DetZ` used as `Lsd` | 11 % geometry error that still "converges" | §4b |
| energy taken from the filename | 1 % λ error → 1 % `Lsd`, wrong absolute lattice parameter | §4a |
| dark read from `exchange/dark` | that group does not exist in DM files; dark silently all-zero | §3d |
| calibrant fit accepted on strain alone | wrong ring assignment fits beautifully | §5d |
| E↔M loop returns its LAST iterate | ships a worse geometry than the run found (72 vs 18 µε) | §5c |
| ring-ratio check skipped | innermost ring mis-assigned; Lsd off by a ring-spacing factor | §5b |
| `ImTransOpt` differs between calibration and recon | geometry mirrored relative to the fit | §6 |
| lattice constant left as the calibrant's | CeO₂ rings predicted for a gold sample | §6 |
| `Rsample`/`Hbeam` set to the REAL sample size | grains plop onto the bounding-box edges — an artefactual pile-up at ±Rsample, ±Hbeam/2 | §6 |
| residual-correction map applied when it made strain worse | v2 discards it automatically — check it did | §5c |
| `darkLoc` left unset (only `darkDataset` set) | all-zero dark → 0 peaks on every frame, invariant to `RingThresh` | §3d |
| `RingThresh` copied from a template | strict size filter shaves spots to single pixels → 0 peaks | §6b |
| `FitRMSE` used as a spot-quality cut | it is an **absolute** residual — cuts the brightest, most certainly-real spots first (58 % of indexed spots) | §6c |
| ω multiplicity (`NImgs`) used as a spot-quality cut | encodes mosaicity, not reality — deletes real single-frame spots, worst for small/undeformed grains | §6c |
| stale `midas-fit-grain` < 0.5.7 | `Grains.csv` DiffPos/DiffOme/DiffAngle cyclically mislabeled | §8 |
| `peakfit: AllPeaks_PS.bin already exists; skip` | results silently inherited from a previous, differently-configured run | §7 |
| peakfit / calc_radius from a tree without the 2026-07-30 determinism fixes | **every re-run gives a different `Grains.csv`**; grain positions jump by >100 µm | Lab Notebook §2 |
| grain position quoted to more than ~100 µm | candidates within a cluster still disagree by 50-280 µm, all at completeness 1.0 | Lab Notebook §2d |
| `indexing(FF): 0 / N seeds with non-zero data` read as a failure | cosmetic — the c-omp backend writes `IndexBest_all.bin`, the counter looks for `IndexBest.bin` | §11 |
| GrainRadius from a tree without the 2026-07-30 ID-space fix | **every grain reported at ~the sample-wide mean radius**; 5.5× too small here | Lab Notebook §3a |
| legacy FF C binaries fed the pipeline's `Spots.bin`/`nData.bin`/`Data.bin` | PF layout vs FF layout — indexer runs minutes instead of 2 s and indexes nothing; looks like bad parameters | §13b, Lab Notebook §3b |
| FF refinement on a tree without the 2026-07-30 `pos_scale` equilibration | **grain positions are the indexer seeds, unrefined** — ~158 µm off the C reference in float32, and the solver reports success | Lab Notebook §3c |
| `midas-process-grains` < 0.7.0 | `Completeness` / `MinNrSpots` are parsed and **discarded** — measured **23710 grains vs 6132** from the same refiner output, no error. Reads as "the peak search is finding noise" | §0, §8b |
| params zipped by `midas-zipper` < 0.1.5 | `BgSubtract`, `BgNSectors` and `MinPeakSNR` are **silently dropped** from the zarr — the peak search runs with settings you did not set | §0, §6c |
| `midas-fit-grain` checked against 0.5.7 and no further | labels are correct, positions are still the **unrefined indexer seeds** (0.6.0) and `c_recipe` is missing (0.7.0) | §0, §8a |
| version floors read from this document instead of from the tree | three floors rose for silent-wrong-answer reasons in the six days after this file was written | §0 |

---

## 0. Verify the install — before anything else

**Do not trust the version numbers in this document. Trust the tree.** Every quantitative
claim here was measured against one state of the code; the packages move faster than the
prose. Between 2026-08-01 and 2026-08-07 alone, three floors in this pipeline rose for
*silent-wrong-answer* reasons, and this handbook carried the old ones.

**The authoritative floors are the `pyproject.toml` files themselves**, not this table.
Run this from the repo root; it takes the **strictest** floor any package in the tree
declares, so one stale dependency list cannot weaken the check:

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
development checkout `importlib.metadata.version()` can report 0.6.0 while the code that
imports is 0.7.0. A gate reading metadata alone fails that tree at step one, for a problem
that does not exist. The reverse is also possible on a stale wheel, which is why the check
takes the **higher** of the two and reports any disagreement rather than silently picking.

**Take the strictest floor, not the one in the orchestrator you happen to use.** They
disagree: `midas_suite` floors `midas-nf-preprocess` at 0.6.0 while `midas_nf_pipeline`
still floors it at 0.4.0 — and 0.4.0 reads `SumFrames` with the *opposite* unit convention
(`NF_HEDM_Handbook.md` §8j). A per-package check against the weaker file certifies a
mixed install.

**Anything below a floor: stop.** These are not import errors. Each one produces a
plausible result that is wrong:

| Package | Below the floor you get |
|---|---|
| `midas-peakfit` | peak fit not reproducible — every re-run a different `Grains.csv` (Lab Notebook §2a) |
| `midas-transforms` | `calc_radius` float atomics; same nondeterminism (Lab Notebook §2b) |
| `midas-fit-grain` | the refiner returns its seed positions and reports success (Lab Notebook §3c); `c_recipe` refine mode absent |
| `midas-process-grains` | `GrainRadius` ~5× low (Lab Notebook §3a); **and** `Completeness`/`MinNrSpots` parsed then discarded — a measured **23710 vs 6132** grains on the same refiner output, no error (`3d9bb427`) |

**Two couplings that packaging cannot express — check them by hand:**

1. **`midas-zipper >= 0.1.5`.** The dependency metadata floors it at 0.1.0 because peakfit
   only needs the zipper in a dev extra. But 0.1.4's allow-lists do not carry
   `BgSubtract`, `BgNSectors` or `MinPeakSNR` — the three keys §6c and §10 tell you to set.
   A parameter file zipped by 0.1.4 **silently drops all three** and the peak search runs
   with settings you did not ask for (`a440bef6`). No error.
2. **`midas_ff_pipeline` still floors fit-grain at 0.6.0 and process-grains at 0.6.1** —
   the pre-fix versions. It is deprecated (§7); use `midas-pipeline run --scan-mode ff`.
   If you install the old orchestrator you get the buggy siblings with no warning.

**When this document and the tree disagree, the tree is right — and say so in your
write-up.** If a number here cannot be reproduced against the code you are running, that
is a finding about this document, not about your data. Record it rather than working
around it.

---

## 0a. THE ORDER — do these steps in this sequence

Each row produces an input the next one needs. Two of them cannot be checked afterwards.

| # | Step | Where | Why it is here and not later |
|---|---|---|---|
| 0 | **Verify the install.** | §0 | Three floors certify silently-wrong versions. Free, and it invalidates everything downstream if skipped. |
| 1 | **Survey the folder** → write `SURVEY.md`. | §0b | Nothing else can start until you know which file is the sweep, which the dark, which the calibrant. |
| 2 | **ω sign** from par field 9. | §2 | **Not detectable afterwards.** A sign error mirrors the microstructure with completeness unchanged. |
| 3 | **Scan definition + dark pairing + `SkipFrame`.** | §3 | `SkipFrame` shifts every ω by one step if wrong — also invisible in the grain list. |
| 4 | **Energy, then distance.** | §4 | Calibration needs λ. `DetZ` is only a seed. |
| 5 | **Calibrate on the calibrant**, overlay the rings. | §5 | Gate: ≤ 100 µε, and the overlay is mandatory. No geometry, no reconstruction. |
| 6 | **Zip the sweep only** — `--only zip_convert`. | §7 | `midas-ring-thresh` reads the zarr, so the zarr must exist before the threshold can be measured. |
| 7 | **Measure `RingThresh`** on that zarr; set it. | §6b | A template threshold is meaningless for your detector and exposure. |
| 8 | **Build the parameter file.** | §6 | Needs the geometry (5), the scan definition (3) and the threshold (7). Replace the calibrant's lattice with the sample's. |
| 9 | **Run the rest of the pipeline.** | §7 | |
| 10 | **Read the result, then report.** | §8, §14 | |

**Step 6 is the one people miss.** §6b tells you to measure `RingThresh` from the data,
but `midas-ring-thresh` operates on `<result>/LayerNr_1/<stem>.MIDAS.zip`, which does not
exist until `zip_convert` has run. Break the cycle by running that stage alone:

```bash
midas-pipeline run --scan-mode ff --params Parameters.txt --result results/ \
    --layers 1-1 --only zip_convert
```

Then verify the dark is non-zero (§3d) — a zero dark makes every threshold return zero
peaks, so measuring `RingThresh` first would give a flat, meaningless table.

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
| `Output/IndexBest_all.bin` | indexing ran (c-omp backend) |
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
pip install "midas-pipeline>=0.8.0" \
            "midas-calibrate-v2>=0.5.3" \
            "midas-zipper>=0.1.5" \
            matplotlib scikit-image
```

`midas-pipeline` pulls `midas-peakfit`, `midas-transforms`, `midas-index`,
`midas-fit-grain>=0.7.0`, `midas-process-grains>=0.7.0`, `midas-hkls`, `midas-stress` and
`midas-diffract` transitively. The three named explicitly are the ones its metadata does
*not* guarantee at the version this runbook needs: `midas-calibrate-v2` for §5, and
`midas-zipper >= 0.1.5` for the peak-search keys (§0). `scikit-image` is a hard
requirement of the v2 auto-seeder (`midas_calibrate_v2/seed/auto_seed.py:523`);
`matplotlib` is needed to produce the mandatory ring overlay (§5d).

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

## 2. STEP 1 — Establish the ω sign convention

**Run this first, on every new dataset.**

```bash
awk '{print $9}' <METADATA_DIR>/<beamtime>_FF.par | sort | uniq -c
```

| field 9 reads | meaning | action |
|---|---|---|
| `aero` / `Aero` | stage turns **clockwise**; **ω_MIDAS = −ω_logged** | negate `OmegaStart` **and** `OmegaStep` |
| anything else | not established by this session | **stop and ask** |

Verified on `bt_1id_jul26`: all **7297** FF rows read `aero`.

Worked example — `Au3_cubes_ff_000008`. The par logs 1441 frames running
ω = −180.25 → +179.75 at step **+0.25**. Negating, and dropping the throwaway frame 0
(§3e), the parameter file gets:

```
OmegaStart 180.25      # omega of RAW frame 0, negated
OmegaStep  -0.25
SkipFrame  1           # -> first frame actually used is at +180.00
OmegaRange -180 180
```

**Why you cannot check this later.** A sign flip in ω mirrors the reconstructed
microstructure. Completeness, grain counts and internal angles are all unchanged. Nothing
inside the reconstruction catches it.

**Corroboration:** `NF_HEDM_Handbook.md` §2 reaches the same rule from the NF par of the
same beamline, and the bundled NF reference paramfile carries `OmegaStart 180` /
`OmegaStep -0.25` for a 360° aero scan.

---

## 3. STEP 2 — Metadata, and the scan definition

### 3a. Where things live

The image tree holds **only frames**. Distances, ω, energy and exposure live in a separate
acquisition-log folder.

| what | where (`bt_1id_jul26`) |
|---|---|
| frames | `/gdata/dm/1ID/2026/bt_1id_jul26/data/ge5/` |
| acquisition logs | `~s1iduser/new_data/bt_1id_jul26/` |
| per-frame FF par | `<logs>/bt_1id_jul26_FF.par` |
| energy monitor | `<logs>/fastsweep_Emon.txt` |
| spec log | `<logs>/FullLog.log` |
| macros | `<logs>/macros_<user>/` |

### 3b. Par-file field map (1-ID FF)

Positional, whitespace-separated. Verified against `bt_1id_jul26_FF.par`:

| field | meaning |
|---|---|
| 1–5 | date stamp |
| 6 | detector tag (`GE_AD`) |
| 7 | scan name |
| 9 | **rotation stage** — the `aero` test (§2) |
| 10, 11 | sweep bounds (logged ω) |
| 17 | **per-frame ω** (logged) |
| 19 | exposure (s) |
| 20 | **file number** |
| 21 | **frame index within the file** (1-based) |

Extract one scan's sweep:

```bash
awk '$20=="000008" && $7=="Au3_cubes_ff" {print $21, $17}' <logs>/<beamtime>_FF.par
```

### 3c. HDF5 layout (DM-converted `.ge5.h5`)

```
exchange/data          (nframes, 2048, 2048) uint16   <- frames, and ALSO the dark file's frames
exchange/data_dark     (1, 2048, 2048)                <- NOT the dark you want
instrument/Detector/PixelSizeX,Y                      <- 200.0 µm
instrument/Detector/ArraySizeX,Y                      <- 2048
instrument/HEM/Energy                                 <- monochromator energy, keV
instrument/DMS/DetZ                                   <- detector STAGE position (§4b)
instrument/SMS/E/HR/samRy                             <- per-frame rotation readback
```

### 3d. The dark — separate file, in `exchange/data`, and the key name is `darkLoc`

**Use the separate dark file, not the in-file `exchange/data_dark`.** Pair it with its
scan by acquisition number: the dark is `dark_before_<N-1>` for data file `<N>` —
`dark_before_000007.ge5.h5` goes with `Au3_cubes_ff_000008.ge5.h5`. Its frames live in
**`exchange/data`**, exactly like the calibrant dark.

**The key `midas_zipper` reads is `darkLoc`, not `darkDataset`.** They are different
spellings consumed by different code:

| key | read by | default |
|---|---|---|
| `dataLoc` / `darkLoc` | `midas_zipper.ff_zip` — `config['darkLoc']`, `ff_zip.py:334` | `exchange/data` / **`exchange/dark`** |
| `dataDataset` / `darkDataset` | downstream consumers (`FF_Parameters_Reference.md` §2) | same |

Set **both**:

```
dataLoc     exchange/data
darkLoc     exchange/data
dataDataset exchange/data
darkDataset exchange/data
Dark /gdata/.../data/ge5/dark_before_000007.ge5.h5
```

> **This is the highest-cost trap in the whole FF path.** Set only `darkDataset` and the
> zipper falls back to `exchange/dark`, finds nothing in the dark file, warns **once** in a
> 1000-line log, and writes an **all-zero dark** into the zarr. Nothing downstream errors.
> The failure surfaces far away, as:
>
> ```
> FrameNr: 0, NrOfRegions: 5, Filtered regions: 0, Number of peaks: 0
> ...
> ValueError: No spots in InputAll.csv. Aborting.        (transforms stage)
> ```
>
> and it is **invariant to `RingThresh`** — lowering the threshold 60 → 10 changes nothing,
> which is the diagnostic signature. Mechanism: with no dark subtracted every pixel sits at
> the ~1900-count pedestal, so the whole frame clears the threshold, labelling returns a
> few enormous blobs, and `filter_regions_by_size` (strict `minNrPx < nPx < maxNrPx`,
> `connected.py:91-100`) discards all of them for exceeding `maxNrPx`.
>
> **Check it directly** rather than trusting the log:
> ```python
> z = zarr.open("<result>/LayerNr_1/<stem>.MIDAS.zip", mode="r")
> assert np.asarray(z["exchange/dark"][0]).max() > 0   # zero == dark was not found
> ```

### 3e. The throwaway first frame — GE / far-field only

**On the 1-ID GE detector the first frame of every acquisition is a settling frame. Always
skip it.**

> **Scope.** This is a **GE (far-field) detector** rule. It does **not** apply to the
> near-field detector: on an NF `DoVolume`/`DoLayer` scan the extra file in the sequence is
> a *trailing* ω-wrap frame at the end, and `StartNr` is the **first** image. Carrying this
> rule to NF drops a real frame and pushes the ω reversal inside the first distance. See
> `NF_HEDM_Handbook.md` §3g.

Measured signature on `bt_1id_jul26` GE5: frame 0 sits ~1.5 % low in baseline versus
every later frame.

| file | frame 0 mean | later frames |
|---|---|---|
| `Au3_cubes_ff_000008` | 1868.96 | ~1898 (frames 720, 1440) |
| `dark_before_000007` | 1870.55 | ~1897.9 (frames 5, 9) |
| `dark_CeO2_..._000003` | — | dropping frame 0 moved the dark mean 2044.1 → 2018.7 |

Set `SkipFrame 1`.

**`SkipFrame` is applied by the consumer, not by the zipper — do not "fix" this.** The
layered design is easy to misread:

| stage | what it does with `SkipFrame` |
|---|---|
| `midas_zipper` | writes **all** raw frames of the first file and the **full** dark stack; records `SkipFrame`; skips leading frames only of files **2+**, which is multi-file concatenation de-duplication, a different thing |
| `midas_peakfit` | does the actual skip: `nFrames -= skipFrame` (`params.py:135`), reads `frame_nr + skipFrame` (`orchestrator.py:181-183`), `dark_arr[skipFrame:]` (`zarr_io.py:301`) |

Consequently **`OmegaStart` is the ω of the first frame you want to USE** (post-skip), and
the zarr's `scan_parameters/start` is deliberately back-dated to
`OmegaStart − SkipFrame·OmegaStep` (`ff_zip.py:294`) so that it describes raw frame 0.
The consumer recovers `start + SkipFrame·step = OmegaStart` for the first frame it
processes. The chain is self-consistent; changing either half alone breaks it.

> Making the zipper physically drop the frame **as well** skips it twice: a 1441-frame
> sweep yields 1439 processed frames instead of 1440. Confirmed the hard way on
> `Au3_cubes_ff_000008` in this tree. Guarded by
> `midas_zipper/tests/test_skipframe.py`.

Sanity check in the peakfit banner: `nFrames` must equal *logged frames − SkipFrame*
(1441 − 1 = **1440**). If it reads 1439, something is skipping twice.

For a hand-reduced average outside the pipeline (calibrant staging, quick looks) there is
no consumer to do it for you, so drop it yourself: `data[1:].mean(axis=0)`, dark included.

---

## 4. STEP 3 — Energy and distance: the two fields that lie

### 4a. Energy

**The filename is not the energy.** On `bt_1id_jul26` the CeO₂ files are named
`..._96keV_...` and the scan was taken at **95.0 keV**.

Sources, in order of trust:

| source | `bt_1id_jul26` | verdict |
|---|---|---|
| `instrument/HEM/Energy` (HDF5) | 95.0 | **use this** |
| `fastsweep_Emon.txt` field 6 (`E_HEM`) | 95.0000 | corroborates |
| spec `FullLog.log` → `Energy (keV):` | 95 | corroborates |
| `instrument/InsertionDevice/IDEnergy` | 95.055 | undulator setting, not the mono |
| `instrument/HRM/Energy` | 78.39 | **different monochromator — ignore** |
| the filename | "96keV" | **stale string** |

`fastsweep_Emon.txt` columns come from `macros_<user>/E_mon.mac`: field 2 is a foil µt, field
6 is `epics_get("1id:userTran3.A")` = the HEM energy readback. **Rows where the last two
columns are `0.000 0.000` had the foil out** (air) and carry no absorption information.

λ[Å] = 12.398419843320026 / E[keV]. At 95.0 keV, λ = 0.130510 Å.

### 4b. Distance — `DetZ` is a stage readback, not `Lsd`

`instrument/DMS/DetZ` is the detector translation-stage position. Its zero is not the
sample rotation centre.

**Measured on `bt_1id_jul26`:** `DetZ` = 1485.00 mm, calibrated `Lsd` = **1666.2 mm** —
an offset of **+181 mm (11 %)**. Using `DetZ` as `Lsd` would have been a catastrophic and
entirely plausible-looking error.

Use `DetZ` as a *seed* only, and expect the fit to move a long way. Differences between
`DetZ` readbacks across a multi-distance scan are trustworthy; the absolute value is not.

---

## 5. STEP 4 — Calibrate on a calibrant

Package: `midas_calibrate_v2` (0.5.2). Entry point `calibrate()` — image + λ + pixel size
+ calibrant name, everything else auto-seeded.

### 5a. Look at the raw frame first

Reduce remotely, plot to PNG, copy back, and *look*. Before any fit you should be able to
state: how many rings are visible, whether they are complete in azimuth, where the
beamstop is, and whether the detector is saturated.

`bt_1id_jul26` CeO₂ reference: rings sharp and complete in azimuth, innermost at
R ≈ 348 px about the fitted BC, beamstop shadow at ≈ (1019, 1076), signal ~54 counts above
a ~2019-count dark after frame-0 removal.

### 5b. Check the ring assignment before you trust the fit

Ring-radius **ratios** depend only on the lattice — λ and `Lsd` cancel. This identifies
which ring the innermost observed one is, independently of any geometry:

```
R_i / R_1  =  tan(2θ_i) / tan(2θ_1)
```

Measure radii from a radial profile about the seeded BC, and compare. On `bt_1id_jul26`
the first 10 CeO₂ rings matched to ≤ 0.0015 in ratio, confirming innermost = (111), and
those 10 rings independently gave `Lsd` = 1667.2 ± 0.3 mm — which is what exposed
`DetZ` (1485 mm) as a stage offset rather than a distance.

> Watch the degenerate families: **(511) and (333) share a d-spacing** and are one ring. A
> naive "i-th observed peak ↔ i-th table entry" pairing slips by one from there on.

### 5c. Run it

```python
from midas_calibrate_v2 import calibrate
res = calibrate(
    img,                      # 2-D, dark-subtracted, frame 0 already dropped
    wavelength=0.130510,      # Å, from §4a
    pxY=200.0,                # µm
    calibrant="CeO2",
    initial_Lsd=1_485_000.0,  # µm; DetZ as a SEED only (§4b)
    output_dir=".../ceo2_calib/",
    n_iter=5,
)
```

Read `res.post_residual_strain_uE`, and the per-iteration `mean / median / trim5%` triple
from the log. **Gate: reject above 100 µε.**

v2 builds an empirical residual-correction map after the fit and **discards it
automatically if it did not reduce strain** (`pipelines/single.py:264-273`). Both outcomes
are normal; check which happened rather than assuming the map is in play.

**The E↔M loop is not monotonic, and that matters.** The E-step re-extracts peaks at the
new geometry, so a late iteration can land in a worse basin than an earlier one. Measured
on this dataset:

```
[v2 iter 2] strain=  59.5μϵ   ty=-0.0230  tz=0.7752
[v2 iter 3] strain=  17.9μϵ   ty=-0.0052  tz=0.9507     <- best
[v2 iter 4] strain=  72.0μϵ   ty= 0.1200  tz=0.8630     <- last
```

`ty` is the weakly-determined direction (it wanders while `tz` holds near 0.9); `Lsd` and
`BC` barely move. Before the 2026-07-30 fix, `autocalibrate` returned the **last** iterate,
so this run shipped a 72 µε geometry when 17.9 µε was in hand — a 4× quality loss, silent,
and still inside the 100 µε gate. It now adopts the best iterate and logs
`adopting best iterate (…) over the last (…)`, matching v1 C
(`FF_HEDM/Example/Parameters.txt`: `nIterations` — "best result is kept"). Guarded by
`midas_calibrate_v2/tests/test_best_iterate.py`.

If your install predates that fix, do not read the final line of the log as the result —
scan all iterations and take the minimum, or re-seed at it.

### 5d. Overlay the rings — mandatory

Predict each ring's radius from the fitted `Lsd`, invert (R, η) → pixel through the **full**
forward model (tilts + distortion + parallax) with
`midas_integrate.geometry.invert_REta_to_pixel_batch`, and draw it on the measured frame.
Look at the inner rings *and* the corners. This is the only check that catches a
well-converged fit sitting on the wrong ring assignment.

### 5e. What a single powder pattern cannot tell you

`Lsd` and λ are near-degenerate: to first order both just scale the ring radii. Fitting at
the wrong energy produces a compensating `Lsd` and a still-good-looking pattern. The
degeneracy is broken only weakly, by the `tan(2θ)` nonlinearity, and **refined distortion
harmonics (`iso_R2/R4/R6`) can absorb most of what is left** — so a distortion-free control
is needed for the comparison to mean anything.

Observed on `bt_1id_jul26` (same image, λ the only change): 95 keV → 19.4 µε,
96 keV → 72.7 µε. Suggestive, and it agreed with the beamline's own confirmation of
95 keV — but treat it as corroboration, not proof.

### 5f. Use the 0/180 pair if you have one

A calibrant measured at two rotations 180° apart gives an independent repeat of the same
detector geometry; the spread between the two fits is an honest uncertainty. On
`bt_1id_jul26`:

| | samRy −90 | samRy +90 | diff |
|---|---|---|---|
| Lsd (mm) | 1666.226 | 1666.008 | 0.218 (0.013 %) |
| BC_y (px) | 1018.720 | 1018.729 | −0.009 |
| BC_z (px) | 1076.540 | 1076.529 | +0.011 |
| ty (°) | 0.0061 | 0.0655 | −0.059 |
| tz (°) | 0.9435 | 0.8977 | +0.046 |

`Lsd` and `BC` repeat superbly; the **tilts** are the weak direction. Note also that the
distortion harmonics differed by up to ~10× between the two fits (`a1`: 0.0001 vs 0.0017)
— the individual harmonic coefficients are fitting noise even when the radial prediction
they sum to is stable. Do not interpret them physically.

### 5g. Export

`midas_calibrate_v2.compat.to_v1.write_v1_paramstest` writes the v1 geometry block
(`Lsd`, `BC`, `tx/ty/tz`, `p0..p14`, `Parallax`, `Wavelength`, `px`, `NrPixelsY/Z`,
`RhoD`, `SpaceGroup`, `LatticeConstant`). v2's harmonic names map back to v1 p-slots via
`_V2_TO_V1_DISTORTION` (`compat/to_v1.py:20-33`) — note the mapping is **not** in index
order (`iso_R2→p2`, `iso_R4→p5`, `iso_R6→p4`, `a2→p0`, …).

`ff_paramstest_from_auto_result` merges the geometry into an existing FF template,
carrying thresholds and scan keys through verbatim.

---

## 6. STEP 5 — Build the parameter file

Start from `FF_HEDM/Example/Parameters.txt` and replace the geometry block. Generate it
from the calibration JSON rather than retyping — hand-copied geometry is how a recon ends
up describing a calibration it isn't using.

**The calibrant's `LatticeConstant` and `SpaceGroup` must be replaced with the sample's.**
The calibration paramstest carries CeO₂ (5.4116, SG 225); a gold sample needs
`4.0782, SG 225`. This is the single most common copy-paste error.

**`ImTransOpt` must match what the calibration was fitted on.** If the calibrant image was
read straight out of `exchange/data` with no transform, the recon must use `ImTransOpt 0`.
A mismatch mirrors the geometry relative to the fit.

Ring selection: only rings with **full azimuthal coverage** are safe defaults. With BC near
the centre of a 2048² detector, that is everything inside the *nearest-edge* radius;
rings between the nearest-edge and far-corner radii are partial and bias η coverage.

`Rsample`, `Hbeam`, `BeamThickness`, `Vsample`, `GlobalPosition` are **not** descriptions
of the sample. **HARD RULE: never set `Rsample`/`Hbeam` to the true sample dimensions.**
They are a deliberately generous *search bound*; tighten them to the real size and any
grain whose true position lies near the boundary is pushed onto it, giving an artefactual
pile-up of grain positions at ±`Rsample` and ±`Hbeam`/2 that reads as real microstructure.
Leave the generous defaults (2000 µm here, matching `FF_HEDM/Example/Parameters.txt`).

---

## 6b. Set `RingThresh` from the data, not from a template

`RingThresh <ring> <threshold>` is the single most consequential number in the peak
search, and the value in any example file is meaningless for your detector, exposure and
sample. Measure it.

The peak finder labels 8-connected blobs above threshold **inside the ring bands only**
(`Width` µm either side of each ring), then applies a **strict** size filter —
`minNrPx < nPx < maxNrPx`, both bounds exclusive (`midas_peakfit/connected.py:91-100`).
With the default `minNrPx 1`, **any single-pixel blob is discarded**. So a threshold that
is slightly too high does not degrade gracefully: it shaves every spot down to a few
isolated pixels and you get exactly zero peaks.

Use the calculator — do not hand-roll the sweep:

```bash
midas-ring-thresh <run>/LayerNr_1/<name>.MIDAS.zip \
    --result-folder <run>/LayerNr_1 --n-frames 40
```

It sweeps thresholds through the **production** peak-search path
(`compute_good_coords` → `preprocess_frame` → `find_regions` →
`filter_regions_by_size`), reports two independent criteria per ring, and prints
paste-ready `RingThresh` lines.

**Do not reimplement the band mask.** Earlier revisions of this section carried a
seven-line snippet that built its own mask from a plain radius-from-beam-centre. That is
wrong: the production band uses the *distortion-corrected* `Rt` after
`apply_image_transformations` + `transpose_square`. Measured on `Au3_cubes_ff_000008` the
naive mask shares only **13.4 %** of its pixels with the real band, which made blob counts
disagree with the pipeline by ~67× and manufactured a spurious "background varies by 20σ
around the ring" result. Through the real band that background is flat (spread 0.4σ).

The two criteria:

- **A — blob SNR.** Lowest threshold at which ≥90 % of surviving blobs have local
  SNR > 5, measured on the *ungated* frame. (It must be ungated: after thresholding, every
  sub-threshold pixel is 0, so the local background and its MAD collapse and every SNR
  reads as 0.) The annulus is restricted to in-band pixels — a band is only `2*Width` wide,
  so an unrestricted annulus is mostly out-of-band zeros.
- **B — expected false positives.** Lowest threshold whose predicted noise-blob count over
  the *whole scan* is under 10, from the per-cell σ and the `minNrPx` size filter. This is
  the criterion that matters when the sample is sparse: a 2-grain dataset has ~1–2 real
  peaks per frame against ~3×10⁵ in-band pixels, so a tiny per-pixel false rate still
  swamps the signal.

They should agree; the tool says so explicitly when they do not, which points at a bad band
or a broken dark rather than at the threshold.

**Why the old "pick the knee" rule was not enough.** Blobs/frame surviving the size filter
on `Au3_cubes_ff_000008` go 5.2 (thr 5) → 1.6 (10) → 0.8 (20) → 0.5 (40). The two-orders
jump is between 5 and 10, but noise keeps falling out well past it. The knee locates where
noise *percolates* into detector-spanning blobs (largest blob 645 px at thr 5 vs 393 at
10), not where noise stops being admitted.

Measured on `Au3_cubes_ff_000008` (20 ms/frame, `Width` 7.5 px) the two criteria agree on
every ring and give **`RingThresh` 10 / 20 / 20 / 10 / 10** for rings 1–5. On ring 5 the
clean fraction goes from 20 % of blobs above SNR 5 at threshold 5 to 100 % at threshold 10.

*(An earlier revision of this section reported "both criteria return 10 on every ring, 92 %
clean at 10 vs 52 % at 5". Those came from a `blob_snr` that restricted its background
annulus to in-band pixels and was over-optimistic; fixed, and the numbers above supersede
them — Lab Notebook §6b.)*

**Caveat:** this tuning is only meaningful once the dark is verified non-zero (§3d). If the
dark is missing, *every* threshold yields zero peaks and the table above is flat — that
invariance is itself the diagnostic.

---

## 6c. Reject spurious peaks by SNR, not by a proxy

`RingThresh` decides what gets *detected*. **`MinPeakSNR`** decides what gets *kept*, and it
is the only quality criterion here that does not smuggle in an assumption:

```
MinPeakSNR 5          # 0 = off (default); (peak - cell_median) / cell_sigma
```

> **`midas-zipper >= 0.1.5` or this key does nothing.** 0.1.4's allow-lists do not carry
> `MinPeakSNR`, `BgSubtract` or `BgNSectors`, so a parameter file zipped by it drops all
> three into the void — the peak search then runs at the defaults with no error and no log
> line (`a440bef6`, §0). The keys are written as **datasets** under
> `analysis/process/analysis_parameters` (`ff_zip.py:159-167`), so check the zarr itself
> before trusting any threshold you set:
>
> ```bash
> python -c "
> import zarr, sys
> g = zarr.open(sys.argv[1], mode='r')['analysis/process/analysis_parameters']
> for k in ('MinPeakSNR', 'BgSubtract', 'BgNSectors'):
>     print(k, list(g[k][:]) if k in g else '*** ABSENT — zipper too old ***')
> " <result>/LayerNr_1/<stem>.MIDAS.zip
> ```

Computed per peak against its own (ring, azimuthal sector) cell during the peak search, so
it costs nothing extra and applies to **FF and PF alike** (both use `midas_peakfit`). It is
the natural knob for the pf-HEDM failure mode where spurious signal is admitted.

**Do not substitute a proxy — each of these has been measured to fail:**

| proxy | why it fails |
|---|---|
| `MinIntegratedIntensity` | no noise estimate, so it cannot tell a weak real spot on a quiet patch of detector from a noise excursion on a hot one |
| `FitRMSE` | an **absolute** residual, so it grows with peak intensity — cutting at `FitRMSE < 2000` discarded **58 % of the indexed spots** on the reference dataset |
| ω multiplicity (`NImgs`) | encodes **mosaicity, not reality**. A small or undeformed grain can satisfy Bragg inside one frame; 45.9 % of credible spots were single-frame and 8 indexed spots reached SNR 2511 on one frame |

**No specific value is recommended yet.** Two SNR estimators (per-cell vs a box on the raw
frame) rank spots differently — 94 % vs 53 % clean at SNR 5 on the reference dataset — and
which is correct decides where the cut belongs (Lab Notebook §6d). Start at 5, and check
what it removes against the raw frames before trusting it.

---

## 7. STEP 6 — Run the pipeline

```bash
midas-ff-pipeline run \
    --params Parameters.txt \
    --result results/ \
    --layers 1-1 \
    --device cuda
```

13 stages, each with a provenance entry in `<result>/LayerNr_N/midas_state.h5`:

```
zip_convert → hkl → peakfit → merge_overlaps → calc_radius → transforms
→ cross_det_merge → global_powder → binning → indexing → refinement
→ process_grains → consolidation
```

`zip_convert` is skipped when the zarr already exists; `cross_det_merge` and
`global_powder` are no-ops for single-detector runs; `consolidation` is gated by
`--generate-h5`. Auto-resolved knobs (`--dtype`, `--shard-gpus`, `--group-size`) are logged
at startup; explicit values always win.

> `midas-ff-pipeline` is **deprecated** as of 0.4.0 — use `midas-pipeline run --scan-mode ff`.
> Same orchestrator underneath.

**Two things to check in the log every time:**

1. `nFrames` in the peakfit banner = logged frames − `SkipFrame` (§3e).
2. **Stage resume is silent.** `peakfit(FF): …AllPeaks_PS.bin already exists; skip.`
   means the peak search did **not** run and results were inherited from a previous
   invocation — which may have used a different threshold, a different dark, or a broken
   config. It costs 0.3 s instead of 55 s, so it is easy to miss. **After changing any
   peak-search or dark parameter, delete `results/` entirely**, do not rely on resume.

Subprocess stages (`peakfit_torch`, `midas_indexer`, `midas_fit_grain`) are invoked by
bare name, so the env's `bin` must be on `PATH` — calling `midas-pipeline` by full path is
not enough and fails with `FileNotFoundError: 'peakfit_torch'`:

```bash
export PATH=/home/beams12/S1IDUSER/opt/envs/midas/bin:$PATH
```

For ≥ 5 k seeds also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Check the zipper's frame accounting in the log.** It prints
`HDF5 scan: N file(s), F frames/file. Skipping the first S frame(s) of every file. Total
frames to write: T`. Confirm `T` equals the frame count you derived from the par file in
§3b. If `T` is one larger, `SkipFrame` did not take effect (§3e version trap).

---

## 8. STEP 7 — Read the result

### 8a. Check the refiner version before reading the residual columns

**`midas-fit-grain` < 0.5.7 writes `DiffPos`, `DiffOme`, `DiffAngle` cyclically
mislabeled.** `driver.py` assigned `calc_angle_errors`'s `(mean_angle, mean_pos, mean_ome)`
straight into the `(pos, ome, angle)` slots, so every FF/PF run through the python/torch
refiner has the three columns rotated (commit `44394e61`; the classic C refiner path was
unaffected).

The tell is obvious once you look: an ω residual of **223°** is impossible on a 0.25° step.
Post-fix the same grain reads `DiffPos 202 µm, DiffOme 0.054°, DiffAngle 0.090°` — all
physical.

0.5.7 fixes the *labels*. **It is not the floor** — the floor is **`>= 0.7.0`** (§0), and
the three reasons stack:

| version | what it fixes |
|---|---|
| 0.5.7 | the cyclic column mislabel described above |
| 0.6.0 | the `pos_scale` fp32 scaling bug — below this the refiner **returns its seed positions unrefined** and reports success (Lab Notebook §3c) |
| 0.7.0 | the `c_recipe` refine mode and its NLopt Nelder-Mead port (`06dd3241`) — the mode that reproduces the C refiner (Lab Notebook §7n) |

Passing the 0.5.7 check and stopping there is the trap: the labels are right and the
positions are still the indexer's seeds.

If you are stuck on 0.5.6, the mapping is: printed `DiffPos` = true DiffAngle, printed
`DiffOme` = true DiffPos, printed `DiffAngle` = true DiffOme.

### 8b. What to check in `Grains.csv`

Before interpreting it:

1. **Grain count vs expectation** — but rule out the plumbing before you blame the physics.
   A calibration cube should give a handful of grains, not thousands. Too many grains has
   **three** causes, in this order:
   1. **Your grain-selection keys were discarded.** `FitSetup` writes `paramstest.txt` for
      the indexer and refiner, which have no use for `Completeness`/`MinNrSpots`, so those
      keys are simply absent from it and every downstream consumer falls back to its own
      default. Measured on a Ni layer: the same refiner output gave **23710 grains via
      `paramstest.txt` and 6132 via the archive that carries the keys** — 3.9×, no error
      anywhere. Fixed by `360cc09e` + `midas-process-grains >= 0.7.0` (§0). **Check this
      first** — it is free, and it looks exactly like a bad peak search.
   2. Genuinely permissive `Completeness` / `MinNrSpots` for the sample.
   3. Only then: the peak search is finding noise (§6b).
2. **Completeness distribution**, not just the mean — a bimodal distribution means two
   populations, usually real grains plus junk. **`midas-process-grains >= 0.7.0` will read
   the cut off that distribution for you**: the antimode of the log₁₀ histogram of the
   quality metric. It is deliberately data-driven, because a fixed threshold does not
   transfer — the EBSD-optimal `DiffPos` cut on one `shade_LSHR` layer was 195.4 µm for the
   C chain and 222.8 µm for the python chain on the *same raw data* (`296368d2`). The gate
   **refuses rather than guesses** when the distribution is not bimodal; a refusal is
   information, not a failure.
3. **Position envelope.** If grain positions pile up against ±`Rsample` or ±`Hbeam`/2, the
   envelope is binding and the positions are not physical. The fix is to make the envelope
   MORE generous, never less — see the hard rule in §6.
4. **Strain sanity.** Whole-grain strains far above ~10⁻³ on an annealed calibration sample
   mean the geometry, not the sample.
5. **What fraction of spots got indexed?** `wc -l InputAll.csv` versus the spots actually
   assigned. A handful of grains explaining a few hundred of several thousand spots is an
   *under-indexed* run, not a sparse sample — confidence 1.0 on the few grains found says
   nothing about the ones missed.
6. **Re-run and compare grain-by-grain.** Grains that appear in one run and not the next
   are indexing noise. On `Au3_cubes_ff_000008` two runs shared only one of their two
   grains; that instability is the signal that `Completeness`, `MinNrSpots`,
   `OverAllRingToIndex` still need work.
7. **`indexing: 0 / N seeds with non-zero data`** in the log deserves an explanation before
   any grain list is trusted.

---

## 9. Reference numbers — `bt_1id_jul26`, GE5 (ADEPT), 95.0 keV

Established in this tree on 2026-07-30. Detector 2048², 200 µm, monolithic.

| quantity | value | how |
|---|---|---|
| energy | 95.0 keV (λ 0.130510 Å) | `HEM/Energy`, Emon, spec log, beamline confirmation |
| `DetZ` readback | 1485.00 mm | `instrument/DMS/DetZ` — **not** Lsd |
| `Lsd` | see `analysis/ceo2_calib_ge5/summary_all.json` | CeO₂ fit, ≈1666 mm |
| `BC` | ≈ (1018.7, 1076.5) px | CeO₂ fit |
| tilts | ty ≈ 0.0–0.07°, tz ≈ 0.90–0.94° | CeO₂ fit, 0/180 spread |
| calibrant strain | ≈ 19 µε (mean), 13 µε (median) | v2, residual map discarded |
| Au sweep (file 8) | 1441 logged frames, 1440 used | par field 21 |
| Au ω (MIDAS) | `OmegaStart 180.00` (first used), `OmegaStep -0.25` | §2 + §3e |
| Au `RingThresh` | 10 (not 60) | measured, §6b |
| Au spots found | 2076 binned rows, but only **229 are credible** — the rest is noise, padding and haloes | Lab Notebook §4d |
| Au grains indexed | **2** (parent + Σ3 twin), confidence 1.000, R ≈ 21 µm, a = 4.07976 Å — a COMPLETE recon | Lab Notebook §4d |
| Au residuals (0.5.7 cols) | DiffPos ≈ 200 µm, DiffOme ≈ 0.05°, DiffAngle ≈ 0.08° | §8a |

Working run directories:
`analysis/ceo2_calib_ge5/` (calibration) and `analysis/au3_cubes_ff_000008/` (recon).
The shared env needed a `PYTHONPATH` overlay at `~s1iduser/opt/midas_overlay` for
`midas-fit-grain` 0.5.7 and `midas-calibrate-v2`; the env itself ships 0.5.6.

---

## 10. Parameter-file reference

Full key list: `FF_Parameters_Reference.md`. Keys this runbook depends on:

| key | note |
|---|---|
| `SkipFrame` | 1 at 1-ID; applies to every multi-frame file and the dark (§3e) |
| `dataDataset` / `darkDataset` | both `exchange/data` for DM files (§3d) |
| `OmegaStart` / `OmegaStep` | negated vs the log for `aero` (§2); `OmegaStart` describes **raw** frame 0 |
| `ImTransOpt` | must match the calibration (§6) |
| `LatticeConstant` / `SpaceGroup` | the **sample's**, not the calibrant's (§6) |
| `RhoD` | µm; distortion normalisation radius |
| `Rsample` / `Hbeam` | generous grain-position search bound — **never set to the real sample size** (§6) |
| `RingThresh <ring> <val>` | per-ring detection threshold; set it with `midas-ring-thresh` (§6b), never from a template |
| `MinPeakSNR` | float, default 0 = off. Minimum local SNR to keep a detected peak (§6c). FF **and** PF |
| `BgSubtract` / `BgNSectors` | 0/1 (default 0) + azimuthal cells per ring. Removes a varying background before thresholding. Only helps where the background actually varies — it did **not** on this beamtime (Lab Notebook §6a) |

---

## 11. Validation status — what is measured, what is convention

Short form. The evidence, the failed hypotheses and the retracted claims live in the
companion **`FF_HEDM_Lab_Notebook.md`**; this section is only what you need in order to
know how far to trust a number.

**Measured on this beamtime** — ω sign (`aero`, all 7297 par rows), the throwaway first
frame (~1.5 % baseline offset, three files), `SkipFrame` as a consumer-side skip,
`DetZ` − `Lsd` = +181 mm by ring ratios, energy 95.0 keV (three instrument records +
beamline), CeO₂ 0/180 repeatability, the `RingThresh` table in §6b.

**Convention, NOT measured** — `ImTransOpt 0` (chosen so the recon matches the frame the
calibration was fitted on; a self-consistent calibration + recon pair can still be
globally *mirrored*, and nothing here pins the absolute handedness). `OmegaStart` as "ω of
the first USED frame".

**Could not verify** — whether `DetZ`'s +181 mm offset is stable across the beamtime
(measured at one distance); whether the 95-vs-96 keV strain gap is a genuine energy
discriminator or partly distortion re-fitting (the distortion-frozen control was not run).

**How far to trust the output** — orientation and lattice parameter are solid; grain
**position is good to ~100 µm, no better** (Lab Notebook §2d). Everything else is
conditional on the install passing §0: `GrainRadius` needs `midas-process-grains >= 0.6.1`
and the **grain-selection keys** need `>= 0.7.0`; bit-reproducibility needs
`midas-peakfit >= 0.4.6` and `midas-transforms >= 0.8.2`; the refiner refines position at
all only from `midas-fit-grain >= 0.6.0` and reproduces the C recipe only from `>= 0.7.0`.
**Run the §0 check and quote its output** — do not assert these from this list.

**Do not judge a reconstruction by the fraction of the spot list it indexes.** On this
dataset 2 grains index 8.9 % of the rows and the recon is nevertheless *complete* — ~98 %
of that list is noise, zero-intensity padding, and over-segmented haloes of the two grains
themselves. A low indexed fraction is a statement about `RingThresh`, not about missing
grains. Classify the spots by own-frame SNR against the raw frames before concluding
anything from it; the method and the numbers are in Lab Notebook §4d.

**Bottom line.** The geometry is trustworthy in *magnitude* — `Lsd` and `BC` repeat to
0.01 % / 0.01 px across an independent 180° repeat, and the rings overlay. Its
**handedness** rests on convention, not measurement. The ω sign and the frame-0 skip
are the two settings that will silently ruin a reconstruction, and both are now pinned
by tests or by measurement rather than by memory.

### `indexing(FF): 0 / N seeds with non-zero data` is cosmetic, not a failure

`midas_pipeline/stages/indexing.py:150-157` counts non-zero rows in
`Output/IndexBest.bin`, but the c-omp backend (the default, and the fast
path) writes `Output/IndexBest_all.bin` + `IndexKey_all.bin` instead — so the counter finds no file and prints 0 even though
indexing succeeded. Judge the stage by `Results/OrientPosFit.bin` and the grain count.

---

## 12. Check reproducibility on a new install

Run the identical pipeline twice into a **clean** result dir and checksum every stage in
pipeline order — not just `Grains.csv`. You want the *first* artifact that diverges,
because that is what names the guilty stage.

`rm -rf` the result dir between runs. The stages resume silently off existing files (§7),
so a "reproducible" result can just be a skipped stage.

```bash
#!/bin/bash
# usage: ff_repro.sh <paramfile> <scratch-dir>
# Runs the pipeline twice into separate trees and reports the FIRST divergence.
set -u
PARAMS="$1"; BASE="$2"
STAGES="Temp/AllPeaks_PS.bin Temp/AllPeaks_PX.bin InputAll.csv
        InputAllExtraInfoFittingAll.csv Spots.bin Data.bin nData.bin ExtraInfo.bin
        SpotsToIndex.csv Output/IndexBest_all.bin Output/FitBest.bin
        Results/OrientPosFit.bin Grains.csv SpotMatrix.csv"

for run in A B; do
  rm -rf "$BASE/$run"; mkdir -p "$BASE/$run"
  midas-pipeline run --scan-mode ff --params "$PARAMS" \
      --result "$BASE/$run" --layers 1-1 > "$BASE/$run.log" 2>&1
done

# md5sum is GNU; on macOS use `md5 -q`. Pick whichever exists.
MD5=$(command -v md5sum >/dev/null && echo "md5sum" || echo "md5 -q")
diverged=0
for f in $STAGES; do
  a="$BASE/A/LayerNr_1/$f"; b="$BASE/B/LayerNr_1/$f"
  [ -e "$a" ] && [ -e "$b" ] || { printf '%-40s MISSING\n' "$f"; continue; }
  ha=$($MD5 < "$a" | cut -d' ' -f1); hb=$($MD5 < "$b" | cut -d' ' -f1)
  if [ "$ha" = "$hb" ]; then printf '%-40s ok\n' "$f"
  else printf '%-40s *** DIVERGED ***\n' "$f"
       [ $diverged -eq 0 ] && echo ">>> FIRST DIVERGENCE: $f  <- this stage is guilty"
       diverged=1
  fi
done
[ $diverged -eq 0 ] && echo "bit-identical across both runs"
```

Glob-named artifacts (`Result_StartNr_*.csv`, `Radius_StartNr_*.csv`) are omitted because
their names carry the start number; add them explicitly once you know it.

With an install that passes §0 this is bit-identical across runs. If it is not, read Lab
Notebook §2 — two separate nondeterminism bugs are documented there with the signature
each produces. They were found exactly this way, and each masked the other until the first
was fixed. Expect to iterate.

---

## 13. Cross-check against the C reference (`FF_HEDM/src`)

The C chain is the reference implementation. When a python result looks wrong, run it —
it is what found five defects in this pipeline. Recipe below; findings in the Lab Notebook.

### 13a. Build it — the shipped binaries are stale

`FF_HEDM/bin/*` on the beamline hosts were compiled in Apr/May 2026 and
`FitPosOrStrainsOMP.c` has changed since. Build fresh. chutoro has no internet, so reuse
the already-fetched dependency tree instead of letting FetchContent phone home:

```bash
cmake -S ~s1iduser/opt/MIDAS_canonical -B $HOME/opt/ffbuild \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DBUILD_OMP=ON \
  -DFETCHCONTENT_BASE_DIR=/home/beams12/S1IDUSER/opt/MIDAS/build/_deps \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build $HOME/opt/ffbuild --target IndexerOMP FitPosOrStrainsOMP ProcessGrains -j 16
```

Note the build rule also copies the binaries into the source tree's `FF_HEDM/bin/`.

`FitPosOrStrainsOMP`'s usage string says `param.txt nBlocks blockNr …`; the code reads
`blockNr = argv[2], nBlocks = argv[3]` (lines 2325-2326), same as `IndexerOMP`. The usage
string is wrong — pass `blockNr nBlocks`.

### 13b. `Spots.bin` is 10 columns now; legacy `IndexerOMP` reads 9

| | `FF_HEDM/src/IndexerOMP.c` | `midas_index/c_src/IndexerUnified.c` |
|---|---|---|
| `N_COL_OBSSPOTS` | 9 (line 63) | 10 (line 100) — col 9 = `ScanNr` |

`midas_transforms.bin_data` writes the **10**-column layout, and
`midas_index/bin/midas_indexer` (built from `IndexerUnified.c`) is the maintained C
indexer the pipeline already calls. Feed the 10-column file to legacy `IndexerOMP` and it
strides through the array wrongly: on this dataset it reported

```
WARNING: SpotId 1177.000000 not found in spots file! Ignoring this spotID.   (×168 of 189)
```

and wrote an all-zero `IndexBest.bin`, after which `FitPosOrStrainsOMP` exits in 0.01 s and
`ProcessGrains` says *"OrientPos file was not found … nothing was indexed"*. **That cascade
is a format mismatch, not a parameter problem** — do not go tuning `Completeness` in
response to it. The tree documents the difference in
`midas_index/dev/c_indexer_diff.md`.

To run the legacy chain anyway, drop col 9 (row order is preserved and `Data.bin`/
`nData.bin` store row indices, not byte offsets):

```python
a = np.fromfile("Spots.bin", dtype=np.float64).reshape(-1, 10)
np.ascontiguousarray(a[:, :9]).tofile("Spots9.bin")
```

After that the warning count drops to 0.

### 13c. `ProcessGrains` needs no re-indexing

It reads only `Results/{Key,OrientPosFit,ProcessKey}.bin`, `Output/FitBest.bin`,
`SpotsToIndex.csv` and `InputAllExtraInfoFittingAll.csv` — never `Spots.bin`. So you can
point it straight at a python pipeline's output and compare grain reduction in isolation:

```bash
cd <copy of layer dir> && $HOME/opt/ffbuild/bin/ProcessGrains -paramFN paramstest.txt -nCPUs 16
```


### 13d. Where the findings are

Everything the comparison turned up — five fixed defects, the Σ3 twin verification, and
the claims that had to be retracted — is in **`FF_HEDM_Lab_Notebook.md`**. Read it before
re-investigating anything in this pipeline; several attractive hypotheses are recorded
there as *refuted*, with the measurement that killed them.

---

## 14. Report — and what "done" means

### 14a. What to hand back

A grain list is not the deliverable; a grain list **with its provenance and its caveats**
is. Write these into the report, not just into the chat:

- The `§0` install-gate output, verbatim. Every claim below is conditional on it.
- `SURVEY.md` (§0b) — the measured inventory, including which file is which and where each
  number came from.
- The calibration result: strain median **and** 5 %-trimmed, the 0/180 spread if you have
  one, and **the ring overlay image** (§5d). The overlay is evidence, not decoration.
- The parameter file actually used, and the `RingThresh` measurement that set it (§6b).
- `Grains.csv` with the §8b checks answered, each with the number that answers it.
- Every assumption you made where this document said *stop and ask* and you proceeded
  anyway — name it explicitly.

**Every quantitative claim names the file and the command that produced it.** A number you
cannot re-derive does not go in the report.

### 14b. Say which bucket each number falls in

§11 splits this pipeline's output three ways — **measured on this beamtime**,
**convention, not measured**, and **could not verify**. Put every number you report into
one of them. The geometry's *magnitude* is measured; its *handedness* is convention. Do
not let a §11 item become a fact by being quoted often enough.

### 14c. Done means

- [ ] §0 install gate run, output pasted, **no package below floor**
- [ ] `SURVEY.md` written, every number read from a file rather than a name
- [ ] ω sign established from par field 9 — or **stopped and asked** if it was not `aero`
- [ ] `SkipFrame` set, and the peakfit banner's `nFrames` = logged frames − `SkipFrame`
- [ ] dark verified **non-zero in the zarr**, not merely configured (§3d)
- [ ] energy from three instrument records, never the filename
- [ ] calibrant strain ≤ 100 µε, reported as median + trimmed
- [ ] **ring overlay produced and looked at** (§5d) — the one check that catches a
      well-converged fit on the wrong rings
- [ ] `RingThresh` measured with `midas-ring-thresh`, not copied
- [ ] sample lattice constant + space group replace the calibrant's (§6)
- [ ] `Rsample`/`Hbeam` left generous; grain positions checked for pile-up at the bounds
- [ ] `Grains.csv` read against all seven §8b checks
- [ ] reconstruction re-run once and compared grain-by-grain (§8b item 6)
- [ ] every number bucketed per §14b, with its provenance

**If a box cannot be ticked, say so in the report rather than leaving it blank.** An
unticked box is a known limit; a silently skipped one becomes a false claim.
