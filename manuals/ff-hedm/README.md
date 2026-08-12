# FF-HEDM Reconstruction Runbook — survey → calibrate → reconstruct → report

**Use this doc to start a fresh session on a far-field dataset this pipeline has never
seen.** Paste it in together with `LAB_NOTEBOOK.md`, then give three lines:

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
absolute paths. Every non-obvious claim carries one, and `utils/doc_citation_check.py`
(wired into the pre-commit hook) fails the commit when a cited file, line or symbol no
longer exists — so a citation here points at real code. **It cannot check the claim, only
the pointer:** the line is right, the sentence about it may still have gone stale.
Claims that are convention, or that
could not be verified, are flagged inline and summarised in §11. **Do not promote a §11
item to a fact.**

**`LAB_NOTEBOOK.md` is the companion to this file.** This document is the
procedure — what to do, in order. The notebook is the evidence: how each defect was found,
the measurements behind every number here, and the claims that had to be *retracted*.
Read the notebook before re-investigating anything in this pipeline; several attractive
hypotheses are recorded there as refuted, and one of them (Lab Notebook §4c) will actively
damage a reconstruction if it comes back — it invites the change hard rule 9 forbids.

Sibling document: `NF_HEDM_Handbook.md`. The two share §2 (ω sign), §3 (metadata) and §4
(energy/distance) almost verbatim, because those are properties of the *beamline*, not of
the modality. Where FF differs, it is called out.

### The doc set — what to read when

**This file is the spine, and the only one you need loaded the whole time.** Everything
else is opened when you reach it. Section numbers are continuous across the set.

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate (§0), the order (§0a), hard rules, halt conditions | always — start here |
| [`phase-0-survey.md`](phase-0-survey.md) | §0b, §0c, §1, §1a, §1b — environment, folder survey, already-processed check | before touching data |
| [`phase-1-geometry.md`](phase-1-geometry.md) | §2–§5g — ω sign, metadata, dark, `SkipFrame`, energy, distance, calibration | the long one; most silent failures live here |
| [`phase-2-configure.md`](phase-2-configure.md) | §6, §6b, §6c, §10 — parameter file, `RingThresh`, `MinPeakSNR`, key reference | when writing `Parameters.txt` |
| [`phase-3-run.md`](phase-3-run.md) | §7, §12 — running the pipeline, resume traps, reproducibility check | when launching |
| [`phase-4-read-report.md`](phase-4-read-report.md) | §8–§8b, §11, §14–§14c — `Grains.csv` checks, validation buckets, report, done-means | when a result exists |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptom → discriminating test → cause → lever | **when something looks wrong** — indexed by symptom, not by step |
| [`RUNBOOK.md`](RUNBOOK.md) | §R1–§R3 — where it runs, what healthy looks like *with conditions*, and the current pick-up point | on resume, and before quoting any number as "normal" |
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | evidence, measurement ledger, **retracted claims** | before re-investigating anything |
| [`ENVELOPE.md`](ENVELOPE.md) | what this measurement can and cannot determine, sorted by whether anything can be done about it | before promising an answer, and **before suggesting a different measurement** |
| [`C_REFERENCE.md`](C_REFERENCE.md) | §13–§13d — the C cross-check recipe | only when a python result looks wrong |

**Citation convention.** A bare `§n` means *this doc set* — use the table above to find
which file. A reference to the notebook is always written `Lab Notebook §n`, because the
notebook has its own §1–§7 that collide with these.

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
| `indexing(FF): 0 / N seeds with non-zero data` dismissed as cosmetic | it **was** cosmetic before `8a594ea5` (the counter read only the legacy `IndexBest.bin`). It is now real: either a hard error, or an honest zero. Reading an older handbook here hides a genuine failure | §11 |
| `n_seeds_attempted` / `n_seeds_indexed` quoted from a pre-`8a594ea5` FF run | FF only ever wrote them into metrics — **both fields read 0 on every FF run** regardless of what indexing did | §11 |
| GrainRadius from a tree without the 2026-07-30 ID-space fix | **every grain reported at ~the sample-wide mean radius**; 5.5× too small here | Lab Notebook §3a |
| legacy FF C binaries fed the pipeline's `Spots.bin`/`nData.bin`/`Data.bin` | PF layout vs FF layout — indexer runs minutes instead of 2 s and indexes nothing; looks like bad parameters | §13b, Lab Notebook §3b |
| FF refinement on a tree without the 2026-07-30 `pos_scale` equilibration | **grain positions are the indexer seeds, unrefined** — ~158 µm off the C reference in float32, and the solver reports success | Lab Notebook §3c |
| `midas-process-grains` < 0.7.0 | `Completeness` / `MinNrSpots` are parsed and **discarded** — measured **23710 grains vs 6132** from the same refiner output, no error. Reads as "the peak search is finding noise" | §0, §8b |
| params zipped by `midas-zipper` < 0.1.5 | `BgSubtract`, `BgNSectors` and `MinPeakSNR` are **silently dropped** from the zarr — the peak search runs with settings you did not set | §0, §6c |
| `midas-fit-grain` checked against 0.5.7 and no further | labels are correct, positions are still the **unrefined indexer seeds** (0.6.0) and `c_recipe` is missing (0.7.0) | §0, §8a |
| version floors read from this document instead of from the tree | **eight** declarations rose for silent-wrong-answer reasons in the nine days after this file was written, across five packages | §0 |

---

## 0. Verify the install — before anything else

**Do not trust the version numbers in this document. Trust the tree.** Every quantitative
claim here was measured against one state of the code; the packages move faster than the
prose. Between 2026-08-01 and 2026-08-09 alone, **eight** declarations across five
packages rose for *silent-wrong-answer* reasons — and at each point this handbook
carried the old ones.

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

**Take the strictest floor, not the one in the orchestrator you happen to use.** The
declarations have disagreed before and can again: `midas_suite` floored
`midas-nf-preprocess` at 0.6.0 while `midas_nf_pipeline` still admitted 0.4.0 — which
reads `SumFrames` with the *opposite* unit convention (`NF_HEDM_Handbook.md` §8j). That
particular gap is closed, but a per-package check against whichever file happens to be
weakest is how a mixed install gets certified, so scan them all.

**Anything below a floor: stop.** These are not import errors. Each one produces a
plausible result that is wrong:

| Package | Below the floor you get |
|---|---|
| `midas-peakfit` | peak fit not reproducible — every re-run a different `Grains.csv` (Lab Notebook §2a) |
| `midas-transforms` | `calc_radius` float atomics; same nondeterminism (Lab Notebook §2b) |
| `midas-fit-grain` | the refiner returns its seed positions and reports success (Lab Notebook §3c); `c_recipe` refine mode absent |
| `midas-process-grains` | `GrainRadius` ~5× low (Lab Notebook §3a); **and** `Completeness`/`MinNrSpots` parsed then discarded — a measured **23710 vs 6132** grains on the same refiner output, no error (`3d9bb427`) |
| `midas-zipper` | `BgSubtract`, `BgNSectors` and `MinPeakSNR` dropped when the params are zipped — the peak search runs at the defaults with no error (`a440bef6`) |

**The floors above are now all declared in the metadata**, so the gate catches them. That
was not true when this document was written: `midas-zipper` was floored at 0.1.0 in both
orchestrators even though `zip_convert` runs the zipper and `midas-peakfit` then reads the
keys it wrote, and `midas_ff_pipeline` still admitted the pre-fix `fit-grain` and
`process-grains`. Both are fixed (`midas-pipeline >= 0.8.2`, `midas-ff-pipeline >= 0.4.3`).

**Two things the gate still cannot see:**

1. **An old zarr.** The floor governs what you *install*, not what is already on disk. A
   `.MIDAS.zip` written earlier by a 0.1.4 zipper is missing those three keys permanently
   — re-zip rather than reuse it, and check the zarr directly (§6c).
2. **`midas_ff_pipeline` is deprecated** (§7). Its floors are correct now, but use
   `midas-pipeline run --scan-mode ff`; the old orchestrator is removed in suite 1.0.

**When this document and the tree disagree, the tree is right — and say so in your
write-up.** If a number here cannot be reproduced against the code you are running, that
is a finding about this document, not about your data. Record it rather than working
around it.

---

## 0a. THE ORDER — do these steps in this sequence

Each row produces an input the next one needs. Two of them cannot be checked afterwards.

| # | Step | Where | Why it is here and not later |
|---|---|---|---|
| 0 | **Verify the install.** | §0 | Several floors exist only to keep out silently-wrong versions. Free, and it invalidates everything downstream if skipped. |
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
