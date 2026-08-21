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

**Scope.** Two single-panel configurations, one layer at a time:

| | **1-ID GE** | **20-ID HT-HEDM Varex** |
|---|---|---|
| files | DM-converted `.ge5.h5` | `.vrx.h5` |
| detector | monolithic GE, 2048² @ 200 µm | Varex, 2880² @ 150 µm |
| dark | `Dark` file, `darkLoc` | `Dark` file, **`darkLoc /exchange/bright`** — `/exchange/dark` exists and is all zeros |
| ω sign | par field 9 = `aero` ⇒ negate (§2) | `OmegaStart 180`, `OmegaStep -0.25` already negated in the file |
| frame 0 | throwaway, `SkipFrame 1` (rule 2) | same |
| `ImTransOpt` | establish per detector (§3f) | **2** (flip-Z), verified on `bt_20id_jul26b` |
| verified on | `bt_1id_jul26` | `bt_20id_jul26b` ti7al / nf709 / ruby |

Multi-panel (GE1–4) and multi-layer scans are *not* covered: `cross_det_merge`
appears in this document only as a no-op. If your data differs in detector count
or layer count, **stop and ask** rather than adapting a recipe.

**Where the two diverge, it is called out inline as “20-ID:”.** The geometry
recipe, the ω sign discipline and every hard rule apply to both. Three things
are genuinely different and each has cost a day: the dark group, `RhoD` (rule
15), and the calibration entry point (§5, `midas-calibrate-v2 --mode ff`).

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
| [`phase-1-geometry.md`](phase-1-geometry.md) | §2–§5h — ω sign, metadata, dark, `SkipFrame`, energy, distance, calibration, and **`tx`/`Wedge` from grains (§5h)** | the long one; most silent failures live here |
| [`phase-2-configure.md`](phase-2-configure.md) | §6, §6b, §6c, **§6d (`RhoD`)**, §10 — parameter file, `RingThresh`, `MinPeakSNR`, key reference | when writing `Parameters.txt` |
| [`phase-3-run.md`](phase-3-run.md) | §7, §12 — running the pipeline, resume traps, reproducibility check | when launching |
| [`phase-4-read-report.md`](phase-4-read-report.md) | §8–§8b, §11, §14–§14c — `Grains.csv` checks, validation buckets, report, done-means | when a result exists |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptom → discriminating test → cause → lever | **when something looks wrong** — indexed by symptom, not by step |
| [`RUNBOOK.md`](RUNBOOK.md) | §R1–§R3 — where it runs, what healthy looks like *with conditions*, and the current pick-up point | on resume, and before quoting any number as "normal" |
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | evidence, measurement ledger, **retracted claims** — Lab Notebook §1–§7 the 1-ID campaign, **Lab Notebook §8 the 20-ID Varex campaign** | before re-investigating anything |
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
| `ImTransOpt` unknown for this detector, with no prior geometry and no asymmetric feature | a wrong flip mirrors the reconstruction and neither the grain list nor the calibrant strain shows it — the mirrored fit scored the *better* strain (§3f) |
| the data is neither 1-ID GE nor 20-ID Varex, or is multi-panel / multi-layer | every field map and geometry recipe below assumes one of the two configurations in the scope table (header, §3) |
| **no calibrant** file in the folder | there is no geometry without one, and `DetZ` is not a substitute (§0b, §4b) |
| any package **below floor** after §0 | three of them produce plausible wrong answers, not errors (§0) |
| calibrant strain **> 100 µε** after §5 | hard gate; a converged fit above it is not usable (rule 6) |
| the ring overlay does not match the frame | the fit is on the wrong rings; nothing downstream can detect it (§5d) |
| the dark reads all zero in the zarr **and the data frames still carry the pedestal** | every threshold returns 0 peaks; tuning `RingThresh` cannot fix it (§3d). **Check both halves.** On 20-ID Varex `exchange/dark` in the zarr is all zeros *by design* while the data is already dark-subtracted (raw frame mean ~1850 → zarr ~0.6): that is cosmetic, not the fault |
| `nFrames` ≠ logged frames − `SkipFrame` | something is skipping twice or not at all; ω is shifted either way (§3e) |
| `RhoD` is not ≈ `corner_px × px`, or `hkls.csv` reaches ring ≥ 500 | the indexer's ring tables are fixed at 500 and older builds write past them; you get 0 seeds indexed and exit 0 (rule 15, §6d) |
| grain positions **pile up** at ±`Rsample` or ±`Hbeam`/2 | the envelope is binding — and the fix is forbidden to you by rule 9 (§6, §8b) |
| **every** grain's strain sits at `1.000e+04` µε with `RMSErrorStrain` ~`1e36` | the strain column is not a measurement — `IDsHash.csv` is missing and d₀ was taken as 0. Grain count, positions, orientations and completeness are all still correct, so nothing else looks wrong (DIAGNOSIS *Strain pegged at its bound*) |
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
10b. **Indexing and refinement are BOTH c-omp — there is no GPU backend for either
   stage, in FF or PF.** This is now **enforced, not advisory**: `--indexer-backend`
   and `--refine-backend` accept `c-omp` and nothing else, and
   `_require_comp_backends` (`midas_pipeline/config.py:666`) re-checks from
   `__post_init__` so a library caller cannot bypass it. Both default to `c-omp`, so passing them is
   optional. The refiner *used* to default to python + torch + CUDA, which made a
   log line reading `indexing(FF, c-omp)` mean the run was silently half on the GPU
   path; it died with a bare `CalledProcessError` and **no child traceback**, and
   every retry cost a full re-index — ~90 min a time on a 24 900-seed layer, twice
   (§7). The Python refiner is known-broken and is not a fallback. The flags that
   tuned it (`--refine-solver`, `--refine-loss`, `--refine-mode`,
   `--pf-refine-mode`, `--use-bounds`, `--bound-*`) are **removed**.

   **Do not talk yourself out of c-omp because its source has no `tx`.** The
   detector roll is applied in `transforms` (`midas_transforms/fit_setup/core.py:376`),
   which corrects the spots *before* either backend sees them; a refiner that
   re-applied it would double-count. Neither backend has `tx` in its geometry
   model and neither needs one (§7).

10c. **`--pg-mode spot_aware` is DISABLED — do not try to re-enable it.** The
   default is `c_parity`, which reproduces the C reference (datasetA Ni: 6150
   grains vs C's 6138, matched pairs agreeing to 0.0000° and 0.000 µm). Adjudicated
   against EBSD on `shade_LSHR`, `spot_aware` bought **+0.1 pp of recall for
   −11.6 pp of precision**, and only **7.2 %** of the 691 grains it added had an
   EBSD partner (vs 80.4 % for the shared population). On a 20-ID alumina rod it
   returned 1652 grains against 533 and put **4.1 % of them outside the physical
   sample**. **Why the branch does this is not yet diagnosed** — it is disabled on
   its output, not on a root cause, so treat any re-enabling proposal as an open
   investigation, not a settled question (§7, Lab Notebook §2e).

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

This one belongs with the first group — it is about distrusting the parameter
file — but is numbered last so the references above keep their numbers:

15. **`RhoD` is the beam-centre-to-farthest-corner distance in MICRONS, and it is
    two things at once (§6d).** It normalises the distortion polynomial
    (`ρ = R_µm / RhoD`) *and*, aliased to `MaxRingRad`, caps hkl generation.
    Compute it, never copy it: `corner_px × px`, which
    `midas-calibrate-v2 --mode ff` does for you. Measured on 20-ID: `RhoD
    2000000` against a 2880 px / 150 µm detector (true value **309 538**)
    generated **745 rings**, overran the indexer's fixed 500-ring array, and
    indexed **0 of 4569 seeds while exiting 0**. The same file with `RhoD`
    corrected gives 208 grains. Two further traps ride on it: the exporter
    `ff_paramstest_from_auto_result` does **not** replace `RhoD`, so a bad value
    survives recalibration; and the damage is **material-dependent** — cubic
    nf709 generated only 70 rings from the same wrong value and reconstructed
    fine, so "it worked on my other sample" is not evidence the value is right.

16. **Ring numbers come from the RUN's own `hkls.csv` — never regenerate them.**
    MIDAS's ring numbering diverges from a fresh `generate_hkls()` above ring
    ~19, and it **fails silently**: on 20-ID alumina, requesting "ring 30"
    believing it was (4,-2,6) at 21 % relative intensity actually selected
    (4,0,-2) at 2.7 %, and "ring 32" selected (2,-1,12) at 0.33 % — nearly
    extinct. They returned 115 and 88 spots, exactly right for what was
    *actually* asked for. Read ring → (hkl, radius) out of the run's
    `hkls.csv` and match on **(h,k,l)**, never on ring number. Two corollaries:
    rings closer than the radial margin get **duplicated, not split** (two rings
    3.9 px apart emitted every one of 2930 peaks **twice**, under both labels,
    byte-identical `YLab`/`ZLab`/`Omega`); and never audit ring signal with a
    **max-projection**, which is dominated by hot pixels — mask any pixel firing
    in more than a few percent of frames first.

17. **A beam narrower than the sample caps what is reconstructable (§ DIAGNOSIS
    `split.illumination_radial`).** The beam is fixed in the lab while the sample
    rotates, so a grain at radial offset *r* is lit only `f(r) =
    (2/π)·arcsin(hw/r)` of the rotation — 13 % at r = 250 µm for a 100 µm beam.
    Below `MinMatchesToAcceptFrac` (default 0.5) such a grain can only be
    accepted on coincidental matches, so the output is not "fewer grains" but
    **fabricated** ones. Measured on a 1 mm alumina rod: **zero** well-fitting
    grains beyond r = 100 µm, and grains lit 8.8 % of the rotation had 85 % of
    their assigned spots recorded while outside the beam. Report the near-axis
    subset and say so; the full cross-section needs a translation scan, not a
    re-run.

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
| `ImTransOpt` differs between calibration and recon | geometry mirrored relative to the fit. Measured: every wrong variant scored a **better** strain than the correct one (47.2 and 55.6 µε vs 58.2), with BC landing exactly on `N-1 − BC` | §3f, §6 |
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
| `RhoD` copied from another sample's file | **material-dependent** silent kill: hexagonal ti7al generated 745 rings and indexed 0 of 4569 seeds; cubic nf709 generated 70 from the same value and was fine | rule 15, §6d |
| `hkls.csv` reaching ring ≥ 500 on a pre-fix `midas-index` | `RingHKL[Rnr]`/`RingTtheta[Rnr]` were written unbounded, through the `data`/`ndata` bin pointers. Fixed builds skip the row and warn; older ones corrupt and exit 0 | rule 15 |
| `ImTransOpt` read with `getattr(v1, ...)` | `CalibrationParams` does not carry it — it lands in `.extra`. Reading it the obvious way silently calibrates with **no** image transform, mirroring an axis. Measured: BC_z 1411.59 instead of 1467.46 (= 2879 − 1467.46) **at 55.6 µε, PASS** — a better strain than the correct fit's 58.2 | §5d |
| a refined parameter sitting exactly on a bound | not a measurement — the fit ran out of room. Seen three times: `Wedge` at +5.0 from a misread ω key, `iso_R4`/`iso_R6` at +0.05 from six grains. `midas-joint-ff-calibrate` ≥ 0.1.9 names it and exits 1 | §5h |
| distortion "refined" by `grain-tx` on 0.1.8 | `v2_coeffs_from_named` builds a numpy array via `float(v)`, detaching the graph — the harmonics got **zero gradient** and never moved, while being reported as refined | §5h |
| `grain-tx` on a parameter file that says `OmegaStart` | pre-0.1.7 read only `OmegaFirstFile` and took the frame count from `NrFilesPerSweep` (= 1 on one-file-per-sweep). 5 matched spots of 12 355, `Wedge` railed at its bound, `rc=0` | §5h |
| running a CLI from the wrong environment | `--mode ff: invalid choice` and friends are version, not syntax. The version number alone may not distinguish builds — check content, not `--version` | §0 |
| version floors read from this document instead of from the tree | **eight** declarations rose for silent-wrong-answer reasons in the nine days after this file was written, across five packages | §0 |
| `--refine-backend` left unset | **CLOSED** — both backends are now `c-omp`-only and `c-omp` by default, enforced in argparse *and* in `PipelineConfig`. Historically the refiner defaulted to python+torch+CUDA while the indexer defaulted to c-omp, so the run went silently half onto the GPU path, died with a bare `CalledProcessError` and no child traceback, and each retry cost a full re-index. A handbook or script still passing `--refine-backend python` is pre-fix | rule 10b, §7 |
| grain counts compared across `--pg-mode` values | the modes are **not** interchangeable, and `spot_aware` is now **disabled** for manufacturing grains — 4.1 % of them outside the physical sample on a 20-ID rod, and only 7.2 % of the ones it added over `c_parity` had an EBSD partner. A higher grain count from it was never evidence of better recall | rule 10c, §7 |
| `processgrains_diagnostics.h5` missing on a default run | **is** a version problem as of `midas-process-grains` 0.9.2, which makes `c_parity` write the residual sidecar. Below 0.9.2 the default mode returned without writing it and `--generate-h5` did not change that — so an older run's missing sidecar indicates nothing about that run's quality. `mode=physics` still has none (`v4_pipeline` never reads FitBest) | §7 |
| a grain count one short of what the seeds justify, on `midas-process-grains` < 0.9.3 | **the LAST seed was silently deleted from every c-omp run.** `FitUnified.c` pwrites only `nSpotsComp` records per seed at a full-slot stride, so `FitBest.bin`/`ProcessKey.bin` end mid-slot; the readers floor-divided by the stride and truncated, and `c_parity_run` then truncated everything else to match, on the false rationale that the dropped seed "gets NrIDsPerID=0 anyway". Measured: a 56,125-seed Ni layer lost seed 56,124 (SpotID 245283, `keep_flag` set, completeness 0.777) — an ordinary live grain, gone with no warning, while `OrientPosFit.bin`/`Key.bin` both saw it. Fixed in 0.9.3 (zero-padded tail); the "truncating to common length" line no longer appears | §7 |
| `DiffPos` re-derived from `residuals/spot_table` | **it does not reconcile.** Per-spot values reproduce the refiner's own FitBest `DiffLen`/`DiffOmega`/`InternalAngle` exactly, and per-grain means reproduce `Grains.csv` `DiffOme`/`DiffAngle` to 6e-7 — but `DiffPos` is not the mean of the per-spot `DiffLen` (median ratio 0.61, datasetA Ni, reproduced inside single-invocation c-omp `Results/FitBest_*.csv`). Different quantities; cause not diagnosed. Provisional | §7 |
| `MinNrSpots` < 3 on a full rotation | a 2-spot "grain" is **under-determined** (orientation has 3 DOF) — the refiner fits it, reports a position and lattice, and it survives every downstream filter, diluting every population statistic. **≥ 3 always; 2 only for a partial rotation; never below 2** | §6 |
| matching margins tightened to "improve" `DiffPos` | **Do not.** `MarginRadial` / `MarginEta` / `MarginRadius` / `MarginOme` bracket what **indexing does not yet know** — position, orientation and strain are all still coarse at that stage (`StepSizePos 100`, `StepSizeOrient 0.1`) — **not** the spot size. Two consequences. (1) Tightening cannot lower `DiffPos` for a given grain: the **refiner never reads these values**, and applying `500 / 500 / 0.5` produced **bit-identical** refiner output. (2) What it actually changes is **which candidates survive indexing**, dropping the ones whose position the coarse search placed least well. On a sample wider than the beam that deletes off-axis grains and *looks* like an improvement while biasing the result toward the rotation axis. Sizing them against the spot width (2–3 px) is the specific error to avoid | rule 9, §6 |
| a run passing `--refine-solver` / `--refine-loss` / `--refine-mode` / `--use-bounds` / `--bound-*` / `--pf-refine-mode` | **removed** (2026-08-19), so argparse fails with `unrecognized arguments`. Every one configured the in-process PyTorch refiner, which is disabled; the c-omp refiner has no configurable solver or loss. Delete the flag — there is no replacement and nothing is lost | rule 10b, §7 |
| a backend judged by grepping its source for `tx` | `tx` is applied in `transforms`, not in indexing or refinement — **no** backend carries it, python included, and re-applying it downstream would double-count. Concluding "c-omp cannot see `tx`" abandons the only supported fast path for a non-defect. The deprecated `FF_HEDM/src/FitPosOrStrainsOMP.c` is also not the binary c-omp runs (that is `midas_fit_grain/c_src/FitUnified.c`) | §7 |
| `--only` given a **comma-separated** list | `--only` is *repeatable*, not comma-separated. `--only a,b` is read as one stage named `"a,b"`, matches nothing, and the run reports **success with zero stages executed** in ~1 s. The orchestrator validates that `--only` omits required *upstream* stages, but does **not** validate that a stage name exists | §7 |

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
| 3b | **Settle `ImTransOpt`.** | §3f | Must be identical in calibration and reconstruction. Wrong, it mirrors everything, and the calibrant strain gets *better*, not worse. |
| 4 | **Energy, then distance.** | §4 | Calibration needs λ. `DetZ` is only a seed. |
| 5 | **Calibrate on the calibrant**, overlay the rings. | §5 | Gate: ≤ 100 µε, and the overlay is mandatory. No geometry, no reconstruction. |
| 6 | **Zip the sweep only** — `--only zip_convert`. | §7 | `midas-ring-thresh` reads the zarr, so the zarr must exist before the threshold can be measured. |
| 7 | **Measure `RingThresh`** on that zarr; set it. | §6b | A template threshold is meaningless for your detector and exposure. |
| 8 | **Build the parameter file.** | §6 | Needs the geometry (5), the scan definition (3) and the threshold (7). Replace the calibrant's lattice with the sample's. |
| 8b | **Check `RhoD`, and the ring count once `hkl` has run.** | §6d | `RhoD` is both the distortion normalisation and the hkl cap. Wrong, it can index **0 seeds and exit 0** (rule 15). |
| 9 | **Run the rest of the pipeline.** | §7 | |
| 9b | **Refine `tx` and `Wedge` from the grains, then re-run 9.** | §5h | The two the calibrant is structurally blind to. Measured: 208 → 226 grains, grain-Z scatter halved. Optional, but do it before quoting positions. |
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
