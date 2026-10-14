---
name: ff-hedm
description: >-
  Take a far-field HEDM (FF-HEDM) dataset from raw frames to a validated grain
  list: survey the folder, establish the omega sign, calibrate on a calibrant,
  measure RingThresh, run the MIDAS pipeline, refine the powder-blind tx from
  the grains and measure the rotation-axis Wedge and OmegaStep from Friedel
  pairs, read Grains.csv, and report with provenance. Use when
  asked to reconstruct, index, calibrate or diagnose an FF-HEDM / far-field
  3DXRD beamtime, when handed a folder of .ge5.h5 / GE or .vrx.h5 / Varex
  detector frames, or when an FF reconstruction looks wrong — including a run
  that finishes with zero grains, zero seeds indexed, a crash in process-grains,
  a refined parameter sitting on its bound, a population residual that will not
  come down, or every grain's strain railed at its bound. Covers 1-ID with a
  monolithic GE panel and 20-ID-D HT-HEDM with a Varex, single panel and one
  layer at a time; stops and asks outside that.
---

# FF-HEDM reconstruction

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites, gets checked by the repo's own
pre-commit hooks, and stays usable without this skill.

## Start here

Read **`manuals/ff-hedm/README.md`** — the spine. It is the only file meant to stay
loaded: scope gate, install gate, the order of operations, the hard rules, and the halt
conditions. It carries an index telling you which file holds which section, and you open
those as you reach them.

Then give, or work out:

```
Data folder:     <ABSOLUTE PATH>     # the image tree
Metadata folder: <ABSOLUTE PATH>     # or "find it"
Sample material: <e.g. gold cubes / unknown, tell me from the data>
```

## Seven things to know before you start

1. **Run the install gate first** (spine §0). Several version floors exist only to keep out
   versions that produce plausible wrong answers rather than errors. It is free, and
   skipping it invalidates everything downstream.

2. **"Get back to me if you get stuck" does not fire here.** A mirrored reconstruction, a
   wrong ring assignment, and an unrefined position all finish and look right. The spine
   carries a list of **named halt conditions** — halt on those, whether or not anything
   seems wrong.

3. **The order is not optional.** Two steps (the ω sign, the frame-0 skip) cannot be
   checked after the fact, because getting them wrong changes the answer without changing
   anything you would look at.

4. **A layer that "succeeds" can have silently lost frames.** Transient filesystem
   permission errors made the peak search zero-fill 847 of 1441 frames of one layer (3
   grains against ~2,900) with the stage marked done. `grep -c 'failed to read'
   driver.log` on every layer, and compare grain counts across a series.

5. **20-ID metadata lies in two places.** `scan_parameters` can describe the previous test
   scan (take ω from the `aero` readback), and a dark file can sit ~100 counts above the
   data floor (build the dark from the layer's own frames; a 1-frame dark is zeroed by
   `SkipFrame`, store two). Older `ring-thresh` printed `RingThresh N 500` when criterion C was
   clean only at the sweep ceiling (a rail, not a floor): read the per-ring criterion lines and do not paste a ceiling value.

6. **Judge a refit on the same grains.** A `tx`/`Wedge` refit improved the whole-population
   spread and match rate purely through 245 newly indexed grains; the grains that were
   already well indexed did not move. Compare the same grains before and after.

7. **Comparing beam settings**: the settings differ in counts, attenuator and slits as well
   as beam height. Report median completeness and the count at ≥ 0.8, and say "as run".
   Orientation matching against another layer or modality: crystal-side symmetry only.
   Lab notebook §15.

## Two configurations

**1-ID / GE** (`.ge5.h5`, 2048² @ 200 µm) and **20-ID-D HT-HEDM / Varex**
(`.vrx.h5`, 2880² @ 150 µm — the **D** branch; FF runs at D and E and
everything verified here is D, so confirm the branch rather than infer it from
"20-ID"). The spine's scope table lists what differs; the
geometry recipe, the ω discipline and every hard rule apply to both. Four
things are genuinely different and each has cost a day:

* **the dark group is per SCAN, not per beamline** (§3d). Measured inside one
  20-ID beamtime: the gold scan's dark is in `/exchange/dark`, the alumina
  scan's in `/exchange/bright`, the CeO2 calibrant's in `/exchange/dark`.
  Carrying one scan's answer to the next left an 1850-count pedestal in place
  and turned every ring band into a single blob, which reads as *"this sample
  is a powder"*. Measure the three group means on every scan;
* **there is no par file at 20-ID** (D or E), so the ω-sign source the 1-ID recipe
  depends on does not exist — and the ω sign is *coupled* to the detector
  mirror, which powder rings cannot break either. **§2b** settles both with
  three independent physical arguments. Do not adopt another run's answer:
  the two pre-existing 20-ID parameter files disagree, and both produced
  plausible results. **Exception, 20-ID-E (HEXM):** the detector there is
  `s20varex2`, not the D Varex; ω is **positive as logged** and `ImTransOpt`
  is **1** (y-flip only, z not flipped; established on PUP_AML_bt_20id_sep26b by the §2b Friedel test on the first grain
  list). Also **measure `OmegaStep`**: 0.25058 (not 0.25, and not the 0.250514
  periodicity estimate) on `PUP_AML_bt_20id_sep26b`, by Friedel pairs and frame
  timing (§5h). Spine scope block has the details;
* **`RhoD`** must be computed, never copied (spine rule 15, §6d). Wrong, it
  indexes **zero seeds and exits 0**, and whether it bites at all depends on the
  sample's symmetry;
* calibrate with **`midas-calibrate-v2 --mode ff`** (§5), which writes the
  parameter file and fixes `RhoD` for you.

After a first reconstruction, **§5h** fixes the three things a powder calibrant
cannot see, with two tools:

* **`Wedge` and `OmegaStep` — `midas-joint-ff-calibrate friedel-wedge`.** Friedel
  mates diffract exactly 180° apart unless the rotation axis is tilted
  (`ω₂ − ω₁ − 180 ≈ −2·W·cot η`) or the ω scale is off (a constant). It fits both
  jointly and returns the **absolute** Wedge, whatever Wedge the run used. This is
  the `FitWedge.c` method; `FitWedgeParallel.c` is **deprecated** (it charges an
  OmegaStep error to the wedge, with opposite signs above and below the beam).
  Read its split table: one fixed axis tilt gives one wedge in every ring, ω range
  and side of the beam. **Pairs come from the raw peak list by default**
  (`InputAllExtraInfoFittingAll.csv`), independent of the reconstruction;
  `--pairs grains` pairs matched spots, which were selected through `MarginOme`
  around the run's own predictions and leak ~6 % of the run's Wedge. It prints an
  estimated wrong-pair fraction (decoys at 180 ± 3° × P(mate missing)²; never the
  raw decoy count). the 20-ID-E ss_cal run: +0.02457 ± 0.00011°, 36,235 pairs, 0.1 % wrong
  (PROVISIONAL; Lab Notebook §17).
* **`tx` — `grain-tx --refine tx`** (the default), with the Wedge from Friedel
  pinned in its input file. **Do not refine Wedge with grain-tx**: its loss is the
  on-detector (Y, Z) distance with the grain poses frozen at the reconstruction's
  values, and it never compares observed ω, where a wedge acts at first order. One
  pass recovers only part of a wedge (synthetic −0.5: −0.207 from 0, then −0.372,
  −0.462, −0.489 over pipeline re-runs; the 20-ID-E ss_cal run pass 1: +0.0119 vs the
  Friedel +0.02457). Friedel gets it in one pass from any reconstruction. ~40 good
  grains are enough for `tx` (HS).
* **Detector readout timing: `OmegaReadoutShift/Axis/Shape/Centre`.** A panel that
  exposes some columns (or rows) earlier than others puts an ω offset on every spot
  **fixed to the pixel**. Friedel pairs cannot see a column pattern (mates share a
  column) and a calibrant has no ω. Detect it in the post-fit `Omega − OmegaExpPost`:
  remove the `cot η` term, then see whether the per-ring medians collapse against
  column, row or |η| (ss_cal: column 0.0021° vs row 0.0307°, |η| 0.0244°). 20-ID-E
  `s20varex2`: centre-out V along columns, suggested `OmegaReadoutShift -0.156`,
  `OmegaReadoutAxis 1`, `OmegaReadoutShape 0`; 20-ID-D Varex: a split-halves step
  (±0.023-0.030°, uncorrected). The A/B cut post-fit `DiffOme` 12 % but met no
  registered threshold: **INCONCLUSIVE, PROVISIONAL** (§5h). `RMSErrorStrain` is
  nearly blind to ω; judge on `DiffOme` and the tangential residual. To try a shift,
  `refresh_analysis_params` the archive and resume `--from transforms
  --no-refresh-params` (no re-zip).

**Never hand-compose the numbers.** Since `midas-joint-ff-calibrate` 0.5.0 (commit
5773de0f) grain-tx reports **absolute** tx/Wedge whenever the input carries a
non-zero Wedge (the raw-frame path); only tx-only on a Wedge-0 file is relative,
and then its `--out` writer composes for you. Feed the `--out` file forward.
Adding the printed value to the previous pass double-counts.

**Never hand-roll a `tx` scan** — a grid of pipeline re-runs gives no uncertainty
and no gradient, at ~12 min a point. **Always read `matched spots` before
believing the number**: current grain-tx raises on 0 matches; older installs
printed `tx=0.000000, rc=0` and wrote `tx 0` (the silent reset). `--paramstest`
may be the master file or the layer `paramstest.txt` on current packages (the
writer emits canonical `Lsd`/`BC`/`tx` keys and the readers map `LsdFit`/`txFit`
on older layer files); on an older install use the master file.

**Measure `OmegaStep` with friedel-wedge, not from 360° periodicity.** On
`PUP_AML_bt_20id_sep26b` periodicity gave 0.250514, Friedel pairs and the frame
timing both 0.25058 (uniform across the sweep); the 0.250514 used there left a
±0.047° ω scale error.

## If a second modality exists, that is the only real accuracy check

Everything the pipeline reports about its own error is **reproducibility** — repeats that
share the detector, `Lsd`, tilts, wedge, energy and reference lattice, so every shared
systematic cancels. EBSD, NF, or any independent measurement gives **distance from truth**.
**§15g** in `phase-5-trust.md` is the procedure. Four things it will not let you skip:

* **match on orientation alone** — it needs no spatial registration, so the position
  residual afterwards is an independent test rather than a fitted one;
* **print the chance-match rate for your own grain count** before quoting any match
  fraction. With ~3900 grains, 0.5° chance-matches 0.3 % but 2° matches **16 %** and 5°
  matches **95 %** — a loose threshold measures fundamental-zone density, not your data;
* **name the reference's segmentation.** Precision moved **13 points** on one dataset
  (72.3 → 85.7 %) with the reconstruction untouched, purely by re-segmenting the reference;
  half of the apparent false positives were real grains it had merged;
* **stratify by grain size before believing any trend** — a clean monotonic |Z| dependence
  there was entirely confounded by size;
* **ask whether the two modalities sample the same volume** before reading the position
  residual. If the layer is a thin slice through the plane the reference sectioned it is a
  real accuracy (5.04 µm there); if not it is inflated by ~the grain radius. Regress the
  residual on grain size — it *falls* in the first case, *grows* in the second. In the
  same-section case the reconstruction's **Z spread is its Z error** (3.11 µm, *better*
  than in-plane with a thin beam), which is otherwise unmeasurable.

## When something looks wrong

Go to **`manuals/ff-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever.
It is indexed by *symptom*, not by step, because the step that produced a symptom is
rarely the step you are on. Every entry carries a test that can come back the other way;
an entry that cannot exonerate the cause it names does not belong there.

Before re-investigating anything, read **`manuals/ff-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one.

## Sibling doc sets

`manuals/defect/` (**the diffuse field this pipeline discards**, skill `defect`),
`manuals/nf-hedm/` (near-field, skill `nf-hedm`), `manuals/dfxm/` (dark-field X-ray
microscopy, skill `dfxm`), `manuals/tomo/` (tomography and the **coordinate-system
reference**, skill `tomo`), and, in the LaueMatching repository, `scripts/pipeline/laue/`
(skill `laue`).

**Reach for `defect` when the grains are not the answer.** This chain fits detected peaks
and returns grains, discarding everything between them. For a deformed material that
discarded field is the measurement: asterism → dislocation density, rods → planar faults,
discrete `n·G/m` satellites → a polytype, and a per-grain mechanics layer on top. It takes
this pipeline's `Grains.csv` as its input.

**Reach for `tomo` whenever an FF analysis touches the sample volume.** Two FF
quantities depend on it and neither is obtainable from the diffraction alone:
the illuminated volume behind absolute grain **size** (`ENVELOPE` §1 — without
it `GrainRadius` is a ratio against the `Vsample` search bound, relative only),
and any registration of FF grain positions to another modality. That doc set
owns `COORDINATES.md`, the single reference for the MIDAS lab frame across FF,
NF, PF and tomo — and its `LAB_NOTEBOOK.md` records which reconstruction checks
have **no power** on which specimens, which is not guessable from the code.
