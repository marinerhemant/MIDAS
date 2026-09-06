---
name: ff-hedm
description: >-
  Take a far-field HEDM (FF-HEDM) dataset from raw frames to a validated grain
  list: survey the folder, establish the omega sign, calibrate on a calibrant,
  measure RingThresh, run the MIDAS pipeline, refine the powder-blind tx and
  Wedge from the grains, read Grains.csv, and report with provenance. Use when
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

## Three things to know before you start

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
* **there is no par file at 20-ID**, so the ω-sign source the 1-ID recipe
  depends on does not exist — and the ω sign is *coupled* to the detector
  mirror, which powder rings cannot break either. **§2b** settles both with
  three independent physical arguments. Do not adopt another run's answer:
  the two pre-existing 20-ID parameter files disagree, and both produced
  plausible results;
* **`RhoD`** must be computed, never copied (spine rule 15, §6d). Wrong, it
  indexes **zero seeds and exits 0**, and whether it bites at all depends on the
  sample's symmetry;
* calibrate with **`midas-calibrate-v2 --mode ff`** (§5), which writes the
  parameter file and fixes `RhoD` for you.

After a first reconstruction, **§5h** refines `tx` and `Wedge` from the grains —
the two a powder calibrant is structurally blind to. `grain-tx` returns a
**residual**: compose it onto what the run already applied and iterate.

**Use the tool. Never hand-roll a `tx` scan** — a grid of pipeline re-runs gives no
`Wedge`, no uncertainty and no gradient, at ~12 min a point. And **`--paramstest` is
the MASTER param file** (the one `midas-pipeline --params` gets), *never*
`<result>/LayerNr_1/paramstest.txt` — that one names its geometry `LsdFit`/`txFit`,
so the model silently falls back to default geometry and reports
`matched spots=0, tx=0.000000, rc=0`, which reads as "tx is already perfect".
**Always read `matched spots` before believing the number.**

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
