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
  a refined parameter sitting on its bound, or every grain's strain railed at
  its bound. Covers 1-ID with a monolithic GE
  panel and 20-ID HT-HEDM with a Varex, single panel and one layer at a time;
  stops and asks outside that.
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

**1-ID / GE** (`.ge5.h5`, 2048² @ 200 µm) and **20-ID HT-HEDM / Varex**
(`.vrx.h5`, 2880² @ 150 µm). The spine's scope table lists what differs; the
geometry recipe, the ω discipline and every hard rule apply to both. Three
things are genuinely different and each has cost a day:

* the dark lives in **`/exchange/bright`** on the Varex — `/exchange/dark`
  exists and is all zeros;
* **`RhoD`** must be computed, never copied (spine rule 15, §6d). Wrong, it
  indexes **zero seeds and exits 0**, and whether it bites at all depends on the
  sample's symmetry;
* calibrate with **`midas-calibrate-v2 --mode ff`** (§5), which writes the
  parameter file and fixes `RhoD` for you.

After a first reconstruction, **§5h** refines `tx` and `Wedge` from the grains —
the two a powder calibrant is structurally blind to.

## When something looks wrong

Go to **`manuals/ff-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever.
It is indexed by *symptom*, not by step, because the step that produced a symptom is
rarely the step you are on. Every entry carries a test that can come back the other way;
an entry that cannot exonerate the cause it names does not belong there.

Before re-investigating anything, read **`manuals/ff-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one.

## Sibling doc sets

`manuals/nf-hedm/` (near-field, skill `nf-hedm`), `manuals/dfxm/` (dark-field X-ray
microscopy, skill `dfxm`), and, in the LaueMatching repository, `scripts/pipeline/laue/`
(skill `laue`).
