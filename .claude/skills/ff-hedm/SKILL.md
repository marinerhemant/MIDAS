---
name: ff-hedm
description: >-
  Take a far-field HEDM (FF-HEDM) dataset from raw frames to a validated grain
  list: survey the folder, establish the omega sign, calibrate on a calibrant,
  measure RingThresh, run the MIDAS pipeline, read Grains.csv, and report with
  provenance. Use when asked to reconstruct, index, calibrate or diagnose an
  FF-HEDM / far-field 3DXRD beamtime, when handed a folder of .ge5.h5 or GE
  detector frames, or when an FF reconstruction looks wrong. Covers 1-ID with a
  single monolithic GE panel; stops and asks outside that.
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

## When something looks wrong

Go to **`manuals/ff-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever.
It is indexed by *symptom*, not by step, because the step that produced a symptom is
rarely the step you are on. Every entry carries a test that can come back the other way;
an entry that cannot exonerate the cause it names does not belong there.

Before re-investigating anything, read **`manuals/ff-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one.

## Sibling doc sets

`manuals/NF_HEDM_Handbook.md` (near-field) and, in the LaueMatching repository,
`scripts/pipeline/Laue_Handbook.md`. Neither has been split into a doc set yet.
