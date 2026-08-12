---
name: pf-hedm
description: >-
  Take a scanning-3DXRD / pf-HEDM dataset from raw frames to a per-voxel grain
  map and, optionally, per-voxel peak-shape strain: survey the scan, fix the
  omega sign and positions.csv convention, calibrate, seed indexing from a
  far-field Grains.csv, run the MIDAS PF pipeline, fit peak shapes with
  midas_pf_odf, and report with provenance. Use when asked to reconstruct,
  index, calibrate or diagnose a pf-HEDM / scanning-HEDM / scanning-3DXRD
  beamtime, when handed a folder of per-translation detector files with a
  positions list, or when a pf-HEDM voxel map or per-voxel strain looks wrong.
  Covers single-panel, one-layer scanning HEDM; stops and asks outside that.
---

# pf-HEDM reconstruction + per-voxel strain

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites, gets checked by the repo's own hooks, and
stays usable without this skill.

## Start here

Read **`manuals/pf-hedm/README.md`** — the spine. It is the only file meant to stay loaded:
scope gate, install gate, the order of operations, the hard rules, and the halt conditions.
It carries an index telling you which file holds which section, and you open those as you
reach them.

Then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # one detector file per translation step
Metadata / geom: <ABSOLUTE PATH>   # calibration + positions, or "find it"
Sample material: <e.g. FCC Ni / unknown, tell me from the data>
Goal:            grain map only | grain map + per-voxel strain
```

## Three things to know before you start

1. **Run the install gate first** (spine §0), including the c-omp→pf-odf bridge check. A
   c-omp PF refine without it silently refines nothing.

2. **"Get back to me if you get stuck" does not fire here.** A mirrored voxel map, a c-omp
   refine that wrote zero voxels, and a strain fit on a raw pedestal all finish and look
   right. The spine carries **named halt conditions** — halt on those whether or not
   anything seems wrong. The ω sign and the `positions.csv` convention cannot be checked
   after the fact.

3. **pf-HEDM is scanning 3DXRD, not far-field or near-field.** Confirm the technique in
   phase 0 before any recipe applies — an FF recipe on a scanning dataset fails silently
   (FF refines position, PF fixes it to the voxel grid; the seed file formats differ).

## When something looks wrong

Go to **`manuals/pf-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
keyed by symptom. Before re-investigating, read **`manuals/pf-hedm/LAB_NOTEBOOK.md`** §5 —
several attractive hypotheses (illumination gating, the naive alignment gate) are recorded
there as refuted, with the measurement that killed each.

## The distinctive phase

pf-HEDM is the only HEDM doc set with a **per-voxel peak-shape strain** phase
(`manuals/pf-hedm/phase-4-strain.md`, `midas_pf_odf`) — fitting the full Bragg-peak shape
per voxel rather than its centroid. Read the envelope before promising a strain map: on an
attenuated scan the strain is signal-limited and its magnitude is provisional.

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/dfxm/` (dark-field X-ray microscopy, skill `dfxm`), and in the
LaueMatching repository the `laue` skill.
