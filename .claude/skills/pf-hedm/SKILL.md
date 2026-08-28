---
name: pf-hedm
description: >-
  Take a scanning-3DXRD / pf-HEDM dataset from raw frames to a per-voxel grain
  map and, optionally, per-voxel peak-shape strain, grain positions and the
  sample boundary: survey the scan, fix the omega sign and positions.csv
  convention, calibrate, seed indexing from a far-field Grains.csv, run the
  MIDAS PF pipeline, fit peak shapes with midas_pf_odf, locate the sample edge
  from the spot-count sinogram, filter contaminated sinogram rows, validate the
  map against an omega-shuffle null, and report with provenance. Use when asked
  to reconstruct, index, calibrate or diagnose a pf-HEDM / scanning-HEDM /
  scanning-3DXRD beamtime, when handed a folder of per-translation detector
  files with a positions list, when a pf-HEDM voxel map, grain shape, grain
  position or per-voxel strain looks wrong, or when asked whether a per-voxel
  map, grain count or acceptance threshold is real rather than chance. Covers
  1-ID scanning and 20-ID HT-HEDM Varex, single-panel and one layer at a time;
  stops and asks outside that. Grain SHAPES from the sinogram reconstruction
  are a known open problem and are gated, not delivered.
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
Station:         1-ID | 20-ID Varex | other (stop and ask)
Sample material: <e.g. FCC Ni / unknown, tell me from the data>
Goal:            grain map only | + per-voxel strain | + grain positions/boundary
```

If the goal is grain **shapes**, say so early — the answer is likely "not from this
measurement" and it is better established before the run than after (see item 4 below).

## Five things to know before you start

1. **Run the install gate first** (spine §0), including the c-omp→pf-odf bridge check. A
   c-omp PF refine without it silently refines nothing. Then identify the station and the
   **code generation** (`INSTRUMENT.md`) — a beamline install may still be the legacy C
   `pf_MIDAS.py`, which lacks two diagnostics and has one 97.7 %-unwritten output.

2. **"Get back to me if you get stuck" does not fire here.** A mirrored voxel map, a c-omp
   refine that wrote zero voxels, and a strain fit on a raw pedestal all finish and look
   right. The spine carries **named halt conditions** — halt on those whether or not
   anything seems wrong. The ω sign and the `positions.csv` convention cannot be checked
   after the fact.

3. **pf-HEDM is scanning 3DXRD, not far-field or near-field.** Confirm the technique in
   phase 0 before any recipe applies — an FF recipe on a scanning dataset fails silently
   (FF refines position, PF fixes it to the voxel grid; the seed file formats differ).

4. **Do not quote grain shapes, and do not assume the grid is all material.** Both are halt
   conditions and both look fine: shapes render as plausible grains (the cause of the
   failure is *unknown* — eleven mechanisms tested, `phase-6-reconstruction.md` §6.7), and
   vacuum voxels inherit a neighbour's orientation and score ~0.92 completeness, so the
   completeness map reads "material everywhere" (`phase-1b-sample-boundary.md`).

5. **Completeness saturates, so a finished map is not evidence.** On a dense layer a
   quarter of voxels sit at exactly completeness 1.0000 and a dense enough spot cloud fits
   almost any orientation. Run the **ω-shuffle null** (`phase-7-validation.md`) and measure
   the **chance ceiling** before quoting a per-voxel result, a grain count, or an acceptance
   threshold — the shipped `MinMatchesToAcceptFrac 0.5` sat *below* the ceiling on both
   layers tested. It is a re-index only, no re-prep. And **never quote a merged-FF grain
   count on scanning data**: its 1-row `positions.csv` turns the beam gate off in the
   matching loop, and the null beat the real arm on every statistic.

## When something looks wrong

Go to **`manuals/pf-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
keyed by symptom. Before re-investigating, read **`manuals/pf-hedm/LAB_NOTEBOOK.md`** §5,
**§7.5** and **§8.7** — a long list of attractive hypotheses is recorded there as refuted,
with the measurement that killed each: illumination gating, the naive alignment gate, six
GPU-crash theories, and on the shape problem eleven mechanisms plus `RawSumIntensity`,
|F|²·Lp normalisation (twice) and edge-padding. **Four of the eleven were later requalified
because they had been scored with dice** — do not score a new attempt that way.

**If two runs of "the same thing" disagree, check `ScanPosTol` before anything else.** The C
adds 0.1 µm to the parsed `BeamSize` before the `BeamSize/2` fallback, so a hand-run without
an explicit `ScanPosTol` silently uses a wider beam gate than the parameter file states —
worth +14.7 % accepted solutions and a changed winner in 10.5 % of voxels. A whole session
was spent wrongly attributing that to a released version change (§8.7). Related:
`paramstest_comp.txt` is written by **two** stages under one filename, and the one on disk is
refinement's, not the indexer's.

## The three distinctive phases

**Per-voxel peak-shape strain** (`manuals/pf-hedm/phase-4-strain.md`, `midas_pf_odf`) — pf-HEDM
is the only HEDM doc set that has one. It fits the full Bragg-peak shape per voxel rather
than its centroid. Read the envelope before promising a strain map: on an attenuated scan
the strain is signal-limited and its magnitude is provisional.

**Reconstruction space** (`manuals/pf-hedm/phase-6-reconstruction.md`) — sinograms, shapes,
and the two shipped diagnostics. Read it even when shapes are not the goal: the
concentration filter (`--sino-conc-threshold 0.35`, calibrated and transferable) took a
grain's fitted position from 5.59 µm to 1.11 µm, and the occupancy flag
(`--out-of-field-occupancy 0.65`) names the grains whose shape cannot be recovered. Both are
**flags and improvements to the point-by-point result, never filters** — deleting flagged
grains took map agreement from 47.8 % to 11.0 %.

**Validation** (`manuals/pf-hedm/phase-7-validation.md`) — the ω-shuffle null, the chance
ceiling, and a threshold-free spatial-coherence screen. Permute ω within `(ring, scan)`,
re-index, and compare: it preserves every marginal the indexer sees, including the beam-gate
statistics, and destroys only the position–ω joint that encodes orientation. On the
reference campaign the PF per-voxel path **passed** (no null voxel above completeness 0.6957
/ 0.8333 against real medians 0.9231 / 0.8943) while **merged-FF failed outright** — its
null found *more* grains than the real data. Cheap: a re-index, no re-prep.

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/dfxm/` (dark-field X-ray microscopy, skill `dfxm`), and in the
LaueMatching repository the `laue` skill.
