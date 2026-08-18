---
name: dct-tt
description: >-
  Take a diffraction-contrast-tomography (DCT) or topotomography (TT) scan from raw
  detector frames to grain orientations, grain shapes, a grain map, or a single
  grain's intragranular orientation field: survey what the frames actually record,
  derive the geometry from the data because the header does not carry it, segment
  and Friedel-pair the spots, self-calibrate lattice and lambda/2a, index grains with
  tolerances set from an omega-scrambled null, assign spots forward, reconstruct
  shapes by SIRT against a spot-swap null, and for TT solve the goniometer tilts,
  check reflection-pair conditioning before promising a rotation tensor, and recover
  the 12-component deformation field. Use when asked to reconstruct, index or
  diagnose a DCT or topotomography scan, when handed rotation-series detector frames
  with discrete diffraction spots, or when a grain map or intragranular field looks
  wrong. Powder-like continuous rings are out of scope and redirect to xrd-ct;
  spotty far-field patterns without tomographic translation redirect to ff-hedm.
---

# DCT and topotomography: grain maps and intragranular fields

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository so
it lives beside the `midas_dct_tt` code it cites and stays usable without this skill.

## Start here

Read **`manuals/dct-tt/README.md`** — the spine. It is the only file meant to stay loaded:
scope gate, install gate, the order of operations, the hard rules, and the halt conditions. It
carries an index telling you which file holds which section; open those as you reach them.

Then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # HDF5/EDF detector frames
Metadata / geom: <ABSOLUTE PATH>   # or "find it" -- usually there is nothing usable
Material:        <e.g. fcc / hcp / unknown -- tell me from the data>
Goal:            grain map | one grain's 3-D shape | intragranular orientation field
```

## Which technique am I looking at?

Both are rotation series on the same instrument, and they are told apart by **what the
grain does as ω advances**:

* **DCT** — grains *flash*. A reflection satisfies Bragg at a few discrete ω and is dark
  elsewhere, so a spot appears for a handful of frames out of thousands. Many grains at
  once, one shape each.
* **TT** — one grain diffracts *continuously* for the whole 360°, because the goniometer was
  tilted to put its **G** along the rotation axis. Hundreds of topographs of one grain, and
  the only route here to what is going on *inside* it.

If the rings are continuous rather than discrete, it is neither — that is `xrd-ct`.

## Six things to know before you start

1. **The header does not carry the geometry, and what it does carry may be wrong.** Every
   geometric quantity on the DCT scan run end-to-end here was derived from the data. The one
   number the header did give was the *sensor* pixel rather than the imaging pixel — wrong by
   a factor **6.65**. A second, TT, dataset had a metadata pixel size wrong by **2×**. Both
   errors produce complete, plausible, entirely wrong reconstructions (phase 1).

2. **Set tolerances from a null or not at all.** At a 4× looser indexing margin, real data and
   an ω-scrambled null indexed **identically** — 2761 of 2902 seeds, completeness 0.250 on
   both. That was believed and written down before the null was run. The adopted margin sits
   above the null's *maximum* completeness of 0.069 (phase 2, `LAB_NOTEBOOK.md`).

3. **Most of a finished grain map is not measurement.** In the adopted map **86 % of the
   labelled volume is dilation**, and ~**22 %** of the domain is uncontested at *any*
   threshold. Lowering the threshold converts uncontested → contested, not uncontested →
   measured. Quote both fractions or the map overstates itself (phase 3).

4. **For TT, conditioning is decided before any photons are collected.** A rotation tensor
   needs two reflections separated by γ ≥ 60°. A 13.3° pair gives sensitivity eigenvalues
   `[0.0067, 0.4933, 0.5]` — one component 75× worse than the others, and no counting
   statistics repair it. `midas_dct_tt.rotation_coverage` and `.goniometer` answer this from
   the grain's orientation and the stage envelope alone (phase 4).

5. **A fit converging on your mask is not evidence the mask is right.** A deliberately wrong
   support scored **0.810** against the true support's **0.860**, and the two fields agreed at
   NCC **+0.940** on their overlap. The data determine the field; the domain is much weaker
   than it feels. Run the wrong-support control before believing a field (phase 4).

6. **Five retractions here came from broken inputs producing plausible numbers**, not from
   new physics: a forward model placing spots at the antipode, an ω conversion that silently
   discarded half a 360° scan, and three conclusions drawn on top of those two. Read
   `LAB_NOTEBOOK.md` before re-investigating anything.

## Starting from raw frames

`phase-2-dct-index.md` (segment → Friedel pairs → self-calibrate → index) and
`phase-3-dct-shapes.md` (assign → per-frame extraction → SIRT → map) carry the DCT spine;
`phase-4-tt.md` carries topotomography end to end. **`RUNBOOK.md` walks a real
multi-tens-of-GB scan from nothing to 862 indexed grains**, verified against the data rather
than written from the API, and `INSTRUMENT.md` holds the detector and stage conventions that
cannot be recovered once a reconstruction exists.

Three things from that runbook worth knowing before you start:

* **Feed the indexer Friedel-pair virtual spots, not raw spots.** `(y+y')/2 − c` and
  `(z−z')/2` are what a point grain on the axis would give, so the sample-radius and
  beam-height parameters drop to the floor. Raw-spot ring assignment was impossible: rings
  84 px apart, grain position moving a spot by up to 150 px.
* **The grain position is fixed in the SAMPLE frame, not the lab.** Each pair flashes at its
  own ω. Adding `Rz(σω)` to the design matrix cut the residual 52 → 41 µm.
* **Threshold with Otsu, never a fraction of the max.** Streak artefacts sit far above the
  grain's own level; `0.5 × max` reported **60 µm grains as 6 µm**.

## When something looks wrong

Go to **`manuals/dct-tt/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever, keyed
by symptom rather than by step. Its first entry is the most dangerous symptom in the
technique: **a grain map that looks clean and space-filling**, which is exactly what dilation
of a sparse measurement produces.

Before re-investigating anything, read **`manuals/dct-tt/LAB_NOTEBOOK.md`** — **seven** results
are recorded there as retracted, and one mechanism as withdrawn while its measurement stood.
None died of new physics.

## Read the envelope before promising an answer

**`manuals/dct-tt/ENVELOPE.md`** separates what the *measurement* can determine from what
these *recipes* apply to. A dataset can be squarely in scope and still unable to support what
is being asked of it: only `λ/2a` is measurable without naming the material, overall
handedness is undecidable from a single scan, and an intragranular *tensor* needs a reflection
pair the stage may not be able to reach.

## Scope

**Discrete-spot rotation series**: DCT (grains flashing at isolated ω) and TT (one grain
diffracting continuously on an aligned axis), through the `midas_dct_tt` chain. Continuous
powder rings are `xrd-ct`. A far-field spot pattern with no tomographic content is `ff-hedm`.

**Status of the capabilities.** Geometry self-calibration, Friedel pairing, indexing, forward
assignment and SIRT shapes are real-data-proven, with nulls. The TT geometry chain reproduces
74 independent real goniometer settings to a median **0.043°/0.050°** against a 25–40° null.
The intragranular field is real-data-proven on **one grain** with a poorly conditioned
reflection pair, and its resolution limit — a recovery window of **1.2–2.0 µm** — is measured,
not assumed. Absolute strain has **not** been demonstrated on real data. Treat that as
calibration for your priors.

## Sibling doc sets

`manuals/ff-hedm/` (far-field HEDM, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/pf-hedm/` (scanning 3DXRD, skill `pf-hedm`), `manuals/xrd-ct/`
(powder-like diffraction tomography — **the right doc set if your rings are continuous**,
skill `xrd-ct`), `manuals/dfxm/` (dark-field X-ray microscopy, skill `dfxm`), and in the
LaueMatching repository the `laue` skill.

## Log a halt

Technique skills carry no verdicts of their own, so there is one thing worth logging: when the
doc set **stops** you. A halt is a designed outcome, and how often the gates fire on real data
is the only evidence that they are load-bearing rather than decorative.

```bash
~/.claude/bin/skill-log --skill dct-tt --event invoked --verdict INVOKED \
  --subject "<which gate halted the work, or 'ran to completion'>" \
  --evidence <the file or reading that triggered it> \
  --note "<what would unblock it>"
```
