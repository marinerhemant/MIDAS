---
name: xrd-ct
description: >-
  Take a diffraction-tomography (XRD-CT) dataset from raw frames or an integrated
  (R, eta) cake to per-voxel phase, strain and texture maps and a report: survey
  the scan and confirm it is powder-like, confirm the sample-to-detector distance
  against the data, extract per-azimuth area and centroid with a ring-free
  background, map deviatoric strain from centroids, attempt a per-voxel ODF
  (gated by peak-to-background and by a positive control), and report with
  provenance. Use when asked to reduce, reconstruct, or diagnose an XRD-CT /
  diffraction-tomography / chemical-tomography scan, when handed per-translation
  detector files or an integrated cake with an omega list, or when a per-voxel
  strain or texture map looks wrong. Covers APS 1-ID and 11-ID-C style
  acquisition through the midas_dt chain; spotty-ringed samples are out of scope
  and redirect to pf-hedm.
---

# XRD-CT per-voxel strain and texture

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository so
it lives beside the `midas_dt` code it cites and stays usable without this skill.

## Start here

Read **`manuals/xrd-ct/README.md`** — the spine. It is the only file meant to stay loaded:
scope gate, install gate, the order of operations, the hard rules, and the halt conditions. It
carries an index telling you which file holds which section; open those as you reach them.

Then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # detector frames, or an integrated (R, eta) cake
Metadata / geom: <ABSOLUTE PATH>   # calibration + translations/omega, or "find it"
Sample material: <e.g. CeO2 / hcp Ti / unknown — tell me from the data>
Goal:            phase map | per-voxel strain | per-voxel texture (ODF)
```

## Six things to know before you start

1. **Area is a difference; centroid is a ratio — and that decides the deliverable.** Every
   azimuthal quantity is one or the other. Measured at 2 % peak-to-background with no planted
   structure: **area scatter 36 %, centroid scatter 0.85 %**. So a scan that cannot support a
   texture map can still give an excellent strain map. **Measure peak/background in phase 2
   before promising anything** (spine, `ENVELOPE.md` §0).

2. **Look at a radial profile before any analysis.** Four analyses on a real DAC Ti scan were
   invalidated by skipping it. One plot showed the background dominated (contrast 1.17×) and
   that α(101) was a doublet being fitted as one line. Five minutes against four wasted
   analyses (phase 0.5).

3. **Do not trust the sample-to-detector distance from metadata.** On an 11-ID-C CeO₂ scan the
   metadata said 1600 mm, the beamline calibration 1579.5 mm, and the data required
   **1632 mm** — both stored values wrong, by 2 % and 3.3 %, i.e. 20,000–33,000 µε of apparent
   strain. **Halt condition.** Otherwise report *relative* strain only (phase 1.1).

4. **"Static in ω therefore instrumental" is WRONG.** `n_s·ẑ = cos θ_B sin η` carries no ω
   dependence, so a fibre about the rotation axis is *necessarily* static in ω. This inference
   was made here and had to be withdrawn. The real limit is the converse: if the unique axis is
   **not** the rotation axis, the uniaxial model cannot fit the texture by construction, and a
   null means nothing (phase 4.10, `DIAGNOSIS.md`).

5. **A texture claim without a positive control is uninterpretable, not merely unverified.** A
   null could equally mean "no texture" or "cannot see texture". Run
   `scripts/odf_positive_control.py` at the measured contrast; it plants discrete crystallites
   (nothing the fit uses), runs the same extraction and fit, and scores **two separate claims** —
   *detect* and *resolve per voxel*. Detect-only means report a sample-average bound, not a map
   (phase 4.8).

6. **Symmetry is the Laue group, and only even harmonic orders are measurable.** Friedel ⇒
   improper operations map to `-R` rather than being discarded; discarding them
   under-symmetrises the **73** space groups with improper operations but no inversion centre.
   And odd `l` is annihilated for *every* scan design — no extra data recovers it, only a
   positivity constraint. Use `midas_hkls.proper_rotations_from_space_group`; quote
   `SymGSH.ghost_dimension()` rather than hiding it (phase 4.2–4.3).

## Starting from raw frames

`phase-1a-reduce.md` (frames → cached (η, R) cake) and `phase-1b-reconstruct.md` (sinogram →
rotation axis → branch → voxel maps) carry the reduction and reconstruction spine.
**`RUNBOOK.md` walks a real 11-ID-C CeO₂ set end-to-end**, verified against the data rather than
written from the API, and `BEAMLINES.md` holds the per-beamline reach, formats and the
conventions that cannot be recovered from a finished reconstruction.

Three things from that runbook worth knowing before you start:

* **Calibrate UNSEEDED.** No `initial_Lsd`, no `BC_guess` — a hand-supplied beam centre
  *overrides* the auto-seeder and the fit then cannot travel. Measured: **1040 µε against
  47 µε** on the same frame. Bound the fit in **detector pixels**, which cannot bias the
  distance.
* **The stored distance is wrong more often than not.** On the CeO₂ set the metadata said
  1600 mm, the beamline calibration 1579.5 mm, and the data required **1632.2 mm**.
* **Verify the cake axis order by collapsing each axis.** It is (η, R) at 11-ID-C and (R, η) at
  1-ID; both reshape cleanly, so a swap is silent. The radial axis shows sharp rings
  (max/median ~181), the azimuthal one is smooth (~1.03).

## When something looks wrong

Go to **`manuals/xrd-ct/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever, keyed
by symptom rather than by step. Every entry carries a test that can come back the other way. Its
first entry is the most dangerous symptom in the technique: **a per-voxel texture map that looks
structured and plausible**, which is exactly what low-contrast extraction error produces.

Before re-investigating anything, read **`manuals/xrd-ct/LAB_NOTEBOOK.md`** §5 — **four**
results are recorded there as refuted, retracted or invalid, and **three** inferences as
withdrawn. None died of new physics. They died of a windowed sum that was 60–85 % background, a
degrees-of-freedom mismatch that made a comparison meaningless, a plant script that accepted a
random seed and never used it, and an inference that was simply backwards.

## Read the envelope before promising an answer

**`manuals/xrd-ct/ENVELOPE.md`** separates what the *measurement* can determine from what these
*recipes* apply to. A dataset can be squarely in scope and still unable to support what is being
asked of it — §0 is a table mapping peak-to-background onto which deliverables are reachable,
and §3 is the identifiability count that makes an L-sweep read correctly (unknowns grow as `L³`
while rows do not, so an underdetermined fit drives its residual to zero for free).

## Scope

**Powder-like XRD-CT**: sample translated across a beam at each of many ω, rings continuous and
integrated azimuthally, reconstruction on a 2-D voxel grid, through the `midas_dt` chain. If the
rings break into **discrete spots** the sample is coarse-grained and this is the wrong tool —
that is scanning-3DXRD, doc set `pf-hedm`. The dividing line is operational: continuous at the
working (R, η) bin size, or not.

**Status of the capabilities.** Reduction, sinograms, reconstruction and per-voxel peak fitting
are real-data-proven. Deviatoric strain from centroids is real-data-proven but the magnitudes on
the one loaded-cell dataset are **provisional**. The texture operator is validated three
independent ways including against third-party TexTOM — but **no XRD-CT dataset here has yet
produced a positive per-voxel ODF result**: one was refuted, one null was itself refuted, one is
parked. Treat that as calibration for your priors.

## Sibling doc sets

`manuals/ff-hedm/` (far-field HEDM, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/pf-hedm/` (scanning 3DXRD — **the right doc set if your rings are spotty**,
skill `pf-hedm`), `manuals/dfxm/` (dark-field X-ray microscopy, skill `dfxm`), and in the
LaueMatching repository the `laue` skill.

## Log a halt

Technique skills carry no verdicts of their own, so there is one thing worth logging: when the
doc set **stops** you. A halt is a designed outcome, and how often the gates fire on real data is
the only evidence that they are load-bearing rather than decorative.

```bash
~/.claude/bin/skill-log --skill xrd-ct --event invoked --verdict INVOKED \
  --subject "<which gate halted the work, or 'ran to completion'>" \
  --evidence <the file or reading that triggered it> \
  --note "<what would unblock it>"
```
