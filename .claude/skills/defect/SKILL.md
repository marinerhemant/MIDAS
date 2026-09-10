---
name: defect
description: >-
  Take the diffuse scattering around and between the Bragg peaks of a rotation-series
  diffraction dataset and turn it into a defect inventory: survey what kind of diffuse
  field is present, ingest raw frames to a mask, a polar background and 3-D spots in
  (omega, row, col), index the cloud and audit the orientation for completeness,
  classify every voxel Bragg versus diffuse and close the intensity budget, detect
  fault rods against a matched control and satellite ladders against a texture-safe
  null, fit asterism to a relative dislocation density, and report with the envelope
  attached — including when the envelope says the number asked for is not obtainable.
  Use when asked to quantify dislocation density, stacking faults, twin walls, fault
  rods, polytype satellites, sub-grains or an intensity budget from HEDM data, when
  handed a q-space voxel cloud, or when a diffuse-scattering result looks wrong.
  Getting the grains in the first place is ff-hedm; continuous powder rings are
  xrd-ct; separating bulk from boundary needs pf-hedm or dfxm and is gated here.
---

# Diffuse-scattering defect metrology

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository
so it lives beside the `midas_defect` code it cites and stays usable without this skill.

## Start here

Read **`manuals/defect/README.md`** — the spine. It is the only file meant to stay loaded:
scope gate, install gate, the order of operations, the hard rules, and the halt conditions. It
carries an index telling you which file holds which section; open those as you reach them.

Then give, or work out from the data:

```
Data:      <ABSOLUTE PATH>   # raw rotation frames, OR an existing q-space voxel cloud
Grains:    <ABSOLUTE PATH>   # Grains.csv / orientations, or "index it from the cloud"
Material:  <e.g. FCC Cu a=3.6356 -- or "tell me from the data">
Goal:      dislocation density | fault rods | polytype ladder | intensity budget | sub-grains
```

## Is this the right doc set?

The standard chain (`midas_index` → `midas_fit_grain` → `midas_process_grains`) works on
**detected peaks** and returns **grains**, discarding everything between them. This doc set is
about what was discarded.

* **Getting the grains** is `ff-hedm` (far-field), `pf-hedm` (scanning), `nf-hedm` (near-field).
* **Continuous powder rings** are `xrd-ct`, not this.
* **Separating bulk from boundary** when both scatter to the same reciprocal direction is not
  possible here at all — that is `pf-hedm` or `dfxm`, and the spine halts on it.

What *is* here: asterism → relative dislocation density; rods → planar faults; discrete
`n·G/m` satellites → polytypes; and the intensity budget that puts a denominator under all of
them.

**Detection and fraction are different deliverables, and the second is often unavailable.**
Whether a rod is present, and along which direction, is answerable and null-tested. What
*fraction of the intensity* it carries needs every diffuse voxel attributed to a reflection,
and that fails whenever the predicted node cloud is finer than the diffuse feature — on the
reference sample 0.0688 Å⁻¹ node spacing against a 0.05–0.15 Å⁻¹ halo, so neither distance nor
direction can attribute a voxel. Check that before promising a fraction (`ENVELOPE.md` §1a).

## Fourteen things to know before you start

1. **Closure is arithmetic; attribution is the claim.** An auto-classifier on the reference
   sample reported **99.8 % intensity-budget closure** and was **~18 % wrong**. At high `|q|`
   the 18°-mosaic node cloud of 232 orientations makes `5G/3`, `220`, asterism and fault-rod
   tails overlap in *every scalar feature*, so no scalar decision tree separates them. Use the
   structural classifier, and quote fractions only with the separation method
   (`phase-3`, `LAB_NOTEBOOK.md` R1).

2. **In a textured sample, a random-direction null is not a null.** A 15° cone diluted <5°
   satellites and the "random" baseline was contaminated by 992 textured grain ⟨111⟩ — signal
   in the denominator. Together they produced a **false negative on a real 9R phase**. The
   package's old estimator now **raises by default**; use
   `polytype.satellite_excess.satellite_radial_excess`, which is count-normalised at the same
   `|q|` and carries a discriminator (`phase-4`, `DIAGNOSIS.md` 1).

3. **The indexer's grain count is not a grain count.** ~230 "grains" per layer on the
   reference sample are a **~100× over-fragmentation** of two Σ3-related families at ~18°
   mosaic. Per-grain statistics over those are statistics over fragments (`DIAGNOSIS.md` 3).

4. **A width is a lower bound while the mosaic is in it.** Satellite FWHM 0.075–0.12 Å⁻¹ gives
   L ≈ 5–10 nm, quoted as a **lower bound** throughout.
   `rod_profile.transverse_width` returns a bound with `coherence_length_A = None` when the
   feature is resolution-limited. That is the honest output; do not convert it back
   (`ENVELOPE.md` §2).

5. **Radial coincidence is worthless in a multi-grain sample.** `5G/3` coincides radially with
   `220`; in full 3-D the nearest `220` of any of 232 grains is **0.27 Å⁻¹** away. Decide by
   3-D vector distance to every allowed reflection of every grain, and check the forbidden
   shells are empty as the corroborating half (`DIAGNOSIS.md` 8).

6. **Summing over ω can hide the thing you are looking for.** The ω-sum superimposes every
   grain's rod at every orientation and smears a 1-D rod into a ring. A real ⟨111⟩ relrod was
   reported **absent** on that evidence, then reinstated by per-frame, per-grain-cluster
   back-projection — bright pixels on integer hkl at median 0.11, PCA 3.7° from ⟨111⟩. Look
   at raw frames before concluding absence (`DIAGNOSIS.md` 12).

7. **Eighteen results here are recorded as retracted, and four of the retractions were
   themselves wrong.** A real two-variant doublet was dismissed as mosaic because the
   diagnostic looked at perpendicular q instead of ω; a real relrod was dismissed on the
   ω-sum. An entire early analysis was retracted for using the **wrong phase**, which
   propagated into the Burgers vector, the selection rules, the fault plane and the strain —
   each independently plausible. The newest is **§3.3**, reported refuted on 2026-09-07 and
   **not** refuted — two disjoint inputs agree to 0.58° and nine of twelve candidate ⟨111⟩
   carry exactly zero voxels; the bad verdict is still sitting in that day's meeting brief
   (`LAB_NOTEBOOK.md` R21). A refutation is a claim and needs the same gate as the positive
   it kills. Read `LAB_NOTEBOOK.md` before re-investigating anything.

8. **Which orientation convention a voxel cloud is in is a property of the CLOUD, not a
   rule.** Two products from this one experiment need **opposite** conventions: the ladder
   fixture needs `OM.T` (bright voxels 0.21 Å⁻¹ from the axis against 4.47 Å⁻¹), the
   all-labels cloud needs `OM` as given (on-lattice 0.978 against 0.212). A docstring that
   stated the transpose universally was wrong for one of them. Run
   `bragg_diffuse.check_orientation_convention` and let the data referee it — but note it is
   **blind on a satellite-only cloud**, where both conventions score near zero and it
   correctly returns `decisive=False`; use the axis test there.

9. **A connected-component label sum is not a budget.** 3-D connectivity merges a reflection,
   its asterism and any streak leaving it into one label, which is then named after whichever
   feature the analyst noticed. On the reference sample the 45 "fault relrod" components
   carried 18.7 % of the intensity and **all 45** had their brightest voxel within 0.02 Å⁻¹ of
   an allowed shell (`LAB_NOTEBOOK.md` R12). Distinct from point 1: not a bad classifier, a
   segmentation unit that is not a physical object.

10. **A null that can fail is necessary but not sufficient — you also need a positive control
    proving the test can see the thing PRESENT.** A directional test returned 29.1° against
    28.95° isotropic and read as a clean absence; a **planted** rod through the same machinery
    scored 29.02° against 28.34° for a planted isotropic blob, i.e. the test had no power at
    all (`LAB_NOTEBOOK.md` R14). Plant the feature, run it through unchanged, and only then
    believe an absence. Point 11 is the same requirement for a *comparison*.

11. **When the groups you are comparing ARE orientations, reproduction proves nothing.** Six
    orientation groups differed in fitted `c`, and the difference held across a re-run with
    different gating. Planting **one identical cell** on every group's real `U` and real hkl
    list, with only the measured origin offset, reproduced the ordering at Kendall 0.73 with a
    span of 0.21–0.27 % against the real 0.41–0.53 % — and was *more* stable between the two
    runs than the real data. Reproduction separates orientation-locked from noise; it cannot
    separate orientation-locked from real. A split-half cannot either, because both halves hold
    the same orientations. **Plant the identical cell first, then compare** (`ENVELOPE.md` §13,
    `DIAGNOSIS.md` 18). Related: a re-analysis sharing 95 % of its input is not a replication
    (§14) — quantify the overlap, test the disjoint part alone, and match groups across runs by
    identity, never by rank.

    **A second, sharper form: the acceptance gate can manufacture the effect.** A gate of the
    form `|x_new/x_seed − 1| < tol` is a truncation band around the *regressor* of the obvious
    test, and produces a positive correlation from data with none. Measured: a 1 % seed-referenced
    gate gave **+0.128 ± 0.042** under a null with zero effect by construction; the same gate
    referenced to a fixed nominal value gave **+0.002 ± 0.053**. Reference every plausibility gate
    to a **fixed nominal**, never to a neighbour — and never change a gate and an algorithm in the
    same edit, or neither can be credited (`DIAGNOSIS.md` 19).

12. **Score two populations the same way before comparing them.** One population measured
    far-only against another measured whole-shell produced a 58× "ranking" that inverted to a
    tie under matched accounting — and that error was the *correction* offered for point 9
    (`LAB_NOTEBOOK.md` R13).

13. **Read the intensity floor off the data, never off the default.** `ENVELOPE.md` §3
    documents 200 counts; the reference cloud bottoms out at **30**, with 71.5 % of voxels
    below 200, and a write-up that assumed the default misreported a fraction that moves 2.5×
    between the two.

14. **A label sum at a fixed absolute threshold is the amplitude raised to a power between 1
    and 5.** For two identically-shaped features whose amplitudes differ by k,
    `V_A/V_B = k^|s|`, and `|s|` is a property of where the threshold lands on that feature's
    intensity distribution, not of the feature — measured here at **0.8 to 5.4** across
    components. It made **Friedel pairs, which must be equal, come out 0.32–1.82 apart** while
    their intensity *per voxel* held at median 0.965. A segmentation-free re-integration moved
    the median volume ratio to **0.991** and cut the log-spread 5.6×. So: never compare
    integrated intensities *or centroids* of components whose peak amplitudes differ by more
    than ~2×; use one fixed support for all members, censored identically **including the
    mirror of any dead region**, on the raw array, threshold swept to convergence — and
    establish the floor on controls known to be equal and matched in **profile width**, since
    the floor is size-dependent (30–40 % for a few-hundred-voxel feature, 2 % above 20,000).
    This is distinct from point 9: it bites even when the label holds exactly the right
    feature. It is also why an intensity-versus-order exponent is extraction-dependent —
    the rungs differ in amplitude along the very axis being fitted, and on this dataset the
    same cloud gave **1.95 and 0.10**, the displacement answer and the composition answer
    (`ENVELOPE.md` §16, §17; `DIAGNOSIS.md` 20, 21).

## The half most people do not know is here

Beyond the diffuse field itself, the package carries a full **diffraction → plasticity** layer
(`phase-6-mechanics.md`): asterism decomposed into lattice rotation versus strain, modified
Williamson–Hall and Warren fault probability, GND and the Nye tensor, anisotropic per-grain
stress and its invariants, Schmid factors and stratification, Σ3 variant assignment, stored
energy and Mecking–Kocks hardening, Hall–Petch and spatial gradients, and writers for DAMASK,
FePX and PRISMS.

Four things govern all of it. **It inherits the phase** — ρ goes as `1/b²`, and on the
reference sample a wrong phase gave b = 4.29 Å where FCC gives 2.571 Å, moving ρ by over an
order of magnitude and taking the selection rules, the fault plane and the strain with it.
**An anisotropy projection is not a mechanical asymmetry**: a stress difference between
two differently-oriented populations appears even when their strains are identical to 1–3 %.

And two that killed a per-variant dislocation-density result outright (`ENVELOPE.md` §8):
**the Nye tensor consumes the refined grain Z**, which here is noise (std ~210 µm about zero,
every layer) — a gradient one of whose directions is noise is not a gradient; and **per-grain
`q_U` is ill-conditioned**, fitted by best-R² over a near-flat surface at R² ≈ 0.02, so only
the ρ *ratio* survives and only if the labels are anchored. On a Σ3 pair they usually are not:
parent and twin are symmetry-equivalent and cannot be told apart from orientation alone.

## Starting from raw frames

Until 2026-09-01 the package had no front end and was handed a voxel cloud by an out-of-tree
script. It now has one: `phase-1-ingest.md` takes frames → mask → polar background → 3-D spots
in (ω, row, col) → cloud, with every selecting step reporting what it discarded. Three things
from it worth knowing before you start:

* **Segment in 3-D.** A reflection sweeps several ω frames; per-frame labelling counts it once
  per frame and cannot tell a genuine second reflection from the same one a frame later.
* **Choose the background by a control that can fail.** Diffraction is positive-only, so every
  coherent *negative* structure the subtraction leaves is an artifact of the model. That is
  what picks the azimuth-sector count, rather than a default.
* **The background cannot follow a ring narrower than its own 2θ smoothing window**, and such
  a ring survives into the spot list as a train of false reflections. Check the window against
  the ring widths actually present.

**Copy the call sequence; do not reconstruct it.** `phase-1-ingest.md` ends with a
**calling contract** — the exact chain, with every signature read from the source — and a trap table.
Nothing in the chain fails loudly on a wrong argument order. The single most expensive one:
`detect_powder_rings` **without `azimuth_deg`** silently skips the occupancy test and discards
about half the real reflections as powder (measured: 105–152 spurious rings instead of 17–50,
kept spots 1133 → 1776). The same section covers calibration, where `make_seed(use_diplib=True)`
segfaults — the process dies at exit 0 with no traceback, so do not opt in — and the naive
pyFAI `.poni` → beam-centre conversion is wrong.

## When something looks wrong

Go to **`manuals/defect/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever, keyed
by symptom rather than by step. Its first entry is the most expensive symptom in the technique:
**a phase you expect reported "not detected" in a textured sample**, because a false negative
closes an investigation and nobody re-opens it.

## Read the envelope before promising an answer

**`manuals/defect/ENVELOPE.md`** separates what the *measurement* can determine from what these
*recipes* apply to. The limit most likely to be asked past: when a defect phase sits on a
reciprocal direction **shared** by two orientation families, parent bulk, twin bulk and a
boundary film all scatter to the same place, and no parameter separates them. That earns a
plain statement and **no suggestion** — the escalation is a different measurement.

**One pattern, wearing three sets of clothes.** That limit (§4), the unobtainable rod fraction
(§1a), and the withdrawn per-variant dislocation densities (§8) look like three separate walls
and are plausibly one: *far-field cannot attribute a diffuse feature to one member of an
over-fragmented mosaic.* The reference sample's indexer splits two Σ3 families into ~230
"grains" per layer, and 232 orientations × 282 allowed hkl tile reciprocal space at 0.0688 Å⁻¹
— finer than any diffuse feature you would want to attribute. Each time this was met it cost
a campaign to rediscover. **At phase 0, before choosing a deliverable, compare the predicted
node spacing against the width of the feature.** It is one line, and it tells you which of the
goals on the intake list are reachable on your sample.

**"Not obtainable" is not "not there."** When the answer is the former, say so in those words:
the rods on the reference sample are directly visible in the raw frames and collinear with the
polytype axis to 1–4°, and none of that is in doubt. What cannot be produced is a number.

## Scope

**The diffuse field of a discrete-spot rotation series**, through the `midas_defect` chain.
Continuous powder rings are `xrd-ct`. Obtaining the grains is `ff-hedm` / `pf-hedm` /
`nf-hedm`.

**Status of the capabilities.** The real-data evidence base is **one material family** — a
deformed Cu-9at%Al single crystal, FCC SG 225, FF-HEDM over 10 layers. Of **946** tests
collected (measured 2026-09-06), **925** pass without `MIDAS_DEFECT_REAL_DATA=1` and exactly
**7** more with it; **11** live in the three real-data files.
All of them are the one sample. The ingest front end has a second, independent anchor not in
the suite (byte-identical mask and 615/615 sub-peaks against a hand-built chain on a
diamond-anvil-cell dataset), and the row/pair indexer has a third (a 621-position
La₃Ni₂O₇ diamond-anvil raster, tetragonal I4/mmm — a different crystal system, a different
instrument and a different failure mode from the Cu-Al anchor). Everything else is synthetic
with fixed seeds.

**A caution the count does not carry.** Two real bugs reached a production analysis of that
raster while the suite was fully green, because both were in the *composition* of correct
primitives by an out-of-tree script, not in any primitive. Neither a passing suite nor a
reproduction caught them; adversarial verification did. `rows.refine_to_convergence` now owns
that composition with its postcondition asserted, but read the pattern in `LAB_NOTEBOOK.md`
R15 before trusting a green suite over a control.

| capability | status |
|---|---|
| rod **detection** and q-space direction | real-data-proven with nulls |
| satellite ladders, polytype decontamination | real-data-proven with nulls |
| relative dislocation density, ratios measured the same way | supported |
| rod / asterism **intensity fraction** | **not obtainable** on this sample — §1a |
| absolute dislocation density, absolute volume fraction | not supported |
| **per-variant** anything on a shared reciprocal direction | refused by `attribution.py` |
| **weak / minority** domain indexing (row + pair seeding, null-gated) | real-data-anchored on a second system — `phase-2` |
| a **between-group** difference in a fitted quantity, validated by reproduction | **not obtainable** — needs a planted identical-cell control, §13 |

Treat that as calibration for your priors.

## Sibling doc sets

`manuals/solve-cell/` (**where the CELL comes from when it is unknown or disputed** —
ab initio indexing, symmetry, distortion mode, phase ID, skill `solve-cell`),
`manuals/ff-hedm/` (far-field HEDM — **where the grains come from** at a KNOWN cell,
skill `ff-hedm`),
`manuals/pf-hedm/` (scanning 3DXRD — **the escalation when bulk and boundary share a
reciprocal direction**, skill `pf-hedm`), `manuals/nf-hedm/` (near-field, skill `nf-hedm`),
`manuals/dct-tt/` (DCT and topotomography, skill `dct-tt`), `manuals/dfxm/` (dark-field X-ray
microscopy — the other escalation for intragranular detail, skill `dfxm`), `manuals/xrd-ct/`
(**the right doc set if your rings are continuous**, skill `xrd-ct`),
`manuals/calibrate-integrate/` (the geometry this consumes, skill `calibrate-integrate`),
and `manuals/tomo/` (the coordinate-system reference, skill `tomo`).

## Log a halt

Technique skills carry no verdicts of their own, so there is one thing worth logging: when the
doc set **stops** you. A halt is a designed outcome, and how often the gates fire on real data
is the only evidence that they are load-bearing rather than decorative.

```bash
~/.claude/bin/skill-log --skill defect --event invoked --verdict INVOKED \
  --subject "<which gate halted the work, or 'ran to completion'>" \
  --evidence <the file or reading that triggered it> \
  --note "<what would unblock it>"
```

**Log a VOID run too, and distinguish it from a halt.** A halt is the doc set refusing before
you spend compute. A void is a measurement that ran, returned a plausible number, and turned
out to have no power — the positive control failed. Those look identical in a results file and
are opposite in meaning, and the void is the one that reaches a manuscript. Use
`--verdict INVOKED --subject "VOID: <what had no power>"`.
