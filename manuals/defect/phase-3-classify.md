# Phase 3 — Bragg / diffuse classification, and what "closure" means

This phase defines the **denominators** every later fraction is quoted against. Get it wrong
and every downstream percentage is wrong in a way that looks fine.

## The split

```python
from midas_defect.bragg_diffuse import (enumerate_hkls, predicted_reflection_points,
                                        classify_voxels, on_lattice_fraction)
```

Given the voxel cloud and the orientations, classify **every** above-threshold voxel by its
distance to the predicted reciprocal lattice `⋃_g (U_g · G_hkl)`. Within `tol` of a predicted
reflection is **Bragg**; the rest is **diffuse**.

Phase-agnostic: the reflection list comes from a `midas_hkls.Crystal` via
`lattice.bragg_shells`, so the systematic absences are never retyped.

**This is a structural test, and that is the point.** The distance is a full **3-D vector**
distance to every allowed reflection of every grain — not a scalar feature, not a radial
coincidence.

## Closure is arithmetic; attribution is the claim

```python
from midas_defect.intensity_budget import intensity_budget
```

The budget decomposes scattered intensity into Bragg / near-Bragg asterism / inter-Bragg
diffuse / low-q halo.

**Closing it does not license a per-feature fraction.** Splitting fault-rod intensity from
deformation asterism needs every diffuse voxel attributed to the reflection it came from, and
that is **impossible whenever the predicted node cloud is finer than the diffuse feature**. On
the reference sample 232 orientations × 282 hkl give a median node spacing of 0.0688 Å⁻¹
against a 0.05–0.15 Å⁻¹ halo, so neither distance nor direction can attribute a voxel, and no
isolated-node subset rescues it (>0.20 Å⁻¹ isolation retains 0.07 % of the halo). Read
`ENVELOPE.md` §1a **before** quoting any "N % fault rods" figure.

Say **"not obtainable"**, never "the rods are absent" — different claims, and only the first
is supported. What survives is structural: rods seen directly in the raw frames and their
collinearity with the polytype axis.

**But closure only means every voxel got a label.** On the reference sample a scalar
classifier reported **99.8 % closure** while being **~18 % wrong** against ground truth. At
high `|q|` the 18°-mosaic node cloud of 232 orientations makes `5G/3`, `220`, asterism and
fault-rod tails overlap in **every scalar feature** — no decision tree over those features can
separate them. `LAB_NOTEBOOK.md` R1, `ENVELOPE.md` §1.

So:

* Use the **structural** classifier, not a scalar one.
* Quote per-class fractions **with the separation method**, or quote neither.
* If the classes genuinely coincide at your `|q|`, report the budget as **closed and
  unattributed**. That is a result, and it is honest.

## A label sum is not a budget

3-D connectivity merges a reflection, its asterism, and any streak leaving it into **one
label** — which then gets named after whichever feature the analyst noticed. On the reference
sample the 45 components called "fault relrod" carried 18.7 % of the intensity and **all 45**
had their brightest voxel within 0.02 Å⁻¹ of an allowed shell: they were reflections with
their wings attached, not rods. `LAB_NOTEBOOK.md` R12.

This is **not** the classifier problem above. It is a segmentation unit that is not a physical
object, and summing over it produces a confident number for a thing that was never isolated.

## Score two populations the same way before comparing them

An accounting difference between two populations is indistinguishable from a physical one. One
population measured far-only against another measured whole-shell produced a **58× "ranking"
that inverted to a tie** under matched accounting — and that error arrived as the *correction*
for the label-sum problem above. `LAB_NOTEBOOK.md` R13.

## Radial coincidence is worthless in a multi-grain sample

Before calling any feature a fundamental, compute the 3-D vector distance to the nearest
allowed reflection of **every** grain. On the reference sample `5G/3` coincides *radially*
with `220`; in 3-D the nearest `220` of any of 232 grains is **0.27 Å⁻¹** away.
`DIAGNOSIS.md` entry 8.

The corroborating half is the forbidden shells: `G/3`, `2G/3`, `4G/3` were **empty across all
grains**. A decontamination that only shows where intensity *is* has done half the work.

## Selection-rule tests

```python
from midas_defect.defect_tests import forbidden_reflection_test
from midas_defect.debye_waller.per_grain_wilson import per_grain_
```

`forbidden_reflection_test` measures intensity at the phase's **systematically absent**
positions against an off-lattice random control at matched `|q|`. A genuine excess means an
anti-phase boundary or another selection-rule-breaking defect; on the reference sample it came
back at the control level on 0 of 248 grains — a clean negative.

**The forbidden positions come from the crystal's own space group.** Supplying them by hand,
or from the wrong phase, is how a 0.46 "smoking gun" turns out to be ordinary Bragg intensity.
`DIAGNOSIS.md` 13.

`per_grain_wilson` gives the per-grain Debye–Waller / Wilson plot, which sets the intensity
scale the budget fractions are quoted against.

## Independence

```python
from midas_defect.honesty import systematic_uq, assert_independent, Probe
```

Two quantities that share their load-bearing inputs are **not** independent probes, however
different they look. A q-space FWHM and its r-space transform keyed to the same axis and the
same voxel attribution agree by mathematical identity, not corroboration. `assert_independent`
refuses to let them be treated as independent.

And UQ must perturb the **systematics** — relabel, re-choose the axis, reseed — not just
resample noise. Resampling-only UQ reports an artifact's *consistency* as *precision*; it once
returned "P = 1.00 / 11.2 σ" for a geometry artifact.
