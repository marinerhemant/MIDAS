# Phase 4 — rods, satellites and polytypes

Two different machineries, told apart by the `phase-0` test: a **satellite** is compact in ω
with a Friedel mate at ω+180; a **rod** is continuous. They have different nulls and must not
be swapped.

## Rods in the cloud

```python
from midas_defect.rod_detect import find_rods, find_rods_iterative_residual
```

Pair-seeded RANSAC over bright cores, scored by counting bright voxels within a tube, then
NMS and a differentiable soft-tube refinement of direction and pivot. Per rod it returns the
integrated intensity, the line-projected profile, and which Bragg shells it crosses. The rod
direction is the **defect-plane normal**.

## Rods on the frame stack

```python
from midas_defect.rod_profile import (rod_path, matched_control_path, profile_along,
                                      rod_significance, ring_L_marks, transverse_width)
```

Use this when the question is *how intensity runs along a known rod*, or *is this rod real*.

**Walk the rod where it is straight.** A rod is a straight line in reciprocal space, but the
map to a flat detector goes through the Ewald sphere and is nonlinear, so on the detector it
is a **curve**. Walking a straight detector line drifts off it, worst at large |L| — which is
exactly where the profile is read. `rod_path` parameterises `(h, k, L)` with L continuous,
solves ω per point, and projects. Each L is then read from **its own frame**, not from a max
projection that piles every frame's background under every point.

**Three ways a point drops out, none of them "no intensity":** no ω solution in the delivered
range, off the detector, or the transverse slice mostly masked. All are counted separately and
returned as **NaN**. A silent gap looks exactly like a real minimum in the quantity being
measured.

**Quote σ, never a ratio.** Per-frame background-subtracted data is centred on **zero**, so a
ratio divides by noise about zero — it returned **−1550×** once. `rod_significance` gives
`(rod − median(control)) / (1.4826 × MAD)`.

**The control must be able to fail.** `matched_control_path` is the same walk at
`(h+½, k+½, L)`: not a lattice rod, but the same |q| range, curvature, ω range and detector
regions. An earlier control — the azimuthal median at the same radius — returned exactly zero
for every row, because a polar-median-subtracted stack *is* an azimuthal median.
`rod_significance` raises on a zero-scatter control rather than returning a number.

**Mark ring crossings, do not delete them.** `ring_L_marks` maps detected powder rings onto L.
A ring puts a bump at one L, imitating exactly the modulation that would encode fault
statistics. Get the radii from the **raw** data, not from a stack the rings were removed from.

**Width is a comparison, not an absolute.** `transverse_width(diffuse_fwhm, bragg_fwhm)` —
the Bragg peaks *on the same rod* carry the resolution function, so
`excess² = diffuse² − bragg²`. If the two are equal the rod is **resolution-limited** and the
function returns a **lower bound** with `coherence_length_A = None`. That is a real result.
Do not convert it back into a value. `ENVELOPE.md` §2.

## A null that can fail is necessary, and not sufficient

**You also need a positive control proving the test can see the feature when it is PRESENT.**
A null tells you the test does not fire on noise. It says nothing about whether the test can
fire at all.

On the reference sample a directional rod test returned **29.1°** against **28.95°** isotropic
and read as a clean absence. Planting a real ⟨111⟩ rod through the identical machinery scored
**29.02°** against **28.34°** for a planted isotropic blob — the test had **no power**, and the
"absence" was meaningless. `LAB_NOTEBOOK.md` R14, `ENVELOPE.md` §1a.

So: plant the feature at a real node, run it through unchanged, and only then believe an
absence. If the planted feature is invisible, the honest report is **"not obtainable"**, not
"absent".

## Satellites and polytypes

```python
from midas_defect.polytype.satellite_excess import satellite_radial_excess
```

**Use this, not `satellite_intensity.polytype_satellite_enhancement`**, which raises by
default and for two opposite reasons — voxel-count inflation (~5× became 700–1600×) and
texture-contaminated nulls (false negatives). `LAB_NOTEBOOK.md` D1.

`satellite_radial_excess` is a **count-normalised** per-voxel mean-intensity ratio between
on-axis and off-axis voxels **at the same |q|**, so detector sampling density and texture
cancel. It carries the discriminator the old metric lacked:

* a real periodic polytype is **peaked at the thirds** (`G/3`, `2G/3`) and **dips to
  background at the half-integers** (`G/6`, `G/2`, `5G/6`);
* a continuous relrod **rises monotonically inward** from the Bragg.

The verdict comes back with the numbers.

**In a textured sample, never use an along-axis-versus-random-direction statistic.** A 15°
cone dilutes a <5° satellite into invisibility and the "random" baseline is contaminated by
the textured axes themselves — both push toward a false negative, and together they produced
one on a phase that is unambiguously present. Use raw voxels and nearest-axis attribution with
a cluster-label cross-tabulation that makes no shared/unique assumption. `DIAGNOSIS.md` 1.

## Finding the axis, and the rest of the polytype API

```python
from midas_defect.polytype.activated_axis import detect_activated_111_axis
from midas_defect.polytype.cell_index import ...          # supercell indexing of the ladder
from midas_defect.polytype.finite_stack import ...        # finite-stack (Hendricks-Teller) model
from midas_defect.polytype.doublet_survey import ...      # doublets across the whole ladder
from midas_defect.polytype.aggregate_thickness import ... # lamella thickness, aggregated
```

`detect_activated_111_axis` finds the modulation axis **from the data** rather than being told
it — the data-refereed route that, cross-checked against the shared-⟨111⟩ construction, gave
99/99 Σ3 pairs agreeing to 0.0° on the reference sample.

A **finite-stack / Hendricks–Teller fit is prone to local minima** — an α = 0.16 on the
reference sample was self-retracted as exactly that. Sweep the starting point and report the
landscape, not the converged value alone.

## Turning a node width into a fault probability — four ways it goes wrong

Every one of these was paid for on La3Ni2O7 (Ruddlesden-Popper n = 2), and three of them
killed a headline number that had already been written down.

**1. Never deconvolve against a floor measured by a DIFFERENT estimator than the signal.**
The instrumental floor was measured with a radial-profile estimator that bins at integer
pixel radius and reports max-minus-min over bin INDICES. That saturates at 1 for any true
width below ~2.2 px, and it duly returned **exactly 1.00 px at all six calibrant radii**.
Re-measured on stacked sub-pixel profiles (25k-67k px per ring) the true floor was
**1.71-2.03 px**. The inferred coherence is degenerate with the floor 1:1, so the first
answer (377 A) was pure rail. **Identical values at every radius are a rail, not precision** --
print the distinct outputs and look at them. Use `midas_defect.polytype._width.fwhm_half_max`,
which interpolates the half-max crossings instead of counting samples above half max; the
counting form is biased low by up to one sample spacing and quantised to it.

**2. State the repeat the measurement actually probes, BEFORE quoting alpha.**
In I4/mmm the (00L) reflections with L odd are FORBIDDEN, so every measured node is even and
the repeat probed is **c/2**, not c. An alpha computed from those nodes is therefore
**per RP block**, and "per unit cell" is 2x larger. This was written down inverted once
("8.0 %/cell = 4.0 %/block"; it is 8.0 %/block = 15.4 %/cell). Name the unit in the same
sentence as the number.

**3. A factor of pi sits on top of whatever envelope you quote.** One width w0 gives
`1/w0` = finite-size coherence and `1/(pi*w0)` = the Hendricks-Teller random-faulting length --
a factor of 3.14 apart, from the same measurement. They are not two results; they are one
measurement under two shape conventions, and node widths alone cannot choose between them.
**Always name the convention in the same sentence as the number.**

**4. The odd-midpoint intensity is an UPPER BOUND, not a second measurement.**
Hendricks-Teller predicts `I(odd midpoint)/I(even node) = alpha^2/4` halfway between even
nodes, and it looks like beautiful independent confirmation of the width result. It is not:
the Bragg TAILS of the two flanking nodes must be subtracted first. Measured on La3Ni2O7 the
tails alone predicted **136 % of the observed midpoint**, and in 2 of 7 gaps the predicted
tail exceeded the observation outright. Fit the node lineshapes, extrapolate them to the
midpoint, subtract, and report a bound.

**Prior-art gate, specific to this family.** The existence of stacking disorder in
La3Ni2O7 is published (Chen et al., JACS 146, 3640 (2024), TEM/LAADF, 1313 vs 2222
polymorphs) and **the odd-L discriminator is published verbatim in that paper**. Run the
prior-art gate before claiming either as a contribution; what is potentially new is seeing it
in a bulk diffraction measurement and mapping it per position, not the discriminator.

## The Friedel/crossing quartet — four spots per reflection, and how to pair them

A single reciprocal-lattice vector **G** gives **four** spots in a 360° scan: **q** and **−q**
each cross the Ewald sphere twice, at ω₀, ω₀+180°, ω₀−δ, ω₀+180°−δ with

    delta = 2*theta / |sin(eta)|

(demk L5 n=5 rung: predicted 8.59° from 2θ = 7.876°, η = −66.5°; observed ≈ 7.5°). On this
instrument the two relationships are geometrically distinct **on the detector**:

| relationship | column | row | Δω |
|---|---|---|---|
| **Friedel mate** (q ↔ −q) | same | mirrored about BC_z | **exactly 180°** |
| **the other Ewald crossing** (same q) | mirrored about BC_y | same | ~180° − δ, NOT 180° |

**Verify the pairing from the q vectors; never assign it from the detector quadrant.** The
test is unambiguous and takes one line: for unit vectors, a Friedel pair has
**dot = −1** and Δframe = exactly half the scan, a crossing pair has **dot = +1** and Δframe
≠ half. Splitting a group of four by which side of BC_y they fall on **looks** like the
crossing split and is not, whenever the four spots are not one quartet.

That failure is not hypothetical. A 220 group of four components on demk L5 was split by
column and averaged — but ids 77 and 130 are **59.3° apart** (dot = +0.51), two different
`<220>` variants of the same grain rather than a Friedel pair, so the group average was taken
over two vectors 59° apart. It reported |q| = 4.246 for a reflection whose every component
sits at **4.89**. Both the offset and the |q| were void. The genuine pairs, verified by dot,
were (77, 84) and (130, 136) at dot = +1.0000.

Before comparing the members of a quartet in intensity or position, read `ENVELOPE.md` §16
(fixed-threshold label sums) and §18 (what bounds a crossing comparison). The short form:
match the support to the feature, censor the mirror of every dead region, do not pool centring
conventions, and remember the Friedel floor is blind to column-antisymmetric errors.

## Telling two orientation variants from a fixed transverse offset

A satellite doublet can be **two orientation variants** (a rotation about the ω axis, so the
pair shares q_lab and lands on the same detector pixel) or **a fixed transverse q-offset**.
The discriminator is how the ω-separation scales with rung order:

* two variants → **Δω constant** with n
* a fixed transverse offset → **Δω ∝ 1/n**

Measured on demk L5 and L7 (`RESULTS_doublet_scaling.md`, preregistered): 6.75° at n=1 and
6.50° at n=2 in **both** layers, against 6.5° / 3.25° predicted by the two hypotheses. **H0
refuted, two variants supported.** Valley depths of 0.00 and 0.02–0.03 confirm the peaks are
fully resolved rather than a shoulder. Measure Δω, not the q-separation: two referees measured
the q-separation and disagreed (0.175–0.205 vs 0.128 at n=2) precisely because it is not
independent of Δω.

## A ladder can be right in |q| and still not be a ladder

demk L5, `RESULTS_L5_ladder_collinearity_negative.md`. Every picked component landed at
`n·G/3` to better than 1 % (0.984–0.998, 1.996–2.009, 3.989–3.995, 4.995–4.999). **The radii
are excellent and the set is still not collinear through the origin.**

Two geometric models were proposed and both died:

* **Four ladders — refuted.** Candidates C and D sit 3.73° and 7.48° from axis A, and a 5°
  search cone around them **re-collects A's and B's own components**. Every member was
  borrowed. **A search cone around a candidate direction will re-collect a neighbour's
  members, so "N members found" is not evidence that the direction exists.** Require members
  that are unique to the axis before counting it.
* **Two ladders — also refuted.** Fitting each axis from the tightest rungs (n=4, n=5), a line
  through the high rungs misses even its own Bragg anchors by 1.9–2.8°. The decisive number:
  the high rungs put A and B **13.89°** apart while their n=1 spots are only **8.5°** apart.
  Two rigid ladders separate their n=1 rungs by the same angle as their n=5 rungs. These do
  not.

What is actually measured is that the **angular spread of the satellite directions grows with
|q|, faster than linearly**: 8.5° at n=1, 12.7° at n=4, 14.8° at n=5, i.e. spread/n of 0.146,
0.219, 0.255. A constant-angle ladder holds spread/n fixed; a fixed transverse offset makes it
fall. Neither describes this, and **no geometric model is currently proposed** — the honest
output is the measurement plus "cause unknown".

**Note the live tension with the section above.** In ω the doublet is a clean, preregistered
pair of variants ~6.5° apart on two layers; in q-space there is no clean pair of directions at
any single angle, and at n=1 there are four discrete directions at 2.03 / 4.43 / 8.16 / 10.51°
from axis A. Both measurements stand. They are not yet reconciled, and a write-up that quotes
one without the other is quoting half the evidence.

## Checking a rod against a simulation, and looking at it

```python
from midas_defect.forward_sim import ...                  # simulate the expected diffuse field
from midas_defect.viz import render_rod_overlay_html      # rods over the data, interactive
from midas_defect.asterism.family_asterism import reflection_directions, family_asterism_arc
```

Related modules: `polytype.ladder`, `polytype.satellite_doublet` (a doublet's signature is an
**ω-split**, not a q-position split — `DIAGNOSIS.md` 6), `polytype.modulation_tilt`,
`polytype.modulation_type`, `polytype.fault_balance`, `delta_pdf` for the r-space view.
