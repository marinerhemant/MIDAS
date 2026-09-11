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

## A Σ3 twin pair puts double-diffraction spots on the n·G/3 ladder positions

Before reading an on-axis `n·G/3` ladder as a polytype in a Σ3-twinned crystal, account for
double diffraction through the parent and the twin. The sum points g_parent + g_twin that lie on
neither lattice fall at exactly **±1/3, ±2/3, ±4/3 and ±5/3 G111 along the shared ⟨111⟩**, and
nowhere off that axis closer than **|q⊥| = 2.82 Å⁻¹**; a random misorientation gives no on-axis
points (`midas_defect/dev/paper/verify_cc54aa5246ae/own_checks/dd_sumlattice.py`, fcc
a = 3.6356 Å). Those are exactly the 9R ladder positions, so **positions cannot separate the
two**, and neither can a cross-tabulation showing the ladder sits on the shared axis.

* A clean **G/2** control does **not** exclude it: double diffraction produces no half-order
  spots. That argument was made on the reference sample and is wrong.
* A large imbalance between a weak rung's two Ewald crossings, with the neighbouring rung
  balanced at the same ω, is a flag for a non-kinematic mechanism (reported on demk L5 by a
  verification lens; not independently checked).
* What does separate them: the polytype's own **off-axis** reflections. Every parent and twin
  reciprocal-lattice point — and so every double-diffraction sum — sits at a multiple of G111/3 along
  the shared axis (a rotation about ⟨111⟩ preserves q·n), so a reflection at any other fraction
  cannot come from double diffraction. For 9R that is l/9 with l not a multiple of 3, e.g. (0 1̄ 4) at
  4/9 (`midas_defect/dev/paper/verify_cc54aa5246ae/own_checks/dd_cannot_make_ninths.py`). An energy
  or ψ scan is a second test.
* **On the reference sample the test is passed:** the 9R-unique (0 1̄ 4) and (1̄ 0 5) at 4/9 and 5/9
  (|q| = 3.12 and 3.28 Å⁻¹) are present, so the 9R is established, and so are the ⟨111⟩ fault rods
  (`LAB_NOTEBOOK.md` E1, E10). Double diffraction there can only modify on-axis intensities.

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

**Verify the pairing from the q vectors; never assign it from the detector quadrant.** Use
`midas_defect.distributions.classify_q_pair(q_a, q_b)` on sample-frame vectors: equal |q| with
**dot = +1** is a crossing pair, **dot = −1** a Friedel pair, anything else unrelated. **Decide
by the sign of the dot product, not by Δω.** The Δω difference in the table is real but small
at low |q| — δ ≥ 2θ, and 2θ is only 1.56° for the n=1 rung (|q| = 0.99 Å⁻¹, λ = 0.172979 Å) —
and centroid scatter on extended features is comparable: on demk L5 the two crossings of the
satellites ran 712.5–726.8 frames apart on a 1440-frame scan while a genuine Friedel pair sat
at 713.7. Splitting a group of four by which side of BC_y they fall on **looks** like the
crossing split and is not, whenever the four spots are not one quartet.

*Correction, 2026-09-10:* this paragraph first said Δframe = exactly half the scan separates
the two kinds (committed in `d9c4cd15`); the reproduction lens on claim `74c222e0b301` showed it
does not.

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
6.50° at n=2 in **both** layers, against 6.5° / 3.25° predicted by the two hypotheses. **H0 is
refuted. H1 is not established:** the registration was read INCONCLUSIVE by its own
pre-declared branch, because n=4 and n=5 are single-peaked and all the leverage sits on two
rungs, and it has not been through `/verify` (`LAB_NOTEBOOK.md` R18). Valley depths of 0.00 and
0.02–0.03 confirm the two peaks are fully resolved rather than a shoulder. Measure Δω, not the q-separation: two referees measured
the q-separation and disagreed (0.175–0.205 vs 0.128 at n=2) precisely because it is not
independent of Δω.

## A ladder can be right in |q| and still not be a ladder

> **VOID NOTE (2026-09-10):** every angle here came from a flat, untilted detector transform (no tilts, no
> distortion). It fails a known-zero test — a grain's 111 against its own 222 comes out 4.2–5.2° instead of 0
> (the corrected transform gives 0.61°) — and it makes the apparent angle between two fixed directions scale
> with |q|. The four- and two-ladder models were artefacts of it, and so is the "angular spread grows with |q|"
> measurement. The radial positions (n·G/3 to better than 1 %) are unaffected, and the search-cone trap still
> stands as a method warning. With the corrected transform the satellites form Friedel/crossing quartets with a
> doublet at n = 1 and 2 only, 6.1° and 6.5° apart (`RESULTS_L5_three_open_points.md`). Source: the VOID section
> of `midas_defect/dev/paper/CHECKPOINT.md`.

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

**There is no such tension once the corrected transform is used** — see the void note at the top of this
section: the doublet shows up in reciprocal-space directions too, at 6.1° and 6.5°.

## Two ways a rod's L-dependence lies about a fault

Both killed a claim on La3Ni2O7 (a Ruddlesden-Popper phase, n = 2) and both are in the
known-limits ledger. They are properties of layered structures and of empirical
normalisation, not of that sample.

**1. The L-period of a Bragg-normalised rod cannot measure a planar-fault displacement.**
For a fault displacement `Rz`, the fault factor `1 - cos(2 pi L Rz / c)` and the parent
`|F(00L)|^2` are the SAME function of L whenever the cell holds n equivalent layers per c.
A fault-free cell with 5 perovskite layers has `|F(00L)|^2 = |sum_j exp(2 pi i L j/5)|^2`,
exactly periodic with period 5, and a period scan on it returns "Rz = c/5 = one layer" from a
crystal with **zero** faults.

**2. An under-sampled envelope normaliser injects its own period.** The obvious fix — divide the
rod by a Bragg "envelope" interpolated through the peaks — does not remove it. For these
structures the peaks sit at even L, so knots at spacing 2 give **< 4 samples per period**: the
interpolation is below Nyquist and preserves ~74 % of the oscillation. Measured on that sample:
the envelope was itself periodic at T = 5.40 with 86-92 % depth, and knots at spacing 2 gave 2.7
samples per cycle with **779 % peak-to-peak interpolation error**. Three consequences, all
measured:

* a synthetic rod with `f(L) = 1` and **no fault at all**, pushed through the same normaliser,
  returned T = 5.15-5.20, i.e. "Rz = 3.70-3.74 A" at p = 0.000 — with a HIGHER R^2 than the real
  data;
* **the answer was a property of the interpolant**: linear 3.70, log-linear 3.67, PCHIP 9.39,
  cubic 2.24 A. A shape-preserving cubic, strictly the better interpolant, deletes the result;
* **the permutation null was vacuous** — a structureless synthetic with no L-structure gave
  p = 0.000, because shuffling only tests "is g a function of L", which any smooth
  q-dependence satisfies. And rod samples are oversampled relative to the transverse window
  (~0.9 px per step in a 13 px box), so N_eff is 10-20x below n; under a circular-shift null that
  preserves the autocorrelation the observed R^2 sat at the null MEDIAN (p = 0.53).

**Before dividing by an empirical envelope:** check the envelope's own spectrum against the
period you are about to claim, run the `f = 1` null through the identical normaliser, and
pre-declare the Friedel/symmetry mate — centrosymmetry forces (h,k,L) and (-h,-k,L) to give
identical periods, so disagreement there kills the result. **If the null returns your answer,
the normaliser is the signal.**

## Which rods carry diffuse intensity — the (h,k) test for an in-plane fault vector

A rod's L-dependence speaks to a fault's displacement along c; WHICH (h, k) rods carry diffuse intensity
speaks to its in-plane displacement R. For planar disorder the diffuse intensity on the (h, k) rod carries

    1 - cos(2 pi (h, k) . R)

so it vanishes where (h, k)·R is an integer and is largest at a half-integer. A Ruddlesden-Popper offset
R = (½, ½) predicts rods on h + k ODD and none on h + k EVEN. A rod walked in q-space from the orientation
(`rod_path`) can be sampled at ANY (h, k), not only on the rows that happened to index.

```python
from midas_defect.rod_profile import (rod_path, matched_control_path, profile_along,
                                      centred_L_nodes, diffuse_to_bragg)
nodes = centred_L_nodes(h, k, L.min(), L.max(), centring="I")     # h + k + L even
rod   = profile_along(stack, mask, rod_path(U, B, h, k, L, ...))
ctl   = profile_along(stack, mask, matched_control_path(U, B, h, k, L, ...))
r     = diffuse_to_bragg(rod, ctl, nodes, exclude=on_ring)       # r["usable"], r["ratio"], r["sigma"]
```

**Normalise by the same rod's Bragg nodes, or the test measures |F|².** Raw diffuse intensity scales with
the parent structure factor, so ranking rods by level recovers "rods where the reflections are bright".
`diffuse_to_bragg` returns the between-node median over the node height; compare THAT across (h, k). Take
the node exclusion from the centring: under I-centring the nodes of even and odd (h + k) rows sit at L of
opposite parity, so a fixed integer grid excludes the wrong points on half the rows.

**A row with no node on the Ewald sphere has no normaliser** and comes back `usable=False`. Report it as
unusable, never as zero: it is noise over noise.

**Measured on 2604, and NOT ESTABLISHED.** Four rows were observable (q-space walk, matched (h+½, k+½)
control, ring crossings removed):

| row | h + k | between nodes | at the gap minima | not observable |
|---|---|---|---|---|
| (0,0,L) | even | +54 sd | +23 sd | 17 % |
| (−1,−1,L) | even | +59 sd | +25 sd | 54 % |
| (1,1,L) | even | +25 sd | +9 sd | 14 % |
| (−2,−1,L) | odd | +4 sd | −0 sd | 34 % |

Rods on three EVEN rows and not on the one ODD row — the opposite of R = (½, ½). But it is ONE odd row,
n = 6, 34 % unobservable and the weakest in the set, so the non-detection may be sensitivity rather than a
selection rule. **Do not quote a parity rule from this**; it needs more odd rows and the per-row
diffuse/Bragg ratio above. (Nickelate project `RUNNING_LOG.md`; the ratio scan, `step23_hk_map.py`.)

## A rod that looks split, or a skirt on one side only — open

Two observations on 2604 are recorded as unresolved, with the three sampling traps that decided what could
be said about them (nickelate project, 2026-08-26; `step51`–`step53`, `step58`).

* **"Fringes on one side of the (0,0,L) rod" — NOT ESTABLISHED.** A first extraction found a skirt on the
  negative-offset side only (median 25.2 against 7.0 counts at 3–10 px, right of centre), and not the
  beamstop arm (nearest mask edge a median 58 px away). Across rods the asymmetry changed sign ((0,0,L)
  L>0 +12.7, (−1,−1,L) L>0 +27.1, (−2,−1,L) L<0 −25.5, (2,2,L) L>0 −29.5), and a per-frame split test over
  all 45 indexed reflections found splitting on BOTH sides — left 20/22 vs right 19/23 (Fisher p = 0.67) at
  one threshold, 21/22 vs 23/23 (p = 0.49) at another. The one-sidedness came from where the unmasked windows
  happened to be.
* **"Double lines in the max projection" — NOT TESTED.** Two resolved maxima in one frame is not two
  persistent tracks after collapsing ω, so the per-frame test does not speak to it. That test also fired on
  44 of 45 reflections, which most likely includes noise, and has no control yet.

The traps, each measured on the same rod:

* **One frame per point keeps a fraction of anything diffuse.** Reading each L from the frame where it
  satisfies Bragg is right for a sharp node and wrong for a feature broad in ω: skirt/core 0.8 % that way,
  4.1 % from the maximum over 36 frames.
* **Neither collapse is unbiased.** The maximum over N noisy frames sits roughly +2σ high; the sum
  accumulates the polar background's small negative per-frame bias (−90.5 counts over 36 frames).
* **A split detector needs a null at positions with no reflection** before its rate means anything. A rate
  of 44/45 is what a detector that fires on noise returns.

## Checking a rod against a simulation, and looking at it

```python
from midas_defect.forward_sim import ...                  # simulate the expected diffuse field
from midas_defect.viz import render_rod_overlay_html      # rods over the data, interactive
from midas_defect.asterism.family_asterism import reflection_directions, family_asterism_arc
```

Related modules: `polytype.ladder`, `polytype.satellite_doublet` (a doublet's signature is an
**ω-split**, not a q-position split — `DIAGNOSIS.md` 6), `polytype.modulation_tilt`,
`polytype.modulation_type`, `polytype.fault_balance`, `delta_pdf` for the r-space view.
