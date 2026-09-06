# Diagnosis — symptom → discriminating test → cause → lever

Keyed by **symptom**, not by pipeline step, because that is how trouble arrives. Every entry
carries a test that can come back the other way.

---

## Local symptoms

Emitted by **this technique's own procedure**, not by `beamreport`'s generic diagnostics,
which key off per-observation residuals against declared coordinates. A random-direction null
in a textured sample, a scalar classifier at high `|q|`, a coherence length that is really a
mosaic width, an ω-split mistaken for mosaic — these are real and nothing generic will detect
them, so they are declared here rather than renamed into the wrong shape.

Every row names where the check lives. A symptom nothing produces is dead text that reads as
coverage.

| symptom | emitted by |
|---|---|
| `null.random_direction_in_textured_sample` | fraction of the "random" baseline that lands within the search cone of a real ⟨hkl⟩ of any indexed grain |
| `budget.closed_but_misattributed` | per-class fractions recomputed by 3-D vector distance to every allowed reflection, against the scalar classifier's labels |
| `grains.over_fragmented` | pairwise misorientation distribution of the indexed "grains" against the number of distinct orientation families |
| `width.mosaic_limited` | diffuse FWHM against the Bragg FWHM **on the same rod**; `rod_profile.transverse_width` returns a lower bound rather than a value |
| `control.cannot_fail` | scatter of the control; `rod_profile.rod_significance` raises when it is identically zero |
| `significance.ratio_on_zero_centred` | sign and magnitude of the control median relative to its own scatter |
| `feature.omega_split_not_q_split` | separation of a candidate doublet in ω against its separation in perpendicular/azimuthal q |
| `satellite.is_relrod_not_reflection` | ω width of the feature (compact spot vs continuous), and whether a Friedel mate appears at ω+180 |
| `coincidence.radial_only` | full 3-D vector distance from the feature to the nearest allowed reflection of **every** grain, not the radial distance |
| `selection.silently_discarded` | count returned by every selecting step; `ingest.find_blobs_3d(return_counts=True)` and `completeness` counts |
| `prediction.masked_not_absent` | fraction of the predicted position's neighbourhood that is masked, before any search |
| `residual.angular_not_radial` | `residual_decomposition.decompose_residuals` — angular (rad) over relative radial |
| `feature.absent_in_omega_sum` | the same feature sought in raw **per-frame** data, per grain-cluster, rather than in the ω-sum |
| `forbidden.wrong_spacegroup_rule` | whether the "forbidden" positions being tested are forbidden **in the phase actually present** |
| `residual.shrinks_with_wing_cutoff` | off-ring residual as a function of the per-spot asterism cutoff |
| `dpdf.termination_ripple` | radial profile along the candidate direction against a control direction, on the same grid |
| `null.beaten_by_a_decoy` | a deliberately wrong model run through the identical machinery |
| `null.censored_by_a_floor` | the null re-run with the estimator's minimum-match floor lowered |

---

## 1. A phase you expect is "not detected", in a textured sample

symptom: null.random_direction_in_textured_sample

**The most expensive symptom here**, because a false negative closes an investigation and
nobody re-opens it. It happened on the reference sample: a real 9R polytype was reported
absent.

**Test.** Take the "random" baseline your enhancement statistic divides by, and ask what
fraction of those random directions land within the search cone of a real ⟨hkl⟩ of *any*
indexed grain. In a textured sample that fraction is large, so the signal is in the
denominator. Separately, ask whether the cone half-angle is larger than the feature: a 15°
cone around a <5° satellite dilutes it into invisibility.

**Cause when it comes back positive.** The null is contaminated and the statistic is
suppressed. On the reference sample the "random" baseline was polluted by 992 textured grain
⟨111⟩ directions and the 15° cone diluted the discrete satellites — together, a false
negative on a phase that is unambiguously present.

**Lever.** Do not use along-axis-versus-random enhancement in a textured sample. Use raw
voxels and **nearest-axis attribution** with a cluster-label cross-tabulation that makes no
shared/unique assumption. Redone that way the same data gave 55.8 % of satellite intensity
within 5° of a shared ⟨111⟩ against a 1.9 % null — ~30×.

> **Already guarded in the package.** `polytype.satellite_intensity.polytype_satellite_enhancement`
> carries this flaw and **raises by default**; `allow_deprecated=True` reaches it only for
> reproducing history. Use `polytype.satellite_excess.satellite_radial_excess`, which is
> count-normalised at the same `|q|` so texture and sampling density cancel, and which
> returns a **discriminator**: a real periodic polytype peaks at the thirds and dips at the
> half-integers; a continuous relrod rises monotonically inward. `LAB_NOTEBOOK.md` D1.

---

## 2. The intensity budget closes to ~100 % and the classes look clean

symptom: budget.closed_but_misattributed

Closure is arithmetic — every voxel got a label. Attribution is the claim, and it is a
different question.

**Test.** Recompute the per-class fractions structurally: forward-model every allowed
reflection of every indexed grain and classify each voxel by full **3-D vector distance**,
then compare against the scalar classifier's labels. Report the disagreement rate.

**Expected on real data.** On the reference sample a scalar classifier reported **99.8 %
closure** and disagreed with the structural ground truth on **~18 %** of intensity. At high
`|q|` the 18°-mosaic node cloud of 232 orientations makes `5G/3`, `220`, asterism and
fault-rod tails overlap in *every* scalar feature.

**Lever.** Use `bragg_diffuse.classify_voxels`, which is structural. Quote the per-class
fractions together with the separation method, or quote neither. If the classes genuinely
coincide at your `|q|`, say the budget is closed and **unattributed** — that is a result.

---

## 3. Hundreds of grains, and the per-grain statistics look beautifully populated

symptom: grains.over_fragmented

**Test.** Compute the pairwise misorientation distribution of the indexed "grains". A genuine
polycrystal gives something near Mackenzie; a fragmented mosaic gives a huge excess at small
angles. Then count distinct orientation *families* by clustering, and compare.

**Expected on real data.** The reference sample gives ~230 "grains" per layer that are really
**two** Σ3-related families, each a ~18° mosaic — a **~100× over-fragmentation**.

**Lever.** Do per-family statistics, not per-"grain". Anything averaged over fragments has an
`n` that is the fragmentation of the indexer, not the sample, and its error bars are
meaningless.

---

## 4. A coherence length comes out suspiciously specific

symptom: width.mosaic_limited

**Test.** Compare the diffuse FWHM against the Bragg FWHM **measured on the same rod**, which
shares the beam, optics, mosaic and detector PSF. If diffuse ≤ Bragg, the feature is
resolution-limited.

**Expected on real data.** Reference sample: satellite FWHM 0.075–0.12 Å⁻¹ → L ≈ 5–10 nm,
mosaic-contaminated, and quoted as a **lower bound** throughout.

**Lever.** `rod_profile.transverse_width` already returns `lower_bound_A` with
`coherence_length_A = None` in that case. Report the bound. Converting it back into a value
because a number is wanted is the failure this entry exists to prevent.

---

## 5. The rod is enormously significant, or negatively significant

symptom: significance.ratio_on_zero_centred

**Test.** Look at the control's median and its scatter. Two distinct faults:

*  If the control's median is near zero and you divided by it, the "ratio" is noise over
   noise. It returned **−1550×** once.
*  If the control's scatter is **identically zero**, the control cannot fail. That is what an
   azimuthal-median control does on data whose azimuthal median has already been subtracted —
   zero by construction.

**Lever.** Quote σ above a matched control: `(rod − median(control)) / (1.4826 × MAD)`. Use
`rod_profile.matched_control_path` — the same walk at (h+½, k+½, L), which shares the |q|
range, the curvature, the ω range and the detector regions. `rod_significance` raises on a
zero-scatter control rather than returning a number.

---

## 6. A feature looks like a mosaic smear, not a doublet

symptom: feature.omega_split_not_q_split

**Test.** Measure the separation in **ω** as well as in perpendicular/azimuthal q. These are
different axes and a doublet can be split in one and not the other.

**Expected on real data.** On the reference sample a real doublet — two polytype variants —
was **first retracted** because the diagnostic looked at perpendicular/azimuthal q, saw
mosaic, and concluded smear. The signature was an **ω-split**: same-rung, same-side pairs
separated in ω. The retraction was itself wrong and was reversed.

**Lever.** For any candidate multiplicity, test every axis before concluding. Reporting "one
broad feature" because you looked along the wrong one is not conservative, it is wrong.

---

## 7. A satellite might just be the Ewald sphere cutting a relrod

symptom: satellite.is_relrod_not_reflection

**Test.** A real reflection is **compact in ω** and has a Friedel mate at ω+180. A continuous
relrod sampled by the Ewald sphere is not compact and its apparent position moves with ω.
Measure the ω width and look for the mate.

**Expected on real data.** Reference sample satellites are compact Bragg spots, σ ≈ **0.6°**
in ω, with Friedel mates where expected — real reflections, not Ewald artefacts.

**Lever.** If the feature is not compact in ω, it is diffuse and belongs in the rod machinery
(`phase-4-rods.md`), not the satellite/polytype machinery. The two have different nulls.

---

## 8. A feature sits exactly on an allowed reflection's radius

symptom: coincidence.radial_only

**Test.** Radial coincidence is nearly meaningless in a multi-grain sample — many shells at
similar `|q|`. Compute the full **3-D vector distance** to the nearest allowed reflection of
**every** indexed grain.

**Expected on real data.** On the reference sample `5G/3` coincides radially with `220`. In
3-D the nearest `220` of any of 232 grains is **0.27 Å⁻¹ away**. It is a satellite, not a
fundamental — and the forbidden shells (`G/3`, `2G/3`, `4G/3`) are empty across all grains,
which is the corroborating half.

**Lever.** Decontaminate by forward-modelling all grains' allowed reflections and taking the
minimum 3-D distance. Never conclude from `|q|` alone.

---

## 9. A rate is quoted over a denominator nobody counted

symptom: selection.silently_discarded

**Test.** For every step that selects, ask what it discarded. `ingest.find_blobs_3d(...,
return_counts=True)` returns `labelled / rejected_small / kept / blobs_split / sub_peaks`;
the completeness audit returns all four verdicts.

**Expected.** "2 of 45 split" turned out to mean "2 of 45 **evaluated**" after a
gap-rejection step had silently discarded 43. The same pattern drove a result four separate
times in one campaign.

**Lever.** Quote the denominator that was actually evaluated, and report the discard beside
it.

---

## 10. A predicted reflection is "absent" from the data

symptom: prediction.masked_not_absent

**Test.** Before searching, check what fraction of the predicted position's neighbourhood is
masked. A prediction landing behind a module gap is not evidence of absence any more than one
landing off the detector is.

**Lever.** `completeness.audit_completeness` applies the mask test *before* the search and
returns MASKED as a distinct verdict from ABSENT. Only ABSENT is a statement about the
sample; MISSED is a statement about the analysis.

---

## 11. The fit will not improve and the residual is large

symptom: residual.angular_not_radial

**Test.** `residual_decomposition.decompose_residuals` splits the misfit into a relative
radial part and an angular part, and reports the dimensionless ratio (angular in radians over
relative radial).

**Cause.** Ratio ≫ 1: the cell is right and the **orientation** is not — suspect multiple
grains; precision will not help. Ratio ≪ 1: the directions are right and the **d-spacings**
are not — suspect the cell, Lsd or λ.

**Expected on real data.** 0.24 % radial against 1.5° angular is a ratio of **10.6** — that
sample was multi-grain and had been treated as a single crystal.

**Lever.** Angular-dominated → search for more orientations. Radial-dominated → re-check the
cell and the distance/wavelength scale.


---

## 12. A feature you can see in the raw frames is "not there" in the analysis

symptom: feature.absent_in_omega_sum

**A frame-of-reference error, not a physics error**, and it produced a retraction on the
reference sample that had to be reversed.

**Test.** Look at the **raw per-frame** data at fixed ω, restricted to one grain cluster, and
back-project to reciprocal space. Compare against what the ω-sum shows.

**Cause.** Summing over ω superimposes every grain's rod at every orientation, which smears a
1-D rod into a ring — a "donut". The rod resolves only **per-frame (fixed ω) + per-grain
cluster**. On the reference sample a real ⟨111⟩ stacking-fault relrod was retracted on the
ω-sum evidence and later reinstated: back-projection put its bright pixels on integer hkl at
median 0.11, PCA gave a 1-D-dominant direction **3.7° from ⟨111⟩** (singular values
[49.4, 9.6, 0.6]) threading 111→200→220→311→400, and the off-node bridge pixels sat a median
16.6° from ⟨111⟩ against 28° for random.

**Lever.** Before concluding absence, look at raw frames. `rod_profile.rod_path` reads each
point from **its own frame** for exactly this reason; a max projection piles every frame's
background under every point and is the wrong instrument for this question.

---

## 13. A forbidden-reflection test shows a huge "smoking gun"

symptom: forbidden.wrong_spacegroup_rule

**Test.** Check that the positions being called forbidden are forbidden **in the phase that is
actually present**, and confirm they are not *allowed* reflections of that phase. Then measure
the forbidden/allowed intensity ratio against an off-lattice random control at matched `|q|`.

**Expected when it is right.** On the reference sample, with the correct FCC rule (mixed
parity forbidden), the ratio is **≈ 0.000** — equal to the random off-lattice background, on
**0 of 248 grains**. A clean negative: no anti-phase boundary, no selection-rule-breaking
defect.

**Cause when it is wrong.** An earlier analysis applied an **I4/mcm c-glide rule** — from a
phase the sample does not have — whose "forbidden" positions **include FCC-allowed
reflections**. It was therefore measuring ordinary Bragg intensity and reported
forbidden/allowed = **0.46** as a smoking gun.

**Lever.** Take the rule from `midas_hkls` via the phase's own space group, never by hand,
and settle the phase first (`midas_hkls.phase_id`). A selection-rule test inherits the phase
assignment completely.

---

## 14. There is unexplained intensity left over — is it a hidden phase?

symptom: residual.shrinks_with_wing_cutoff

**Test.** Vary the per-spot asterism cutoff and watch the off-ring residual. A real extra
phase does **not** care how you define a spot's wings; asterism wings do.

**Expected on real data.** Reference sample: the off-ring residual fell monotonically
**1.24 → 1.22 → 0.60 → 0.32 → 0.12 %** as the cutoff widened, and the far-field floor
(distB ≥ 0.7) was **0.20 %** and featureless. It was asterism, not a hidden phase.

**Lever.** Report the residual as a function of the cutoff, not at one setting. The full
attribution on that sample closed as Bragg 64.8 % / asterism 31.5 % / 9R satellites 0.04 % /
inter-Bragg ~0.6 % / smooth background ~1.6 % / unexplained ≤ 0.2 %.

**Name the bin, not the defect.** That ~0.6 % was long written as "⟨111⟩ SF relrod"; the label
is withdrawn (`LAB_NOTEBOOK.md` E7 amendment, R12). It is a distance-bin occupancy. Rod and
asterism are not separable in this sample by distance *or* by direction (`ENVELOPE.md` §1a),
so a bin named after a defect asserts an attribution the measurement cannot make.

---

## 15. The 3-D ΔPDF shows a peak at the expected real-space period

symptom: dpdf.termination_ripple

**Test.** Run the identical radial profile along a **control** direction on the same grid. If
the control oscillates with similar amplitude, you are looking at FFT termination ripple, not
structure.

**Expected on real data.** Reference sample, q_max = 3.4, 128³: the profile oscillated with
similar amplitude along the satellite direction **and** along a ⟨100⟩ control (ratio ~1), and
the central slice was a ripple star. The 6.09 Å "peak" was one ripple maximum.

**Two further reasons not to headline it.** The real-space period is the trivial Fourier
conjugate of the satellites, so it is **not an independent probe** — `honesty.assert_independent`
exists to refuse exactly this pairing. And a publishable real-space figure needs apodization,
Bragg-model subtraction and a finer grid: a presentation nicety, not new evidence.

**Lever.** Keep the q-space satellites as the primary evidence — `satellite_radial_excess`,
with its dip at the half-integers, is immune to FFT artifacts.

---

## 16. A null is cleared, and the result feels too good

symptom: null.beaten_by_a_decoy

**Test.** Run a **decoy** — a cell, an axis or an assignment you know to be wrong — through
the identical machinery. It must fail.

**Cause when the decoy also clears.** The null is measuring the machinery, not the model. On
one campaign a deliberately wrong cell beat the ω-permutation null just as the real one did;
the question had to be settled by a structural test instead (a (00L) ladder: 9 spots,
L = −4..+16, |G| linear in L, collinear to 0.23°).

**Lever.** Report the decoy's score beside the model's. A null without a decoy answers a
weaker question than the one being asked.

---

## 17. A null and a decoy both report exactly zero

symptom: null.censored_by_a_floor

**Test.** Re-run the null with the estimator's minimum-match floor **lowered**.

**Cause.** A refinement that returns `None` below *n* matches makes every null draw and every
decoy score zero, which reads as an overwhelming p-value. It is censoring, not significance.

**Expected.** Uncensored, one such "p = 0.000, decoy 0" became **null median 3, max 5, decoy
4** — p ≈ 0.03–0.05, and the decoy nearly as good as the model.

**Lever.** Never let an estimator's guard clause double as a null result. Lower the floor, or
score the null with a statistic that is defined at every *n*.

## 18. A per-group difference reproduced when you re-ran it, and you want to call it robust

**Symptom.** Groups (grains, domains, variants) differ in some fitted quantity — a lattice
parameter, a width, a density. You re-ran with a changed gate, or on more data, and the pattern
held. It feels established.

**Discriminating test — do this before believing it.** Plant **one identical value** on every
group's real `U` and real reflection list, propagate through the real geometry with the
systematics you have actually measured, and refit with the same fitter. Compare the manufactured
between-group spread with the observed one.

```python
# identical crystals, real orientations, real hkl lists, real geometry
q_fake = [U @ (B_identical @ h) for U, h in zip(group_U, group_hkl)]
```

**Cause when it fails.** Anything locked to orientation — an unmodelled reciprocal-space origin
offset, a residual tilt, an ω convention, a detector distortion — produces a *between-group*
difference from identical crystals, because each group samples the detector differently. On one
dataset this manufactured 0.21–0.27 % of an observed 0.41–0.53 %, at Kendall 0.73 to the real
ordering, and was **more** stable across the two runs than the real data.

**Lever.** None, in the sense of a parameter. If the manufactured spread is comparable to the
observed one, the number is not obtainable — say that (`ENVELOPE.md` §13). Fitting the suspected
systematic free is worth one try, but check the direction: here a free-origin refit made the
observed span *grow*, which excluded the origin offset as the mechanism without rescuing
anything.

**What this is not.** Not a noise problem. A noise-only planted control returned 0.007–0.016 Å
against a real 0.10 Å and read as strong reassurance — noise averages away over 100+ domains per
group, and a systematic does not. Two lenses independently reported that a noise control had
been mistaken for a systematics control.

**Related.** §17 (a null and a decoy both report zero) is the same failure at the level of a
single test; this is it at the level of a comparison.

## 19. A seeded domain's cell comes out looking like its neighbour's

**Symptom.** Domains found by seeding from an already-solved domain's cell have lattice
parameters suspiciously close to that seed, and domains found independently do not.

**Discriminating test.** Regress the non-seed domain's parameter on the seed's, and do it
**within positions that carry both kinds** so the comparison is not confounded by real spatial
variation:

```python
# slope ~0 is healthy; a positive slope means the seed is leaking into the fit
scipy.stats.linregress(seed_c[both], nonseed_c[both])
```

**Cause.** The reflection **set** was selected with the seed's cell and never re-selected with
the domain's own. Real as a correctness defect: selecting reflections with a foreign cell is
wrong however large the consequence turns out to be.

**BUT THE OBVIOUS MEASUREMENT IS CIRCULAR — read this before quoting any slope.** If the
pipeline gates acceptance on `|c/c_seed − 1| < tol`, that gate is a **truncation band around the
regressor** and manufactures a positive c-on-c slope from nothing. Under a y-permutation null
with zero anchoring by construction, a 1 % seed-referenced gate produced **+0.128 ± 0.042**; a
gate referenced to a fixed nominal cell produced **+0.002 ± 0.053**. An observed slope of +0.226
under a seed-referenced gate is therefore *mostly or entirely* the gate. The apparent branch
control (seeded +0.329 vs independently-seeded −0.169) is contaminated the same way: it compares
a **gated** population against an **ungated** one.

**Lever.** Reference every cell-plausibility gate to a **fixed nominal cell**, never to a
neighbour's. Then, and only then, `rows.refine_to_convergence` to re-match on the domain's own
cell and refit on the converged set.

**What the loop itself buys, measured honestly.** Run the 2x2 one change at a time:

**The 2x2, run one change at a time** (233 positions, all four arms re-run; paired bootstrap):

|                     | OLD `\|c/c_seed-1\|<1%` | NEW `\|c/c_nom-1\|<1.5%` |
|---|---|---|
| **no loop** (pre-fix) | +0.2260 | **+0.0836** |
| **loop + refit**      | +0.2073 | +0.1247 |

Loop alone: **−0.019** [−0.098, +0.055] old gate, **+0.041** [−0.045, +0.135] new gate — zero,
and *positive* on the new gate. Gate alone: **−0.142** [−0.236, −0.059], larger than the whole
apparent improvement. The buggy code with only the gate swapped (+0.084) beats the "fixed"
pipeline (+0.125). Gate-corrected anchoring does not fall at all: **+0.095 → +0.123**.

The improvement first reported for the loop was the gate re-referencing, mis-credited. Three
fingerprints confirm it: only `c` moved (gamma, whose bound never changed, moved −0.006 ± 0.049);
24 of 530 domains carry 110 % of the effect; and the weakest stratum, named in the fix as the
smoking gun, went +0.424 → +0.478 — untouched. **Change a gate and an algorithm one at a time.**

**Keep the loop anyway.** It moves cells off the seed, which is the point, and a seed-referenced
gate then rejects them — 43 pair domains lost (532 → 489) on the old gate. 4.9 % of *ungated*
independently-seeded domains sit >1 % from the seed cell, so that bound was clipping a real
population. Loop and nominal gate ship together or not at all.

**Second, quieter form.** Check that the reported cell is the least-squares fit to the
reflections stored alongside it. Assigning the candidate inside a refine→rematch loop reports a
cell fitted to the previous iteration, or to a rejected round: **26 %** of one run's stored `c`
values failed this, median 0.0041 Å, against a 0.035 Å effect under study.
