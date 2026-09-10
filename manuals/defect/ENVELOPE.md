# Envelope — what diffuse-defect metrology can and cannot determine

**Owner:** Hemant Sharma. **Last reviewed 2026-09-02.**

A dataset can be squarely in scope and still unable to support what is being asked of it.
**Read this before promising an answer.** §0 is the table to check first.

## Tiers — which limits can move, and which cannot

The tier decides what a report is allowed to say: a *configured* limit may be suggested as a
change; a *fixed* or *intrinsic* one must be called unobtainable rather than tuned at.

| tier | meaning | sections |
|---|---|---|
| **Fixed** | set by the instrument or this beamline cycle | §2 (mosaic sets the coherence-length floor), §5 (ω sampling sets what a "compact" spot can mean) |
| **Configured** | chosen per run, changeable next time — the only tier a report may propose changing | §3 (threshold, background model, split ratio), §6 (ω range and step) |
| **Intrinsic** | the measurement cannot determine it at all; no parameter recovers it | §1/§1a (attribution at high \|q\|; rod fraction), §4 (bulk vs boundary on a shared direction), §7 (absolute volume fraction), §11 (ΔPDF is not an independent probe), **§13 (a per-group difference cannot be validated by reproduction when the groups are orientations)**, **§14 (a re-analysis sharing its input is not a replication)** |

## 0. Deliverable vs what it requires

| Deliverable | Requires | Available? |
|---|---|---|
| **Rods present / absent**, and their q-space direction | a lattice + a matched control | ✅ yes |
| **Rod direction → defect-plane normal** | the orientation matrix | ✅ yes |
| Rod **transverse width** → lateral coherence | Bragg peaks on the same rod as the resolution function | ⚠️ a **lower bound** whenever mosaic-limited — see §2 |
| **Satellite ladder** → polytype identification | forbidden-gap reflections resolved from fundamentals | ✅ yes, with a discrete test — §5 |
| Per-grain **dislocation density** | asterism breadth + a contrast factor | ⚠️ **relative** comparisons only — see §8 |
| **Intensity budget** closed to 100 % | classification of every voxel | ⚠️ arithmetic yes, **attribution no** at high \|q\| — §1 |
| A **fault-rod intensity fraction** (rod separated from asterism) | per-voxel attribution to a reflection | ❌ **not obtainable** when the node cloud is finer than the feature — §1a |
| Defect **volume fraction**, absolute | a gap-free denominator and a resolved boundary | ❌ lower bound only — §7 |
| **Bulk vs boundary** distribution of a phase on a shared direction | something with spatial resolution | ❌ needs pf-HEDM / DFXM — §4 |
| **Absolute strain** from the diffuse field | a demonstration that does not exist here | ❌ not demonstrated |
| **Stacking-fault probability** α | a rigorous Warren analysis from {111}/{200} shifts | ⚠️ the shipped route is a **proxy** — §10 |
| **Total** dislocation content | more than the line profile can see | ⚠️ WH sees a fraction; quote it — §10 |
| A **real-space** 9R/polytype period as independent evidence | something that is not the Fourier conjugate of the satellites | ❌ the ΔPDF is the conjugate, and ripple-dominated — §11 |
| **Mechanical** asymmetry between two orientation populations | separating anisotropy projection from real difference | ⚠️ §12 |

## 1. Attribution at high \|q\| — intrinsic

Closure is arithmetic. Attribution is the claim, and the two come apart.

On the reference sample, at high `|q|` the 18°-mosaic node cloud of 232 indexed orientations
makes `5G/3`, `220`, asterism and fault-rod tails **overlap in every scalar feature** —
radius, breadth, intensity, local density. A scalar decision tree over those features cannot
separate them, and one that was built reported **99.8 % closure while being ~18 % wrong**.

**Consequence.** Budget fractions are trustworthy only where the classes are separated by
something *structural* — a forward-modelled 3-D vector distance to every allowed reflection
of every grain, not a scalar threshold. `bragg_diffuse.classify_voxels` does the structural
version. Quote the per-class fractions with the separation method, or quote neither.

**Not changeable by a parameter.** More care in the classifier does not help; the features
genuinely coincide.

## 1a. A rod fraction needs node attribution, and a fragmented mosaic forbids it — intrinsic

Separating fault-rod intensity from deformation asterism requires attributing each diffuse
voxel to the reflection it came from. **That is impossible whenever the predicted node cloud
is finer than the diffuse feature**, and an over-fragmented mosaic makes it so.

On the reference sample, 232 indexed orientations × 282 allowed hkl = 65,424 predicted nodes
give a median node-to-node spacing of **0.0688 Å⁻¹**, against a near-Bragg halo of
**0.05-0.15 Å⁻¹**. Consequences, both measured:

* **Distance channel.** Every label group is 94-100 % within 0.15 Å⁻¹ of some node
  (whole cloud 97.87 %, `BRAGG_111` 98.04 %, `FAULT_RELROD` 97.14 % — *below* the cloud mean).
  The statistic carries no information; against a same-support null the relrod excess is
  **+1.3 percentage points**.
* **Direction channel.** A **planted** ⟨111⟩ rod at a real 220 node is detected at 11.58°
  against ~29° isotropic **only for the 6.8 % of its voxels that attribute to the correct
  node**; pooled, it scores 29.02° against 28.34° for a planted isotropic blob — i.e.
  invisible. A test that cannot see a planted rod cannot report one absent.
* **No isolated-node subset rescues it.** Restricting to nodes isolated by > 0.20 Å⁻¹ retains
  **0.07 %** of the halo intensity; by > 0.30 Å⁻¹, **none**.

**Say "not obtainable", never "the rods are absent".** The two are different claims and only
the first is supported. The rod evidence that survives is structural, not fractional: rods seen
directly in the raw frames and their collinearity with the polytype axis.

**Not a limit of the technique** — a limit of this sample's orientation count and mosaic. Fewer
or better-resolved orientations would not hit it. Escalation is pf-HEDM or DFXM, as in §4.
Measured 2026-09-01 -> `packages/midas_defect/dev/paper/RESULTS_rod_directionality.md`,
`RESULTS_budget_reconciliation.md`.

## 2. Coherence length — fixed by mosaic

A rod's transverse width is the true width convolved with the instrument **and the crystal
mosaic**. Bragg peaks on the same rod carry the instrument, so
`excess² = width(diffuse)² − width(Bragg)²` removes it. They do **not** carry a mosaic that
differs between the fault lamellae and the parent.

On the reference sample satellite FWHM 0.075–0.12 Å⁻¹ gives L ≈ **5–10 nm**, quoted
throughout as a **lower bound** because it is mosaic-contaminated.
`rod_profile.transverse_width` returns `lower_bound_A` with `coherence_length_A = None`
whenever the feature is resolution-limited. That is the honest output, not a failure; do not
convert it into a value.

## 3. Threshold, background and split ratio — configured

These are the knobs a report may propose changing.

| knob | default | what it decides | how to set it |
|---|---|---|---|
| intensity threshold | 200 counts | what enters the cloud at all | from the negative-structure control, not by eye. **Check the cloud's actual minimum, do not assume the default** — the demk `all_labels_qvox` cloud bottoms out at **30**, and a write-up that quoted 200 misreported a fraction that moves 2.5× between the two |
| background sectors | 8 | whether azimuthal absorption is followed | `ingest.choose_sectors`, by the control |
| `smooth_bins × tth_bin` | 5 × 0.02° | the narrowest ring the model can remove | must be **below** the narrowest ring present, or it survives into the spot list |
| watershed `split_ratio` | 3.0 | how far a maximum must rise above its saddle | scale-free (ratio, not counts); plateau-test it |
| `core_frac` | 0.5 | which voxels set the position | flat over 0.3–0.7, so not tuned |

## 4. Bulk versus boundary on a shared direction — intrinsic

**The one limit most likely to be asked past.**

When a defect phase sits on a reciprocal direction **shared** by two orientation families,
a far-field measurement cannot say which side of the boundary it came from — parent bulk,
twin bulk and a boundary film all scatter to the same place. On the reference sample the 9R
sits on the ⟨111⟩ common to parent and twin (55.8 % of satellite intensity within 5° of
**both**, against a 1.9 % null — ~30× enrichment; 1018 of 1025 Σ3 pairs at 0.0°). Whether it
is a bulk population or a 1–3 nm interfacial film is **not decidable from these data**.

**No suggestion is appropriate for this tier.** The escalation is a different measurement —
pf-HEDM (`manuals/pf-hedm/`) or DFXM (`manuals/dfxm/`) — not a parameter.

## 5. What "compact" can mean — fixed by ω sampling

Distinguishing a genuine reflection from a continuous relrod sampled by the Ewald sphere is a
**discrete** test, and its resolution is the ω step. On the reference sample each satellite is
a compact Bragg spot with σ ≈ **0.6° in ω**, and Friedel mates appear at ω+180 as real
reflections must. At a 1° step that is barely resolved; at a coarser step the test cannot be
made at all.

Related fixed limit: a reflection has **two** Ewald crossings per 360°, verified against
predicted ω to <1°. A pipeline that assumes one silently discards half the data.

## 6. ω range and step — configured

The only tier where a counterfactual has an answer. Use `midas_xaf.rotation_budget` to turn
"we need more ω" into a table — reflections per grain, how many are sensitive to the quantity
of interest, and the resulting σ — rather than an opinion.

## 7. Volume fraction — intrinsic lower bound

A defect volume fraction from a satellite/fundamental intensity ratio is a **lower bound**:
detector gaps remove signal asymmetrically, and the denominator is a fundamental whose own
tails have been assigned elsewhere. The reference sample gives f ≳ **1 %** from gap-free
labels only (311 sat/Bragg = 0.7 %, 511 = 1.7 %). Quoting it as a value overstates it, and no
threshold recovers the missing signal.

## 8. Dislocation density — relative only, here

The package's modified Williamson–Hall uses a cubic `H²` anisotropy correction in `q_U`. The
reference re-analysis used a per-grain radial-breadth fit. The two do **not** agree in
absolute density, and the real-data regression compares **relative** quantities (matrix/twin
ratio, `q_U` ordering) for exactly this reason.

**Consequence.** Ratios and orderings between populations measured the same way are supported.
An absolute dislocation density in m⁻² is not, until the contrast factor and the anisotropy
convention are pinned against an independent method.

**And a PER-VARIANT ratio is not supported here at all** — the 2026-06-23 audit withdrew one.
Three independent reasons, each sufficient:

* **GND via the Nye tensor consumes the refined grain Z**, which on this data is noise
  (std ~210 µm about a mean of ~0, every layer). A spatial gradient one of whose three
  directions is noise is not a gradient. `gnd/nye_tensor.py` is marked per-variant unreliable;
  `gnd/scalar_gnd.py` is safer only if it genuinely avoids Z.
* **Per-grain `q_U` in `line_profile/modified_wh.py` is ill-conditioned** — chosen by best-R²
  over a near-flat surface, R² ≈ **0.02**. Only the ρ *ratio* survives (the Wilkens F-prefactor
  cancels), and only if the FWHMs and the labels are good. Absolute ρ additionally carries
  ~3× from that prefactor (ρ ∝ 1/F², default 0.30, range 0.30–0.50).
* **The parent/twin labels were not anchored.** The two are Σ3-symmetric and cannot be told
  apart from orientation alone; the original assignment was "matrix = the count-majority
  cluster", and re-anchoring to the loading axis **inverted** several contrasts. The sign was
  not secure, let alone the magnitude.

Even at face value the withdrawn contrast was **AUC 0.57** — P(a random twin grain exceeds a
random parent grain), against 0.50 for identical distributions. A tendency, not a separation.
The clean per-spot-local re-analysis gives parent ≈ twin to **1–3 %** (`LAB_NOTEBOOK.md` E8).

**What may still be said about which family is more deformed** — and it is in the manuscript —
rests on three *family-level* measures that agree in all ten layers: per-family mosaic 7–11°,
indexing completeness (twin lower, mean Δ 0.034), and the isolated-200 asterism arc
(twin/parent **1.25×**, range 1.23–1.27). None of them is a dislocation density.

## 9. How far one anchor licenses you

The real-data evidence base is **one material family**: a deformed Cu-9at%Al single crystal,
FCC SG 225, a = 3.6356 Å, FF-HEDM over 10 layers. Of **571** tests collected, **564** pass with
`MIDAS_DEFECT_REAL_DATA=1` (7 skip on MPS float64/complex128 grounds or a missing copland
file). **32** touch real data and all 32 are that sample; **22** of those compare against
stored result tables rather than recomputing from voxels. Everything else is synthetic with
fixed seeds.

The front end (`ingest`) has a second, independent anchor that is **not** in the suite: on a
La₃Ni₂O₇ diamond-anvil-cell dataset it reproduced a hand-built chain byte-identically for the
mask and 615 of 615 sub-peaks to 0.0008 px (recorded in that project's `PORT_TO_MIDAS.md`).

**Treat that as calibration for your priors.** FCC-with-planar-faults is well travelled here.
A hexagonal polytype, a modulated incommensurate phase, or a sample whose mosaic is small
enough that the coherence length stops being a lower bound are all *plausible* and all
*unproven*.


## 10. Dislocation density and fault probability — what the line profile can see

Two bounds in **opposite** directions, routinely confused:

* **ρ is an UPPER bound** when instrumental broadening has not been subtracted. Every width
  the instrument contributes is read as strain.
* **A coherence length is a LOWER bound** while the mosaic is in the width (§2).

Beyond that, the line profile does not see all the dislocation content.
`thermodynamics.taylor_inversion.wh_visible_fraction` compares the WH-visible ρ against the
total ρ that Taylor-inverting the flow stress requires. **Quote that fraction.** A density
presented without it implies a completeness the measurement does not have.

The WH prefactor (`k ≈ 16` in `ρ = kε²/b²`) is convention-dependent, so the defensible
statement on the reference sample was the order of magnitude — `ρ ~ 10¹²–10¹³ m⁻²` — not the
median.

**Fault probability.** `line_profile.warren_beta_proxy_per_grain` is a rod/Bragg intensity
ratio, an order-of-magnitude **proxy**, not a rigorous Warren α from {111}/{200} peak shifts.
Reference sample: α_proxy ≈ 0.005 typical, ≈ 0.022 in the faulted third — sparse faults,
~200 layers between them typically. An earlier α = 0.16 was self-retracted as a
Hendricks–Teller local-minimum artifact.

**And all of it inherits the phase.** ρ scales as `1/b²`, so a wrong phase gives a wrong
Burgers vector and a wrong density by the square: b = 4.29 Å from the wrong phase against the
correct FCC 2.571 Å moved ρ by more than an order of magnitude on the reference sample.

## 11. The 3-D ΔPDF is not an independent probe here — intrinsic

The real-space period is the **Fourier conjugate** of the q-space satellites, so agreement
between them is a mathematical identity, not corroboration.
`honesty.assert_independent` refuses the pairing.

On the reference sample the aggregate 3-D ΔPDF (q_max 3.4, 128³) was additionally **dominated
by FFT termination ripple**: the radial profile oscillated with similar amplitude along the
satellite direction and along a ⟨100⟩ control, and the central slice was a ripple star.

**Consequence.** Keep the q-space satellites as the primary evidence. A real-space figure is a
presentation nicety needing apodization, Bragg subtraction and a finer grid — not new
evidence, and never a second independent probe.

## 12. Anisotropy projection versus mechanical asymmetry — configured/intrinsic boundary

A stress or stored-energy difference between two populations that differ in **orientation**
will appear even when their strains are identical, because the stiffness is anisotropic.

On the reference sample the twin population showed higher stress and stored energy than the
parent, and the asterism rotation and strain widths were **identical parent ≈ twin to 1–3 %**.
The honest reading was an elastic-anisotropy projection of equal strain, and it was not
headlined.

**To claim a real asymmetry** you need a quantity that is not projected through the stiffness
— the asterism widths themselves, a GND density, a fault probability — measured the same way
in both populations.

## 13. A per-group difference cannot be validated by reproduction — intrinsic

When the groups being compared **are orientations** — grains, domains, variants, texture
components — no amount of re-analysis, re-running or split-half testing distinguishes a real
structural difference from a systematic that is locked to orientation. Both halves contain the
same orientations, so both halves carry the same bias.

This was measured, not reasoned. Six orientation groups from a raster differed in fitted `c`,
and the difference was stable across a re-run with different gating. Planting **one identical
cell** on every group's real `U` and real hkl list, propagating through the real geometry with
only the project's own measured unmodelled origin offset, and refitting with the same
`refine_lattice`:

| | span run A | span run B | Spearman | Kendall | max abs Δc |
|---|---|---|---|---|---|
| real data | 0.529 % | 0.414 % | +0.943 | +0.867 | 0.0170 Å |
| **identical crystals + measured offset** | 0.265 % | 0.210 % | +0.829 | +0.733 | **0.0118 Å** |
| identical crystals, noise only | 0.044 % | 0.036 % | −0.143 | −0.200 | 0.0116 Å |

Identical crystals reproduce the ordering at Kendall 0.73, are **more** stable between the two
runs than the real data, and shrink their span 21 % between runs against the real 22 %.
Reproduction separates *orientation-locked* from *noise*. It cannot separate *orientation-locked*
from *real*.

**The consequence for what may be claimed.** A between-group spread smaller than the planted
control's spread is not a result, whatever its p-value. In the case above the surviving "safe"
statement — a two-level split at 0.18 % — sat **inside** the artifact's 0.21–0.27 % range and
had to be withdrawn as well. A Kruskal–Wallis of p = 1.9e-48 across the groups did not help: it
tests whether the groups differ, and the artifact makes them differ.

**What has power.** Plant an identical cell on the real orientations and the real reflection
lists, push it through the real geometry and the real fitter, and compare the manufactured
spread against the observed one. That is the only control in this class that can fail for the
right reason, and it is cheap. Run it **before** the between-group comparison, not after.

**What does not have power, and looks like it does.** A split-half of one run (the halves share
the orientations). A re-run with different gating (see §14). A noise-only planted control —
noise averages away over 100+ domains per group and a systematic does not, so a noise control
returning 0.007–0.016 Å against a real 0.10 Å reads as reassurance and is not.

**Tier: intrinsic** when the grouping variable is orientation. It becomes tractable only if the
suspected systematic can be *fitted and removed* — and note that on real data a free-origin
refit made the observed span **grow** (0.0884 → 0.0969 Å), which excludes the origin offset as
the operating mechanism without rescuing the claim.

## 14. A re-analysis that shares its input is not a replication — intrinsic

Re-running a pipeline with a changed gate, on the same frames, and getting the same answer
measures the size of the overlap and nothing else.

Measured on the same raster: 933 of 1281 domains in the second run were the **same crystal at
the same position** as a domain in the first (< 0.05°), i.e. **95.4 %** of the first run
survived into the second, 766 of them with a byte-identical `c`, over the same live positions.
An overlap-matched null predicted 1.20 discordant pairs in the group ordering; the real
comparison scored 1. **Zero independent information.**

Three rules follow, and each of them caught a real error here:

* **Quantify the overlap in the records before using the word "reproduces."** State it.
* **Test on the disjoint part alone.** The 348 genuinely new domains ordered *differently* —
  4/15 discordant — and the group anchoring the whole story was no longer the extreme one.
* **Match objects across runs by identity, never by rank.** The two runs group independently, so
  "5th largest" in one is not "5th largest" in the other; matched by orientation, two groups
  swapped and the ordering inverted.

A split-half of a **single** run is a strictly better robustness test than a re-run of it — here
disjoint spatial halves reproduced the full six-group ordering in **2.3 %** of splits — but read
§13 first: against an orientation-locked bias, a split-half has no power either.

## 15. A coherence length from node widths carries a factor ~5 and a factor π — intrinsic

Measured on La3Ni2O7 (RP n = 2), (00L) nodes 2–9 px against a re-measured instrumental floor
of 1.71–2.03 px. The honest deliverable was **c-axis coherence 10–55 unit cells, α ≈ 3–14 %
per RP block** — a factor-5 envelope, and **that interval IS the result**, not a fit that
needs tightening. It is the union over four independent re-analyses and spans the
instrumental floor, the width estimator, the lineshape and the node subset. Quoting the
central value (~20 cells) alone is not supported.

Three limits stack, and each is intrinsic to reading a length off a width:

* **The floor is degenerate with the answer 1:1.** Whatever residual error the instrumental
  floor carries transfers directly. Here re-measuring the floor moved the answer by 2×.
  See phase-4-rods §"Turning a node width into a fault probability", trap 1.
* **A factor of π on top.** `1/w` (finite size) and `1/(πw)` (Hendricks-Teller random
  faulting) differ by 3.14 on one measurement, and node widths cannot choose between them.
* **The unit is per RP block, not per unit cell** — the (00L) odd reflections are forbidden,
  so the probed repeat is c/2. Per cell is 2× larger.

**What could NOT be decomposed.** Splitting the width into a coherence term plus a spacing
spread (Δc/c) does not survive: on 7 nodes AICc prefers a ONE-parameter model (flat width, or
floor + linear with no coherence term at all), the 95 % joint region contains both components
being zero, and against pure replicate scatter (duplicate nodes at the same |L| differ by a
factor 2.2) lack-of-fit is p = 0.98. **Report one width with a wide interval and no
decomposition.** Width growth with |L| is suggestive only — Spearman +0.730 (p = 0.007) falls
to +0.554 (p = 0.062) once the rod pedestal is modelled.

**Prerequisite before any of this is meaningful:** confirm the detector was inside its
validated count-rate range (phase-1-ingest trap table). Node widths that inflate with
brightness are a detector effect wearing a defect's clothes.

## 16. A label sum at a fixed absolute threshold is the amplitude raised to a power between 1 and 5 — intrinsic

Measured on demk L5 (Cu-9at%Al, MIDAS zip floor T = 30), 2026-09-08. For a segmentation that
keeps voxels above a fixed absolute value T, two identically-shaped features whose amplitudes
differ by a factor k give

    V_A / V_B = k^|s|        s = d log V / d log T  at the operating threshold

and **|s| is a property of where T happens to sit on that feature's intensity distribution,
not of the feature.** Per component, by peak: 246 → |s| = 0.99, 362 → 0.78, 88 → 2.06,
76 → 2.42, 67 → 2.30, 63 → 3.23, 56 → 3.93, **49 → 5.38**. The weaker the feature the worse
it gets, because T climbs relative to its peak.

**This is not the same failure as `LAB_NOTEBOOK.md` R12** (3-D connectivity merging a
reflection, its asterism and any streak leaving it into one label) **or the intensity-floor
default (§3).** It bites even when the label contains exactly the right feature and nothing
else.

What it did on real data:

* **Friedel pairs — which must be equal — came out 0.32 to 1.82 apart in label sum**, while
  intensity **per voxel** held at median 0.965 and within ±10 % for nearly every pair. The
  variation was entirely in how many voxels the label contained.
* A segmentation-free protocol on the same voxels moved the median volume ratio
  0.880 → **0.991** and collapsed the log-spread **5.6×** (sd 0.237 → 0.042), stable across
  8 support variants. On the integrated ratios, rms|ln| went 0.287 → **0.134**, spread
  2.87 → 1.53, within ±15 % of unity 4/11 → **8/11**.
* Prediction check, so the mechanism is not merely plausible: amplitude ratio 1.136 with
  |s| = 4.6 predicts a label-volume ratio of 1.86 against **1.75** observed; another pair
  gives 1.172^1.24 = 1.22 against **1.191**.
* **The floor is size-dependent.** Size-matched control pairs that must obey Friedel returned
  rms|ln| = **0.420**; strong pairs (> 20,000 vox) returned **0.022**. For a few-hundred-voxel
  feature the method floor is 30–40 %.

**How to apply.** Never compare integrated intensities — or centroids — of components whose
peak amplitudes differ by more than ~2× if they were segmented at a fixed absolute threshold.
Instead: **one fixed support for all members, censored identically (including the mirror of
any detector dead region), integrated on the raw array, with the threshold swept to
convergence.** Establish the method's floor on control features that are *known* to be equal
and **matched in profile width** — not merely in size and |q| — before quoting any difference
as real.

Corollary for fitting: **any power law fitted ACROSS features of different amplitude — e.g.
satellite intensity versus order n — inherits a bias that varies with amplitude along the very
axis being fitted.** See §17, which is that corollary biting.

Related: the rms of a ratio is a SCATTER statistic. Under a true null an estimator with
scatter σ returns rms|ln| ≈ σ, so "rms fell from 0.287 to 0.134" says the second estimator has
less variance and says nothing about whether an asymmetry exists. Test the SIGNED mean against
controls.

## 17. An intensity-versus-order exponent is extraction-dependent, and the spread covers the competing physical answers — configured

Found 2026-09-07 on demk L5 while cross-checking a number already on a slide.
`RESULTS_exponent_extraction_conflict.md`.

The satellite intensity scaling `I ∝ n^p` is the discriminator between a **displacement**
modulation (theory p = 2) and a **composition** modulation (p = 0). Two extractions of the
same four rungs from the same cloud gave **1.95** and **0.10** — one on each answer.

| | n=1 | n=2 | n=4 | n=5 | exponent |
|---|---|---|---|---|---|
| re-extraction, raw I/I₁ | 1.000 | 1.322 | 0.392 | 0.256 | |
| Lorentz factor J/J₁ = sin 2θ | 1.000 | 1.999 | 3.994 | 4.989 | |
| re-extraction, corrected | 1.000 | 2.644 | 1.567 | 1.279 | **0.10** |
| manuscript, raw | 1.00 | 0.45 | 1.92 | 4.42 | |
| manuscript, corrected | 1.00 | 0.90 | 7.68 | 22.04 | **1.95** |

The disagreement is not a tolerance question: the manuscript puts n=2 **below** n=1 (0.45)
where the re-extraction puts it **above** (1.32). Extraction geometry for the re-extraction:
tube `perp < max(0.05, tan(1.6°)·|par|)`, window `|par − nG/3| < 0.06`.

**Do not quote a modulation type from a rung-intensity exponent** until the extraction is
pinned and the exponent is shown stable across the tube half-angle, the radial window and the
integration method. §16 is one reason it moves — the rungs differ in amplitude by more than
2×, so a fixed-threshold extraction biases them differently along the fitted axis.
**Status: OPEN.** Displacement versus composition is not decided on this dataset.

## 18. Comparing the two Ewald crossings of a reflection — what actually bounds it

Established on demk L5 across 2026-09-08/09. The comparison is worth making — it is one of the
few internal checks a single rotation scan supports — but four things bound it, and three of
them changed a verdict here.

**The Friedel floor is structurally BLIND to the errors that survive in a crossing comparison.**
Friedel mates are mirrored in **row** about BC_z (same column, Δω = 180°); the two Ewald
crossings are mirrored in **column** about BC_y. A beam-centre error in y, a detector `ty`
tilt and a rotation-axis wedge are all **column-antisymmetric**, so they cancel in the Friedel
comparison and survive in the crossing one. A crossing offset "N× above the Friedel floor"
therefore means *not truncation and not generic centroid noise*. It does **not** mean *not
instrumental*.

**But a calibration residual has closed-form leverage and is common-mode — which makes it
falsifiable.** Finite-differencing every geometry parameter, only BC_y moves the crossing
offset at all:

    d|dq| / d(BC_y)  =  2 · k · px / Lsd  =  0.0188 1/A per pixel

(the factor 2 because the two crossings sit ~180° apart in ω, so a column shift adds rather
than cancelling). Every tilt contributes ≤ 0.003 1/Å per 0.1°, and p0–p3 ≤ 1e-4. It is
**common-mode to ±6 % across ten reflections spanning |q| = 0.99–7.04.** So a
column-antisymmetric residual predicts **one** |dq| for every reflection. On this dataset the
observed spread was a factor **12.6**, and the tightest reflection (222 at 0.0035) caps the
residual at ≤ 0.19 px, which predicts ≤ 0.0035 everywhere. A bounded 11-parameter fit left
95–112 % of every offset standing. **Use this as a test:** a large reflection-to-reflection
spread in crossing offset *excludes* a calibration residual as a sufficient cause. It does not
tell you what does cause it.

**Compare features of similar size, with a support that ENCLOSES each one.** Satellites and
fundamentals differ by ~10× in width here; a sweep of h ≤ 30 px enclosed all five satellite
groups (which need h = 11–23) and **zero of five** Bragg groups (which need h = 36–100).
Identical parameters on features of different size is a different measurement in each arm, and
here it **inverted the ordering** between the two arms. Size the support per group and say so.

**Do not pool centring conventions on extended features.** Between a label centroid and a
windowed argmax, the Bragg 111 crossing offset moved **2.49×** (0.0327 vs 0.0132) where every
compact satellite moved 1.09–1.23×. A median over both is the answer in neither.

**And note what cannot be done at all:** a superlattice reflection has **no |q|-matched
fundamental**. `|q_111| = 2.99 1/A` is the lowest allowed fcc reflection and the n=1 satellite
sits at `|q_111|/3 ≈ 0.99`, where no fundamental exists by construction. A |q|-matched Bragg
control at n=1 is impossible in principle, which forces the comparison into |dq| against the
whole control spread rather than against one partner — and the choice of invariant (|dq| versus
angle about the origin, which is |dq|/|q|) can itself flip the verdict.
