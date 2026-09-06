# Phase 5 — Decide whether to trust the reconstruction

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 15. One command

```bash
midas-grain-qa results/LayerNr_6 --json qa.json --csv qa_pergrain.csv
```

Point it at a finished layer directory. Five stages, one report:

| stage | what it answers | needs |
|---|---|---|
| `d0` | is the reference lattice right? | `Grains.csv` only |
| `twins` | are the Σ3 twins real, or chance CSL pairs? | `Grains.csv` only |
| `attribution` | which contested spots does a grain not own? | `SpotMatrix.csv` |
| `uncertainty` | 1σ on all 12 refined parameters | geometry + `ProcessKey.bin` |
| `calibration` | **are the error bars right?** | an adjacent layer |

`--skip <stage>` drops any of them; `uncertainty` is much the slowest
(~0.05 s/grain on GPU, ~0.5 s on CPU), and `--max-grains-sigma` caps it.
Implementation: `midas_process_grains/grain_qa.py`.

---

## 15a. The rule that makes any of this worth reading

**A check that cannot fail is not a check.** Before believing any number below,
ask what it would have looked like had the thing been broken. Every stage here
carries a control that can come out the wrong way, and on the dataset this doc
set was written from, several did:

| control | what it caught |
|---|---|
| far-pair null on the twin rate | nothing — twins were real (0.00 % chance rate) |
| matched-random spot removal | contested spots real: −38 % held-out, against **+5.6 %** for removing the same *number* at random |
| a control arm that reproduced production byte-for-byte | proved the d0 harness *before* its result was read |
| repeat measurement across layers | the error bars were **2.9× too large** |
| synthetic data through the same pipeline | killed a "σ ∝ 1/√n law" that was true by construction |

The last two are the ones a user is most likely to need, and only the fourth is
automated here (`calibration`). The fifth is the habit worth copying: **push
fabricated data through your analysis and see whether the result survives.** If
it does, the result was a property of the method, not the sample.

---

## 15b. `calibration` — the stage that can tell you your error bars are wrong

A grain straddling the boundary between two layers is measured **twice**, from
independent spot sets. If σ is right then

```
z = (m1 − m2) / (sqrt(2) · sigma)
```

has unit variance. `std(z)` is therefore the factor your error bars are wrong by,
measured, with nothing assumed.

**It reports and never applies.** The output names the implied `sigma_obs_px` and
asks you to re-run with it. That is deliberate: a silent placeholder is what
caused the problem in the first place, and silently correcting it would be the
same mistake pointed the other way.

Read the two gates it prints. It matches grains within a position window and then
measures the scatter of that same difference, which is selection on the dependent
variable — so it reports a tight and a 4× looser gate. If they disagree much, the
gate is shaping the answer. On 1-ID the robust `std(z)` moved only 0.348 → 0.360,
but the *tail* fractions moved a great deal, which is why no tail fractions are
printed.

**What it cannot see.** The difference cancels everything the two measurements
share — detector, `Lsd`, tilts, wedge, energy, the reference lattice. So this
calibrates **reproducibility**, not distance from truth. On 1-ID the hydrostatic
strain carried a **+348 µε** common-mode offset with a per-layer spread of only
2.5 µε: invisible to this check, and 5.7× larger than anything it reports. Carry
calibration error separately, as a global registration term on the whole map,
not as a per-grain inflation.

---

## 15c. `sigma_obs_px` is an assumption, and it is the whole scale

Every σ from the `uncertainty` stage scales **linearly** with `sigma_obs_px`, and
carries no information from your data about it. The default of 1.0 px is a
placeholder, not a typical value — on 1-ID GE5 the measured figure was 0.35 px,
so the default inflated every bar ~2.9×. The stage warns when you leave it.

Two traps:

1. **A frame is not a pixel.** ω noise is in frames. Pass `sigma_obs_frames`
   separately; carrying the pixel value across over-assumed the ω channel ~2.5×
   on its own (1 frame = 0.25° against a measured core of 0.098°).
2. **Do not estimate it as χ²/dof over the refiner's residuals.** MIDAS minimises
   a sum of *absolute* internal angles — an L1/LAD estimator, not least squares
   (`midas_fit_grain/c_src/FitPosOrStrainsOMP.c`). The residual is heavy-tailed
   (15–17 % of spots carried 82–86 % of Σr² on 1-ID), so χ²/dof answers a
   question about a fit that is never performed: it gave 0.84 px against a true
   0.35. Use `calibration`, or the density of the residual core.

---

## 15d. What the σ do and do not mean

They come from a **Fisher** information matrix, and in that construction the
observations are detached: `H = JᵀJ` with `J = ∂pred/∂g`. Replacing every
observed spot with the model prediction, or with prediction plus 4× the real
noise, leaves σ **bit-identical**.

That is correct behaviour for expected information, and it has a consequence
worth stating plainly: **σ describes how well a grain *could* be determined given
its spot geometry and your assumed noise. It does not respond to whether that
grain actually fitted well.** A grain with a terrible residual and 200 spots gets
a small σ. Use `DiffPos`, the attribution stage, and `RMSErrorStrain` to judge fit
quality; use σ for precision. They are different questions.

It is also the **frozen-calibration** variant (Σ_cc = 0) — it disclaims Lsd, beam
centre, tilts, wedge and energy by construction
(`midas_process_grains/compute/position_uncertainty.py`).

---

## 15e. Strain: which convention, and when neither will do

`Grains.csv` carries both. They are computed by different routes and do not agree
closely (Pearson r ≈ 0.64 on ε₁₁ for 1-ID).

- **eFab** comes from the refined lattice parameters. The refiner fits position
  *before* the lattice, so the position degree of freedom absorbs radial residual
  before eFab's inputs see it. A projection cannot increase variance, so **eFab is
  guaranteed smoother** — and it reads about 20 % low in amplitude. Never quote it
  as a strain *magnitude*.
- **eKen** solves the six components from the spots directly, so a misassigned
  spot hits it much harder.

Reproducibility favours eFab by ~1.65× in raw microstrain, but most of that is the
compression: normalised per unit signal the advantage is only **~1.15×**. Do not
repeat 1.65.

**`|eKen − eFab|` is a free per-grain quality flag** — both tensors are already in
`Grains.csv`, so it needs no spot-level analysis. On 1-ID it tracked poorly
determined grains well (`DiffPos` ρ = +0.75, completeness ρ = −0.70). It is *not*
specifically a contested-spot flag: that correlation (+0.52) is largely screened
off by general grain quality.

**And check whether per-grain strain is resolved at all before mapping it.**
Compare the between-grain spread to the within-grain noise from `calibration`. On
1-ID that ratio was ≈ 1 for both conventions, with `RMSErrorStrain` median 863 µε
against a 267–385 µε signal — **per-grain strain was not resolved**, and no choice
of convention would have fixed it. Population and layer averages were still fine.

---

## 15g. Validating against an external reference (EBSD, NF, a second modality)

Everything above measures **reproducibility** — repeat measurements that share the
detector, `Lsd`, the tilts, the wedge, the energy and the reference lattice, so every
systematic they share cancels. An external modality is the only thing that measures
**distance from truth**. If you have one, this section is the most informative check in
the doc set. Implementation used below:
`~/Desktop/analysis/shirley_paper/ff_ebsd_compare/ff_ebsd_match.py`.

### The design rule

**Never fit the FF→reference registration and then validate with the same data.**
Three ways to enforce it, in order of preference:

1. **Match on orientation alone.** Orientation matching needs no spatial registration at
   all — only the frame convention. The position residual of those pairs is then an
   *independent* test, because position never entered the matching.
2. **Where a discrete choice is unavoidable** (which reference axis maps to which FF axis,
   the rigid offset), pick it on a random **half** and report on the held-out half. On
   `shade_LSHR` train and test agreed to 0.1 µm, which is what says the choice generalised.
3. **Every number carries a null that can come out the wrong way.** Use a
   **texture-preserving** null — the same orientation set, rigidly rotated — so it has the
   same support as the real comparison and cannot succeed for trivial reasons.

### The threshold is not a free choice — measure the chance rate

With many grains the cubic fundamental zone is densely populated and a loose threshold
measures **zone density, not your reconstruction**. Measured on `shade_LSHR`, ~3900 grains:

| match threshold | real | chance (texture-preserving null) | lift |
|---|---|---|---|
| **0.5°** | 81.3 % | **0.3 %** | **×271** |
| 1.0° | 82.9 % | 2.1 % | ×39 |
| 2.0° | 85.5 % | 16.2 % | ×5.3 |
| 5.0° | 99.2 % | 94.8 % | ×1.0 |

Print this table for **your** grain count before quoting any match fraction.

### The frame convention is settled by SCATTER, not by offset

Matrix-vs-transpose, and any axis mapping, is a discrete choice: try both and let the null
decide. **Read the scatter, not the mean offset** — on the ring overlay for the same
dataset the *wrong* beam-centre convention gave the *better* median offset (0.05 px against
0.87) and was exposed only by its scatter (8.12 px against 0.12) and its ring contrast.
Orientation convention on `shade_LSHR`: as-written lift ×236, transposed ×0.8 — decisive.

### Separate a systematic frame error from random scatter

A median misorientation mixes the two. Fit the best **global lab rotation** between the two
orientation sets (resolve symmetry first, then average the residual rotations as
quaternions) and report the angle, its axis, and the median after removing it.

* An axis on **z** of ~one ω step is the `SkipFrame`/`OmegaStart` trap — the one error that
  changes every orientation while leaving every internal diagnostic healthy. This is the
  only way to exclude it from inside a finished reconstruction.
* Measured on three independent `shade_LSHR` reconstructions: **0.029°, 0.025°, 0.025°**,
  none on z — the trap excluded against an external reference rather than by inspection.

**Symmetry acts on the RIGHT.** MIDAS orientation matrices map crystal→lab, so an
equivalent orientation is `B·S`, never `S·B`. Applying it on the left gave a *plausible*
median (0.199°) with a maximum of 49° on one dataset and 38.6° on another. Guard the
result: if removing a best-fit rotation ever **increases** the median, that is
arithmetically impossible for a best fit — raise, do not report.

### FIRST establish whether the two modalities sample the same volume

**This changes what the position residual means, and getting it wrong cost a day.**

* If the FF layer is a **thin slice through the same plane** the reference sectioned —
  the usual design of an EBSD-comparison dataset — then FF's fitted position is the
  centroid of grain ∩ beam slab, i.e. a *section* centroid too. The two are directly
  comparable and the residual is a real **accuracy**.
* If FF integrates a volume much thicker than the reference's section, the two centroids
  differ by ~the grain radius *by definition*, and the residual is only an upper bound.

**Do not assume — test it, because the two cases predict opposite things.** Regress the
position residual against grain size:

| | prediction |
|---|---|
| same section | residual **falls** with size (more signal, less measurement error) |
| FF volume ≫ section | residual **grows** with size (the definitional gap scales with radius) |

Measured on `shade_LSHR`, all three independent reconstructions: slope **−0.36, −1.03,
−1.47** µm per µm of section radius — it *falls*, in every one. Residual by EBSD section
radius for the best run: 6.59 → 5.21 → 4.22 → **3.71 µm** across 0–5, 5–8, 8–12, 12–50 µm.
Second, independent check: FF `GrainRadius` correlates with the EBSD **section** radius at
Spearman **ρ = 0.76–0.81** (shuffled-pairing control −0.009). Same section, confirmed.

**Misorientation stays an upper bound regardless** — it contains the reference's own
angular precision, which is not separable. 0.216° on `shade_LSHR` bounds FF orientation
error; it does not estimate it.

### Report the position error AGAINST THE GRAIN SIZE, not on its own

A position figure in µm means nothing until it is divided by the grain radius. Measured on
`shade_LSHR`: **5.04 µm against a median section radius of 6.28 µm — a ratio of 0.80**, so
**40.6 % of correctly-matched grains have their fitted position outside the footprint of
the very grain they match.**

**This splits "matched" into two different questions, and they give different answers:**

| question | answer |
|---|---|
| does an EBSD grain *somewhere* have this orientation? (orientation-only) | **85.7 %** |
| does the EBSD voxel *directly beneath this grain's position* have it? | **56.8 %** |

Both are correct. The first is what the rest of §15g measures and is the accuracy result;
the second is what a **visual overlay** of grains on a reference map actually shows. An
overlay therefore looks far worse than the orientation agreement, and captioning one as
"see how well they line up" is an over-claim — it is a picture of the *position* limit.

**Practical rule:** use orientation to identify a grain across two modalities; use position
only when the error is small compared with the grain. Say which one a figure is showing.

### When the layer IS the section, the Z spread IS the Z error

A thin-beam layer scan is the rare case where truth is known by construction: every
indexed grain physically lies in a ~1.5 µm slab, so **the reconstruction's Z distribution
is its Z error distribution** — a quantity that is otherwise unmeasurable, because you
never know a grain's true Z.

Measured (robust sd about the layer's own median Z, which is *where the layer sits*, not
an error):

| run | in-plane error | **Z error** | ratio |
|---|---|---|---|
| 2024 published | 12.90 µm | **4.35 µm** | 0.34× |
| Aug 2026 | 6.61 µm | **3.24 µm** | 0.49× |
| best run | **5.04 µm** | **3.11 µm** | 0.62× |

True slab sd is 0.43 µm, so essentially all of it is measurement error (deconvolved 3.08 µm).

> **With a beam thinner than the grains, Z is BETTER determined than in-plane, not
> worse.** That inverts the usual expectation, and `ENVELOPE.md` §3's "Z is the
> badly-conditioned coordinate" row was measured with a **100 µm** beam. The beam itself is
> the constraint: a grain outside a 1.5 µm slab cannot diffract at all.

**A large |Z| is therefore a fit failure, not a position.** It is a free quality flag
needing no ground truth — filtering |Z − layer| ≤ 5 µm kept 75.7 % of grains and lifted the
match rate 72.3 → 81.3 %. **But it is not a better *ranking* than `DiffPos`**: ranked by
|Z| the top-2000 match rate was 95.0 % against `DiffPos`'s 98.5 %. Use it as a filter or a
diagnostic, not as the sort key.

### The same-section case also calibrates `GrainRadius`

`GrainRadius` is normally a ratio against the `Vsample` search bound and comparable only
within a run. When the layer is the section, the reference's **section radius** is the
physical truth for it. Measured: this run's `GrainRadius` sits at **1.23×** the EBSD
section radius — approximately calibrated — while the 2024 and Aug-2026 runs, with
different `Vsample`/`BeamThickness`, sit at **5.49×** and **5.51×**. So absolute grain
sizes were meaningful in one run and not in the others, which is invisible without an
external reference.

The relation is **monotonic but compressed** — log–log slope **0.407 ± 0.005**, not 1 — so
use it for ordering and for a rough scale, never as a calibrated size distribution.

With that scale, "where is this reconstruction trustworthy" becomes a physical statement:

| `GrainRadius` | ≈ section radius | matched |
|---|---|---|
| 10–14 µm | 8.1–11.4 µm | **99.4 %** |
| 8–10 µm | 6.5–8.1 µm | 96.8 % |
| 6–8 µm | 4.9–6.5 µm | 86.9 % |
| < 6 µm | < 4.9 µm | **53.9 %** |

### THE REFERENCE IS A SEGMENTATION, NOT GROUND TRUTH

The single most important finding of the `shade_LSHR` campaign, and it is not about FF:

> **Precision moved 13 points — 72.3 % → 85.7 % — with the reconstruction untouched,
> purely by re-segmenting the reference.**

The supplied EBSD grain list had **3893** grains; re-growing it from the raw `.mic`
(239 984 voxels, 2 µm grid, 4-connected, union-find on `misorientation_om_batch`, min 4
voxels) gives **4625**, and Sparks et al. 2024 Table 3 report **4496** for the same layer.
**49.5 % of that run's apparent false positives were real grains the coarse segmentation
had merged away.**

The count was nearly tolerance-independent (4646 at 0.5°, 4581 at 5°) because grain
boundaries are sharp — 82.1 % of adjacent voxel pairs below 1°, only 0.3 % between 1° and
3°. So the discrepancy was never the tolerance; it was a different segmentation criterion
entirely.

**Control that makes it a property of the reference and not of your run:** re-score a
*second, independent* reconstruction. The same re-segmentation rescued 62.6 % of the 2024
run's unmatched grains against 49.5 % of the new one — it lifts the older run *harder*, so
it cannot be the new pipeline inventing grains a permissive reference happens to accept.

**Never quote an FF precision figure without naming the reference segmentation.** Recall is
comparable *between runs on one reference*; it is never a fraction of the grains that
physically exist.

### Comparing two reconstructions of different size

Precision falls mechanically when a run reports more grains than the reference contains
(4312 FF against 3893 reference caps it at 90.6 % before any physics). The obvious fix —
truncate both to equal grain count by an internal, reference-blind metric — **carries its
own artefact**: it assumes that ranking is equally discriminating in both runs.

Measured: of 144 reference grains the older run found at equal N and the newer did not,
**103 (72 %) were in the newer run's list within 0.5°**, 81 of them merely ranked below the
`DiffPos` cut. Only 25 were genuinely absent. Report equal-N *and* full-count, and resolve
any disagreement by asking, of each "missing" grain, whether it is absent or merely
ranked low.

### Is an unmatched grain spurious? Ask whether it owns its spots

A grain indexed from coincidence must build itself from spots that other grains also claim.
Compute, per grain from `SpotMatrix.csv`, the fraction of its spots claimed by **no other
grain**, and compare unmatched against matched. **This needs no external reference at all.**

Measured on `shade_LSHR`: private fraction **0.496** unmatched against **0.507** matched,
~268 spots each so ~134 private; and at the extreme the unmatched were *less* borrowed
(private < 0.3: 1.1 % against 2.7 %). No borrowed-spot population existed, which is what
sent the investigation to the reference instead. Note 28.2 % of all spots were claimed by
more than one grain, so judge the *difference*, not the absolute level.

### Stratify before believing a trend

Match rate against `shade_LSHR` fell monotonically with |Z| — 91 % → 38 % — which reads as
"EBSD is a 2-D section and FF is not". **Stratifying by grain size killed it**: within the
largest size quartile the match rate is flat at ~98 % across every depth, and within the
smallest it is ~43 % *even at Z ≈ 0*. Size is the driver; |Z| bites only for small grains
beyond 10 µm. Small grains have badly-conditioned Z, so the two are confounded and the
marginal trend belonged to size.

The useful end state is a table like this one, which says where the reconstruction is
trustworthy and where it is not:

| our GrainRadius quartile | still unmatched after the better reference |
|---|---|
| largest (> 9.6 µm) | **0.7 %** |
| 7.7 – 9.6 µm | 4.4 % |
| 6.3 – 7.7 µm | 11.8 % |
| smallest (< 6.3 µm) | **39.1 %** |

---

## 15f. Done means

- [ ] `midas-grain-qa` run, and every stage either produced a number or printed a
      reason it could not.
- [ ] `d0` within a few tens of ppm, or the reference lattice corrected and the
      run repeated. A d0 error biases **hydrostatic strain only**; deviatoric,
      orientations, positions and sizes are unaffected.
- [ ] `calibration` run against an adjacent layer, and `sigma_obs_px` either
      confirmed or re-run with the implied value.
- [ ] The twin adjacency null read, not just the twin fraction.
- [ ] The attribution self-check separation read; below ~30 pp, do not use it to
      drop spots.
- [ ] Per-grain strain checked for resolvability before any strain map is shown.
- [ ] Anything surprising pushed through a control that could have failed.
- [ ] If an external modality exists, §15g run — matched on orientation alone, position
      reported on a held-out half, the chance-rate table printed for **your** grain count,
      and the global lab rotation checked for a `SkipFrame`-shaped error on z.
- [ ] Any precision figure quoted **with the reference segmentation named**, and the
      reference re-segmented at least once to see how far the number moves.
- [ ] Unmatched grains tested for spurious with the private-spot fraction before being
      called false positives.
- [ ] Every trend stratified against grain size before it is believed.
