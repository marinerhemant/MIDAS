# XRD-CT lab notebook — evidence, ledger, and what has been refuted

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Read §5 before re-investigating anything.** It records **five** results as refuted or
invalid, **one as downgraded**, **one superseded with its cause since identified**, **one
withdrawn for a coding defect**, and **three** inferences as withdrawn — each with the
measurement that killed it. None died of new physics.
They died of a windowed sum that was mostly background, a degrees-of-freedom mismatch that made a
comparison meaningless, a plant script that accepted a random seed and never used it, a positive
control whose forward model was gentler than reality, a fix aimed at a mechanism that could not
produce the symptom, a `nan_to_num` that turned a dead azimuthal column into a 34-pixel centroid
shift, and an inference that was simply backwards.

**Last checked:** 2026-08-20 · **Owner:** MIDAS maintainers

---

## 1. The controlling measurement: area vs centroid

**Established, and it governs everything else in this doc set.**

| Quantity | Arithmetic | Scatter at 2 % contrast | Fate |
|---|---|---|---|
| Integrated area (→ texture) | difference of large numbers | **36 %** | unusable |
| Intensity-weighted centroid (→ strain) | ratio | **0.85 %** | usable |

Synthetic, no planted azimuthal structure, so all of it is extraction error
(`tests/test_azimuthal.py::test_centroid_survives_low_contrast_where_area_does_not`).

> **★ THE REAL-DATA LEG OF THIS ENTRY WAS WITHDRAWN 2026-08-21.** It read:
> *"Reproduced on real data: the DAC Ti scan gave strain consistent across six
> independent reflections while every texture signal was incoherent."* Both
> halves are now void. The six-reflection agreement was **BUG A** — a shared
> `nan_to_num` systematic, not a shared measurement (§5h) — and on a corrected
> re-extraction α(012) **reverses sign**. And the DAC Ti scan **fails the scope
> gate**: ~4 crystallites per 0.3° azimuthal column (§5i), so its azimuthal
> quantities were never in scope, and "every texture signal was incoherent" is
> what crystallite-count fluctuation looks like, not evidence about texture.
>
> **What stands: the synthetic result only.** The area-vs-centroid asymmetry is
> a property of the arithmetic — area is a difference, centroid is a ratio — and
> the synthetic test demonstrates it cleanly. **It has no surviving real-data
> corroboration in this project.** Do not cite one until an in-scope dataset
> provides it.

**Not asserted:** that the area degrades *faster* in relative terms as contrast falls. It
does not — measured 2.7× for area against 3.3× for centroid between 50 % and 2 % contrast,
because the area is already at 14 % scatter at high contrast and has little headroom. The
claim that survives is about the **absolute** magnitudes.

## 2. The pole-figure operator — ESTABLISHED, three independent routes

The fibre integral of a GSH mode is closed form,
`(4π/(2l+1)) · conj(Y_l^m(y)) · Y_l^n(h)`, so the operator is a table of spherical harmonic
values. **No Rodrigues mesh, no tetrahedra, no fibre-tetrahedron intersection** — which
retires what a handoff plan called its critical-path item.

| Route | Result | Where |
|---|---|---|
| Closed form vs brute-force fibre quadrature, mode by mode | **5.6e-15** relative | `tests/test_gsh.py::test_fibre_integral_equals_closed_form` |
| Our own Monte-Carlo pole figure from sampled orientations | corr **0.99968** | `tests/test_texture_kernel.py`, slow marker |
| **TexTOM** (Frewein et al., IUCrJ 11 2024; Zenodo 10.5281/zenodo.12543638) — third-party, its grid, its Haar measure, its fundamental zone | corr **0.999992** | `scripts/validate_gsh_vs_textom.py` |

Analytic cross-check on the normalisation chain: with only `a₀ = 1`, the pole density equals
the **family multiplicity exactly** (8 for {111}, 6 for {100}, 12 for {110}, 24 for {210}).
Every factor in the operator has to be right for that to land on an integer.

**Prior art.** The closed form is Bunge's classical pole-figure equation; the determinacy
count stands; and TexTOM plus Carlsen et al. 2025 (J. Appl. Cryst. 58,
10.1107/S1600576725001426) already occupy this exact geometry. What is ours is the
implementation and its validation, not the mathematics.

**A note on TexTOM's cubic fundamental zone.** Their `ressources/symmetries.py` case `'432'`
imposes seven octahedral constraints; the eighth, `-R₁+R₂-R₃ ≤ 1`, is absent. The validation
script measures whether that omission is real (acceptance against the analytic 1/24, and
whether the accepted set is a fundamental domain at all). This is a statement about their
code, and it does not gate our result.

## 3. Symmetry — ESTABLISHED

* **All 230 space groups** give the correct Laue-proper order, and every element is
  orthogonal with det +1 (`tests/test_point_group.py`).
* `point_group_rotations('432')` reproduces an **independently written** octahedral group
  (signed permutation matrices, no generators, no closure) **element for element** — which
  is what transfers the cubic validation to the general machinery.
* **Improper operations must map to `-R`, not be discarded.** Friedel ⇒ the recoverable
  symmetry is the Laue group. Counted: of 230 space groups, **92** are centrosymmetric (`-R`
  already present, nothing changes), **65** are Sohncke (no improper operations to drop), and
  **73** are neither — those are the ones that break. Pm (#6) collapses to order 1 instead
  of 2; Pmm2 (#25) to 2 instead of 4.
* **Key group elements on the matrix, never on a quaternion.** `q` and `-q` are the same
  rotation and the sign tie-break is unstable exactly at `w = 0` — the 180° elements, of
  which every one of these groups has several. A quaternion-keyed closure returned **28**
  elements for 432.
* **Cubic has `M(2) = 0`**: there is no l=2 cubic invariant, so a cubic model cannot express
  what an l=2 term carries. 622 *does* have one, which is why a cubic-only model cannot
  represent hcp basal texture at all.

## 4. Truncation, identifiability, and regularisation — ESTABLISHED

**Required `L` rises with the number of crystallites a voxel holds:** L=6 at ~1000, L=8 at
~3000, L=10 at ~10⁴ — measured **in the overdetermined regime**, which is the only regime
where the measurement means anything.

**The trap that makes an L-sweep read backwards.** Unknowns grow roughly as `L³` while the
row count is fixed, so the system goes underdetermined at high `L`, and an underdetermined
least-squares fit drives its residual toward zero *for free*. Measured
unknowns/rows for 156 voxels and 24192 rows: L=6 → 0.30×, L=8 → 0.52×, L=10 → 0.79×,
L=12 → 1.43×, L=14 → 1.81×. "Reaches the noise floor at L ≥ 12" is therefore **not evidence
of anything**. Report the ratio with every residual.

**The discrepancy principle fixes the WEIGHT, not the PRIOR.** Three different priors reached
the same residual floor, and the run with the **best** residual (0.0606) gave among the
**worst** reconstructions (MAE 0.250 against 0.056). So choosing λ by discrepancy against a
measured noise floor is right, and it buys you nothing about whether the prior is correct.
Never rank reconstructions by residual alone.

**The odd-`l` ghost subspace is unrecoverable by any scan design** — `Y_l^n(-h) = (-1)^l
Y_l^n(h)` and diffraction cannot separate `h` from `-h`. This is *linear* algebra only: a
positivity constraint escapes it (Matthies), which is what both published texture-tomography
codes rely on. An earlier statement here claimed more than "linear"; it was weakened to
this on the prior-art gate.

**Kernel truncation loss, measured** (cubic-symmetrised kernel peak recovered vs truncation):

| half-width | L=6 | L=10 | L=16 | L=22 |
|---|---|---|---|---|
| 8° | **5.8 %** | 22.7 % | 61.6 % | 89.2 % |
| 16° | 46.8 % | 88.5 % | 99.8 % | 100 % |
| 40° | **100 %** | 100 % | 100 % | 100 % |

So it is **sharp** kernels that lose amplitude, not wide ones — a 40° kernel really is
band-limited by l=6. The mechanism is symmetry, not bandwidth: `M(2) = 0` for cubic
annihilates the l=2 term, which is the *largest* coefficient for any kernel sharper than
~30°, and L=6 leaves a cubic kernel with only l = 0, 4, 6.

**Two quantities that saturate, worth knowing before blaming the data:**

* `Ahat_l` itself **rises before it falls** (`χ_l(0) = 2l+1`); it is `Ahat_l/(2l+1)` that
  decays monotonically. For a 16° kernel `Ahat` peaks at l=4.
* The **Hermans `S` ceiling with only `a₂` free is ≈ 0.61**: +0.59 at `a₂ = 1`, maximum
  +0.611 near 1.35, then +0.586 at 2 and +0.447 at 5. A fit stalling near `S ~ 0.6` may be
  at this ceiling rather than at the data's limit; a sharper axial texture needs `a₄` and
  `a₆`.

## 5. REFUTED, RETRACTED and WITHDRAWN — do not resurrect

### 5a. `N_c* = 100–300` — **RETRACTED**

A claimed crystallite-count bracket for per-voxel ODF recoverability. **Refuted by all four
`/verify` lenses.** The plant was 99.52 % a single global cubic texture; `n = 1`; no rung was
significant at the 5 % level; and the bracket held at **one** η binning only. The generating
script's `plant_field` accepted an `rng` argument and never used it, so every "per-voxel"
plant was the same field.

**Lesson that generalises:** a ladder study needs its plant audited for actual per-voxel
content *before* the ladder is run, not after the answer is interesting.

### 5b. CeO₂ null — **REFUTED at L = 6, 8 and 10**

Spurious structured "texture" on a sample that should have none, median rising 0.25 → 0.45
with `L`. The residual was **flat in `L`**, so truncation is not the cause. Absorption was
excluded quantitatively (predicted effect 0.000 %).

> **★ SUPERSEDED 2026-08-20 by §5b-ter.** The text below records what was believed until then:
> that peak movement was the leading suspect and peak-fitted areas were the fix. **Both are
> wrong.** Movement is real, but a window wider than ~2× FWHM has an area that is *invariant* to
> sub-pixel movement, so it cannot be the cause — and peak-fitting therefore cannot be the fix.
> Kept here because the reasoning is instructive, not because it is current.

**Leading suspect, diagnosed but NOT proven: the peak is moving inside a fixed radial
window.** The two radial halves of every ring were **anti-correlated at −0.72**, which is the
direct signature of movement rather than of an amplitude change. The fix is peak-fitted
areas rather than windowed sums; this has not yet been run.

### 5b-bis. CeO₂ re-measured 2026-08-18 — the diagnosis firms up, and one story dies

Run on `11idc` against `cake_cache_resid.h5` with the promoted `midas_dt.azimuthal`, one
(translation, ω) frame, `max_half_px=16`, `block_bins=30`. Also the **first real-data exercise of
that module**, which ran unmodified.

| ring | R px | peak/bg | SNR/η | half-corr | | ring | R px | peak/bg | SNR/η | half-corr |
|---|---|---|---|---|---|---|---|---|---|---|
| 111 | 404.0 | 137.5 | 198.9 | **+0.219** | | 222 | 809.3 | 22.5 | 57.7 | **+0.265** |
| 200 | 466.6 | 53.5 | 101.9 | −0.024 | | 400 | 935.2 | 25.1 | 57.9 | −0.610 |
| 220 | 660.3 | 133.1 | 158.2 | −0.121 | | 331 | 1019.6 | 46.6 | 92.1 | −0.740 |
| 311 | 774.7 | 103.8 | 144.1 | **−0.707** | | 420 | 1046.3 | 30.1 | 72.3 | −0.717 |
| | | | | | | 422 | 1146.9 | 38.8 | 85.5 | **−0.737** |

**1. The −0.72 REPRODUCES independently** — five rings at −0.61 to −0.74, measured with different
code from the original finding.

**2. Window truncation is EXCLUDED.** The boring explanation was that outer rings are broader and
a fixed window clips them. Tested, and it fails: **FWHM is 2–4 px** on seven of the nine rings
against a 32-px window (8–16× wider), and the correlation is **stable across a 5× window sweep**
(half-width 6 → 32 px: 422 moves −0.803 → −0.641, 311 moves −0.737 → −0.660). Truncation would be
destroyed by widening. Peak *movement* now stands on much firmer ground than one number.

**3. A story died: it is NOT a smooth function of R.** The natural reading — a given strain shifts
outer rings further, so the effect grows with R — is **wrong**. **311 (−0.707) and 222 (+0.265)
are adjacent in R and opposite in sign.** It does not track the cubic orientation factor either:
200 and 400 both have Γ = 0 yet read −0.024 and −0.610. **The hkl pattern is unexplained, and is
recorded as an open question rather than a finding.**

**4. ★ CeO₂ is HIGH-contrast data: peak/bg 22–137**, against 0.005–0.17 on the DAC Ti scan —
three to four orders of magnitude apart. So none of the low-contrast area-vs-centroid pessimism in
§1 applies here, and the texture gate in `ENVELOPE.md` §0 passes comfortably. The CeO₂ null is
therefore **not** an SNR problem.

*(This paragraph originally continued "…it is a pipeline problem, which is exactly what makes it
the right dataset to fix against." **That conclusion is withdrawn** — see §5b-ter. It is a
*sample* problem, and CeO₂ is the **wrong** dataset to fix a pipeline against.)*

**5. ★ (331) and (420) are unusable at this geometry — a NEW gate.** They sit **26.7 px apart**,
closer than the windows they want. `ring_windows` correctly narrows both to half=12 on its gap
rule, but their apparent FWHM comes out at **29 px** against 2–4 px for every clean ring — that is
the neighbour inside the window. `count_maxima` still calls each a singlet, because within its own
narrowed window each *is* one. **A ring-centre gap check is a separate test from a within-window
multiplet test, and this dataset shows you need both.** Added to the spine's halt conditions.

### 5b-ter. CeO₂ — final state after four adversarial lenses

> **The RANDOM component is crystallite counting — now positively CONFIRMED. The FLOOR is a
> mixture, and at least a quarter of it is our own integrator.**

> **Status after adversarial verification (statistics + independent-reproduction lenses),
> 2026-08-20: the floor EXISTS, the CAUSE does not, and it is an EXTRAPOLATION not an observed
> plateau.**
>
> **Survives, independently reproduced from raw data with a from-scratch implementation:**
> random **3.741 %** (claimed 3.73), systematic **0.867 %** (claimed 0.900, agreement 4 %),
> per-frame RMS median **3.76 %** (claimed 3.7), Poisson excess **3.36×** (claimed 3.5),
> ω-half reproducibility **+0.822** (claimed +0.849) — and all six are insensitive to a
> structurally different background (≤1.2 % change). The floor is also **not** a sampling
> artefact: a pure-random null on the exact draw scheme yields ≤0.35 % against ~0.9 % observed.
>
> **Died:** the attribution to grain counting (chord exponent, both lenses), and the claim that
> the curve *plateaus*.
> Evidence: `~/Desktop/analysis/11idc_ceo2_dt/peakfit/RESULTS_ceo2_{peakfit,area_origin,sqrtn,floor}.md`.

**The structure is finite crystallite counting, at two levels:**

| component | size | behaviour |
|---|---|---|
| random per ω | **3.73 %** | averages down with N_ω — grain count *per frame* |
| **ω-locked floor** | **0.90 %** | **never** averages down — grain count *for the whole scan*; it **is** the ω-average's own residual |

Four of nine rings have an ω-locked component above 1.0 %.

**⛔ The positive evidence is WITHDRAWN (adversarial verification, 2026-08-20).** It was: "the
floor's amplitude scales as chord^(−0.572) against −0.50 predicted by 1/√N_grains". Three
independent kills:

1. **The exponent is manufactured by the analysis.** The amplitude was measured on
   `specific = pattern_t − mean_over_translations`. Subtracting the across-translation mean makes
   the residual largest at the *extreme* translations, which are also the *shortest* chords — so
   amplitude and chord anti-correlate **by arithmetic**, for any η pattern varying smoothly with
   position and with **zero** chord dependence. A simulated null with true exponent 0, calibrated
   to the observed corr = −0.877, returns median **−0.628** and
   **P(null ≤ −0.572) = 0.67**.
2. **It is a readout of an unfitted nuisance parameter.** The capillary radius was hard-coded at
   0.45 mm. Sweeping it: R = 0.42 → −0.303, 0.45 → −0.572, 0.48 → −0.785, 0.55 → −1.278. An 11 %
   change in an *assumed* constant moves the exponent by ±0.35, five times the gap being called
   "consistent with −0.50".
3. **The exponent cannot discriminate anyway.** Photon shot noise also gives chord^(−0.5);
   additive read/background noise gives chord^(−1.0). The measured −0.6 ± 0.13 sits in the
   mixture zone §4 of `RESULTS_ceo2_area_origin.md` had already fitted.

### ★ CONFIRMED on an axis nobody in the chain tested: η-bin-width scaling

The artifact lens ran the one clean positive test for a counting process — sweep the azimuthal
bin width and watch the relative RMS. A counting process gives slope **−0.5**; a smooth systematic
gives **0.00**.

**Measured over 45° → 2°, on 9 of 9 rings: slope −0.520 (range −0.447 … −0.538)**, with the noise
floor 20–60× below, and lag-1 η autocorrelation ≈ 0 (white) at every binning. Implied crystallite
size ~3 µm at 100 % packing (N_eff = 816 per 10° bin per ω on (111)) — self-consistent.

**So the ω-random ≈3.7 % bulk IS finite-crystallite counting statistics.** That part is
established, and on better evidence than the chord argument ever was. Every number in
`RESULTS_ceo2_*.md` was taken at one fixed 10° binning; the scaling axis was free the whole time.

### ★ But at least 25 % of the FLOOR is the integrator itself

`integrate_hard` returns `sums / counts` — a **mean over whichever pixels land in each (R, η)
bin** — so each bin reports intensity at the mean R of *its own* pixel set, and that effective R
jitters with the pixel lattice. Pushing a **perfectly uniform, perfectly noiseless** synthetic
image through the identical reducer and extraction manufactures **0.19–0.44 % azimuthal area RMS
from a sample with zero azimuthal structure and zero noise** — fixed in the detector frame, i.e.
ω-locked and translation-invariant.

That artifact correlates **+0.50** (up to +0.72) with the measured translation-shared floor,
**explaining ~25 % of its variance**. Its own ring-to-ring correlation is **+0.035** — so it
passes straight through observables D (−0.011) and R (+0.089), the very tests used to "exclude
flat-field / detector response". **The "unexplained ~1/3 translation-invariant residue" is, in
substantial part, the integrator.**

### ★ Process failure: not one measurement in the chain touched a raw frame

Every number — `area_origin`, `poisson_null`, `sqrtn`, `floor_frame`, `floor_absorption` — was
read from `cake_cache*.h5`, a pre-integrated product of someone else's script. **That is how a
0.2–0.4 % integrator artefact went unnoticed through five analyses.** Raw data before models is a
standing rule and it was not followed.

*(Two worries that died: the `np.clip` hypothesis — identical to 3 d.p. with clipping off — and
"the background estimator manufactures it" — a background built from the 361-ω average, with 19×
smaller noise, leaves the RMS unchanged. The `p0..p14` vs `iso_R2/a1/phi1` key naming is the same
15 numbers permuted, so that worry was unfounded too.)*

**A fourth kill, from the physics lens, is the most damaging:** the same model-free exponent
applied to the **random** component — the one this entry says *is* crystallite counting — returns
**−0.813**, identical to the floor's. **A test that gives the same answer for the component you
believe is grain counting and for the component you are trying to identify carries no
information.**

And the chord model is **empirically false**: the assumed cylinder (r = 0.45 mm, centre 0.713 mm)
predicts **zero** illuminated volume at hxz = 0.2 and 1.2, where the data carry **18 %** and
**28 %** of the peak ring intensity. Using the *measured* illuminated-volume proxy instead (the
ω-averaged net ring area) moves the exponent to **−0.805** (jackknife −0.818 ± 0.235).

**So grain counting is now an unsupported hypothesis, not an identification.**

### ★ The leading candidate is interpolation error in our own `background_from_ring_free`

Proposed by the physics lens, and it fits **every** observable — which grain counting does not:

| observable | measured | background-interpolation predicts | grain counting predicts |
|---|---|---|---|
| amplitude vs illuminated volume | **−0.81** | −1.0 (constant *absolute* error ÷ signal) | −0.5 |
| harmonic content P1 | **0.297** | ≈0.25 (white — percentile estimator ~independent per η bin) | no prediction |
| across rings R | **+0.089** | uncorrelated | uncorrelated |
| ω-locked? | yes | yes — the background field is fixed at a given translation | yes |
| translation-specific? | yes | yes — the scatter field moves with the sample | yes |
| **corr(floor, 1/S)** | **+0.859** | high | — |
| **corr(floor, 1/√multiplicity)** | **+0.601** | — | this is the grain-counting prediction, and it fits *worse* |

The two **weakest** rings (222 at S = 4463, 400 at S = 4372) carry the two **largest** floors
(1.544 %, 1.591 %); the brightest (111, S = 41725) carries nearly the smallest (0.663 %).
**Crystallite counting has no mechanism to make the floor track |F|².** A constant absolute
background error does exactly that. Across all 81 (ring, translation) points the
constant-additive model fits **1.65× better** than √S.

**This matters beyond CeO₂: `background_from_ring_free` is shipped package code.** If its
interpolation error is the dominant systematic on low-signal rings, that is a property of the
*extraction*, not of any sample.

### ★ Two of the six "exclusions" do NOT hold

* **Background model was never actually excluded.** Observable B is a 3×3 *parameter-sensitivity*
  sweep, not a model-**correctness** test: an η-structured error common to all nine settings scores
  0 % spread and reads as "excluded". It was also run on the wrong quantity (12 ω frames, so
  random-dominated rather than the floor). **And ring 222 measured 0.3084 — above its own
  registered 0.30 fire threshold** — while only the 9.6 % median was reported. That is a
  preregistration violation: a ring fired and the median hid it. 222 has the second-largest floor
  and the steepest model-free exponent (−1.190, exactly additive).
* **Flat-field was only partly excluded.** R = +0.089 is the *across-ring* η correlation, and
  different rings sit at different radii sampling entirely different pixels — so a pixel-level
  gain or defect structure gives R ≈ 0 **by construction**. R excludes only a detector response
  that is smooth in η and common to all radii.

### Attacks that FAILED — these exclusions stand

* **Peak movement is PRESENT — the earlier wording "excluded" was too strong.**
  `ceo2_area_origin_result.json` stores `half_corr` = −0.610, −0.631, −0.831, −0.967, −0.685,
  −0.879, −0.934, −0.958, −0.959 on **all nine rings**, and the planted-truth control calibrates
  ~0.25–0.55 px of azimuthal ring-radius movement at −0.88. **Neither results file quoted those
  numbers.** What is excluded is movement *as the cause of the area RMS*, not movement itself. A
  real ring displacement exists: the ω-averaged radial
  centroid carries a cos η modulation with **common phase across all nine rings**, 0.45 px at
  −96° (hxz 0.30) through ~0 at hxz ≈ 0.75 to 0.29 px at +99° (hxz 1.10) — a clean 180° flip. But
  re-integrating with the window **tracking** the centroid changes the translation-specific
  amplitude by **0.0 %**. Movement stays excluded for the floor.
* **Shot noise and absorption** are genuinely negligible: Poisson 0.50–1.76 % ÷ √361 =
  0.026–0.093 %; CeO₂ at 106.9 keV attenuates only 5–9 % more at centre than at edge, and with the
  wrong sign.
* **The control I skipped, run by the refuter, passed.** H was measured on the *full* pattern; the
  *translation-specific* part's reproducibility was never checked. It is **H_spec = +0.840**. The
  specific part is real.

**Everything else excluded, each by a test that could have come back the other way:**

| candidate | how it died |
|---|---|
| **peak movement** | on *mechanism*: a window > 2× FWHM has an area invariant to sub-pixel movement. Measured inflation from a planted 0.26 px shift: **1.00×** from 2.3× to 16.3× FWHM (1.37× only at 1.7×). CeO₂ windows were 8–16× FWHM |
| **residual geometry** | η pattern between the two caches correlates **+0.972**, RMS change 4.3 % |
| **background model** | RMS spread **9.6 %** across a 3 × 3 (block_bins × percentile) grid |
| **flat-field / detector response** | ring-to-ring η correlation **+0.089** |
| **photon shot noise** | a real **3.5×** excess above the Poisson prediction (range 2.4–8.1) |
| **capillary absorption** | **0 of 3** predictions, with the decisive one *inverted*: amplitude correlates with chord at **−0.877**, i.e. largest where there is *least* material |

**Why the absorption inversion is the answer.** Least material = fewest crystallites = largest
counting fluctuation. That is what pointed at grain counting, and the chord exponent then
confirmed it quantitatively.

**Consequence — now UNSUPPORTED, since the evidence for it was withdrawn above.** The reading was:
the CeO₂ null was never an analysis defect but a **sample limitation** — too few crystallites in the illuminated volume — and no change to
background modelling, peak fitting, geometry or harmonic truncation can fix it. The levers are
physical: finer powder, larger gauge volume, or more ω (and more ω removes only the 3.73 %, never
the 0.90 % floor). **So CeO₂ is the wrong null for a texture pipeline.**

**What is NOT settled:**
* **~1/3 of the floor is translation-invariant and unexplained.** Read against the reproducibility
  ceiling (ω-half correlation **+0.849**), the between-translation correlation of **+0.272** gives
  a translation-invariant fraction of **0.32**. Grain counting accounts for the translation-
  *specific* two-thirds. The residue is ~0.3 % absolute — small but real.
* **Detector gain unmeasured.** The additive noise term fitted `k = 2.36` against Poisson's 1.0,
  close to the **2.23** gain recorded in `known-limits.md` for an integrating sCMOS. Only one Air
  frame exists and there are no repeat exposures, so a photon-transfer measurement needs new data.
* **No independent grain-size measurement.** The identification rests on the chord exponent plus
  the exclusion of everything else.

**⛔ "Plateaus from N ≈ 64" is REFUTED (reproduction lens).** With 40 draws per point instead of
1, the curve is **monotone and still falling at the largest N**: ring 111 drops a further 9.5 %
from N=64→128 and 5.7 % from 128→256, reaching **0.664 %** where a single draw had read 0.762 %.
My own N=128 → N=256 points went *up*, which is single-draw scatter, not a plateau. At N=256 the
measurement is still 8 % above the fitted asymptote — and since N=256 of 361 shares ~71 % of its
frames between draws, that point cannot independently test the floor anyway.

**The floor is therefore a two-parameter EXTRAPOLATION**, not something the data settle onto. It
survives as a fitted quantity with a real value; it has not been observed directly.

**⛔ The registered slope threshold is background-fragile.** The "§5 REFUTED" verdict fired on
median slope −0.235 against a −0.25 line. Independent backgrounds give **−0.267** and **−0.245** —
straddling it. The refutation verdict flips with a defensible background choice; the substantive
finding (a non-zero ω-locked component) does not.

**★ The exact Poisson null was available and was thrown away.** `FrameReducer` computes and
propagates a closed-form Poisson variance; `build_cache.py` stores only `intensity` and discards
it. Worse, the cake holds the **per-bin mean** (`Σw·I/Σw`), not a sum — so `sqrt(Σ cake values)` is
**not** `sqrt(N_photons)`; it is off by `sqrt(gain/n_pix)`. Measured directly from ring-free
high-frequency scatter, `var/I` = 0.59–1.12 over I = 54–389, i.e. it lands near 1 only because the
Varex ADU gain (~100–200 at 105 keV) nearly cancels n_pix (70–200). **The 3.4× excess stands to
about +35 %/−0 %, but the gain reasoning in `RESULTS_ceo2_area_origin.md` §4 (k = 2.36 ≈ 2.23) is
not the operative factor.** Re-cache with the propagated variance and the null becomes exact.

**A method lesson worth more than the result.** The 1/√N_ω sweep was *registered as a hypothesis
and refuted* (slope −0.235 against a −0.25 refute line), yet it produced the key number — because
the registration pre-committed to fitting `RMS² = sys² + rand²/N` **regardless of slope**, since
separating the ω-locked residue was the actual point. Write registrations that stay informative
under their own refutation.

### 5c. Ti axial uniaxial ODF — **REFUTED**

| model | params | χ² | improvement |
|---|---|---|---|
| uniform null | 349 | 2.7491e5 | — |
| **one global texture** (+3 shared `a₂,a₄,a₆`) | 352 | 2.7461e5 | **0.11 %** |
| per-voxel (+1047) | 1396 | 2.7445e5 | 0.17 % vs null, 0.06 % vs global |

Registered refute line was < 5 %; measured 0.17 %. Hermans `S` median +0.141 with an IQR of
0.54 against a registered usability limit of 0.1 — scatter, not a map.

**The global rung is what makes this interpretable.** A per-voxel fit that fails could mean
"no texture" or "texture present but not spatially resolvable". A single sample-average fibre,
three parameters, buys **0.11 %** — so it is not a resolution limit.

**And the global rung is where the positive control is strong.** Planted texture gives 23–34 %
improvement on that rung across every contrast tested, against 0.17 % here — a gap of more than
100×. Note that the control's *per-voxel* recovery is much weaker than originally recorded
(§5f), but this refutation does not depend on it.

**Scope, stated because it is the obvious alternative and remains open.** The registration
fixed the fibre axis along the rotation axis. **If the cell was loaded in RADIAL geometry**
the texture varies with ω and this model cannot fit it by construction. So: *refuted* for a
fibre about the rotation axis; **not tested** for a fibre about any other axis. Resolving it
needs the cell's loading geometry — a fact about the experiment, not the data.

### 5d. RBF / non-negativity comparison — **INVALID, not a result**

The two arms had different degrees of freedom, so their residuals were not comparable and
**no conclusion about either basis follows**. Recorded so it is not cited as evidence that
non-negative radial bases do not help. The published claim (Carlsen et al. 2025) that radial
coefficients go sparse when a voxel holds few grains remains untested by us.

### 5e. "A 40° kernel loses most of its amplitude at L=6" — **CORRECTED, direction inverted**

Recorded during the RBF work and **wrong as stated**. Measured (§4): a 40° kernel keeps
100 % at L=6; it is the **8°** kernel that keeps 5.8 %. The underlying lesson — that
non-negativity constrains the full function, so a kernel used as a positivity basis needs its
own expansion order well above the operator's `L` — survives; the example did not.

### 5f. "The pipeline recovers planted texture at corr 0.60–0.75" — **DOWNGRADED**

**Found while promoting this code out of `dev/`, 2026-08-18.** The original DAC Ti positive
control reported per-voxel recovery at corr **+0.75 → +0.60** holding down to peak/bg = 0.02,
and that number was carried in the ledger as *established*. It does not reproduce.

Two optimistic choices in the original harness, both invisible in its output:

1. **The background was exactly known.** Its forward model added a flat pedestal and subtracted
   *the same constant* (`area = counts - BG*npx`). There was no background *model*, no window,
   no ring masking — so the extraction error that dominates real data was entirely absent.
2. **The fit was under-converged**, capped at 40 iterations with a numerical Jacobian on 276
   parameters. That cap acted as implicit regularisation.

Re-measured with the real extraction chain and a converged analytic-Jacobian fit, on a clean
plant (6 % noise, 308 normals per azimuthal bin), planted pole-figure `S` median −0.167:

| peak/bg | global improvement, **known** bg | \|corr\|, known | global improvement, **estimated** bg | \|corr\|, estimated |
|---|---|---|---|---|
| 0.50 | 22.9 % | 0.23 | 30.8 % | **0.35** |
| 0.20 | 28.6 % | 0.54 | 30.6 % | 0.15 |
| 0.10 | 32.7 % | **0.67** | 24.0 % | 0.13 |
| 0.05 | 33.6 % | 0.49 | 12.1 % | 0.21 |
| 0.02 | 25.4 % | 0.34 | **2.6 %** | 0.05 |

**Background-model cost: best `|corr|` falls 0.67 → 0.35** when the background must be estimated
rather than known. That gap is the term real low-contrast data is dominated by, and it does
**not** average down with more frames the way Poisson noise does.

**What survives:** *detection* on the global rung, at the contrasts where the Ti rings actually
sat. Planted texture gives 24–31 % improvement at peak/bg 0.1–0.5 against **0.11 %** on real Ti.
So §5c's conclusion — that the Ti data carry no coherent azimuthal texture — **stands**.

**What does not survive:** the claim that *per-voxel* recovery was demonstrated. `|corr|` runs
0.05–0.67 and is **non-monotonic in SNR**, peaking near peak/bg = 0.1 in the `known` arm and
never exceeding 0.35 in the realistic one.

**A new qualification the original control could not have found, and it matters.** With a
realistically estimated background, planted texture at **peak/bg = 0.02 yields only 2.6 %**
improvement — *below* the 5 % refute line §5c was registered against. So at the very bottom of
the Ti contrast range the test would refute genuinely textured data. The Ti conclusion is
therefore properly stated as: **no texture at the level a planted ~25 %-amplitude fibre would
produce in the rings whose contrast was 0.1 or better.** It is not a statement about the weakest
rings, and the sensitivity floor is contrast-dependent rather than a flat 5 %.

**Why non-monotonic, which is the interesting part.** At high SNR the limit is **model
mismatch**, not noise: the plant is a discrete-crystallite fibre distribution and the fit is a
4-parameter squared-modulus expansion, so the fit chases an azimuthal shape it cannot represent
and distributes the mismatch into the per-voxel coefficients. Noise was regularising the fit.
Consistent with this, the `estimated` arm — which carries *more* error — scored **better** than
`known` at peak/bg 0.5 (0.35 against 0.23).

**Consequence for practice.** Score *detect* and *resolve* separately, which
`scripts/odf_positive_control.py` now does. A null on the **global** rung is interpretable as a
statement about the sample; a null on the **per-voxel** rung is not, and a sample-average bound
should be reported rather than a map.

**Lesson that generalises:** a positive control's own forward model needs auditing as carefully
as the analysis it validates. This one was gentler than reality in a way that its own output
never revealed, and it had already been promoted to *established*.

### 5g. Three inferences WITHDRAWN

1. **"The η pattern is static in ω, therefore instrumental."** **Wrong.** `n_s·ẑ = cos θ_B
   sin η` carries no ω dependence, so a fibre about the rotation axis is *necessarily* static
   in ω. Static in ω is what an axial fibre looks like. This was used as a discriminator for
   several steps.
2. **"The η pattern is hkl-dependent, therefore texture."** Withdrawn: those sums were
   **60–85 % background**, and the background varies with both R and η, so hkl-dependence
   proves nothing.
3. **"Counts, not crystallites, set the limit"** — a physics-lens reading that was amplified
   into a design conclusion before being verified. Re-derivation **reversed** it: a
   thinned-to-300 case gave +0.506, not +0.023.

### 5h. DAC Ti S1 per-voxel radial gradient — **REFUTED twice, 2026-08-20**

A per-voxel centre-to-rim apparent-d gradient (+1829…+4996 µε, claimed
sign-consistent across five rings and two phases) was refuted by `/verify` on all
**four** lenses, then refuted again by a **pre-registered** re-run after the
extraction bugs were fixed. Both verdicts are in `~/.claude/skill-log.jsonl`;
working directory `~/Desktop/analysis/dac_ti_strain/`.

**Two extraction bugs, one of them inherited by §6's withdrawn number.**
*BUG A* — `np.nan_to_num` maps a dead raw-η column's centroid to **0 px**, then
ten raw columns are averaged into a bin: one dead column moves the bin ~34 px out
of ~345 (1–2 × 10⁵ µε). |t|-dependent, so it back-projects into a centred radial
pattern and *predicts* the sign consistency that was offered as physics.
*BUG B* — the rebin stored mean(area) and mean(centroid), so the moment was
mean(a)·mean(c) rather than mean(a·c).
**Fix:** drop dead raw columns from both sums *before* the rebin and carry
`area = Σaᵢ`, `moment = Σaᵢcᵢ`. Also `Σsig/√Σnoise²`, not the mean of per-column
SNRs — the two differ by √10, so the old SNR ≥ 3 gate was never the gate it
looked like.

**All three nulls were vacuous, in three different ways.** They planted a
*spatially uniform* field into a *linear* reconstruction, so `M = d·I` survives
masking elementwise and the ratio returns `d` regardless of truncation, missing
wedge or mask — the operator ledger `known-limits.md` (outside this repo) says a planted-uniform-field test cannot
see this, in advance. One null also planted a variation in **R** while the
measurement reported **d** (`dd/d = −0.99·dR/R`, so it was subtracted with
inverted sign), and its annuli had **zero voxel overlap** with the measurement's.

**After the fix, the pre-registered test refuted the sample-level claim too:**
α(100) **+2850**, α(110) **+1041**, ω(300) **+1071**, α(012) **−323** µε; ring
spread **8.8×** against a 3× limit. **α(012) reversed sign** from +4970. And the
decisive internal check: **α(100) and α(110) both have c-axis fraction χ = 0 —
they measure the same a-axis — yet differ by ~1800 µε with non-overlapping CIs.**
No strain field can do that; it bounds what this dataset supports regardless of
model.

**α(101) vindicated the doc set.** `manuals/xrd-ct/ENVELOPE.md:99` says to exclude it as a
doublet; v1 kept it on a gap/FWHM of 2.04 against a 2.00 gate. Under a correct
`n_live == 10` selection it yields **zero** usable bins.

**Window width was suspected and then EXCLUDED by measurement.** `E_even` looked
monotonic in window/FWHM across the four rings (2.52× → −323 … 5.71× → +2850),
which would have been the signature of tail leakage. A 16-arm sweep at full ω
resolution, with the **background peak-mask held fixed** so only the integration
window varied, shows all three testable rings **converge**:

| ring | arms ≥ 2.3× FWHM | span |
|---|---|---|
| α(100) | +2748, +2778, +2844, +2864, +2936 (to 6.1× FWHM) | **188 µε** |
| α(110) | +1136, +962 | **174 µε** |
| ω(300) | +1080, +1020, +1220 | **200 µε** |

The n=4 monotonicity was a coincidence. `ti_s1_windowsweep.py`.

### ★ THE MEASUREMENT FLOOR FOR CENTROID STRAIN ON THIS DATASET IS ~1750 µε

Because the window is excluded, **α(100) ≈ 2800 µε and α(110) ≈ 1050 µε stand as
robust measurements — and both have χ = 0, i.e. both measure the same α-Ti
a-axis.** They differ by a factor **2.7** at every window width tested. No strain
field can give one lattice parameter two values, so this is an uncontrolled
systematic of ~1750 µε, and it is a property of the *data*, not of a
specification choice.

**Consequence:** the pre-registered meaningful effect size was 1000 µε. The floor
is above it. **DAC Ti S1 cannot answer the radial-gradient question at all** —
definitively, not provisionally. Do not re-attempt it on this dataset without
first explaining the α(100)/α(110) split.

**Diagnostic hint for whoever takes it up.** Converting each `E_even` to the
centroid shift it implies, `ΔR = ε·R`:

| ring | R (px) | E_even (µε) | implied ΔR (px) |
|---|---|---|---|
| α(100) | 344.9 | +2800 | **+0.966** |
| α(110) | 599.4 | +1050 | +0.629 |
| ω(300) | 666.9 | +1100 | +0.734 |
| α(012) | 513.6 | −323 | −0.166 |

An additive sub-pixel offset compresses the spread from **2.67× to 1.53×** — so
the systematic behaves more like a **sub-pixel centroid offset** than like a
strain, though neither model is clean and α(012) fits neither.

### 5h-bis. Chasing the α(100)/α(110) split — two mechanisms tested

**Note on the physics first.** "Both measure the same a-axis so they cannot
differ" is **not a valid argument for a polycrystal** — different hkl sample
different grain families, and diffraction elastic constants plus type-II
intergranular strains routinely differ between reflections. The argument that
*does* hold here: **hcp is elastically transversely isotropic about c**, and both
α(100) and α(110) have `l = 0`, so their plane normals lie in the basal plane
where all directions are elastically equivalent. Their diffraction elastic
constants are therefore equal. Plastic anisotropy could still split them; a
factor 2.7 is far too large for that.

**The window sweep excludes two mechanisms analytically**, because both scale
with window width while the result does not (span ≤ 200 µε over 2.3×→6.1× FWHM):
a linear background residual shifts the centroid by `~e₁(2W³/3)/A` (**W³**), and
neighbouring-ring tail leakage grows with **W**.

**H1 pixel locking — RULED OUT.** Sub-bin phase of the measured centroids is
uniform on all four rings (KS statistic 0.002–0.005, p = 0.16–0.98, n ≈ 54 000
each). Expected, since the cake is binned at 0.25 px against a 5.25–8.5 px FWHM,
but measured rather than assumed. `ti_s1_shapediag.py`.

**H2 width change × off-centre window — LIVE, and it exposed a real defect.**
The peaks **broaden markedly at the edge chords**: α(100) **+45 %**, ω(300)
+43 %, α(110) +18 %, α(012) +13 % — and `E_even` is **perfectly rank-correlated**
with relative width change (Spearman 1.0, n = 4, p = 0.042). This is degenerate:
a real strain gradient *along the ray* also broadens the peak and shifts its
centroid. But the diagnostic showed the **ring-centre positions are hard-coded
from the 2021 ring assignment and were never fitted to the data**:

| ring | window offset from its own peak |
|---|---|
| α(100) | −0.563 px |
| α(110) | **+0.044 px** |
| ω(300) | **−1.524 px** (19 % of its FWHM) |
| α(012) | −0.848 px |

An off-centre window truncates one flank more than the other, so a **width**
change becomes an apparent **centroid** shift with a per-ring gain set by the
per-ring offset — and a fixed fractional offset does **not** wash out as the
window widens, which is why the window sweep could not see it.

**★ But the mechanism is quantitatively too weak, and that was checked rather
than assumed.** Modelled on a Gaussian with the measured +45 % width change
(`tests/test_azimuthal.py`), the artefact depends steeply on `half/σ`:

| `half/σ` | window/FWHM | fake shift | µε at R = 345 px |
|---|---|---|---|
| 5.00 | 4.2× | 0.02–0.22 bins | **15–161** |
| 2.50 | 2.1× | 0.72–2.38 bins | 522–1727 |
| 1.75 | 1.5× | 0.82–2.42 bins | 591–1753 |

The Ti rings sit at `half/σ` = 6.73 α(100), 4.71 ω(300), 3.39 α(110), 2.96
α(012). The model therefore predicts **α(012) most affected and α(100) least**,
which is exactly the re-centring result — but for ω(300) it predicts ~20 µε
against **585 µε observed**, i.e. it is **1–2 orders of magnitude too small**.
So Gaussian tail truncation is NOT why re-centring matters on this data;
non-Gaussian content moving in and out of the window (background residual,
neighbour tails) is. **Do not quote the off-centre window as the explanation for
the ring split — it is a real defect worth fixing, not the cause.**

**Standing lesson regardless of that outcome: do not centre a radial window on a
catalogued ring position.** Fit the peak, or centre on the measured centroid.
This is the same disease as §5b's CeO₂ null, whose prescribed fix was
peak-fitted areas rather than windowed moments — one fix serves both.

**Re-centring result.** Windows re-centred on each ring's measured peak, all else
identical (`ti_s1_recentred.py`): α(110), whose offset was +0.044 px, is
**bit-identical** — the control works. ω(300) moves **+55 %** (1100 → ~1685) and
α(012) **flips sign to ≈ 0** (−323 → +21…+152). So centring is a real error
source worth up to 55 % and a sign. **But the α(100)/α(110) split survives at
2.8×** (2950 vs 1050), so centring is not its cause and the ~1900 µε floor stands.

### ★ 5h-ter. The decisive argument: the broadening has the WRONG SIGN

For a disc with a radial profile `d(r)`, a chord at offset `|t|` samples radii
`|t| → R`. The **centre** chord therefore spans the full `d(0)…d(R)` and must
give the **broadest** peak; an edge chord spans almost nothing and must give the
**narrowest**. A real radial gradient predicts *edge narrower than centre*.

**Measured: edge peaks are BROADER in all four rings** — α(100) +45 %, ω(300)
+43 %, α(110) +18 %, α(012) +13 % — and `E_even` tracks that broadening at
**Spearman 1.000** (Pearson +0.899, n = 4). The broadening is **not** a
low-count artefact: the SNR drop at the edge chords (ratios 0.38–0.75) is
**uncorrelated** with it (Pearson +0.050, Spearman −0.400), and the area ratio
tracks the SNR ratio to three figures, so signal loss does not drive it either.

**Conclusion.** Whatever generates `E_even` is driven by the edge-chord
broadening, and that broadening has the opposite sign to the radial gradient it
was being used to support. **`E_even` on this dataset is not a measurement of a
radial strain gradient.** This is a geometric argument, not a statistical one,
and it does not weaken with n = 4. `ti_s1_snrwidth.py`.

**Still unexplained, and the honest open question:** *why* the edge chords
broaden. Candidates not yet separated — the ray passing close to the gasket where
the stress state differs sharply, partial illumination / penumbra at the chamber
edge, or relative growth of a contaminant contribution as the sample signal falls
26–63 %.

### ★ 5h-quater. The raw frames were not gone, and the geometry is exonerated

Earlier entries here said this needed the `.ge5` frames and that they were absent.
**They are not.** `/gdata/dm/1ID/2021/hpldrd_dec21/data/ge5` on **copland**
(5.7 TB, DM status *live*) holds all 50 `DAC_Ti_S1_PFocus_seg1_*` — **both**
y-layers — and this beamtime's **own CeO₂ calibrant**. The DM tree is
year-partitioned, so `ls /gdata/dm/1ID/` misses it; use the 1-ID gdata inventory.

The byte count settles the format for good: 8192 header + **653** frames of
2048²×2, so `HeadSize 8396800` (header + one throwaway) leaves **652** and
integrated frame 0 sits at ω = −169.0, exactly as the macro states.

**Unseeded calibration on that calibrant** (no `BC_guess`, bounds in detector
pixels only): **Lsd = 1 002 136.5 ± 1.9 µm**, BC (1022.354, 991.032) matching the
par to 0.002/0.004 px, **residual strain 7.1 µε** against the <100 µε gate, and an
overlay of 16 rings agreeing to **−0.82…+0.53 px**.

**So the distance was right all along.** The gap to `bt_1id_dec20.txt` is 590 µm —
316σ on a 2 ppm determination, but only **−0.059 %**, i.e. **589 µε** of scale
error, comfortably below the ~1900 µε floor. **Geometry does not explain the
α(100)/α(110) split.** The doc set's standing warning that a stored distance is
usually wrong did not hold here, and it is worth recording that it did not.

*Two caveats on the new calibration:* `phi5` came back unconstrained (<1σ), and
RhoD resolved to 294.4 mm against an outermost fitted ring at 183.7 mm, so the
high-order radial terms are weakly determined. The 2021 integration used
**RhoD 200000 µm**, so its `p0–p3` are **not** interchangeable with these.

### ★ 5h-quinquies. The 2021 cake and `midas_integrate_v2` disagree by up to 0.33 px

With the raw frames in hand, one S1 frame (file 754, integrated frame 326 = raw
frame 327) was re-integrated at the **identical 2021 geometry** — same Lsd, BC,
tilts, `p0–p3`, RhoD 200000, same R and η axes read from the 2021
`.REtaAreaMap.csv` rather than assumed. Any difference is then the **integrator**
and nothing else.

| ring | 2021 cake (px) | re-integrated (px) | Δ | as µε |
|---|---|---|---|---|
| ω(001) | 309.4982 | 309.2898 | **−0.2084** | +673 |
| α(100) | 344.9973 | 344.8687 | −0.1286 | +373 |
| α(101) | 391.0262 | 390.9829 | −0.0433 | +111 |
| α(012) | 512.8751 | 512.5439 | **−0.3311** | +646 |
| α(110) | 599.5933 | 599.5068 | −0.0864 | +144 |
| ω(300) | 666.7927 | 666.5998 | −0.1930 | +289 |

Median −0.161 px, **ring-to-ring spread 0.288 px ≈ 562 µε**.

**Normalisation is excluded as the cause.** The two cakes differ in mean by
8.55×, but the ratio is **flat inside every clean ring window** — slopes −0.04 to
+0.015 %/px, implying centroid shifts of only −0.008…+0.0015 px. A constant scale
cannot move a centroid. (α(101) alone shows −1.12 %/px, which is its 382.4 px
doublet neighbour inside the window, not normalisation.)

**So a ring-dependent radial offset of a few hundred µε is baked into the 2021
cake** — precisely the failure `known-limits.md` predicts for an analysis that
never touches a raw frame. It is a real contributor to the ring incoherence,
though at 562 µε of spread it is **not the whole** ~1900 µε floor.

**Not established:** which integrator is right. `midas_integrate_v2` 0.5.1 is the
tested one and the 2021 code is unseen, so the presumption is against the cake —
but that is a presumption, not a measurement.

**Two integration gotchas found doing this, both silent:**
* `integrate()` returns **(n_eta, n_r)** — the transpose of the 1-ID `.bin`
  `[nR][nEta]`. Both reshape cleanly. Verify by collapsing each axis.
* `np.median(np.diff(eta_axis))` on a 0.3° axis gives `0.29999999999999716`, and
  the integrator then computes 1207.0000…1 → **1208** bins, one too many. Round
  bin sizes before putting them in an `IntegrationSpec`.

### 5i. ★★ DAC Ti S1 is COARSE-GRAINED and was out of scope all along — 2026-08-21

The finding that retires everything above, and the one that should have come first.

Measured on raw frames, integrated with the verified geometry, at the **full 0.3° azimuthal
resolution** (`ti_scope_gate.py`, `ti_scope_gate2.py` in
`~/Desktop/analysis/dac_ti_strain/`):

| ring | cv_robust / cv_Poisson | crystallites per 0.3° column | % of ring intensity in >3 MAD spikes |
|---|---|---|---|
| α(100) | **225** | **4.7** | 4.1 % |
| α(110) | **223** | **3.8** | 4.1 % |
| α(012) | **174** | **4.4** | 2.3 % |
| ω(300) | **247** | **4.1** | 2.9 % |

Azimuthal intensity varies **40–59 %** where shot noise allows **0.2 %**. The bright spikes are
**not** the story — 0.2–1.7 % of columns carrying 2–6 % of intensity; the apparent *continuum*
is itself ~4 grains per column. At the 3° working bin that is ~40 grains, so cv ≈ 1/√40 ≈ 16 % —
**exactly the amplitude of the Ē(η) structure that four `/verify` lenses had just refuted as
strain.**

**This explains every earlier observation at once**, which no other hypothesis did:
reproduces across the two layers (2 µm apart in a 20 µm sample — *the same grains*);
uncorrelated between rings (different hkl diffract from different grain subsets);
survives widening the radial window (a spot is inside either window); and carries power at
**harmonics n ≥ 3**, which `E(η) = q̂·ε·q̂` cannot produce for **any** strain tensor.

**Consequences.**
1. **Azimuthal intensity here is crystallite-count fluctuation.** Per the spine, every texture
   number this pipeline produces from it is noise. The technique for this sample is
   **scanning-3DXRD → `pf-hedm`**, not `xrd-ct`.
2. **§1's real-data leg is withdrawn** (see the box there), and with it the spine's
   "deviatoric strain from centroids is real-data-proven" — that rested on this dataset.
3. **The scope gate was checked on ring CONTINUITY and passed.** It was never checked *per
   azimuth at the working bin size*. `phase-0-survey.md` §0.1b now carries the measurement.

**The methodological lesson, which cost a day.** Four artefact hunts came back negative and
were read as converging on "it must be the sample". They were not: they were **scope checks
being run as artefact checks**. A dataset outside the technique's stated scope will pass
artefact tests, because the structure is real — it is simply not the quantity being claimed.
**Check the scope gate quantitatively before, not after, the analysis that depends on it.**

## 6. Provisional — labelled provisional, and must stay that way

> **These have NOT been through `/verify`.** The label travels with the number into any text
> that leaves the session, not only into working notes.

**Ti deviatoric strain 0.3–0.7 %** — ~~six reflections, 3352–7324 µε (5–95 % inter-percentile),
MAD 1228–3354 µε. Credible **because the six independent reflections agree**~~
**→ WITHDRAWN 2026-08-20. Do not quote.** The extraction it came from
(`ti_full_extract.py`) zero-fills a dead raw-η column's centroid via
`np.nan_to_num` and then averages ten raw columns into a bin, so one dead column
drags that bin by ~34 px out of ~345. Measured on 12 real frames, bins that
*passed* the SNR gate while carrying a zeroed centroid: α(012) 2 @ 199 652 µε,
α(110) 4 @ 100 035 µε, ω(300) 7 @ 200 120 µε. The defect is |t|-dependent, so it
concentrates in the low-SNR outer translations. On a corrected re-extraction
**α(012) reverses sign**. The "six reflections agree" argument is therefore void —
they shared a common systematic, not a common measurement. See
`~/Desktop/analysis/dac_ti_strain/` and §5h.

**Ti texture bound |S| ≲ 0.1** — from amplitude² scaling of the measured 0.17 % against a
planted 25 %.
**Softener that must travel with it:** the positive control carries **Poisson noise only**,
while the real data is dominated by systematic per-frame *background-model* error, which does
not average down the same way. **The true bound is looser than this scaling implies.**

## 7. Dataset facts, decoded — do not re-derive

**DAC Ti (hcp α + ω), 1-ID.** `652 ω × 1207 η × 2400 R`, float64, layout **`[nR][nEta]`**,
25 files = 25 translations. `startOme = −169`, `omeStep = 0.25`, **ω NEGATED** (1-ID aero
convention), first frame already skipped via `HeadSize 8396800`. Use the `.REtaAreaMap.csv`
for R↔2θ — it carries the detector tilts and distortion coefficients, which an idealised
`arctan(r/L)` does not. **Peaks are 0.5–17 % above background**, the fact that governs
everything. Ring assignment good to 0.2 px; 14 rings assigned, **6–9 vetted**. α(101) at
393.6 px is a **doublet**.

**CeO₂, 11-ID-C.** Sample-to-detector distance **1632 mm, measured from the data**. The
metadata value (1600 mm) and the beamline calibration (1579.5 mm) are **both wrong** for this
data as collected. 9 rings, cubic.

**U₃O₈, orthorhombic.** Parked. The 2023 run kept 20-px windows and only ~2 rings survive;
it needs re-integration. In its parameter file, `Wavelength 0.136994 # 55.618 keV` — **the
comment is wrong**, that λ is 90.5 keV. Trust the number, not the comment.

## 8. What has never been done

Stated so absence is not mistaken for a negative result:

* **No XRD-CT texture dataset has yet produced a positive per-voxel ODF result here.** One
  scan was refuted, one null was itself refuted, one is parked. The operator is validated
  three ways; the *application* has no positive real-data result.
* ~~CeO₂ with peak-fitted areas~~ — **no longer worth doing**: peak movement was excluded as
  the cause on mechanism (§5b-ter), so peak-fitting cannot fix the area structure.
* U₃O₈ re-integration, regressed against the 2023 output.
* A **free-axis** (non-axial) fibre fit. The forward model supports it; it has not been run.
