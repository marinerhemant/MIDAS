# XRD-CT lab notebook — evidence, ledger, and what has been refuted

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Read §5 before re-investigating anything.** It records **four** results as refuted or
invalid, **one as downgraded**, and **three** inferences as withdrawn — each with the
measurement that killed it. None died of new physics. They died of a windowed sum that was
mostly background, a degrees-of-freedom mismatch that made a comparison meaningless, a plant
script that accepted a random seed and never used it, a positive control whose forward model was
gentler than reality, and an inference that was simply backwards.

**Last checked:** 2026-08-18 · **Owner:** MIDAS maintainers

---

## 1. The controlling measurement: area vs centroid

**Established, and it governs everything else in this doc set.**

| Quantity | Arithmetic | Scatter at 2 % contrast | Fate |
|---|---|---|---|
| Integrated area (→ texture) | difference of large numbers | **36 %** | unusable |
| Intensity-weighted centroid (→ strain) | ratio | **0.85 %** | usable |

Synthetic, no planted azimuthal structure, so all of it is extraction error
(`tests/test_azimuthal.py::test_centroid_survives_low_contrast_where_area_does_not`).
Reproduced on real data: the DAC Ti scan gave strain consistent across six independent
reflections while every texture signal was incoherent.

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
therefore **not** an SNR problem; it is a pipeline problem, which is exactly what makes it the
right dataset to fix against.

**5. ★ (331) and (420) are unusable at this geometry — a NEW gate.** They sit **26.7 px apart**,
closer than the windows they want. `ring_windows` correctly narrows both to half=12 on its gap
rule, but their apparent FWHM comes out at **29 px** against 2–4 px for every clean ring — that is
the neighbour inside the window. `count_maxima` still calls each a singlet, because within its own
narrowed window each *is* one. **A ring-centre gap check is a separate test from a within-window
multiplet test, and this dataset shows you need both.** Added to the spine's halt conditions.

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

## 6. Provisional — labelled provisional, and must stay that way

> **These have NOT been through `/verify`.** The label travels with the number into any text
> that leaves the session, not only into working notes.

**Ti deviatoric strain 0.3–0.7 %** — six reflections, 3352–7324 µε (5–95 % inter-percentile),
MAD 1228–3354 µε. Credible **because the six independent reflections agree**, which is the
right reason and still not verification.

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
* CeO₂ with **peak-fitted areas** — the test of the peak-movement diagnosis (§5b).
* U₃O₈ re-integration, regressed against the 2023 output.
* A **free-axis** (non-axial) fibre fit. The forward model supports it; it has not been run.
