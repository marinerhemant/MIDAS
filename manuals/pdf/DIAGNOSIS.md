# Diagnosis — symptom → discriminating test → cause → lever

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).

Indexed by **symptom**, because the step that produced a symptom is rarely the step you are
on. Every entry carries a test that can come back the other way. An entry whose test cannot
exonerate the cause it names does not belong here.

Numbers are from the reference beamtime: 1-ID-E, Varex 2880² at 150 µm, 67.4 keV, Lsd 470 mm,
beam centre at the detector edge, CeO2 / Ni / IPA / Kapton / air. On other detectors the
*test* transfers; the *value* may not.

---

## Local symptoms

| symptom | emitted by |
|---|---|
| `geometry.rhod_missing` | median of the per-pixel R map after `to_integration_spec()` is far beyond the detector diagonal |
| `geometry.map_dropped` | `spec.ResidualCorrectionMap` empty while the calibration directory holds `residual_corr.bin` |
| `q_axis.unverified_range` | Q of the last ring in the crest check against the Q_max of the transform |
| `intensity.uncorrected` | corrected / uncorrected 1-D profile ratio ≡ 1 |
| `grid.nonuniform_q` | `np.ptp(np.diff(q)) > 0` on the array handed to the transform |
| `sigma.uncalibrated` | robust std of interleaved-sliver z outside [0.8, 1.25] |
| `sq.rising_tail` | \|⟨S⟩[10,16] − 1\| with the scale anchored on the high-Q tail only |
| `gr.lowr_slope` | max \|G + 4πρ₀r\| below the first peak, as a fraction of the first-peak height |
| `refine.sigma_unscaled` | a refined σ quoted without √χ²_ν and √(N/N_eff) |
| `multiphase.degenerate` | NaN weight uncertainties from `refine_multi_phase` |
| `strain.blind_leak` | `strain_crlb` rank < 6 with every `per_component_ue` infinite |
| `rmc.species_X` | `set(supercell.species) == {"X"}` |

---

## G(r) of a liquid or amorphous sample has a huge peak below r ≈ 0.3 Å

**Test.** Run the identical chain on the uncorrected cake (`mean2d_uncorrected`), and on the
corrected one.

**Cause if it tracks the correction:** the polarization × solid-angle correction was never
applied (hard rule 1). On IPA the uncorrected arm gives G ≈ +410 at r ≈ 0.12 Å and ⟨S⟩ error
10.9; the corrected arm removes the spike (`$ANALYSIS/out/pdf_edge_q21_lorch/summary.json`).

**Lever.** Pass the per-pixel correction as `correction=` to the variance integrator (phase-5 §5.1).

---

## Corrected S(Q) keeps rising at high Q, and sits below zero at low Q

**Tests,** in the order that exonerates the most:
1. **Geometry.** Reduce with several calibrations. Four geometries agreed to three decimals
   on H4, so geometry was not it.
2. **Polarization plane.** Find the ring-intensity minimum in the uncorrected cake. It sat
   at η ≈ 83° (CeO2) and 93° (Ni), i.e. horizontal, so the default plane 90 is right
   (`$ANALYSIS/out/02b_polarization_edge_map.json`).
3. **Absorption magnitude.** Compare the package μ/ρ with xraylib `CS_Total`. They agree to four
   digits, and Ni's A_s,sc is flat to ±0.5 % over Q 1–21.
4. **Compton exponent.** k = 3 made both liquids worse.
5. **Detector obliquity efficiency.** CsI 200 µm took IPA H4a 1.38 → 0.09 and Kapton
   0.49 → 0.07; 600 µm did half as much.

**Cause.** Most likely a missing oblique-incidence detector efficiency. **Not established:**
the sensor thickness is unknown on this detector.

**Lever.** `detector_efficiency` with the data-sheet thickness. Do not use the thickness that
flattens S(Q) (hard rule 14).

`$ANALYSIS/out/pdf_sep11_none_map_q21_lorch{,_k3,_deteff200,_deteff600}/summary.json`.

---

## Low-r G(r) of a crystalline powder does not follow −4πρ₀r

**Test.** Hold every correction fixed and vary only the assumed number density and packing.
Then refine ρ₀ with `refine_normalization(..., fit_number_density=True)`, report-only.

**Cause.** An assumed bulk density for a dispersed powder. On Ni in carbon black, H4b stayed
≥ 0.30 in every arm, including the arm that fixed the liquids. A refined ρ₀ was not run.

**Lever.** A measured packing fraction and composition, stated as inputs (hard rule 15).

---

## Outer calibrant rings sit ~1 px off while the inner rings are perfect

**Test.** Crest positions of **isolated** rings only (≥ 12 px from both neighbours), off the
raw profile, over several distortion models.
- On the reference frame the drift was real and its **sign followed the model**: +482 µε at
  Q 17.2 with `none` + map, −533 µε with the ring-separation cut + map
  (`$ANALYSIS/out/01g_isolated_ring_crests.json`).
- GSAS-II's own geometry shows a drift too: −1300 µε at Q 21.

**Cause.** A dense calibrant at high energy. Beyond R ≈ 1100 px the rings sit closer than the
E-step window (±5.3 px), and beyond ≈ 1300 px they carry almost no intensity.

**Lever.** The recipe that passed:
- `calibrate(..., min_ring_separation_px=11, refine_distortion="none", n_iter=10)`;
- the residual map set by hand (hard rule 2);
- the Q scale beyond the last measurable ring reported as unmeasured (hard rule 4).

It gave 50.3 µε ring-mean RMS over 72 rings to Q 17.2, a best-of-14 selection.

---

## The E-step finds no measurable ring beyond some radius, even with wider η bins

**Test.** Run one E-step at EtaBinSize 5, 10, 15, 20° and count rings with SNR ≥ 2.
- Counting noise predicts ~2× SNR for a 4× wider bin.
- Measured: 36 / 38 / 39 / 35 rings, the outermost always at R 1203 px, and median SNR at
  R 1300–1500 px of 0.75–0.90 at every bin size (`$ANALYSIS/out/01i_estep_eta_scan.json`).

**Cause.** SNR is set by ring structure (neighbours inside the window, weak outer rings), not
by pixel noise.

**Lever.** None by binning. A per-ring quality filter buys stability, not reach
(`manuals/calibrate-integrate/LAB_NOTEBOOK.md` §10).

---

## The ring overlay looks shifted vertically, or mirrored

**Test.** Numbers, no plotting. Along fixed columns, find where each inner ring crosses in the
raw frame, above and below the beam centre, and compare with the R map.
- A beam-centre error shifts both crossings the same way.
- A plotting mirror changes neither.

On the reference frame the crossings sat −0.046 / +0.022 px from the map (MAD 0.06 px), against
a predicted mirror of 68.8 px (`$ANALYSIS/out/cal_edge/overlay_check.json`).

**Cause.** `contour(R[::4, ::4], extent=(0, W, H, 0))` with `origin=None` puts row 0 at the
bottom of an `imshow(origin='upper')` axis.

**Lever.** Contour with explicit coordinate arrays, and pin the axis limits.

---

## Every pixel's R is ~1e33 px, or the integration is empty

**Cause.** `RhoD` was not passed to `to_integration_spec()` on a calibration that refined
distortion (hard rule 2). **Test:** median of `eval_pixel_REta(spec)[0]`.

---

## G(r) looks plausible but every peak is slightly off in position and amplitude

**Test.** `np.ptp(np.diff(q))` on the array handed to the transform.

**Cause.** A non-uniform Q grid (hard rule 3).

---

## A refined lattice constant comes back with σ ≈ 1e-6 Å

**Test.** Compare χ²_ν with 1, and N with N_eff = (r_max − r_min)·Q_max/π.

**Cause.** An unscaled Hessian σ (hard rule 7). On Ni: 1.7e-6 Å raw, 4.5e-4 Å after scaling.

---

## U_iso is far larger than expected for the material

**Test.** Refit on the unwindowed G(r). On Ni U_iso went 0.0093 → 0.0051 Å².

**Cause.** A Lorch-windowed G(r) against a model with no window (hard rule 8).

---

## The calibrant's own lattice constant comes back hundreds of ppm off

**Test.** Refit across window, background order and FT Q_max. On CeO2 against 5.4116 Å:

| arm | Δa/a |
|---|---|
| headline (Lorch, bg 0) | −1107 ppm |
| background order 2 | −1114 ppm |
| no window | −611 ppm |
| FT truncated at Q 18 (Lorch) | −1155 ppm |

χ²_ν was 1e5–1e6 throughout (`$ANALYSIS/out/06a_smallbox_sep11_none_map.json`).

**Cause candidates**, none isolated:
- the model lacks termination, window and Qdamp;
- one U_iso is shared by Ce and O;
- CeO2 is the sample whose S(Q) fails H4 worst.

**Lever.** None in the package today. Carry the deviation as a chain uncertainty. **Do not**
rescale Q to make it vanish: the Q axis is already anchored to that calibrant, so the
correction would be circular.

---

## A Δ-PDF between replicate halves shows > 3σ features everywhere

**Test.** Robust std of the null z at r > 5 Å, plus a planted Q-scale control on one half.
- Null: 5.2 % of points beyond 3σ, robust std 1.54.
- After rescaling by that std: 0 %.
- Plants: +100 µε reached 33 % of r > 5 Å points, +500 µε 85 %
  (`$ANALYSIS/out/08a_delta_pdf_sep11_none_map.json`).

**Cause.** An uncalibrated σ (hard rule 6).

**Limit.** Interleaved halves share geometry, background and every correction, so a quiet
null bounds pixel noise only.

---

## strain-PDF reports percent-level strain on an unloaded powder

**Test.** Eigen-decompose the Fisher matrix and read the estimate along each determinable
eigen-direction.

**Cause.** Blind directions leaking through a pinv step (hard rule 11).

---

## A minority phase fits with a large weight and NaN uncertainties

**Test.** Compute the amplitude fraction w·s, fit a decoy phase known to be absent, and run a
raw-data matched filter with a planted control.

**Cause.** The weight–scale degeneracy (hard rule 9), or a phase that is not there (hard rule 10).

---

## An RMC output CIF has `X` atoms, or swap moves fail

**Cause.** `Supercell.from_crystal` species fallback (hard rule 12). **Lever:** set
`supercell.species` explicitly before refining.

---

## Two independent reductions agree, but the physical checks fail

**Cause.** They share the frame, the calibrant and the normalisation, so their agreement is
consistency (hard rule 16). **Test:** the H4 checks on each one, not the correlation between them.

## A Pearson-r agreement between two reductions passed adversarial verification's negative control — or didn't

**Test.** Compute the identical agreement statistic between the reference reduction and at
least one reduction of the SAME frame using a setting already known to be wrong (a bad Compton
exponent, a wrong absorption geometry, a mismatched azimuthal wedge). If the known-wrong arm
also passes, the statistic has no power here.

**Measured.** On the wheel-1 Ni frame, a Compton-exponent error, a 3× detector-efficiency
thickness error, and abandoning the matched wedge entirely ALL scored Pearson r ≥ 0.997 against
GSAS-II — statistically indistinguishable from the registered arm's 0.9974. Only a window
setting mismatch (Lorch on vs off) failed. H5 was REFUTED on this basis even though every
number in it independently reproduced exactly (hard rule 16).

**Lever.** A same-crystal, same-Q-calibration G(r) comparison needs a statistic sensitive to the
correction chain specifically — e.g. a residual after subtracting the shared crystallographic
peaks, or a comparison restricted to the high-Q tail where S(Q) → 1 is being tested (H4), not a
raw Pearson r on the full curve.

## A Bayesian posterior "confirms" the MAP on every model but one, and the one failure looks like a fluke

**Test.** Compare the prior width to the Hessian-implied likelihood width at the MAP (rescaled
by √χ²_ν). If the prior is more than, say, 10× wider, Bayes' rule alone will pin the posterior
near the MAP regardless of whether the sampler is doing anything meaningful. Then check whether
the one model that fails has a mundane numerical explanation (near-collinear nuisance
parameters a mean-field guide cannot represent; a single unconverged NUTS chain) before reading
its failure as a physics signal.

**Measured.** On the wheel-1 Ni small-box model, the prior on `a` was 170× wider than the
rescaled likelihood width; every model but one passed "within 2σ of the MAP," and the one
failure traced to three nearly-collinear background coefficients (pairwise correlation up to
0.98) defeating `AutoNormal`'s diagonal covariance — not to the physical model
(hard rule 18).

**Lever.** Also re-derive any z-score's likelihood σ against this project's own √(N/N_eff)
correlated-points convention (hard rule 7) before trusting it: an inconsistent σ can make a
real pass look like a narrow escape, or a real failure disappear.

---

## `RuntimeWarning: midas_hkls refused 1 of 11 shipped ions … Ce4+`

The shipped Ce⁴⁺ coefficients violate the electron-count sum rule: f(0) = 56.0 against Z − charge
= 54 (`packages/midas_pdf/midas_pdf/ionic_form_factors.py:108`, `:216`). Ce falls back to
neutral-atom scattering. Neutral Ce has f(0) = 58 where Ce⁴⁺ would be 54, so a ceria S(Q) carries a
7 % form-factor excess at Q = 0 that shrinks with Q.

---

## `RuntimeError: expected scalar type Double but found Float` from `bayesian_refine_nuts`

**Test.** Does it happen only with `bg_order` set, and only under NUTS? On the reference data SVI
with the same background ran.

**Cause.** The background coefficients are drawn from `dist.Normal(0.0, width)` in the default
float32, then multiplied into the float64 model
(`packages/midas_pdf/midas_pdf/bayesian_refine.py:122-130`).

**Lever.** `torch.set_default_dtype(torch.float64)` before the call. It changes no model choice.

**Next failure, same call:** `KeyError: 'bg_0'`. The chain is seeded from `map_init`, which
supplies `bg_j` only if you put it there, and pyro needs every latent site
(`bayesian_refine.py:285-296`). **Lever:** add the MAP's background coefficients to `map_init`, as
`bg_0`, `bg_1`, ….

---

## An SVI posterior mean sits several σ from the MAP of the same model

**Test.** Compute (mean − MAP) / posterior σ for each model, against **that model's own** MAP.
Then rerun SVI seeded from that MAP, with more steps.

On Ni, every model was seeded from the bg-0 MAP (`$ANALYSIS/out/12a_bayes_sep11_none_map.json`):

| model | (mean − own MAP) / σ |
|---|---|
| bg None | +0.48 |
| bg 0 | −1.79 |
| bg 2 | **+4.13** |

The reseeded rerun was not done.

**Cause candidates**, none isolated: the seed, the mean-field `AutoNormal` guide, the a prior
(Normal(init, 0.02)), too few steps.

**Lever.** Do not report an SVI mean for a model that has not been checked against its own MAP.
