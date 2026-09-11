# Lab notebook — evidence, measurements and retractions

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).

The spine says what to do. This says **why**, and records what had to be withdrawn. Read it
before re-investigating anything here.

**One beamtime, two analyses.**
- **The data.** Wheel-1 PDF frames at 1-ID-E: CeO2, Ni powder in carbon black, liquid IPA, an
  empty Kapton capillary and air. Varex 2880² at 150 µm, 180-frame sums, Ta K edge 67.4164 keV,
  Lsd ≈ 470 mm, beam centre at the detector's left edge.
- **§1** is an earlier analysis of those frames (2026-06, on pre-release midas-pdf), and what did
  not survive.
- **§2 onward** is a from-scratch re-run of 2026-09-10 on current packages, read against a
  pre-registration: 12 hypotheses with confirm/refute thresholds fixed before any output
  (`$ANALYSIS/PREREGISTER.md`, read in `$ANALYSIS/RESULTS.md`).

**Status labels are load-bearing.**
- ESTABLISHED = survived adversarial review.
- PROVISIONAL = measured, not independently attacked.
- REFUTED = a stated claim or threshold the measurement contradicts.

**Nothing below has been through `/verify` yet.** Every positive entry is PROVISIONAL, and the
refutations are measurements that have not themselves been attacked.

`$JUNE/...` names the earlier analysis directory (`midas_pdf_test/output`); like `$ANALYSIS`, it
is provenance, not a link.

---

## §1. The earlier analysis of these frames (2026-06) — what did not survive

The earlier report (`$JUNE/REPORT_for_Leighanne.md`, `$JUNE/06_full_demo_summary.txt`, notebook
`$JUNE/midas_pdf_leighanne.ipynb`) made the claims on the left.

| claim | what the re-run measured | status |
|---|---|---|
| a(Ni) = **3.52449 ± 0.00001 Å** (report); 3.5245 ± 0.0000 (summary); 3.52493 in the notebook | a_Ni = 3.524673 Å with σ_total **3.9e-3 Å** by a pre-registered recipe; σ_stat alone is 4.5e-4 Å (§7). All three earlier values sit well inside that. | the ±1e-5 Å uncertainty **REFUTED** (394×) |
| Hessian σ on a "reflects that robustness honestly", at χ²/ndof 1456 after inflating σ ×20 | the Hessian σ carries neither √χ²_ν nor √(N/N_eff). On the re-run's own fit those two factors multiply σ(a) by 267. | **REFUTED** |
| Ni/air = 0.166 "matches the attenuator transmission" | the attenuator PV read 0 on every file; the ion chambers put Ni at **0.098** of air | **REFUTED** |
| 335 µε "is the documented Varex 4343CT basis-incompleteness floor" | the accepted re-run geometry reached **50.3 µε** ring-mean RMS on the same frame, over rings to Q 17.2 (§3) | **REFUTED** as a floor |
| IPA's low-r feature is "from incomplete Kapton-as-empty subtraction" | leaving out polarization × solid angle reproduces it exactly (+410 at r ≈ 0.12 Å); applying the corrections removes it (§4) | cause **REFUTED** |
| "S(Q) → 1 at high Q for all three samples" | with the corrections applied, S(Q) of every sample rises past 1 from Q ≈ 15–16, and the pre-registered ⟨S⟩ check fails on all of them (§4) | **REFUTED** |
| Δ-PDF: "1499 of 1500 r-points differ at > 3σ" | the two G(r) were two reductions of the same counts, compared with an independence formula. On genuinely disjoint pixel sets the propagated σ is itself 1.5× too small (§5). | **REFUTED** at the statistic |
| midas-pdf "ticks all 8" corrections of the standard taxonomy, oblique incidence included | no oblique-incidence or detector-efficiency correction was applied, and the integrator applied no polarization or solid angle (§4, §10) | **REFUTED** |
| calibration seeded from a manual ring picker, `auto_seed=False` | `calibrate()`'s own `make_seed`, from the image alone, recovers the beam centre within 0.44 px and Lsd within 0.04 % of GSAS-II (§2) | superseded |

**Why it matters procedurally.** Every row is a silent wrong answer: a converged fit, a clean
figure, a plausible number. They became hard rules 1, 5, 6, 7 and 16. The report sent with those
claims was not retracted with its recipients at the time of writing.

---

## §2. Seeding from the image — PROVISIONAL (H1 INCONCLUSIVE as registered)

`calibrate()` was given no beam-centre or distance hint and chose `make_seed` (17 rings, rms
0.51 px). It refined BC (1.515, 1473.879) px and Lsd 470 618.8 µm.
- Against GSAS-II's calibration of the same frame: ΔBC_y +0.015 px, ΔBC_z −0.436 px,
  ΔLsd −0.040 %.
- The registration also required the contours to sit on the crests at every radius. They did
  not beyond R ≈ 1000 px (§3), so H1 is INCONCLUSIVE.

`$ANALYSIS/out/cal_edge/summary.json`, `verify.json`.

**A plotting error, caught by the user.** The first full-view overlay was mirrored top to bottom
by `contour(R[::4, ::4], extent=(0, W, H, 0))` over an `imshow(origin='upper')`, a predicted
shift of 68.8 px. Ring crossings measured numerically, along four columns above and below the
beam centre, sat −0.046 / +0.022 px from the R map (MAD 0.06 px), so the geometry was right and
the figure wrong (`$ANALYSIS/out/cal_edge/overlay_check.json`). The misfit of the outer rings in
that figure was exaggerated. The misfit itself stands on the crest table.

---

## §3. The outer calibrant rings — measured misfit, weak rings, what was accepted — PROVISIONAL

**The package default does not converge here.** With n_iter 4 and full distortion, in-loop
strain went 3880 → 630 → 932 → 607 µε, and cross-validation failed (held-out 927.6 against
train 337.4 µε). Ring-mean crest RMS off the raw profile was 717 µε.

Variants on the same frame (`$ANALYSIS/out/cal_<tag>/summary.json`, `verify{,_map}.json`):

| variant | in-loop µε | post-residual µε | package CV | crest RMS µε, without / with the residual map |
|---|---|---|---|---|
| default (full distortion) | 606.5 | 291.2 | FAIL | 717.3 / 581.1 |
| `refine_distortion="radial"` | 445.4 | 242.3 | FAIL | 296.3 / 332.9 |
| `"none"` | 588.7 (still falling) | 147.2 | ok | 460.4 / 152.4 |
| `min_ring_separation_px=11`, full | 924 best, 1298 final (oscillating) | 107.0 | FAIL | 505.9 / 190.6 |
| separation 11, `"radial"` | 195.2 | 93.6 | WARN | 363.6 / 403.4 |
| **separation 11, `"none"`, n_iter 10 (accepted)** | 351.3 | 43.5 | ok | **264.6 / 50.3** |

**The drift is real, and its sign follows the model.** Crests of isolated rings (≥ 12 px from
both neighbours) remove the crest finder as a suspect. At Q 17.2 the drift was +482 µε for
`none` + map and −533 µε for separation 11 + map (`$ANALYSIS/out/01g_isolated_ring_crests.json`).
GSAS-II's own geometry leaves +700 µε at Q 7–9 and −1300 µε at Q 21 on the same frame
(`01d_gsas2_ceo2_crests.json`). Neither is independent evidence; both are model-dependent
residuals.

**The outer rings carry almost no signal.**
- One E-step at the `none` geometry gave per-ring SNR of 15–46 inside R 530 px, 2–3 at
  R 1000–1230 px, and below 1 beyond R ≈ 1300 px (Q ≈ 13). Only 36 of 169 rings reached SNR ≥ 2
  (`01h_estep_ring_quality.json`).
- Where rings are measurable, the E-step centroids agree with the raw crests.
- Widening the η bins from 5° to 20° left the count at 35–39 and the outermost measurable ring at
  R 1203 px (`01i_estep_eta_scan.json`). Counting noise would have given ~2× SNR, so the limit is
  structure, not noise.

**λ was not the cause (H2b).** A calibration at the monochromator readback energy (λ +0.125 %)
moved Ni peak Q by ≤ 11.3 ppm below Q 8, but by −444 ppm at Q 19.6. If Q followed λ the shift
would be +1253 ppm, of the opposite sign. So the high-Q divergence is two fits disagreeing where
neither is constrained (`01c_lambda_anchor.json`).

**Accepted (the user's decision, 2026-09-10: "Good enough calibration! Move on!!!!!" / "Outer
rings have nothing, no intensity").** The last row of the table.
- It is a best-of-14 selection on the very statistic it is quoted by, so 50.3 µε is not a clean
  confirmation, and H2a as registered stays REFUTED.
- The crest check's ring list was built with `_ring_table(n_rings=120)`, which ends at Q ≈ 17.2.
  So the Q scale from 17.2 to the transform's 21 Å⁻¹ is **unmeasured**, not measured-good
  (hard rule 4).

---

## §4. Intensity corrections, and whether S(Q) is physical — PROVISIONAL (H4 REFUTED)

**Package defect.** The variance integrators apply no intensity correction and do not read the
spec flags (`packages/midas_integrate_v2/midas_integrate_v2/corrections/integrated.py:117-135`).
On this detector the per-pixel polarization × solid-angle factor reaches ×2.3 at Q 20 Å⁻¹ and
×4.8 at 26. Omitting it reproduces the earlier IPA artifact (§1).

**The pre-registered checks.** H4a is |⟨S⟩[10,16] − 1|, with the scale anchored on [18, 21]
only; confirm ≤ 0.05 (Ni ≤ 0.10), refute > 0.20. H4b is max |G + 4πρ₀r| below the first peak ÷
the first-peak height; confirm ≤ 0.10, refute > 0.30. On the accepted geometry
(`$ANALYSIS/out/pdf_sep11_none_map_q21_lorch*/summary.json`):

| arm | Ni H4a / H4b | IPA | Kapton | CeO2 (report-only) |
|---|---|---|---|---|
| **headline** | 0.276 / 0.363 | 1.380 / 4.303 | 0.490 / 1.348 | 1.069 / 0.731 |
| no intensity corrections | 0.461 / 0.340 | 10.92 / 9.87 | 4.21 / 6.71 | 0.048 / 0.365 |
| Compton k = 3 | 0.283 / 0.368 | 1.587 / 5.102 | 0.570 / 1.605 | 1.074 / 0.737 |
| GSAS-II's wedge and Q max | 0.233 / 0.349 | 1.234 / 3.331 | 0.477 / 1.285 | 1.006 / 0.936 |
| detector efficiency, CsI 600 µm (report-only) | 0.233 / 0.330 | 0.672 / 1.873 | 0.220 / 0.540 | 1.010 / 0.673 |
| detector efficiency, CsI 200 µm (report-only) | 0.187 / 0.300 | **0.091 / 0.517** | **0.071 / 0.305** | 0.947 / 0.621 |

**H4 is REFUTED on every registered sample and on every one of four geometries**, which agree
to three decimals.

What was ruled out, each by a test that could have implicated it:
- **Polarization plane.** The ring-intensity minimum sits at η ≈ 83° (CeO2) and 93° (Ni)
  (`02b_polarization_edge_map.json`).
- **Absorption.** Package μ/ρ equals xraylib `CS_Total` to four digits, and Ni's A_s,sc is
  flat to ±0.5 %.
- **Compton exponent.** k = 3 made both liquids worse.

**The only lever that moved the liquids by an order of magnitude is an oblique-incidence
detector efficiency, and thinner CsI helps more.** That points to an obliquity-type angular
factor; GSAS-II's own reduction carries ObliqCoeff 0.3. It is **not evidence**: the
registration left detector efficiency out because the thickness is unknown, and choosing the
thickness that passes would fit the correction to the test (hard rule 14).

Ni H4b stays ≥ 0.30 in every arm, which an assumed bulk density for a dispersed powder would
explain; that was not tested. CeO2 barely moves and is not diagnosed.

---

## §5. σ calibration — H3 REFUTED for every model; H8 INCONCLUSIVE by rule

**H3.** Disjoint sets of interleaved 1° slivers, z = Δ/σ; confirm robust std(z) ∈ [0.8, 1.25]
(`$ANALYSIS/out/03c_sigma_edge.json`).
- The planted control passed: 1.034 / 1.034 / 0.985 / 1.970 against 1 / 1 / 1 / 2.
- Robust std (air / Kapton / IPA): `poisson` 19.4 / 18.1 / 15.0; `azimuthal` and `hybrid`
  2.98 / 2.99 / 2.80; `measured` (photon transfer) 3.52 / 3.58 / 3.52.
- Report-only: Ni `measured` 1.04, CeO2 1.50.
- Contiguous halves read 44–364, which is background structure. That is why the registration
  was amended to slivers before integration.
- **Post hoc, not attacked:** a fractional per-sliver systematic of f = 0.05–0.12 % brings every
  frame to 1 (`03d_sigma_structure_edge.json`). That is what signal-proportional azimuthal
  structure predicts. It hides under the ~2400 ADU readout noise on the dim Ni frame.

**H8.** The same slivers on Ni, reduced through the full chain
(`$ANALYSIS/out/08a_delta_pdf_sep11_none_map.json`).
- The null exceeds 3σ on 5.2 % of r points; robust std(z) 1.54; 0 % after rescaling by it.
- Planted Q-scale changes on one half: +100 µε → 33 % of r > 5 Å points beyond the rescaled 3σ;
  +500 µε → 85 %. So the test has power at the registered 100 µε.
- By the registration's own rule an uncalibrated σ makes H8 INCONCLUSIVE.
- **Limit:** interleaved halves share geometry and every correction, so a quiet null bounds
  pixel noise only.

---

## §6. Agreement with GSAS-II — PROVISIONAL (H5 CONFIRMED as registered, not /verify-d)

MIDAS reduced GSAS-II's wedge (MIDAS η 90 ± 30.5°) with GSAS-II's Q max (20.5) and its Lorch
setting for Ni (on). Compared on r 1.8–10 Å, Ni gave Pearson **0.9974** (confirm ≥ 0.98), and
first-peak positions 2.4897 vs 2.4906 Å, Δr **−0.0009 Å** (confirm ≤ 0.005)
(`$ANALYSIS/out/05e_h5_pdf_sep11_none_map_q20.5_lorch_eta90w61.json`).

- **This is consistency, not accuracy.** Both reductions share the frame and Faber-Ziman, and the
  same MIDAS S(Q) fails H4 (hard rule 16).
- The η mapping onto GSAS-II's 60–121° wedge was derived, not read: GSAS-II's powder pattern
  reaches 2θ 42°, which only the direction into the detector allows
  (`05d_gsas2_wedge_edge_map.json`).
- The CeO2 Pearson of 0.926 compares MIDAS Lorch-on with GSAS-II Lorch-off, per its own PDF
  Controls. That is a settings mismatch, not a finding.
- The `.gpx` was read with a restricted unpickler admitting only numpy and `_codecs` globals.

---

## §7. Small box — PROVISIONAL (H6 REFUTED; H7: the earlier ±1e-5 Å REFUTED)

`refine_structure` (cubic a, one U_iso, scale, constant background), r 1.5–15 Å, the q21 Lorch
G(r) (`$ANALYSIS/out/06a_smallbox_sep11_none_map.json`).

**H6. The calibrant's own a comes back 1107 ppm low.** CeO2 refined to a = 5.405608 Å against
5.4116 Å, the value `calibrate()` used (NIST SRM 674b). Refute is > 300 ppm; σ_stat was
383 ppm and χ²_ν 4.1e5. The other arms:

| arm | Δa/a |
|---|---|
| background order 2 | −1114 ppm |
| no window | −611 ppm |
| FT truncated at Q 18 | −1155 ppm |
| no window + FT at Q 18 | −569 ppm |

**The window alone moves a by ~+500 ppm.** The model has no termination, no window and no Qdamp,
and one U_iso shared by Ce and O. CeO2 is also the sample whose S(Q) fails worst. No mechanism
is claimed.

**H7. a_Ni = 3.524673 Å, σ_total 3.94e-3 Å** (4.68e-3 with the registered-run calibration term):

| term | Å | from |
|---|---|---|
| σ_stat | 4.49e-4 | Hessian 1.68e-6 × √4787 × √(1351 / 90.2) |
| σ_rscale | 1.77e-4 | 50.3 µε (geometry used); 717 µε in the registered run gives 2.53e-3 |
| σ_chain | 3.90e-3 | 1107 ppm (H6) × a_Ni |
| σ_choice | 1.77e-4 | half-range of 3.524476–3.524829 over {FT Q_max 18, 21} × {Lorch, none} × {bg 0, 2} |

- (a_Ni − 3.5240)/σ_total = 0.17.
- Without σ_chain, whose transfer from CeO2 to Ni is questionable (the Ni fit is 86× better by
  χ²_ν and passes H5), σ_total is 5.1e-4 Å. That is still 10× the refute threshold for the
  earlier ±1e-5 Å.
- **U_iso depends on the window:** 0.0093 Å² with Lorch, 0.0051 without (hard rule 8).
- The `midas-pdf-refine` CLI reproduced the API fit to 1e-10 Å and reports the same raw
  Hessian σ.

---

## §8. strain-PDF on an unloaded powder — H9 INCONCLUSIVE as registered; post hoc in-plane reading

**Setup** (`$ANALYSIS/out/09a_strain_pdf_sep11_none_map.json`). Wedges at η 45 / 75 / 105 / 135°,
30° wide; the 15° and 165° wedges do not reach Q 21 on an edge-centred beam. Kernel m 8, 96
quadrature directions, empirical wedge σ_G 0.0212.

**What the package returned.**
- **`strain_crlb`:** Fisher rank 3, and **every** Voigt component marked undetermined. The null
  space is e12, e13, and e11 carrying −0.05 of e22 and e33. The per-component cut marks anything
  with a null projection above 1e-6, which catches e22, e33 and even e23
  (`packages/midas_pdf/midas_pdf/strain_pdf.py:205-206`).
- **`recover_strain`:** e11 = **+50 534 µε**, e22 = e33 = −3262, e13 −9308, e12 +1985, e23
  −273 µε, with no uncertainty. That is ~5 % along the blind direction.
- No refute clause can fire and no component is determinable, so the verdict is INCONCLUSIVE.

**Post hoc (not registered)** (`09b_strain_inplane_sep11_none_map.json`). A linear in-plane fit
at zero strain, J by `jacfwd`:

| component | ε | CRLB, empirical σ_G | × √(N/N_eff) |
|---|---|---|---|
| e22 | +69 µε | 7.7 | 30 |
| e33 | −123 µε | 15.3 | 59 |
| e23 | −25 µε | 5.8 | 23 |

- All are ≤ 300 µε. They sit 4–9 raw CRLB out, and within 3 after inflation.
- The Fisher eigen-directions agree: −140 ± 16, −25 ± 6 and −20 ± 5 µε.
- e33 − e22 ≈ −190 µε is the size of the accepted geometry's residual cos 2η amplitude (median
  A2 113 µε), which the known-limits ledger predicts for one powder calibration. Geometry is the
  boring candidate. Not claimed.

---

## §9. Phases: raw data first — NiO ABSENT; decoy INCONCLUSIVE; core-shell null non-zero

**NiO(111) at 2.605 Å⁻¹ in the raw Ni I(Q)** (`$ANALYSIS/out/10a_nio_raw_sep11_none_map.json`).
- **Method:** matched filter, quadratic background plus a fixed-width Gaussian, with the null
  from the same statistic elsewhere, because H3 left no calibrated σ.
- **Result:** z = **−0.83** at instrument width (σ_Q 0.0115 Å⁻¹, from Ni(111)) and −0.92 at 3×.
- **Controls:** a planted 5σ peak at 2.45 returned 5.13 / 5.19; Ni(111) itself gave 3419.
- The largest positive excursions in 2.2–2.9 were z 2.50 at 2.3215, where CeO2(200) sits at
  2.322, and 2.39 at 2.700. Neither is claimed.

**CeO2 decoy on the Ni G(r)** (`10b_decoy_coreshell_sep11_none_map.json`, `13a_cli_roundtrip_*.json`):

| decoy fit | amplitude or weight | χ² change | decoy a | decoy U_iso |
|---|---|---|---|---|
| API, start weights 0.5 / 0.5 | −1.1 % (negative scale) | −4.0 % | 5.639 Å | — |
| API, start weights 0.98 / 0.02 | +1.5 % | −3.4 % | 5.353 Å | — |
| CLI | weight 0.234 | χ²_ν 4448 vs 4787 | 5.972 Å | 2.23 Å² |
| structure fixed at the calibrant | **0.29 ± 0.39 %** (z 0.75) | −0.6 % | fixed | fixed |

The free decoy never looks like CeO2, yet takes 3–7 % off χ²: the extra parameters absorb misfit
(hard rule 9). As registered, INCONCLUSIVE.

**Core-shell (Ni core, NiO shell) as a null.** It does not return zero, and the reason is the
model:
- **API:** shell fraction 0.36, shell a 5.03 Å, shell U_iso 25 Å², R_core 63.8 Å, χ² −47 %.
- **CLI:** shell fraction 0.58, shell U_iso 0.43 Å².
- **Uncertainties:** five of eight (API) and six of eight (CLI) are exactly 0.0.

A shell with U_iso of tens of Å² has no structure. The χ² gain is consistent with the particle
envelope standing in for the Qdamp the model lacks (not tested). `--pin-geometry` changed
nothing (hard rule 17).

---

## §10. Package defects found — engineering, each re-checked against source

1. **Integration corrections.** The variance integrators ignore `PolarizationCorrection` /
   `SolidAngleCorrection` (`packages/midas_integrate_v2/midas_integrate_v2/corrections/integrated.py:117-135`),
   so `midas_pdf.image_to_iq`, `integrate_to_Gr_with_variance` and `midas-integrate-v2-pdf` never
   correct. The `midas_pdf.frontend` docstring claims otherwise (`packages/midas_pdf/midas_pdf/frontend.py:1-9`).
2. **The geometry hand-off.** `AutoCalibrationResult` does not carry `RhoD`, and
   `to_integration_spec()` sets neither `RhoD` nor `ResidualCorrectionMap`
   (`packages/midas_calibrate_v2/midas_calibrate_v2/pipelines/auto.py:197`; the rho_d fallback is
   at `.../forward/geometry.py:217-225`). `midas_dt.geometry.spec_from_calibration` omits `RhoD`
   too.
3. **`midas-integrate-v2-pdf`.** `--mask` is not applied on the G(r) path; `--compton` and
   `--absorption-mu-R` are parsed and unused; there is no `.h5` input; ⟨f²⟩ = 1
   (`packages/midas_integrate_v2/midas_integrate_v2/cli.py:678-843`).
4. **Ce⁴⁺ form factor.** The shipped coefficients break the electron-count sum rule (f(0) 56.0
   against 54), so midas_hkls refuses them and Ce falls back to neutral
   (`packages/midas_pdf/midas_pdf/ionic_form_factors.py:108`, `:216`).
5. **`refine_structure`.** σ is cov = 2H⁻¹ with no χ²_ν or correlation scaling
   (`packages/midas_pdf/midas_pdf/structure.py:340-346`). There is no termination, window or
   refined Qdamp in the model.
6. **`refine_multi_phase`.** Weights are degenerate with the per-phase scales
   (`multi_phase.py:83-101`), and weight σ is NaN (`:228`).
7. **Strain-PDF.** `strain_crlb` marks every component infinite once any null-space leak
   exceeds 1e-6, and `recover_strain` returns values along blind directions with no uncertainty
   (`strain_pdf.py:205-206`, `:250`).
8. **RMC species.** `Supercell.from_crystal` labels every atom `X`, because the midas_hkls tensor
   exposes `.elements`, not `atomic_symbols` (`rmc/supercell.py:93-97`). Reproduced from the API
   (CeO2: 12 × `X`) and from the CLI (the output CIF).
9. **CIF U_iso.** `read_cif_to_crystal` ignores `_atom_site_U_iso_or_equiv`
   (`cif.py:262-270`): U_iso 0.0063 came back as B 0.0.
10. **Core-shell CLI.** `midas-pdf-coreshell --pin-geometry` is parsed and echoed, never applied
    (`cli/coreshell_cmd.py:59`, `:103`); output is identical with and without it.
11. **Core-shell uncertainties.** `refine_core_shell` reports σ = 0.0 for most parameters when
    its Hessian is degenerate.
12. **`fourier_sine_transform`.** It takes dQ = q[1] − q[0] and so assumes a uniform grid
    (`packages/midas_integrate_v2/midas_integrate_v2/pdf.py:169`).
13. **`bayesian_refine_nuts` with a background crashes.** It stops at once with
    `RuntimeError: expected scalar type Double but found Float`, because the background
    coefficients are sampled from `dist.Normal(0.0, width)` in the default float32 and then
    multiplied into a float64 model (`packages/midas_pdf/midas_pdf/bayesian_refine.py:122-130`).
    SVI with the same background does not hit it. The workaround that keeps the model is
    `torch.set_default_dtype(torch.float64)` before calling it
    (`$ANALYSIS/logs/12a_nuts_sep11_none_map_attempt1_dtype_error.log`).
    **Past that, it fails again** with `KeyError: 'bg_0'`. The chain is seeded from `map_init`,
    which contributes `bg_j` only if the caller supplied it, and pyro needs every latent site
    (`bayesian_refine.py:285-296`). The fix is to pass the MAP's background coefficient as
    `bg_0` in `map_init` (`$ANALYSIS/logs/12a_nuts_sep11_none_map_attempt2_keyerror_bg0.log`).

Nothing here was fixed as part of this work, and the maintainers have not been asked.

---

## §11. RMC on the real Ni G(r) — H11 INCONCLUSIVE; the position miss is the disorder bias — PROVISIONAL

**Setup** (`$ANALYSIS/out/11a_rmc_sep11_none_map.json`):
- 4 chains on a 4³ FCC Ni supercell (256 atoms, box at the small-box a = 3.524673 Å).
- 10 000 displacement moves each, r 1.5–6.5 Å.
- Target divided by a small-box scale; kernel U 0.001 Å²; 48 min wall.

**Result:**
- χ² fell from 19–24 k to 380–395 in every chain, a spread of 4.0 %. Acceptance 0.59–0.60.
- Pooled first shell: mean **2.49828 Å** against a/√2 = 2.49232 Å (**+0.0060 Å**), width
  √(var + 2U_kernel) = 0.1250 Å against √(2U_iso) = 0.1364 Å (ratio 0.92), Z 11.88–11.93.
- The chains agree and the width passes. The position lands between confirm (≤ 0.005 Å) and
  refute (> 0.02 Å), so the verdict is INCONCLUSIVE.

**Post hoc, and the reason to read RMC positions carefully.** A mean pair distance under
isotropic disorder sits above the static distance by about s²/d. With the measured spread
s ≈ 0.117 Å that predicts +0.0055 Å, against +0.0060 observed. The box was fixed at the
small-box a, so the first-shell position could not have measured a anyway. What the registered
clause measured is the disorder bias.

## §12. Bayesian posterior and model ranking — H12 CONFIRMED on the registered model by SVI and NUTS; SVI fails the same check on another model — PROVISIONAL

**Setup** (`$ANALYSIS/out/12a_bayes_sep11_none_map.json`). Ni q21 Lorch, r 1.5–15 Å. The
likelihood σ is the chain σ_G × √χ²_ν of the bg-0 MAP. SVI ran 2000 steps with 500 samples,
seeded from the bg-0 MAP.

| model | SVI a (Å) | posterior σ(a) | own MAP a (Å) | (mean − MAP)/σ | WAIC |
|---|---|---|---|---|---|
| bg None | 3.524772 | 2.20e-4 | 3.524667 | +0.48 | −1320 ± 100 |
| **bg 0** (registered) | 3.524306 | 2.05e-4 | 3.524673 | −1.79 | −1334 ± 96 |
| bg 2 | 3.525530 | 2.06e-4 | 3.524682 | **+4.13** | −1464 ± 72 |

- **Registered check (bg 0): passes.** All of a, U_iso and scale are within 2 posterior σ
  (worst |z| 1.79).
- **The same check fails on bg 2, so the pass is not a property of the method on this data.**
  The cause is not isolated (DIAGNOSIS).
- **Widths:** σ(a) ≈ 2.1e-4 Å, ≈ 8e-4 Å after √(N/N_eff). No absolute claim is made.
- **Ranking:** WAIC and LOO agree to four digits. The best raw z is 3.22 for bg 2 over bg 0,
  but after the registered √(N/N_eff) inflation it is **0.83**, so nothing is decisive. A raw
  z > 3 on correlated G(r) points is exactly the false certainty the registration anticipated.
- **SVI cost:** 453–786 s per model.
- **NUTS** failed twice on package defects before sampling (§10 item 13). The third attempt,
  with both workarounds and the model unchanged, ran for 59 min, past the registered 30-min
  budget, which the user waived.
  - **Result:** a = 3.524680 ± 1.20e-4 Å, and every parameter within 0.06σ of the MAP.
  - **Width:** σ(a) is within 3 % of the Hessian σ × √χ²_ν.
  - **Mixing:** r̂ ≤ 1.004 for a, U_iso and scale, but 1.037 (n_eff 50) for the background. One
    chain, no divergences (`$ANALYSIS/out/12a_nuts_sep11_none_map.json`).
- **So the weak link is SVI, not the Laplace width.** SVI's bg-0 mean sits 3.1 NUTS-σ from the
  NUTS mean, with a posterior 1.7× wider; its registered pass leaned on that width.

### /verify outcome — REFUTED (physics, statistics, artifact lenses; reproduction lens SURVIVES)

**Statistics — the simplest and most decisive attack.** No seed exists anywhere in the SVI/NUTS
call path. Rerunning the exact bg-0 pipeline twice more, no code changed: $z_a$ went
$-1.79 \to +1.68 \to -2.23$ — sign flip, and the third run alone crosses the registered
$|z| \leq 2$ line into refute. The three-run spread ($\approx 2.1$) is the size of the pass/fail
bar itself. A single unseeded stochastic-optimizer run cannot support a confirm/refute call.

**Physics.** The priors on a/U_iso/scale are centred at the MAP by construction
(`map_init = map_vals`) with widths 2–4 orders of magnitude broader than the Hessian-implied
likelihood at the MAP, even after the registered √χ²_ν rescale (still ~170× broader for a).
"Posterior mean within 2σ of the MAP" then follows from Bayes' rule for almost any
non-divergent sampler — it is not a validation of SVI or NUTS. The one model that failed (bg 2
under SVI) failed because its three background coefficients are nearly collinear (pairwise
correlation −0.94 to +0.86, checked by re-running `refine_structure(bg_order=2)` directly), a
ridge the mean-field `AutoNormal` guide cannot represent — a generic SVI weakness, not a
physical signal. Confirmed directly: rerunning bg-2's SVI at the registered 2000 steps with a
different RNG draw gave z_a = +2.65 (registered: +4.13); at 8000 steps, z_a = −1.44 — sign flip,
"failure" becomes "pass." NUTS ran a single chain (pyro's default, never overridden), seeded at the MAP;
its r̂ is a split-single-chain diagnostic that cannot detect a chain that never left the basin
it was dropped into, and the one parameter it does flag as marginal (bg_0, r̂ 1.037, n_eff 50)
is exactly the direction expected to mix slowest.

**Artifact.** The SVI/NUTS likelihood σ (`sigma_G · sqrt(chi2_nu)`, `12a_bayes.py`) omits
the √(N/N_eff) correction this same project's own small-box recipe applies to the identical
parameter (`06a_smallbox.py`, verified to reproduce H7's 4.49e-4 Å exactly). Re-scoring on that
consistent footing: SVI bg-0 z_a −1.79 → −0.46; NUTS bg-0 z_a +0.05 → +0.01; **SVI bg-2 z_a
(the one reported failure) 4.13 → 1.08 — also passes.** On the project's own convention,
consistently applied, there is no case in the record where this check is capable of failing.

**Reproduction (complete, SURVIVES).** The MAP reproduces bit-for-bit; both NUTS bug tracebacks
reproduce exactly; pyro's `num_chains=1` default and split-R-hat diagnostic, and the WAIC/LOO
formulas, were confirmed directly from source (real, standard, not fabricated); an independent
85-minute NUTS re-run landed within 0.02–0.07σ of the registered one. Its own from-scratch SVI
re-run gave $z_a$ = +0.38 — a fourth distinct value across four unseeded runs of identical code
(registered −1.79; statistics-lens reruns +1.68, −2.23; this one +0.38). Nothing this lens
checked was fabricated, so its verdict is SURVIVES on its own terms — it independently
re-confirms, rather than contradicts, why the other two lenses call the claim REFUTED.

**What survives:** NUTS agreeing with the MAP to 0.06σ, with a width matching the rescaled
Hessian to 3%, is a legitimate sanity check that the sampler runs correctly on this problem. It
is not evidence the model itself is right, and the registered "confirm" criterion cannot tell
the difference.
