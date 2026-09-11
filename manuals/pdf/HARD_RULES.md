# Hard rules — total scattering and PDF

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).
> Written 2026-09-10 from a from-scratch re-run of one real beamtime. Every rule here was
> written after something silently produced a wrong answer, or would have; none is a style
> preference. `$ANALYSIS/...` paths are provenance, defined at the end of the spine.

1. **Apply the intensity corrections yourself — the variance integrators do not.**
   `integrate_hard_with_variance` and the polygon/subpixel variants apply no polarization or
   solid-angle correction and never read `spec.PolarizationCorrection` /
   `spec.SolidAngleCorrection`; only `integrate_with_corrections` does
   (`packages/midas_integrate_v2/midas_integrate_v2/corrections/integrated.py:117-135`). So
   `midas_pdf.image_to_iq`, `integrate_to_Gr_with_variance` and `midas-integrate-v2-pdf` all
   produce uncorrected profiles, whatever `packages/midas_pdf/midas_pdf/frontend.py:1-9` says.
   Build the per-pixel correction and pass it as `correction=` (phase-5 §5.1). On a 470 mm
   Varex at 67 keV the factor reaches ×2.3 at Q = 20 Å⁻¹ and ×4.8 at 26. Without it an IPA
   G(r) carries a spike of ≈ +410 at r ≈ 0.12 Å, which is the artifact in the earlier report
   on this beamtime (Lab Notebook §1, §4).

2. **Carry the geometry across whole: `RhoD` and the residual map.**
   `AutoCalibrationResult.to_integration_spec()`
   (`packages/midas_calibrate_v2/midas_calibrate_v2/pipelines/auto.py:197`) sets neither.
   - **`RhoD`.** Without it, refined distortion is evaluated at ρ = R_µm / 1 µm
     (`packages/midas_calibrate_v2/midas_calibrate_v2/forward/geometry.py:217-225`): median
     |ΔR| was **7.4e33 px** on the reference frame. Reconstruct it the way `calibrate()` did
     (`auto.py:629-641`): the corner distance from the **seed** beam centre, through
     `midas_distortion.rhod.resolve_rho_d_um`.
   - **The residual map.** The map `calibrate()` writes (`residual_corr.bin`) reaches
     integration only if you set `spec.ResidualCorrectionMap` by hand. On the accepted geometry
     it took ring-crest RMS from **264.6 to 50.3 µε**.

   `$ANALYSIS/out/cal_edge/verify.json` (`rhod_trap`), `$ANALYSIS/out/cal_sep11_none/verify{,_map}.json`.

3. **Resample onto a uniform Q grid before the transform.** `fourier_sine_transform` takes
   dQ = q[1] − q[0] (`packages/midas_integrate_v2/midas_integrate_v2/pdf.py:169`). An R-binned
   profile is non-uniform in Q, and the transform returns a plausible, wrong G(r) with no
   warning.

4. **Verify the Q axis out to the Q_max you transform to, and build the check's ring list that
   far.** A powder fit's residual says nothing about rings it did not measure.
   - **The ring list truncates silently.** `_ring_table(..., n_rings=N)` stops at the N-th
     unique ring (`packages/midas_calibrate_v2/midas_calibrate_v2/seed/auto_seed.py:155`). With
     N = 120 the CeO2 crest check at 67.4 keV stopped at **Q ≈ 17.2 Å⁻¹** while the transform ran
     to 21, and every crest statistic still looked complete.
   - **The outer rings may not be measurable at all.** On that frame, rings beyond R ≈ 1300 px
     (Q ≈ 13) had median E-step SNR below 1 at every η bin from 5° to 20°.

   The Q scale between the last measured ring and Q_max is then **unmeasured**, and is reported
   that way (Lab Notebook §3).

5. **Normalise to the ion chambers, not to the attenuator PV.** `Attenuator/Position` read 0 on
   all five files. The ion chambers downstream (net C/IC1) put Ni and CeO2 at **×0.098** of
   air. A normalisation that trusted the PV would have been 10× off on exactly the samples
   that matter (`$ANALYSIS/out/int_edge/meta.json`).

6. **A propagated σ is a model until an azimuthal sliver test calibrates it.**
   - **The test.** Use two disjoint sets of interleaved 1° slivers and compute z = Δ/σ. The
     robust std of z must be ≈ 1. A planted ×4 variance must read 2 (it did: 1.97).
   - **Measured on air, Kapton and IPA.** `poisson` (√S on ADU) was 15–19× too small, in-bin
     azimuthal scatter 2.8–3.0×, and a measured photon-transfer model 3.5×
     (`$ANALYSIS/out/03c_sigma_edge.json`).
   - **Everything downstream inherits it:** G(r) bands, Hessian σ, Δ-PDF significance, CRLB,
     WAIC SE.
   - **Contiguous halves are no substitute.** Background structure made their robust std read
     44–364 (Lab Notebook §5).

7. **Scale every refined σ by √χ²_ν and √(N/N_eff) before quoting it.** `refine_structure`
   returns cov = 2 H⁻¹ of the weighted χ²
   (`packages/midas_pdf/midas_pdf/structure.py:340-346`). That is right only if the weights
   are right and the r points independent, and here they are neither. G(r) points are
   Fourier-correlated: N_eff = (r_max − r_min)·Q_max/π is 90 for r 1.5–15 Å at Q_max 21,
   against 1351 points.

   For σ(a) on Ni:

   | stage | σ(a) |
   |---|---|
   | raw Hessian | 1.7e-6 Å |
   | × √χ²_ν × √(N/N_eff) | 4.5e-4 Å |
   | fixed recipe (+ calibration, chain and choice terms) | 3.9e-3 Å |

   `$ANALYSIS/out/06a_smallbox_sep11_none_map.json`. An uncertainty smaller than the r-grid
   step is a tell.

8. **The model and the data must share the window.** `refine_structure` / `pdffit_gr`
   (`structure.py:141`, `:264`) model no Q_max termination, no Lorch window and no refined
   Qdamp. Fit Lorch-windowed data with them and the window's broadening goes into U_iso:
   - Ni U_iso: 0.0093 Å² with Lorch, 0.0051 without (**+84 %**).
   - CeO2 a moved **+497 ppm** between the two.

   Quote U_iso only from unwindowed data, or label it window-inflated. Carry the window choice
   in the uncertainty (H7's σ_choice).

9. **A multiphase weight is not a phase fraction, and has no σ.** `multi_phase_gr` sums
   w_i/Σw · G_i, where each G_i already carries its own scale s_i
   (`packages/midas_pdf/midas_pdf/multi_phase.py:83-101`). Only the product w_i·s_i is
   identifiable, and the refiner's weight uncertainties start as NaN (`:228`). Read the
   amplitude fraction w_i s_i / Σ w_j s_j, fit a decoy phase known to be absent, and look at
   the raw pattern first (rule 10).

   On the Ni G(r), a freely refined CeO2 decoy never looked like CeO2 and still took 3–7 % off
   χ²:
   - API: a 5.35–5.64 Å, one start with a negative scale.
   - CLI: a 5.97 Å, U_iso 2.23 Å².

   With its structure held at the calibrant, its amplitude was 0.3 ± 0.4 %
   (`$ANALYSIS/out/10b_decoy_coreshell_*.json`, `13a_cli_roundtrip_*.json`). A decoy must keep
   the decoy's structure, or it is only extra parameters.

10. **Look for a phase in the raw I(Q) before fitting it.** A two-phase fit always uses its
    extra freedom. For NiO in the Ni sample, the test had four parts:
    - a matched filter on I(Q) at NiO(111);
    - a null measured from the same statistic elsewhere in the pattern;
    - a planted 5σ Gaussian that must come back at z ≈ 5;
    - the parent phase's own peak as a positive control.

    Result: z = **−0.8** at 2.605 Å⁻¹, the plant came back at 5.1, Ni(111) at 3419
    (`$ANALYSIS/out/10a_nio_raw_sep11_none_map.json`). Only a phase that is present gets a
    model.

11. **One detector frame cannot see e11, e12 or e13, and `recover_strain` still returns
    numbers for them.** Every probe direction lies, to first order, in the detector plane.
    - **`strain_crlb`** finds Fisher rank 3, then marks **every** Voigt component undetermined:
      the null space leaks a few per cent into e22 and e33 and ~1e-6 into e23, and the cut is
      a projection > 1e-6 (`packages/midas_pdf/midas_pdf/strain_pdf.py:205-206`).
    - **`recover_strain`** (pinv steps, no uncertainty, `:250`) returned **e11 = +50 534 µε**
      on an unloaded powder.
    - **What to read instead** is the Fisher eigen-directions. The three determinable ones
      came back at −140 ± 16, −25 ± 6 and −20 ± 5 µε, with σ from the empirical wedge scatter,
      not inflated (`$ANALYSIS/out/09a_strain_pdf_*.json`, `09b_strain_inplane_*.json`).

12. **RMC needs its species and its scale set by hand.**
    - **Species.** `Supercell.from_crystal` falls back to labelling every atom `"X"` when the
      crystal tensor has no `atomic_symbols` (`packages/midas_pdf/midas_pdf/rmc/supercell.py:93-97`).
      Swap moves and written CIFs are then wrong.
    - **Scale.** The RMC χ² has no scale term, so divide the target by a small-box scale first.
    - **Weighting.** The forward model is an unweighted pair sum, so with more than one
      element it is not an X-ray G(r).

13. **Do not reduce a PDF with `midas-integrate-v2-pdf`.** Its `--mask` is not applied on the
    G(r) path, `--compton` and `--absorption-mu-R` are parsed and never used, `.h5` frames are
    not readable, and S(Q) is formed with ⟨f²⟩ = 1
    (`packages/midas_integrate_v2/midas_integrate_v2/cli.py:678-843`). Use the Python chain
    (phase-5).

14. **Do not tune a correction to the check it is supposed to pass.**
    `detector_efficiency` needs a sensor thickness, has no default, and models a flat sensor
    normal to the beam (`packages/midas_pdf/midas_pdf/corrections.py:76-95`). On the reference
    data it was the only lever that moved H4 by an order of magnitude: at CsI 200 µm, IPA
    ⟨S⟩ error went 1.38 → 0.09 (`$ANALYSIS/out/pdf_sep11_none_map_q21_lorch_deteff200/`). Choosing
    the thickness that makes S(Q) flat fits the correction to the test. Take the thickness from
    the detector's data sheet, or report the arm as a sensitivity.

15. **Sample facts are inputs, and the low-r slope tests them.** Below the first peak, G(r)
    must follow −4πρ₀r. With bulk density assumed for Ni dispersed in carbon black, the low-r
    check stayed ≥ 0.30 in every arm, including the one that fixed the liquids. State ρ₀,
    packing and composition as assumptions, and do not adjust them to pass.

16. **Agreement with a second reduction is consistency, not accuracy — and check that with a
    negative control before trusting it as even that.** MIDAS and GSAS-II agreed on the Ni G(r)
    at Pearson **0.9974** with first-peak Δr **−0.0009 Å**
    (`$ANALYSIS/out/05e_h5_pdf_sep11_none_map_q20.5_lorch_eta90w61.json`). The same MIDAS
    S(Q) failed the physical checks (H4). Both reductions used the same frame, the same
    Faber-Ziman normalisation, and — check this explicitly — the same wavelength: MIDAS's
    calibrated λ (0.183908 Å) equals the nominal K-edge value to 15 significant figures, and
    GSAS-II's `.imctrl` fixes the identical constant with `varyList {'wave': False}`. Neither
    side independently fitted it.

    **This agreement was verified and REFUTED.** Run the identical comparison against MIDAS
    reductions of the same frame using known-wrong settings, already on disk: a deliberately
    wrong Compton exponent scored Pearson 0.9973 — indistinguishable from the registered arm's
    0.9974. So did a detector-efficiency thickness wrong by 3×, and the full detector instead of
    the matched azimuthal wedge. Only removing the Lorch window (an acknowledged settings
    mismatch, not a physics error) failed the test, at 0.9364. **A Pearson-r-on-the-full-curve
    test between two G(r) of the same crystal, both anchored to the same Q-calibration, is
    dominated by shared crystallography and cannot tell a correct correction chain from a wrong
    one.** Before reporting "agreement" as evidence a reduction is right, run the same statistic
    against at least one reduction of the identical frame using a setting already known to be
    wrong; if the known-wrong arm also passes, the test has no power and the passing number is
    not evidence.

17. **An uncertainty of exactly 0.0 is a degenerate Hessian, not a measurement.**
    `refine_core_shell` returned σ = 0.0 for five of its eight parameters on the Ni G(r), and
    the `midas-pdf-coreshell` CLI for six. In both, a shell carried U_iso of 0.43–25 Å², i.e. no
    structure at all, and a volume fraction of 0.36–0.58 on a sample whose raw I(Q) shows no
    NiO.

    Two related traps in the same tools:
    - `--pin-geometry` changes nothing: the output is identical with and without it
      (`packages/midas_pdf/midas_pdf/cli/coreshell_cmd.py:59` parses it, `:103` only echoes it).
    - A CIF's `_atom_site_U_iso_or_equiv` is dropped on read; U_iso 0.0063 came back as B 0.0
      (`packages/midas_pdf/midas_pdf/cif.py:262-270`).

    `$ANALYSIS/out/10b_decoy_coreshell_*.json`, `$ANALYSIS/out/13a_cli_roundtrip_*.json`.

18. **A Bayesian posterior seeded and centred at its own MAP, with a likelihood far sharper
    than the prior, cannot fail a "near the MAP" check — verify that before reporting it.**
    `bayesian_refine_svi`/`_nuts`'s prior means default to `map_init` (`bayesian_refine.py:92-103`),
    and the Hessian-implied likelihood width at the MAP was 2–4 orders of magnitude tighter
    than the prior widths even after the registered √χ²_ν rescale. "Posterior mean within 2σ of
    the MAP" then follows from Bayes' rule almost regardless of whether SVI or NUTS ran
    correctly. On the wheel-1 Ni fit, the one model variant that DID fail this check (a
    quadratic background under SVI) failed for a generic, physics-independent reason: its three
    background coefficients are nearly collinear (pairwise correlation up to 0.98), a ridge the
    mean-field `AutoNormal` guide cannot represent — not a signal about the data.

    **Also check the likelihood σ against this project's own √(N/N_eff) convention before
    trusting a z-score built from it.** `12a_bayes.py` set the SVI/NUTS likelihood σ to
    `sigma_G · sqrt(chi2_nu)`, omitting the same √(N/N_eff) correlated-points correction this
    project's own small-box uncertainty recipe applies to the identical parameter (hard rule 7).
    Applying it consistently took the one reported failure (z = 4.13) to z = 1.08 — passing —
    which means the check could not have failed anywhere in the record on a consistent
    footing.

    **NUTS defaults to one chain** (`pyro.infer.MCMC`, never overridden here), seeded at the
    MAP; its reported r̂ is a split-single-chain diagnostic, blind to a chain that never leaves
    the basin it was dropped into. Run ≥ 2 independently-initialised chains before trusting r̂
    as evidence of real mixing, and check that priors are wide relative to the *rescaled*
    likelihood — not the raw one — before reading anything from where the posterior mean lands.

    **Neither `bayesian_refine_svi` nor `bayesian_refine_nuts` seeds the RNG anywhere.**
    Re-running the identical SVI pipeline on the wheel-1 Ni fit three times, with no code
    change, gave z_a = −1.79, +1.68, −2.23 — sign and confirm/refute outcome both flipped
    between runs, with a run-to-run spread the same size as the pass/fail bar. A single
    unseeded SVI run is not a result; rerun at least twice before reading anything from where a
    variational posterior landed.

19. **Energy and geometry rules are inherited, not restated.**
    - λ comes from the tabulated K edge of the monochromator foil, never from a fit
      (`manuals/calibrate-integrate/HARD_RULES.md` rule 9).
    - A ring list is crystallography, not a measurement (its rule 13).
    - No correction is applied before the geometry is verified against the raw rings (its
      spine §6).

---
