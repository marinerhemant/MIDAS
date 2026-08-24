# Hard rules — calibrate → integrate

> Part of the **calibrate-integrate doc set**. Spine: [`README.md`](README.md).
> Split out of the spine 2026-08-21. Every rule here was written after something
> silently produced a wrong answer; none is a style preference.

## §8. Hard rules

1. **`SubPixelLevel` stays at 1.** Above 1 the CUDA integrator truncates the
   fractional sub-pixel coordinate and reads the neighbouring pixel — measured
   24.3× on in-band bins, and `IntegratorZarrOMP` is unaffected because it
   interpolates. `0` is bit-identical to `1`; write `1`.
   (`FF_HEDM/src/IntegratorFitPeaksGPUStream.cu:916`, Lab Notebook §2.)

2. **Never seed the calibration from an existing parameter block.** `make_seed`
   works from the image. A prior answer's errors are inherited silently.

3. **Remove the sentinels before calibrating, and do not assume they are
   negative.** `-1` / `-2` (Pilatus) and `2**32-1` (EIGER, and any unsigned
   dtype-max) are gaps and bad pixels, not counts. `img[img < 0] = 0` catches
   only the first kind and fails **open** on the second, which is the more
   dangerous direction: the fitter gets 4.29e9 instead of a small negative.
   Use `read_image(..., return_mask=True)`, which handles both and returns the
   mask. Verified across the 1-ID archive: GE uint16 frames carry **zero**
   pixels at 65535, so this costs nothing on the detectors that never had the
   problem. (Lab Notebook §12.)

4. **Fix δLsd to 0 and move the modules.** A module is misplaced *in the
   detector plane*, giving a constant ΔR; a per-panel δLsd gives ΔR ∝ R. Fitting
   the first with the second rails it — 16 of 48 panels in one run. Bound the
   in-plane shift at ~2 px and refine that. (Lab Notebook §5.)

5. **Measure before enabling `GradientCorrection`.** It is off everywhere by
   default. On the reference data the cardinal bands were *quieter* than a
   non-cardinal control, i.e. there was no aliasing to correct. Test with a
   control band before switching it on. (Lab Notebook §4.)

6. **Match rings by position, not by rank.** (Lab Notebook §3.)

7. **Turn on the expansion gauge for a tiled detector.** `fix_panel_id` and
   Σ panel = 0 remove the *translation* nullspace. They do not touch a second
   one: pushing every module outward in proportion to its radius shifts ring
   radii exactly the way an `Lsd` error does, so the fit trades freely between
   them. Measured: 11 % of the fitted panel field sat in that mode, ~73 % of it
   absorbable into `Lsd`. Without the gauge, panels rail — 9 of 48 in one run.
   `add_panel_no_expansion_constraint(spec)`.

8. **A powder ring cannot determine a module's tangential shift.** With η spread
   much below 90° on a module the 2 × 2 Fisher block is rank-1: only the radial
   component is identifiable. Do not report the tangential part as a measurement.

9. **λ is NOT determined by a single-distance powder pattern.** Wavelength and
   `Lsd` are degenerate: a ring at radius R constrains only the ratio, so a
   wrong λ is absorbed into `Lsd` and **the strain gate still passes**. A 1 %
   energy error becomes a 1 % distance error, silently, with a calibration that
   looks clean.

   So take λ from the beamline (monochromator/undulator), never from the fit,
   and cross-check it against the filename and the metadata.

   **"From the beamline" is more specific than it sounds.** At 1-ID the
   monochromator is tuned to an absorption **K edge** and left there, so the
   number to use is the *tabulated edge energy of the foil element* — read the
   element from `~/new_data/<expt>/fastsweep_Emon*.txt` and look the edge up in
   MIDAS's own table, `midas_pdf/midas_pdf/data/fluor_edges.json`. Measured over
   116 beamtimes (Lab Notebook §13): 74 of the 82 with a logged energy sit
   within 0.3 % of a foil K edge; the monochromator readback in
   `fastpar_*.par` field 10 runs a median **0.040 % below** the tabulated edge;
   and `exp_setup.yml`'s `EDGE:` key is **stale** — where it disagrees with the
   Emon element (9 of 18 beamtimes) the Emon element is right 7 times and the
   yml value never. Some beamtimes are deliberately off-edge at a round setting
   (95, 100 keV); there the readback is all there is, and it carries ±0.1 %.

   To break the degeneracy with a measurement rather than a claim about the
   monochromator, you need **several exactly-known distances** —
   `midas-calibrate-v2 --lsd-offsets`, which refines one shared `L0` plus known
   offsets and a shared λ. Measured on a planted 1 % error: the two hypotheses
   differed by 0.063 vs 0.083 px RMS, i.e. barely distinguishable.

   **Do not try to recover λ by calibrating at candidate energies and taking the
   lowest residual. It cannot work, and it produces a convincing wrong answer.**
   The only thing that breaks the degeneracy is the `tan`/`asin` nonlinearity,
   and to first order that is

       ln[R(λ')/R(λ₀)] = c₀ + c₂ρ² + c₄ρ⁴ + O(ρ⁶)

   which is **exactly the span of `{Lsd, iso_R2, iso_R4, iso_R6}`**. So with
   `refine_distortion=True` — the default — the fit absorbs the entire signal
   into the radial distortion block. Measured for a −5.94 % energy error
   (Lab Notebook §15):

   | free parameters | residual left |
   |---|---|
   | `Lsd` only | **555 µε** — trivially detectable |
   | `Lsd` + `iso_R2` | 0.70 µε |
   | `Lsd` + full radial block (the default) | **8.8e−06 µε** |

   against an empirical noise floor of **11.3 µε**. The default configuration
   makes this rule *stricter* than it reads: not "weakly broken", but **not
   broken at all**. A 30-unit control that appeared to recover the energy 83 %
   of the time was matched exactly by a constant, data-blind guess.

   **What does work: use the degeneracy instead of fighting it.** `Lsd` tracks
   the assumed energy at log-log slope **1.0066 ± 0.0037**, so an independently
   recorded distance *is* an energy estimator. Picking the candidate whose
   fitted `Lsd` matches a distance written in the filename recovered the true
   energy in **17 of 17** units, using no residual at all.

10. **`FixPanelID` is a gauge choice, not a measurement.** Panel shifts from two
   calibrations with different anchors are not directly comparable.

11. **Refine only the distortion the azimuth supports.** Every `a_k`/`phi_k` pair
   is a k-fold azimuthal harmonic and needs azimuth to be identifiable. Over a
   narrow wedge they are degenerate with the beam centre (1-fold) and the tilts
   (2-fold), so they rail at their bounds and the E↔M loop stops converging.
   Measured on a 4-panel detector whose beam centre lies off the corner, giving
   66–73° of each ring: the shipped calibration had **3, 4, 7 and 7 of its 15
   coefficients pinned at ±0.002**, and a refit with them free oscillated between
   84 and 4692 µε across iterations.

   Use `refine_distortion="radial"` — or an explicit list — instead of the
   all-or-nothing boolean
   (`packages/midas_calibrate_v2/midas_calibrate_v2/forward/distortion.py:49`).
   On that frame even `"radial"` was not enough and `"none"` was required: 181 µε
   diverging → 72 µε. **Check, do not assume:** run the azimuth gate
   (`.../pipelines/diagnostics.py:281`), refine the largest block that passes,
   and confirm the loop settles.

   A second calibrant does **not** help here. Both powders illuminate the same
   wedge, so multi-phase adds rows to the Jacobian, not a new direction.

12. **Set `RhoD` to the outer ring radius, in µm.** The distortion polynomial
   lives in `ρ = R_µm / RhoD`, so `RhoD` is a normalisation, not a measurement —
   but it sets the dynamic range of every radial term. Left far beyond the
   outermost ring, ρ stays small and the high powers collapse: at ρ_max = 0.32,
   ρ⁶ is 1e-03 and `iso_R4` / `iso_R6` came back with 1σ of 0.9 to 15 on
   coefficients of order 1e-03, railed at their bounds. `calibrate()` derives a
   sane value; a *template* may not. Gate: `.../pipelines/diagnostics.py:401`.

13. **A ring table is crystallography, not a measurement of this exposure.**
   Weak, vignetted or grainy rings still produce a centroid per η bin, and those
   centroids are noise the geometry absorbs. Filter rings on what the frame
   actually carries — `MinEtaBinsPerRing` / `MinRingSNR`
   (`.../pipelines/_common.py:127`) — and note the count is absolute, so it
   scales with `EtaBinSize`: a fully-covered ring carried 13 fits at 5° bins and
   ~36 at 2° on the same frame. Read the distribution off `ring_quality()` rather
   than copying a threshold.

---
