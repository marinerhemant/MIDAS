# Phase 2 - spots, integration, corrections, run scaling, merge, and reading the floor

Sources: `packages/midas_sx/midas_sx/{pipeline,integrate,reduce,workflow,merge}.py`, `midas_hkls.corrections`, `validation/PREREGISTER.md` STEP 14 (A36) and STEP 17, `validation/real/PREREG_a39_floor.md`, `validation/results/a39_floor/CLAIM_floor.md`.

## 1. Spots

`pipeline.find_spots(raw, geom, threshold=None, noise_sigmas=7.0, n_sectors=8, low_count_threshold=20.0, min_vol=10)` is the `midas_defect` front end: mask, polar background, 3-D blobs. It returns `(spots, mask, info)`. `spots` has `frame, row, col, integrated, volume_vox, n_frames, peak_counts, omega_width_frames, ...`. Pass `found=(spots, mask, info)` to `reduce_scan` to skip the 20 minutes of spot finding when you already have them; cache `spots` and `mask` yourself.

Pilatus: `threshold=3.0, low_count_threshold=0.0` (background ~0 counts; the default collapses).

## 2. reduce_scan, the options that matter

```python
red = reduce_scan(raw, crystal, U0, setup,
                  refine_tilts=True, refine_wedge=True,            # real scans (phase 1)
                  background=None,                                   # "mean" (default), "quadratic" (A36: INCONCLUSIVE, default stays "mean")
                  margin=None,                                       # override the swept support margin
                  robust_merge=True,                                 # within-run merge (default)
                  integrate_predicted=False, predicted_d_min=None,   # opt-in, NO GAIN on the L-alanine floor
                  refine_wavelength=False,                           # opt-in, PROVISIONAL (phase 1 section 6)
                  found=None, find_kwargs=None, reduce_kwargs=None, allow_unconverged=False)
```

It returns a `Reduction` with `obs` (one row per integrated reflection: `h k l frame row col omega omega_meas tth eta L PT I sigI bg n_box F2 sigF2 blob_sum hkl_residual`), `rejected`, `counts`, `merged`, `r_int`, `U`, `geom` (the REFINED geometry), `orientation`, `profile`, `spots`, `mask`, `info`, `targets`. `reduce_kwargs` go to `reduce.reduce_spots` (e.g. `hkl_tol`, `crowd_px`, `min_tth_deg`, `eval_at`, `max_factor_variation`).

### Why every blob is accepted or rejected, and how to read `counts`

`reduce.REJECT_REASONS` = `low_angle, no_hkl, no_crossing, frames_edge, detector_edge, masked_pixels, no_background, near_axis, outside_opening, factor_variation, crowded`; `red.counts` has one entry per reason plus `input` and `kept`, and `input == kept + sum(reasons)` is asserted. On the L-alanine runs 70-80 % of blobs end in `no_hkl` (weak noise, ring-like scatter, blobs no lattice explains) and 370-490 per run in `crowded` (`validation/results/a39/out_p0/run_0N.log`, `counts`).

* **`hkl_tol` (default 0.15)** is the largest distance from an integer hkl for a blob to be assigned. A weak noise blob inside it is integrated AS that reflection: reflection (1 0 2) of L-alanine was two such blobs (I = -3 +- 284 where ~72000 counts were expected); tightening to 0.05 improved R_int by about 1.4 points on that data and left the default unchanged. [`validation/real/PREREG_a39_floor.md` Amendment 1; `README.md`]
* **`crowded`** (`crowd_px` 12): another spot within 12 px and the same frames. It removes the strongest, highest-Lorentz reflections (24 of the 41 strongest blobs that the base diagnostic left unmatched on run 02); both blob-driven and predicted-position modes do it. It is conservative and untouched.
* **The reduction is blob-driven**: the box is centred on the measured blob, the corrections on the CALCULATED crossing. A reflection with no blob is never integrated; `integrate_predicted=True` integrates every predicted crossing (adds ~6 % unique reflections on L-alanine; NO GAIN on the systematic floor, `packages/midas_sx/README.md`).

## 3. Integration support and the margin sweep

`integrate.integrate_box(raw, centre, half, mask, shell=(2, 3), background)`: a fixed box minus a shell background estimated OUTSIDE it from raw counts with a high-outlier-clipped mean (median + 4.5 sqrt(median + 1)); the variance is honest Poisson, `var(I) = sum(raw in box) + n_box^2 bg / n_shell`. `build_profile` measures the extents (frames `ext_f0 + ext_f1 L`, pixels `ext_p`) and `choose_margin` sweeps an integer margin over `(0, 1, 2, 3, 4, 6)` and takes the smallest on the plateau (`plateau=0.995`).

* **The sweep may end without a plateau, and then it warns** (`RuntimeWarning: build_profile ... still growing with the support margin`). On L-alanine the median I(m)/I(m=6) was 0.76 / 0.81 / 0.86 / 0.90 / 0.93 / 1.00 for m = 0..6 on every run, and the largest margin (6) is used. Margin 1 gave R_int 25.4 %, margin 10 gave 15.4 % (margin 6: 15.67 %), and the floor f was 12.2 / 14.2 / 15.8 % at margins 1 / 6 / 10: larger boxes did not lower the floor. [`PREREG_a39_floor.md` Amendment 1; STEP 17 READING] Treat the warning as "the integration is not converged in margin", report the margin used, and run the margins 1 / 6 / 10 once; if they disagree on the thing you are quoting, say so.
* Background: `"mean"` assumes a flat background; a curved one biases the blended rows, and the error grows with box volume (A36: the quadratic estimator removes the bias and costs sigma; registered reading INCONCLUSIVE, default stays `"mean"`).

## 4. Corrections

At the calculated Ewald crossing, from `midas_hkls.corrections`: `rotation_lorentz` (NaN within `min_abs` of the rotation axis), `polarization_factor(tth, eta, pol_fraction, pol_plane_eta_deg)`, `slab_self_absorption`/`diamond_anvil_transmission` for the DAC. Never re-implement them here. Inputs YOU supply and that default silently:

* **`PolarizationModel(fraction=0.99, plane_eta_deg=90.0)`**: the plane is 0 or 90 deg in the MIDAS frame and the default is 90. Rule used on L-alanine, registered before the data: reduce with both and take the plane with the lower merged R_int (plane 0: 15.67 %; plane 90: 24.53 %); report both.
* **`DacModel`** defaults to NO absorption (`mu_per_cm=0.0`, no opening cone). For a DAC scan pass `t_plus_mm`, `t_minus_mm`, `mu_per_cm`, `half_opening_deg`. The absorption ablation was below the preregistered detection threshold in the simulations: absorption is UNTESTED.
* Evaluating L at the measured spot instead of the calculated crossing gave a median ratio of 1.000 (robust sd of the log ratio 0.001) on L-alanine and left the floor by L unchanged: not a cause. [`PREREG_a39_floor.md` Amendment 1]

## 5. Several runs: scale, then merge

```python
from midas_sx.workflow import merge_runs
mr = merge_runs({"r02": red02, "r03": red03, "r04": red04}, crystal.space_group)       # scale="smooth", robust=True, n_knots=8
mr.merged, mr.obs, mr.scales, mr.r_int
```

`merge_runs(runs, space_group, scale="smooth"|"group"|None, robust=True, group="run", n_knots=8, **scale_kwargs)`; `runs` is a mapping of `Reduction` (or anything with an `obs` table) or a DataFrame with the group column.

* **`scale_smooth`** (`merge.scale_smooth`): a piecewise-linear LOG scale per run over `knot="frame"`, `n_knots=8`, fitted from symmetry-equivalent observations only (log domain, weights `1/((sigma/F2)^2 + rel_floor^2)` with `rel_floor=0.1` so the strong reflections do not dominate, gauge: weighted mean of log k = 0). With fewer than `min_obs_per_knot` (20) shared observations per knot it uses fewer knots, down to one (a plain run scale); with no equivalents it is the identity. `fit_mask` restricts what CONSTRAINS the fit (cross-validation).
* **`scale_groups`** is the older one-scale-per-run; keep it for old numbers.
* **`merge_observations(robust=True)`**: error model `sigma'^2 = sigma^2 + (g <F2>)^2`, `g` estimated from the data (`estimate_error_model_g`; 0.073 on the smooth-scaled L-alanine data, 0.182 on the run-scaled), and worst-first leave-one-out rejection at `reject_sigma=4.0`, never with n = 2 (you cannot tell which of two is bad). The plain default (`robust=False`) is the `1/sigma^2` weighted mean, biased low and dominated by a failed integration.
* **Placebo.** The smooth scale lowers R_int by construction when it is flexible. Before believing it, permute the frame labels within each run and fit again: on L-alanine the placebo left R_int at ~15.3 % (default merge) while the real frame order took it to 6.8 % (smooth scale + robust merge). With the robust merge the permuted-frame fit is 14.2 %, because the robust merge alone lowers R_int to 14.4 %: both steps contribute (R_F2 against the independent reference: smooth + default 9.31 %, run scale + robust 11.33 %, both 6.33 %). Cross-validate when you can: fit the scale on a random half of the unique reflections and evaluate the other half (L-alanine: 15.67 -> 6.88 %).
* **What the scale is.** Anything smooth in the frame index that multiplies a whole run: beam or sample-volume drift, crystal centring, the illuminated volume. On L-alanine it swung by +-20 % over 80-160 deg and differed between the three runs (fitted log range 0.35-0.58 within a run; `PREREG_a39_floor.md` Amendment 1). Its cause was NOT established. It cannot be told apart from a real intensity variation with the same frame dependence, and it needs equivalents spread over each run's whole range. Record I0 (or a transmitted-beam / monitor value) per frame so the scale is measured, not inferred.

## 6. The metrics, and what each can and cannot tell you

* **R_int** (`merge.r_int(obs, sg, merge_kwargs=...)`): `sum|F - <F>| / sum F` over reflections observed at least twice, unweighted. Pass `merge_kwargs=dict(robust=True)` to measure against the robust mean. It is an in-sample consistency, flattered by any fitted scale.
* **R_split** (`validation/real/a39_combined.py`): merge runs {A} and {B} separately and compare on the common unique reflections, `sum|Fa - Fb| / sum (Fa + Fb)/2`. L-alanine {02, 03} vs {04}: 21.4 % -> 10.1 %.
* **The systematic floor** (`merge.systematic_floor(obs, sg, by=None, min_obs=30)`): the leave-one-out log deviation of equivalents, robust sd, minus the counting prediction (sigma/F2)^2 (1 + 1/(n - 1)), over sqrt 2. It separates "noise the counting statistics explain" from "noise they do not". L-alanine: 14.2 % against 2.7 % counting before the fixes, about 5.8 % after; it rose with the Lorentz factor (10.8 % at L < 1.5, 23.5 % at L > 5), at low angle, and on the run that touched the count cutoff. **If f is several times the counting prediction, more exposure will NOT help**; look at the dependence (L, resolution, run, frame) and at the drift first.
* **A model-free reference readout** (only when an independent measurement of the same compound exists): see `phase-3-refine.md` section 5. Fix its definition before reading it.

## 7. The advice module (PROVISIONAL, unvalidated at the time of writing)

`midas_sx.advice.assess(runs, crystal, merged_runs=None, acquisition=Acquisition(...), checks=None)` collects ten registered checks (coverage: `scan_range`, `completeness_redundancy`, `next_setting`, `resolution_limit`; quality: `systematic_floor`, `intensity_drift`, `dynamic_range`, `omega_sampling`, `geometry_consistency`, `contamination`) into an `AdviceReport` (`.text()`, `.to_dict()`, `.ok`). Each finding states the number, the rule (`advice.THRESHOLDS`), the evidence, the action for the beamline, and what it does not cover. It has unit tests, a validation on injected defects in simulated frames and a run on L-alanine; the /verify of that validation returned PROVISIONAL (4 of 4 lenses UNCERTAIN) and two checks (`dynamic_range` saturation, `omega_sampling`) were repaired afterwards and are not re-verified. Use it to look, quote its numbers only after reproducing them. See `ADVICE.md` and `ENVELOPE.md`.
