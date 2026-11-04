# midas-sx

Single-crystal structure-factor reduction and refinement for MIDAS: rotation-series frames (monochromatic beam, area detector, optionally in a
diamond-anvil cell) to corrected, merged F² with honest uncertainties, a SHELX `.hkl`, and a weighted least-squares structure refinement.

**Status: alpha. Most claims in the table below are validated on SIMULATED data** (an independent forward simulator in `validation/`, never imported by
the package). Three real data sets have been tried (section "Real frames"). Only the L-alanine set (Diamond I19-1, Pilatus, three runs) passed a four-lens `/verify` for the multi-run merge and for the non-hydrogen positions, as a LOCAL refinement at the noise floor; the other two are INCONCLUSIVE and none of them validates the method in general. The status words are the ones the adversarial `/verify` procedure gave them; `validation/DESIGN.md` and
`validation/PREREGISTER.md` carry every result, every amendment and what each test does not show. PROVISIONAL means "not (yet) established",
not "probably true".

## Quick start

```python
from midas_hkls import Crystal
from midas_sx.workflow import ScanSetup, reduce_scan, refine_structure
from midas_sx.reduce import DacModel

setup = ScanSetup(wavelength_A=0.4133, dist_mm=200.0, pixel_mm=0.1, centre_rc=(1023.5, 1023.5),
                  omega_start=-35.0, omega_step=0.25,                    # START EDGE of frame 0
                  dac=DacModel(axis0=(1, 0, 0), t_plus_mm=2.5, t_minus_mm=2.5, mu_per_cm=0.9, half_opening_deg=35.0))
red = reduce_scan(frames, crystal, U0, setup)       # frames: (n_frames, rows, cols); crystal: the START model; U0: a rough orientation
red.merged                                          # unique reflections: h k l n F2 sigF2 ...
fit = refine_structure(crystal, red.merged)         # model_options={"aniso_sites": ("La1",), "refine_extinction": True, "wavelength_A": 0.4133}
fit.table(); fit.converged, fit.goof, fit.at_bound, fit.null_params
```

`reduce_scan` raises `OrientationNotConverged` when the orientation/geometry refinement fails its gates (with `allow_unconverged=True` it continues with a warning);
`refine_structure` warns when the refinement did not converge or has GooF > 3. Read `converged`, `at_bound`, `null_params` and `goof` before
quoting any parameter. For a crystal twinned by the 90° rotation about c use `reduce_twin90_scan` and `refine_twin90_structure` (it warns when the pair-model orientation falls back to the centroid one, and when twin.m sits on 0.5: domain 1 must be the majority). The default reflection pool (`d_min`) is the inscribed detector edge, the one the validated runs used; `ScanSetup` defaults to NO anvil absorption (`DacModel().mu_per_cm = 0`).

## Real frames (PREREGISTER STEPS 15 to 17 and A39-FLOOR have every number; the first two attempts are 2026-10-21)

- **Rochelle salt, Australian Synchrotron MX1 (ADSC Q210, 360 x 1 deg phi, CBF), INCONCLUSIVE:** the reader, the convention scan and the blind lattice finder ran; the lattice that 50 % of the strong spots share is not any published Rochelle cell and does not refine (3.5 px / 6 frames). Untested causes: the Q210's module distortion (no distortion model here), a crystal off the axis, a different crystal.
- **Thiourea 130 K, Bruker D8 VENTURE PHOTON-III, Mo (Zenodo 18910433, five omega runs of a kappa stage), INCONCLUSIVE overall:** blind lattice 98.9 % of the strongest spots (cell +1.0 % on the short axis; the scale follows the distance, which was chosen with the vendor's reflection list), orientation refinement converged in all five runs (0.34-0.40 px, 0.29-0.32 frames); merged F² against the CIF and against the vendor's SAINT / SADABS reduction agree to 7.5 % / 7.3 % on the 46 merged reflections with I/sigma > 10 (scale and an unphysical B = -0.48 fitted; 23.6 % on all 239; two lenses of /verify read it PROVISIONAL); BUT R_int is 22 % (55 % of the unique reflections seen twice), and the refined non-H positions are 0.07 A (median) from the published ones with pulls of 3.6-6.5 sigma: systematic error (no extinction or absorption model, 239 unique reflections, hydrogens fixed). The phi-scan run of that set is not usable (rotation about a tilted axis).
- **L-alanine 100 K, Diamond I19-1 (Pilatus 2M, 0.689 A, swung detector, three omega runs; Zenodo 11946282; reference COD 1574527), PREREGISTER STEP 17 and A39-FLOOR:**
  the registered blind-indexing tier is REFUTED as written (the 95 % bar ignored that the first 100 frames hold only about 280 reflections: the geometry was not the limit), the merge tiers at the
  baseline processing are INCONCLUSIVE (R_int 15.7 %). The campaign that followed found a missing frame-dependent scale and a merge without outlier rejection; with `workflow.merge_runs` defaults
  R_int is 6.8 % and the model-free R_F2 against the independent reflection file embedded in the COD CIF is 6.3 % (was 14.2 %): ESTABLISHED by a four-lens `/verify` for this crystal and these three runs
  (`validation/results/a39_floor/CLAIM_floor.md`). The non-hydrogen positions refine to a median 0.0014 A (max 0.0031 A) from COD, R1 3.0 %: ESTABLISHED as a LOCAL refinement at the noise floor;
  the hand and origin come from the start and the fixed hydrogens, the displacement parameters are not validated (`results/a39_floor/CLAIM_tierC_cfgA.md`). Rotation-axis wedge (0.3-0.4 deg), detector
  tilts and a joint wavelength/cell scale had to be refined for the geometry to hold.
- The geometry of a real scan is NOT in its header: the Rochelle header's beam centre was 4 px off, the Bruker header lists two distances, and the image orientation and sense of rotation come from the indexing score (`midas_sx.conventions`; the four mirror-image conventions tie exactly and are equivalent for |F|²). Read `io_frames.read_cbf_scan` / `read_sfrm_scan` for what each header states, and expect to scan the geometry.

## Pipeline

frames -> spots (`midas_defect` front end: mask, polar background, 3-D blobs; `pipeline.py`) -> orientation and geometry refinement from the
measured spots (`orient.py`: joint least squares of U, beam centre and distance, hkl re-assigned as it converges) -> hkl assignment ->
fixed-support integration with a measured-extent support (`integrate.py`) -> corrections at the CALCULATED Ewald crossing
(`midas_hkls.corrections`: rotation Lorentz, partially polarized beam, diamond-anvil absorption and opening cone) (`reduce.py`) -> symmetry
merge and SHELX `.hkl` (`merge.py`) -> refinement (`refine.py`) with symmetry-derived constraints and a gated convergence verdict.
Physics lives in `midas_hkls`; this package holds the workflow only.

## What is implemented, and how far it is verified

| Feature | Where | Status (simulated data) |
|---|---|---|
| Lorentz, polarization, anvil absorption | `midas_hkls.corrections` | ESTABLISHED as an independent-implementation cross-check against the simulator's Monte-Carlo (mu, the flat-slab/cone model and the horizontal polarization plane are shared inputs) |
| Reduction to F² | `reduce.py`, `integrate.py` | PROVISIONAL (R_F2 about 1 % on Cu-like and 1.7 % on the S2 bilayer-oxide-like simulated frames; absorption ablation below detection) |
| Coverage forecast (Fisher) | `validation/forecast_c4.py` | PROVISIONAL, narrowed after /verify: matches the chain\'s formal sigma to 0.88-0.92 on ONE orientation (calibrated); 8-9 of 10 parameters determined at 1e5 counts is a MEDIAN over 8 orientations (2-9 each) |
| Orientation + beam centre + distance | `orient.py` | PROVISIONAL (U median 0.008 deg from a 0.3 deg / 0.5 px / 1 % start; ONE crystal, ideal case; a 0.5 % cell error already costs 0.19 deg) |
| Effect of the orientation error on the structure | validation A32 | with the TRUE cell: median shift of the eight determined parameters 0.15 / 0.30 sigma at 0.01 / 0.02 deg; 0.05 deg is the onset of breakdown (0.87 sigma over 20 seeds); it enters through the Lorentz factor only; a cell error of 0.5-1 % in the reduction gives 1-3 sigma (REFUTED as first worded, restated) |
| Primary extinction (exact perfect-crystal Laue factor) | `midas_hkls.laue_extinction_factor` | ESTABLISHED (the old tanh factor understated it by about π²) |
| Refining the block thickness t_b | `refine_extinction=True` | PROVISIONAL (t_b recovered to 1 % at 5 and 10 µm; at 10 µm the upper bound is hit) |
| Anisotropic U on chosen sites | `aniso_sites=` | ESTABLISHED at the orientation it was verified on (isotropic refinement of anisotropic data is biased by 4.7 to 11.7 sigma on Ni.z, La2.z); a second typical orientation is INCONCLUSIVE (coordinate pulls of -0.6 sigma remain), a hard one (c* 13 deg from the beam) is UNINFORMATIVE |
| Exactly overlapping 90° twin, twin fraction | `refine_twin=True` | ESTABLISHED, narrow scope: one signal pair, one orientation, one split size; detection 68 % at m = 0.1; the majority domain is an input; a second typical orientation: INCONCLUSIVE (m recovered to 0.004, two coordinate pulls just above 0.5 sigma) |
| Extinction + anisotropy + twin at once | `refine_extinction`, `aniso_sites`, `refine_twin` | PROVISIONAL (simulated, one orientation; m and t_b recovered with every effect needed; t_b is an EFFECTIVE thickness; see below) |
| Two resolved domains, one scale each | `n_domains=2` | PROVISIONAL |
| Domain finding from scratch | `midas_defect.domains.find_domains` | PROVISIONAL (found both domains in 50/50 seeds; the cell it returns follows the detector-distance error; a wrong declared cell is recovered up to an effective 4 %; the declared CENTRING must be right: a primitive lattice declared for an I-centred crystal finds both domains in 2 of 25 seeds, and 6 of 25 return a wrong list silently; any space group with the right centring works after the `allowed_multiple_parity` fix) |
| Pseudo-merohedral twin, split spots (a != b) | `predict.py`, `twin_orient.py`, `refine_twin90_structure` | PROVISIONAL, with failed registered criteria (below) |
| Studentized outlier rejection | `outlier_sigma=` | PROVISIONAL |
| Likelihood-ratio bound test (per parameter) | `refine(bound_test=True)` | restated claim fixed after /verify found a pooled test; not re-verified (sampler skipped it) |
| Polar-axis origin pin, fixed atoms | `RefineModel(fix_origin=True, fixed_atoms=...)` | unit-tested; on the thiourea data the pinned refinement stays within 0.14 A (unpinned: 0.28 A rigid drift) |
| CBF / Bruker SFRM readers, convention scan, `setting.match_setting`, run scales | `io_frames`, `conventions`, `setting`, `merge.scale_groups` | unit-tested; used on the two real sets above |
| Multi-run scaling + merge: smooth per-run scale over the frame index, robust merge | `workflow.merge_runs` (defaults), `merge.scale_smooth`, `merge_observations(robust=True)` | ESTABLISHED on ONE real data set (L-alanine, Diamond I19-1 Pilatus, 3 runs, /verify 4 lenses 2026-10-23): R_int 15.7 -> 6.8 % (6.9 % with the scale fitted on half of the reflections), model-free R_F2 against an independent reference reflection file 14.2 -> 6.3 %; both steps contribute; the d < 0.8 A shell does not improve; the cause of the +-20 % scale variation is not established; wording in `validation/results/a39_floor/CLAIM_floor.md` |
| Wavelength / cell-scale refinement | `index_and_refine(refine_wavelength=True)` | PROVISIONAL: recovers injected eps in simulation, but on real data the formal sigma is 20-40x smaller than the run-to-run spread (about 5e-4); with the cell fixed it is NOT an energy calibration |
| Friedel-pair estimate of tx and the rotation-axis wedge | `friedel_geometry` | INCONCLUSIVE on real data (agrees with the orientation fit within 0.035 deg, registered rule met in 1 of 3 runs); works in simulation, needs pairs inside a run (scans of 180 deg or more give them all) |
| Integration of every predicted crossing (not only found blobs) | `reduce_scan(integrate_predicted=True)` | NO GAIN on the systematic floor (L-alanine); +6 % unique reflections; opt-in |
| Data-adequacy advice (what is missing in the data, with the action for the beamline) | `midas_sx.advice.assess`, `Acquisition`, `advice.THRESHOLDS` (checks: scan_range, completeness_redundancy, next_setting, resolution_limit, systematic_floor, intensity_drift, dynamic_range, omega_sampling, geometry_consistency, contamination) | PROVISIONAL (claim a0c60827daf1: 4 of 4 /verify lenses UNCERTAIN, every number reproduced). On simulated frames with one injected defect, intensity_drift, scan_range, completeness/redundancy, the wedge note, the contamination ring and systematic_floor discriminate the defect from clean frames; `dynamic_range` (saturation) and `omega_sampling` did NOT as first written; they were repaired and re-verified (claim a56630731be7: PROVISIONAL, 3 lenses SURVIVES, physics UNCERTAIN: the clipping test is blind above about 1 % background and the single-frame bar is specific to the simulator's spot width). 0 of 12 fresh clean scenes raised a blocking/limiting finding (6 independent). Thresholds cite L-alanine numbers; the 14 of 14 real-data match is a regression check. Wording and gains untested. `manuals/sx/ADVICE.md`, `validation/real/PREREG_advice.md` |

Known failures and caveats you must know about:

- **A smooth multiplicative drift is the first thing to check on real data.** On the L-alanine runs the systematic floor in F2 was 14 % against 2.7 % from counting
  statistics until a per-run smooth scale over the frame index and a robust merge were added (`workflow.merge_runs`; the single-run merge in `reduce_scan` is
  robust by default, `robust_merge=False` restores the plain mean). Margin, count-rate and the place where the Lorentz factor is evaluated were tested and are NOT the cause there.
  The fitted scale swings by +-20 % over 80-160 deg and differs between runs; its physical cause (crystal centring or illumination is the guess) is untested. Record I0 per frame if you can.
- **`hkl_tol`:** a tighter assignment tolerance (0.05 instead of 0.15) improves R_int by about 1.4 points on the blob-driven path of the L-alanine data (poor assignments of weak noise blobs); the default is unchanged.
- **Blind indexing of the strongest N blobs cannot exceed the number of reflections the window contains** (L-alanine, first 100 frames: about 270-290 predicted reflections for 600 blobs, so at most about 48 % can index); a low indexed fraction is not a calibration failure by itself.
- **Split spots (a != b, b/a = 1.011 and 1.03):** the registered read of the first test was REFUTED on convergence; after the fix the read is
  INCONCLUSIVE, not "fixed". At b/a = 1.011 La2.U and Ni.U are biased (+1.4 and -0.9 sigma) and the orientation of the first domain reaches 0.0055
  deg; at b/a = 1.03 it is 0.0208 deg, short of the 0.0143 deg that was registered. The fitted weight of the second component is a nuisance
  weight (0.06 to 0.08 against an intensity fraction of 0.30), NOT a twin fraction. The twin law and the cell (a, b, c) are inputs.
- **Background curvature (simulated, PROVISIONAL):** the integration subtracts the MEAN of a thin shell around the box, which assumes a flat background. A smooth curved background makes the shell mean exceed the mean inside the box by the curvature, and the error grows with the box volume (about half-width cubed): in the b/a = 1.011 twin simulation blended-pair rows lose 43 counts (median) at 5415-voxel boxes, against 2 counts for an untwinned crystal, which biased La2.U by +1.7 sigma and the twin fraction to 0.24 for 0.30. A plane fit, a median or a gap do not remove it; a quadratic fit to the shell does, and it is in the package as `background="quadratic"` (`reduce_scan(..., background=)`, `integrate_box`): on background-only simulated frames it leaves a mean of -1 count against -57 for the shell mean (`validation/results/a36t1tail_*`). It is NOT the default: the registered tests read T1 INCONCLUSIVE (median and slope met, the intercept criterion missed by signal tails in the shell, which any shell estimator has), T2 mixed (blend-row z -0.150 -> -0.004 and the P11 La2.U pull +1.13 -> -0.28, but sigma of the twin fraction x 1.42), T3 INCONCLUSIVE (formal sigmas x 1.17 on the untwinned set); the bias is removed at +17 to +42 % formal uncertainty (PREREGISTER STEP 14). A bright neighbour whose tail lies in the shell hurts the quadratic fit somewhat more than the mean. Whether a real background is curved enough to matter is untested.
- **Cell and geometry are inputs:** an error of 0.5-1 % in the cell used for the reduction shifts the structure parameters by 1-3 sigma (it enters the Lorentz factor at the Ewald crossing); the orientation error alone is much less harmful (see the table).
- **t_b is an effective thickness:** with different block thicknesses in the two twin domains, or a thickness distribution, the refined t_b is 10-12 sigma high while GooF is unchanged; the twin fraction is robust (<= 1 sigma). t_b converges only from a start below about 20 um (the failure is flagged, not silent).
- **The twin law swaps x and y:** the in-plane displacement parameters Uxx and Uyy of a cation are nearly degenerate (correlation -0.91 at m = 0.3, exactly degenerate at m = 0.5); they touch their bound in about a third of the fits. `refine_twin90_structure(gate_free=...)` accepts them; tie Uxx = Uyy yourself if the site symmetry allows.
- **c* close to the beam:** an orientation with c* 13 deg from the beam gave 25-37 unique reflections instead of 54 to 99; in the simulated refinements 0 to 1 parameters were determined, the anisotropic fit converged in 0 of 25 seeds and the twin fit in 5 of 25, always with an undetermined U on its bound; the chain flagged this and returned no confident wrong value, but it does not say WHY (read n_unique and the sigma table). A platelet loaded along c is the typical DAC case.
- **Counts:** about 1e5 integrated counts on the strongest reflection are needed to determine 8 to 9 of the 10 parameters of the S2 bilayer-oxide-like
  test structure; at 1e4 nearly nothing is determined (forecast, PROVISIONAL, formal sigma).
- **Misspecification map (C3; the first write-up was REFUTED by /verify, the paired version is PROVISIONAL):** with U, cell and the hkl assignment held at truth, a missing Lorentz correction fails the refinement, a missing polarization correction shifts Ni.z by about 0.02 A, a missing absorption correction shifts O2.z by about -0.018 A, a polarization fraction of 0.50 shifts six parameters, a U error of 0.1 / 0.3 / 0.6 deg about (1,1,1) shifts 3 / 6 / 6 of the 10 parameters (`validation/results/c3_misspecification_map_paired.txt`; one noise realisation; entries near threshold are not robust to the noise draw; "anvil -20 %" and "mu x0.80" are the same operation). Errors in omega zero-point, beam centre and distance that do not change which spots are assigned have NO effect in this reduction BY CONSTRUCTION (corrections are evaluated at the crossing calculated from the assigned hkl and U); the open-loop effect of a wrong start geometry is what the orientation refinement and its A19 runs measure.
- `midas_hkls.lorentz_polarization` is the full-ring average and ignores the polarization fraction; per-spot work uses
  `midas_hkls.corrections.polarization_factor` (this package does).

## Not implemented

A default background model other than the shell mean (the quadratic shell fit is available but not the default: T1 INCONCLUSIVE, T2 mixed, T3 INCONCLUSIVE); rotation about a tilted axis (a phi scan of a kappa stage); a detector distortion or panel-gap model; secondary extinction and non-block dynamical effects; anomalous dispersion in the merge and refinement; refining the cell (it is held fixed in the
orientation refinement because it trades off against the distance); anisotropy on O sites and positive-definiteness constraints on U; occupancies;
an unknown twin law or an unknown/wrong b/a; refining U2 independently of U1; more than two components; joint refinement of orientation and
structure (orientation is refined first and then held fixed; the effect of the orientation precision reached, 0.01 to 0.02 deg, on the
structure parameters has not been measured directly, the misspecification map starts at 0.1 deg); a command-line interface; real-data validation beyond the three sets above (one crystal verified).

## Layout

`midas_sx/` the package (`workflow.py` is the front door); `tests/` unit tests (use the simulator in `validation/` for small scenes);
`validation/` the independent simulator, every preregistered run script, results and the synthesis page (read `DESIGN.md` first).
