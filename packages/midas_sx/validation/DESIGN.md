# sx_refinement: design and state (2026-10-15)

Thread: single-crystal F²_obs reduction + structure refinement for mono-beam DAC rotation data
(request from a 16-ID-B user; no real data yet). Target package: `midas-sx` (name reserved on PyPI by HS,
nothing published, nothing scaffolded yet). Judged against `PREREGISTER.md` (read it first, including A1-A3).

## Where things live (updated 2026-10-15, later)

| piece | place | status |
|---|---|---|
| per-spot corrections | `packages/midas_hkls/midas_hkls/corrections.py` | done, 46 tests, uncommitted |
| Lp ring-average fix | `midas_hkls/intensity.py` | done, uncommitted; claim `7054b7947ba7` PROVISIONAL |
| NF Lp imports `midas_hkls.corrections` (dep `midas-hkls>=0.17.0`) | `midas_nf_fitorientation` | done, uncommitted; hkls 0.17.0 must release first |
| package | `packages/midas_sx` (0.0.1, name reserved on PyPI by HS, NOT published) | geometry, assign, integrate, reduce, merge/.hkl, pipeline: done, 26 tests |
| simulator + preregistration + this note | `packages/midas_sx/validation/` (moved from `midas_hkls/dev/sx_refinement`, which is gitignored) | tracked-eligible |
| torch LS refiner on F2 (`midas_sx/refine.py`) | `midas_sx` | done, 12 tests; C2 and C4 run (below) |
| orientation + geometry refinement (`midas_sx/orient.py`) | `midas_sx` | done, 10 tests; A19 run on 50 seeds (below) |
| predict-and-integrate (weak reflections) | `midas_sx` | NOT built: A12, the forecast says the weak reflections cut sigma by only 6-33 % |
| release.sh | `midas_sx/release.sh` (adapted from midas_saxs, `bash -n` clean, not run) | done; README still minimal |

Release order: midas-hkls 0.17.0 -> midas-nf-fitorientation -> midas-sx. Message sent to the release chat (midas-75).

## Simulator (`simulate.py`) in one paragraph

gemmi |F|² truth (two traps encoded, PREREGISTER A2); Monte-Carlo mosaic blocks crossing a finite Ewald
shell, so the Lorentz factor is the shell-crossing time and has no closed form in the simulator; Thomson-vector
polarization; explicit ray/slab lengths through two diamond anvils that rotate with the sample; hard opening cone;
random-depth crystal-plate absorption (off by default); detector PSF; Poisson noise per seed on a cached signal
volume. Detector convention of the simulator: `col = cx - y/px`, `row = cz - z/px`; ω right-handed about +z; frame k
covers `[omega_start + k*step, +step)`. 512x512 px, 0.3 mm, 100 mm, λ = 0.4133 Å (30 keV), ±40° in 0.25° steps; about
0.5-0.6 GB resident (Mac limit is < 3 GB/process).

## Self-test results (`results/simulator_selftest_2026-10-15.txt`, bandwidth_rel = 1e-3)

* Emergent (Monte-Carlo) Lorentz / closed-form `rotation_lorentz`: S1 n=39 median 1.00040, robust spread 0.29 %,
  range 0.9956..1.0085; S2 n=516 median 0.99962, spread 0.36 %, range 0.9886..1.0118. The simulator and the closed
  form agree with no shared code. (The spread is the MC noise floor of A3, ≥ 5e4 hits per spot there.)
* P·T from the simulator (Thomson + ray tracing) vs `corrections`: S1, spots > 1.5° inside the cone, max
  |ratio-1| = 1.3e-5 (untilted) and 1.5e-5 (tilted DAC, asymmetric anvils); S2 tilted 3.5e-4.
* Positive control: flipping the ω sign in the DAC axis moves P·T by up to ±3 % (S1 tilted) and blocks 23 of 39
  spots (untilted geometry), so the ω-sign convention IS detectable by this comparison.

## Things the self-test taught us (design rules for the reduction)

1. **Rocking width matters, not only the nominal ray.** At `bandwidth_rel = 3e-3` a low-angle spot spans ~0.9° of ω
   and its tail is clipped by the opening cone ~1° before the nominal edge: ratio sim/corrections for one spot went
   0.745 → 0.997 → 0.99999 as the shell narrowed 3e-3 → 1e-3 → 3e-4. Physics, not a bug.
2. **High-L spots are unreliable at the factor level.** (2,3,9) has L = 42.7 (median 2.98); its shell crossing spans
   ~2.4° of ω (median spot 0.17°) and its diffracted direction sweeps through the cone edge during the crossing.
   Rule: the reduction must reject a spot when the correction factors, evaluated at the ends of ITS OWN ω-profile,
   differ by more than a stated tolerance, and count what it rejected. A fixed angular margin is not enough.
3. Spots touching the first/last frame are incomplete (`complete` in the truth table); reject them.
4. gemmi traps: attach the space group with `setup_cell_images()`; call `change_occupancies_to_crystallographic()`.

## Not yet done / open

* Reduction must reproduce a positive control for each ablation (no L, no P, no A, wrong ω sign), per PREREGISTER.
* Intensity protocol: fixed-support integration on the background-subtracted raw array vs `find_blobs_3d` label sums
  (KL-106 arm). Both read the same simulated frames through the REAL `midas_defect` front end.
* Indexing arm: first runs hand the reduction the true U; a later arm uses `midas_defect.indexing`.
* Extinction cannot be tested here (kinematic simulator, KL-028).
* Other duplicated Lp/polarization code, not touched: `midas_pink/inverse.py` (own torch Lp, mathematically the
  correct mixture), `midas_integrate_v2/corrections/intensity.py` (torch, differentiable). Deduplicating needs a torch
  backend in `corrections`.

## C1 / C5 results through the real front end (2026-10-15; `results/c1_*.txt`; `run_c1.py`)

All PROVISIONAL: synthetic, one noise draw per run, not /verify'd. Claim logged (see skill-log).

| run (results file) | unique refl. | R_F2 FULL | slope FULL (A^2), regression SE | R_F2 N1 | R_int FULL / blob |
|---|---|---|---|---|---|
| S1 orient 1, noise 11 (`c1_S1_Cu_100000_n11_o1_E30_t2.5.txt`) | 10 | 0.0122 | -0.0150 +- 0.054 | 0.4621 | 0.0114 / 0.0084 |
| S1 orient 2, noise 12 (`..._n12_o2_...`) | 10 | 0.0122 | +0.0085 +- 0.041 | 0.2140 | 0.0125 / 0.0146 |
| S1 orient 3, noise 13 (`..._n13_o3_...`) | 11 | 0.0064 | +0.0004 +- 0.017 | 0.3478 | 0.0074 / 0.0089 |
| S1 20 keV, 3.5 mm anvils, noise 14 (`..._n14_o4_E20_t3.5.txt`) | 4 | 0.0028 | -0.0861 +- 0.032 | NaN (too few reflections to score N1/L) | 0.0047 / 0.0047 |
| S2, noise 21 (`c1_S2_bilayer_like_..._n21_o1_...`) | 52 | 0.0168 | -0.0006 +- 0.048 | 0.6206 | 0.0159 / 0.0178 |

All five regenerated 2026-10-15 after the scoring fix below (every arm is scored on FULL's reflection set). The first S1 orient-1 run, before
the fix, had used lambda = 0.4133 A where the regenerated one uses 12.3984/30 = 0.41328 A; that 5e-5 change reshuffles the Monte-Carlo
shell sampling and moved the FULL slope from -0.0043 to -0.0150 (R_F2 0.0113 -> 0.0122): a direct look at the ~0.01-0.04 slope scatter.
Never quote a slope without its SE.

* C1 numeric criteria (R_F2 <= 5 %, |slope| <= 0.2, FULL beats N1 by >= 2x) met in 4 of 4 scorable runs (N1/FULL = 37.9, 17.5, 54.1, n/a, 36.9).
* Sensitivity: L and P ablations detected everywhere scorable. **T (absorption) ablation is NOT detected in any run** (scored on FULL's
  set: R_F2 0.0223 / 0.0110 / 0.0183 / 0.0205 / 0.0227 vs FULL 0.0122 / 0.0122 / 0.0064 / 0.0028 / 0.0168; slopes within their SE). T
  spans only 0.55-0.61 over all spots at 30 keV / 2.5-3 mm anvils / 35 deg opening, a ~1 % effect on F2. Per the preregistration the T
  correction is UNTESTED here, and it is not a likely cause of a bad refinement at these settings. (The earlier 'ablate T DETECTED' in
  one seed was a scoring artefact, see Verification.)
* Wrong omega sign: fails at hkl assignment (0 of 37-171 blobs survive), the strongest possible detection.
* C5 (fixed support vs blob label sum): R_F2 favours fixed support in every run (1.2/1.2/0.6/0.28/1.7 % vs 3.6/2.5/1.4/0.35/4.5 %)
  and the blob-sum slope is large in the four runs with real signal (-0.44, -0.28, -0.18, -0.06 vs fixed support ~0). R_int favours fixed
  support in 3 of 5 (orient 2, orient 3, S2), blob in 1 (orient 1), equal in 1 (20 keV). By the letter of the registration (R_int must also be
  lower) C5 therefore holds in 3 of 5. R_int cannot see the KL-106 bias, because equivalents of equal amplitude share it.
* Completeness is the real limit: S2 has 516 true reflections and the blob finder yields 143 (28 %, threshold 7 sigma). The
  weak ones are exactly those that carry O positions and ADPs. Predict-and-integrate from the indexed orientation is needed
  before any refinement claim.
* Front end artefacts: with no beamstop in the simulator, the steep radial background at 2theta -> 0 yields 26-37 blobs near the
  beam centre; they assign to (0,0,0) with zero residual and are rejected by the `low_angle` reason. A real cell has a beamstop.
* Memory: peak 1.5 GB resident for the full chain on 512x512x320 frames; front end ~85 s.

## Still open before any refinement claim

* C2 (refinement pull test, 50 noise seeds), C3 (misspecified geometry map), C4 (coverage), per PREREGISTER.
* Absorption sensitivity needs a geometry where T varies by > 2 % across spots, or accept it as negligible here.
* Indexing arm: all runs above were given the TRUE orientation U.
* Orientation/geometry refinement from the found spots (real data have neither exactly).

## Verification of the C1 claim (verify, 2026-10-15): PROVISIONAL

Claim aae4c7af7e9b, four fresh Sonnet lenses, mechanical rule: no lens REFUTED, two UNCERTAIN, so PROVISIONAL.
Lens verdicts are in `~/.claude/skill-log.jsonl`; their workdirs are
`/Volumes/T5 EVO/verify_workdirs/Desktop/analysis/midas_sx_c1/refute_*`.

* physics SURVIVES: L and P agree with an independent finite-difference of the Ewald function (median 1.00000) and an algebraic
  derivation of the simulator's Thomson mixture; truth lookup by `asu_representative` is sound (gemmi F2 is constant inside every
  class, S1 83 classes, S2 545); the wrong-omega control gives 0 kept spots in every run. Absorption is nearly invisible here
  (T 0.55-0.61 over all spots; the back-reflection branch is never exercised: 0 spots in S1 and S2).
* reproduction SURVIVES: the stored S1 seed-12 run reproduces exactly (every scientific line to 4 decimals); an independent chain
  (own box sum, own L/P/T, no midas_sx) gets R_F2 1.4-2.9 % on the strongest 8-20 spots and 18x worse without corrections.
* statistics UNCERTAIN: **'|slope| < 0.01' is a point value, not a precision.** With 10-11 unique reflections the slope scatter
  between replicates is ~0.04 (three S1 slopes -0.0043, +0.0085, +0.0004; 95 % interval of the mean [-0.015, +0.018]). The
  preregistered bar was 0.2 A^2 and it is met; do not quote a tighter number.
* artifact UNCERTAIN: re-noising the same simulated signal gave S1 slope mean +0.002, sd 0.045 over 18 reruns (33 % under 0.01),
  and S2 slopes -0.0006, -0.008, -0.053, -0.029, -0.079 over 5 reruns (mean -0.034, SE ~0.014: **a small negative bias of order
  -0.03 A^2 in S2 cannot be excluded**). The reported -0.0006 is the best of the five.

Corrections made because of it:
1. **Ablations are now scored on FULL's reflection set** (`run_c1.py`): switching a correction off also switched off its
   `factor_variation` rejection, so an unrestricted ablation scored a different reflection set. The seed-12 'ablate T DETECTED'
   was this, not absorption.
2. **Slope is printed with its regression standard error**; no slope is to be quoted to better than the replicate scatter.
3. 'N1 is 19-54x worse' is an order-of-magnitude statement (N1 is scored on fewer reflections and the ratio's denominator is the
   noisy R_F2). Independent chain got ~18x.

Open gaps the lenses named (not yet fixed):
* The registered detector-flip control is not implemented (L and P are invariant to flipping eta or a pixel axis; only hkl
  assignment sees it).
* The test is closed loop: the reduction is handed the true orientation, cell, DAC axis/thicknesses/mu/cone and polarization
  fraction, and the simulator has a Gaussian profile, flat slabs, no extinction. The misspecification map (C3) is what
  measures sensitivity to that.
* `factor_variation` was designed after the simulator exposed a high-L outlier; removing it moves S1 seed-12 R_F2 from 1.2 % to
  2.0 %, so the S1 numbers include that one-spot rejection (S2 rejects 3).
* Support plateau k = 2 is probably optimistic for a detector with real profile tails (untested).
* 12 of 39 S1 true spots and 373 of 516 S2 true spots are not found by the blob finder; S2 recovers 52 of 138 unique reflections
  that cross the scan.


## Refinement (C2, C4), 2026-10-15: all PROVISIONAL, synthetic, closed-loop (true orientation and DAC model are handed to the reduction)

Files: `results/c2*_S*.csv` (per seed, per start), `results/c2*_analysis_*.txt`, `results/c4_*`, `results/diag_*.txt`, `results/c4_weak_reflection_value.txt`.
Amendments A4 to A13 in `PREREGISTER.md` record every decision, in order, with its reason. Nothing in C2/C4 was read before its reading rule was fixed.

### What the refiner is
`midas_sx/refine.py`: bounded trust-region least squares of F2_obs against k |F|^2 from `midas_hkls`, an autograd Jacobian, coordinate shifts
derived from each site's symmetry (so a site on a 4-fold axis gets one), one U_iso per site, formal sigma from the column-scaled (J^T J)^-1,
and a verdict that is gated: converged only if the optimiser succeeds, no parameter is on a bound, and the Jacobian has full rank.
Tests (`tests/test_refine.py`): exact data of the model's own F2 recovered to 1e-8 (S1 and S2); gemmi data to the cross-code floor
(midas_hkls vs gemmi differ by up to 9e-8 of the largest F2, which moves parameters by up to 4e-7); an 80-draw Monte-Carlo with calibrated pulls;
a bound hit and a rank-deficient case are flagged, with infinite sigma in the null space; the false-minimum finding below is a test.

### Three things found on the way (each changed the work, none was tuned away)
1. **A start 0.41 A from truth is on a chi2 ridge.** 0.02 FRACTIONAL on S2's 20.5 A axis is 0.41 A; along c the chi2 of La2.z (others at truth)
   peaks at about +-0.4 A (1.16e6 at -0.4 A) and falls beyond. Some sign patterns end in a genuine false minimum (restart with a second algorithm
   does not move; autograd Jacobian matches finite differences to 4e-10). C2 therefore runs from two starts: N3 as registered and N3p (0.10 A).
2. **The first reduction (v1) lost 5-17 % of high-Lorentz spots** (noise-free: L = 35 -> 0.835, 43 -> 0.862, 45 -> 0.911, 18 -> 0.939), because
   its support was a Gaussian k-sigma rule tuned on L < 8 while a high-L spot's omega profile is the flat-topped shell crossing. On noisy frames
   z = (I - I_exact)/sigma_I had sd 1.86 (4.2 for the 1e4-3e4-count class). v2 measures the extent (flat-top profile fit, 99.5 % of the fitted
   area, regressed on L over all strong spots up to L = 62, plateau margin): noise-free min ratio 0.981, median within 0.04 % of 1; noisy z sd
   1.15 / 1.07 / 1.16 in the three scored classes (limit 1.3). Pre-registered acceptance (A11) met; v1 results are kept below.
3. **The ensemble-mean offset of the pulls was mostly a defect in the REDUCTION, not simulator noise. (CORRECTED 2026-10-15: I first wrote the opposite.)**
   A noise-free run of the whole chain reproduces the 50-seed mean error of every parameter (O2.z 0.0124 vs 0.0116 +- 0.0013 A; Ni.z -0.0034 vs -0.0030 +- 0.0004 A;
   O1.U -1.05e-3 vs -1.06e-3 +- 0.18e-3 A^2), so the offset is deterministic. I then divided exact counts by the simulator's realised Monte-Carlo Lorentz factor and every
   error collapsed to 1e-5-1e-4, and concluded "simulator Monte-Carlo noise". **That oracle was confounded**: it replaced the reduction's Lorentz factor, which is evaluated
   at the spot's MEASURED centroid direction, with one at the nominal crossing, so it removed a reduction error together with the noise. The verify physics and artifact
   lenses both refuted the attribution and split it: the simulator's Lorentz noise (0.30 % per reflection, unbiased, Poisson) accounts for about 10-12 % of the offset,
   evaluating L at the measured centroid for about 60 %. My own test agrees: evaluating L, P, T at the CALCULATED Ewald crossing of the assigned reflection shrinks the
   noise-free errors 5-50x (Ni.z -3.35e-3 -> -5.9e-4 A, O2.z 1.24e-2 -> 2.4e-4 A, O1.U -1.05e-3 -> -1.4e-4 A^2) and the measured/calculated Lorentz factors differ by 0.57 % rms
   (omega by 0.043 deg rms). That is the v3 reduction (PREREGISTER A15, `assign.calculated_crossing`, `reduce_spots(eval_at="calculated")`, now the default).
   The A15 acceptance (every determined parameter within 0.1 formal sigma, noise-free) is NOT MET (worst Ni.z 0.247 sigma): I did not allow for the simulator's own 0.30 % floor (A16).
   `results/diag_chain_bias.txt` (current) and `results/diag_chain_bias_before_predicted_arm.txt` (the v2 numbers).

### C2 on S2 (bilayer-like, 50 noise seeds, 54 median unique reflections, 11 parameters)
| | v1 reduction | v2 reduction |
|---|---|---|
| converged from N3 (registered, 0.41 A on c) | 35 / 50 (70 %) | 36 / 50 (72 %) |
| converged from N3p (0.10 A) | 49 / 50 (98 %) | **50 / 50** |
| median GooF (N3p) | 1.87 | **1.06** |
| pull sd, determined parameters (N3p) | 0.83 - 2.30 | **0.82 - 1.23** (all inside 0.7-1.4) |
| largest \|pull mean\| (N3p) | 3.03 | 1.12 |
| coordinate bias, determined (N3p), A | La2.z -0.0018, Ni.z -0.0057, O1.z -0.0001, O2.z +0.0087 | La2.z -0.0005, Ni.z -0.0030, O1.z +0.0031, O2.z +0.0116 |

Preregistered readings (v2, N3p): **sd criterion MET; mean criterion (|mean pull| <= 0.3) NOT MET for all 8 determined parameters; coordinate bias <= 0.02 A MET;
"converges from the registered N3 start" NOT MET (72 %; the failures end on a U bound).** The not-met mean was NOT explained by simulator noise as I first claimed: finding 3 (corrected) shows it was mostly the reduction. 
the v2 mean criterion failure is therefore partly the reduction (corrections at the measured centroid), fixed in v3; see the v3 column once it is run.
Determined (median formal sigma below 0.02 A / 0.00127 A^2 = B 0.1 A^2): La2.z, Ni.z, O1.z, O2.z, La1.U, La2.U, Ni.U, O1.U. **Undetermined: O2.U (sigma 0.0022) and O3.U (0.0038).**
Formal sigmas (v2, N3p): La2.z 0.0009 A, Ni.z 0.0026, O1.z 0.0069, O2.z 0.0102.

### C2 on S1 (Cu-like, v2, 30 seeds - FEWER than the registered 50, because S1 has one free parameter and the front end dominates the cost)
30 / 30 converged, GooF median 0.91, U_iso bias -1.6e-4 A^2 (B -0.013 A^2), sigma 2.2e-4, pull sd 1.07 (MET), pull mean -0.62 +- 0.19 (NOT MET).

### C4 (coverage), forecast validated against C2 at the nominal geometry (A5, A10, A13)
The first forecast check FAILED (forecast sigma 0.58-0.64 of the measured, uniformly, because it used the v1 support; kept in `results/*_v1profile*`).
With the v2 support, the forecast is 0.88-0.92 of the C2 formal sigma for all ten parameters (within the 30 % rule) and predicts 54 unique reflections vs 54 measured;
it is about 10 % optimistic (internal-only merge sigma). It is therefore a usable planning tool for the FORMAL sigma. Measured RMS error was 1.2-1.8x the forecast sigma for the determined parameters.
Median over 8 orientations, strongest reflection integrated counts N, detectability cut calibrated on orientation 1 (1138 counts; the literal-A9 cut 2350 gives the same qualitative answer):
* **N = 1e5: 8-9 of 10 parameters determined at every geometry** (O3.U never; O2.U borderline). Unique reflections: 54 (opening 25 deg), 67 (30), 77 (35), 87 (40). La2.z sigma 0.00068 -> 0.00048 A over that range.
* **N = 1e4: at most 1 of 10 determined; the refinement is rank deficient for 25-100 % of orientations (13-18 unique reflections).** Counts matter far more than opening angle.
* **An omega range beyond the opening half-angle adds nothing** (the cell axis is near the beam, so the incident ray leaves the cone at |omega| = half-opening): W = 30/45/60 give identical unique counts for opening <= 30.
* **Predict-and-integrate (A6/A12) is not worth building on this evidence**: lowering the detectability cut to every reflection with any signal doubles the unique reflections (77 -> 150) but lowers the formal sigmas by only 6-33 %.

### What this does NOT show
Nothing here touches real data. The reduction is handed the true orientation, cell, DAC axis, anvil thicknesses, mu, opening cone and polarization; the simulator is kinematic (no extinction, KL-028),
Gaussian-profiled in-plane, flat-slab, one crystal, no twin, no diamond Bragg spikes or gasket rings. The C3 map (what a wrong DAC model or geometry does to the parameters) has NOT been run, and it is the experiment
that says how much of this survives a real, imperfectly known setup. The absorption correction remains UNTESTED (sub-threshold at 30 keV, 2.5-3 mm).


### C3: what a wrong assumption does, v2 reduction (`results/c3_misspecification_map_v2.txt`, `run_c3.py`; A14). SUPERSEDED for the geometry rows by the v3 map further down
Paired shifts of the refined parameters, wrong minus correct, on the SAME spots (S2, one noise realisation, one orientation). Baseline formal sigma: La2.z 0.0009 A,
Ni.z 0.0026, O1.z 0.0065, O2.z 0.0097. A shift "matters" if it exceeds min(formal sigma, tolerance 0.02 A / 0.00127 A^2).
* **Fatal:** no Lorentz correction (not converged, GooF 32).
* **Large:** no polarization correction (Ni.z +0.0197 A, La2.z +0.0026; 6 parameters matter); polarization fraction 0.50 instead of 0.99 (7 parameters).
* **Geometry:** a 1.0 px (0.3 mm) beam-centre error moves O1.z by -0.0183 A, O2.z -0.0097, Ni.z -0.0034 and several U (Ni.U +0.0016, O1.U +0.0072 A^2); 0.5 px already moves four U.
* **Absorption:** dropping it entirely moves O2.z by -0.0219 A (above tolerance) and Ni.U, O1.U, O2.U matter. Absorption is only ~1 % of F2 but the oxygen parameters live at that level.
* **Does NOT matter within the ranges tried:** anvil thickness +-20 %, mu +-25 %, DAC axis tilt absent or 2 deg wrong (only O1.U moves, ~0.001 A^2), polarization fraction 0.90, omega zero-point up to 0.3 deg, detector distance +0.5 %.
Caveat: one realisation, one orientation: this says how far an assumption MOVES the answer, not a precision on the movement.

### C1 re-run with the v2 support (`results/c1v2_*`)
S2: FULL R_F2 1.45 %, slope -0.025 +- 0.044, N1 0.626 (43x worse), C1 numeric criteria met. S1 (orient 2, noise 12): FULL R_F2 1.05 %, slope -0.003 +- 0.031; N1 and the L ablation
cannot be scored (only 2 scorable merged reflections), so the 2x criterion is not scorable for that seed. The earlier v1 C1 files (`results/c1_*`) are kept.

### What this means for the real question (a 16-ID-B user with bad refinements), as far as a simulator can say
* If the strongest reflection has about 1e5 integrated counts and the cell opens to 35 deg, a bilayer-like structure is refinable to heavy-atom coordinates of ~0.001-0.003 A, oxygen z of ~0.007-0.010 A and most U_iso; the oxygen U of the sites with the least scattering stays undetermined.
* At 1e4 counts on the strongest reflection nothing useful is determined (13-18 unique reflections, rank deficient). **Counts are the first thing to check.**
* The polarization correction (use the true beam polarization, ~0.99 for a storage ring) and the Lorentz factor are not optional; the beam centre must be good to well under a pixel; the DAC absorption model can be rough (+-20 %) but must be present for oxygen.
* An omega range beyond the opening half-angle adds nothing; the opening angle sets the number of reflections.
* All of this assumes a kinematic, single, untwinned crystal with a good starting model (within ~0.1 A; 0.4 A can land in a false minimum), which a real DAC dataset may not be.


## v3 (corrections at the calculated crossing, A15/A16), after the second verification: the current state

### Verification outcome (claim 3fbe7add7210, first C2 write-up): REFUTED as worded, numbers reproduce
Four fresh lenses: reproduction SURVIVES (an independent refiner with gemmi F2 and a finite-difference Jacobian matches the repo's per-seed parameter errors to 1.7e-7 A and the formal sigmas to 4e-7;
`diag_chain_bias.txt` reproduces byte for byte); statistics UNCERTAIN; physics and artifact REFUTED the sentence "the offset is caused by the simulator's Monte-Carlo Lorentz noise" (the oracle
was confounded; MC noise is ~10-12 % of the offset, evaluating L at the measured centroid ~60 %). Two REFUTED lenses kill the claim as worded (mechanical rule); I corrected the reduction (v3) and re-ran.
The statistics lens also found: all 13 seeds with La2.z displaced - and Ni.z displaced + fail from the registered start (0/13) while 36 of the other 37 converge, so the 72 % is the draw of start signs,
not a property of the data; and the 50 seeds share one signal volume and one orientation, so the standard error on a mean pull is conditional on that realisation.

### C2 on S2, 50 seeds, three reductions (N3p start unless stated; `results/c2*_analysis_*.txt`, `c2*_S2_*.csv`)
| | v1 | v2 | **v3** |
|---|---|---|---|
| converged, N3p (0.10 A) | 49 / 50 | 50 / 50 | **50 / 50** |
| converged, N3 (registered, 0.41 A on c) | 35 / 50 | 36 / 50 | 35 / 50 |
| median GooF | 1.87 | 1.06 | **0.95** |
| pull sd, 8 determined | 0.83 - 2.30 | 0.82 - 1.23 | **0.77 - 1.17** |
| \|pull mean\|, 8 determined | up to 3.03 | 0.46 - 1.12 | **<= 0.28** |
| coordinate bias, A (La2.z / Ni.z / O1.z / O2.z) | -0.0018 / -0.0057 / -0.0001 / +0.0087 | -0.0005 / -0.0030 / +0.0031 / +0.0116 | **-0.0001 / -0.0006 / -0.0008 / +0.0004** |
| rms coordinate error, A | 0.0021 / 0.0066 / 0.0093 / 0.0114 | 0.0010 / 0.0043 / 0.0088 / 0.0148 | **0.0008 / 0.0030 / 0.0080 / 0.0090** |
Preregistered readings on v3 (N3p): **sd criterion MET, mean criterion MET (|mean| <= 0.3 for all 8; the largest is La1.U 0.28), coordinate bias <= 0.02 A MET. "Converges from the registered N3 start" NOT MET (70 %).**
The mean criterion passes on v3 and failed on v1 and v2: it measured a reduction defect, as the lenses said. Formal sigmas (v3): La2.z 0.0009 A, Ni.z 0.0026, O1.z 0.0070, O2.z 0.0099. Determined: the same 8; O2.U and O3.U undetermined.
Caveat: still one signal realisation and one orientation, so the mean-pull test is conditional on them; the v3 noise-free chain still has errors up to 0.25 formal sigma (Ni.z), which is consistent with the simulator's own 0.30 % Lorentz noise (A16) but that is not shown separately.

### C3 on v3 (`results/c3_misspecification_map.txt`; closed-U version `..._v3_closedU.txt`; v2 version `..._v2.txt`)
* **Geometry (omega zero-point, beam centre, detector distance): no shift for beam centre <= 0.5 px, distance <= 0.5 %, omega <= 0.3 deg when U is exact** (corrections follow the calculated crossing, so measured geometry enters only through hkl assignment). **It is not zero at 1 px beam-centre error** (O2.z -3.4e-3 A, O2.U +1.4e-3, O3.U +3.5e-3 A^2; 23 of 147 spots fail the hkl gate, a 16 % data loss) and at +2 px nearly every spot fails the gate. The error moved into the hkl gate; it was not removed. Re-indexing U under a wrong geometry absorbs only 0.03-0.08 deg, so the U rows below are NOT a stand-in for geometry errors.
* **Orientation error is the dominant sensitivity.** U rotated by 0.1 deg about (1,1,1): La2.z +1.4e-3 A (above its formal sigma 0.9e-3), O1.U and O3.U move; 0.3 deg: 9 of 10 parameters move, O2.z by -0.012 A; 0.6 deg: O2.z by -0.028 A (above the 0.02 A tolerance), Ni.z +0.011 A; 0.3 deg about the beam axis: the fit does not converge. A rotation about the rotation axis (= an omega zero-point) changes nothing. **The orientation must be good to about 0.1 deg or better, and the effect depends on the rotation axis** (La2.z moves by more than 1 sigma on 3 of 6 axes at 0.1 deg; the rows above used one axis). The artifact lens's noise-free scan: v3's worst systematic error is 0.25 sigma for an exact U, median 0.33 / max 0.82 at 0.02 deg, 0.70 / 1.54 at 0.05 deg, 1.42 / 2.16 at 0.10 deg, against 1.39 sigma for v2 at any U: **v3 only beats v2 while U is good to ~0.05-0.1 deg.** midas-sx does not yet refine U from the observed spots (all runs use the true U).
* Unchanged from v2: no Lorentz correction is fatal (GooF 32, no convergence); no polarization correction moves Ni.z by +0.0196 A and 5 other parameters; a polarization fraction of 0.50 moves 6; no absorption correction moves O2.z by -0.0184 A and Ni.U, O1.U, O2.U; anvil thickness +-20 %, mu +-25 %, DAC tilt absent or 2 deg wrong, polarization fraction 0.90: only O1.U (and O3.U for the absent tilt) move, by ~0.001-0.002 A^2.

### What the whole exercise says about the user's real question (still only what a simulator can say)
1. Check counts first: ~1e5 integrated counts on the strongest reflection for a bilayer-like cell; at 1e4 nothing useful is determined.
2. Check the ORIENTATION: it must be good to ~0.1 deg, and the refinement start to ~0.1 A (0.4 A can land in a false minimum from some sign patterns).
3. Apply Lorentz and polarization properly (true beam polarization); apply an absorption correction for oxygen but do not worry about +-20 % in anvil thickness or mu.
4. Evaluate the corrections at the CALCULATED reflection position, not at the measured centroid of an extended spot (this was a 5-50x effect on the parameters here).
5. All of this is for a kinematic, untwinned single crystal with the true DAC model; real data add extinction, twinning, diamond reflections and gasket rings, none of which are simulated.

### Open
* U refinement from observed spots and an indexing arm (midas_defect has the pieces); the claim above all use the true U.
* A higher-hit simulation (target_hits >= 1e6, Lorentz noise 0.1 %) would separate the simulator's own residual from anything left in the reduction.
* Real data. Nothing here has touched any.


### Second verification (claim f5116b2d711d, v3): PROVISIONAL (no lens refuted)
statistics SURVIVES (narrowly; the mean-pull test has little power, the 50 seeds measure one fixed offset, N3 and N3p are the same evidence, and 35/50 from the registered start is the draw of start signs: La2.z - with Ni.z + fails 13/13);
reproduction SURVIVES (an independent refiner on all 50 v3 tables matches the repo to 1.7e-7 A and 8.5e-5 relative in sigma; the calculated crossing matches the simulator's nominal one to 3e-6 deg, a gap traced to the simulator's own grid interpolation);
physics UNCERTAIN (corrections-at-the-crossing physics and the U-sensitivity size confirmed by an independent propagation; the geometry wording and the stand-in claim corrected, A18);
artifact UNCERTAIN (every number reproduces; all of it is for an exact U, which a real analysis does not have). The v3 noise-free acceptance (A15) was NOT met (worst 0.247 sigma) and v3 is the third reduction judged on the same 50 seeds and signal.

## Orientation refinement from the measured spots: EXPLORATORY probe (not preregistered; `run_u_refine.py`, `results/u_refine_exploratory.txt`)
Question (HS): the MIDAS tools can refine U, so can we do better than the ~0.05 deg the structure refinement needs? Probe: S2, C2 seed-0 noisy frames through the real front end, 142 assigned spots
(hkl assignment used the TRUE U, so indexing itself is not tested), U refined with the existing `midas_defect.seed_index.refine_U_from_centroids` (cosine loss on the sample-frame q directions) from deliberately wrong starts.
* **Refined U lands 0.016 deg from truth from every start tried** (0.1, 0.3, 0.6 deg about (1,1,1), 0.3 deg about the beam, 0.3 deg random axis): 0.0163 deg in 4 of 5 starts at 400 steps, 0.010-0.023 deg across step counts (Adam jitter).
  A bootstrap on 70 % of the spots gives a spread of 0.014 deg. One pixel is 0.172 deg here, so this is ~0.1 px: centroid noise, not the optimiser, sets the floor of this crude method.
* **Propagated to the structure refinement (same spots and noise): errors in formal sigma, exact U vs U refined from the spots (0.016 deg off) agree to within 0.6 sigma for every parameter and the GooF is unchanged (1.22 vs 1.23).** With U left 0.3 deg off the parameters move by up to 3-4 sigma (La1.U -3.94, La2.U +2.62, O1.U -2.34) while the GooF is STILL 1.22: **an orientation error of this size is invisible in the goodness of fit.** U left 0.05 deg off is within ~1 sigma of exact.
* **Reading, corrected after four lenses (claim ac0610024833, PROVISIONAL, none refuted):**
  - 0.0163 deg is reproduced by a closed-form Wahba/Kabsch solution (the loss at that U is 0.0022879 vs 0.0022969 at the true U), so it is a genuine minimum of the loss, not an Adam artefact (the 0.010-0.023 deg scatter across step counts is optimiser jitter; the tool has no lr decay and no convergence test). It holds for EQUAL weights; weighting by intensity gives 0.066 deg (the bright spots carry a small centroid bias), so do not weight by intensity.
  - **It is mostly a simulator floor, not a statement about real centroids**: the measured-minus-truth centroid error is 0.24 px rms, does not shrink with brightness, and correlates at 0.92 (row) with the simulator's own integer rounding of hit positions before the PSF blur. Real photon-limited centroids could do better; real geometry and cell errors do worse: with exact centroids a beam-centre error of 0.5 px moves the refined U by 0.016 deg (col) to 0.039-0.042 deg (row), a distance error of +2 % by 0.030 deg. So the refined-U error is of order 0.015-0.04 deg depending on geometry, which is at or below the 0.05 deg the structure needs only in the good cases.
  - **My earlier "about 3x inside the 0.05 deg requirement" is not supported**: with U refined to 0.016 deg the structure parameters move by up to 0.68 sigma (rms 0.32; the earlier "within 0.6 sigma" was wrong for O3.U), and with U left 0.05 deg off by up to 0.80 sigma (rms 0.45). The paired shifts do not scale with misorientation; they depend on the direction of the error vector, which is random in one draw. Only the 0.3 deg row shows a clear effect (up to 4.95 sigma from the exact-U row), and the "0.05 deg requirement" itself is an estimate, not derived.
  - One draw: a misorientation from one noise realisation has a 90 % range of roughly 0.007-0.04 deg.
  - The hkl circularity is not hurting here: re-assigning hkl from a U that is 2 deg off and iterating assign-then-fit recovers 0.009 deg, so indexing from a wrong start can work for this sparse single crystal.
  - A rotation of U about the rotation axis is an omega zero-point: the fit cannot see it and the structure does not care, but the half-frame convention (0.125 deg) moves U by exactly that amount.
  - GooF unchanged (1.22 vs 1.23) is an expected consequence of fitting ~11 parameters to ~54 F^2; it carries no information about U accuracy, so real data have no internal check on a 0.3 deg orientation error.
* **What this does not show:** one realisation; the simulator's geometry is exact (a wrong beam centre or distance would bias U and was not tried together with the refinement, which is the real test); hkl assignment used the true U; the refinement used q directions only. A preregistered version should (i) index from a wrong U and a wrong geometry, (ii) refine U and the geometry jointly (`midas_fit_grain` or a least-squares in 2theta, eta, omega), (iii) be run over the 50 noise seeds and judged on the structure parameters, not only on the angle.


## Phase 2, step 1: orientation and geometry refinement from the measured spots (PREREGISTER A19, A20)
`midas_sx/orient.py` (`index_and_refine`): joint weighted least squares of U (rotation vector), beam-centre row and column, and relative distance against the MEASURED (row, col, frame) of the assigned
reflections; prediction by `midas_defect.geometry.ewald_crossing_omegas` + `qlab_to_pixel` (production maps); equal weights (intensity weights were 4-5x worse), soft-L1 loss; hkl re-assigned at tolerance 0.25, 0.20, 0.15; cell held fixed.
Formal sigma from a Jacobian built by central differences of the sigma-normalised residuals (scipy's returned Jacobian was identically zero for soft-L1 in some fits). The pixel map needs `tol_px=1e-8` (its 1e-3 px default is coarser than a finite-difference step).
Gate: converged only if the optimiser succeeds and the ROBUST rms residual is below 2 px and 1 frame (a plain rms tripped on 4 of 50 seeds because of a few mis-centroided spots, after which the gate was made robust; the 4 'not converged' seeds in the first csv are that gate artefact, their U errors are as good as the rest, and the plain rms is still reported as `rms_px_all`).
Unit tests (`tests/test_orient.py`, 10): noise-free recovery from every start axis; formal sigma within a factor ~2 of the scatter; the omega-origin gauge (a half-frame shift = a 0.125 deg rotation about z, nothing else); a 2 % wrong cell is invisible in pixels (distance absorbs it) but visible in omega (1.7 frames); too few spots refused; outlier robustness.

### Result (A19, 50 noise seeds, S2; start: U 0.3 deg about a random axis, beam centre 0.5 px in a random direction, distance +-1 %) -- PROVISIONAL, synthetic
* **O1 MET:** refined U error median **0.0082 deg**, p90 0.0117, max 0.0163, 100 % within 0.05 deg. The existing direction-only `refine_U_from_centroids` from the same start: median 0.046 deg, bimodal (about 64 % near 0.03 deg, 36 % near 0.4 deg), p90 0.389, only 52 % within 0.05 deg (it can leave U WORSE than it started when the geometry is wrong). The median is a fair comparison; the tail is partly optimiser behaviour (Adam, lr 5e-3 rad, no decay, jitters even on exact data: p90 0.017 deg) and is not shown to be hkl misassignment (physics lens).
  All 50 seeds share ONE signal volume and ONE orientation (only the Poisson noise changes), so 0.0082 deg and the scatters below are repeatability on this crystal, not accuracy over orientations. With 12 fresh random orientations at the same noise (lens re-run, own code) the U error has median 0.0071 deg, max 0.0100 deg, but the beam-centre scatter is 0.014 px (row) / 0.026 px (col) and the distance scatter 1.2e-4, i.e. about 3-5x the 50-seed figures.
  Geometry recovered: beam centre to 0.005 px scatter (column bias +0.014 px, inside the fresh-orientation spread), distance error 1e-2 -> -3.6e-4 (identical for both start signs, about 3 sigma of the fresh-orientation scatter: a fixed offset of this crystal, cause not found). Residuals 0.24 px, 0.18 frames (the injected noise). Formal rotation sigma ~0.002-0.006 deg.
* **REFUTE conditions: none triggered** (0 % of seeds above 0.10 deg; 0 % hkl collapse; 6 % of seeds with a paired structure shift above 1 sigma).
* **O2 FAILED as registered (do not read it as met):** pull sd 0.76-1.16 (MET) and paired mean shift vs the exact-U run <= 0.23 sigma (MET), but the paired rms is above 0.5 sigma for La2.z (0.52), O1.z (0.75), La2.U (0.72), O1.U (0.59).
  The wrong start left unrefined moves the parameters by 1.4-3.7 sigma rms (La1.U 3.7).
* **Control (A20, post hoc, 25 seeds):** keeping the exact U and exact geometry in the reduction but finding the spots with the wrong-start geometry (v5): `v4 - v5` paired rms is 0.05-0.14 sigma for all 8 parameters (the refined U and geometry change the answer by far less than the bound), while `v5 - v3` is 0.13-0.90 sigma. **The most likely reading of the unmet criterion is
  spot-finding / selection noise from using a 0.5 px / 1 % wrong geometry for the polar-background maps, not the orientation estimate. This attribution is post hoc: n = 25, one orientation, and direct for seed 9 only (the misses concentrate in seeds 9, 33, 47).** The earlier suggestion that it was a difference in which reflections are present was tested (refining on the common unique reflections) and NOT supported.
* **Practical consequence (untested):** after refining the geometry, re-run the front end with the refined geometry before integrating.
* Verification of this claim (4 lenses): the independent reproduction (own Ewald solver, pixel map, rotation parametrisation, hkl assignment) reproduced two seeds (0 and 4) only: U 0.0116 / 0.0080 deg as in the csv; the physics lens validated the forward model against the simulator's own code (max 3e-6 px, 1.2e-5 frames), the Jacobian (condition 1.36) and noise-free exactness; with the geometry left at its wrong start the orientation-only fit gives 0.021-0.022 deg, so the joint geometry fit improves U by about 2-3x. The sigma calibration (predicted rms 0.0086 vs observed 0.0088 deg) is the statistics lens.
* Artifact lens (UNCERTAIN, so the verdict is PROVISIONAL): (i) the comparison with the existing tool is not like for like. `run_c2.py` v4 drives `refine_U_from_centroids` once, at hkl tolerance 0.25, with no re-assignment and a non-robust loss; first-pass hkl are correct, optimiser choice (Adam lr 5e-3/5e-4, scipy) makes no difference, and a few gross-frame-error spots dominate its loss (seeds 36 and 5 stay at 0.87 / 0.35 deg even with exact geometry). Given the same three-round re-assignment it reaches 0.010-0.026 deg with exact geometry and 0.036-0.087 deg with the start geometry (4 seeds), so the 0.389 deg p90 is mostly driving, the geometry fit is worth about 0.03 deg, and direction-only vs full-position is not isolated. (ii) 0.0082 deg is the ideal case: exact cell, no detector roll, omega origin shared with the simulator. In the lens' model-spot stress test (150 spots, 8-12 seeds per row) a +0.5 % / -0.5 % anisotropic cell error gives 0.19 deg, an unmodelled 0.2 deg detector roll gives 0.14 deg, an isotropic +0.5 % cell gives 0.018 deg, and all of these still pass the convergence gate. Detector tilts ty, tz of 0.2 deg are absorbed (0.0077 deg). Recovery does not depend on the start size up to 2 deg / 3 px / 5 %. (iii) The 'converged 46/50' line in `results/c2v4_analysis_orientation.txt` predates the robust gate and is stale (the gate never changes U).
* **Not shown:** indexing from nothing (a rough start is given), the cell (held at truth: the distance and the cell trade off), tilts and distortion, real data; the simulator's centroid error is dominated by its own integer rounding of hit positions (the physics lens of the earlier probe), so 0.008 deg is a statement about this simulator.


## Phase 2, step 2: primary extinction in the refinement (PREREGISTER A21, A21a, A21b) -- PROVISIONAL, synthetic
Prerequisite finding (verify claim 0ffd5eab3724, ESTABLISHED, 4 lenses): `midas_hkls.primary_extinction_factor` = tanh(t/Lambda)/(t/Lambda) understates the exact perfect-crystal symmetric-Laue extinction (1/2A) int_0^{2A} J0, A = pi t/Lambda, by pi^2 at small t/Lambda (`extinction_convention_check.py`, `results/extinction_convention_check.txt`). The exact factor is now `midas_hkls.laue_extinction_factor` (+ `laue_kinematical_path_limit_um`, 10 % at 0.1785 Lambda); `midas_dct_tt.extinction_weights` uses it (dct_tt suite 597 passed, 8 skipped); the old function stays, with warnings. `torch.special.bessel_j0` is not differentiable and is off by 4e-7, so J0 comes from scipy with -J1 as the backward.
Built: `RefineModel(refine_extinction=True, wavelength_A=...)`, parameter e = t_b^2 (um^2; lower bound 1e-8 is the physical zero and is not a rail), `result.t_b_um`, `correlation()`; truth in `extinction_truth.py` (TT closed form, gemmi |F|, hooked into `simulate.py` as `f2_factor`); `run_c2.py` with env `EXT=T1_5|T1_10|T2`; `run_ext_refine.py` (arms K kinematic, E t_b free, O oracle = data divided by the true multiplier); `analyse_ext.py` -> `results/ext_analysis.txt`. 50 seeds per variant, v3 reduction, exact U and DAC model, 0.10 A start.

Result (read against A21; INCONCLUSIVE: nothing CONFIRMED, no REFUTE condition triggered):
* Extinction matters: at 5 um the kinematic fit is biased by up to ~10 sigma (K-O mean, oracle formal sigma; La1.U -10.3, La2.U -8.0, La2.z +6.3), GooF 3.3, 50/50 converged. At 10 um the kinematic fit rails on the U lower bound (K 0/50 converged; La1.U 100 %, La2.U 98 %, O1.U 82 % of seeds at 1e-4), so the quoted -28 / -43 sigma are floors set by the bound, NOT a measured extinction bias. At 10 um the bias is therefore "at least ~43 sigma". The mechanism is the generic scale / U_iso / z trade-off [KL-036] (K scale 0.074 vs O 0.084 at 5 um, 0.086 vs 0.136 at 10 um); E's scale matches O's (0.0842 vs 0.0841).
* E recovers t_b: median 5.054 um (IQR 4.947-5.127) and 10.012 um (9.989-10.042), 50/50 converged, GooF 0.96-0.97. This is repeatability on ONE signal volume and orientation, with a small detectable offset (+0.046 um, 2.4 SE at 5 um) and a function shared by truth and refiner (A21a): it checks the implementation and the fit machinery, NOT the model. Pull sd 0.85-1.23 for the 8 determined parameters (SE of a pull sd at n = 50 is ~0.1; La1.U at 5 um has sd 1.23 and mean +0.54, also present in the oracle arm: shared signal volume).
* Registered E1 criteria: all met except the paired E-O rms <= 0.5 sigma: worst La1.U 0.563 (5 um) and La2.U 0.546 (10 um) => NOT MET as registered. Post hoc, not a rescue: the nested-model expectation sqrt(sigma_E^2/sigma_O^2 - 1) is 0.523 (La1.U, 5 um), 0.666 (La2.U, 10 um), so the threshold left no room for the cost of the extra parameter (t_b is 0.92-0.93 correlated with the scale; never above 0.98). The same criterion is already 0.512 on the no-extinction data (E-K, La1.U), so it measures that cost, not misfit.
* E2 (T2, thickness spread sigma_ln 0.5, median 7 um): |mean pull| <= 1 met (max 0.83), E-O rms <= 1 sigma NOT MET (La2.U 1.23, La1.U 1.14); effective t_b 7.64 um. The misspecification costs about 1 sigma on U.
* E3 (T0, no extinction): t_b <= 1 um in 62 % (needed 80 %), formal sigma of t_b 0.83 um (so the threshold was about 1 sigma wide: a poor choice of mine), E-K rms 0.51 (La1.U). Sigma inflation max 1.11 and no bias above 1 sigma, so no REFUTE.
* Not shown: that t_b is not absorbing another dimming mechanism; Bragg or asymmetric blocks (symmetric Laue assumed, C = 1 both sides); redistribution of intensity out of the integration box; secondary extinction; real data.
* Verification (4 lenses, PROVISIONAL): physics and reproduction SURVIVES (own refiner reproduces t_b and chi^2 on 15 seeds at both thicknesses: 5.075 / 10.032 um medians, chi^2 58.5469 vs CSV 58.5468); statistics and artifact UNCERTAIN (the 10 um bias figure is a bound floor; recovery is self-consistency).



## Phase 2, step 3: anisotropic displacement parameters (PREREGISTER A22) -- ESTABLISHED (synthetic, 4 lenses SURVIVES), scope as below
Built: `RefineModel(aniso_sites=("La1","La2","Ni"))`; constraints DERIVED as the null space of `U - R_c U R_c^T` over the site stabilizer in Cartesian coordinates (`aniso_constraints`), parameters = independent Cartesian entries (Uxx, Uzz for a 4mm site) in A^2, converted to the CIF U_ij of `midas_hkls` (`to_cif`; gemmi agreement 1.000000 on tetragonal, hexagonal 3-fold-site and triclinic cells; a symmetry-forbidden tensor is detected). `result.absolute` carries U_eq (trace/3) and the smallest Cartesian-U eigenvalue (positive-definiteness is NOT imposed; min over 100 seeds 1.7e-3). Truth S2A: La1 U11=U22/U33 = 0.004/0.012, La2 0.0045/0.0100, Ni 0.0035/0.0080 A^2; gemmi `Site.aniso`; same simulator, reduction (v3) and merge as before. Scripts: `run_aniso_refine.py`, `analyse_aniso.py` -> `results/aniso_analysis.txt`.
Result (E1 and E2 CONFIRMED as registered; no REFUTE condition):
* S2A, arm A: 50/50 converged, GooF median 0.96; the 9 determined parameters (rule on the T0 data: La2.z, Ni.z, O1.z, O2.z, La1.Uxx, La2.Uxx, La2.Uzz, Ni.Uxx, Ni.Uzz) have pull sd 0.88-1.12 and |mean pull| <= 0.34. Arm I (isotropic U, today's practice): 49/50 converged (one on a bound), GooF median 3.26, mean error / sigma_A = 4.7 (La2.z) and 11.7 (Ni.z); in arm-I sigma 4.8 and 16.6; scaled by GooF 1.5-2.7 and 5-9. In Angstrom: La2.z +0.0049, Ni.z +0.038 against about 0.0002 and 0.0003 for arm A. U_eq biases of arm I: La1 -2.1e-3, La2 +1.6e-3, Ni -3.6e-3 A^2.
* T0 (isotropic truth): Uzz-Uxx within 2 sigma in 98 / 94 / 92 % (La1 / La2 / Ni); sigma_A/sigma_I of determined coordinates max 1.22 (median per-parameter ratio on T0; on S2A the Ni.z ratio is 1.40); A-I paired rms under 1.2 x the nested-model expectation + 0.1 (the A21 lesson applied).
* La1.Uzz (the largest anisotropy) is outside the determined set by the registered rule (median sigma 0.00158 vs 0.00127) but is recovered: mean 0.01218 vs 0.012, pull mean 0.12, sd 0.89; zero anisotropy excluded at > 2 sigma in 96 % of seeds.
* DEPENDENCE: arm A needs the `midas_hkls` mixed aniso/iso Debye-Waller fix. With the old `_debye_waller` (iso atoms got a factor of 1) 0/50 converge and GooF is 2.43 (artifact lens, `refute_artifact/old_vs_new.py`).
* Not shown: O-site anisotropy, a second anisotropy ratio, lower-symmetry structures in a refinement (the constraint code is unit-tested on hexagonal, cubic, triclinic and I4/mmm sites), the joint refinement with extinction, real data. One signal volume and one orientation; mean pulls are conditional on them (T0 La1.Uxx mean pull +0.48).


## Phase 2, step 4 (stage 1): two crystal domains (PREREGISTER A23, A23a) -- PROVISIONAL, synthetic; registered information-gain clause REFUTED
Scope (stated in the registration, nothing beyond it is claimed): two domains of the SAME structure with an arbitrary relative orientation (67.8 deg), mostly separate spots; domain orientations REFINED from a 0.3 deg start (not found from scratch); one scale per domain; domain intensity ratio 0.3/0.7 = 0.4286 at the same absolute scale (`simulate(scale_override=...)`, `run_c2.py` with `DOM2=1`). Exact overlap (merohedral twins), twin fractions on summed contributions and domain finding from scratch (`midas_defect.domains.find_domains` exists for the bilayer raster case) are stage 2 and are NOT done.
Built: `RefineModel(n_domains=2)` + `refine(..., domain=labels)` (one scale per domain, `scale`, `scale.d2`), `reduce_spots(..., neighbours=...)` (the crowding guard sees all found spots when one domain's subset is reduced), `run_twin.py` (S1 = single-domain chain on the two-domain frames; M = registered two-domain chain; M2 = exploratory looser labelling), `analyse_twin.py` -> `results/twin_analysis.txt` (registered arms also in `twin_analysis_registered_M.txt`; the rerun that added M2 reproduces the S1 and M analysis byte for byte).
Result, 50 seeds:
* C1 MET: domain 1 U error median 0.0093 deg (max 0.0174), domain 2 median 0.0058 deg (max 0.0151), 100 % within 0.05 deg, from a 0.3 deg start (domain 2 searched among the spots domain 1 does not explain; no truth used, but the start is 0.3 deg off, so this is a refinement statement).
* C2 MET: arm M 49/50 converged, GooF median 0.97, pull sd 0.81-1.14, |mean pull| <= 0.27 on the 8 determined parameters (mean pulls are conditional on one signal pair; SE of each ~0.12-0.16).
* C3 MET: median k2/k1 = 0.4301 (IQR 0.4292-0.4312) vs 0.4286 (+0.4 %). Physics lens: this is an intensity-scale ratio, not a volume fraction; the +0.3-0.4 % comes from a domain-dependent mismatch (weighted 1.0032) between the simulator's Monte-Carlo mosaic weight and the reduction's closed-form Lorentz x polarization x absorption.
* C4 NOT MET and the registered REFUTE condition TRIGGERED: M's median formal sigma is 1.05 times S1's (above S1 for 7/8; per-parameter ratios 0.98-1.10; paired per-seed, M is above S1 in >= 79 % of seeds for those 7). The registered hypothesis is therefore REFUTED in its information-gain clause.
* WHY (artifact lens, tested on this data): M keeps 45 unique domain-1 reflections, S1 has 53.5 (median): my labelling rule (ambiguous unless the other domain's residual > 0.25) throws away about 9 reflections per seed. At a MATCHED domain-1 set the second domain lowers sigma by 3-10 % (M / domain-1-block-only 0.90-0.97, 8/8). Exploratory M2 (ambiguous only when both domains are inside the 0.15 gate): 48/50 converged, 51 domain-1 + 33 domain-2 unique reflections, sigma 2-5 % below S1 (ratios 0.954-0.984, 8/8; paired CI narrow, all 48 seeds below 1). The A23a joint reading (domain-1 reflections >= 52) was missed by one (51), so by the registered rule the cause is "not supported (or only partly)"; the matched-set comparison is the direct evidence and is post hoc.
* S1 is slightly flattered: its 430 extra reflections (in S1, not in M) have z sd 1.94 (4.0 % beyond 3 sigma) because they sit next to a domain-2 spot and their sigmas are understated; the shared reflections have z sd 0.96. The 'contamination' column (median 0.019, kept domain-1 reflections with a true domain-2 spot in the support box) misses that.
* The sigma gain is FORMAL sigma. The realised RMS error of M2 relative to S1 has bootstrap CIs that include 1 for 7 of 8 parameters (statistics lens): no demonstrated gain in actual error.
* Not shown: domains found from scratch, near or exact overlap, a shared geometry error, a different relative orientation or intensity ratio, more than two domains, real data. One pair of signal volumes.
* Verification (4 lenses, PROVISIONAL): physics, reproduction (independent refiner, all 50 seeds, GooF to 1.5e-6) and artifact SURVIVES; statistics REFUTED on a wording slip of mine (pull sd range 0.82 should read 0.81; S1 'unique 54' is a rounded median of 53.5), which by the rule leaves the claim PROVISIONAL.


## Phase 2, step 4, stage 2: exactly merohedral twin and a refinable twin fraction (PREREGISTER A24) -- ESTABLISHED (synthetic, 4 lenses SURVIVES), scope as below
Built: `RefineModel(refine_twin=True)`: `F2_calc(h) = k [(1 - m) |F(h)|^2 + m |F(T h)|^2]`, `T h = twin_matrix @ h` (default the 90 deg rotation about c, (k, -h, l)), m bounded to [0, 0.5] (domain 1 = majority; lower bound is the physical zero, not a rail), `result.twin_fraction`, `sigma_twin_fraction`; valid when T normalises the merge point group (checked for Immm, F2_tw constant on mmm-equivalents to 3e-13). Truth: S2O = the S2 cell (a = b) in I m m m with the equatorial O split in z (O1a 0.105, O1b 0.085; 0.4 A apart), `simulate.structure_s2o`, `twin_f2` + `simulate(f2_override=...)` (gemmi F2 of h and of T h), `run_c2.py` with `TWIN=<percent>`; only 12 % of the reflections have F2(h) and F2(T h) differing by more than 5 %. Scripts `run_twin2_refine.py`, `analyse_twin2.py` -> `results/twin2_analysis.txt`.
Result (E1, E2, E3 CONFIRMED as registered; no REFUTE condition), 50 seeds per variant, arm U = untwinned (today's practice), arm T = twin fraction free (start 0.15):
* m = 0.30: median m 0.2959 (IQR 0.277-0.320), sigma_m 0.027, pull of m mean -0.14 / sd 1.15, detected at > 2 sigma in 100 %; GooF U 1.34 vs T 0.93; T converged 49/50.
* m = 0.10: median m 0.0925 (IQR 0.063-0.121), sigma_m 0.032, detected in 68 % (a calibrated sigma_m would give about 85 %: sd(m)/sigma_m is 1.25 in both twinned variants); GooF U 1.01 vs T 0.96.
* m = 0 (null): median m 0.0000, m <= 0.05 in 92 %, m > 0.10 in 0 %, no pull meaningful at the bound; sigma inflation of arm T over U max 1.37 (O1a.z), paired T-U shifts under the nested bound (A22 lesson).
* What ignoring the twin does: the untwinned fit of the m = 0.30 data reaches an ACCEPTABLE GooF (1.34; below the registered 1.5x bar, so 'informative' was met by the coordinates) but biases O1a.z by -0.082 A and O1b.z by +0.087 A (-6.6 / +6.6 sigma of arm T; -8.8 / +6.5 in arm U's own sigma): the twin averages the two inequivalent O sites and pulls them together. Noise-free refit of the untwinned model on exact twinned F2 gives -0.080 / +0.090 A (physics lens). A twin of this size is NOT visible in the goodness of fit.
* Identifiability: |corr(m, O1a.z)| median 0.49 (m = 0.30), 0.64 (0.10), 0.68 (0); never above 0.98. Start without the O1 split (O1a.z = O1b.z) lands on the same optimum with the same m (0.3065 / 0.1000 / 0.0000; artifact lens, 20 seeds): m is not read off the known split.
* Limits that are INPUTS: the majority-domain labelling. A start with the O1 split reversed rails m at 0.5 and does not converge on twinned data (tw30 0/20 converge); on the null that failure is quiet (GooF 0.98).
* Not shown: pseudo-merohedral twins (a != b, split spots), more than two components, other twin laws, one orientation and one 0.4 A split size, domain finding from scratch, real data. All seeds of a variant share one signal volume. Verification: physics, statistics, reproduction (independent refiner, all 150 fits, m to the CSV's 4 digits) and artifact SURVIVES.



## Phase 2, step 5: domain finding from scratch with `midas_defect.domains.find_domains` (PREREGISTER A25, A25a) -- PROVISIONAL, synthetic
Scope: the A23 two-domain frames (S2, orientation seeds 1 and 3, intensity ratio 0.4286; the raw matrix angle is 67.8 deg, the D4-folded crystallographic separation 33.7 deg), NO orientation given to the search, the declared cell (3.83, 20.5, space group 139), blobs found under the A19 wrong-start geometry (0.5 px, 1 %). The tool's own parameter `sigma_rtn` is per-dataset by its docstring: measured on ten UNSCORED seeds (100-109) from the true orientations as the median (radial, transverse, normal) q-residual std of the spots they index: (0.0221, 0.0374, 0.0398), about 3x the defaults (0.0071, 0.0145, 0.0094); with the defaults seeds 0-2 give the same two domains (artifact lens), so the result is not sensitive to the calibration. Everything else at default. Scripts `run_finddomains.py`, `analyse_fd.py` -> `results/fd_analysis.txt`, per-seed JSON lines `results/fd_*.jsonl`.
Result (50 two-domain seeds + 25 single-domain null seeds; read as registered INCONCLUSIVE because of C2):
* C1 MET: the two largest found domains match different truth orientations within 1 deg (D4-folded) in 50/50 seeds; largest domain median 0.212 deg (p90 0.273), second largest 0.360 deg (p90 0.525); no domain beyond 2 deg. Reflections claimed: 140 / 66 (median). Convention: `Domain.U` is the simulator's U (q = U B h) in 50/50 seeds, checked on hkl residuals, not only on fold angles.
* C3 MET (94 %, 47/50): from the found U, `index_and_refine` reaches domain 1 median 0.0093 deg and domain 2 0.0063 deg (<= 0.017 deg in all 47). The 3 failures are the 3 seeds that returned a third domain, where I refined the second FOUND domain (a fragment) instead of the second LARGEST; not re-run.
* C4 MET: 25/25 single-domain frames return exactly one domain, within 1 deg of the truth.
* C2 NOT MET (14 %): the recovered absolute cell scale follows the injected distance error: c/20.5 - 1 = -0.81 % (28 seeds, sd 0.013 %) for a -1 % start distance and +0.94 % (22 seeds) for +1 % (correlation 0.999; the tool's own lattice fit reproduces -0.81 / +0.81 % when the truth spots are rescaled). c/a is within 0.3 % (sd; 0.14 % with the mean of a and b), median +0.09 %. A cell from `find_domains` is only as good as the detector distance.
* Spurious third domain: 3/50 two-domain seeds (33, 36, 38; 21, 21 and 25 reflections; 95 % CI 1.3-16.5 %; all three had a -1 % start distance, Fisher p = 0.25). It is a fragment of the REAL domain 2 split off at a wrong orientation by the row branch (16-18 of its 21 spots are indexed by the true domain 2); the real domain 2 then comes out of the pair branch with 43-49 reflections instead of 62-72. The null (0/25) cannot bound this rate (upper 95 % limit 13.7 %) because the fragment only appears when domain 2 is present. Any downstream code that takes "the second domain found" is wrong in about 6 % of seeds; take the second largest.
* The second domain's larger orientation error (0.36 vs 0.21 deg) is not explained by the start geometry (beam-centre/distance errors do not correlate with the fold angle; the geometry alone moves the true-U fit by 0.014 / 0.053 deg); it comes from the free-cell lattice fit and may contain a systematic floor repeated in every seed (one pair of signal volumes), which the stored fold angles cannot separate from noise.
* Not shown: other structures or space groups, more than two domains, small misorientations between domains, other intensity ratios, noisier data, a wrong declared cell, a distance error beyond ~1.5 % (the tool's declared-c gate is 1.5 %; the recovered c error reaches 1.15 %), real data. One pair of orientations.
* Verification (4 lenses, PROVISIONAL): physics, reproduction (independent driver, seeds 0-4: n_found, sizes, cell and fold angles reproduce, fold to < 1e-3 deg) and artifact SURVIVES; statistics UNCERTAIN (rate CIs, null bound, shared signal volume).


## Phase 2, step 6: pseudo-merohedral twin with split spots, a != b (PREREGISTER A26, A26a) -- registered verdict REFUTED (convergence); PROVISIONAL, synthetic
Scope: S2P = S2O with b = a x (b/a); second component = first rotated +90 deg about c (U2 = U1 Rz(+90), the twin law and cell are GIVEN); the reflections h and T h = (k,-h,l) are split by delta-q = (1/b - 1/a)(h,-k,0) (zero only on 00l; checked against the simulator's truth tables to 2e-4 px median). Variants: P11 (b/a 1.011, m = 0.30: every pair a blend, median separation 0.38 px), P30 (1.03: 405 blended + 29 + 31 resolved targets per seed, median separation 0.99 px), P00 (1.011, domain 1 only). 50 seeds each, one shared signal pair per variant.
Built (all in `midas_sx`): `predict.py` (`enumerate_hkl`, `predicted_crossings`, `twin_targets`: a pair closer than 2x the spot half-width in pixels and frames is ONE blended target at the midpoint, otherwise one target per component), `reduce_spots(extra_px / extra_frames)` (the box widened by half the separation), `refine(partner=, flags=)` (per-observation component weights), `make_cache_p.py`, `run_pm.py`, `analyse_pm.py`, `analyse_pm_posthoc.py`. Predict-and-integrate (A12 'not built') is now built and was tested here: the corrections at the first component's crossing differ from the second's by a median 0.2-0.4 % (95th percentile 0.8-1.4 %).
Result (read as registered: REFUTED because arm T converged in 25/50 and 26/50 < 80 %):
* m: median 0.2851 (IQR 0.257-0.310, sigma_m 0.043) at P11, 0.2990 (IQR 0.290-0.303, sigma_m 0.0039) at P30, 0.0027 on the null (m <= 0.05 in 94 %). m within 0.02 of 0.30 at both splits. An independent refiner reproduces these medians exactly over all 100 twinned fits.
* Precision gain from the split: sigma_m drops from 0.043 (blends only) to 0.0039 (resolved components present) and the untwinned fit biases O1a.z by -2.9 (P11) and -5.5 (P30) sigma of arm T. The A24 'exact-overlap' arm X reached m -> 0 at P30, but that arm mislabelled the resolved components; correctly summed pairs give m = 0.29. So the split does not require a different model; it makes the same model much more precise.
* Why 'REFUTED': the gate. All P11 failures are bound hits of the undetermined oxygen U (50/50 pass without those). At P30 43/50 pass; the other 7 have a determined parameter (Ni.U, La1.U, O1a.U) on a bound, and non-converged P30 seeds have 3-4x larger errors (m scatter 0.0175 vs 0.0044) with four seeds whose strongest reflections are ~20 % low. The convergence defect at 3 % is therefore real, not only a gate artefact; sigma_m = 0.0039 holds on the converged seeds only. m pull sd: 1.01 (converged), 2.06 (43 post hoc seeds), 4.12 (all 50; two seeds at +7 / +9).
* Orientation: refining U1 on blended centroids costs accuracy: 0.049 deg at P11, 0.029 deg at P30 (6 % of seeds above 0.10 deg), 0.006 deg untwinned; a crude emulation reproduces the order (merged pair centroids). It does not explain the La2.U offset (+1 sigma at P11, +0.8 sigma already in the null).
* P11 pulls: not calibrated as registered (La2.U mean +0.98, Ni.U -0.86). Weak rows: the truth model over-predicts the weak rows of the twinned data (z = -0.11 at P11, -0.057 at P30, -0.003 in the null); the cause is not identified.
* Null (P00): calibrated (sigma_m 0.0126, m > 2 sigma in 2 %); the T-U nested-bound criterion is missed by one seed (O1a.z 0.48 vs 0.40, CI contains the bound). There is no single-domain null at b/a = 1.03.
* Not shown: unknown twin law or cell (both given), refinement of b/a, more than two components, other splits, an independent U2, one orientation, real data, any fix for the P30 outlier seeds (e.g. refining the orientation on the pair model rather than on centroids).
* Verification (4 lenses, PROVISIONAL): reproduction SURVIVES; physics, statistics and artifact UNCERTAIN (the corrections above).


## Phase 2, step 8: a wrong declared cell for domain finding (PREREGISTER A28, A28a, A28b) -- PROVISIONAL, synthetic
Procedure: `run_fd_sweep.py` on the cached blobs of the 50 A25 two-domain frames (seeds 0-24), 11 declared-cell conditions (isotropic s = -5 ... +5 %, c only +-3 %, a only +-3 %), single pass P1 and a self-calibrating second pass P2 (rerun with the cell and `nominal_c` of the largest domain of the first pass, kept when it explains more spots). Success = A25's rule. Results and the corrections of the verification are in PREREGISTER A28b; the headline, as the data support it:
* The tool's own cell gate (`cell_tol_c` = 1.5 %) is centred on the cell the tool returns; with the +-1 % distance error of this data that cell is -0.8 % / +0.9 % off the truth, so the single pass finds both domains over a window of about 3 % width that is shifted by the distance error (not a symmetric +-1.5 % around the truth). It loses the second domain exactly in the six seeds whose second domain only the pair branch can find.
* A second pass that re-centres c on the first domain's returned cell gives 100 % for effective errors up to ~4.2 % and fails near 5.8 %; recalibrating `a` alone does nothing and an a-only error is harmless to the score.
* Independent driver (350 find_domains calls) reproduces every stored success exactly.
* Practical rule: if the first pass finds only one domain, rerun with the c (and `nominal_c`) the tool returned. Not shown: other geometries, other signal pairs, other lattice types.


## Phase 2, step 7: a fix for the b/a = 1.03 outlier seeds of A26 (PREREGISTER A27, A27a) -- registered read INCONCLUSIVE, PROVISIONAL, synthetic
Built: `refine(outlier_sigma=5, gate_free=...)` (worst-first rejection on the internally studentized residual; a convergence verdict that ignores bound hits of named parameters), `midas_sx/twin_orient.py` (orientation and geometry refined on the pair model: blob at (1 - m) p1 + m p2, resolved components at p1 or p2, m a global nuisance weight bounded to [0, 0.5]), `run_pm2.py` (tables with that orientation), `refit_pm.py` (arms U, T, Ur, Tr on any table), `analyse_a27.py` -> `results/a27_analysis.txt`. Cause of the A26 failures, post hoc: single rows with a quoted sigma far too small for what is wrong with them (a box that missed its spot), not the orientation.
Result, 50 seeds per variant (full discussion and the verification corrections are in PREREGISTER A27a):
* Convergence under the revised gate: 49/50 at b/a 1.011 and 49/50 at 1.03 (the plain arm under the same gate: 49/50 and 43/50). At 1.011 this is the gate redefinition alone; the outlier rejection adds six seeds and a calibrated m pull (sd 1.43 -> 1.06) at 1.03 only. The rule is inert on seeds 14 and 25 in the pair-model tables.
* m: 0.2965 (b/a 1.011) and 0.2985 (1.03); GooF 1.03 / 1.01; at most 2 rows rejected per seed (median 0); null: m <= 0.05 in 100 %.
* Orientation of domain 1: 0.0055 deg against 0.0491 (b/a 1.011, better in 50/50 seeds) and 0.0208 against 0.0287 (1.03, 49/50), 0.0068 vs 0.0062 on the null. The registered 0.0143 deg at b/a 1.03 is missed, and the remaining floor is unexplained.
* Still failing: the registered pull calibration at b/a 1.011 (La2.U +1.42, Ni.U -0.87); the La2.U bias is present without the twin fit (untwinned robust arm +1.24) and tracks an additive offset of the blend rows (-36 +- 2 vs -0.7 +- 0.6 untwinned) that is not understood.
* The fitted blend weight is a nuisance shift (0.057 / 0.077), not the intensity fraction 0.30.
* Verification (PROVISIONAL): reproduction SURVIVES (independent refiner and rejection rule, 100 fits); statistics REFUTED on the attribution wording (corrected above); physics and artifact UNCERTAIN.


## Phase 3 (2026-10-21): verification of what ships, the unexplained residuals, a second orientation, the joint model

Everything below is simulated data. Per-claim lens files: `results/verify_*_2026-10-21.md`. The sampler (skill-log) wanted one claim in three; all were verified. The rule is mechanical (>= 2 REFUTED, or reproduction REFUTED -> REFUTED; any UNCERTAIN -> PROVISIONAL; 4 x SURVIVES -> ESTABLISHED).

### Verification outcomes
| Claim | id | Verdict | One line |
|---|---|---|---|
| `midas_hkls.lorentz_polarization` = full-ring average, K ignored | 7054b7947ba7 | ESTABLISHED | old (1-K)+K cos^2 2theta is the single-direction value; legitimate for a fixed scattering plane |
| mixed aniso/iso Debye-Waller fix (`_debye_waller`) | 4739ca7e4ffb | ESTABLISHED | gemmi agreement 3e-7; worst 1.6e-5 over 10 structures x 4 masks; no gradient at exactly U = 0 |
| `corrections` Lorentz and P.T vs the simulator | 81fe013a9b1f | ESTABLISHED | 'independent implementation' only: mu, slab/cone model and polarization plane are shared |
| `allowed_multiple_parity` from the centring translations (midas_defect) | 927641617f34 | ESTABLISHED | find_domains with I-centred groups 87/107/119/140/71: 25/25 (9/25 before); centring only, screws/glides ignored |
| extinction sizing with the tanh factor | 4c303c76ef31 | RETRACTED | rests on the factor shown wrong by pi^2 |
| C4 Fisher forecast | ec9b556b8e3b | REFUTED as worded | the weak-reflection table was stale (4-29 %, not 6-33 %); the 0.88-0.92 match reproduces but is calibrated to one orientation |
| C3 misspecification map | 5017f82f9b6f | REFUTED as worded | baseline not paired to the variant's spots; geometry rows null by construction; corrected map `c3_misspecification_map_paired.txt` |
| extinction per twin domain before the sum (`refine.py`) | c1873418e535 | REFUTED as worded | the old ordering is 2.83e-3 off (not 7.7e-4: my misread of a truncated array); matters at t_b >= 15 um or 10x counts, not at 8 um with repository noise |
| `midas_2d.solid_angle_polarization` per pixel | eab0fff812f8 | PROVISIONAL | values match an independent Thomson to 1e-15; my detach changed dQ-gradient 23 % (found by the artifact lens, fixed by a torch-aware `polarization_factor`) |
| `midas_sx.workflow` composes the validated chain | 49b86bbe41de | PROVISIONAL | bit-identical rows on seeds 3, 8, 21 with explicit d_min; the DEFAULT d_min was the corner (pool 4702 vs 2354), fixed to the validated definition |
| orientation precision (A32) | 0e1d4287b7f5 | REFUTED as worded | holds at 0.01/0.02 deg; 0.05 deg is the onset of breakdown; needs an exact cell (0.5-1 % cell error gives 1-3 sigma) |
| background-deficit mechanism (A34/A35) | 21f14d2f24bb | REFUTED as worded | mechanism sharper than stated: curvature of the background x blend-inflated box; quadratic shell fit removes it |
| joint model (A33) | 3e5f69ee9f55 | PROVISIONAL | independent numpy fit matches all 30 seeds to 1e-8; t_b is an effective parameter |

### What the investigations found
1. **The P11 La2.U / Ni.U bias and the underestimated twin fraction (A27's unexplained residual R1)** come from a background-subtraction bias in the integration, not from the twin model: blended-pair rows on noise-free frames lose 43 counts (median, expected < 3000) at the 5415-voxel boxes the measured-extent profile gives when it is built from blended blobs, against -5.5 (b/a 1.03) and -1.9 (untwinned). The cause is the curvature of a smooth 2theta background: the mean of the shell around a box exceeds the mean inside it by the curvature, times n_box (97 % of the deficit; flat background: exact at every box size; deficit ~ half-width^3). A plane fit to the shell, a median and a gap do not remove it; a quadratic fit does (+0.01 counts, 766 rows). Registered A34 CONFIRMED (the oracle removes it); A35 INCONCLUSIVE as registered with a registration flaw; the structure-bias numbers rest on about 3 effective fits. See `results/verify_background_deficit_2026-10-21.md`. NOT done: the fix (a quadratic shell background in `integrate_box`), a test on noisy frames, and the check on whether real backgrounds are curved enough. The earlier hypothesis (tails of the support in the shell) is refuted: the signal part of the shell mean is 0.00000 per voxel.
2. **The centroid orientation floor is a pure bias** (A29 D2: clean frames give the same error as noisy ones: P00 0.0056 / 0.0063 deg, P30 0.0272 / 0.0273, P11 0.0511 / 0.0482). The pair model's own floor at b/a 1.03 (0.0208 deg, registered target 0.0143) was NOT tested this way and remains unexplained.
3. **A wrong declared space group** (A31): the tool uses the group only through the centring mask and `allowed_multiple_parity`; a primitive lattice declared for an I-centred crystal finds both domains in 2/25 seeds (and 6/25 return a wrong list silently); the parity defect for the other I-centred groups is fixed (9/25 -> 25/25).
4. **Orientation precision (A32)** and its limits: see the table row and `results/verify_orientation_precision_2026-10-21.md`. The U error reaches F2 only through the Lorentz factor at the crossing; a cell error of 0.5-1 % does much more harm than the orientation.
5. **A second orientation (A30)**: typical O4: INCONCLUSIVE for both tests (22/25 and 23/25 converged under the registered gate, 25/25 with the oxygen U gate_free; a coordinate bias of 0.4-0.7 sigma remains: La2.z -0.6, O1.z -0.7, O1b.z -0.6 sigma); hard O6 (c* 13 deg from the beam): UNINFORMATIVE (0-1 determined parameters), the chain flags non-convergence and returns no confident wrong value (0/325 and 2/325 estimates beyond 3 sigma). The ESTABLISHED claims on anisotropic U and the twin fraction keep their status for the orientation they were verified on; their replication at O4 is INCONCLUSIVE, not a refutation.
6. **The joint model (A33)**: registered reading REFUTED on convergence (arm F 20/30; every failure an undetermined in-plane cation U on a bound, GooF 0.7-1.2); the twin law swaps x and y so Uxx and Uyy are nearly degenerate (corr -0.91 at m = 0.3, exactly degenerate at m = 0.5), which predicts the bound hits (12 expected, 10 observed); with them gate_free (post hoc) 30/30, m 0.2995 (pull sd 1.23), t_b 4.95 um. Every ablation is needed. t_b is an effective parameter (+10-12 sigma if the domains differ in thickness; GooF unchanged); it converges only from a start below about 20 um.

### Defects found and fixed during the verification (all uncommitted)
`midas_defect.rows.allowed_multiple_parity` (SG 139/69 special cases) -> derived from the centring; `midas_sx.refine` extinction applied to the twin sum -> per component, with the partner's own Bragg angle; `midas_sx.refine` read-only hkl arrays (torch warning, order-dependent test); `midas_2d.instrument.solid_angle_polarization` single-direction polarization -> per pixel and differentiable, with `polarization_axis` in `detector_signal`; `midas_hkls.corrections.polarization_factor` accepts torch tensors; `midas_sx.workflow` default d_min (corner -> inscribed edge), silent fallbacks now warn, minority-domain hint; `run_c3.py` pairing (symmetric); `c4_weak_reflection_value.txt` is STALE (written before the last edit of `forecast_c4.py`: re-run before quoting).
Not fixed (found): other `== (139,)` gates in `midas_defect/rows.py` (`orientation_grid` order 8; `index_by_grid` uses the 422 operators for every group); `midas_2d/forward.py rod_intensity` passes K to a function that now ignores it with a warning; `reduce_twin90_scan` has no unit test (equivalence checked by hand on 3 seeds); one refuter used `pkill -f` against the rules.

### Open (needs a decision)
(a) a quadratic shell background in the integration (registered test on held-out seeds, noisy frames, S2 untwinned and P11/P30), (b) the pair-model floor at b/a 1.03, (c) the coordinate bias of 0.4-0.7 sigma at O4, (d) tying Uxx = Uyy for twinned structures where the twin law swaps x and y, (e) real data, (f) the release (hkls 0.17.0 -> NF -> dct-tt -> midas-defect -> midas-2d -> midas-sx).
