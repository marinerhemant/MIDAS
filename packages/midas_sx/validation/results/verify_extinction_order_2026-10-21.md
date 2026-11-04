# /verify of claim c1873418e535 (extinction per twin domain), 2026-10-21: REFUTED as worded (a number), the fix stands

Lenses: physics UNCERTAIN, statistics UNCERTAIN, reproduction REFUTED, artifact SURVIVES -> REFUTED (reproduction). Workdirs .../verify_workdirs/Desktop/analysis/midas_sx/refute_c187_*/.

## What was wrong
The claim said the old ordering was 7.7e-4 off exact data. That number was MY MISREAD of a truncated numpy array in a failing assertion message (the printed fragment contained 7.7e-4; the maximum was not shown). Recomputed independently by me and by two lenses: **2.83e-3** at the test point (S2O, m 0.30, t_b 8 um, 208 reflections; worst hkl (0 1 13)), median 2e-7. The new ordering: 1.5-1.6e-6 (= the cross-code numerics floor: the model's kinematic F2 vs gemmi 4.5e-6, its extinction factor vs the exact Bessel integral 3.7e-6; 4.6e-6 even with no twin).

## What holds
- Physics: extinction per incoherent block, weights m and 1-m applied after extinction (intensity fractions), flags multiply after extinction, flags (1,0)/(0,1)/(1,1) reproduce the single components and their sum exactly.
- Regression: model bit-identical to the old one for extinction only, twin only, aniso only, aniso+twin, aniso+extinction (torch.equal); differs only for twin+extinction, as intended. gradcheck passes; Jacobian finite at e = 1e-8, m = 0 and 0.5 (NaN at e = 0 in old and new, unreachable: lower bound 1e-8). Cost +8 % forward at most. The A33 joint tables/fits were produced with the new code.
- The test tolerance (2e-5) discriminates at all 20 points of a (t_b, m) grid (smallest old error 1.8e-4 at t_b 2 um).

## How much it mattered (statistics lens; 30 noise draws, repository noise level)
t_b 8 um, m 0.3: old ordering biases m by -0.02 sigma and t_b by -0.09 sigma (paired, vs new); GooF 0.998 for both (the error is invisible in the fit statistic). At 10x better counting statistics: -0.18 / -0.89 sigma and a coordinate -1.17 sigma. At t_b 30 um: m -0.5 sigma. So the old ordering did not matter at the tested size with this noise; it matters for t_b >= 15 um or high counts. At m = 0.5 neither ordering converges.

## Limitation found (latent, not a regression)
theta_deg is computed from the primary hkl for BOTH components: the partner T h has a different d when a != b (b/a 1.011: 0.14 deg, 5.1-5.6e-5 relative; b/a 1.03: 0.37 deg, 1.5e-4), larger than the 2e-5 test tolerance but 20x below the old-ordering error. Also: both domains share one t_b; Lambda is for sigma polarization only (the pi component cos 2theta is ignored); truth and refiner share the perfect-block symmetric-Laue model; interleaved domains below the extinction length are not separate blocks.

## Amendments made after the verification (tested, not re-verified): partner theta from the partner's own d; read-only hkl array copied; the test comment and the registration text carry the corrected number.
