# Phase 3 - refining the structure, and what the result does and does not say

Sources: `packages/midas_sx/midas_sx/refine.py`, `workflow.refine_structure`, `validation/results/a39/CLAIM_tierC.md`, `validation/results/a39_floor/CLAIM_tierC_cfgA.md`, `validation/RELEASE_READINESS.md`, `validation/real/a39_refine.py`, `validation/real/a39_ref_hkl.py`.

## 1. The start model

A refinement here starts from a `Crystal` that carries cell, space group, atoms with the displacement parameter as B = 8 pi^2 U (`refine.EIGHT_PI2`), and the site labels you will refer to. Build it from the CIF (gemmi) and keep the labels:

```python
Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label)
```

* **`fixed_atoms=("H00A", ...)`** holds atoms (hydrogens) at the start positions and displacement: they are not refined and they are not free to move the origin.
* **`fix_origin=True` (default)** pins the polar-axis origin always, also with fixed atoms (an unpinned refinement of a polar structure drifts rigidly: 0.28 A on thiourea against 0.14 A pinned). Unit-tested; verified on one real set only.
* **`aniso_sites=("O001", ...)`** anisotropic U on the named sites. Isotropic refinement of anisotropic data biases the coordinates; the simulated claim is ESTABLISHED at the orientation it was verified on and INCONCLUSIVE on a second one (`packages/midas_sx/README.md`). On real L-alanine: iso GooF 3.14 against aniso 1.64 on the same table.
* **`refine_extinction=True` with `wavelength_A=`** (primary extinction, exact perfect-crystal Laue factor; the fitted `t_b` is an EFFECTIVE thickness; PROVISIONAL), **`refine_twin=True`** (declared law and majority domain; ESTABLISHED with narrow scope), **`n_domains=2`** (PROVISIONAL): read `packages/midas_sx/README.md` before using any of them.
* The start does NOT need to be the answer, but it decides the hand and the origin (section 4). Say which it was.

## 2. The call and the gates

```python
from midas_sx.refine import RefineModel, refine
model = RefineModel(start, fixed_atoms=fixed, aniso_sites=aniso)
fit = refine(model, hkl, F2, sigma, outlier_sigma=5.0)       # or workflow.refine_structure(crystal, merged, model_options=..., outlier_sigma=5.0)
fit.table(); fit.converged, fit.reason, fit.status, fit.goof, fit.R1, fit.wR2
fit.at_bound, fit.null_params, fit.full_rank, fit.sv_ratio, fit.rejected, fit.sigma, fit.sigma_rescaled
```

Read before quoting any parameter: `converged`/`status` (the verdict is gated, not just "least_squares stopped"), `at_bound` and `at_bound_harmless`, `null_params` (not determined by the data), `full_rank` and `sv_ratio`, `goof`. `refine_structure` warns for GooF above 3.

* **The per-parameter likelihood-ratio bound test** (`bound_test=True`, `BOUND_LR_CHI2` = 2.71): separates a parameter truly on a bound from one the data merely cannot determine. ESTABLISHED after a REFUTED first version (a pooled test; claim 97f96459b1e5 -> 38c79b5ee7fe, 4 lenses). Limits to state: a true rail with a chi2 drop of 3.2 is caught about 55 % of the time at 1 % noise (one-sided 5 % test), the family-wise false alarm is 6 / 14 / 16 % when 1 / 2 / 3 parameters truly sit on a bound. [`validation/RELEASE_READINESS.md`]
* **Formal esds are understated.** On L-alanine the pull rms was 2.36 with the formal sigma and 1.44 after rescaling by GooF (the baseline merge: 4.06 and 1.85). Quote `sigma_rescaled` or a bootstrap, and say which.

## 3. Outlier rejection decides GooF and can decide the positions

`outlier_sigma` defaults to `None`: every row counts. The verified L-alanine numbers used `outlier_sigma=5.0`, which removes rows WORST-FIRST on the internally studentized residual, one at a time, at most 10 rows, never below `n_par + 5` rows. What that did:

* New merge, aniso: 10 rows rejected, all 10 starts end at one point; GooF 1.64, R1 3.02 %, median 0.0014 A, max 0.0031 A from the reference. 8 of the 10 rejected rows have n >= 3 (up to 13): they are strong, well-measured reflections the model misfits (extinction is the guess), not bad singletons. The ten rejected rows are a DIFFERENT set from the baseline merge's (only (1 0 2) and (1 0 5) are shared).
* With no rejection the fit does not reach those values (median 0.041 A, GooF 10.9). Dropping ONLY the failed integration (1 0 2) by hand gives median 0.0013 A, max 0.0037 A, GooF 1.82, R1 3.05 %. Removing 1 to 16 rows worst-first keeps the median at 0.0013-0.0018 A and the max at 0.0029-0.0037 A; GooF falls from 1.82 to 1.59. So the POSITIONS do not depend on the cap of 10, the GooF does.
* The baseline merge was more sensitive (max 0.0124 A with one row removed; median 0.049 A with none).
* R1 3.02 % is summed over all reflections with F2 > 2 sigma including the 10 rejected rows (2.91 % on the kept rows). Say which.

[`results/a39_floor/CLAIM_tierC_cfgA.md` AMENDED WORDING; `results/a39/CLAIM_tierC.md`]

## 4. This is a LOCAL refinement; hand and origin come from the start; ten starts are one result

* **Start offset.** `refine.perturbed_crystal(crystal, rng, shift_A=0.05)` moves each site by 0.05 A along each of its three allowed directions with random sign, that is 0.087 A per atom. Starts converged up to about 0.43 A per atom (0.25 A per direction), partially at 0.52 A, and failed by 0.69 A; random starts with the right atom types did not converge (baseline merge: basin about 0.3 A, `results/a39/CLAIM_tierC.md`). The claim is "refines from a start within a few tenths of an angstrom", not "solves the structure".
* **Hand and origin.** Friedel-merged |F|^2 cannot fix the hand of a non-centrosymmetric structure: an inverted heavy-atom start with the reference hydrogens fixed stays inverted (4.3 A from the reference); with no hydrogens in the model both hands give the identical chi2 (`results/a39/CLAIM_tierC.md`). The start and the fixed hydrogens select the setting. The package has no anomalous dispersion and says nothing about absolute structure.
* **Ten starts at 0.05 A that all end at one point are ONE result**, not ten confirmations (agreement to 1e-9). Count n = 1 for the statistics.
* **Permutation null.** A refinement of the same table with F2, sigma and n permuted among the reflections must FAIL from the same starts (L-alanine: no convergence in 6 of 6 draws on the new table, R1 0.53-0.61, median 0.39-1.05 A). Run it; a method that also "recovers" the structure from scrambled data is telling you about the start.

## 5. Reading the result against a reference

Positions: compute the distance modulo the space-group operations and lattice translations (P 2_1 2_1 2_1: 4 operators), and state the origin/hand treatment; the minimum picks the identity when the start is the reference setting. `validation/real/a39_refine.py` does it, and the four-lens /verify reproduced the distances with independent symmetry-aware code.

* **The agreement is at the noise floor.** Gaussian noise on the refined Fc2 at the GooF-matched level predicts a median of 0.0013 A for the new table; the observed 0.0014 A is not better than the data allow. About 0.001 A of the offset is systematic (table error and/or the reference differing from the truth). The baseline merge carried about 0.003 A of systematic error (median 0.0035 A against 0.0020 A predicted by its noise).
* **Hydrogens set GooF and R1, not the heavy-atom positions.** No H: median 0.0027 A, max 0.0077 A, GooF 3.55, R1 6.2 %; H displaced 0.1 A at random: median 0.0017-0.0021 A, GooF about 1.9.
* **Do not compare GooF between merges whose error model differs.** The baseline table with the same error floor (g = 0.073) added to its sigmas gives GooF 1.22, R1 5.35 %. Compare R1 and positions.
* **ADPs are not validated.** They moved by about 30 % between merges (20-30 % below the published values for the baseline merge, 7-16 % above for the new one) and fail the rigid-bond test (0.0016 A^2 against 0.0004 A^2 for the reference). Quote coordinates.

**Model-free readout against an independent reflection file** (only when one exists; the CIF of the L-alanine reference embeds its own measured hkl file): `python validation/real/a39_ref_hkl.py <merged.csv> <cif>` merges the reference equivalents (mean) and compares ours (n >= 2) with ONE scale, the one that minimises R = sum|Fo2 - k Fref2| / sum Fo2 (L1). Fix this definition BEFORE reading it: on the same baseline table the least-squares scale gave 18.66 % and the L1 scale 14.15 %; the registered readout is the L1 scale. L-alanine: 14.15 % (baseline merge) -> 6.33 % (new merge); 100 strongest reference reflections 17.1 % -> 7.0 %; the d < 0.8 A shell does not improve. [`PREREG_a39_floor.md` Amendment 2; `CLAIM_floor.md`]

## 6. What a report must carry

n reflections used and rejected (and which), start model and offset, number of starts and the statement "n = 1", GooF and R1 with the rejection stated, coordinates with the sigma kind (formal, rescaled, bootstrap), the reference readout with its registered definition, the permutation null, the margin and polarization plane used, and the list of what is NOT claimed (hand, ADPs, other crystals). Status words as logged; claims about one crystal stay claims about one crystal.
