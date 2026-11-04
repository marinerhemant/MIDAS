# /verify of the three midas_hkls code claims (2026-10-21): full caveats (the skill-log notes were truncated to 300 characters)

All three: 4 lenses (physics, statistics, reproduction, artifact), 12/12 SURVIVES, consolidated ESTABLISHED by the mechanical rule. Refuter workdirs:
/Volumes/T5 EVO/verify_workdirs/Desktop/analysis/midas_sx/refute_{7054,4739,81fe}_{physics,statistics,reproduction,artifact}/

## 7054b7947ba7  lorentz_polarization = full-ring average (midas_hkls/intensity.py)
- The OLD form (1-K)+K cos^2 2theta is legitimate for a fixed scattering plane (monochromator/Bragg-Brentano, K = cos^2 2tm/(1+cos^2 2tm)); the claim is only that it is not the ring average. That use case has lost its K parameter in this function; the warning names midas_hkls.corrections.polarization_factor. Release notes should say so.
- Ring average independent of f, polarization direction and circular component: checked to 12 digits (Stokes density matrix) and 2e-16 (eta grid).
- "No caller passes K != 0.5": supported by grep of packages/, dev/, untracked trees, 17 worktrees (no hit). One latent route: cfg.vmap.polarization (midas_pipeline/config.py:465, default 0.5, no CLI flag, no setter in the repo). Scripts outside the repo are not searched.
- The old and new factors differ by < 1 % below about 8 deg 2theta (K=0 or 1), so low-angle rings would not have shown the old error.
- Outside the claim, to fix separately: packages/midas_2d/midas_2d/instrument.py:76-78 solid_angle_polarization still uses the one-direction (1-K)+K cos^2 2theta; stale docstrings in midas_hkls/intensity.py (module header lines 6 and 10, powder_intensity / intensity_from_crystal) still say "polarization fraction is preserved".
- The test only calls with K=0.5 except the warning test; K-independence through powder_intensity/intensity_from_crystal is not tested.

## 4739ca7e4ffb  mixed aniso/iso Debye-Waller (midas_hkls/structure_factor.py _debye_waller)
- Agreement with gemmi is 3e-7 on the claim cell (the 2e-4 in the test is a tolerance); worst 1.6e-5 over 10 structures x 4 aniso masks x ~350 reflections to s = 1.2 (cubic, tetragonal, hexagonal, trigonal, monoclinic, triclinic, centred, special positions).
- HEAD error is "about 1 %" only on the claim cell at low s (0.2-1.5 % on the six reflections); on other structures and at high s it reaches several 10 % and more.
- Gradient: the branch is chosen by sum|U_aniso| > 0, so there is no gradient at exactly U = 0 (an iso atom set to U11 = 1e-9 jumps to the anisotropic branch, F2 changes 13984 -> 13996). A refine start with B_iso = 0 on an aniso site would sit on that jump (not run through refine.py). HEAD already had this for the all-zero case.
- Not caused by the fix, but found: an aniso atom on a special position whose U breaks the site symmetry disagrees with gemmi (5e-4 to 8e-3) because midas_hkls averages U over coincident images; near-special positions are merged by gemmi at ~0.01 and by midas_hkls at 1e-4; float32 on a mixed cell raises a dtype error (also at HEAD).
- The test covers P1, tetragonal, diagonal U only; the non-orthogonal and symmetry-expanded checks are the refuters', not in the repo.

## 81fe013a9b1f  corrections.rotation_lorentz and P*T vs simulate.py
- "Independent" means independent IMPLEMENTATION. Shared inputs: mu table (midas_hkls.absorption via make_experiment), flat-slab / point-sample / hard-cone transmission model, horizontal polarization plane. P*T agreement (1e-5 to 1e-4) is therefore an algebra and convention check (it does catch omega-sign, polarization-plane and anvil-thickness-swap errors); it cannot test mu, thicknesses, opening angle or the model itself.
- The Lorentz Monte Carlo (time in an Ewald shell) is genuinely independent of 1/|e.(s0 x s)|: a brute-force sweep of a cubic cell gives median 1.000000 (worst 6e-5 at 5e-4 shell), tilted axis 6.8e-4, tilted beam 2.5e-3.
- The Lorentz spread is the binomial noise floor (z std 0.93-0.96): MAD spread 0.29 % (S1) and 0.36 % (S2, not 0.3 %). The median (1.0004 / 0.9996) is one Monte-Carlo realisation, good to about +-0.1 % (S2 offset -1.7 se at seed 2; seeds 7 and 11 give 1.00019 and 0.99993).
- Power: constant offsets of 0.5 % are caught in both samples, 0.1 % in S2 only; a tilt of the rotation axis about x is invisible to the median and caught by the spread from about 0.1-0.25 deg.
- The 1.5 deg margin is where the cone-edge blur from the rocking width begins (disagreement up to 30 % inside that margin, inf outside the cone); any threshold from 1.0 to 5 deg gives the same 3.47e-4. It depends on bandwidth_rel = 1e-3 and mosaic 0.10 deg. One spot (2,3,9) 3.7 deg outside the cone has simulator P*T = 4.3e-5 where corrections gives T = 0; not diagnosed.
- Scope: one tilt setup, 35 deg half-opening, d down to |e.(s0 x s)| = 0.012 (L up to 83); d between 1e-3 and 0.007 untested.
