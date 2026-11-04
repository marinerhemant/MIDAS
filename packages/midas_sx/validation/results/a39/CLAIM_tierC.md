# Claim a1a8381d5796 (full text; the skill-log subject is truncated at 300 chars)

midas-sx, run on L-alanine at 100 K (Diamond I19-1, Pilatus 2M, Zenodo 10.5281/zenodo.11946282, omega runs 02/03/04 at phi 0/120/240) with this configuration:
polarization K = 0.990 with plane 0 deg; fixed-support integration with support margin 6; mean shell background; header wavelength 0.68890 A; detector tilts, wedge, beam centre and distance refined per run from the spots;
cell and space group P 2_1 2_1 2_1 taken from COD 1574527; hydrogens fixed at the COD positions; merge in Laue mmm with Friedel mates merged and inter-run scales (scale_groups),
refines the six non-hydrogen atoms (O001 O002 N003 C004 C005 C006) with anisotropic U against the merged F2 (793 unique reflections, 716 with n >= 2; 783 used) to
  median |delta position| 0.0035 A and max 0.0051 A from COD 1574527, R1 (I > 2 sigma) 5.96 %, GooF 2.19, in 10 of 10 starts displaced 0.05 A from the reference (all 10 end at the same minimum),
and the same refinement run on the same merged table with F2, sigmaF2 and n permuted among the reflections (one permutation) does not recover those positions
  (iso-U arm: 0 of 10 converged, R1 0.674, median |delta position| 0.95 A, max 1.71 A; the aniso-U arm of the null had not finished when this claim was registered).

Not claimed: that the registered tiers A or B passed (A is REFUTED, B1/B2 INCONCLUSIVE in PREREGISTER STEP 17 READING); that the absolute structure is determined; that 10 starts are independent; the iso-U arm confirms tier C (GooF 3.38 > 2.5).
Effect size that matters: positions 0.02 A (registered tier C bar), R1 8 %.

Files (Mac, relative to packages/midas_sx/validation): results/a39/out_p0/{refine.log, refine_tierC.csv, merged.csv, merge_reading.json, refine_scr0.log}, real/{a39_run.py, a39_merge.py, a39_refine.py}, PREREGISTER.md (STEP 17 and READING).
Data on copland (read-only for refuters): /gdata/dm/MPE/OrthrosJr/analysis/sharma_work/midas_sx_real/lalanine/{frames (CBF), 1574527.cif, out_p0/r02|r03|r04 (obs.pkl, U.npy, reduction.json, spots_blind.csv, mask.npy), out_p0/merged.csv, out_p0/merged_scr0.csv}, code/{midas_sx,midas_hkls,midas_defect,midas_stress}.
Run recipe on copland: cd to that directory; MIDAS_SX_PATH=$PWD/code/midas_sx /home/beams12/S1IDUSER/opt/envs/midas/bin/python -I code/midas_sx/validation/real/a39_refine.py lalanine/out_p0 lalanine/1574527.cif [merged csv name]  (about 1 h with the host idle; two arms x 10 starts).

## AMENDED WORDING after /verify (2026-10-23; all four lenses SURVIVES; the amendments are the refuters' wording fixes)
- The refinement applies the package's studentized worst-first outlier rejection (outlier_sigma = 5, hard cap 10 rows; 7 of the 10 are n = 1 reflections, e.g. (1 0 2) observed 14.7 vs calculated 13044). WITHOUT rejection the aniso fit stays near the start (median 0.049 A, GooF 11.8, R1 15.6 %); dropping only (1 0 2) keeps the median at 0.0035 A but gives max 0.0124 A and GooF 3.06. The quoted max 0.0051 A and GooF 2.19 need the full 10-row rejection.
- The start displacement is 0.05 A per free direction = 0.087 A per atom (median). The "same minimum" holds only inside a ball of roughly 0.3 A around the reference; a 0.43-0.52 A start does not converge (0.88 A, GooF 14.9), random starts with the right atom types do not converge (R1 ~0.56). This is a LOCAL REFINEMENT claim, not a structure solution.
- Hand and origin of P 2_1 2_1 2_1 are not determined by the data (an inverted start with the H fixed stays inverted at 4.3 A from COD, bit-identical chi2; with no H both hands give chi2 6082.7568). They come from the starting model and the fixed COD hydrogens.
- The 10 starts are ONE distinct result (n = 1 independent confirmation). Pulls: 4.06 formal, 1.85 GooF-rescaled (esds understated ~2x). Refined ADPs come out 20-30 % below the published ones (data stop near d = 0.66 A; the reference reaches 0.34 A).
- Supporting (lens results, provisional detail): model-free comparison of our merged F2 with the F2 embedded in the COD CIF (an independent measurement at 0.2489 A): R_F2 14.7 % with a scale only, no resolution trend; positions unchanged (median 0.0024-0.0036 A) when dropping H, dropping tth < 25 deg, dropping strong observations, dropping harmonic partners, or using margin 10; margin 1 and polarization plane 90 do not converge. Aniso nulls (shell and global permutations, 3 draws completed) all fail (median 0.54-1.05 A, R1 0.55-0.75).
- Still NOT established: tiers A and B (A REFUTED as registered, B1/B2 INCONCLUSIVE); the iso arm (GooF 3.38 > 2.5); any claim about the absolute structure; transfer to other data sets or beamlines.
