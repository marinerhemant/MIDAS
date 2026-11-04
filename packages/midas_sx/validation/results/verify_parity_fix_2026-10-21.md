# /verify of claim 927641617f34 (midas_defect.rows.allowed_multiple_parity from the centring translations), 2026-10-21: ESTABLISHED

Lenses: physics SURVIVES, statistics SURVIVES, reproduction SURVIVES, artifact SURVIVES. Workdirs .../verify_workdirs/Desktop/analysis/midas_sx/refute_9276_*/.

## What holds
- Derivation: reflections on a row n*G_step; allowed n form a subgroup of the integers (multiples of one n0); smallest n is the right quantity. Textbook rules for I, C, A, F, R (obverse hex) vs the function: 0 mismatches over all 230 groups x every step in [-4,4]^3 (167,440 cases); brute force from gemmi's centring translations over [-2,2]^3 for 139, 87, 140, 71, 69, 65, 166, 123: 0 mismatches. n never exceeds 3 (range 1..6 is enough); B centring is unreachable by number.
- No regression for the groups that worked: old == new for SG 139 and 69 on 31,250 steps (-12..12), 10,000 random steps up to +-1e6; stored A31 SG139 results identical before/after for all 25 seeds. Whole midas_defect suite 1228 passed, 38 skipped, 7 xfailed, 0 failed; test_rows 75 passed.
- find_domains: reproduced on 10 seeds (25..34) with the OLD rows.py swapped in: SG87 2/10 old vs 10/10 new, SG139 10/10 both, SG123 0/10 both; all 30 new records equal the saved ones. The raw rows of the five fixed groups are field for field identical to SG139 (one measurement repeated, not five independent tests). 95 % interval for 25/25 is 0.86-1.0 and conditional on ONE crystal / orientation pair (25 noise draws of fixed signals).
- Before the fix the failure was loud (0 domains in 16 of 25 seeds; the 9 non-empty results were all correct), after: identical to SG139.

## Limits (all outside the claim as worded)
- Centring only: screws and glides are ignored. Full-group n0 differs from the centring-only value for 2,584 of 22,540 primitive steps over 155 groups; e.g. SG 141 (0,0,1): 4 against 2; SG 140 (0,1,1): 2 against 1; SG 76 (0,0,1): 4 against 1. The old code returned 1 everywhere, so this is an incomplete improvement, not a regression. For SG 139 the centring rule equals the full group on every step tested. The docstring (rows.py:69) said 'allowed reflection': now says centring-allowed.
- A primitive lattice declared for an I-centred crystal is unchanged by the fix: 2/25 correct and 6 of 25 return a wrong list silently (seeds 26, 30, 38, 41, 44, 47; domains 20-56 deg off); the rest return nothing.
- Other `== (139,)` gates remain: rows.py:1301 orientation_grid (order 8 for 139 else 1: a full-SO(3) grid, about 8x costlier, not wrong output) and index_by_grid uses the 422 operators for every group.
- Tests: no end-to-end find_domains test with a non-139 I-centred group (only the manual run_fd_sg.py); A and B centring untested. Cost +5 % on find_lattice_rows (7 us per call for SG139, 33 us for SG166); invalid group numbers now raise ValueError (the old code returned 1 silently); a non-integer step is truncated.
