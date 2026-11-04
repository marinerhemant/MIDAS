# Package notes - properties of the CODE, not of any one analysis

Recorded because someone teaching from this doc set will hit them. Sources are named; the date of the facts is the 2026-10-23/28 working tree (nothing in `packages/midas_sx` was committed at that time).

## 1. Where code lives

| what | where | note |
|---|---|---|
| corrections (polarization, rotation Lorentz, anvil absorption), frames, B matrix, primary extinction | `midas_hkls.corrections`, `midas_hkls.frames`, `midas_hkls.laue_extinction_factor` | never re-implement them in `midas_sx`; `midas_hkls.lorentz_polarization` is the full-ring average and ignores the polarization fraction, per-spot work uses `corrections.polarization_factor` |
| symmetry, Lattice, SpaceGroup, blind lattice | `midas_hkls`, `midas_hkls.ab_initio.index_ab_initio` | `two_pi` defaults differ across the MIDAS chain: pass it every time |
| mask, polar background, 3-D spots, pixel -> q | `midas_defect` (`ingest.build_mask`, `geometry.pixel_to_qlab`, `qlab_to_qsample`) | `qlab_to_pixel` diverges for a strongly swung detector (phase 1) |
| orientation grid, Laue rotations | `midas_stress.orientation_sampling` | `midas_defect.rows` now wraps it |
| the independent simulator | `packages/midas_sx/validation/simulate.py` | never imported by the package; unit tests import it through a `sys.path` insert and need `gemmi` |
| the swung-detector geometry builder | `validation/real/a39_geom.py::swing_geometry` | a VALIDATION script, not package code |
| the model-free reference readout | `validation/real/a39_ref_hkl.py` | a script |
| the front door | `midas_sx/workflow.py` | `reduce_scan`, `merge_runs`, `refine_structure`, `reduce_twin90_scan`, `refine_twin90_structure`, `predicted_targets` |

## 2. Defaults that choose silently

| default | where | consequence |
|---|---|---|
| `PolarizationModel(fraction=0.99, plane_eta_deg=90.0)` | `reduce.py` | the wrong plane gave R_int 24.53 % instead of 15.67 % on I19-1 (plane 0 was right there) |
| `DacModel(mu_per_cm=0.0, half_opening_deg=None)` | `reduce.py` | no absorption and no opening cone |
| `refine(..., outlier_sigma=None)` | `refine.py` | no rejection; the verified numbers used 5.0 |
| `merge_observations(robust=False)` | `merge.py` | plain 1/sigma^2 mean (biased low); `reduce_scan(robust_merge=True)` and `merge_runs(robust=True)` are robust by default |
| `scale_groups` vs `scale_smooth` | `merge.py` | `merge_runs` defaults to the smooth scale; `scale_groups` is one scale per run |
| `background="mean"` | `integrate.py` | assumes a flat background; `"quadratic"` is available, A36 INCONCLUSIVE |
| `hkl_tol=0.15`, `crowd_px=12`, `min_tth_deg=2.0`, `max_factor_variation=0.02` | `reduce.reduce_spots` | weak noise blobs accepted as reflections; strongest reflections dropped as crowded |
| `choose_margin` sweep `(0, 1, 2, 3, 4, 6)` | `integrate.py` | may end without a plateau and only WARNS |
| `index_and_refine(refine_tilts=False, refine_wedge=False, refine_wavelength=False)` | `orient.py` | the real-scan geometry needs the first two |

## 3. Opt-in features and their status words

| feature | status |
|---|---|
| `refine_tilts`, `refine_wedge` | used on three real runs of one data set; unit-tested (`tests/test_wedge.py`, `tests/test_orient.py`); no independent verification |
| `refine_wavelength` | PROVISIONAL: recovers injected scales in simulation; real-data formal sigma 20-40 times too small; degenerate with the cell scale |
| `friedel_geometry` | INCONCLUSIVE on real data (registered rule met in 1 of 3 runs) |
| `integrate_predicted` | NO GAIN on the L-alanine floor; +6 % unique reflections; also fixed a real bug in `assign.calculated_crossing` (a target exactly on the crossing returned None), regression-tested |
| `margin=` | override of the swept margin |
| `background="quadratic"` | A36 INCONCLUSIVE |
| `scale_smooth`, robust merge, `merge_runs` | ESTABLISHED on one real data set (claim f55319a6e282) |
| `aniso_sites`, `fix_origin`/`fixed_atoms` | aniso ESTABLISHED in simulation at one orientation; on L-alanine the aniso arm met every registered bar; origin pin unit-tested |
| `refine_extinction` | PROVISIONAL (`t_b` effective, converges only from a start below ~20 um) |
| `refine_twin`, `reduce_twin90_scan`, `refine_twin90_structure` | see section 5 |
| `midas_sx.advice` | PROVISIONAL: unit-tested, first real-data run on L-alanine; validation on injected defects in progress; no `/verify` yet |

## 4. Not in the package (say so when asked)

Rotation about a TILTED axis (a phi scan of a kappa stage; run 01 of the L-alanine set was excluded for this); a detector distortion or panel-gap model (boxes touching a masked pixel are rejected); secondary extinction; anomalous dispersion (so no absolute structure); absorption for anything but the DAC path; refining the cell (it is held fixed); joint refinement of orientation and structure (orientation first, then fixed); occupancies; a command-line interface. (`packages/midas_sx/README.md` "Not implemented"; its last item, "any real-data test", is out of date: three real data sets were tried.)

## 5. Twins: what the package does and does not do

Declared 90-degree twin about c (`twin_matrix` default `((0,1,0),(-1,0,0),(0,0,1))`): twin fraction refined, ESTABLISHED with narrow scope (one signal pair, one orientation, one split size; detection 68 % at m = 0.1; the majority domain is an input). Split spots (a != b): PROVISIONAL with failed registered criteria; the fitted second-component weight is a nuisance weight, not a twin fraction; the twin law and the cell are inputs. In-plane displacements Uxx and Uyy are nearly degenerate under the law (correlation -0.91 at m = 0.3). `domains.find_domains` (in `midas_defect`) finds two domains from scratch if the CENTRING is declared right; it does not discover a twin law. An unknown twin law is NOT IN SCOPE.

## 6. Fixed defects you may meet in old numbers

Each was found on real or simulated data and fixed with a test; results produced before the fix may differ.

* `choose_margin` certified the reference margin as "converged" (itself); it no longer does and `build_profile` warns when there is no plateau.
* `match_setting` accepted a reflection (det < 0) because it is also orthogonal; it now requires det(U0) > 0.
* The polar-axis origin was not pinned when atoms were fixed; the pin is now always applied.
* The pooled likelihood-ratio bound test was REFUTED by two lenses; replaced by the per-parameter test plus a Poisson-quantile clip (claim 38c79b5ee7fe).
* The quadratic background clipped with a Gaussian bound, biasing low counts; it uses a Poisson quantile.
* `predict_positions` returned omega values wrapped by 360 deg for a scan through 360; it now returns the representative nearest the hint.
* `calculated_crossing` returned None for a root exactly on a grid node (7 of 46 targets in a simulated scene).
* `scale_smooth` crashed when no reflection had an equivalent; it now returns the identity.
* The Poisson-only weighted merge (default `merge_observations`) is still the low-level default; see section 2.

## 7. Practical

* The unit tests are slow (the full suite takes 30-40 minutes on the Mac); run individual files while developing.
* Mac: under 3 GB per process. Spot finding on a full Pilatus run (2.1 G voxels) belongs on copland.
* Install order for a fresh tree: `midas_stress` (>= 0.16.0), `midas_hkls` (the version that has `corrections`, >= 0.17.0), `midas_defect` (>= 0.9.1), then `midas_sx`.
* `validation/COMMIT_SCOPE_PROPOSAL.md` says what of `validation/` belongs in git.
