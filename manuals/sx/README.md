# sx - single-crystal rotation-method structure refinement with midas-sx

The spine. Keep this loaded; open the phase files as you reach them.

**Scope.** Rotation frames of ONE crystal (monochromatic beam, area detector, discrete Bragg spots,
optionally in a diamond-anvil cell) where the cell and space group are known or come from a
reference CIF, and you want a merged F2 table, a `.hkl`, and/or a refined structure. Out: an unknown
cell (`solve-cell`), many grains (`ff-hedm`), diffuse scattering (`defect`), powder rings (`xrd-ct`),
white-beam Laue (`laue`). A twinned crystal is in scope only with the twin law and the cell DECLARED
(`PACKAGE_NOTES.md`); finding the law is not.

## Status in one paragraph

Alpha (`midas-sx` 0.0.1, `packages/midas_sx/README.md`). The structure-quality claims rest on an
independent forward simulator and on public real data. On real data: Rochelle salt (A37) INCONCLUSIVE,
thiourea (A38) INCONCLUSIVE overall, L-alanine (A39, Diamond I19-1 Pilatus, three runs) tier A REFUTED
as registered and tiers B1/B2 INCONCLUSIVE for the baseline processing (`validation/PREREGISTER.md`
STEP 15, 16, 17 READING). After the A39-FLOOR campaign (`validation/real/PREREG_a39_floor.md`) two
claims were four-lens verified and are ESTABLISHED on that one crystal and those three runs: the
processing fix (smooth per-run scale + robust merge: R_int 15.67 -> 6.77 %, model-free R_F2 against an
independent reflection file 14.15 -> 6.33 %; claim f55319a6e282,
`validation/results/a39_floor/CLAIM_floor.md`) and a LOCAL refinement of the six non-H atoms (median
0.0014 A from the reference, R1 3.02 %, GooF 1.64; claim 2f5c452c9687,
`validation/results/a39_floor/CLAIM_tierC_cfgA.md`; the baseline-merge version is claim
a1a8381d5796, `validation/results/a39/CLAIM_tierC.md`). Wavelength refinement, Friedel-pair geometry,
predicted-position integration and the advice module are PROVISIONAL, INCONCLUSIVE, NO GAIN and
unvalidated respectively; `ENVELOPE.md` lists all of it, with the claim ids; `ADVICE.md` explains the advice report.

## Order of operations, and why it is this order

| # | phase | file | why it cannot move |
|---|---|---|---|
| 0 | survey: what does the header carry and what does it NOT | `phase-1-ingest-and-setup.md` | beam centre, distance, image orientation, omega sense, polarization plane, swing are routinely wrong or absent |
| 1 | read frames, resolve orientation + omega sense, swing class | `phase-1-ingest-and-setup.md` | the 16 conventions are chosen by the indexing score; a wrong class does not index |
| 2 | blind lattice from the first ~20 deg, match to the reference setting | `phase-1-ingest-and-setup.md` | the whole run indexes badly before the geometry is refined (STEP 17, Amendment 1) |
| 3 | spots, then orientation + geometry (centre, distance, tilts, wedge) from them | `phase-1` + `phase-2` | a cell/geometry error enters the Lorentz factor at the Ewald crossing |
| 4 | integrate on a fixed support, correct at the CALCULATED crossing | `phase-2-reduce-and-merge.md` | the corrections belong to the calculated crossing, not the centroid |
| 5 | scale the runs (smooth, per run) and merge robustly | `phase-2-reduce-and-merge.md` | a Poisson-weighted plain mean is biased low and dominated by a failed integration |
| 6 | read the floor: counting-limited or systematic? | `phase-2-reduce-and-merge.md` | more exposure only helps if the floor is counting |
| 7 | refine (local), read the gates | `phase-3-refine.md` | start model decides hand/origin; rejection decides GooF |
| 8 | reference readout (if a reference exists) and the report | `phase-3-refine.md` | the readout definition is fixed BEFORE you look |

## Install gate

See `.claude/skills/sx/SKILL.md`: `midas-sx` 0.0.1 needs `midas-hkls>=0.17.0` and `midas-defect>=0.9.1`
(`packages/midas_sx/pyproject.toml`); optional `fabio`, `gemmi`.

## The calling contracts - copy these, do not reconstruct them

```python
import numpy as np, torch
from midas_hkls import Crystal, Lattice, SpaceGroup, Atom
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.io_frames import read_pilatus_cbf_scan            # also read_cbf_scan, read_sfrm_scan
from midas_sx.conventions import candidate_conventions, transform_frame, rank_conventions
from midas_sx.setting import match_setting
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.workflow import ScanSetup, reduce_scan, merge_runs, refine_structure
from midas_sx.refine import RefineModel, refine, perturbed_crystal

# 1. frames. keep_invalid=True keeps the -1/-2 gap and bad-pixel flags so the mask sees them
stack, hdr = read_pilatus_cbf_scan("frames/", prefix="run02_", keep_invalid=True)   # hdr: ScanHeader
raw = transform_frame(stack, conv.dihedral).astype(np.float32)                      # conv chosen by the indexing score

# 2. blind lattice from the strongest ~600 blobs of the FIRST ~100 frames (not the whole run)
spots, mask, info = find_spots(raw, geom0, threshold=3.0, low_count_threshold=0.0)  # Pilatus: counts ~0 background
q = pixel_to_qlab(rows, cols, geom0, device="cpu", dtype=torch.float64)
g = qlab_to_qsample(q, torch.as_tensor(omega_rad), 0.0).numpy()
res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=30)
U0, P, err = match_setting(np.asarray(res.UB), crystal.lattice)                      # requires det(U0) > 0

# 3. reduce one run: polarization plane is an INPUT (default 90), the geometry is refined from the spots
setup = ScanSetup(wavelength_A, dist_mm, pixel_mm, centre_rc, omega_start, omega_step,
                  pol=PolarizationModel(fraction=0.99, plane_eta_deg=0.0), tilts_deg=(0.0, 0.0, -20.0))
red = reduce_scan(raw, crystal, U0, setup, refine_tilts=True, refine_wedge=True,
                  find_kwargs=dict(threshold=3.0, low_count_threshold=0.0))          # red.obs, red.merged, red.orientation, red.geom

# 4. several runs: smooth per-run scale + robust merge are the DEFAULTS of merge_runs
mr = merge_runs({"r02": red02, "r03": red03, "r04": red04}, crystal.space_group)    # mr.merged, mr.obs, mr.scales, mr.r_int

# 5. refine. outlier_sigma defaults to None (no rejection): the verified numbers used 5.0
model = RefineModel(start_crystal, fixed_atoms=("H1",), aniso_sites=("C1", "N1", "O1"))
fit = refine(model, hkl, F2, sigma, outlier_sigma=5.0)                               # or refine_structure(crystal, mr.merged, model_options={...}, outlier_sigma=5.0)
fit.converged, fit.goof, fit.R1, fit.at_bound, fit.null_params, fit.rejected
```

## Things to know before you start

Each cost a real retraction or a changed reading on the three real data sets or on the simulator.
Sources are the files named in brackets (all under `packages/midas_sx/` unless said).

1. **The geometry of a real scan is not in its header.** Rochelle salt's header beam centre was 4 px
   off; the Bruker thiourea header lists two distances; image orientation and omega sense come from the
   indexing score, and four mirror-equivalent conventions tie exactly and are equivalent for |F|^2.
   [`README.md` "Real frames"; `validation/PREREGISTER.md` STEP 15, 16]
2. **A swung detector: `Beam_xy` is the perpendicular foot, not the beam.** For the 20 deg swing of
   I19-1 the beam hits D tan(20 deg) = 58 mm = 339 px away along the swing axis, and Lsd = D / cos(20 deg);
   the plane is tilted by +-alpha about the lab axis the swing direction maps to. The helper that builds
   this is a validation script (`validation/real/a39_geom.py::swing_geometry`), NOT package code.
   [`validation/PREREGISTER.md` STEP 17 design choice 1]
3. **Blind-index from the first ~20 deg, not the whole run.** On run 02 the blind scan on the strongest
   600 blobs of the whole run found no lattice (best 83/600) while the first 99 frames found the right
   cell at once (277/600, 0.46 % from the reference). [STEP 17 Amendment 1]
4. **The rotation axis is tilted (the MIDAS `wedge`).** The orientation crept by 0.66 deg over 170 deg of
   rotation; robust rms went 1.52 px (U, centre, distance only) -> 1.05 px (+ wedge +0.285 deg) -> 0.88
   px (+ sideways tilt) -> 0.78 px (+ detector tilts). Refined wedge on the three runs: +0.41 / +0.31 /
   +0.29 deg. Use `refine_tilts=True, refine_wedge=True` on a real scan. [STEP 17 Amendment 1;
   `validation/results/a39/out_p0/r0N/reduction.json`]
5. **`midas_defect.qlab_to_pixel` diverges for a strongly swung detector** (79 of 108 test rays at 20
   deg); `predict_positions` falls back to a damped Newton solver (`orient._newton_pixels`). Nothing to
   do, but do not call `qlab_to_pixel` yourself on such a geometry. [STEP 17 Amendment 1]
6. **A low blind-indexed fraction is not a calibration failure.** The first 100 frames of an L-alanine
   run hold only about 270-290 predicted reflections, so 600 blobs cannot index above ~48 %; blind
   indexing gave 288/276/265 of 600 with the header geometry and 285/277/266 with the refined one.
   Registered tier A (>= 95 %) was REFUTED as worded for exactly this reason. [STEP 17 READING, addendum]
7. **The polarization plane is an input and the default is 90.** `PolarizationModel(plane_eta_deg=90.0)`
   is the default; for I19-1 plane 0 was right. The registered rule is "the plane with the lower merged
   R_int": plane 0 gave 15.67 %, plane 90 gave 24.53 %. [`midas_sx/reduce.py`; STEP 17 READING]
8. **Check a smooth multiplicative drift of each run before you trust a merge.** Systematic floor in F2
   14.2 % against 2.7 % from counting; after a per-run smooth scale (8 knots over the frame index) and a
   robust merge: R_int 15.67 -> 6.77 % (6.88 % with the scale fitted on half the reflections), R_split
   21.4 -> 10.1 %, model-free R_F2 14.15 -> 6.33 %. Margin, count-rate exclusion, and where the Lorentz
   factor is evaluated were tested and are NOT the cause. The fitted scale swings by +-20 % over 80-160
   deg and differs between runs; its physical cause is NOT established (crystal centring or illumination
   is the guess). [`validation/real/PREREG_a39_floor.md`; `results/a39_floor/CLAIM_floor.md`]
9. **A Poisson-weighted plain mean is biased low and dominated by a failed integration.** Reflection
   (0 0 8): observations 4015, 4183, 2825 and -1.2 merged to 834.6 +- 793 (the three good ones average
   ~3.7e3). `merge_observations(robust=True)` (error model, leave-one-out rejection, never with n = 2)
   is the default in `merge_runs` and in `reduce_scan`. [`PREREG_a39_floor.md` Amendment 1]
10. **Wavelength and an isotropic cell scale are exactly degenerate when the cell is fixed.** A header
    wavelength versus reference cell disagreement of 0.19 % removed a one-sided omega offset; the data
    cannot say which of the two is off. `refine_wavelength=True` exists, its formal sigma is 20-40 times
    smaller than the run-to-run spread (about 5e-4), and it is NOT an energy calibration. A calibrant
    pattern at the same energy and distance is the only way to separate them.
    [`validation/real/PREREG_lowangle_dframe.md` READING; `README.md`]
11. **Fix the reference-readout definition before you read it.** On the same L-alanine table R_F2
    against the independent reflection file was 18.66 % with a least-squares scale and 14.15 % with the
    scale that minimises R (L1); the registered readout is the L1 scale. [`PREREG_a39_floor.md` Amendment 2;
    `validation/real/a39_ref_hkl.py`]
12. **A sigma-weighted scale-and-B fit to a merged table is fragile.** The same A39 merge gave R_F2 55.3 %
    (sigma-weighted, B = -1.63 A^2) and 19.0 % (unweighted, B = +0.78 A^2), because a few reflections
    with a Poisson-only sigma dominate the weighted fit. Registered tier B2 was INCONCLUSIVE for this
    reason. [STEP 17 READING]
13. **`refine` rejects nothing by default, and the verified numbers need the rejection.** The
    studentized worst-first rule (`outlier_sigma=5.0`, at most 10 rows) removes strong, well-measured
    reflections the model misfits (8 of the 10 in the new merge have n >= 3). Without it the fit does not
    reach the quoted values (median 0.041 A, GooF 10.9); with only (1 0 2) dropped by hand it gives
    median 0.0013 A, max 0.0037 A, GooF 1.82. GooF 1.64 needs the full 10 rows.
    [`results/a39_floor/CLAIM_tierC_cfgA.md` AMENDED WORDING]
14. **The refinement is local; hand and origin come from the start model.** The start offset was 0.05 A
    per free direction (0.087 A per atom); starts converge up to about 0.43 A per atom and fail by 0.69 A;
    an inverted heavy-atom start with the reference hydrogens fixed stays inverted; Friedel-merged F2
    cannot fix the hand. Ten starts that end at one point are ONE result. [same file]
15. **Formal esds are understated and the ADPs are not validated.** Pull rms 2.36 formal, 1.44 after
    GooF rescaling; refined ADPs sit 7-16 % above the published ones for the new merge and 20-30 % below
    for the baseline merge, and fail the rigid-bond test (0.0016 A^2 against 0.0004 A^2 for the
    reference). Quote positions, not ADPs. [`results/a39_floor/CLAIM_tierC_cfgA.md`;
    `results/a39_floor/refine_cfgA.log`]
16. **The margin sweep may never plateau.** On L-alanine the median I(m)/I(m=6) was 0.76, 0.81, 0.86,
    0.90, 0.93, 1.00 for m = 0, 1, 2, 3, 4, 6 and `build_profile` warns. The floor f was 12.2 / 14.2 /
    15.8 % at margin 1 / 6 / 10, so a larger margin did not lower it; margin 6 stays. The warning is a
    `RuntimeWarning`, easy to miss. [STEP 17 READING; `PREREG_a39_floor.md` Amendment 1]
17. **The reduction is blob-driven.** A predicted reflection with no blob is never integrated, and a weak
    noise blob inside `hkl_tol` = 0.15 is integrated as that hkl ((1 0 2): I = -3 +- 284 where about 72000
    counts were expected). A tighter `hkl_tol` (0.05 instead of 0.15) improved R_int by about 1.4 points on the
    blob-driven path; `integrate_predicted=True` gave NO GAIN on the floor. The default is unchanged.
    [`PREREG_a39_floor.md` Amendment 1; `README.md` caveats]
18. **`DacModel` defaults to NO absorption** (`mu_per_cm = 0.0`, no opening cone). A DAC scan needs the
    anvil model passed explicitly. [`midas_sx/reduce.py`]

## Hard rules

* **Name the status of every number as the log recorded it.** ESTABLISHED / PROVISIONAL /
  INCONCLUSIVE / REFUTED; never promote one in a summary.
* **Positions, not ADPs.** And a position result is a local refinement from a stated start.
* **Compare R1 and positions between merges, not GooF** when the error model changed (the same table
  with the same error floor added to its sigmas gave GooF 1.22 against 2.19). [`CLAIM_tierC_cfgA.md`]
* **Report n actually used, rows rejected, and the start offset** with any refined parameter.
* **Preregister the reading before you look at the reference**, and say what was seen before.
* **One crystal is one crystal.** The numbers above are L-alanine, three runs, one beamline.

## Halt conditions

Stop and ask when: no convention indexes; `OrientationNotConverged`; a parameter is `at_bound` or in
`null_params`; GooF above 3; the sigma-weighted and unweighted R_F2 disagree by more than a factor of
two; two files give one quantity two values; the answer changes with the polarization plane, margin, or
start. Name the check and the number.

## Index

| file | holds |
|---|---|
| `phase-1-ingest-and-setup.md` | readers, conventions, swung detector, blind lattice, setting, geometry refinement |
| `phase-2-reduce-and-merge.md` | spots, integration, corrections, polarization plane, run scaling, robust merge, the floor |
| `phase-3-refine.md` | refinement, rejection, local basin, hand, multiple starts, the reference readout |
| `PACKAGE_NOTES.md` | where code lives, traps, opt-in and unverified features |
| `ENVELOPE.md` | every claim with its status word and claim id; what the data cannot determine |
| `ADVICE.md` | how to read an `AdviceReport`, what each check does and how far it was tested |
