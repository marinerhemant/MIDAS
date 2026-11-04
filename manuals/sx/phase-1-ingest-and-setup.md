# Phase 1 - frames, conventions, swung detector, blind lattice, setting, geometry

What the header does NOT tell you, and the order in which you find it out. Sources: `packages/midas_sx/midas_sx/{io_frames,conventions,setting,orient}.py`, `validation/PREREGISTER.md` STEP 15-17, `validation/real/a39_*.py`.

## 0. Survey before you run anything

Read the header of the first frame of each run and write down, with the file it came from: wavelength, distance, pixel size, beam centre, scan axis, start angle and step, exposure, count cutoff, detector 2theta (swing), polarization, chi/phi of each run, number of frames. Then list what you will NOT trust: the beam centre (4 px off on Rochelle salt), the distance (the Bruker header lists two), the image orientation, the sense of rotation, the polarization plane, any angle of a kappa stage. The Pilatus CBF header is parsed by `io_frames.pilatus_header_fields(text)`; the values it finds end up in `ScanHeader.extra`.

Do not infer the energy from file names; read the beamtime log. If a calibrant was measured, use it (see section 6).

## 1. Readers

```python
from midas_sx.io_frames import read_pilatus_cbf_scan, read_cbf_scan, read_sfrm_scan, list_sfrm_frames, read_sfrm_mask, ScanHeader
stack, hdr = read_pilatus_cbf_scan("frames/", prefix="run02_", keep_invalid=True)
```

* **Pilatus / imgCIF** (`read_pilatus_cbf_scan`, `read_cbf_scan`): negative pixels are flags (-1 gap, -2 bad pixel). With `keep_invalid=True` they stay in the stack (int32) so that `midas_defect.ingest.build_mask` sees them; without it they become 0 and are only counted in `ScanHeader.n_undefined`. Use `keep_invalid=True` for a Pilatus. Values above the range of `dtype` raise instead of wrapping. The pixel size is in the header as metres (0.000172 -> 0.172 mm).
* **Bruker SFRM** (`read_sfrm_scan`, `list_sfrm_frames`, `read_sfrm_mask`): the SFRM header carries two DISTANC values and the second one is the distance used; mask frames named `_xa_` are excluded from the scan list and read with `read_sfrm_mask`.
* A 850 x 1679 x 1475 Pilatus run is 2.1 G voxels: do not hold it in float64 on the Mac; use float32 and copland (about 20 minutes per run for spot finding).
* `ScanHeader` fields you will use: `shape`, `wavelength_A`, `distance_mm`, `pixel_mm`, `angle_start`, `angle_step`, `angle_steps_regular` (False means the step is not constant: stop), `exposure_s`, `n_undefined`, `extra`.

## 2. Image orientation and omega sense: choose by the indexing score

The array can sit in the MIDAS frame in 8 ways (the dihedral group of the square) and the scan can turn either way: 16 candidates, `conventions.candidate_conventions()`; `Convention(dihedral, omega_sign)`; `transform_frame(stack, dihedral)` applies one. The choice is made by blind indexing, never by the header. `rank_conventions(scores, min_margin=0.2)` returns `(best, runner_up_score, resolved)` and reports a tie as unresolved, it never breaks it.

The mirror-equivalent conventions tie EXACTLY and are equivalent for |F|^2 (four of them on L-alanine); resolve at the class level and take the first. A margin below 0.2 between non-equivalent classes is a halt.

## 3. A swung detector is a geometry class of its own

`Detector_2theta` non-zero: the beam centre in the header (`Beam_xy`) is the perpendicular foot of the beam on the detector, not the beam position. The beam hits D tan(alpha) away along the swing axis (58 mm = 339 px at D = 160 mm, alpha = 20 deg); the effective distance is D / cos(alpha); the plane is tilted by +-alpha about the lab axis the swing direction maps to after the dihedral transform (tz for the columns, ty for the rows). The sign of the tilt and the sense of the offset are hypotheses: the A39 convention scan tried (no swing) + (axis in {cols, rows} x tilt sign x offset sign) = 9 hypotheses for each of the 16 conventions, 144 blind indexings of ~1 s each (`validation/real/a39_step2_conventions.py`), and the winner was `identity | omega-`, swing along cols, tz -1, offset -1.

The builder is a VALIDATION script, not package code: `validation/real/a39_geom.py::swing_geometry(H, conv, axis, tz_sign, off_sign)`. If you need it twice, move it into the package with a test.

## 4. Blind lattice, from the first ~20 degrees

```python
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
spots, mask, info = find_spots(raw, geom0, threshold=3.0, low_count_threshold=0.0)    # Pilatus: background ~ 0
st = spots[spots.frame < 100].sort_values("integrated", ascending=False).head(600)
g = qlab_to_qsample(pixel_to_qlab(st.row, st.col, geom0, device="cpu", dtype=torch.float64), omega_rad, 0.0).numpy()
res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=30)
```

* **Use the first ~100 frames (20 deg), not the whole run.** With the header geometry the rotation-axis tilt and the detector tilts have not been refined yet, and over the whole run the blind indexer found no lattice (best 83/600) while the first 99 frames found the right cell at once (277/600). [STEP 17 Amendment 1]
* **The indexed fraction has a ceiling.** The window holds only the reflections that cross inside it (about 270-290 for L-alanine at 20 deg), so 600 blobs cannot index above ~48 %; most of the rest are weak compact blobs at low 2theta (the strong unexplained ones are ring-like and consistent with ice Ih lines; that is a consistency check, not an identification). The fraction is not a calibration measure: it was 288/276/265 with the header geometry and 285/277/266 with the refined one. The cell you get must be within a stated tolerance of the reference (L-alanine: within 0.5 % on every axis) and the orientation refinement must converge (`OrientationResult.converged`, rms <= 2 px and <= 1 frame by default).
* Pilatus: the explicit blob threshold 3 counts is needed; the MAD-based default collapses to 0 when the background is ~0. `low_count_threshold=0.0`.

## 5. Match the found lattice to the reference setting

`setting.match_setting(UB, lattice_ref, max_orthogonality_error=0.03)` returns `(U0, P, err)`: the signed axis permutation with det(U0) > 0 that makes the found basis the reference one. A reflection is also orthogonal, so the determinant condition is what rejects a mirror; err around 0.1 or more means a wrong permutation or a lattice that is not the reference, and above the limit it raises `ValueError`. For an orthorhombic mmm lattice the mirror-equivalent permutations give the same intensities (the first best one is returned); for monoclinic and triclinic lattices only proper settings are considered and the mirror image is a different setting.

## 6. Refine orientation and geometry from the measured spots

`reduce_scan` does it first (`orient.index_and_refine`): joint least squares of U, beam centre and distance, hkl re-assigned as it converges. On a real scan also ask for:

* `refine_tilts=True`: the three detector tilts (a swung detector's starting tilt is the swing angle).
* `refine_wedge=True`: the rotation-axis wedge (MIDAS convention `q_lab = R_y(-W) R_z(omega) q_sample`). L-alanine needed +0.41 / +0.31 / +0.29 deg on the three runs; without it the robust rms was 1.52 px against 0.78 px with wedge plus tilts [STEP 17 Amendment 1]. A wedge or tilt above ~0.1 deg means the axis or the detector is not where the header says: say so in the report.
* `refine_wavelength=True` (opt-in, PROVISIONAL): one relative scale of the wavelength, `lambda = lambda0 (1 + eps)`. With the cell fixed this is EXACTLY degenerate with an isotropic cell scale of -eps, so it absorbs any error of the reference cell and is NOT an energy calibration; its formal sigma is 20-40 times smaller than the run-to-run spread (eps 1.5e-3 to 2.5e-3 on L-alanine, spread about 5e-4). On L-alanine it removed a one-sided omega offset of low-angle spots (INCONCLUSIVE as registered, post hoc explanation: header wavelength and reference cell disagree by ~0.19 %; `validation/real/PREREG_lowangle_dframe.md`). The only thing that separates wavelength from cell is a calibrant pattern (CeO2, LaB6) at the same energy and distance, or a trusted cell AND a trusted energy.
* Friedel pairs (`midas_sx.friedel_geometry`, INCONCLUSIVE on real data): an estimate of tx and the wedge that does not use U. It needs pairs inside a run: a 170 deg scan has few (35 / 57 / 47 on the three L-alanine runs, about 1800 of 3120 signed hkl have their mate in ANOTHER run, which needs the runs' relative rotation); a scan of 180 deg or more gives them all. Agreement with the orientation fit was within 0.035 deg (`validation/real/PREREG_friedel_geometry.md`).

`OrientationNotConverged` is raised when the gates fail; `allow_unconverged=True` continues with a warning, which is for diagnosis, not for results.

## 7. ScanSetup

```python
ScanSetup(wavelength_A, dist_mm, pixel_mm, centre_rc, omega_start, omega_step,
          dac=DacModel(...), pol=PolarizationModel(fraction, plane_eta_deg), wedge_deg=0.0, tilts_deg=(0.0, 0.0, 0.0))
```

`omega_start` is the START EDGE of frame 0, with the omega sign of the chosen convention applied. `centre_rc` is (row, column) of the BEAM in the transformed array (for a swung detector: the foot plus the offset of section 3). `PolarizationModel.plane_eta_deg` defaults to 90, `DacModel` defaults to no absorption: both are inputs you must set (phase 2).
