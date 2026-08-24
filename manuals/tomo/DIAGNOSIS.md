# Tomography-for-diffraction diagnosis reference

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches
to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here — it turns the report into a machine for
confirming whatever its author already believed.

Scope: a tomogram used as a **sample shape** for an FF/PF/NF experiment — the illuminated
volume that replaces `V_gauge`, and the path length an absorption correction needs. Not
tomographic image quality for its own sake.

---

## Grain sizes carry a canned constant, not a measured volume

symptom: scale.inflated

**Test.** Read `Hbeam`, `Rsample` and `Vsample` out of the run's own parameter file with
`GaugeVolume.from_param_file`, and compare `Σ(4/3 π R³)` over `Grains.csv` against
`V_gauge`. On the FF reference run (`ff_refiner_prepost/result/LayerNr_1`, 6112 grains)
there is no `Vsample` line, so `V_gauge = Hbeam·π·Rsample² = 2000·π·2000² = 2.513e10 µm³`
and the grains sum to **6.5 %** of it.

The other answer: if `Vsample` was set to a measured volume for this specimen, the scale is
someone's measurement and this entry does not apply — check `GaugeVolume.is_template_default`,
which flags the `Rsample 1000 / Hbeam 1000 / Vsample 50000000` written by the
`midas_calibrate_v2` templates.

**Cause.** `radius/core.py:172` builds the gauge volume from `Hbeam` and `Rsample`, which
are **search bounds** by a hard project rule and are never the specimen. Absolute grain
size is therefore a template constant. Relative sizes within a ring are unaffected.

**Lever.** `midas_transforms.radius.shape_correction.correct_grain_volumes` with a
`SampleShape`. Volume scales by `V_illum / V_gauge`, radius by its cube root. Emit
`GrainRadius_shape` **alongside** `GrainRadius`, never over it.

---

## The correction was applied to the numerator only

symptom: scale.inflated

**Test.** Compute `⟨f⟩_r` over each ring after the correction. It must be exactly 1. If it
is not, the per-spot factor is inflating every volume by `⟨1/A⟩` — about 1.6× in volume and
17 % in radius at μD ≈ 0.5, uniform across the dataset and in the direction everyone expects.

The discriminating check: apply a **constant** correction. Radii must come back
bit-identical. If they move, the normalisation is missing.

**Cause.** `powder_int` (`radius/core.py:153-160`) is a sum of *observed* spot intensities,
so it already carries every effect the numerator does. Only the *spread* of a correction
survives the ratio.

**Lever.** `normalise_per_ring` before use. It is not optional and it is not a refinement.

---

## The sample mask is mirrored

symptom: systematic.per_object

**Test.** `meta_null(check, shape, ...)` — rerun the registration check on
`shape.mirrored()`. The statistic must degrade by a stated margin. If it does not, the
check has **no power over handedness** and its PASS is not evidence about `in_plane`.

The other answer: a genuinely chiral cross-section degrades under the mirror, and then a
PASS means something. A centred cylinder or a centred box does not — verified in
`test_registration_checks.py`, where the meta-null returns `NO_POWER` on a box and `PASS`
on an L-shape.

**Cause.** `in_plane` is one of eight signed axis permutations and nothing in any
reconstruction format records it. The wrong choice mirrors the sample: the reconstruction
is still sharp, the mask still smooth, and path lengths acquire a spatial gradient that
reads as real microstructure.

**Lever.** Establish the handedness per experiment from a known asymmetry, not from the
reconstruction. Until then say the registration is unverified — `SampleShape.provenance`
carries `registration: NOT verified` from every reader for exactly this reason.

---

## The sinogram registration check could not have failed

symptom: null.not_cleared

**Test.** `sinogram_check` computes the predicted modulation `std/mean` of
`V_illum(ω)` **before** comparing anything, and returns `NO_POWER` below `min_modulation`
(default 0.02). A cylinder on the rotation axis has a lit volume that does not vary with ω
at all, so any measured curve "agrees" with the flat prediction and a χ² sails through.

The other answer: an elongated or off-axis shape modulates, and then the correlation is
informative — the three failure modes have three signatures (handedness flips the odd
Fourier components; an axis offset injects a one-cycle `cos ω` whose amplitude *measures*
the offset; a wrong pixel size scales the curve without moving its phase).

**Cause.** The check was run on a shape it has no power over.

**Lever.** Use a check with power (V2 held-out containment, or V4 NF grain-map Dice), or
report the registration as unverified. `NO_POWER` is not a pass, and `CheckResult.__bool__`
returns False for it.

---

## The mask includes reconstruction padding

symptom: scale.inflated

**Test.** `recon_xdim = next_power_of_2(det_xdim)` (`midas_tomo/config.py:198`), so a
1365-wide detector reconstructs onto a 2048 grid and a third of every slice is padding no
ray ever sampled. Count occupied voxels outside the disc of radius `det_xdim/2` about the
rotation axis. It must be exactly zero — the readers in
`midas_transforms.geometry.tomo` raise otherwise.

The other answer: if the occupancy is inside the disc, the mask is at least reconstructible
and the volume error is elsewhere — check the threshold with `threshold_sensitivity`.

**Cause.** Parallel-beam FBP can only reconstruct the inscribed disc. Values outside it are
ringing, and thresholding picks them up.

**Lever.** Raise the threshold, or correct `rot_axis_ix/iy`. Pass `det_xdim` to the reader
so the check is the strong one rather than the grid's own inscribed circle.

---

## The reported volume is whatever threshold was chosen

symptom: uncertainty.miscalibrated

**Test.** `threshold_sensitivity(volume, thresholds, voxel_volume_um3=…)`. A high-contrast
sample is stationary — the volume barely moves across a plausible range. A smooth
phase-contrast blob is not: a Gaussian gives >30 % in reported *radius* over a 0.2–0.8
sweep.

The other answer: `stationary: True` with a fractional spread under a few percent means the
threshold is not doing the work and a single value is defensible.

**Cause.** The threshold multiplies `V_illum` directly and there is no principled value for
it. On a phase-contrast reconstruction there may be no plateau at all, because the interior
has no contrast to threshold.

**Lever.** Phase retrieval before thresholding. Report the band, not the midpoint. If the
volume is not stationary, the mask is not usable as a volume estimate — say so.
