# pf-HEDM diagnosis reference

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches to
a symptom the generic diagnostics detect. Keyed by *symptom*, not by step — the step that
produced a symptom is rarely the step you are on.

**Every entry carries a test that can come back the other way.** Before re-investigating,
read [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) §5 — several attractive hypotheses are recorded
there as refuted.

---

## The grain map is mirrored / reflected

symptom: systematic.per_object

**Test.** Reconstruct a **known feature** — a sample edge, a notch, or a fiducial — and check
its handedness against a microscope or tomography image. If the feature is flipped about a
diagonal or an axis, the geometry convention is wrong. If the feature is where it should be
but *individual grain orientations* look wrong, that is a different cause (seed / Friedel),
not a mirror. A genuinely mirrored map is internally consistent and plausible — the feature
test is the only thing that separates it from a correct one.

**Cause.** A flipped **ω sign** (the frame→ω mapping disagrees with the `SMS/aero` encoder)
or a wrong **`positions.csv`** order (sorted where file-order is expected) or sign.

**Lever.** Fix the ω sign (Handbook phase 1.1) or the position convention (phase 1.2) and
re-run from binning — cheap. Do **not** flip the output map; the seeds and per-voxel
attribution are also mirrored underneath.

---

## Almost no voxels were refined / very low indexed fraction

symptom: quality.low_fraction

**Test.** Count `Results/Result_OrientPos_voxel_*.csv` against the voxel count, and inspect
`SpotsToIndex.csv`: is it **5-column** (`voxNr SpId nSpotsBest _ bestSolIdx`) or a single
column / spot list? A 1-column file with a c-omp refine means the refiner read a malformed
seed and refined nothing (it still exits 0). If the seed file is correct and completeness is
simply low across the board, the cause is the data, not the wiring — check the raw peak SNR.

**Cause.** Missing/malformed **5-col PF seed** for the c-omp refiner (wiring) **vs** a
signal-limited scan where most voxels genuinely lack matched spots (physics).

**Lever.** Wiring: synthesise the seed (`midas_fit_grain.scan_seed.write_pf_seed_file`) — the
pipeline now does this automatically. Physics: accept the limit and report completeness; a
weaker seed threshold will not manufacture spots that are not there.

---

## Per-voxel strain magnitude is inflated

symptom: scale.inflated

**Test.** Read the **border ring** of a spot's patch (away from the centred peak). A flat
pedestal of hundreds–thousands of counts means the frames are raw, not dark-subtracted. Re-fit
with `subtract_background=True` and compare: if the median strain drops sharply and the field
pattern changes (low raw-vs-corrected correlation), the pedestal was biasing it. If the border
is already ~0 and nothing changes, the patches were dark-subtracted and this is not the cause.

**Cause.** Raw frames fed to pf-odf without dark subtraction — the purely multiplicative
per-spot scale inflates on the pedestal and the residual is dominated by unmatched background.

**Lever.** `assemble_grain_patch_data(..., subtract_background=True)` for raw frames; leave it
**off** for already-dark-subtracted caches (Handbook phase 4.4).

---

## Strain floors or runs away on the bright rings

symptom: floor.limited

**Test.** Measure the **saturation fraction** on the low-order rings (pixels at the detector
clamp). A ring that is 90 %+ saturated has a flat top the narrow-splat forward cannot match;
it drags the per-spot scale and floors the data loss. Mask saturated pixels
(`saturation_threshold`) and re-fit: if the strain settles, saturation was the cause; if it
does not move, look elsewhere.

**Cause.** Saturated low rings — fitting Gaussian splats against clamped flat-tops. On an
attenuated scan this is coupled to the *fixed* dynamic-range limit (envelope §1).

**Lever.** Mask saturated pixels, or restrict the fit to unsaturated rings. The real fix is
acquisition (HDR / graded attenuation) — a next-experiment change (envelope §2).

---

## Neighbour-misorientation is bimodal (salt-and-pepper map)

symptom: split.bimodal

**Test.** Histogram the misorientation between adjacent voxels. **~85 % ≤ 1° with a tail
> 20° and almost nothing between** is isolated wrong-solution picks, not real sub-grains —
real intragranular spread is continuous, not bimodal. Check the completeness of the outlier
voxels: comparable to their neighbours (a marginally-better wrong pick) confirms it; much
lower suggests genuinely unindexed voxels instead.

**Cause.** The top-completeness candidate at scattered voxels landed on a wrong orientation
variant that scored marginally higher than the neighbour-consistent one.

**Lever.** Neighbour-consensus cleanup before grain segmentation (Handbook phase 5.2); for a
principled fix, re-pick each voxel's candidate closest to its neighbours' orientation
(`seed_om_table` in the refiner) rather than by completeness alone.
