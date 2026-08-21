# pf-HEDM diagnosis reference

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches to
a symptom the generic diagnostics detect. Keyed by *symptom*, not by step — the step that
produced a symptom is rarely the step you are on.

**Every entry carries a test that can come back the other way.** Before re-investigating,
read [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) §5 — several attractive hypotheses are recorded
there as refuted.

---

## Local symptoms

Emitted by **this technique's own procedure**, not by `beamreport`'s generic
diagnostics, which key off per-observation residuals against declared coordinates.
Comparing two of pf-HEDM's own outputs against each other — the reported strain
tensor against the strain implied by the refined cell in the same row — is real and
useful, and nothing generic will ever detect it, so it is declared here rather than
renamed into the wrong shape.

Every row names where the check lives. A symptom nothing produces is dead text that
reads as coverage, which is exactly what the generic vocabulary existed to prevent.

| symptom | emitted by |
|---|---|
| `consistency.strain_vs_lattice` | this entry's own cross-check: `O E Oᵀ` from `Result_OrientPos_voxel_*.csv` cols 27-35 (sample frame, already µε) against strain rebuilt from cols 15-20 (crystal frame) |

Strain railing at the Kenesei `MargStrain` bound is **not** listed here: it is the
generic `bound.pileup` (objects piling against a declared parameter bound).

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

## Per-voxel strain components sit exactly on ±10000 µε

symptom: bound.pileup
coord: strain

**Test.** Count components within ~1 µε of **10000**:
`sum(abs(abs(E) - 10000) < 1)` over `Result_OrientPos_voxel_*.csv` cols 27–35.
Any hit is diagnostic — that is the `MargStrain` box (default ±0.01), not a
measurement. Then fit the cell from the observed rings (phase-2 §2.5) and compare
with `LatticeParameter`: a mismatch of more than ~2000 µε is the cause. **Do not
raise `MargStrain` to make it go away** — a wider box hides a bad reference
instead of exposing it.

**Cause.** `LatticeConstant` is not the sample's cell. `StrainTensorKenesei`
measures `(dsObs − ds0)/ds0` against the *nominal* `ds0`, so the reference error
is spent out of the ±10000 µε budget before any real strain is measured.
Measured on NMC811: a pristine reference (0.7 % off a charged cell) railed
**11.9 %** of voxels; pinning the cell took it to **0 %**, and lifted completeness
0.618 → 0.833 and voxels 84 → 123.

**Lever.** Pin `LatticeConstant` with
`midas_hkls.refine_lattice_from_d_spacings` and re-run (phase-2 §2.5). Cross-check
with `midas_stress.recover_d0_anisotropic`; the two should agree to ~1000 µε.

---

## Reported strain looks unrelated to the refined lattice in the same row

symptom: consistency.strain_vs_lattice

**Test.** Rotate before comparing. The reported `E11..E33` are in the **sample**
frame (`StrainTensorKenesei` fits on `gobs`, the observed G-vector directions),
while strain rebuilt from `a,b,c,α,β,γ` is in the **crystal** frame. Compare
`O E Oᵀ` against the reported tensor, not the raw components. On the reference
dataset that took the correlation from **−0.08 (meaningless) to +0.84…+0.94**.
Also check units: the `E` columns are **already microstrain** — multiplying by
1e6 again manufactures a fake 5×10⁹ "rail".

**Cause.** A frame (or unit) mismatch in the *analysis*, not a defect in the
refiner. Both of these were live mistakes in a real session before being caught.

**Lever.** Compare invariants (trace, eigenvalues) first — they are frame-free —
then rotate. If the correlation is still low **after** rotating, only then suspect
the refiner.

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
