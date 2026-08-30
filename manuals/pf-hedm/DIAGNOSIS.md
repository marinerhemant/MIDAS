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
| `contamination.sino_rows` | `sinogram_concentration` (`midas_pipeline/find_grains/_sinogen.py:91`) → `sinoConc_*.bin`, when `--sino-conc-threshold` is set |
| `coverage.out_of_field` | the `reconstruct` stage's own warning, read from `sinoOccupancy_<nG>.bin` (`midas_pipeline/stages/reconstruct.py:58`) — written unconditionally by find_grains |
| `consistency.half_split` | this entry's own cross-check: reconstruct each grain from two disjoint halves of its sinogram rows and correlate the images |
| `baseline.majority_class` | this entry's own cross-check: any map-vs-map agreement score against the score of calling every voxel the most common grain |

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

---

## Grain shapes look streaky, scattered, or wrong — while positions look fine

symptom: consistency.half_split

**Test — and run it in this order.** Three cheap discriminations come before any hypothesis:

1. **Occupancy.** Read `sinoOccupancy_<nG>.bin`. Above ~0.65 the grain fills or exceeds the
   scanned field and its shape is **geometrically** unrecoverable — the projections never
   see it end. That is an answer, not a defect (envelope §2).
2. **Registration.** Is `midas_pipeline ≥ 0.11.0`? Below it every reconstruction is one
   voxel low in both axes for odd `n_scans` — a constant offset that looks like nothing
   (Handbook phase 6 §6.2).
3. **Half-split.** Reconstruct each grain from two disjoint halves of its rows and
   correlate. **This is the discriminating test**: ~0.8–0.9 means the sinogram is
   self-consistent and the shape failure is downstream of the data; ≈ 0 means the sinogram
   itself is noise and nothing reconstructed from it means anything.

**Cause.** If occupancy and registration are clean and half-split is high, this is the
**open problem** — the residual sits at 0.82–0.84 invariant across FBP/SIRT/MLEM and the
cause is unknown (notebook §7.5). Positions are unaffected and remain quotable.

**Lever.** **Do not quote the shapes** (spine halt condition). Quote positions, which fit to
1.3–2.1 µm. If shapes are the deliverable, absorption tomography of the same sample is the
better instrument.

⚠️ **Before re-investigating, read notebook §7.5.** Eleven mechanisms are recorded there as
tested — four of them requalified — plus `RawSumIntensity`, |F|²·Lp normalisation (twice)
and edge-padding, each with the preregistered criterion it failed. **Do not score a new
attempt with dice**: it is blind to sub-amplitude artifacts and caused four separate
requalifications on this exact problem.

---

## A few sinogram rows are vertical stripes / a grain's fitted position is far from its voxels

symptom: contamination.sino_rows

**Test.** Compute per-row concentration — the fraction of a row's intensity lying within
`±max(D, 4 µm)/2` of the grain's own fitted sinusoid — and histogram it. A contaminated row
smears across *every* scan position. Cross-check against intensity: on the reference
campaign the flagged rows were **stronger** than clean rows (median total I 470 k vs 310 k,
`corr(conc, log I) = −0.254`) and spanned 44 % more scan bins. If your low-concentration
rows are instead *weak*, you are looking at noise, not contamination, and the filter will
not help.

**Cause.** A reflection that also collected a neighbouring grain's spot, or a spot from a
grain outside the scanned field. The extra intensity is not on the grain's sinusoid, so it
biases the position fit hard.

**Lever.** `--sino-conc-threshold 0.35`, or
`apply_concentration_filter(raw_sino, conc, 0.35)`
(`midas_pipeline/find_grains/_sinogen.py:216`). Measured: 16/958 rows dropped, grain 3's
position fit **5.59 → 1.11 µm**. **Do not retune the 0.35** — it transferred unchanged to a
different sample at 21× coarser sampling. Expect a *smaller* gain there (16–34 %, and
slightly negative on the occasional grain), and expect **no** change in the reconstruction
residual: this fixes position, not shape.

---

## The completeness map shows no sample edge — "there is material everywhere"

symptom: coverage.out_of_field

**Test.** Two, and the first one is usually enough:

1. **Fit the scan-geometry null**, `f(r) = 1 − (2/π)·arccos(S/r)`. If it explains the radial
   falloff (R² ≈ 0.9), the disc you are looking at is the *scan*, not the sample. The null
   predicts a **circle**; a real straight edge predicts a **chord**.
2. **Histogram the (s, ω) spot count** from `InputAllExtraInfoFittingAll*.csv`. Its
   **support** is the silhouette. A straight edge appears as a wedge, and **twice, 180°
   apart with the sign of s flipped** — look for the partner. No partner, no edge.

**Cause.** Completeness cannot see vacuum. A vacuum voxel shares beam lines with material
further along, **inherits that grain's orientation**, and scores ~0.92. Measured: a floor of
0.445 with *nothing* below 0.40 across a grid that was 21.5 % vacuum.

**Lever.** Mask from the spot-count sinogram support (Handbook phase 1b), then recompute
every per-voxel statistic. Masked, the same layer read median completeness **1.0000** in
material against 0.9219 in vacuum.

⚠️ **Do not use the tomographic max-grain-intensity map as ground truth for the boundary.**
It answers "did one of the *listed* grains reconstruct here", not "is there material here";
using it produced a wrong edge placement that had to be retracted (notebook §7.5).

---

## Two maps "agree" — but the number has no null

symptom: baseline.majority_class

**Test.** Compute the **constant-map null**: the agreement you would get by calling every
voxel the single most common grain. Report both, plus Cohen's κ. On the reference campaign
the tomographic and point-by-point maps agreed on **60.1 %** of voxels — against a null of
**65.2 %**, κ = 0.399.

**Cause.** Grain-ID maps are dominated by one or two large grains, so raw voxel-wise
agreement is mostly measuring "both maps found the big grain". The statistic is not wrong;
quoting it without its null is.

**Lever.** Report agreement *relative to* the null (spine hard rule 9). If it falls below
the null, say so plainly — that is not weak agreement, it is none. And note that this
applies to **filtered-vs-unfiltered** comparisons too: excluding out-of-field grains scored
11.0 % against 47.8 % precisely because the large grain carries the map.

---

## Solution counts differ between a pipeline run and a hand-run of the same binary

symptom: consistency.reproduction

**Test.** Compare `Params->ScanPosTol` between the two runs. Grep the parameter
file each actually read for `ScanPosTol`; if the hand-run's file has no such
line, that is the cause. Confirm by appending `ScanPosTol <BeamSize/2>;` to the
hand-run's file and re-indexing a handful of voxels — the counts should match
the pipeline's exactly, voxel for voxel.

**Cause.** Two things compound.

1. `IndexerUnified.c:2608-2627` **adds 0.1 µm to the parsed `BeamSize`**:
   ```c
   sscanf(line, "%s %lf", dummy, &BeamSize);
   BeamSize += 0.1;          /* silently inflates the parsed value */
   ```
   and the gate (lines 1006 *and* 3447 — matching **and** seeding) is
   `scanTol = (ScanPosTol > 0) ? ScanPosTol : (BeamSize / 2)`.
2. `stages/indexing.py` computes `scan_pos_tol_um = beam_size/2` **in Python from
   the true value** and `Indexer._emit_c_omp_paramstest` writes it out, so a
   pipeline run gets the intended tolerance. A hand-run without that line falls
   back to `(BeamSize + 0.1)/2`.

At `BeamSize 1.5` that is a gate of **0.80 µm instead of 0.75** — 6.7 % wider,
which measured **+14.7 % more accepted solutions** across a layer and changed the
per-voxel winner in 10.5 % of voxels.

**Lever.** Always pass `ScanPosTol` explicitly when driving the binary by hand.
Never rely on the `BeamSize/2` fallback — it does not mean what the parameter
file says.

⚠ **`paramstest_comp.txt` is written by TWO different stages under one filename**
— `Indexer._emit_c_omp_paramstest` (indexing) and
`stages/_comp_params.comp_backend_paramstest` (refinement). Refinement runs later
and **overwrites indexing's copy**, and only indexing's carries `ScanPosTol`. So
the file on disk is *not* the file the indexer read, and reconstructing a run from
it reproduces the wrong gate. This cost a full session to diagnose; it was
initially and wrongly attributed to a released version change.

---

## The per-voxel map passes every internal check — but might be chance

symptom: quality.no_null

**Test.** Run the ω-shuffle null ([`phase-7-validation.md`](phase-7-validation.md)):
permute ω within `(ring, scan)`, re-bin, re-index with the **same binary and the
same `ScanPosTol`**, and compare best-completeness per voxel. The **chance ceiling**
is the highest best-completeness any null voxel reaches.

**Cause.** On a dense scanning layer completeness **saturates** — a quarter of
voxels sit at exactly 1.0000 — so it has no dynamic range left to discriminate,
and a spot cloud dense enough will fit almost any orientation. Nothing internal to
the run exposes this: the map is full, the completeness map reads healthy, and the
orientations are spatially plausible.

**Lever.** Gate just above the measured ceiling. The shipped
`MinMatchesToAcceptFrac 0.500000` sat **below** the ceiling on both layers tested
(0.6957 and 0.8333, the latter measured at a wider gate and so an over-estimate), admitting one null voxel for every two real ones. **Measure
the ceiling per layer** — it rose with spot density, and the dense layers were the
more contaminated ones (39.5 % of voxels in the chance band, against 25.0 % on the
sparse layer). Where completeness has saturated, use **IA** (separated 2.6×) and
the threshold-free **spatial-coherence ratio** (§7.5) instead.

---

## merged-FF reports far more grains than the per-voxel map

symptom: quality.no_null

**Test.** Check `positions.csv` in the merged-FF run directory. One row means
`nScans_ == 1`, so `doScanFilter` is **0** (`IndexerUnified.c:1005`) and the
beam-position gate is off in the matching loop entirely. Then run the ω-shuffle
null on the merged spot list.

**Cause.** Collapsing the scans into one FF pattern deletes the `scannrobs`
column, which is the only thing restricting which observed spot a theoretical spot
may match. Every theoretical spot may then match anything in its (ring, η, ω) bin,
and the bins are ~13× fuller. Measured: the null **beat** the real arm (97.5 % vs
97.1 % of seeds finding a solution; IA 0.2896 vs 0.3243; 7 652 vs 6 086 distinct
orientations). The grain count carried no information about the sample.

**Lever.** Do not use merged-FF as a grain-counting route on scanning data — it is
structurally wrong for it, not merely mistuned (phase 7 §7.6), and costs 5.6× more
core-hours than PF unseeded. Raising thresholds until the merged list is sparse
does restore information, but only by discarding ~92 % of the spots — the weak
small-grain spots the technique exists to capture. merged-FF as a **seeding**
route is a different question and is unaffected.

---

## Spot counts do not add up / ~20 % of rows are all zeros

symptom: data.placeholder_rows

**Test.** `awk 'FNR>1 && int($6)==0' InputAllExtraInfoFittingAll*.csv | head` —
if the rows are all-zero apart from `SpotID`, `OrigSpotID` and `ReturnCode`, they
are failed-transform placeholders carrying `RingNumber 0`.

**Cause.** The transforms stage writes a placeholder row for every spot that
failed, rather than dropping it. On the reference campaign **235 334 of 1 170 954
rows (20.1 %)** were placeholders.

**Lever.** **Always filter `RingNumber > 0` before counting spots** in these
files. Counting raw rows against a merged file that has already dropped them
manufactures a fake "20 % collapsed on merge" — the real merge collapse was
0.09 %. The same placeholders are what a stray `ring 0` in a per-ring tally means.
