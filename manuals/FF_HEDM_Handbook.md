# FF-HEDM Reconstruction Runbook

**Audience: a Claude Code session with no prior context that must take a new far-field
beamtime from raw frames to a grain list.** Not a tutorial. Follow the steps in order;
each one names the file to read, the command to run, the field to look at, and the branch
to take.

Citations are `path:line` relative to `$MIDAS = /Users/hsharma/opt/MIDAS`. Read them with
absolute paths. Every non-obvious claim carries one. Claims that are convention, or that
could not be verified, are flagged inline and listed again in §11. **Do not promote a §11
item to a fact.**

Sibling document: `NF_HEDM_Handbook.md`. The two share §2 (ω sign), §3 (metadata) and §4
(energy/distance) almost verbatim, because those are properties of the *beamline*, not of
the modality. Where FF differs, it is called out.

Maintained code = `midas_zipper` (0.1.4), `midas_calibrate_v2` (0.5.2), `midas_peakfit`,
`midas_transforms`, `midas_index`, `midas_fit_grain`, `midas_process_grains`, and the
orchestrator `midas_ff_pipeline`. `FF_HEDM/` is soft-deprecated C; only its example
parameter files are used here. **The bundled c-omp binaries in `midas_index/bin` and
`midas_fit_grain/bin` are the preferred fast path** — "deprecated C source" does not mean
"don't use the c-omp indexer."

---

## STOP — read this before touching anything

### Hard rules

1. **Determine the ω sign convention first (§2).** Field 9 of `<beamtime>_FF.par`. If it
   reads `aero`, the stage turns **clockwise** and **ω_MIDAS = −ω_logged**: negate
   `OmegaStart` *and* `OmegaStep`. Get this wrong and the reconstruction is **mirrored**,
   which is **not detectable from the grain list**. Step 1 of every new dataset, no
   exceptions.
2. **On the 1-ID GE detector, frame 0 of every acquisition is a throwaway — set
   `SkipFrame 1` (§3e).** This applies to the FF data sweep, the dark, and the calibrant.
   It is required for *correctness*, not tidiness: the frame is real data taken before the
   stage settled, and if the sweep is defined off the raw frame count it also shifts ω by
   one step for every subsequent frame. **Scope: GE / far-field only. Do NOT carry this to
   the near-field detector** — on an NF `DoVolume`/`DoLayer` scan the spare file is a
   *trailing* ω-wrap frame, not a leading throwaway, and `StartNr` is the first image
   (`NF_HEDM_Handbook.md` §3g).
3. **`DetZ` is not `Lsd` (§4b).** The detector-stage readback carries an arbitrary zero
   offset. On `pokharel_jul26` the offset was **+181 mm** on a 1666 mm distance — 11 %.
   Only *differences* between `DetZ` readbacks are trustworthy. Lsd comes from the
   calibrant, always.
4. **The filename is not the energy (§4a).** `pokharel_jul26` wrote
   `CeO2_..._96keV_000001.ge5.h5` for a scan taken at **95.0 keV**. Three instrument
   records agreed on 95 and the string was stale. Read `instrument/HEM/Energy` from the
   HDF5, cross-check `fastsweep_Emon.txt` field 6 and the spec log's `Energy (keV):`.
5. **Never trust a calibration you have not overlaid on the image (§5d).** A converged fit
   with a good strain number can still sit on the wrong ring assignment. Overlay predicted
   rings on the measured frame and look at it.
6. **Report calibrant strain as median and 5 %-trimmed, not just mean (§5c).** The mean is
   dominated by a handful of bad fits. **Hard gate: no calibrant geometry above 100 µε
   goes downstream.**
7. **Units: µm, degrees, Å** (Å for wavelength and lattice parameters only). Output Euler
   angles are **radians**.
8. **`Lsd` and `Wavelength` are a *pair*.** They are near-degenerate against a single
   powder pattern (§5e). Whatever λ you fit at, downstream must use the *same* λ. A 1 %
   λ error mostly cancels in relative strain but scales absolute lattice parameters by 1 %.
9. **`Rsample` and `Hbeam` are a SEARCH BOUND, never the sample size (§6).** Setting them
   to the true dimensions plops grains onto the bounding-box edges, manufacturing a
   pile-up of positions at ±`Rsample` and ±`Hbeam`/2. Keep the generous defaults. If
   indexing is slow, suspect the binned-file format (§13e), never the envelope.
10. **Check what the pipeline actually skipped.** `midas-ff-pipeline` no-ops stages that
   don't apply; a silent no-op and a silent failure look identical in the log tail. Read
   the per-stage provenance in `<result>/LayerNr_N/midas_state.h5`.

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `aero` ω sign | mirrored microstructure, plausible completeness | §2 |
| first frame kept | one bad frame + every ω off by one step | §3e |
| "fixing" the zipper to skip the first file too | double skip — 1440 frames become 1439 | §3e |
| `DetZ` used as `Lsd` | 11 % geometry error that still "converges" | §4b |
| energy taken from the filename | 1 % λ error → 1 % `Lsd`, wrong absolute lattice parameter | §4a |
| dark read from `exchange/dark` | that group does not exist in DM files; dark silently all-zero | §3d |
| calibrant fit accepted on strain alone | wrong ring assignment fits beautifully | §5d |
| E↔M loop returns its LAST iterate | ships a worse geometry than the run found (72 vs 18 µε) | §5c |
| ring-ratio check skipped | innermost ring mis-assigned; Lsd off by a ring-spacing factor | §5b |
| `ImTransOpt` differs between calibration and recon | geometry mirrored relative to the fit | §6 |
| lattice constant left as the calibrant's | CeO₂ rings predicted for a gold sample | §6 |
| `Rsample`/`Hbeam` set to the REAL sample size | grains plop onto the bounding-box edges — an artefactual pile-up at ±Rsample, ±Hbeam/2 | §6 |
| residual-correction map applied when it made strain worse | v2 discards it automatically — check it did | §5c |
| `darkLoc` left unset (only `darkDataset` set) | all-zero dark → 0 peaks on every frame, invariant to `RingThresh` | §3d |
| `RingThresh` copied from a template | strict size filter shaves spots to single pixels → 0 peaks | §6b |
| stale `midas-fit-grain` < 0.5.7 | `Grains.csv` DiffPos/DiffOme/DiffAngle cyclically mislabeled | §8 |
| `peakfit: AllPeaks_PS.bin already exists; skip` | results silently inherited from a previous, differently-configured run | §7 |
| peakfit / calc_radius from a tree without the 2026-07-30 determinism fixes | **every re-run gives a different `Grains.csv`**; grain positions jump by >100 µm | §12 |
| grain position quoted to more than ~100 µm | candidates within a cluster still disagree by 50-280 µm, all at completeness 1.0 | §12e |
| `indexing(FF): 0 / N seeds with non-zero data` read as a failure | cosmetic — the c-omp backend writes `IndexBest_all.bin`, the counter looks for `IndexBest.bin` | §11 |
| GrainRadius from a tree without the 2026-07-30 ID-space fix | **every grain reported at ~the sample-wide mean radius**; 5.5× too small here | §13d |
| legacy FF C binaries fed the pipeline's `Spots.bin`/`nData.bin`/`Data.bin` | PF layout vs FF layout — indexer runs minutes instead of 2 s and indexes nothing; looks like bad parameters | §13b, §13e |
| FF refinement on a tree without the 2026-07-30 `pos_scale` equilibration | **grain positions are the indexer seeds, unrefined** — ~158 µm off the C reference in float32, and the solver reports success | §13f |

---

## 1. Environment

All APS hosts share `/home/beams*`. conda is **not** on the non-interactive ssh PATH, so
call the shared env by full path:

```bash
/home/beams12/S1IDUSER/opt/envs/midas/bin/python
```

GPU prefix: `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE`. Pick a GPU by
**utilisation**, not free memory.

| Host | GPU | Note |
|---|---|---|
| chiltepin | driver dead | **only host with internet — install here** |
| copland | 2× A6000, 96 cores | general workhorse; jump host for toro/shannon |
| alleppey | 4× H100 | |
| sentosa | 2× H200 + 2× RTX PRO 6000 | most GPU memory |
| chutoro | 2× A6000, 64 cores | no internet |

**The shared env is not complete.** Verified 2026-07-30: `matplotlib` and `scikit-image`
were both absent, and `scikit-image` is a hard requirement of the v2 auto-seeder
(`midas_calibrate_v2/seed/auto_seed.py:523`). Install from chiltepin:

```bash
ssh chiltepin '/home/beams12/S1IDUSER/opt/envs/midas/bin/pip install matplotlib scikit-image'
```

Long jobs need `setsid`/`nohup` + a redirect or they die on SSH hangup. Write scripts to a
file and `scp` them; do not inline `cat > file && python &`.

Outputs go under the beamtime's own `analysis/` tree, e.g.
`/gdata/dm/1ID/<year>/<beamtime>/analysis/<task>/`. **Never leave results in `/tmp`.**

---

## 2. STEP 1 — Establish the ω sign convention

**Run this first, on every new dataset.**

```bash
awk '{print $9}' <METADATA_DIR>/<beamtime>_FF.par | sort | uniq -c
```

| field 9 reads | meaning | action |
|---|---|---|
| `aero` / `Aero` | stage turns **clockwise**; **ω_MIDAS = −ω_logged** | negate `OmegaStart` **and** `OmegaStep` |
| anything else | not established by this session | **stop and ask** |

Verified on `pokharel_jul26`: all **7297** FF rows read `aero`.

Worked example — `Au3_cubes_ff_000008`. The par logs 1441 frames running
ω = −180.25 → +179.75 at step **+0.25**. Negating, and dropping the throwaway frame 0
(§3e), the parameter file gets:

```
OmegaStart 180.25      # omega of RAW frame 0, negated
OmegaStep  -0.25
SkipFrame  1           # -> first frame actually used is at +180.00
OmegaRange -180 180
```

**Why you cannot check this later.** A sign flip in ω mirrors the reconstructed
microstructure. Completeness, grain counts and internal angles are all unchanged. Nothing
inside the reconstruction catches it.

**Corroboration:** `NF_HEDM_Handbook.md` §2 reaches the same rule from the NF par of the
same beamline, and the bundled NF reference paramfile carries `OmegaStart 180` /
`OmegaStep -0.25` for a 360° aero scan.

---

## 3. STEP 2 — Metadata, and the scan definition

### 3a. Where things live

The image tree holds **only frames**. Distances, ω, energy and exposure live in a separate
acquisition-log folder.

| what | where (`pokharel_jul26`) |
|---|---|
| frames | `/gdata/dm/1ID/2026/pokharel_jul26/data/ge5/` |
| acquisition logs | `~s1iduser/new_data/pokharel_jul26/` |
| per-frame FF par | `<logs>/pokharel_jul26_FF.par` |
| energy monitor | `<logs>/fastsweep_Emon.txt` |
| spec log | `<logs>/FullLog.log` |
| macros | `<logs>/macros_PK/` |

### 3b. Par-file field map (1-ID FF)

Positional, whitespace-separated. Verified against `pokharel_jul26_FF.par`:

| field | meaning |
|---|---|
| 1–5 | date stamp |
| 6 | detector tag (`GE_AD`) |
| 7 | scan name |
| 9 | **rotation stage** — the `aero` test (§2) |
| 10, 11 | sweep bounds (logged ω) |
| 17 | **per-frame ω** (logged) |
| 19 | exposure (s) |
| 20 | **file number** |
| 21 | **frame index within the file** (1-based) |

Extract one scan's sweep:

```bash
awk '$20=="000008" && $7=="Au3_cubes_ff" {print $21, $17}' <logs>/<beamtime>_FF.par
```

### 3c. HDF5 layout (DM-converted `.ge5.h5`)

```
exchange/data          (nframes, 2048, 2048) uint16   <- frames, and ALSO the dark file's frames
exchange/data_dark     (1, 2048, 2048)                <- NOT the dark you want
instrument/Detector/PixelSizeX,Y                      <- 200.0 µm
instrument/Detector/ArraySizeX,Y                      <- 2048
instrument/HEM/Energy                                 <- monochromator energy, keV
instrument/DMS/DetZ                                   <- detector STAGE position (§4b)
instrument/SMS/E/HR/samRy                             <- per-frame rotation readback
```

### 3d. The dark — separate file, in `exchange/data`, and the key name is `darkLoc`

**Use the separate dark file, not the in-file `exchange/data_dark`.** Pair it with its
scan by acquisition number: the dark is `dark_before_<N-1>` for data file `<N>` —
`dark_before_000007.ge5.h5` goes with `Au3_cubes_ff_000008.ge5.h5`. Its frames live in
**`exchange/data`**, exactly like the calibrant dark.

**The key `midas_zipper` reads is `darkLoc`, not `darkDataset`.** They are different
spellings consumed by different code:

| key | read by | default |
|---|---|---|
| `dataLoc` / `darkLoc` | `midas_zipper.ff_zip` — `config['darkLoc']`, `ff_zip.py:334` | `exchange/data` / **`exchange/dark`** |
| `dataDataset` / `darkDataset` | downstream consumers (`FF_Parameters_Reference.md` §2) | same |

Set **both**:

```
dataLoc     exchange/data
darkLoc     exchange/data
dataDataset exchange/data
darkDataset exchange/data
Dark /gdata/.../data/ge5/dark_before_000007.ge5.h5
```

> **This is the highest-cost trap in the whole FF path.** Set only `darkDataset` and the
> zipper falls back to `exchange/dark`, finds nothing in the dark file, warns **once** in a
> 1000-line log, and writes an **all-zero dark** into the zarr. Nothing downstream errors.
> The failure surfaces far away, as:
>
> ```
> FrameNr: 0, NrOfRegions: 5, Filtered regions: 0, Number of peaks: 0
> ...
> ValueError: No spots in InputAll.csv. Aborting.        (transforms stage)
> ```
>
> and it is **invariant to `RingThresh`** — lowering the threshold 60 → 10 changes nothing,
> which is the diagnostic signature. Mechanism: with no dark subtracted every pixel sits at
> the ~1900-count pedestal, so the whole frame clears the threshold, labelling returns a
> few enormous blobs, and `filter_regions_by_size` (strict `minNrPx < nPx < maxNrPx`,
> `connected.py:91-100`) discards all of them for exceeding `maxNrPx`.
>
> **Check it directly** rather than trusting the log:
> ```python
> z = zarr.open("<result>/LayerNr_1/<stem>.MIDAS.zip", mode="r")
> assert np.asarray(z["exchange/dark"][0]).max() > 0   # zero == dark was not found
> ```

### 3e. The throwaway first frame — GE / far-field only

**On the 1-ID GE detector the first frame of every acquisition is a settling frame. Always
skip it.**

> **Scope.** This is a **GE (far-field) detector** rule. It does **not** apply to the
> near-field detector: on an NF `DoVolume`/`DoLayer` scan the extra file in the sequence is
> a *trailing* ω-wrap frame at the end, and `StartNr` is the **first** image. Carrying this
> rule to NF drops a real frame and pushes the ω reversal inside the first distance. See
> `NF_HEDM_Handbook.md` §3g.

Measured signature on `pokharel_jul26` GE5: frame 0 sits ~1.5 % low in baseline versus
every later frame.

| file | frame 0 mean | later frames |
|---|---|---|
| `Au3_cubes_ff_000008` | 1868.96 | ~1898 (frames 720, 1440) |
| `dark_before_000007` | 1870.55 | ~1897.9 (frames 5, 9) |
| `dark_CeO2_..._000003` | — | dropping frame 0 moved the dark mean 2044.1 → 2018.7 |

Set `SkipFrame 1`.

**`SkipFrame` is applied by the consumer, not by the zipper — do not "fix" this.** The
layered design is easy to misread:

| stage | what it does with `SkipFrame` |
|---|---|
| `midas_zipper` | writes **all** raw frames of the first file and the **full** dark stack; records `SkipFrame`; skips leading frames only of files **2+**, which is multi-file concatenation de-duplication, a different thing |
| `midas_peakfit` | does the actual skip: `nFrames -= skipFrame` (`params.py:135`), reads `frame_nr + skipFrame` (`orchestrator.py:181-183`), `dark_arr[skipFrame:]` (`zarr_io.py:301`) |

Consequently **`OmegaStart` is the ω of the first frame you want to USE** (post-skip), and
the zarr's `scan_parameters/start` is deliberately back-dated to
`OmegaStart − SkipFrame·OmegaStep` (`ff_zip.py:294`) so that it describes raw frame 0.
The consumer recovers `start + SkipFrame·step = OmegaStart` for the first frame it
processes. The chain is self-consistent; changing either half alone breaks it.

> Making the zipper physically drop the frame **as well** skips it twice: a 1441-frame
> sweep yields 1439 processed frames instead of 1440. Confirmed the hard way on
> `Au3_cubes_ff_000008` in this tree. Guarded by
> `midas_zipper/tests/test_skipframe.py`.

Sanity check in the peakfit banner: `nFrames` must equal *logged frames − SkipFrame*
(1441 − 1 = **1440**). If it reads 1439, something is skipping twice.

For a hand-reduced average outside the pipeline (calibrant staging, quick looks) there is
no consumer to do it for you, so drop it yourself: `data[1:].mean(axis=0)`, dark included.

---

## 4. STEP 3 — Energy and distance: the two fields that lie

### 4a. Energy

**The filename is not the energy.** On `pokharel_jul26` the CeO₂ files are named
`..._96keV_...` and the scan was taken at **95.0 keV**.

Sources, in order of trust:

| source | `pokharel_jul26` | verdict |
|---|---|---|
| `instrument/HEM/Energy` (HDF5) | 95.0 | **use this** |
| `fastsweep_Emon.txt` field 6 (`E_HEM`) | 95.0000 | corroborates |
| spec `FullLog.log` → `Energy (keV):` | 95 | corroborates |
| `instrument/InsertionDevice/IDEnergy` | 95.055 | undulator setting, not the mono |
| `instrument/HRM/Energy` | 78.39 | **different monochromator — ignore** |
| the filename | "96keV" | **stale string** |

`fastsweep_Emon.txt` columns come from `macros_PK/E_mon.mac`: field 2 is a foil µt, field
6 is `epics_get("1id:userTran3.A")` = the HEM energy readback. **Rows where the last two
columns are `0.000 0.000` had the foil out** (air) and carry no absorption information.

λ[Å] = 12.398419843320026 / E[keV]. At 95.0 keV, λ = 0.130510 Å.

### 4b. Distance — `DetZ` is a stage readback, not `Lsd`

`instrument/DMS/DetZ` is the detector translation-stage position. Its zero is not the
sample rotation centre.

**Measured on `pokharel_jul26`:** `DetZ` = 1485.00 mm, calibrated `Lsd` = **1666.2 mm** —
an offset of **+181 mm (11 %)**. Using `DetZ` as `Lsd` would have been a catastrophic and
entirely plausible-looking error.

Use `DetZ` as a *seed* only, and expect the fit to move a long way. Differences between
`DetZ` readbacks across a multi-distance scan are trustworthy; the absolute value is not.

---

## 5. STEP 4 — Calibrate on a calibrant

Package: `midas_calibrate_v2` (0.5.2). Entry point `calibrate()` — image + λ + pixel size
+ calibrant name, everything else auto-seeded.

### 5a. Look at the raw frame first

Reduce remotely, plot to PNG, copy back, and *look*. Before any fit you should be able to
state: how many rings are visible, whether they are complete in azimuth, where the
beamstop is, and whether the detector is saturated.

`pokharel_jul26` CeO₂ reference: rings sharp and complete in azimuth, innermost at
R ≈ 348 px about the fitted BC, beamstop shadow at ≈ (1019, 1076), signal ~54 counts above
a ~2019-count dark after frame-0 removal.

### 5b. Check the ring assignment before you trust the fit

Ring-radius **ratios** depend only on the lattice — λ and `Lsd` cancel. This identifies
which ring the innermost observed one is, independently of any geometry:

```
R_i / R_1  =  tan(2θ_i) / tan(2θ_1)
```

Measure radii from a radial profile about the seeded BC, and compare. On `pokharel_jul26`
the first 10 CeO₂ rings matched to ≤ 0.0015 in ratio, confirming innermost = (111), and
those 10 rings independently gave `Lsd` = 1667.2 ± 0.3 mm — which is what exposed
`DetZ` (1485 mm) as a stage offset rather than a distance.

> Watch the degenerate families: **(511) and (333) share a d-spacing** and are one ring. A
> naive "i-th observed peak ↔ i-th table entry" pairing slips by one from there on.

### 5c. Run it

```python
from midas_calibrate_v2 import calibrate
res = calibrate(
    img,                      # 2-D, dark-subtracted, frame 0 already dropped
    wavelength=0.130510,      # Å, from §4a
    pxY=200.0,                # µm
    calibrant="CeO2",
    initial_Lsd=1_485_000.0,  # µm; DetZ as a SEED only (§4b)
    output_dir=".../ceo2_calib/",
    n_iter=5,
)
```

Read `res.post_residual_strain_uE`, and the per-iteration `mean / median / trim5%` triple
from the log. **Gate: reject above 100 µε.**

v2 builds an empirical residual-correction map after the fit and **discards it
automatically if it did not reduce strain** (`pipelines/single.py:264-273`). Both outcomes
are normal; check which happened rather than assuming the map is in play.

**The E↔M loop is not monotonic, and that matters.** The E-step re-extracts peaks at the
new geometry, so a late iteration can land in a worse basin than an earlier one. Measured
on this dataset:

```
[v2 iter 2] strain=  59.5μϵ   ty=-0.0230  tz=0.7752
[v2 iter 3] strain=  17.9μϵ   ty=-0.0052  tz=0.9507     <- best
[v2 iter 4] strain=  72.0μϵ   ty= 0.1200  tz=0.8630     <- last
```

`ty` is the weakly-determined direction (it wanders while `tz` holds near 0.9); `Lsd` and
`BC` barely move. Before the 2026-07-30 fix, `autocalibrate` returned the **last** iterate,
so this run shipped a 72 µε geometry when 17.9 µε was in hand — a 4× quality loss, silent,
and still inside the 100 µε gate. It now adopts the best iterate and logs
`adopting best iterate (…) over the last (…)`, matching v1 C
(`FF_HEDM/Example/Parameters.txt`: `nIterations` — "best result is kept"). Guarded by
`midas_calibrate_v2/tests/test_best_iterate.py`.

If your install predates that fix, do not read the final line of the log as the result —
scan all iterations and take the minimum, or re-seed at it.

### 5d. Overlay the rings — mandatory

Predict each ring's radius from the fitted `Lsd`, invert (R, η) → pixel through the **full**
forward model (tilts + distortion + parallax) with
`midas_integrate.geometry.invert_REta_to_pixel_batch`, and draw it on the measured frame.
Look at the inner rings *and* the corners. This is the only check that catches a
well-converged fit sitting on the wrong ring assignment.

### 5e. What a single powder pattern cannot tell you

`Lsd` and λ are near-degenerate: to first order both just scale the ring radii. Fitting at
the wrong energy produces a compensating `Lsd` and a still-good-looking pattern. The
degeneracy is broken only weakly, by the `tan(2θ)` nonlinearity, and **refined distortion
harmonics (`iso_R2/R4/R6`) can absorb most of what is left** — so a distortion-free control
is needed for the comparison to mean anything.

Observed on `pokharel_jul26` (same image, λ the only change): 95 keV → 19.4 µε,
96 keV → 72.7 µε. Suggestive, and it agreed with the beamline's own confirmation of
95 keV — but treat it as corroboration, not proof.

### 5f. Use the 0/180 pair if you have one

A calibrant measured at two rotations 180° apart gives an independent repeat of the same
detector geometry; the spread between the two fits is an honest uncertainty. On
`pokharel_jul26`:

| | samRy −90 | samRy +90 | diff |
|---|---|---|---|
| Lsd (mm) | 1666.226 | 1666.008 | 0.218 (0.013 %) |
| BC_y (px) | 1018.720 | 1018.729 | −0.009 |
| BC_z (px) | 1076.540 | 1076.529 | +0.011 |
| ty (°) | 0.0061 | 0.0655 | −0.059 |
| tz (°) | 0.9435 | 0.8977 | +0.046 |

`Lsd` and `BC` repeat superbly; the **tilts** are the weak direction. Note also that the
distortion harmonics differed by up to ~10× between the two fits (`a1`: 0.0001 vs 0.0017)
— the individual harmonic coefficients are fitting noise even when the radial prediction
they sum to is stable. Do not interpret them physically.

### 5g. Export

`midas_calibrate_v2.compat.to_v1.write_v1_paramstest` writes the v1 geometry block
(`Lsd`, `BC`, `tx/ty/tz`, `p0..p14`, `Parallax`, `Wavelength`, `px`, `NrPixelsY/Z`,
`RhoD`, `SpaceGroup`, `LatticeConstant`). v2's harmonic names map back to v1 p-slots via
`_V2_TO_V1_DISTORTION` (`compat/to_v1.py:20-33`) — note the mapping is **not** in index
order (`iso_R2→p2`, `iso_R4→p5`, `iso_R6→p4`, `a2→p0`, …).

`ff_paramstest_from_auto_result` merges the geometry into an existing FF template,
carrying thresholds and scan keys through verbatim.

---

## 6. STEP 5 — Build the parameter file

Start from `FF_HEDM/Example/Parameters.txt` and replace the geometry block. Generate it
from the calibration JSON rather than retyping — hand-copied geometry is how a recon ends
up describing a calibration it isn't using.

**The calibrant's `LatticeConstant` and `SpaceGroup` must be replaced with the sample's.**
The calibration paramstest carries CeO₂ (5.4116, SG 225); a gold sample needs
`4.0782, SG 225`. This is the single most common copy-paste error.

**`ImTransOpt` must match what the calibration was fitted on.** If the calibrant image was
read straight out of `exchange/data` with no transform, the recon must use `ImTransOpt 0`.
A mismatch mirrors the geometry relative to the fit.

Ring selection: only rings with **full azimuthal coverage** are safe defaults. With BC near
the centre of a 2048² detector, that is everything inside the *nearest-edge* radius;
rings between the nearest-edge and far-corner radii are partial and bias η coverage.

`Rsample`, `Hbeam`, `BeamThickness`, `Vsample`, `GlobalPosition` are **not** descriptions
of the sample. **HARD RULE: never set `Rsample`/`Hbeam` to the true sample dimensions.**
They are a deliberately generous *search bound*; tighten them to the real size and any
grain whose true position lies near the boundary is pushed onto it, giving an artefactual
pile-up of grain positions at ±`Rsample` and ±`Hbeam`/2 that reads as real microstructure.
Leave the generous defaults (2000 µm here, matching `FF_HEDM/Example/Parameters.txt`).

---

## 6b. Set `RingThresh` from the data, not from a template

`RingThresh <ring> <threshold>` is the single most consequential number in the peak
search, and the value in any example file is meaningless for your detector, exposure and
sample. Measure it.

The peak finder labels 8-connected blobs above threshold **inside the ring bands only**
(`Width` µm either side of each ring), then applies a **strict** size filter —
`minNrPx < nPx < maxNrPx`, both bounds exclusive (`midas_peakfit/connected.py:91-100`).
With the default `minNrPx 1`, **any single-pixel blob is discarded**. So a threshold that
is slightly too high does not degrade gracefully: it shaves every spot down to a few
isolated pixels and you get exactly zero peaks.

Measure the in-band blob population versus threshold before running:

```python
band = np.zeros_like(img, bool)
for R in ring_radii_px:
    band |= np.abs(radius_from_BC - R) <= width_px
for thr in (5, 10, 15, 20, 30, 40, 60):
    lab, n = ndimage.label((img - dark) * band > thr, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:]
    print(thr, n, (sizes > 1).sum(), sizes.max() if n else 0)
```

Measured on `Au3_cubes_ff_000008` (20 ms/frame, `Width` 7.5 px):

| threshold | blobs/frame | with > 1 px | largest blob |
|---|---|---|---|
| 60 | 2–4 | 1–2 | 26–37 px |
| 20 | 5–7 | 4–6 | 40–66 px |
| **10** | **6–10** | **3–8** | **61–95 px** |
| 5 | 481–591 | 10–19 | noise-dominated |

Pick the lowest threshold that has not yet started admitting noise — the blob count jumps
by two orders of magnitude when you cross into noise, so the knee is obvious. Here that is
**10**, not the 60 a template would have given.

**Caveat:** this tuning is only meaningful once the dark is verified non-zero (§3d). If the
dark is missing, *every* threshold yields zero peaks and the table above is flat — that
invariance is itself the diagnostic.

---

## 7. STEP 6 — Run the pipeline

```bash
midas-ff-pipeline run \
    --params Parameters.txt \
    --result results/ \
    --layers 1-1 \
    --device cuda
```

13 stages, each with a provenance entry in `<result>/LayerNr_N/midas_state.h5`:

```
zip_convert → hkl → peakfit → merge_overlaps → calc_radius → transforms
→ cross_det_merge → global_powder → binning → indexing → refinement
→ process_grains → consolidation
```

`zip_convert` is skipped when the zarr already exists; `cross_det_merge` and
`global_powder` are no-ops for single-detector runs; `consolidation` is gated by
`--generate-h5`. Auto-resolved knobs (`--dtype`, `--shard-gpus`, `--group-size`) are logged
at startup; explicit values always win.

> `midas-ff-pipeline` is **deprecated** as of 0.4.0 — use `midas-pipeline run --scan-mode ff`.
> Same orchestrator underneath.

**Two things to check in the log every time:**

1. `nFrames` in the peakfit banner = logged frames − `SkipFrame` (§3e).
2. **Stage resume is silent.** `peakfit(FF): …AllPeaks_PS.bin already exists; skip.`
   means the peak search did **not** run and results were inherited from a previous
   invocation — which may have used a different threshold, a different dark, or a broken
   config. It costs 0.3 s instead of 55 s, so it is easy to miss. **After changing any
   peak-search or dark parameter, delete `results/` entirely**, do not rely on resume.

Subprocess stages (`peakfit_torch`, `midas_indexer`, `midas_fit_grain`) are invoked by
bare name, so the env's `bin` must be on `PATH` — calling `midas-pipeline` by full path is
not enough and fails with `FileNotFoundError: 'peakfit_torch'`:

```bash
export PATH=/home/beams12/S1IDUSER/opt/envs/midas/bin:$PATH
```

For ≥ 5 k seeds also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Check the zipper's frame accounting in the log.** It prints
`HDF5 scan: N file(s), F frames/file. Skipping the first S frame(s) of every file. Total
frames to write: T`. Confirm `T` equals the frame count you derived from the par file in
§3b. If `T` is one larger, `SkipFrame` did not take effect (§3e version trap).

---

## 8. STEP 7 — Read the result

### 8a. Check the refiner version before reading the residual columns

**`midas-fit-grain` < 0.5.7 writes `DiffPos`, `DiffOme`, `DiffAngle` cyclically
mislabeled.** `driver.py` assigned `calc_angle_errors`'s `(mean_angle, mean_pos, mean_ome)`
straight into the `(pos, ome, angle)` slots, so every FF/PF run through the python/torch
refiner has the three columns rotated (commit `44394e61`; the classic C refiner path was
unaffected).

The tell is obvious once you look: an ω residual of **223°** is impossible on a 0.25° step.
Post-fix the same grain reads `DiffPos 202 µm, DiffOme 0.054°, DiffAngle 0.090°` — all
physical.

```bash
pip show midas-fit-grain      # need >= 0.5.7
```

If you are stuck on 0.5.6, the mapping is: printed `DiffPos` = true DiffAngle, printed
`DiffOme` = true DiffPos, printed `DiffAngle` = true DiffOme.

### 8b. What to check in `Grains.csv`

Before interpreting it:

1. **Grain count vs expectation.** A calibration cube should give a handful of grains, not
   thousands. Thousands means the peak search is finding noise.
2. **Completeness distribution**, not just the mean — a bimodal distribution means two
   populations, usually real grains plus junk.
3. **Position envelope.** If grain positions pile up against ±`Rsample` or ±`Hbeam`/2, the
   envelope is binding and the positions are not physical. The fix is to make the envelope
   MORE generous, never less — see the hard rule in §6.
4. **Strain sanity.** Whole-grain strains far above ~10⁻³ on an annealed calibration sample
   mean the geometry, not the sample.
5. **What fraction of spots got indexed?** `wc -l InputAll.csv` versus the spots actually
   assigned. A handful of grains explaining a few hundred of several thousand spots is an
   *under-indexed* run, not a sparse sample — confidence 1.0 on the few grains found says
   nothing about the ones missed.
6. **Re-run and compare grain-by-grain.** Grains that appear in one run and not the next
   are indexing noise. On `Au3_cubes_ff_000008` two runs shared only one of their two
   grains; that instability is the signal that `Completeness`, `MinNrSpots`,
   `OverAllRingToIndex` still need work.
7. **`indexing: 0 / N seeds with non-zero data`** in the log deserves an explanation before
   any grain list is trusted.

---

## 9. Reference numbers — `pokharel_jul26`, GE5 (ADEPT), 95.0 keV

Established in this tree on 2026-07-30. Detector 2048², 200 µm, monolithic.

| quantity | value | how |
|---|---|---|
| energy | 95.0 keV (λ 0.130510 Å) | `HEM/Energy`, Emon, spec log, beamline confirmation |
| `DetZ` readback | 1485.00 mm | `instrument/DMS/DetZ` — **not** Lsd |
| `Lsd` | see `analysis/ceo2_calib_ge5/summary_all.json` | CeO₂ fit, ≈1666 mm |
| `BC` | ≈ (1018.7, 1076.5) px | CeO₂ fit |
| tilts | ty ≈ 0.0–0.07°, tz ≈ 0.90–0.94° | CeO₂ fit, 0/180 spread |
| calibrant strain | ≈ 19 µε (mean), 13 µε (median) | v2, residual map discarded |
| Au sweep (file 8) | 1441 logged frames, 1440 used | par field 21 |
| Au ω (MIDAS) | `OmegaStart 180.00` (first used), `OmegaStep -0.25` | §2 + §3e |
| Au `RingThresh` | 10 (not 60) | measured, §6b |
| Au spots found | 2078 (`InputAll.csv`), ~8.5 peaks/frame | §6b |
| Au grains indexed | **2**, confidence 1.000, R ≈ 21 µm, a = 4.07976 Å | provisional, §11 |
| Au residuals (0.5.7 cols) | DiffPos ≈ 200 µm, DiffOme ≈ 0.05°, DiffAngle ≈ 0.08° | §8a |

Working run directories:
`analysis/ceo2_calib_ge5/` (calibration) and `analysis/au3_cubes_ff_000008/` (recon).
The shared env needed a `PYTHONPATH` overlay at `~s1iduser/opt/midas_overlay` for
`midas-fit-grain` 0.5.7 and `midas-calibrate-v2`; the env itself ships 0.5.6.

---

## 10. Parameter-file reference

Full key list: `FF_Parameters_Reference.md`. Keys this runbook depends on:

| key | note |
|---|---|
| `SkipFrame` | 1 at 1-ID; applies to every multi-frame file and the dark (§3e) |
| `dataDataset` / `darkDataset` | both `exchange/data` for DM files (§3d) |
| `OmegaStart` / `OmegaStep` | negated vs the log for `aero` (§2); `OmegaStart` describes **raw** frame 0 |
| `ImTransOpt` | must match the calibration (§6) |
| `LatticeConstant` / `SpaceGroup` | the **sample's**, not the calibrant's (§6) |
| `RhoD` | µm; distortion normalisation radius |
| `Rsample` / `Hbeam` | generous grain-position search bound — **never set to the real sample size** (§6) |

---

## 11. Validation status

### Verified in this tree (2026-07-30)

- `aero` on all 7297 `pokharel_jul26` FF par rows; ω negation rule shared with NF.
- Throwaway first frame: measured ~1.5 % baseline offset in three separate files (§3e).
- `SkipFrame` is a **consumer-side** skip; the zipper's first-file exemption and the
  back-dated `start_omega` are both correct as shipped. Verified by reading the three
  `midas_peakfit` call sites and by the double-skip (1439-frame) failure that results from
  changing it. Locked by `midas_zipper/tests/test_skipframe.py`.
- `DetZ` − `Lsd` offset of +181 mm, from an assignment-free ring-ratio measurement (§5b).
- Energy 95.0 keV: three instrument records, plus beamline confirmation.
- CeO₂ 0/180 repeatability (§5f).
- Shared env missing `matplotlib` and `scikit-image` (§1).
- `darkLoc` vs `darkDataset`: the zipper reads `config['darkLoc']` (`ff_zip.py:334`) and
  writes an all-zero dark when it is unset. Confirmed by reading
  `zarr["exchange/dark"]` before (max 0) and after (mean 1870.55) (§3d).
- `midas-fit-grain` 0.5.6 column rotation, and that 0.5.7 fixes it: the same grain's ω
  residual went from 223.87° to 0.054° (§8a).
- Threshold sensitivity table in §6b, measured on this dataset.

### Convention — not verified by measurement

- **`ImTransOpt 0` for this beamtime.** Chosen because the calibration was fitted on the
  untransformed `exchange/data` array, so the recon must match. It has **not** been checked
  against an independent handedness reference (e.g. a known-orientation sample). A
  self-consistent calibration + recon pair can still be globally mirrored. Treat the
  absolute handedness as unestablished until something external pins it.
- `OmegaStart` semantics as "ω of raw frame 0". This is what the fixed zipper implements
  and what §2's worked example assumes; it is a convention choice, not a measurement.

### Provisional — the Au3 cubes reconstruction is still not a finished result

The `Au3_cubes_ff_000008` run completes all 13 stages, is bit-reproducible, and yields
**2 grains** — a parent and its Σ3 twin, both with independent evidence in the raw frames
(§13g). What it is not yet is a *complete* reconstruction:

- **2 grains explain 185 of 2077 spots (8.9 %).** ~91 % of the spot list is unindexed.
  That is the open item; it is indexing-parameter work (`Completeness`, `MinNrSpots`,
  `OverAllRingToIndex`), not geometry.
- **Grain positions are good to ~100 µm, not better** (§12e). Quote them accordingly.
- `Rsample`/`Hbeam` are the generous search bound, which is correct and must stay that
  way (hard rule 9) — not a pending item.

Current output, for reference:

```
 ID          X          Y          Z    DiffPos   GrainRadius
 80    2.685097   1.982347  -5.349646   190.089      114.6207
185  -10.289690 -22.492229  -3.493707   182.207       99.9628
```

Both grains sit within ~25 µm of the rotation centre, and DiffPos (190.1 / 182.2 µm)
is now within ~1 % of the C reference's own residual (188.7 / 181.0 µm).

**Superseded claims — do not resurrect.** Earlier revisions of this document said (i)
"only one of the two grains reproduces across runs", (ii) "the reported position is a
tie-break" over ~500 µm, and (iii) that indexing cost scales with `Rsample`/`Hbeam`. All
three were artifacts of bugs since fixed — the determinism defects (§12b, §12c), the fp32
refiner (§13f), and a malformed `nData.bin` (§13e) respectively.

### `indexing(FF): 0 / N seeds with non-zero data` is a cosmetic bug, not a failure

The message comes from `midas_pipeline/stages/indexing.py:150-157`, which counts non-zero
rows in `Output/IndexBest.bin`. With `indexer_backend="c-omp"` (the default, and the fast
path) the binary writes `Output/IndexBest_all.bin` + `IndexKey_all.bin` instead, so the
counter finds no file and reports 0. Indexing had in fact succeeded. Judge the stage by
`Results/OrientPosFit.bin` and the grain count, not by this line.

### Could not verify — do not upgrade these

- Whether `DetZ`'s +181 mm offset is stable across the beamtime. Measured at one distance
  only.
- Whether the 95-vs-96 keV strain gap (19.4 vs 72.7 µε) is a genuine energy discriminator
  or partly an artifact of the distortion harmonics re-fitting. The distortion-frozen
  control was not run.

### Bottom line

The geometry is trustworthy in magnitude (`Lsd`, `BC` repeat to 0.01 % / 0.01 px across an
independent 180° repeat, and the rings overlay). Its **handedness** rests on convention,
not measurement. The ω sign and the frame-0 skip are the two settings that will silently
ruin a reconstruction, and both are now pinned by tests or by measurement rather than by
memory.

The pipeline itself is bit-reproducible (§12) and agrees with the C reference on grain
reduction, grain radius (2e-7) and refined position (~14 µm) (§13). Five defects found by
that cross-check are fixed and released; every one of them was **silent** — wrong numbers,
never an error message. That is the lesson to carry: on this pipeline, check the answer
against an independent implementation and against the raw frames, because nothing will
tell you it is wrong.

---

## 12. Reproducibility — is the same input guaranteed to give the same answer?

**As of 2026-07-30, yes** — but only after two defects were fixed. If you are running an
older tree, assume it is not.

### 12a. How to check it yourself (do this once per new install)

Run the identical pipeline twice into a clean result dir and checksum every stage, in
pipeline order. Do not just compare `Grains.csv` — you want the **first** artifact that
diverges, which is what identifies the guilty stage:

```bash
for f in Temp/AllPeaks_PS.bin Temp/AllPeaks_PX.bin Result_StartNr_*.csv \
         Radius_StartNr_*.csv InputAll.csv InputAllExtraInfoFittingAll.csv \
         Spots.bin Data.bin nData.bin ExtraInfo.bin SpotsToIndex.csv \
         Output/IndexBest_all.bin Output/FitBest.bin Results/OrientPosFit.bin \
         Grains.csv SpotMatrix.csv ; do
  printf "%s  %s\n" "$(md5sum < "$L/$f" | cut -d' ' -f1)" "$f"
done
```

`rm -rf results/ runinfo/` between runs — the stages resume silently off existing files
(§7), so a "reproducible" result can just be a skipped stage.

Two independent bugs were found this way, and each was masked by the other until the first
was fixed. Expect to iterate.

### 12b. Fixed: `midas_peakfit` — batched-LM grouping was timing-dependent

Symptom: three runs of the same FF pipeline, same `Parameters.txt`, same host, gave three
different `Grains.csv`. The first diverging artifact was `Temp/AllPeaks_PS.bin`, while
`Temp/AllPeaks_PX.bin` (the raw connected-pixel sets) was byte-identical — so thresholding
and connected components were fine and the **peak fit** was not. Isolated: two runs of
`midas_peakfit.orchestrator.run` alone on one fixed zarr moved **1167 of 8599 peaks**, and
not in the last bit — one intensity column differed by 344 counts.

Cause. `lm_solve` is mathematically independent per region but not numerically: batching
`B` regions into one call selects a different cuBLAS/MAGMA batched-GEMM and Cholesky
kernel, so a region's fit depends on *which regions it was solved alongside*.
`RegionPool` made that grouping vary run to run in two ways — it derived the batch quantum
from **live** free VRAM and host `MemAvailable` and re-keyed the cache on the live bucket
count, and the consumer pulled *every* queued entry rather than one quantum, so chunk
boundaries landed wherever the consumer thread happened to be scheduled. On top of that
`lm.py` set `torch.backends.cuda.matmul.allow_tf32 = True` **at import, process-wide**;
that flag's design note only covers the fp64 path, but with the FF default
`--dtype float32` it also caught the plain fp32 `Jt @ J`, assembling the normal equations
at a 10-bit mantissa.

Fixed in `midas_peakfit/pool.py` (quantum decided once per `(n_peaks, m_pixels)`, quantized
to a power of two, consumer pulls exactly one quantum including at drain; host residency
moved to a separate global backstop that logs loudly if it ever fires) and
`midas_peakfit/lm.py` + `lm_generic.py` (TF32 scoped to the fp64 matmul that asked for it,
never a global import side effect). Locked by
`midas_peakfit/tests/test_pool_determinism.py` — 6 of its 7 CPU tests fail on the pre-fix
source. No throughput cost: 34.1 frames/s after vs 30.8 before.

### 12c. Fixed: `midas_transforms` — `calc_radius` used floating-point atomics

With peakfit deterministic, one divergence remained. `Result_StartNr_*.csv` (merge output)
was bit-identical while `Radius_StartNr_*.csv` was not, and exactly three columns moved:

| column | max relative difference between two runs |
|---|---|
| `PowderIntensity` | 2.23e-07 |
| `GrainVolume` | 3.75e-07 |
| `GrainRadius` | 4.49e-07 |

That is float32 epsilon, and it is the whole chain: `powder_int` → `GrainVolume` →
`GrainRadius`. `radius/core.py` summed per-ring intensity with
`powder_int.scatter_add_(0, spot_match, ...)`, which on CUDA lowers to floating-point
`atomicAdd` — arbitrary summation order per launch. It surfaced in `Grains.csv` as
`GrainRadius` 20.775146 vs 20.775148 µm and nothing else.

Fixed by replacing it with a per-ring masked `sum` (`torch.sum` is a fixed-order tree
reduction and deterministic on every backend). The reduction is over the number of
**configured rings**, so this costs nothing. Locked by
`midas_transforms/tests/test_calc_radius_determinism.py`, which also checks the value
against an independent numpy reference so determinism can't be bought with a wrong answer.

**General rule for this codebase:** `scatter_add_`, `index_add_`, and `index_put_` with
`accumulate=True` on float CUDA tensors are all nondeterministic. If a scientific output
depends on one, it is not reproducible.

### 12d. Verified

Three independent full-pipeline runs on `Au3_cubes_ff_000008`, `rm -rf results/` between
each, on chutoro: **bit-identical at all 27 checkpointed artifacts**, `Grains.csv` md5
`0449046c4a1eaa698d447fa480f10671` all three times.

### 12e. Grain position is weakly determined — but far less than first reported

**CORRECTED 2026-07-31.** An earlier revision of this section reported cluster members
spanning **356/193/553 µm** and called the reported position "a tie-break, not a
measurement". Those numbers were measured with the **broken fp32 refiner** (§13f), which
was not refining position at all — so most of that spread was the bug, not the data.
Re-measured on the same seeds after the fix:

| candidate cluster | pre-fix (fp32) span X/Y/Z | **after the fix** |
|---|---|---|
| cluster 0 (8 members) | 245 / 333 / 534 µm | **47 / 71 / 238 µm** |
| cluster 1 (9 members) | 642 / 294 / 517 µm | **133 / 56 / 283 µm** |

So position determinacy improved 3–5× in X and Y. It is still not tight: the candidates
within a cluster disagree by ~50–130 µm in X/Y and ~240–280 µm in Z, and they all sit at
completeness 1.0000, so `Completeness` cannot separate them — with
`MarginRadius`/`MarginRadial`/`MarginEta` at 500 µm (2.5 px at this pixel size) a grain
can move a long way with every spot still inside the matching window. The remaining
discriminator is `DiffPos`, and its minimum is shallow.

**What this means in practice:** quote grain positions to ~100 µm on this dataset, not to
the six decimals `Grains.csv` prints. If you need better, tighten the matching margins
toward 1 px — do NOT touch `Rsample`/`Hbeam`, which are a search bound (hard rule 9).

### 12f. Which candidate becomes "the grain" — and the two modes disagree

**In `--mode spot_aware` (what `midas-pipeline --scan-mode ff` runs), the grain IS one
candidate.** `midas_process_grains/pipeline.py:416-417` picks
`rep_pos = argmin(ias[members])` — the member with the smallest **internal-angle**
residual, `OrientPosFit.bin` **column 24** — and then copies that one candidate's
`position`, `orient_mat`, `lattice`, `grain_radius` and `confidence` straight into the
output (lines 452-456, 524-528). There is no averaging of any kind. Confirmed numerically
on this run: `Grains.csv` ID 80 = (10.882587, 47.793549, −108.281662) and ID 185 =
(−100.816216, −8.876349, 51.627029) are exactly the argmin members, and are **not** the
cluster medians (65.44, 23.35, −120.27) and (−34.62, 5.92, −8.66).

**`--mode physics` does the opposite.** `v4_pipeline.py:723` reports
`np.median(positions[members])` with a rotation-mean orientation, and its comment argues
that a single representative gives a >20 % rate of grains whose stored OM fails to
re-predict its own spots. So the two code paths take opposite positions on the same
question. That is unresolved in this tree — do not assume the mode you ran did what the
other one's comment describes.

**The C reference settles which one is the reference.** `FF_HEDM/src/ProcessGrains.c`
picks `BestGrainPos` by minimum internal angle (lines 688-695) and then copies that
member's row verbatim — `FinalMatrix[kk][jj+1] = OPs[rown_l][jj]` (line 1041) — for
orientation, position and lattice. Only the strain is recomputed, from the cluster's
pooled spots. So **`spot_aware` matches the C reference and `physics` deviates from it.**
Verified by running the freshly-built C binary on the identical refinement output: the two
grains it shares with `spot_aware` agree to six decimals in X/Y/Z, lattice parameter,
DiffPos, DiffOme and DiffAngle. See §12h.

For this dataset the trade is measurable. Per cluster, member position sd is
(73, 106, 193) µm; the bootstrap SE of the median over 9 members is (31, 53, 99) µm. So
the median **would** be the more precise position estimator here — at the cost of
publishing an X/Y/Z that no single fit actually produced. `spot_aware`'s single-candidate
choice is self-consistent but inherits the full member scatter.

**Knock-on from the 0.5.6 refiner bug (§8a):** since the rep is chosen by `argmin(col 24)`,
and 0.5.6 wrote `col 22 = mean_angle, col 23 = mean_pos, col 24 = mean_ome` (0.5.7 writes
`pos, ome, angle`), a pre-0.5.7 tree selected the representative by **mean ω error** while
believing it was internal angle. That changes which candidate becomes the grain, not just
a printed label. One more reason not to run < 0.5.7.

### 12g. Why the answer used to jump between runs

Before the fixes, a single flipped peak renumbered the spot IDs, which changed cluster
membership and so changed which candidate won the `argmin` — which is why one run reported
(18.28, −5.46, 150.11) µm and another (10.89, 47.83, −108.04) µm for the same grain. Both
were members of the same candidate cluster.

Two separate defects fed that: the peak list was not reproducible (§12b, §12c), and the
fp32 refiner was not moving position at all so the candidates stayed scattered across the
indexer's coarse position grid (§13f). With both fixed the run is bit-reproducible and the
candidate spread is 3–5× tighter (§12e).

If you want the position tighter still, tighten `MarginRadius` / `MarginRadial` /
`MarginEta` from 500 µm (2.5 px) toward ~1 px so `Completeness` regains discriminating
power, and re-check the candidate spread by reading `Results/OrientPosFit.bin` (cols 11-13
are position; col 26 > 0 selects the alive candidates). Do **not** reach for
`Rsample`/`Hbeam` — shrinking the envelope narrows the spread only by clamping candidates
against the bound, replacing an honest ambiguity with a fabricated pile-up (hard rule 9).

---

## 13. Cross-checking against the C reference (`FF_HEDM/src`)

The C chain is the reference implementation. When a python result looks wrong, run it.
Two things make that harder than it should be; both are recorded here so the next session
does not rediscover them.

### 13a. Build it — the shipped binaries are stale

`FF_HEDM/bin/*` on the beamline hosts were compiled in Apr/May 2026 and
`FitPosOrStrainsOMP.c` has changed since. Build fresh. chutoro has no internet, so reuse
the already-fetched dependency tree instead of letting FetchContent phone home:

```bash
cmake -S ~s1iduser/opt/MIDAS_canonical -B $HOME/opt/ffbuild \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DBUILD_OMP=ON \
  -DFETCHCONTENT_BASE_DIR=/home/beams12/S1IDUSER/opt/MIDAS/build/_deps \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build $HOME/opt/ffbuild --target IndexerOMP FitPosOrStrainsOMP ProcessGrains -j 16
```

Note the build rule also copies the binaries into the source tree's `FF_HEDM/bin/`.

`FitPosOrStrainsOMP`'s usage string says `param.txt nBlocks blockNr …`; the code reads
`blockNr = argv[2], nBlocks = argv[3]` (lines 2325-2326), same as `IndexerOMP`. The usage
string is wrong — pass `blockNr nBlocks`.

### 13b. `Spots.bin` is 10 columns now; legacy `IndexerOMP` reads 9

| | `FF_HEDM/src/IndexerOMP.c` | `midas_index/c_src/IndexerUnified.c` |
|---|---|---|
| `N_COL_OBSSPOTS` | 9 (line 63) | 10 (line 100) — col 9 = `ScanNr` |

`midas_transforms.bin_data` writes the **10**-column layout, and
`midas_index/bin/midas_indexer` (built from `IndexerUnified.c`) is the maintained C
indexer the pipeline already calls. Feed the 10-column file to legacy `IndexerOMP` and it
strides through the array wrongly: on this dataset it reported

```
WARNING: SpotId 1177.000000 not found in spots file! Ignoring this spotID.   (×168 of 189)
```

and wrote an all-zero `IndexBest.bin`, after which `FitPosOrStrainsOMP` exits in 0.01 s and
`ProcessGrains` says *"OrientPos file was not found … nothing was indexed"*. **That cascade
is a format mismatch, not a parameter problem** — do not go tuning `Completeness` in
response to it. The tree documents the difference in
`midas_index/dev/c_indexer_diff.md`.

To run the legacy chain anyway, drop col 9 (row order is preserved and `Data.bin`/
`nData.bin` store row indices, not byte offsets):

```python
a = np.fromfile("Spots.bin", dtype=np.float64).reshape(-1, 10)
np.ascontiguousarray(a[:, :9]).tofile("Spots9.bin")
```

After that the warning count drops to 0.

### 13c. `ProcessGrains` needs no re-indexing

It reads only `Results/{Key,OrientPosFit,ProcessKey}.bin`, `Output/FitBest.bin`,
`SpotsToIndex.csv` and `InputAllExtraInfoFittingAll.csv` — never `Spots.bin`. So you can
point it straight at a python pipeline's output and compare grain reduction in isolation:

```bash
cd <copy of layer dir> && $HOME/opt/ffbuild/bin/ProcessGrains -paramFN paramstest.txt -nCPUs 16
```

### 13d. What that comparison found (Au3_cubes_ff_000008, 2026-07-30)

Identical refinement input, C vs python `spot_aware`:

| | C `ProcessGrains` | python `spot_aware` |
|---|---|---|
| grains | **6** | **2** |
| shared grains' X/Y/Z, a, DiffPos/Ome/Angle | — | identical to 6 decimals |
| GrainRadius (grain 80) | 114.620659 µm | 20.775146 µm ← **was wrong** |

1. **The reduction rule agrees.** Both take the minimum-internal-angle member and copy its
   fit verbatim. Nothing to fix.
2. **The clustering does not — and C is the one that is wrong (§13g).** C walks the
   *shared-spot adjacency* in `ProcessKey.bin` and merges neighbours with misorientation
   < 0.4° (`FindInternalAngles`, `ProcessGrains.c:140`), `MinNrSpots` defaulting to 1.
   python Phase-1 clusters globally on misorientation at `MisoriTol` 0.5°, then runs a
   Pass-A spot-overlap merge. C keeps 6 where python keeps 2 — but those 6 are only **two
   orientations**, and python's 2 are the physical answer. Resolved in §13g.
3. **GrainRadius was a genuine python bug — now fixed.** `midas_process_grains` built its
   per-spot radius lookup from `Radius_*.csv`. That file and `ExtraInfo.bin` hold the same
   spots numbered 1..N but in **different orders** (`calc_radius` renumbers, then
   `bin_data` sorts by `(RingNumber, Omega, Eta)` and renumbers again), and every id
   downstream of the binner is in the ExtraInfo space. The join therefore averaged ~112
   arbitrary spots, so every grain came out near the *global* mean radius (~22 µm) instead
   of its own. Fixed to read `ExtraInfo.bin` col 3 keyed by col 4 — the same source the
   refiner uses. python now reports 114.620677 / 99.962755 µm against C's
   114.620659 / 99.962738. Locked by
   `midas_process_grains/tests/test_spot_radius_id_space.py`.

**Reported grain sizes from any run before this fix are too small — by 5.5× on this
dataset.** The error is not a constant factor; it is "your grain's radius was replaced by
the sample-wide average", so it compresses the whole size distribution toward the mean.

### 13e. If `IndexerOMP` is slow, your binned files are in the wrong format

**`IndexerOMP` is not slow.** On this dataset it indexes 189 seeds in **2.03 s** on 16
threads (~94 seeds/s), comparable to the unified `midas_indexer` at 3 s, and finds the
same 20 candidates. Full legacy chain end to end: index 2.03 s → `FitPosOrStrainsOMP`
0.29 s → `ProcessGrains` 0.023 s.

If you see it take minutes, the cause is §13b — the pipeline's binned files are in the
PF/unified layout and legacy FF C reads a narrower one. **Three widths differ, and
converting only one of them leaves the indexer reading garbage:**

| file | PF / unified (what the pipeline writes) | legacy FF (what `IndexerOMP` reads) |
|---|---|---|
| `Spots.bin` | `(N, 10)` float64 | `(N, 9)` float64 — col 9 = `ScanNr` |
| `nData.bin` | `(B, 2)` **int64** (count, offset) | `(B, 2)` **int32** |
| `Data.bin` | `(T, 2)` **int64** (rowno, scanno) | `(T,)` **int32** rowno |

`nData.bin` is the one that bites. `IndexerOMP.c:122` does `nspots = ndata[Pos*2]` on an
`int *`; against an int64 array a bin lookup lands on the wrong bin and frequently reads
an **offset as a count** — up to 220,925 here instead of ≤24. The inner loop then scans
~10⁴× too many rows. That is both the slowdown *and* the reason nothing matches, from one
cause. Diagnose it by checking the file against the bin count:

```python
total_bins = n_ring_bins * ceil(360/EtaBinSize) * ceil(360/OmeBinSize)
nd = np.fromfile("nData.bin", dtype=np.int64).reshape(-1, 2)
assert nd.shape[0] == total_bins          # wrong dtype if this fails
assert nd[:, 0].sum() == n_data_entries   # counts must sum to Data.bin's length
```

Read at the correct width the numbers are self-consistent (counts ≤ 24 summing to 220,925,
offsets non-decreasing); at the wrong width they are absurd (counts up to 220,925 summing
to 9.4e12, all offsets 0). Converter: `utils/pf_to_ff_bins.py` — it asserts all three invariants before
writing.

**Corrected 2026-07-30.** An earlier revision of this section claimed indexing cost scaled
with `Rsample`/`Hbeam` and tabulated 16 s at 200 µm vs >420 s at 2000 µm. Those timings
were measured on the malformed `nData.bin` and are **void** — the apparent envelope
sensitivity was the corrupt bin lookup, not the search space. With correct input the
indexer takes 2.03 s at `Rsample 2000`. Do not use runtime as an argument for touching the
envelope; see hard rule 9.

**Resolved:** legacy `IndexerOMP` indexing 0/189 had the same single cause. With correctly
converted binaries it indexes 20/189 — the same 20 the unified indexer finds.

### 13f. The refiner returned its input in float32 — a SCALING bug, not precision

Same seeds in (the C `IndexerOMP`'s `IndexBest.bin`), three refiners out:

| refiner | \|pos\| mean | DiffPos med | median \|Δpos\| vs C | moved from seed |
|---|---|---|---|---|
| C `FitPosOrStrainsOMP` | 52.5 µm | 193.89 | — | 158.3 µm |
| py **float64** (cpu & cuda) | 71.0 / 69.5 µm | 199.1 | **13.4 µm** | 149.7 µm |
| py **float32** (cpu & cuda) | 227.1 µm | 231.9 | 158.2 µm | **0.0 — 20/20** |

In float32 the refiner did not refine position **at all** — it emitted the
seed. cpu and cuda agree to 3 s.f. within each dtype.

**The cause is parameter scaling, not arithmetic.** The optimizer carries
position as `pos_scaled = pos / pos_scale`, and the shipped `pos_scale = 100`
left the gradient blocks wildly unbalanced. Measured on the synthetic fixture:

| `pos_scale` | \|g\|position | \|g\|euler | ratio | fp32 error vs truth |
|---|---|---|---|---|
| **1e2 (shipped)** | 95.8 | 1.47e5 | **1537** | **154.27 µm** |
| 1e3 | 958 | 1.47e5 | 154 | 0.75 µm |
| 1e4 | 9581 | 1.47e5 | 15.4 | 0.013 µm |
| **1e5** | 9.58e4 | 1.47e5 | **1.5** | **0.004 µm** |

L-BFGS applies **one step length to the whole concatenated vector**, so
position advanced ~1500× less per step than orientation. fp64 has the mantissa
headroom to keep resolving that; fp32, whose gradient carries ~1e-4 relative
rounding noise (~600× eps), does not — the position component of each step
lands under the noise. After the first orientation-dominated step the
strong-Wolfe line search finds no further descent and returns t = 0 forever:

```
f64: 177.35 → 49.08 → 5.02 → ... → 2.27e-08   (descends; final error 0.00 µm)
f32: 177.36 → 70.25 → 70.25 ×9                (frozen EXACTLY; error 154.27 µm)
```

Everything else is clean, which is how the scaling was isolated: the fp32
gradient **direction is exact** (`cos(g64,g32) = +1.00000000` on every block),
fp32 **resolves** loss decreases down to t = 1e-8 and position steps of
0.0001 µm, and **LM fails too** — so it is not specific to L-BFGS. The error
is also *not* Lsd-driven (tested at 10 / 100 / 1000 mm: ~1e-4 throughout).

**Fix.** `midas_fit_grain.refine_block` now derives the scale from the entry
gradient — `s = |g_other| / |g_pos_µm|`, the value that makes the ratio 1 —
instead of a fixed 100. It is a pure reparameterization, so the sample-cylinder
clamp (which divides the µm bounds by `pos_scale`) stays consistent. Result,
over three seed offsets:

| seed offset | f64 old | f64 auto | f32 old | f32 auto |
|---|---|---|---|---|
| (90, −60, 110) | 0.0031 | **0.0002** | 154.27 | **0.0032** |
| (−200, 150, −80) | 0.0261 | **0.0002** | 67.62 | **0.0041** |
| (15, −5, 25) | 0.1054 | **0.0002** | 29.58 | **0.0049** |

fp64 improves too (15–500×).

**Confirmed on the real dataset** (189 C-indexer seeds, vs the C reference):

| config | median \|Δposition\| vs C | DiffPos median |
|---|---|---|
| fp32, fixed `pos_scale`=100 | **158.24 µm** | 231.9 |
| fp64, fixed `pos_scale`=100 | 13.38 µm | 199.07 |
| **fp32, auto** | **13.65 µm** | **196.94** |
| **fp64, auto** | 13.96 µm | 199.11 |
| *(C reference itself)* | — | *193.89* |

End to end through the pipeline, the two reported grains moved to where they
should be — near the rotation centre, and onto the C chain's answer:

| grain | before the fix | after | full C chain |
|---|---|---|---|
| 80 / 1000 | (10.9, 47.8, **−108.3**) | **(2.7, 2.0, −5.3)** | (−4.6, 4.8, −20.8) |
| 185 / 1177 | (−100.8, −8.9, 51.6) | **(−10.3, −22.5, −3.5)** | (−7.3, −27.8, 4.0) |

Distance from the C reference fell 99 → 17 µm and 107 → 10 µm, and DiffPos to
190.1 / 182.2 against C's 188.7 / 181.0. Refinement also got *faster*
(51.9 s → 16.4 s at fp64) because balanced blocks converge sooner.

`RefinementConfig.dtype` still defaults to **float64** — cheap at this scale and
the conservative choice — but fp32 is now a supported trade for throughput on
large runs, not a correctness risk.

**The silence was the second bug**, and it is fixed independently:
`refine_block` reports `max_position_move_um` / `median_position_move_um` /
`n_unmoved_position`, `midas_fit_grain/driver.py` emits `UNREFINED-POSITIONS: …`
when no grain moved more than **px/1000**, and
`midas_pipeline/stages/refinement.py` re-surfaces that into the run log.
(px/1000, not one pixel — a healthy fp64 fit moved only 0.77 px here, so a
one-pixel threshold would flag good fits; fp32 moved 2.5e-06 px.)

**If you see that warning, the grain positions in that run are indexer seeds,
not fits.** Do not quote them.

NOTE the PF scanning path (`refine.py::refine_grain`, used by `scan_driver`)
is deliberately left on the fixed scale: `position_mode="fixed"` locks the
voxel to the scan grid there, so position is not a free parameter, and PF
carries C-parity gates. Apply the same equilibration if you enable
`position_mode="voxel_bounded"`.

### 13g. RESOLVED — C's 6 grains are 2 orientations, and one is a Σ3 twin

C reports 6, python `spot_aware` reports 2, from identical refinement. Measuring pairwise
misorientation settles it: the 6 are **two families**, not six orientations.

```
           1000      899     1014      918     1177     1176
 1000         -    0.638    0.079    0.203   59.973   59.850
  899     0.638        -    0.570    0.448   59.918   59.959
 1014     0.079    0.570        -    0.127   59.995   59.882
  918     0.203    0.448    0.127        -   59.977   59.899
 1177    59.973   59.918   59.995   59.977        -    0.620
 1176    59.850   59.959   59.882   59.899    0.620        -
```

**C is over-segmenting.** 1000 and 1014 differ by **0.079°** and share **70 %** of their
spots (Jaccard 0.697); 1000/1014/918 are mutually < 0.21°, comfortably inside C's own 0.4°
merge threshold. They stayed separate only because `FindInternalAngles` requires
misorientation < 0.4° **AND** shared-spot adjacency in `ProcessKey.bin` — near-identical
orientations that miss that adjacency link never get compared. The duplicates are also the
worse fits: the two python-matching grains (1000, 1177) have the lowest DiffPos
(188.7, 181.0) and Conf 1.000, while the extras run 206–333 with Conf 0.875–0.983.

python's two land on C's best members at **0.031°** and **0.018°**, and no C grain is more
than 1° from a python counterpart. **python `spot_aware` gives the right answer here.**

**The cross-family ~60° is a Σ3 annealing twin**, and it is about ⟨111⟩ — every one of the
eight cross pairs, several within 0.01–0.11° of the ideal axis. So the specimen is one
grain plus its twin.

#### Verified against the raw frames

Orientation algebra alone would not settle this — a phase can score highly on reflections
it *borrows* from another (exactly what killed a Zn/Cu epitaxy claim in the Fuller Laue
campaign). The honest test is whether the twin has spots that are **its own**:

```
parent 1000: 112 spots     twin 1177: 116 spots     shared: 43
                                    -> 73 spots belong to the twin alone
```

Those 73 were checked directly in the dark-subtracted frames, at the predicted pixel:

| spot set | SNR > 5 | median SNR | min |
|---|---|---|---|
| **twin-only** | **30/30** | **1718** | 351 |
| parent-only | 30/30 | 1634 | 135 |
| shared | 30/30 | 1736 | 820 |

The pixel convention was **not assumed** — it was calibrated on spots already known to be
real, and came out unambiguous: `img[z, y]` with `row = BCZ + z/px`, `col = BCY − y/px`
gives median SNR **1469** where all seven alternative axis/sign combinations give ~1.0.
Residual registration is ~1 px (median +0.77 px in column, 1.03 px radial), because this
check ignores the distortion terms p0–p14 and the detector tilt; the ±6 px signal box
absorbs that.

Figure + generators: `twin_vs_frames.png`, `twin_vs_frames.py`, `twin_frames_fig.py`
(beamtime `analysis/au3_cubes_ff_000008/`).

**Method worth reusing:** when two codes disagree on grain count, ask (a) pairwise
misorientation — are these even different orientations? (b) spot-set Jaccard — are they
explaining the same observations? and (c) for any orientation you want to believe, does it
have *exclusive* spots with real intensity? (a)+(b) separate over-segmentation from real
grains; (c) stops you believing a phase that only ever borrows peaks.
