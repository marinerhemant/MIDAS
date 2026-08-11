# Phase 1 — Geometry: ω sign, metadata, energy, calibration

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

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

Verified on `bt_1id_jul26`: all **7297** FF rows read `aero`.

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

| what | where (`bt_1id_jul26`) |
|---|---|
| frames | `/gdata/dm/1ID/2026/bt_1id_jul26/data/ge5/` |
| acquisition logs | `~s1iduser/new_data/bt_1id_jul26/` |
| per-frame FF par | `<logs>/bt_1id_jul26_FF.par` |
| energy monitor | `<logs>/fastsweep_Emon.txt` |
| spec log | `<logs>/FullLog.log` |
| macros | `<logs>/macros_<user>/` |

### 3b. Par-file field map (1-ID FF)

Positional, whitespace-separated. Verified against `bt_1id_jul26_FF.par`:

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

Measured signature on `bt_1id_jul26` GE5: frame 0 sits ~1.5 % low in baseline versus
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

**The filename is not the energy.** On `bt_1id_jul26` the CeO₂ files are named
`..._96keV_...` and the scan was taken at **95.0 keV**.

Sources, in order of trust:

| source | `bt_1id_jul26` | verdict |
|---|---|---|
| `instrument/HEM/Energy` (HDF5) | 95.0 | **use this** |
| `fastsweep_Emon.txt` field 6 (`E_HEM`) | 95.0000 | corroborates |
| spec `FullLog.log` → `Energy (keV):` | 95 | corroborates |
| `instrument/InsertionDevice/IDEnergy` | 95.055 | undulator setting, not the mono |
| `instrument/HRM/Energy` | 78.39 | **different monochromator — ignore** |
| the filename | "96keV" | **stale string** |

`fastsweep_Emon.txt` columns come from `macros_<user>/E_mon.mac`: field 2 is a foil µt, field
6 is `epics_get("1id:userTran3.A")` = the HEM energy readback. **Rows where the last two
columns are `0.000 0.000` had the foil out** (air) and carry no absorption information.

λ[Å] = 12.398419843320026 / E[keV]. At 95.0 keV, λ = 0.130510 Å.

### 4b. Distance — `DetZ` is a stage readback, not `Lsd`

`instrument/DMS/DetZ` is the detector translation-stage position. Its zero is not the
sample rotation centre.

**Measured on `bt_1id_jul26`:** `DetZ` = 1485.00 mm, calibrated `Lsd` = **1666.2 mm** —
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

`bt_1id_jul26` CeO₂ reference: rings sharp and complete in azimuth, innermost at
R ≈ 348 px about the fitted BC, beamstop shadow at ≈ (1019, 1076), signal ~54 counts above
a ~2019-count dark after frame-0 removal.

### 5b. Check the ring assignment before you trust the fit

Ring-radius **ratios** depend only on the lattice — λ and `Lsd` cancel. This identifies
which ring the innermost observed one is, independently of any geometry:

```
R_i / R_1  =  tan(2θ_i) / tan(2θ_1)
```

Measure radii from a radial profile about the seeded BC, and compare. On `bt_1id_jul26`
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

Observed on `bt_1id_jul26` (same image, λ the only change): 95 keV → 19.4 µε,
96 keV → 72.7 µε. Suggestive, and it agreed with the beamline's own confirmation of
95 keV — but treat it as corroboration, not proof.

### 5f. Use the 0/180 pair if you have one

A calibrant measured at two rotations 180° apart gives an independent repeat of the same
detector geometry; the spread between the two fits is an honest uncertainty. On
`bt_1id_jul26`:

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
