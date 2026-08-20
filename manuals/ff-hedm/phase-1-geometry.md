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

> **20-ID Varex:** the dark is in **`/exchange/bright`**. `/exchange/dark` exists
> in the same file and is all zeros, so pointing at the obvious name leaves the
> ~1500-count pedestal in the image. Set `darkLoc /exchange/bright`.
>
> Separately, `exchange/dark` **inside the zarr** also reads all zero on these
> datasets, and that one is harmless: the data was already dark-subtracted at zip
> time (raw frame mean ~1850 → zarr ~0.6). Check the data frames before chasing
> it — see the halt-condition wording in the spine.

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

### 3f. `ImTransOpt` — the detector flips

A list of codes applied **in order** to every frame, before anything else sees
it (`midas_peakfit/midas_peakfit/preprocess.py`,
`apply_image_transformations`):

| code | effect | as indices |
|---|---|---|
| 1 | flip horizontal, along Y / the row axis | `image[l, m] := image[l, N-m-1]` |
| 2 | flip vertical, along Z / the column axis | `image[l, m] := image[N-l-1, m]` |
| 3 | transpose | `image[l, m] := image[m, l]` |

`ImTransOpt 2` on 20-ID Varex; establish it per detector, not per run.

**It is a convention, like the ω sign, and it belongs in the same category of
danger.** A wrong flip does not fail. It mirrors the reconstruction, and a
mirrored microstructure has a perfectly normal grain count, normal completeness
and normal strain. You cannot see it in `Grains.csv`.

**The rule is that calibration and reconstruction must use the *same* value.**
A mismatch mirrors the geometry relative to the fit, and then the two disagree
in a way that no downstream number reveals.

#### Why the calibrant will not save you

A powder pattern is concentric rings. Flipping it about either axis maps rings
onto rings, so the fit converges just as happily on the mirrored image — the
ring overlay (§5d) looks *correct*, because it is correct, for the mirrored
geometry.

Measured on 20-ID CeO2, same exposure, only the transform differing
(Lab Notebook §8f):

| `ImTransOpt` | BC_y (px) | BC_z (px) | strain | gate |
|---|---|---|---|---|
| **2** — correct | 1450.86 | 1467.46 | 58.2 µε | PASS |
| *omitted* | 1450.90 | **1411.59** = 2879 − 1467.46 | 55.6 µε | PASS |
| **1** — wrong axis | **1427.98** = 2879 − 1450.86 | **1411.62** | **47.2 µε** | PASS on strain |

Both wrong geometries scored a **better** strain than the correct one, and the
mirror is exact: each affected coordinate lands on `N-1 − BC`. Strain alone
would have chosen the worst of the three. This is the concrete reason rule 6's
gate is necessary but not sufficient, and why the BC-mirror check below is a
gate in its own right rather than advice.

#### How to establish it, and how to check it

1. **Inherit it.** If a previous reconstruction on this detector worked, take its
   value. This is the normal case and the only one that needs no thought.
2. **Check against a prior beam centre.** A refined BC landing within a pixel or
   two of `N-1 − BC_prior`, rather than near `BC_prior`, is the mirror
   signature and is decisive. `midas-calibrate-v2 --mode ff` runs this check
   automatically and **fails the gate** when it fires, precisely because strain
   will not.
3. **Use a physically asymmetric feature.** The beamstop shadow, a dead region,
   or a panel edge sits somewhere known on the real detector. Locate it in the
   transformed frame and confirm it is where the hardware says it is.
4. **Otherwise stop and ask.** With no prior geometry and no asymmetric feature,
   a single powder exposure genuinely cannot tell you, and guessing costs a
   mirrored dataset that looks fine.

#### The reading trap

`CalibrationParams` does **not** expose `ImTransOpt` as an attribute — the key
lands in `.extra`. Reading it the obvious way returns nothing and the caller
then silently calibrates with *no* transform at all, which is how the mirrored
fit above was produced. Read it from `.extra`, or from the parameter-file text.

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

Package: `midas_calibrate_v2`. Entry point `calibrate()` — image + λ + pixel size
+ calibrant name, everything else auto-seeded.

**Prefer the one-call route**, which does §5a–§5g and writes the parameter file:

```bash
midas-calibrate-v2 <template paramstest> --mode ff \
  --image <calibrant file> \
  --dark-group exchange/bright \        # 20-ID Varex; omit or change elsewhere
  --initial-lsd 900000 \
  --raw-folder <SAMPLE data folder> \
  --output calib/ps_calibrated.txt
```

The positional file is a **template**: thresholds, ring numbers, ω scan, lattice
and file naming are carried through; geometry, distortion, `px` and **`RhoD`**
are replaced. It is therefore correct to hand it the very file that was failing.
It writes a ring overlay every time and exits non-zero above the 100 µε gate.

It also fixes three things that are easy to get wrong by hand and do not raise:
the generic `--image` HDF5 loader takes the file's *first top-level key* (on a
`.vrx.h5` that is the `WM` metadata group, not the data); the beam centre must
be auto-seeded, never guessed; and `RhoD` must be rewritten (§6d).

Omit the template entirely and pass the experiment keys as flags (`--px`,
`--lattice`, `--space-group`, `--omega-start/--omega-step`, `--ring-thresh`, …)
when there is no previous reconstruction to inherit from. Note that
`--ring-to-index` then defaults to the *lowest* ring given, which is often not
the strongest: on ti7al, ring 1 gave 1630 seeds and 173 grains where ring 3 gave
4512 and 208.

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

### 5h. `tx` and `Wedge` — the two the calibrant cannot see

Run this **after** a first reconstruction, then reconstruct again with the file
it writes. It is the one geometry step that needs grains rather than powder.

Neither parameter is recoverable from a calibrant, for reasons of symmetry
rather than precision:

* **`tx`** is a rotation of the detector about the beam. Powder rings are
  concentric, so rotating them about their own centre changes nothing. `tx` is
  *structurally* invisible to a calibrant, and `midas-calibrate-v2` therefore
  seeds it at 0 and never refines it.
* **`Wedge`** is the departure of the rotation axis from perpendicular. A still
  image never sees the rotation axis at all.

Both act on single-crystal spots followed across ω, which is what a grain list
is.

```bash
midas-joint-ff-calibrate grain-tx \
  --paramstest <calibrated params> \
  --layer-dir  <result>/LayerNr_1 \
  --refine tx,Wedge --max-grains 100 --max-iter 120 \
  --out ps_txwedge.txt
```

**Measured, 20-ID ti7al layer 1** — feeding the result back and re-running:

| | before | after |
|---|---|---|
| grains | 208 | 226 |
| grain-Z scatter (sd) | 152.6 µm | **76.4 µm** |
| completeness (median) | 0.580 | 0.630 |
| X / Y scatter | 271 / 265 µm | 273 / 272 µm — unchanged |

Z halving while X and Y stand still is the signature of a real geometry
correction: it tightened the badly-conditioned coordinate and left the
well-conditioned ones alone. A fit that moved all three would be absorbing
error.

**`--refine` is a freeze/thaw list**, not a fixed pair. Also available: `Lsd`,
`BC_y`, `ty`, `tz`, and the distortion harmonics (`iso_R2/R4/R6`, `a1..a6`,
`phi1..phi6`). Naming any of `BC_y`/`ty`/`tz`/distortion switches the residual
to a raw-pixel path that recomputes the observations at the trial geometry,
Stage-4 spline included.

`BC_z` is **refused**: a vertical beam-centre shift is degenerate with a global
shift of the grain Z positions, which is exactly the coordinate far-field
determines worst.

**`--fix KEY=VALUE` pins a parameter to a value you know** — a lattice measured
on a standard, grain positions a focused beam already defines — and holds it
there while the rest refines. That is different from leaving it out of
`--refine`, which keeps whatever the parameter file said. A single row
broadcasts to every grain:

```bash
  --fix grain_lattice=4.1569,4.1569,4.1569,90,90,90     # LaB6
```

#### Two checks before believing any of it

**`matched spots` must be a large fraction of the grains' spots.** A handful
means the predicted pattern is not landing on the data at all — nearly always
the ω-scan keys.

**No refined value may sit on a bound.** ≥ 0.1.9 names it and exits 1, because
this has produced plausible-looking wrong answers three times: `Wedge` at +5.0
from a misread ω key, `iso_R4`/`iso_R6` at +0.05 from six grains.

Conditioning is checked up front and will warn you: `tx` needs ω-coupling
**across** grains to be separable from each grain's own orientation, so fewer
than ~5 grains makes it poorly determined; the distortion is a detector-wide
field and wants ~50+, which is why it belongs on the calibrant.

#### What this cannot give you

`Lsd` is refinable here, but the data determines the **product** `Lsd·λ/a`, not
`Lsd`. Measured on nf709 (9077 grains) by sweeping the assumed cell:

| assumed `a` (Å) | fitted `Lsd` (µm) | final cost |
|---|---|---|
| 3.5960 | 895 241 | 1.0666e9 |
| 3.5990 | 896 006 | 1.0663e9 |
| 3.6020 | 896 771 | 1.0661e9 |

`Lsd` tracks the assumed lattice **linearly** — about 249 µm per mÅ — while the
cost is flat to 0.05 % (Lab Notebook §8e). The fit is not choosing `Lsd`; you are, through the
lattice and through λ. Breaking that degeneracy needs several detector
distances with known relative travel (`midas-calibrate-v2 --mode multi
--lsd-offsets`), which is the only route here that makes λ identifiable rather
than asserted.

---
