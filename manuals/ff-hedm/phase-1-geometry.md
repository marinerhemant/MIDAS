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
| *there is no par file* | 20-ID and anything else without one | **§2b** — settle ω and the detector mirror together |

Verified on `bt_1id_jul26`: all **7297** FF rows read `aero`.

Worked example — `Au3_cubes_ff_000008`. The par logs 1441 frames running
ω = −180.25 → +179.75 at step **+0.25**. Negating gives raw frame 0 at +180.25; dropping
the throwaway frame 0 (§3e), the **first frame actually used** is at +180.00, and that is
what `OmegaStart` names:

```
OmegaStart 180.00      # omega of the first frame USED, i.e. AFTER SkipFrame (§3e)
OmegaStep  -0.25
SkipFrame  1           # raw frame 0, at +180.25, is discarded
OmegaRange -180 180
```

> **This example said `OmegaStart 180.25` in revisions of this file before 2026-08-31, and
> that is a silent one-step (0.25°) ω error in every reconstruction that followed it.**
> `OmegaStart` is the ω of the first frame you want to **USE**, post-skip — §3e is the
> authority and is code-cited; §2 and §10 were the two places that contradicted it.
>
> **Symptom, and how to check an existing run.** Nothing errors and nothing in
> `Grains.csv` moves: it is a rigid 0.25° rotation of every orientation about ω, exactly
> the cost already documented for the 20-ID zero-point offset in §3e (≤ 2.2 µm of position
> at r = 500 µm; every difference — misorientation, relative orientation, strain —
> unaffected). Read the zarr instead. The zipper stores
> `measurement/process/scan_parameters/start = OmegaStart − SkipFrame·OmegaStep`
> (`ff_zip.py:250`), so on this sweep the stored `start` **must equal the negated raw
> frame 0, +180.25**. Measured on the 1441-frame `aero` sweep: `OmegaStart 180.25` stored
> 180.50 — one step past raw frame 0, wrong — and `OmegaStart 180.00` stored 180.25,
> correct.
>
> ```python
> z = zarr.open("<result>/LayerNr_1/<stem>.MIDAS.zip", mode="r")
> float(z["measurement/process/scan_parameters/start"][0])   # == negate(raw frame 0 omega)
> ```
>
> [`RUNBOOK.md`](RUNBOOK.md) §R2b has carried `OmegaStart 180.00` for this same detector
> and an identical sweep throughout; where the runbook and this example disagreed, the
> runbook was right.

**Why you cannot check this later.** A sign flip in ω mirrors the reconstructed
microstructure. Completeness, grain counts and internal angles are all unchanged. Nothing
inside the reconstruction catches it.

**Corroboration:** `NF_HEDM_Handbook.md` §2 reaches the same rule from the NF par of the
same beamline, and the bundled NF reference paramfile carries `OmegaStart 180` /
`OmegaStep -0.25` for a 360° aero scan.

---

### 2b. No par file — settle the ω sign and the detector mirror TOGETHER

**20-ID HT-HEDM has no par file at all.** Metadata lives in EPICS NDAttributes inside
each `.vrx.h5` (§3b-2). There is no field 9, so §2 has no input — and the problem is
worse than one missing convention, because **the ω sign and the detector mirror are
coupled and neither the calibration nor the grain list can break either one:**

* **A powder calibrant cannot see the mirror.** Rings are centro-symmetric, so
  `ImTransOpt 1` (flip-Y) and `ImTransOpt 2` (flip-Z) converge with the *same* `Lsd`,
  the same tilts and the same strain. Only the refined beam centre differs, and it
  lands exactly on `N-1 − BC` — which is why it reads as a plausible fit rather than a
  failure. `midas_calibrate_v2/pipelines/ff_calibrate.py` `_check_not_mirrored` exists
  for precisely this and is the gate that now warns. Worse, on `bt_20id_jul26b`
  every wrong variant scored a *better* strain than the correct one (47.2 and 55.6 µε
  against 58.2 — §3f, Lab Notebook §8f).
* **A grain list cannot see the ω sign.** It mirrors the microstructure with
  completeness, grain count and internal angles unchanged (§2).
* So a **wrong pair is self-consistent**, and it is reached by exactly the route you
  would take. Measured: the two pre-existing 20-ID parameter files disagree —
  `ps_au.txt` is `ImTransOpt 1` with ω positive, the `bt_20id_jul26b` run is
  `ImTransOpt 2` with ω negated — and **both produced plausible reconstructions**.
  Adopting either is not evidence.

**Do not proceed on a fitted quantity.** Every argument below is either a physical fact
about the instrument or a parameter-free relation between observed spots. Use all three
you can get; they are independent of each other and of the geometry fit.

**Argument 1 — the stage's physical sense fixes ω.** Ask which way the rotation stage
turns, viewed from above, and confirm it against the per-frame readback's *direction*.
On `bt_20id_jul26b` the aero turns **clockwise viewed from above** ⇒ ω_MIDAS = −ω_logged,
so a scan commanded −180.25 → +180.25 at step +0.25 becomes `OmegaStart 180.0`,
`OmegaStep -0.25` after negating and dropping the throwaway frame 0 (§3e).

> **Read the step from `scan_parameters`, not from the per-frame readback.** The readback
> is asynchronous EPICS polling: on the alumina scan it alternates ≈0.2246 / 0.2695, stalls
> (Δ = 0) and catches up (Δ up to 3.26°). Its *mean* is 360.234/1441 = 0.25006°/frame,
> matching the commanded 0.25 exactly. The jitter is the PV, not the stage — see
> §3e's per-frame roll measurement, which bounds real stage jitter at 0.0031°.

**Argument 2 — an asymmetric feature in the frame fixes the flip axis.** The beamstop is
the usual one, because its support tells you which way is physically down. On
`bt_20id_jul26b` the rod is supported **from below**, so the high row index is physically
down ⇒ flip Z ⇒ `ImTransOpt 2` (code 2 is `image[::-1, :]` in
`midas_peakfit/midas_peakfit/preprocess.py` `apply_image_transformations` — §3f);
confirmed by looking at the frame after `flipud` and seeing the rod point to low Z. A
dead region, a panel notch or a known-asymmetric mask serves equally well. This fixes
**z**.

**Argument 3 — a within-grain Friedel quadruplet fixes y, given ω.** Parameter-free, and
it uses only the spots you already have. Each reflection satisfies Bragg twice per turn,
so **G** and **−G** give four spots; the two from the same **G** sit at equal z and
opposite y, separated by 2α with `cos α = Gx0/ρ`. Pair them **inside one grain, over
`SpotMatrix.csv`**, and compare the observed |Δω| against the prediction from each spot's
own `G_z`.

Measured on the gold scan, 24 pairs: median residual **+0.389°** against an ω step of
0.25°, 21 of 24 within ~2.3°, and the **sign fraction ω(+y) > ω(−y) = 0.0417, i.e. 23 of
24** — a definite handedness.

> **Two ways to get this wrong, both hit here.**
> **(a)** Pair over the ungrouped `InputAll.csv` and you get 22,588 "pairs" at sign
> fraction 0.494 — pure noise. Friedel pairing is a *within-grain* operation.
> **(b)** Note what the test proves: flipping **both** y and ω flips the sign twice, so
> it cannot pick the absolute pair on its own. It fixes **y given ω**. That is why it is
> argument 3 and not argument 1 — the chain is stage sense fixes ω, the beamstop fixes z,
> the quadruplet fixes y.

**Cross-check on the pixels once you have a candidate convention.** Grid-search all eight
flip/frame combinations and score each by how much of a spot's reported `IMax` you actually
recover at the patch it predicts. On the gold scan the winner —
`row = 2879 − ZCen`, `col = YCen`, `frame = rint((ω − 180.25) / −0.25)` — recovered
**44,539 counts** against a reported `IMax` of 50,476, while the un-flipped convention
recovered **3**. That is not a marginal preference, and it is a cheap last gate before
committing.

**What does not settle it.** A near-field anchor on the same sample does not: a mirror
**preserves a cube's radius**, and the NF azimuth convention was itself fitted to the
reconstruction, so it cannot arbitrate without being re-derived from raw shadow geometry.
Do not reach for it.

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

**20-ID HT-HEDM has no acquisition-log folder and no par file.** Everything is EPICS
NDAttributes written inside each `.vrx.h5` — see §3b-2. The data tree is
`/gdata/dm/20ID/HT_HEDM/<cycle>/<proposal>/data/varexD/`, one folder per scan, each
holding a `_dark_before`, the data file, and a `_dark_after`.

> **Access.** The DM tree is group-restricted (`drwxr-x---+`) and the owning group is
> per proposal. A beamline account that reads it today may not tomorrow: on
> `nfdev_jul26` the `s1iduser` ACL entry disappeared mid-campaign, and every stage that
> touches raw data had to move to `s20hedm`. `ProxyJump` cannot reach that account —
> nest the ssh instead. See [`RUNBOOK.md`](RUNBOOK.md) §R1.

### 3b. Par-file field map (1-ID FF)

Positional, whitespace-separated. **The head of the row is stable across beamtimes; the
tail is vintage-dependent and shifts by one.** Pin the tail columns on your own file
before using them — the two verified mappings are below.

| field | meaning | verified on |
|---|---|---|
| 1–5 | date stamp | both |
| 6 | detector tag (`GE_AD`) | both |
| 7 | scan name | both |
| 9 | **rotation stage** — the `aero` test (§2) | both |
| 10, 11 | sweep bounds (logged ω) | both |
| 17 | **per-frame ω** (logged) | both |

**These six transfer**, which is why the ω-sign rule (§2) worked unchanged on a
three-year-old file. The tail does not:

| | exposure (s) | **file number** | **frame index** (1-based) | verified on |
|---|---|---|---|---|
| **2026 vintage** | 19 | 20 | 21 | `bt_1id_jul26_FF.par` |
| **2023 vintage** | **18** | **19** | **20** | `bt_1id_mar23_FF.par` (42 fields), `/gdata/dm/1ID/2023/bt_1id_mar23/data/metadata/bt_1id_mar23/` |

Everything after field 17 is therefore **off by one between these two beamtimes**, and
neither row is a property of the station. Reading a 2023 file with the 2026 map silently
returns the wrong column: field 21 is not the frame index there, and an `awk` selection on
`$20` selects on the frame index instead of the file number — which matches nothing, or
matches the wrong scan.

**Pin the columns empirically, on your own file.** The frame index is the column that runs
1, 2, 3, … and resets per file; the file number is the column that is constant within a
scan and matches the six-digit suffix of the image filename; the exposure is the small
constant float. One pass over a few rows settles all three:

```bash
awk '{print NF; exit}' <logs>/<beamtime>_FF.par                 # 42 on bt_1id_mar23
head -1 <logs>/<beamtime>_FF.par | tr -s ' ' '\n' | cat -n      # field number -> value
awk 'NR<4{for(i=17;i<=NF;i++) printf "%d:%s ", i, $i; print ""}' \
    <logs>/<beamtime>_FF.par                                    # watch which one counts 1,2,3
```

Only then write the extraction. **This next line is an example to verify, not to copy** —
it is the 2026 mapping, and on a 2023 file the two column numbers move down by one:

```bash
awk '$20=="000008" && $7=="Au3_cubes_ff" {print $21, $17}' <logs>/<beamtime>_FF.par
```

This is hard rule 13 in the metadata: never take a number from a position you did not
check on the file in front of you.

### 3b-2. HDF5 metadata map (20-ID-D HT-HEDM Varex, `.vrx.h5`)

There is no par file, so this table replaces §3b. Established by a full `visititems()`
walk of both a calibrant and a data file on `bt_20id_jul26b`; **dump the file yourself
before relying on it**, because it is a Bluesky/areaDetector layout and not a contract.

```
exchange/data                                   (nframes, 2880, 2880)  <- frames
exchange/dark        )  which of these holds the dark is PER SCAN — §3d
exchange/bright      )
measurement/process/scan_parameters/{start,end,step}   <- the ω scan AS COMMANDED. Use this.
measurement/process/scan_parameters/stage              <- e.g. '20idhedmA:m1'
measurement/instrument/SMS/aero                        <- per-frame ω readback, degrees.
                                                          JITTERY — direction only, never the step (§2b)
measurement/instrument/SMS/sam{X,Y,Z}                  <- sample translations, mm
measurement/instrument/SMS/sam{Rx,Rz}                  <- sample tilts, degrees
measurement/instrument/Detector/detY                   <- detector STAGE position, mm — a SEED, not Lsd (rule 3, §4b)
```

Three things this layout does **not** carry, each of which has to come from elsewhere:

| missing | consequence | what to do |
|---|---|---|
| **energy / wavelength** | nothing anywhere in the file records it — an exhaustive search of dataset names *and* `NDAttrSource` PVs for `energy\|mono\|undulator\|wave\|lambda\|keV\|dcm` returns **none** | it is user-supplied and stays **provisional** (§4a). `bt_20id_jul26b` ran at an asserted 63.000 keV in one campaign and 63.314 in the other — 0.5 % apart, which scales every absolute lattice parameter by 0.5 % and mostly cancels in relative strain (rule 8) |
| **ω sign** | §2's input does not exist | §2b |
| **the layer step** | the folder name claims it | read it: `samY` advanced 0.075 mm per scan across files 002569–002574, which is what confirmed BH100/OL25. Rule 13 |

Use `detY` to pick the calibrant. On `nfdev_jul26` the Ceria/LaB6 **0723** sets read
900.00000 mm and the 0721 sets 1040.00000 mm, while every `HPcat_aluminaRod_*` and
`Au_FF_box_*` read 900.00000 — so 0723 is the calibrant for that data, decided on a
measured readback rather than a date. (`ps_au.txt` had fitted `Lsd` 894264.41 µm against
that 900 mm readback: a 5.7 mm offset, exactly the rule-3 point.)

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

> ### 20-ID Varex: `darkLoc` is per SCAN. Measure it every time.
>
> An earlier revision of this file said "on the 20-ID Varex the dark is in
> `/exchange/bright`". **That was one scan's answer promoted to a station property, and
> it is wrong.** Measured means across a single beamtime (`nfdev_jul26`, 10-frame dark
> stacks):
>
> | file | `exchange/dark` | `exchange/bright` | `exchange/data` | ⇒ `darkLoc` |
> |---|---|---|---|---|
> | `Au_..._dark_before_001180` | **7.95** | 0.00 | 0.00 | `/exchange/dark` |
> | `HPcat_aluminaRod_*` dark | 0.00 | **1946** | — | `/exchange/bright` |
> | `Ceria_0723_..._001188` | **1484.07** | 0.00 | 1692.18 | `/exchange/dark` |
>
> Three cases in one folder, including a **calibrant and its sample disagreeing**. The
> `bt_20id_jul26b` campaign in the same cycle measured `/exchange/bright` and was right —
> for its scans.
>
> **Also per scan: whether the data is already dark-subtracted.** In the same folder the
> Au *data* is DAQ-subtracted (mean 0.10, 97 % exact zeros, residual dark ≈ 7) while the
> CeO2 calibrant is not (pedestal ≈ 1484). Pixel encoding was unscaled (min gap 1) on
> both. So "the dark looks empty" is a statement about one scan and never about the next.
>
> **Check, don't inherit** — three numbers per scan, seconds to read:
>
> ```python
> import h5py, numpy as np
> with h5py.File(dark_before_file) as f:
>     for g in ("exchange/dark", "exchange/bright", "exchange/data"):
>         if g in f:
>             print(g, float(np.asarray(f[g][:10]).mean()))
> ```
>
> **What it costs to get wrong.** The pedestal survives, every pixel clears the
> threshold, and each ring band labels as one ~42,000-px blob. It does not error, and it
> does not read as a dark problem — it reads as *"this sample is a powder, not spots"*,
> which is exactly the retraction it caused (Lab Notebook §9h).
>
> Separately, `exchange/dark` **inside the zarr** reads all zero on some of these
> datasets, and that one is harmless: the data was already dark-subtracted at zip time
> (raw frame mean ~1850 → zarr ~0.6). Check the *data frames* before chasing it — see the
> halt-condition wording in the spine.

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
`OmegaStart − SkipFrame·OmegaStep`, which the zipper computes as
`start_omega` (`ff_zip.py:250`), so that it describes raw frame 0.
The consumer recovers `start + SkipFrame·step = OmegaStart` for the first frame it
processes. The chain is self-consistent; changing either half alone breaks it.

**This is the authority for `OmegaStart`, and §2 and §10 used to contradict it.** Both
said `OmegaStart` describes *raw* frame 0; both were corrected on 2026-08-31. Checked
against the code and the data on a 1441-frame `aero` sweep negated to +180.25 → −179.75:
`OmegaStart 180.25` wrote `scan_parameters/start = 180.50`, a step past the raw frame 0 it
is supposed to describe, while `OmegaStart 180.00` wrote 180.25 and is correct. The
symptom of the wrong value is a **silent 0.25° rigid rotation about ω** — see the note in
§2 for how to test an existing run.

> Making the zipper physically drop the frame **as well** skips it twice: a 1441-frame
> sweep yields 1439 processed frames instead of 1440. Confirmed the hard way on
> `Au3_cubes_ff_000008` in this tree. Guarded by
> `midas_zipper/tests/test_skipframe.py`.

Sanity check in the peakfit banner: `nFrames` must equal *logged frames − SkipFrame*
(1441 − 1 = **1440**). If it reads 1439, something is skipping twice.

For a hand-reduced average outside the pipeline (calibrant staging, quick looks) there is
no consumer to do it for you, so drop it yourself: `data[1:].mean(axis=0)`, dark included.

> #### 20-ID carries a known **one-step (0.25°) ω zero-point offset**. Quote it, do not chase it.
>
> **This is documented, accepted and not a defect to fix mid-analysis.** It is recorded
> here because it is invisible in every output and would otherwise be rediscovered once
> per campaign — it has already been found twice, on `bt_20id_jul26b` (symptom only) and
> `nfdev_jul26` (symptom and a proposed mechanism).
>
> **What is measured.** On the same 20-ID data the current pipeline's spot ω range is
> **[−179.75, +180.25]** where the legacy-C run gave **[−180, +180]** — a one-step shift
> in the absolute ω zero-point between code generations. Frame accounting is correct in
> both (`nrFramesDone: 1441` either way), so this is **not** a `SkipFrame` error and the
> double-skip check above will not catch it.
>
> **What it costs, and why it is tolerable.** It rotates every orientation by 0.25° about
> ω. Grain positions move **≤ 2.2 µm at r = 500 µm**. Relative orientations, misorientations
> and every strain quantity are untouched, because they are differences. So:
>
> * **fine for** grain counts, positions, sizes, strain, misorientation, texture *shape*,
>   and any comparison between layers or samples processed the same way;
> * **state the offset** if you publish absolute orientations, or compare orientations
>   against a measurement outside this pipeline (EBSD, NF, a legacy-C reconstruction).
>   A 0.25° rigid rotation about ω is well inside most misorientation tolerances but is
>   not zero.
>
> **Attribution — open, and it does not need closing to use the data.** The `nfdev_jul26`
> campaign proposed that the zipper negates the raw commanded start (−180.25 → +180.25)
> without advancing it one step for the dropped frame 0. Note that this sits at odds with
> the back-dating contract described immediately above, under which a stored
> `scan_parameters/start` of 180.25 is *correct* for `OmegaStart 180.0`, `OmegaStep −0.25`,
> `SkipFrame 1`. Either the negated path does not honour that contract or the attribution
> is wrong; the **symptom** is measured either way, and that is what a report quotes.
> Do not "fix" one half of the chain on the strength of the attribution alone — §3e's
> warning about changing either half by itself applies here more than anywhere.

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
| `fastsweep_Emon.txt`, the `E_HEM` column (§ below — **not a fixed field number**) | 95.0000 | corroborates |
| spec `FullLog.log` → `Energy (keV):` | 95 | corroborates |
| `instrument/InsertionDevice/IDEnergy` | 95.055 | undulator setting, not the mono |
| `instrument/HRM/Energy` | 78.39 | **different monochromator — ignore** |
| the filename | "96keV" | **stale string** |

**The `E_HEM` column index is vintage-dependent — identify it by its VALUE, not by its
position.** `fastsweep_Emon.txt` columns come from `macros_<user>/E_mon.mac`, and the
leading timestamp is not a fixed width:

| vintage | `E_HEM` is | why | verified on |
|---|---|---|---|
| 2026 | field **6** | field 2 is a foil µt; field 6 is `epics_get("1id:userTran3.A")` | `bt_1id_jul26`, read 95.0000 |
| 2023 | field **10** | the timestamp occupies five whitespace fields (`Tue Mar 28 10:56:12 2023`), pushing everything right; **field 6 there is a foil µt, not an energy** | `bt_1id_mar23`, read 71.6800 across the whole window |

Taking field 6 on a 2023 file therefore returns an absorption number, not an energy, and
it will not look like one. **The energy is the column sitting in the tens of keV and
constant across the scan window** — pick it that way and the vintage stops mattering:

```bash
awk 'NR==1{for(i=1;i<=NF;i++) printf "%d:%s ", i, $i; print ""}' <logs>/fastsweep_Emon.txt
```

Then corroborate against the other two records before believing it. On `bt_1id_mar23`,
field 10 read **71.6800** flat, `FullLog.log` said `HEM energy is set to 71.68 keV`, and
the tomography scan record said `Energy (keV): 71.63` — three records agreeing to 0.07 %,
which is the level at which the pair rule (rule 8) starts to matter for an absolute
lattice parameter (see [`RUNBOOK.md`](RUNBOOK.md) §R2f).

**Rows where the last two columns are `0.000 0.000` had the foil out** (air) and carry no
absorption information.

λ[Å] = 12.398419843320026 / E[keV]. At 95.0 keV, λ = 0.130510 Å; at 71.630 keV,
λ = 0.173090 Å.

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
  --dark-group exchange/dark \          # MEASURE IT on THIS calibrant (§3d) — per scan,
                                        # not per station. On nfdev_jul26 the CeO2 dark
                                        # was in exchange/dark while its own sample's
                                        # was in exchange/bright
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

### 5f. Use the 0/180 pair if you have one — and read the spread before you average it

A calibrant measured at two rotations 180° apart is **two different measurements, and
which one you have depends on where the calibrant sat.** There are two regimes and they
call for opposite actions:

| regime | `Lsd` spread | `BC` spread | what it is | what to do |
|---|---|---|---|---|
| **small, unstructured** | ≲ 0.05 % | agrees | an independent repeat — honest fit uncertainty | quote the spread as the uncertainty; either fit will do |
| **large, systematic** | ≫ the repeatability, tilts and BC unmoved | agrees to ≪ 1 px | the diffracting volume is **displaced along the beam**; a 180° turn flips the offset, so the two fits **bracket** the true distance | **average — this is mandatory, not a tidy-up.** The mean is the rotation-axis-to-detector distance; either single exposure is wrong by the offset |

**Do not decide by the size of the number.** Decide by the three-part discriminating test
below, because a large spread that is *noise* and a large spread that is a *displacement*
call for different geometries.

#### The discriminating test — all three must hold for the displacement reading

1. **`BC` is unchanged.** A displacement **along the beam** scales the pattern about a
   fixed centre. A displacement **transverse** to the beam translates it, which moves
   `BC` instead. `BC` unmoved with `Lsd` split ⇒ the offset is along the beam.
2. **The ring radii scale UNIFORMLY**, measured on the images at a **fixed** BC — one
   ratio, flat across every ring. A tilt, a distortion error or a ring mis-assignment
   (§5b) all produce a radius-dependent ratio instead.
3. **It reproduces on a second, independent pair.** A fit landing in two basins does not
   repeat; a geometric offset does.

Also confirm from the log that only ω moved between the two exposures — every sample and
detector position identical. If a stage moved, this entry does not apply.

#### Regime 1, measured — `bt_1id_jul26`

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

#### Regime 2, measured — `bt_1id_mar23` CeO2, GE5 (2026-08-31)

The same procedure on a different campaign returned a spread **34× larger**, and it was
not noise:

| quantity | 0 vs 180 |
|---|---|
| `Lsd` | **7.34 mm apart = 0.96 %** — against 0.013 % on `bt_1id_jul26` above |
| `BC` | **0.0008 px apart** |

All three tests fired:

* the par shows **only ω changed** between the two exposures — fields 12–16, every sample
  and detector position, byte-identical;
* the ring radii scale **uniformly**: ratio **1.009604** measured directly from the images
  at a fixed BC, flat across 8 rings, against an `Lsd` ratio of **1.009602**;
* an independent 2 s CeO2 pair reproduces the split to **0.23 µm**.

**Cause.** The calibrant's diffracting volume sat **3.669 mm off the rotation axis along
the beam**. A 180° rotation flips that offset, so one exposure sits 3.669 mm nearer the
detector and the other 3.669 mm further; the two fits bracket the truth and **the mean is
the rotation-axis-to-detector distance**, which is the distance the reconstruction needs.
Using either single exposure would have put `Lsd` **0.48 %** out.

**Confirmed by a known-answer test.** With the corrected `Lsd = 767 765.75 µm`, gold cubes
on the same geometry fitted a = **4.07898 Å** against gold's literature 4.0782 —
**+191 ppm**. Either single exposure would have given **4.0977** or **4.0587 Å**, i.e.
±0.478 % — **25×** further from the known answer. (Two grains, related by a Σ3 twin at 59.9724°
about ⟨111⟩ — the same parent-plus-twin signature as the 1-ID gold reference,
Lab Notebook §3d.) Full numbers: [`RUNBOOK.md`](RUNBOOK.md) §R2f, Lab Notebook §10b.

**With only one exposure you cannot see any of this**, and the fit will converge and pass
the 100 µε gate at the displaced distance. That is the argument for acquiring the pair,
not just for using it: it is the only calibrant-side handle on a sample-position error
that otherwise propagates into every `Lsd`-scaled quantity. See
[`DIAGNOSIS.md`](DIAGNOSIS.md) *Sample displacement or distance error*.

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

> **`grain-tx` returns a RESIDUAL, not an absolute. Compose and iterate.**
> It reports the roll left over from whatever `tx` the reconstruction already ran with,
> so `tx_total = tx_applied + tx_reported`, then re-run from `transforms` and do it
> again — each pass recovers only part of what remains
> (`midas_joint_ff_calibrate/grain_refine.py`). Reading the reported number as the answer
> silently under-corrects, and the under-correction looks like convergence.
>
> Measured on the `nfdev_jul26` gold scan: pass 1 **−0.158497**, pass 2 **−0.087265**,
> composed **−0.245762**; an independent ring/η estimate gave −0.2455, and extrapolating
> the zero-crossing of a 4-point scan gave ≈ **−0.267**. That last value is an
> **extrapolation, not a converged fit** — a third pass would settle it. `Lsd`, `ty`,
> `tz` and `BC` are **absolute** on the non-raw path and must *not* be composed.
>
> **Two independent samples agreeing is the check that it is real.** On gold, the tx scan
> moved `DiffPos` 476 → 365 → 305 → 297 → **237 µm** monotonically, with the mean
> tangential residual 337 → **13 µm** and `frac(dtan>0)` 0.806 → **0.517**. Feeding the
> *same* tx to alumina moved its tangential bias 52.4 → **−2.6 µm** and `frac(dtan>0)`
> 0.560 → **0.495**. A fit absorbing error does not transfer between samples.

**Measured, 20-ID ti7al layer 1** — feeding the result back and re-running:

| | before | after |
|---|---|---|
| grains | 208 | 226 |
| grain-Z scatter (sd) | 152.6 µm | **76.4 µm** |
| completeness (median) | 0.580 | 0.630 |
| X / Y scatter | 271 / 265 µm | 273 / 272 µm — unchanged |

Z halving while X and Y stand still was, on *that* dataset, the signature of a
real geometry correction: it tightened the badly-conditioned coordinate and left
the well-conditioned ones alone. A fit that moved all three would be absorbing
error.

> **The Z-halving signature does NOT generalise — do not use it as the
> acceptance test.** It appears only when Z was loose to begin with. Measured on
> `bt_1id_mar23` LSHR (fcc, a = 3.59028 Å, 2321 grains, 1-ID GE5), a much larger
> roll than ti7al's:
>
> | | before | after | read as |
> |---|---|---|---|
> | `DiffPos` p50 | 251.3 µm | **62.3 µm** (4.0×) | the real signal here |
> | `DiffPos` p5 | 238.5 µm | **20.4 µm** | the **hard floor at 235 µm disappeared** |
> | `DiffAngle` p50 | 0.152° | 0.074° | |
> | **`DiffOme` p50** | 0.072° | **0.074° — unchanged** | **the control** |
> | X / Y scatter | 283 / 285 µm | 290 / 291 µm — unchanged | |
> | grain-Z scatter | 36.5 µm | 32.4 µm — barely moved | Z was already tight against a 200 µm beam |
>
> **`DiffOme` is the control and it must not move.** A roll of the detector about
> the beam moves spots within the detector plane; it cannot change which ω frame
> a spot appears on. A "tx refinement" that improves `DiffOme` is absorbing
> something else and should not be believed.
>
> The refinement converged in **two passes** — pass 1 residual **+0.123253°**,
> pass 2 **+0.003433°**, a **36× drop**, composed **`tx` = 0.126686°,
> `Wedge` = 0.000320°** — where the gold case above only halved per pass. Compose
> and iterate regardless; how fast it converges is a property of the dataset, not
> of the method.
>
> **The disappearing floor is the more useful diagnostic**, and it is available
> *before* you refine anything: see [`DIAGNOSIS.md`](DIAGNOSIS.md) *A hard floor
> in the `DiffPos` distribution*. Full numbers in Lab Notebook §10c.

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

> **Version gate first: `midas-joint-ff-calibrate` ≤ 0.4.0 cannot read a current
> `Grains.csv` or `SpotMatrix.csv`, and only one of the two halves crashes.** Its
> loader assumed the legacy 21-column `Grains.csv` behind a `len(cols) < 21`
> guard, which passes on a 47- or 53-column file — so it read `DiffPos` as
> `GrainRadius` and `DiffOme` as `Confidence`, and since grains are **selected**
> by `argsort(-confidence)` it was refining the **worst-fitting** grains. The
> `SpotMatrix.csv` half raises `KeyError: -1` on the ~3.3 % of rows that are
> predicted-but-never-observed. Fixed in the working tree (unreleased at the time
> of writing) by routing both through `midas_process_grains.io.read`; floor
> `midas-process-grains >= 0.11.0`. Full numbers: spine trap table,
> Lab Notebook §10d.

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
