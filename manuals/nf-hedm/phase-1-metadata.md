# Phase 1 — ω sign, metadata, energy and distance

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 2. STEP 1 — Establish the ω sign convention

**Run this first, on every new dataset.**

```bash
# field 9 of the per-frame log = the rotation stage name
awk '{print $9}' <METADATA_DIR>/<beamtime>_NF.par | sort | uniq -c
```

Decision:

| field 9 reads | meaning | action |
|---|---|---|
| `aero` / `Aero` | **recorded ω is opposite to MIDAS convention: ω_MIDAS = −ω_aero** | negate both `OmegaStart` and `OmegaStep` relative to the log |
| anything else | not established by this session | **stop and ask.** Do not assume it matches MIDAS |
| **no `.par` file exists** | you are not at 1-ID | 20-ID: **§2a** — settled, `aero`, negated. Any other beamline: **stop and ask** |

Worked example, verified (`bt_1id_jun25`, copland):

```
$ awk '{print $9}' /home/1-id/s1iduser/1id_old_data_2026preJune/bt_1id_jun25/bt_1id_jun25_NF.par \
    | sort | uniq -c
 441397 aero
```

All 441397 rows are `aero`. The acquisition macro for `Au4_cubes_nf_96keV` logged

```
Ran DoSingleLayer( -180, 0.25, 720, 1, Au4_cubes_nf_96keV, 1, 1 )
sweep stage -180 0 720 1
```

i.e. a logged sweep **−180 → 0 with step +0.25**. In the MIDAS paramfile that is

```
OmegaStart 180
OmegaStep -0.25
```

**Corroboration inside the repo:** the bundled reference paramfile already carries exactly
this pair — `ps_au.txt:65` `OmegaStart 180`, `ps_au.txt:66` `OmegaStep -0.25` — for a
360° Au scan (1440 frames × 0.25°, `ps_au.txt:70-74`).

### 2a. At 20-ID-D there is no `.par` file — the sign came from the beamline

**Determined 2026-08-28, instrument scientist: the 20-ID-D HT-HEDM rotation stage is `aero`,
and the sign is negated — the same convention as 1-ID.** ω is recorded per frame in
`exchange/theta` (§3h), so what the beamline supplied is the *sign*, not the values:

```
exchange/theta   -180 -> +180.25, step +0.25      # what the file records
OmegaStart  180                                   # what the paramfile takes
OmegaStep    -0.25
NrFilesPerDistance 1440                           # from the omega RANGE, not the frame count
```

`180 + 1439 × (−0.25) = −179.75`, i.e. a full 360° sweep. **This is what the completed
20-ID reconstructions already used** — `params_au0802.txt` and `params_ss316.txt` both
carry `OmegaStart 180` / `OmegaStep -0.25` — so the determination *confirms* those maps
rather than invalidating them. They were run on a correct assumption that had not yet
been checked; it has now been checked.

**What this closes.** Every 20-ID orientation map used to carry the label *handedness
undetermined, map may be mirrored*. **That label is retired.** Do not re-apply it, and do
not re-derive the sign from the data — the two results that look like they settle it
cannot (a cube-2 **radius** is mirror-invariant, and `θ = −φ − 90°` was calibrated against
the reconstructions themselves, so both are invariant under a global mirror). A sign is
settled by the beamline or not at all, which is why this row reads *determined* and names
who determined it.

**Why you cannot check this later.** A sign flip in ω mirrors the reconstructed
microstructure. Confidence values, grain counts and spot overlap are all unchanged. There
is no self-consistency check inside the reconstruction that catches it. That is true at
both beamlines, and it is why §2a is a *provenance* record rather than a measurement.

**Note for calibration work only:** an ω sign error *cancels* in a two-distance ray-bundle
solve, because the flip applies identically at both distances and spots are matched by
frame index within a sweep. It does **not** cancel in the paramfile or in forward
simulation. Evidence: `$ANALYSIS/bt_1id_jun25_nf/PREREGISTER.md:49-52`.

---

## 3. STEP 2 — Find the beamtime metadata and extract the scan definition

### 3a. Locate the metadata folder

The image tree has **only TIFFs**. The acquisition logs live elsewhere. For
`bt_1id_jun25` on copland:

| what | path |
|---|---|
| images | `/gdata/dm/1ID/2025/bt_1id_jun25/data/nf/` |
| **metadata** | `/home/1-id/s1iduser/1id_old_data_2026preJune/bt_1id_jun25/` (= `~s1iduser/new_data/1id_old_data_2026preJune/bt_1id_jun25/`) |

If you have not found a folder containing `FileCount.txt` + `fastsweep_Emon.txt` +
`*_SequenceOfEvents.log`, **you cannot write a paramfile.** Search for it:

```bash
# Try the convention first -- ~s1iduser/new_data/<beamtime>/, see the table in 3a.
ssh copland 'ls /home/1-id/s1iduser/<beamtime>/'

# Only if that misses. Measured 2026-08-12: this did not return within 120 s on copland
# and produced permission noise from unrelated home directories. Background it or bound it.
ssh copland 'find /home/1-id -maxdepth 5 -name "FileCount.txt" 2>/dev/null'
```

### 3b. What each metadata file answers

| file | field layout | answers |
|---|---|---|
| `FileCount.txt` | date is 5 whitespace fields, then **f10 = DetZ (mm)**, f11 = detector name, **f12 = exposure (s)**, **f15 = scan prefix**, **f16/f17 = image-number range, HALF-OPEN `[f16, f17)`** | the scan inventory: which prefixes, how many sweeps, which detector distances, which image numbers. One line per sweep. |
| `fastsweep_Emon.txt` | date is 5 fields, then **f10 = energy in keV** | **the only reliable energy record.** Match by timestamp to the scan's start time (rows are sparse — take the nearest preceding row). |
| `<prefix>_SequenceOfEvents.log` | free text | the acquisition macro and the per-sweep ω range. `DoVolume(nLayers, samY, layerStep, nDistances, DetZstart, DetZstep, omegaStart, omegaStep, nFrames, sweepTime, prefix nSweeps)`, then `umv volDetZmot <mm>` per distance and `sweep stage <from> <to> <nframes> <secs>`, and `Begin this sweep at image : <prefix> <N>`. |
| `<prefix>_WA.log` | SPEC `wa` dump | all motor positions at scan time: DetZ, samX/Y/Z. **Contains no energy field.** |
| `<beamtime>_NF.par` | **f6** = detector, **f7** = prefix, **f8** = image number, **f9** = rotation stage, f10 = ω start, f11 = ω step, **f17 = per-frame ω** | per-frame log. Source of the §2 sign rule and the §3d duplicate check. |
| `<beamtime>.spe` | SPEC log | **`#U Energy:` is stale/wrong — do not use it** (§4). Calibrant filenames inside it (e.g. `CeO2_..._96keV`) are a usable cross-check. |
| `<beamtime>_<E>keVBeamPos_*_BeamPosScan.txt` | 32 lines each | the DetZBeamPos calibration scan logs (§4). |

### 3c. Extraction commands

```bash
MD=/home/1-id/s1iduser/1id_old_data_2026preJune/bt_1id_jun25
SCAN=Au4_cubes_nf_96keV

# --- the sweep inventory: DetZ, exposure, image ranges ---
awk -v S=$SCAN '$15==S {printf "DetZ=%s mm  exp=%s s  imgs=[%s,%s)  n=%d\n",
                        $10,$12,$16,$17,$17-$16}' $MD/FileCount.txt

# --- the acquisition macro and per-sweep omega ---
cat $MD/${SCAN}_SequenceOfEvents.log

# --- energy: nearest preceding Emon row to the scan start time ---
awk '{print $1,$2,$3,$4,$5,"  E_keV="$10}' $MD/fastsweep_Emon.txt | less
```

Verified output for `Au4_cubes_nf_96keV`:

```
DetZ=9 mm       exp=1 s  imgs=[403783,404503)  n=720
DetZ=13.0001 mm exp=1 s  imgs=[404503,405223)  n=720
```

and from `fastsweep_Emon.txt`, the row preceding the 17:43:56 scan start:

```
Fri Jun 27 16:59:25 2025 ... f10=96.0000     -> 96.0000 keV, lambda = 0.1291502 A
```

`f12 = 1` (exposure) is corroborated two ways: `sweep stage -180 0 720 1` (the trailing
`1` is seconds/frame) and wall clock — sweep 1 ran 17:31:28 → 17:44:05 = 757 s for 720
frames ≈ 1.05 s/frame.

So the derived paramfile block for this scan is:

```
nDistances 2
NrFilesPerDistance 720    # RAW images per distance (§8j: not divided by SumFrames)
StartNr 0
                          # EndNr: omit — optional for NF, derived and logged (§10d)
RawStartNr 403783
OmegaStart 180            # NEGATED, aero  (logged -180 -> 0, step +0.25)
OmegaStep -0.25           # RAW step (§8j: not multiplied by SumFrames)
Wavelength 0.1291502
px 1.48
NrPixels 2048
```

### 3d. Consistency checks — run all three

**(i) Rows vs unique image numbers.** A duplicated image number at a sweep boundary means
one sweep lost a frame.

```bash
awk -v S=$SCAN '$7==S{print $8}' $MD/<beamtime>_NF.par | wc -l      # rows
awk -v S=$SCAN '$7==S{print $8}' $MD/<beamtime>_NF.par | sort -u | wc -l  # unique
```

Verified for `Au4_cubes_nf_96keV`: **1442 rows, 1441 unique.** The duplicate:

```
$ awk '$7=="Au4_cubes_nf_96keV" && $8==404503 {print $8, $17}' bt_1id_jun25_NF.par
404503 0.000000      <- last frame of sweep 1
404503 -180.000000   <- first frame of sweep 2
```

The later write wins, so **sweep 1 silently lost its ω = 0 frame.** Do not derive
frames-per-distance from `NF.par` row counts. Use `FileCount.txt` ranges as half-open
`[f16, f17)`: 404503−403783 = 720 ✓, 405223−404503 = 720 ✓.

**(ii) TIFF count vs the inventory.**

```bash
ls -1 /gdata/dm/1ID/2025/bt_1id_jun25/data/nf/$SCAN/*.tif | wc -l   # -> 1441
```

1441, not 1440: image 405223 (the half-open end) also exists on disk as a stray. Expect
stragglers; trust `FileCount.txt`.

**(iii) Pad width and contiguity.** The pipeline TIFF reader hard-codes a **6-digit zero
pad** and builds `<DataDirectory>/<OrigFileName>_<NNNNNN>.<extOrig>`
(`process_images/io.py:24-36`). If the pad is not 6, the reader will not find frames even
though the viewer can display them (the viewer infers pad width from the file you pick,
`nf_qt.py:1210-1218`).

```bash
python3 -c "
import glob,os,re,collections,sys
d=sys.argv[1]
ns=[(len(m.group(1)),int(m.group(1))) for f in glob.glob(os.path.join(d,'*.tif'))
    for m in [re.search(r'_(\d+)\.tif$', f)] if m]
w=collections.Counter(p for p,_ in ns); v=[n for _,n in ns]
print('pad widths:',dict(w),'count:',len(v),'min:',min(v),'max:',max(v),
      'contiguous:',max(v)-min(v)+1==len(v))" /gdata/dm/1ID/2025/bt_1id_jun25/data/nf/$SCAN
```

**Frame-index formula the reader uses** (`process_images/io.py:24-36`), for the *j*-th
frame of "layer" *L* (1-based):

```
idx = RawStartNr + (L-1)*WFImages + (L-1)*NrFilesPerDistance + j     j in [0, NrFilesPerDistance)
```

**Trap: in `process_images`, "layer" means DETECTOR DISTANCE, not sample layer.** The
argument is spelled `layer_nr`/`LayerNr` throughout, but `process_all` defaults it to
`range(1, n_distances+1)` (`process_images/pipeline.py:259-260`) and `layer_nr-1` indexes
the *distance* axis of the bitmask (`process_images/pipeline.py:237-238`). Sample layers
are handled entirely by rewriting `RawStartNr` (§8c).

### 3e. Folder-name conventions

The **only** folder-name rule in code is the BeamPos special case: if the basename of the
**current working directory** contains `BeamPos` or `DetZBeamPos`, the viewer globs `*.tif`
in the cwd and makes `Frame` an index into that sorted list (`nf_qt.py:127-142` — it tests
`os.getcwd()`, not the `folder` argument, so you must `cd` in). Documented in
`manuals/NF_Calibration.md:37-43` and `manuals/NF_GUI.md:95-97`.

`*_nf_*` for sample-layer scans is **convention only** — nothing in this repo parses it.
Confirm against frame counts, never against the name.

### 3f. Already-processed folder? Check before recomputing

- `<stem>_Median_Background_Distance_<d>.bin` (`nf_qt.py:1239`) and
  `<stem>_{MaximumIntensity,SumIntensity}[MedianCorrected]_Distance_<d>.bin`
  (`nf_qt.py:1258-1264`) — raw `uint16`, `NrPixelsY*NrPixelsZ`. Their presence means the
  median step already ran.
- `grid.txt`, `hkls.csv`, `SpotsInfo.bin`, `DiffractionSpots.bin`, `OrientMat.bin`,
  `Key.bin`, `MicFileBinary`, `*.mic`, `*_consolidated.h5`, `*_pipeline.h5` — a previous
  reconstruction, all flat in one directory (§8d).
- One frame per file: the loader asserts each TIFF's shape equals `(NrPixelsZ, NrPixelsY)`
  and raises otherwise (`process_images/io.py:64-68`). A 3-D `tifffile.imread` return means
  this is not the layout the code expects.
- **Row/column order is (Z, Y)** — first axis Z (detector rows), second Y
  (`process_images/io.py:45-51`). The viewer displays `imarr2[::-1,::-1]`
  (`nf_qt.py:1305`), NF display origin bottom-right (`nf_qt.py:654`); the spot writer
  applies the matching `y → NrPixelsY-1-y`, `z → NrPixelsZ-1-z` flip
  (`process_images/spots_io.py:33-43`). **Do not "fix" an apparent flip** without tracing
  which of these three frames you are in.

---

### 3g. Derive the sweep structure per scan — never inherit it from the calibrant

**Scans in the same beamtime differ in distance count and ω step.** The geometry
(`Lsd`, `BC`, tilts) transfers from the calibrant; the *scan definition* does not.
Re-derive these four numbers for every scan: `nDistances`, `NrFilesPerDistance`,
`StartNr`, `OmegaStep`.

Derive them from the **per-frame log** (`<beamtime>_NF.par`), not from the
`DoVolume(...)` arguments — those are what was *requested*, the per-frame log is
what was *written*.

```bash
M=~/new_data/<beamtime>; PFX=<exact scan prefix>
# f7 = prefix, f8 = image number, f17 = per-frame omega
awk -v p="$PFX" '$7==p {print $8, $17}' $M/<beamtime>_NF.par > ome.txt

# 1. DEDUP FIRST -- sweep starts are logged twice (see trap below)
awk '{o[$1]=$2} END{for (k in o) print k, o[k]}' ome.txt | sort -n > dedup.txt

# 2. distance boundaries = where omega jumps BACKWARDS
awk 'NR>1 && $2<prev {print "boundary at image", $1} {prev=$2}' dedup.txt

# 3. reconcile against what is actually on disk -- both lists must match exactly
ls /gdata/dm/1ID/<year>/<beamtime>/data/nf/$PFX | grep -oE '[0-9]{6}' | sort -n > files.txt
comm -3 <(awk '{print $1}' dedup.txt) files.txt      # must print nothing
```

**Trap: sweep starts appear twice in `NF.par`.** For `nf_sampleB_htB_s2_0p5deg` the
log had **756 rows for 721 images** — 35 duplicated image numbers, one per sweep
restart (`nCrashedFrames : 1` in the `_Sweep.log`). Here both copies carried the
*same* ω, so dedup is lossless — **but check that**, because a duplicate with a
*different* ω means a genuinely lost frame and a shifted ω axis for the rest of
the sweep. Counting rows without dedup inflates the frame count by 5%.

**Worked contrast, both from `bt_1id_jul26`:**

| | `Au5_cubes_nf_96keV` (calibrant) | `nf_sampleB_htB_s2_0p5deg` (sample) |
|---|---|---|
| distances | 4 (DetZ 7/9/11/13) | **2** (DetZ 7/9) |
| ω step (logged) | +0.25 | **+0.5** |
| frames per distance | 720 | **360** |
| `StartNr` / stride | 286 / 720 | **8733 / 360** |
| ω logged | −180 → −0.25 | −180 → −0.5 |
| ⇒ `OmegaStart` / `OmegaStep` | 180 / −0.25 | 180 / **−0.5** |
| ⇒ `OmegaRange` | 0 180 | 0 180 |

Both are `aero`, so `OmegaStart`/`OmegaStep` are the **negated** logged sweep
(§2) and `OmegaRange` is the same 0–180 window — while the step, the frame count
and the distance count all differ. Copying the calibrant paramfile and editing
only the paths gets three of these wrong.

**The last frame of the last distance is a real frame.** Each sweep nominally
ends on ω = 0, but that frame is overwritten by the first frame of the next
sweep — except for the final sweep, which keeps it. That is why the sampleB scan has
721 files for 2×360 frames, and why a stride of `NrFilesPerDistance` is still
exact. Ignore the extra file. **Do not set `EndNr`** — it is optional for NF and the
pipeline derives `StartNr + NrFilesPerDistance − 1` and logs it (`60dcc94c`, §10d).

**The 1-ID "skip the first frame" rule is a GE / far-field detector rule and does
NOT apply to NF.** On the GE FF detector the first frame of every acquisition —
sweep, dark, calibrant — is a settling throwaway (1441 logged frames ⇒ usable
2..1441), and it must be dropped from averages and from the sweep definition. The
NF detector does not do this. **Never carry that rule across to an NF scan**:
`StartNr` is the first image. Confirmed with the instrument scientist.

Independent evidence from the data, which also holds on any such scan:

- The `+1` extra file sits at the **end**, not the start. The per-frame ω log
  reverses exactly at the second distance's first image (9093 for the sampleB scan),
  and the final image carries the wrap ω = 0. Skipping a leading frame would push
  the reversal *inside* the first distance, which contradicts the log.
- `Au5_cubes_nf_96keV` reconstructed to confidence **1.000000** on the calibrant
  with `StartNr` = the first image. A one-frame ω error would not survive that.

So: on NF volume scans the extra file is a trailing wrap, and `StartNr` is the
first image. Re-derive this from the ω log for any new scan rather than assuming
either convention — the two differ by one ω step, which is invisible in the
`.mic` and degrades confidence without ever raising an error.

---

### 3h. 20-ID-D HT-HEDM (Bluesky + HDF5) — a different world; inherit nothing

`§3a-§3g above are 1-ID.` At 20-ID (`/gdata/dm/20ID/HT_HEDM/<cycle>/<beamtime>/`) the
acquisition is **Bluesky/ophyd running tomography-style fly scans**, the detector is a FLIR
optical camera, and the data is **HDF5 in DXchange layout**. Lab notebook §7 has the
measured detail; what changes operationally:

| | 1-ID | 20-ID |
|---|---|---|
| Frames | one TIFF per frame, 6-digit pad | **one HDF5 per (distance, layer)**, `exchange/data` `(N, Z, Y)` uint16 |
| Metadata | `~/new_data/<beamtime>/` (`FileCount.txt`, `fastsweep_Emon.txt`, `*.par`) | `data/metadata/<bt>/.logs/ipython_logger.log` — **there is no `~/new_data` equivalent** |
| ω | derived from the paramfile, sign from `NF.par` f9 | **`exchange/theta` is IN the file**, per frame |
| Distance / energy / px | `FileCount.txt` f10 / `fastsweep_Emon.txt` f10 | **NOT in the HDF5** — only areaDetector camera attributes are. Read the `nfscan(...)` call out of the ipython log |
| Darks | separate scan | may be **all-zero placeholders**; check before trusting (`data_dark`, `data_white`, `data_white_post`) |

**Which scan folder?** Neither this file nor the spine names one, and a data directory
holds many. The measured detail for a given beamtime is in **Lab Notebook §7**, which names
the folder it worked (`nfdev_jul26` → `Au_cube`; at 1-ID `bt_1id_jul26` → the Au sweep
`Au5_cubes_nf_96keV` with its paired `Au_DetZBeamPos_95keV`). Read that before picking, and
do not choose on filename plausibility — a fresh session on `nfdev_jul26` picking by name
would reasonably have taken `Au_63keV_Z20mm` or `test_nf`.

**Read the scan definition out of the acquisition command**, e.g.

```bash
M=/gdata/dm/20ID/HT_HEDM/2026-2/nfdev_jul26/data/metadata/nfdev_jul26
grep -n "User input: RE(nfscan" $M/.logs/ipython_logger.log
#  nfscan(0.2, fname=..., nfz_start=7, nfz_end=11, ndz=3, y0=8.04, dy=0.01, y_nlayers=2)
#  -> ndz COUNTS DISTANCES: linspace(7,11,3) = 7, 9, 11 mm ;  2 layers ;  6 files
```

`ndz` counts distances, not steps — established from the log's own API rename (lab
notebook §7a). Energy is an **absorption edge**; the foil table is printed in the log by
`foilA.about` (element, thickness, position, K-edge keV).

> **63.314 keV is CONFIRMED for `nfdev_jul26` and `bt_20id_jul26b`** — instrument scientist,
> 2026-08-28. It is no longer a caveat on those two beamtimes, and reports quoting it need
> not hedge. **The procedure gap that made it a caveat is still open**, so the next
> beamtime starts in the same place:
>
> **The foil table is not a measurement of the energy, and 20-ID has no equivalent of
> `fastsweep_Emon.txt` f10.** `foilA.about` prints a *static* 13-row reference table of
> every foil in the wheel (Pr/Sm/Yb/Lu/Hf/Ta/W/Re/Pt/Au/Pb/Bi) and is never followed in
> the log by a call recording which foil was selected. Checked 2026-08-12 on
> `nfdev_jul26`: the only corroboration *available in the data* for 63.314 keV (Lu K-edge)
> was `_63keV` in the scan *filenames*, which rounds to Lu rather than the neighbouring Yb
> (61.332) or Hf (65.351) — plausible, and **exactly the inference hard rule 22 forbids**
> ("never take a number from a name"). The confirmation above came from **asking**, which
> is the lever this row exists to point at. On a 20-ID beamtime nobody has confirmed,
> record the energy as **inferred, not measured**, say so in any report that quotes it, and
> ask. A value rigorous *from the data alone* still needs the foil selection logged, or an
> edge scan.

**Entering the pipeline at 20-ID.** This used to be impossible — `midas_nf_preprocess` had
no HDF5 reader and `process_all` began by loading a whole layer into RAM, which is 141 GB at
fp32 on this detector. Both are now closed (`process_images/io.py`,
`process_images/median.py::streaming_temporal_median`), so the reduction runs on the HDF5
directly. Four keys do it:

```
extOrig     h5          # one file per DETECTOR DISTANCE, not per frame
DataLoc     exchange/data
PixelScale  1           # or 64 -- CHECK np.unique, do not inherit it
RawStartNr  708         # -> _000708.h5 _000709.h5 _000710.h5 for nDistances 3
```

Streaming is automatic for HDF5 (`StreamFrames 0`, the default): frames are read a block at
a time and the layer is never resident. The streamed and materialised paths produce an
**identical** `SpotsInfo.bin` — verified on 8 real frames of `NF_Au_cube_0802_000708` as
well as on synthetic fixtures — so this is a memory route, not a second reduction.
`PARAMETERS.md` §10f has the full key list, the sample-layer convention and the
`MedianFrames` measurement.

Verified against the real file (`validate_h5_reader.py`, 2026-08-19): frames read identical
to `h5py` at indices 0/1/719/1439; `NrFilesPerDistance 1440` against a 1442-frame dataset;
`PixelScale` warns on a wrong setting and stays quiet on the right one.

Do **not** convert to TIFF to avoid any of this. The pixel scaling and the ω-sign question
(hard rule 1) both survive the conversion and become harder to see afterwards.

**Still 20-ID-specific, still yours to do by hand:**

- read `exchange/theta` range and step to set `NrFilesPerDistance` from the ω RANGE
- check `data_dark` / `data_white` for the all-zero placeholder case
- `np.unique` on one frame to fix `PixelScale` **before** anything else
- recover the scan definition from the `nfscan(...)` call in the ipython log
- `midas_nf_preprocess.beam_calib.{shadow,triangulate}` for BC and Lsd (§6, §6e)

**Traps specific to this format:**

1. **The pixel encoding is PER SCAN, not per detector — measure it, never inherit it.**
   `nfdev_jul26` is 10-bit stored ×64: values are multiples of 64 and saturation is
   65472 = 1023×64. On the **same detector serial**, two weeks later, `NF_Au_cube_0802`
   and the SS316L NF scan are 12-bit **unscaled** (max 4092, unique values 0, 2, 4, 6, …)
   — while the SS316L *tomography taken the same day* is ×64 again. Declare it as
   `PixelScale` (§10f); it defaults to 1, warns in both directions and **never infers**.

   ```python
   np.unique(frame)[:8], frame.max()      # multiples of 64 -> PixelScale 64; else 1
   ```

   A wrong `PixelScale` is the most expensive mistake recorded at this beamline: it turns
   the §5d production threshold of "2 counts" into 128 counts, thresholds the **pedestal**
   so the background reads as signal, and it produced three successive wrong distance
   answers on SS316L before it was found (lab notebook §8b).
2. **The frame count may exceed 360°.** `nfdev_jul26` has 1442 frames spanning
   −180 → +180.25; the last two duplicate the first two. Set `NrFilesPerDistance` from the
   ω range, not from the frame count.
3. **Chunking wastes disk and I/O.** Chunks (1,1500,1960) on a 4600×5320 frame pad to
   6000×5880 — **44 % overhead** (101.77 GB file for 70.58 GB of payload). Reading a single
   frame costs ~35 MB of chunk reads, not 49 MB of frame.
4. **The chunk layout is not the frame layout.** Read whole frames through the source
   abstraction rather than slicing `exchange/data` by hand, or every read pays the
   padding above.

**The two blockers that used to sit here are CLOSED** (opened 2026-08-01, closed
2026-08-19; the shared env carries the fix — see `RUNBOOK.md` §R1). They were: no HDF5 reader in `midas_nf_preprocess`, and `process_all` loading a
whole layer into RAM. Recorded because the *shape* of the fix matters —

- The reader is chosen from `extOrig`, and one HDF5 holds one **distance**, so the file
  index advances per distance rather than per frame (`process_images/io.py::layer_file`).
  `process_images` used to address distances as frame-index ranges inside one series, which
  is why a per-distance container needed a source abstraction and not a second loader.
- The median is row-blocked and reads through the same source
  (`median.py::streaming_temporal_median`), so peak memory is a block, not a layer.

**Do not inherit the 1-ID pixel convention** — with one exception now measured. `ybc =
2047 − col`, `zbc = 2047 − row` encodes *that* detector and *that* viewer/writer chain, and
on a different beamline the array→lab mapping must be re-derived; getting it wrong **mirrors
the microstructure invisibly**. At 20-ID it *was* re-derived rather than assumed: one
reduction pass wrote both bitmasks, and the Y-reversed twin scored **maxC 0.000000 with zero
voxels ≥ 0.5** against 0.6957 / 634 voxels for the 1-ID convention (Lab Notebook §7g). So
the flip transfers to the 20-ID Oryx **as a measurement**. The method — build both masks
from one expensive reduction and let the calibrant decide — is what to repeat on the next
new detector, not the constant.

The ω sign is a different matter and is **not** settled by that test (hard rule 1).

---

## 4. STEP 3 — Energy and distance: the two fields that lie

### 4a. Energy

**Use `fastsweep_Emon.txt` field 10. Nothing else.**

Two decoys, both verified in `bt_1id_jun25`:

**Decoy 1 — `NF.par` field 29.** For the first frame of `Au4_cubes_nf_96keV` it reads
`96.033875`, which looks exactly like "96 keV" for a scan named `..._96keV`. It is a
fluctuating beam monitor:

```
$ # per-scan range of NF.par f29
Au4_cubes_nf_96keV: n=1442  min=93.076934  max=98.299410
Au5_cubes_nf_96keV: n=2163  min=85.054687  max=92.116068   <- also genuinely 96 keV
```

Au5 was at 96 keV and f29 never reaches 96. Successive Au5 frames read
89.013416 / 89.300548 / 89.331455. It fluctuates frame-to-frame and disagrees with the
true energy by up to 11 %. **It is not energy.**

> **What f29 contains is beamtime-specific — do not use its shape as a check.** The
> energy-like 93–98 range above is `bt_1id_jun25`. On `bt_1id_jul26` the same field
> reads mostly `0.000000` with the rest clustered 1.00–1.05 (checked 2026-08-12), i.e.
> nothing like energy at all. The rule is unchanged and does not depend on this: use
> `fastsweep_Emon.txt` f10 and nothing else. A reader who tries to *recognise* the decoy
> by its range will simply not find it on some beamtimes.

**Decoy 2 — `<beamtime>.spe` `#U Energy:`.**

```
$ grep -m3 "#U" bt_1id_jun25.spe
#U Beam Current: 200.4
#U Energy: 8.0509
#U Preamp Settings:
```

`8.0509` while the beamline ran at 52 and 96 keV. Stale. Ignore.

Cross-check available: calibrant scan names inside the `.spe` (e.g. `CeO2_..._96keV`).

### 4b. Distance

**DetZ in `FileCount.txt` is a stage readback in mm, not `Lsd`.** Only the *difference*
between distances is trustworthy:

```
DetZ 9.0000 -> 13.0001 mm  =>  dLsd = 4000.1 um   (exact, matches DetZstep 4 in the macro)
```

The **absolute** sample-to-detector distance requires calibration. `*DetZBeamPos*` scans
are meant to supply it: direct beam plus sample at ω = 0/90/180, macro
`DetZBeamPos(startDetZ endDetZ step exposure prefix)`, logged to
`<beamtime>_<E>keVBeamPos_*_BeamPosScan.txt`.

**In `bt_1id_jun25` those image directories exist and are EMPTY:**

```
$ for d in /gdata/dm/1ID/2025/bt_1id_jun25/data/nf/*BeamPos*; do echo "$d : $(ls -A $d|wc -l)"; done
.../Au_DetZBeamPos2 : 0 entries
.../Au_DetZBeamPos3 : 0 entries
.../Au_DetZBeamPos4 : 0 entries
```

The 27 `*_BeamPosScan.txt` logs (32 lines each, at 51.931 / 51.951 / 96 keV, with
`NoBBNoAu` / `NoBBwithAu{0,90,180}` / `withBBNoAu` variants) **do** exist. So for this
beamtime the absolute `Lsd` had to be triangulated from sample spots instead (§6).

**Decision tree:**

| you have | do |
|---|---|
| non-empty `*DetZBeamPos*` TIFFs | measure the direct beam per distance in the viewer (`manuals/NF_Calibration.md:31-34, 74-78, 82-100`) → BC and absolute Lsd |
| empty `*DetZBeamPos*` but ≥ 2 sample distances | §6 (ray-bundle triangulation from sample spots) |
| one distance only, no BeamPos | **stop.** Lsd and BC are not recoverable; ask |

**Peak finding does not need geometry, so you are not blocked.**
`midas_nf_preprocess.process_images.params.ProcessParams` (`process_images/params.py:21-106`)
has **no** `Lsd`, `BC`, `px`, `tx/ty/tz` or `Wavelength` field — its whole key list is
`RawStartNr, DataDirectory, OutputDirectory, NrPixels, NrPixelsY, NrPixelsZ, WFImages,
NrFilesPerDistance, OrigFileName, ReducedFileName, extOrig, extReduced,
BlanketSubtraction, MedFiltRadius, DoLoGFilter, LoGMaskRadius, GaussFiltRadius,
WriteFinImage, Deblur, nDistances, WriteLegacyBin, SoftTemperature`
(`process_images/params.py:83-105`). Run peak finding *before* the geometry is known; this
breaks the apparent chicken-and-egg. `ProcessImagesPipeline.process_frame`
(`process_images/pipeline.py:143-149`) returns a `FrameResult` exposing `.labels`
(connected components, `process_images/pipeline.py:45-47`) and `.n_spots`
(`:49-51`) for centroiding.

---
