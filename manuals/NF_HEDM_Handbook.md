# NF-HEDM Reconstruction Runbook

**Audience: a Claude Code session with no prior context that must run an NF-HEDM
reconstruction from scratch.** Not a tutorial. Follow the steps in order; each one names
the file to read, the command to run, the field to look at, and the branch to take.

Citations are `path:line` relative to `$MIDAS = /Users/hsharma/opt/MIDAS`. Read them with
absolute paths (`/Users/hsharma/opt/MIDAS/<path>`). Every non-obvious claim carries one.
Claims that are convention, or that could not be verified, are flagged inline and listed
again in §11. **Do not promote a §11 item to a fact.**

Maintained code = four Python packages: `midas_nf_pipeline` (0.1.1), `midas_nf_preprocess`
(0.1.2), `midas_nf_fitorientation` (0.3.2), `midas_hkls` (0.5.0), plus the viewer
`gui/nf_qt.py`. `NF_HEDM/` is soft-deprecated C; only its example paramfile and seed cache
are used here.

---

## STOP — read this before touching anything

### Hard rules

1. **Determine the ω sign convention first (§2).** Field 9 of `<beamtime>_NF.par`. If it
   is `aero`, then **ω_MIDAS = −ω_aero** and the paramfile needs the *negated* sweep. Get
   this wrong and the reconstruction is **mirrored**, which is **not detectable from the
   `.mic` alone**. This is step 1 of every new dataset, no exceptions.
2. **The TIFF tree does not contain the metadata (§3).** `/gdata/dm/1ID/<year>/<beamtime>/data/nf/`
   holds only images. Distances, ω, energy, exposure live in a *separate* acquisition-log
   folder. Find it or stop.
3. **Energy comes from `fastsweep_Emon.txt` field 10, and nowhere else (§3, §4).** Two
   other fields look like energy and are wrong.
4. **Never count spots off a raw max-projection (§5).** It is dominated by cosmic rays.
   Use the temporal-median + LoG path.
5. **Do not run `midas-nf-pipeline run` (§8a).** Five call sites in the orchestrator do not
   match their callees, re-verified in this tree. Use the nine-command route in §8b.
6. **`--all-layers` is mandatory** on `process-images`; without it only the last detector
   distance's bits survive (`process_images/pipeline.py:229-243`,
   `process_images/cli.py:57-60`).
7. **Read `TriEdgeSize` from column 5 of a data row, never from the `%TriEdgeSize` header**
   (§9a).
8. **`/grains/` in the consolidated H5 is not grains (§9c).** Use `/maps/grain_id` or run
   `mic2grains`.
9. **Units: µm, degrees, Å** (Å for wavelength and lattice parameters only). Output Euler
   angles are **radians**; so are `.map.kam` and `.map.grod`.
10. **DetZ ≠ Lsd (§4).** Only *differences* between DetZ readbacks are trustworthy.
11. **`BC` pixel convention is `2047 − raw_index`, on BOTH axes (§6b).** `ybc = 2047 − col`,
    `zbc = 2047 − row`, where `raw` is what `tifffile.imread` returns. Validated against an
    operator reading to 0.3 px. The constant is **2047, not 2048** — one pixel matters at
    `BCTol 0.2`.
12. **Never borrow the beam tilt β between beamtimes (§6f).** Measure it from that
    beamtime's own DetZBeamPos scan. Borrowing it was wrong by 62× in y.
13. **BC comes from DetZBeamPos; Lsd comes from spots (§6a).** Neither measurement can give
    the other's quantity. Run DetZBeamPos first.
14. **Confidence 1.0 does NOT mean the geometry is right (§7b).** It is a *plateau*: on real
    Au data, `ty` seeds 2 deg apart all reach exactly 1.0000. Never close out a calibration
    on the confidence number alone.
15. **Never re-seed a refinement with its own output (§7b).** `TiltsTol` is relative to the
    seed, so iterating ratchets the tilts ~1 deg per pass while confidence stays high.
    Use `NumIterations` inside ONE invocation instead.
16. **`-multiGridPoints` does not fix an under-determined geometry on a single-crystal
    calibrant (§7b).** All voxels are one grain, so N voxels give one grain's constraint.
17. **Check `BoxSize` before blaming the geometry (§7d).** Unset, it costs exactly the last
    few percent of confidence (0.949153 vs 1.000000) and looks like a small geometry error.
18. **Compare reconstructions by field, never by checksum (§8f).** `MicFileBinary` records
    carry a per-voxel `RunTime`, so two bit-identical *physics* results have different md5s.

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `aero` ω sign | mirrored microstructure, plausible confidence | §2 |
| `NF.par` field 29 looks like energy | wrong wavelength → wrong ring radii | §4 |
| `<beamtime>.spe` `#U Energy:` is stale | same | §4 |
| Duplicated image number at sweep boundaries | one sweep silently loses a frame | §3d |
| DetZ readback used as absolute Lsd | systematically wrong geometry | §4 |
| Cosmic rays in a max-projection | fictitious spot counts, fictitious "coverage" | §5 |
| `%TriEdgeSize 0.000000` header | `mic2grains` silently falls back to global merging; grain radii collapse to 0 | §9a |
| `Confidence == 0` rows dropped from text `.mic` | row *i* of `.mic` ≠ voxel *i* | §9a |
| four mislabelled H5 datasets | reading the wrong quantity under a plausible name | §9c |
| `LatticeConstant` instead of `LatticeParameter` | `KeyError` in the HKL stage | §10b |
| BC left in raw array indices | mirrored geometry; **invisible in y**, because BC sits near the detector centre so a flip still looks plausible | §6b |
| `2048 − index` instead of `2047 − index` | 1 px BC error, i.e. 5× `BCTol` in z | §6b |
| β borrowed from another beamtime's paramfile | per-distance BC wrong by tens of px | §6f |
| centroiding `1 − T` over the whole illuminated band | axis scatter 66 px instead of 0.2 px, and it looks like real data | §6e |
| row-permutation "null" in a spot-matching test | null silently re-runs the real analysis and passes | §6i |
| confidence 1.0 read as "geometry solved" | a whole beamtime reconstructed on the wrong plateau | §7b |
| refinement re-seeded from its own output | tilts drift ~1 deg/iteration, confidence stays high | §7b |
| `GridPoints` given 6 tokens instead of a 12-column `.mic` row | parses fine, refines nothing | §7c |
| `BoxSize` parsed but not applied | calibrant plateaus at 0.949 instead of 1.000 | §7d |
| blob size compared using radius from the GRID ORIGIN | sample offset misread as a geometry difference | §7e |
| md5 of `MicFileBinary` used to check reproducibility | `RunTime` differs every run; always "fails" | §8f |
| voxel-count blow-up in `screen()` | 1704 GiB allocation on a full grid | §8g |
| `EdgeLength` set to a fixed value while `GridSize` differs | grid stops tiling — e.g. 1 µm triangles on a 10 µm lattice is ~1 % coverage; `mic2grains` areas wrong by `(GridSize/EdgeLength)²`; worsens every multi-res loop | §10e |
| `MinMisoNSaves` left at its **1.0 default** with `SaveNSolutions 1` | a per-window symmetry misorientation dominates runtime, AND a later higher-confidence solution is silently discarded | §8h |

---

## 1. Environment

**On the APS hosts** (chiltepin, copland, alleppey, sentosa, chutoro — all share
`/home/beams*`), call by full path; conda is not on the non-interactive ssh PATH:

```bash
/home/beams12/S1IDUSER/opt/envs/midas/bin/python
```

Verified contents (2026-07-29, `importlib.metadata`): `midas-nf-pipeline 0.1.1`,
`midas-nf-preprocess 0.1.2`, `midas-nf-fitorientation 0.3.2`, `midas-hkls 0.5.0`,
`numpy 2.4.6`, `tifffile 2026.3.3`, `h5py 3.16.0`, `scipy 1.17.1`, `torch 2.11.0+cu128`.
Same versions as this repo tree.

**`matplotlib` is NOT installed in that env.** Therefore: **reduce remotely, plot
locally.** Write an `.npz` of the reductions on the host, `scp` it to the Mac, plot there.
See §5a for the pattern actually used.

GPU prefix on any of those hosts: `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE`.
Pick a GPU by *utilization*, not free memory. Long jobs: `setsid`/`nohup` + redirect to a
log, or SIGHUP kills them.

On the Mac: no CUDA. Activate the project env (`midas_env` or `hsharma_midas`); ask if
unsure. Do not assume.

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

Worked example, verified (`pokharel_jun25`, copland):

```
$ awk '{print $9}' /home/1-id/s1iduser/1id_old_data_2026preJune/pokharel_jun25/pokharel_jun25_NF.par \
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

**Why you cannot check this later.** A sign flip in ω mirrors the reconstructed
microstructure. Confidence values, grain counts and spot overlap are all unchanged. There
is no self-consistency check inside the reconstruction that catches it.

**Note for calibration work only:** an ω sign error *cancels* in a two-distance ray-bundle
solve, because the flip applies identically at both distances and spots are matched by
frame index within a sweep. It does **not** cancel in the paramfile or in forward
simulation. Evidence: `/Users/hsharma/Desktop/analysis/pokharel_jun25_nf/PREREGISTER.md:49-52`.

---

## 3. STEP 2 — Find the beamtime metadata and extract the scan definition

### 3a. Locate the metadata folder

The image tree has **only TIFFs**. The acquisition logs live elsewhere. For
`pokharel_jun25` on copland:

| what | path |
|---|---|
| images | `/gdata/dm/1ID/2025/pokharel_jun25/data/nf/` |
| **metadata** | `/home/1-id/s1iduser/1id_old_data_2026preJune/pokharel_jun25/` (= `~s1iduser/new_data/1id_old_data_2026preJune/pokharel_jun25/`) |

If you have not found a folder containing `FileCount.txt` + `fastsweep_Emon.txt` +
`*_SequenceOfEvents.log`, **you cannot write a paramfile.** Search for it:

```bash
ssh copland 'find /home/1-id /home/beams -maxdepth 5 -name "FileCount.txt" 2>/dev/null'
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
MD=/home/1-id/s1iduser/1id_old_data_2026preJune/pokharel_jun25
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
NrFilesPerDistance 720
StartNr 0
EndNr 719
RawStartNr 403783
OmegaStart 180            # NEGATED, aero  (logged -180 -> 0, step +0.25)
OmegaStep -0.25
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
$ awk '$7=="Au4_cubes_nf_96keV" && $8==404503 {print $8, $17}' pokharel_jun25_NF.par
404503 0.000000      <- last frame of sweep 1
404503 -180.000000   <- first frame of sweep 2
```

The later write wins, so **sweep 1 silently lost its ω = 0 frame.** Do not derive
frames-per-distance from `NF.par` row counts. Use `FileCount.txt` ranges as half-open
`[f16, f17)`: 404503−403783 = 720 ✓, 405223−404503 = 720 ✓.

**(ii) TIFF count vs the inventory.**

```bash
ls -1 /gdata/dm/1ID/2025/pokharel_jun25/data/nf/$SCAN/*.tif | wc -l   # -> 1441
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
      'contiguous:',max(v)-min(v)+1==len(v))" /gdata/dm/1ID/2025/pokharel_jun25/data/nf/$SCAN
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

**Trap: sweep starts appear twice in `NF.par`.** For `nf_Ce_ht525_s2_0p5deg` the
log had **756 rows for 721 images** — 35 duplicated image numbers, one per sweep
restart (`nCrashedFrames : 1` in the `_Sweep.log`). Here both copies carried the
*same* ω, so dedup is lossless — **but check that**, because a duplicate with a
*different* ω means a genuinely lost frame and a shifted ω axis for the rest of
the sweep. Counting rows without dedup inflates the frame count by 5%.

**Worked contrast, both from `pokharel_jul26`:**

| | `Au5_cubes_nf_96keV` (calibrant) | `nf_Ce_ht525_s2_0p5deg` (sample) |
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
sweep — except for the final sweep, which keeps it. That is why the Ce scan has
721 files for 2×360 frames, and why a stride of `NrFilesPerDistance` is still
exact. Set `EndNr = StartNr + NrFilesPerDistance − 1` and ignore the extra.

**The 1-ID "skip the first frame" rule is a GE / far-field detector rule and does
NOT apply to NF.** On the GE FF detector the first frame of every acquisition —
sweep, dark, calibrant — is a settling throwaway (1441 logged frames ⇒ usable
2..1441), and it must be dropped from averages and from the sweep definition. The
NF detector does not do this. **Never carry that rule across to an NF scan**:
`StartNr` is the first image. Confirmed with the instrument scientist.

Independent evidence from the data, which also holds on any such scan:

- The `+1` extra file sits at the **end**, not the start. The per-frame ω log
  reverses exactly at the second distance's first image (9093 for the Ce scan),
  and the final image carries the wrap ω = 0. Skipping a leading frame would push
  the reversal *inside* the first distance, which contradicts the log.
- `Au5_cubes_nf_96keV` reconstructed to confidence **1.000000** on the calibrant
  with `StartNr` = the first image. A one-frame ω error would not survive that.

So: on NF volume scans the extra file is a trailing wrap, and `StartNr` is the
first image. Re-derive this from the ω log for any new scan rather than assuming
either convention — the two differ by one ω step, which is invisible in the
`.mic` and degrades confidence without ever raising an error.

---

## 4. STEP 3 — Energy and distance: the two fields that lie

### 4a. Energy

**Use `fastsweep_Emon.txt` field 10. Nothing else.**

Two decoys, both verified in `pokharel_jun25`:

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

**Decoy 2 — `<beamtime>.spe` `#U Energy:`.**

```
$ grep -m3 "#U" pokharel_jun25.spe
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

**In `pokharel_jun25` those image directories exist and are EMPTY:**

```
$ for d in /gdata/dm/1ID/2025/pokharel_jun25/data/nf/*BeamPos*; do echo "$d : $(ls -A $d|wc -l)"; done
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

## 5. STEP 4 — Look at the raw frames before building anything

### 5a. Pattern: reduce remotely, plot locally

The shared env has no matplotlib (§1). Working example from this session:

```
remote reducer  -> writes au4_reduced.npz  (keys: detz{9,13}_{max,med,f0,f360,f719})
scp to Mac
/Users/hsharma/Desktop/analysis/pokharel_jun25_nf/plot_au4.py       -> 4 PNGs
/Users/hsharma/Desktop/analysis/pokharel_jun25_nf/check_artifacts.py -> the null-model check
```

Both scripts are checked in at that path; `check_artifacts.py` re-runs in seconds and
reproduces every number in §5b. Run it before trusting any spot count:

```bash
cd /Users/hsharma/Desktop/analysis/pokharel_jun25_nf
/Users/hsharma/miniconda3/envs/midas_env/bin/python check_artifacts.py
```

### 5b. Reference sanity numbers for a real 1-ID NF dataset

`Au4_cubes_nf_96keV`, 2048² uint16, px = 1.48 µm, 720 frames/distance, 2 distances.
All re-derived on 2026-07-29 by `check_artifacts.py`:

| quantity | DetZ 9 mm | DetZ 13.0001 mm |
|---|---|---|
| temporal-median background, mean | **6.68 counts** | 6.53 |
| median background, left half / right half | 3.16 / **10.20** | 2.99 / 10.06 |
| brightest pixel, single raw frame (frames 0/360/719) | 939 / 933 / 688 | 935 / 794 / 850 |
| brightest pixel, max-projection | 1294 | 1441 |
| brightest pixel, temporal median | 260 | 243.5 |
| static-hot pixels (median > 50) | **347** | 328 |
| static-hot set overlap | intersection 328 → **Jaccard 0.945** (i.e. the same pixels) | |
| max-projection px > 100 | **8423** | 8647 |
| of which coincide between distances | **152 → Jaccard 0.009** | |
| of which are static-hot | **164** | 139 |

Blob-size histogram of max-projection px > 100 (DetZ 9): **4893 blobs of 1 px, 435 of
2–3 px, 9 of 4–9, 4 of 10–29, then 17 blobs ≥ 30 px** (8 of 30–99, 9 of ≥ 100), totalling
2535 px. DetZ 13 gives 5149 / 427 / 3 / 4 / 17 — the same structure.

**Interpretation, and the operational rule.** The starfield is **cosmic rays**, not hot
pixels and not spots: Jaccard 0.009 between distances means those bright pixels are
transient, and only 164 of 8423 are static-hot. The **17 blobs ≥ 30 px are the real Bragg
spots.** A naive `npix > threshold` count on a max-projection therefore overestimates spot
content by ~500×.

**Rule: never count spots off a raw max projection.** Subtract the temporal median, then
LoG + connected components (`DoLoGFilter 1`), then filter by blob area.

**But `DoLoGFilter 1` is NOT an unconditional production default.** Operator knowledge
(1-ID, 2026-07-30): *the LoG path sometimes kills real signal*, and on weak-scattering
samples that loss matters more than the cosmic-ray suppression it buys. The rule above is
about **counting spots for a sanity check**, where you must not be fooled by cosmics; it is
not a blanket instruction for the production reduction.

Decision guide:

| Situation | Setting | Why |
|---|---|---|
| counting spots / auditing frame content (§5b) | `1` | cosmics dominate a raw max-projection ~500:1 |
| strong scatterer, dense spots (e.g. the Au calibrant) | either; Au was reconstructed to confidence 1.000 with `0` | |
| **weak signal** (e.g. `nf_Ce_ht525_s2`) | **`0`** | LoG can suppress genuine weak peaks; cosmics are then left in, and must be tolerated downstream |

If you change this key you **must regenerate `SpotsInfo.bin`** — it is baked into the
reduction, not applied at fit time.

Note the dynamic range: background ≈ 6.7, single-frame spot peaks ≈ 700–950, i.e. **~2
decades**. The §5b reductions were computed over 72 of the 720 frames (every 10th) for the
max projection and 18 frames for the temporal median
(`/Users/hsharma/Desktop/analysis/pokharel_jun25_nf/plot_au4.py:49,65`).

### 5c. Decision tree on what you see

| observation | conclusion | action |
|---|---|---|
| median background ~5–10 counts, single-frame spot peaks ~700–1000 | normal | proceed |
| left/right halves differ ~3× in the median | detector panel asymmetry, expected here | do not "correct" it; `BlanketSubtraction` after the temporal median is the intended knob |
| thousands of 1-px bright dots in max-proj | cosmic rays | ignore; they die in the median + LoG path |
| a few hundred pixels bright in the *median* | fixed hot pixels | expect ~330 at 2048²; they persist across distances |
| max-proj has < 10 blobs ≥ 30 px | too few spots to index | check ω range, energy, and that you have the right scan |
| `tifffile.imread` returns 3-D | multi-page TIFF | wrong layout for this code (§3f) |

---

## 6. Detector-distance and rotation-axis calibration (DetZBeamPos)

Validated on `pokharel_jul26` (95.0000 keV, Retiga, px 1.48 µm) against an independent
operator reading taken in `gui/nf_qt.py`. Everything below is measured, not inferred,
except where explicitly flagged.

### 6a. The split — neither measurement alone is enough

| measurement | gives | CANNOT give |
|---|---|---|
| **DetZBeamPos** (this section) | `BC` per distance, β (beam tilt vs the DetZ stage axis), sample position w.r.t. the rotation axis | **absolute Lsd** |
| **NF spot backprojection** (§6i) | **absolute Lsd** (the DetZ offset δ) | β |

Why DetZBeamPos cannot give Lsd: the sample sits on the rotation axis, and a point on the
beam axis casts its shadow at `BC` *regardless of L*. The beam is parallel (verified: shadow
width constant to 0.3 px across four distances), so there is no magnification cue either.

Why spots cannot give β: a tilt common to every ray is absorbed into the measured ray
slopes. See §6i for the derivation.

**Run DetZBeamPos first, then spots.** BC and the sample position come out of this section;
Lsd comes out of the spot data; together they close the geometry.

### 6b. Pixel convention — VALIDATED, do not re-derive it

```
ybc = 2047 − raw_column_index          zbc = 2047 − raw_row_index
```

where `raw_*` are indices into the array as `tifffile.imread` returns it.

Provenance, in order of strength:

1. `gui/nf_qt.py:1305` applies `self.imarr2 = self.imarr2[::-1, ::-1].copy()`
   **unconditionally**, on both the TIFF and `.bin` load paths — a 180° rotation, i.e. both
   axes reversed.
2. The cursor readout indexes into *that reversed array* (`gui/gui_common.py:714-731`:
   `mapSceneToView` then `_raw_data[iy, ix]` with `ix = int(x+0.5)`). `origin='br'` only
   calls `vb.invertX()`, which changes the drawn axis direction, **not** data coordinates.
3. **Empirical confirmation.** On `pokharel_jul26_95keVBeamPos_redH_NoBBwithAu0_000267.tif`
   (DetZ 7 mm), this pipeline measured (997.41, 38.26); an independent operator visual read
   in `nf_qt.py` gave **(997.7, 38.2)**. Agreement 0.29 px in y, 0.06 px in z.

**The constant is 2047, not 2048.** It is the plain index reversal on a 2048-long axis. The
2048 form is off by exactly one pixel (39.26 vs the measured 38.2 above) — which matters,
because `BCTol` in `ps_au.txt` is 0.2 px in z.

MIDAS consumes these as one entry per distance: `ybc`/`zbc` are lists and
`midas_nf_fitorientation/params.py:357` raises if `len(ybc) != n_distances`. The forward
model is `y_pixel = yBC + ydet/px`, `z_pixel = zBC + zdet/px`
(`midas_diffract/forward.py:1283-1284`, NF runs `flip_y=False`), and the image stack is
`[N, Z, Y]`, so **zbc pairs with the row axis and ybc with the column axis** in the reversed
frame above.

### 6c. Locate and decode the scan

Logs live in the acquisition-log folder (§3a), one per condition:

```bash
ls ~/new_data/<beamtime>/*BeamPosScan.txt
```

Condition names decode as:

| token | meaning |
|---|---|
| `NoBB` / `withBB` | beam block **out** / **in** |
| `NoAu` | **no sample** — this is the direct beam |
| `withAu0`, `withAu90`, `withAu180` | sample in beam at ω = 0, 90, 180° (as logged; apply the §2 sign rule before using ω for anything else) |
| `redH` | reduced beam height |

The macro line gives the distance series; the per-block summary lines give image → DetZ:

```bash
cd ~/new_data/<beamtime>
grep -E "^Ran|^Image #" <bt>_<E>keVBeamPos_NoBBNoAu_BeamPosScan.txt
#   Ran : DetZBeamPos( <startDetZ> <endDetZ> <step> <exposure> <prefix> )

# image -> DetZ, one row per position:
grep -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+" \
     <bt>_<E>keVBeamPos_NoBBNoAu_BeamPosScan.txt | awk '{printf "%s->%s  ", $1, $3}'
```

**Extract this per condition and never assume it from the macro** — `withBBNoAu` in
`pokharel_jul26` was three *separate* macro invocations with a different exposure, so its
mapping does not follow the others.

Images for all conditions usually share one folder, e.g.
`data/nf/Au_DetZBeamPos_95keV/<prefix>_<image:06d>.tif` (zero-padded to 6, §3).

### 6d. STEP A — zbc from the direct beam

The beam is vertically focused, so on the detector it is a **thin horizontal stripe**
(~10–15 rows). That makes the vertical centroid sharp and unambiguous.

For each DetZ, on the **`NoAu`** condition:

```python
p = a.mean(axis=1) - np.median(a.mean(axis=1))     # row profile, background removed
idx = np.where(p > 0.2 * p.max())[0]               # stripe rows
lo, hi = idx.min(), idx.max()
rows = np.arange(lo - 6, hi + 7)                   # pad, then intensity-weighted centroid
seg = np.clip(p[rows], 0, None)
row_c = (rows * seg).sum() / seg.sum()
zbc = 2047 - row_c
```

Do it for **both** beam heights if available — they must agree (see §6g).

### 6e. STEP B — ybc from the sample shadow at ω = 0 and 180

**The direct beam cannot give ybc.** Horizontally the stripe is a broad, slit-defined band
(833 px wide at full height in the reference dataset). Its centre is the centre of the
*illuminated region*, which is set by the slits and has nothing to do with where the
rotation axis is.

The sample gives it. A sample off the axis by `u` projects to `+u` at ω = 0 and `−u` at
ω = 180, so

```
axis = ( dip_centre(ω=0) + dip_centre(ω=180) ) / 2
```

cancels the sample's own offset **exactly**. Since the sample sits on the axis and a point
on the beam axis shadows at `BC`, this axis position **is** `BC_y(L)`.

Build the transmission profile along the stripe, using the matching `NoAu` image at the
same DetZ as the reference:

```python
lo, hi = stripe_rows(ref)                     # from the NoAu image, as in 6d
Iref = ref[lo:hi+1, :].sum(axis=0)
I    = au [lo:hi+1, :].sum(axis=0)
band = Iref > 0.05 * Iref.max()
T    = np.where(band, I / Iref, np.nan)
```

> **ESTIMATOR TRAP — this is the one that will bite you.** Do **not** centroid `1 − T` over
> the illuminated band. The shadow is a ~28 px dip to T ≈ 0.57, but the band is up to 833 px
> wide with ~2% noise, so the noise integral swamps the dip. Doing it that way on the
> reference dataset gave **66 px of scatter and non-monotonic** axis positions, and a
> sample offset swinging ±29 µm. Corrected, the same data gives **0.2 px** agreement.

Use the **midpoint of the two half-depth edges** — a knife-edge measurement, robust to the
dip's flat bottom:

```python
base, bottom = np.nanmedian(T[band]), np.nanmin(T[band])
half = 0.5 * (base + bottom)
imin = int(np.nanargmin(np.where(band, T, np.nan)))
# walk left and right out of the dip to the half-depth crossing, linear-interpolate each
xl = <left crossing>;  xr = <right crossing>
dip_centre = 0.5 * (xl + xr)                  # sub-pixel
ybc = 2047 - dip_centre
```

Reference implementation: `axis_from_dip.py` (`dip_centre()`), alongside `beam_center.py`
in `~/Desktop/analysis/pokharel_jul26_beampos/`.

**ω = 90 is the cross-check**, not an input: `dip_centre(90) − axis` is the orthogonal
component of the sample offset. In the reference dataset it came out −2.0 to −2.8 µm at
every distance and both beam heights.

### 6f. STEP C — fit β, then emit BC per distance

Fit each axis linearly against the **motor readback** (not absolute L — δ is unknown at this
stage, and it only shifts the intercept, leaving β unchanged):

```python
A = np.column_stack([np.ones(n), detz_um])
intercept, beta = np.linalg.lstsq(A, bc_values, rcond=None)[0]
```

Report `BC(DetZ) = intercept + β · DetZ[µm]` and evaluate it at each distance used by the
sample scan. Because β is a property of the beam/stage alignment, **this transfers to DetZ
values the calibration scan never visited.**

> **β MUST be measured per beamtime. Never borrow it.** Borrowing β from `ps_au.txt` (by
> differencing its two `BC` lines across its two `Lsd` lines) and applying it to
> `pokharel_jul26` was wrong by **62× in y** and **2.1× in z**, with y's magnitude
> underestimated so badly that the horizontal misalignment — which is in fact the *dominant*
> one, 4.6× larger than vertical — looked negligible.

**`BCTol 2 0.2` from `ps_au.txt` is too tight** for a seed carrying any per-distance
uncertainty. If β is measured as above, the seed is good to ~0.5 px and the stock tolerance
is fine. If β is *not* available, seed all distances with the same BC and open BCTol in the
affected axis to tens of pixels.

### 6g. Acceptance gates — check all five before using the numbers

| # | check | reference dataset achieved |
|---|---|---|
| 1 | full-height vs `redH` agree at every distance | 0.13–0.20 px (ybc), 0.10 px (zbc) |
| 2 | BC linear in DetZ, max residual | 0.74 px (y), 0.85 px (z) |
| 3 | sample offset `u` small and *consistent* across distances | −0.4 to −1.2 µm, all 4 distances |
| 4 | shadow width constant across distances (⇒ parallel beam) | 28.1–28.4 px = 41.8 µm |
| 5 | ω=90 offset small and consistent | −2.0 to −2.8 µm |

Gates 1 and 3 are the ones that caught the estimator bug in §6e: the broken estimator
passed neither.

### 6h. Reference numbers — `pokharel_jul26`, 95.0000 keV, px 1.48 µm

Images 251–285 in `data/nf/Au_DetZBeamPos_95keV/`, DetZ 7/9/11/13 mm, 9 conditions.

| DetZ (mm) | ybc | zbc |
|---|---|---|
| 7 | 997.00 | 38.31 |
| 9 | 1014.01 | 41.83 |
| 11 | 1029.68 | 44.13 |
| 13 | 1043.94 | 48.80 |

```
ybc(DetZ) = 942.91 + 0.007825 · DetZ[µm]        beta_y/p = +0.007825 px/um
zbc(DetZ) =  26.38 + 0.001689 · DetZ[µm]        beta_z/p = +0.001689 px/um
```

Sample on the rotation axis to 0.5 µm; sample width 41.8 µm (ω=0/180) and 47.5 µm (ω=90).

### 6i. Fallback when there is no DetZBeamPos scan

This happens — in `pokharel_jun25` all three `Au_DetZBeamPos*` folders exist but are
**empty** (§3f). Then BC has to come from the sample's own diffraction spots, with a real
loss of information.

Match one spot across two distances. Its slope is measured exactly and the unknown offset δ
cancels:

```
r_k = Δy_k / ΔD          s_k = Δz_k / ΔD          (ΔD from the DetZ motor)
```

Substituting back at distance 1 leaves a **linear** system in three unknowns
`[A_y, A_z, L1]`, two equations per spot, so N ≥ 2 spots suffice:

```
y_k1 = A_y + L1 · r_k          z_k1 = A_z + L1 · s_k
```

Equivalently, the distance-1 → distance-2 map is a pure radial scaling about `A` by
`k = L2/L1`; two correspondences give `(A_y, A_z, k)` in closed form, which is the RANSAC
hypothesis. **This is where absolute Lsd comes from** (`δ = L1 − DetZ₁`).

What you get and do not get:

- **`L1`, hence δ — yes.** On `pokharel_jun25` Au this gave δ = 153–178 µm across six
  accepted solves at two energies, bootstrap ±4 µm within a dataset.
- **β — no, structurally.** With a tilt β the true model is
  `y_k(L) = A_y + L·(β + a_k)/p`, and the measured slope `r_k` *is* `(β + a_k)/p`. β is
  absorbed. The fit returns `A`, the projection of the **rotation axis** at L = 0 — which is
  **not** MIDAS's `BC`. They differ by `β·L`: `BC(L) = A + β·L`.
- Therefore **per-distance BC is unrecoverable from spots alone.** Seed one BC for all
  distances and widen `BCTol`.

Mandatory controls, both cheap, both of which caught real problems:

- **Position-scrambled null:** permute the *position* columns of the distance-2 spot list
  independently of (ω, area). Permuting whole rows is a **no-op** — the ω/area gate matches
  on column values, so row order changes nothing, and the "null" silently re-runs the real
  analysis. This bug produced a falsely tight 5.1 µm null scatter before it was found.
- **ω-shuffled null:** pair distance-1 spots with distance-2 spots at a *different* ω. Every
  pair is then physically impossible.

Both must fail decisively. On Au4 they returned **0 of 200** consensus solutions against 136
inliers for the real pairing.

Also gate every solve on: y-only vs z-only `L1` agreement (< 200 µm), `cond(A)` (7–8 when
healthy), and leave-one-out stability. On `pokharel_jun25` Au3 the `6→7` pair failed five
gates at once (δ = 5012 µm, cond 115, y/z split 947 µm); naively averaging all three pairs
would have given δ = 1786 µm instead of 174 µm — a 10× error.

---

## 7. STEP 5 — Refine the geometry on a calibrant, and know what it cannot do

§6 gives `BC` per distance and a starting `Lsd`. That is *not yet a usable
geometry*: the three detector tilts `tx/ty/tz` are still unknown, and `Lsd` is
only as good as the triangulation. This step refines them against a known
single-crystal calibrant (a gold cube), then hands the result to the real sample.

**Read §7b before running anything here.** The obvious ways to do this
refinement have all been tried on real data and all fail silently — they return
a confident, wrong answer rather than an error.

### 7a. What is measured, what is refined, what is fixed

| Quantity | Where it comes from | Refined here? |
|---|---|---|
| `BC` per distance | DetZBeamPos direct beam + shadow (§6) | only within `BCTol`, tiny |
| `Lsd` per distance | spot triangulation (§6a), or DetZ + δ | yes |
| `tx` | direct-beam stripe slope (§6f) | yes, from that seed |
| `ty`, `tz` | **nothing measures them** — start at 0 | yes |
| ω convention | `NF.par` f9 (§2) | never — fixed input |
| energy / λ | `fastsweep_Emon.txt` f10 (§4a) | never — fixed input |

A single direct-beam stripe carries no first-order signature of `ty` or `tz`, so
they must come out of the calibrant fit. This is exactly why the fit is
under-determined in the way §7b describes.

### 7b. Three verified negatives — do not rediscover these

All three were established on `pokharel_jul26` / `Au5_cubes_nf_96keV`, 4
distances, 95 keV. They are properties of the *problem*, not bugs.

**(1) Confidence 1.0 is a plateau, not a unique solution.**
Single-voxel refinements seeded at `ty` = 0.559, 1.507 and 2.622 deg all converge
to confidence **exactly 1.0000**. Reaching confidence 1 tells you the geometry is
*self-consistent with the spots you kept*; it does not tell you it is *the*
geometry. Never report "confidence 1.0, therefore calibrated".

**(2) `-multiGridPoints` does NOT break the degeneracy.**
The natural fix — refine against many voxels at once — was run. Seeded from the
`ty`=0.559 plateau it converged to `ty` 0.683 (mean confidence 0.9562); seeded
from `ty`=2.622 it converged to `ty` 2.985 (mean confidence 0.9753). The two
answers are **2.3 deg apart in `ty` and ~48 µm apart in `Lsd`**, and both look
excellent. Cause: on a calibrant cube every voxel belongs to **one grain**, so
twelve voxels contribute one orientation's worth of constraint, not twelve.
Multi-point helps only if the voxels sample *different* grains.

**(3) Never iterate single-point refinement.**
`TiltsTol` is interpreted **relative to the current seed**, not as an absolute
bound. Feeding a refinement's output back in as the next seed therefore ratchets
the tilts outward by roughly the tolerance each pass (~1 deg/iteration observed);
`ty` walked to 4.6 deg while confidence stayed high the whole way. Run the
refinement **once** from a defensible seed. If you want more iterations, use
`NumIterations` inside a single invocation, which does respect the original seed.

**Also observed, unexplained:** `LsdRelativeTol 5` (a value used successfully in
the past) **stalled** at confidence 0.27 on this dataset, while
`LsdRelativeTol 1` succeeded. Do not assume 5 is a safe default here; it is
recorded as an anomaly, not as a rule.

### 7c. The procedure that works

Single voxel, one invocation, tight tolerances, everything you actually measured
held nearly fixed.

```bash
# params.txt for calibration -- differences from a recon paramfile:
#   Rsample small (30), GridSize 2         -- a calibrant cube is tiny
#   MinConfidence 0.7                      -- reject junk seeds
#   BCTol 0.02 0.02                        -- BC is MEASURED (§6); barely let it move
#   LsdTol 500 / LsdRelativeTol 1          -- see the LsdRelativeTol 5 anomaly above
#   TiltsTol 1                             -- NOT the 0.05 code default: ty/tz start
#                                             at 0 but may really be ~0.5 deg, and a
#                                             0.05 deg tanh box makes them unreachable
#   NumIterations 3                        -- iterate INSIDE one call, not by re-seeding
#   GridPoints <a full 12-column .mic row> -- see the format trap below
```

`GridPoints` takes a **raw 12-column `.mic` data row**, not an abbreviated
coordinate triple (`FitOrientationParametersMultiPoint.c:697` `sscanf` reads 12
tokens). Passing 6 tokens parses without error and silently refines nothing
useful. Take the row straight out of a previous `.mic`.

Then verify the fit reproduces the *known* answer on one voxel before trusting it
anywhere:

```bash
midas-nf-fit-orientation params.txt 0 1 16 --device cuda --fp32
# read Confidence out of the .mic; on Au it should be 1.000000 with BoxSize set
```

### 7d. `BoxSize` — implemented in Python, and it changes the answer

`BoxSize` was parsed but **never applied** in the Python path until this tree.
The gate is:

- units **µm on the detector at `Lsd[0]` only** — applied before displacement,
  tilts and `BC`;
- comparisons are **strict** (`>` low, `<` high);
- paired with `OmegaRange` **by index**; a spot is kept if it satisfies **any**
  (`OmegaRange`, `BoxSize`) pair;
- rejected spots are **excluded from the confidence denominator**, not counted as
  misses.

That last point is why it matters so much: with the gate off, a single Au voxel
scores **0.949153**; with it on, **1.000000**, matching the C reference exactly.
The Triton fused kernel implements the same gate (verified: eager 1.000000 /
triton 1.000000 gate-on, 0.949153 / 0.949153 gate-off).

**If confidence plateaus just below 1 on a calibrant, check `BoxSize` before
touching the geometry.** A missing gate looks exactly like a slightly-wrong
geometry.

### 7e. Discriminating two candidate geometries with a full map — a WEAK test

When §7b leaves two plausible geometries, the instinct is to reconstruct under
each and pick the better map. Do it, but calibrate your expectations: **on
`pokharel_jul26` this test did not cleanly separate them.**

Both geometries produced a single coherent gold crystal, and:

| | geometry A | geometry B |
|---|---|---|
| mic rows | 5012 | 5044 |
| median confidence (whole grid) | 0.170 | 0.273 |
| voxels `C >= 0.9` | 384 | 477 |
| equiv-area radius at `C>=0.9` | 14.55 µm | 16.22 µm |
| Eul1 (high-C) | 2.5807 rad | 2.5800 rad |
| Eul1 spread | 0.148 deg | 0.186 deg |
| high-C voxels beyond shadow radius, about own centroid | **0.000** | **0.000** |

The two orientations agree to **0.0007 rad (0.04 deg)** — far below either map's
internal spread. B has uniformly higher confidence at every threshold; that is a
shift of the confidence scale, not a structurally different microstructure.

**The metric trap that nearly inverted this conclusion.** Measuring "fraction of
high-confidence voxels outside the known sample radius" **from the grid origin**
gave A 0.02 and B 0.09, which reads as B smearing signal into empty space. It is
an artifact: **the sample is not on the rotation axis.** Both blobs are offset
(A centroid −4.62 µm, B −6.89 µm), and once radius is measured about **each
map's own high-confidence centroid** the difference vanishes entirely — both are
0.000. Always centroid-correct before comparing blob sizes.

Useful external anchor: the DetZBeamPos shadow gives a sample radius (20.9 µm
here) that is **not** derived from the fits being compared, so it is the only
independent size check available. But note it is threshold-dependent — at
`C>=0.90` both blobs read ~15-16 µm, at `C>=0.70` both read ~23 µm — so it
bounds gross errors rather than resolving fine ones.

**Bottom line: budget for the map being a weak discriminator.** If two geometries
survive §7b, expect to choose on aggregate confidence and operator judgement, and
label the choice provisional.

### 7f. Reference — `pokharel_jul26` Au5, the geometry actually adopted

95.0000 keV, λ 0.1305097 Å, px 1.48 µm, 2048², DetZ 7/9/11/13, stage `aero`.
Adopted **geometry A**:

```
Lsd 7228.584913     BC 996.716776  37.941506
Lsd 9229.709611     BC 1013.675328 41.313334
Lsd 11229.713336    BC 1029.377525 44.328838
Lsd 13228.960327    BC 1043.678358 47.979195
tx 0.788229   ty 0.683384   tz 0.082687
```

Geometry B (`Lsd` 7276.153002/9278.833913/11280.854593/13281.673523, `tx` 0.705447
`ty` 2.985390 `tz` 0.446467) was the competing plateau and was judged slightly
worse. **The margin was small — treat A as the working geometry, not as a
refutation of B.**

Because the calibrant and the real samples sit on the same rotation axis at the
same DetZ positions, these `Lsd`/`BC` values transfer directly to any scan in the
beamtime that uses the same DetZ set. That transfer is the entire reason to
reconstruct the gold first.

---

## 8. STEP 6 — Run the reconstruction

### 8a. DO NOT use `midas-nf-pipeline run`

Five call sites in the orchestrator do not match their callees. **Re-verified 2026-07-29
by reading both sides**; all five survive. The pipeline was **not executed** — the
exception *types* below follow from Python semantics, the exception *text* is not quoted
because it was not observed.

Summary: #1, #2 and #4 fire on **every** run; #3 only with `TomoImage` set; #5 only on
`refine-params` without `--multi-point`.

**Defect 1 — `run_diffr_spots` → `diffr_spots.cli.run`. Fires on every loop-0 run.**

```python
# packages/midas_nf_pipeline/midas_nf_pipeline/stages.py:233-237  (CALL SITE)
    args = Namespace(
        parameter_file=str(param_file),
        device=None, dtype=None, output_dir=None,
    )
    diffr_run(args)

# packages/midas_nf_preprocess/midas_nf_preprocess/diffr_spots/cli.py:40-49  (CALLEE)
def run(args: argparse.Namespace) -> int:
    params = DiffrSpotsParams.from_paramfile(args.parameter_file)
    pipe = DiffrSpotsPipeline(
        params,
        device=args.device,
        dtype=args.dtype,
        hkls_csv=args.hkls_csv,                    # :46
        seed_orientations_csv=args.seeds,          # :47
    )
    result, paths = pipe.run(output_dir=args.output_dir)
```

`AttributeError`: neither `hkls_csv` nor `seeds` is on the Namespace. (`output_dir` is.)

**Defect 2 — `run_image_processing` → `process_images.cli.run`. Fires on every run with
image processing.**

```python
# packages/midas_nf_pipeline/midas_nf_pipeline/stages.py:287-293  (CALL SITE)
        args = Namespace(
            parameter_file=str(param_file),
            distance_nr=d,
            n_cpus=int(p.get("nCPUs", 1)),
            device=None, dtype=None,
        )
        proc_run(args)

# packages/midas_nf_preprocess/midas_nf_preprocess/process_images/cli.py:49-60  (CALLEE)
def run(args: argparse.Namespace) -> int:
    params = ProcessParams.from_paramfile(args.parameter_file)
    pipe = ProcessImagesPipeline(
        params, device=args.device, dtype=args.dtype, n_cpus=args.n_cpus
    )
    if args.all_layers:                            # :54
        bitmask = pipe.process_all()
    else:
        bitmask = pipe.process_layer(args.layer_nr)  # :57
    out = args.output or str(Path(params.output_directory) / "SpotsInfo.bin")  # :59
```

`AttributeError` on `all_layers` at `:54`. The callee also wants `layer_nr` (not
`distance_nr`) and `output`, neither of which is supplied.

**Defect 3 — `run_tomo_filter` → `filter_grid_by_tomo`. Only when `TomoImage` is set.**

```python
# packages/midas_nf_pipeline/midas_nf_pipeline/stages.py:201-203  (CALL SITE)
    new_grid_path = filter_grid_by_tomo(
        str(grid_path), tomo=tomo, tomo_pixel_size=tomo_pixel_size,
    )

# packages/midas_nf_preprocess/midas_nf_preprocess/tomo_filter/filter.py:121-125  (CALLEE)
def filter_grid_by_tomo(
    grid_points: torch.Tensor,
    tomo: Union[np.ndarray, torch.Tensor],
    px_tomo_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
```

Three mismatches: `TypeError` — unexpected keyword `tomo_pixel_size` (the parameter is
`px_tomo_um`); a `str` is passed where an `(N,5)` Tensor is required
(`filter.py:142-145` raises on `grid_points.ndim`); and a 2-tuple is returned where
`stages.py:205` expects a path.

**Defect 4 — both seed stages → `write_seeds_csv`. Fires on every run.**

```python
# packages/midas_nf_pipeline/midas_nf_pipeline/stages.py:150-151  (CALL SITE, FF path)
    seeds = read_grains_orientations(grains_path)
    write_seeds_csv(out_path, seeds)
# packages/midas_nf_pipeline/midas_nf_pipeline/stages.py:169-171  (CALL SITE, cache path)
    seeds = load_seeds_for_space_group(sg, seed_dir=seed_dir)
    out_path = p["SeedOrientations"]
    write_seeds_csv(out_path, seeds)

# packages/midas_nf_preprocess/midas_nf_preprocess/seed_orientations/io.py:24-31  (CALLEE)
def write_seeds_csv(
    quats: torch.Tensor,
    path: Union[str, Path],
    *,
    fmt: str = "%.7f",
) -> None:
    """Write a ``(N, 4)`` quaternion tensor to a comma-separated CSV."""
    if quats.ndim != 2 or quats.shape[1] != 4:     # :31
```

Arguments **transposed** — `quats` receives a `str`, which has no `.ndim`. Return types
confirmed: `load_seeds_for_space_group(...) -> torch.Tensor`
(`seed_orientations/from_cache.py:127-133`); `read_grains_orientations(...) ->
list[GrainOrientation]` (`seed_orientations/from_grains.py:41-46`) — the FF path
additionally needs `write_seeds_with_lattice_csv` (`io.py:53-56`), not `write_seeds_csv`.

**Defect 5 — `cmd_refine_params` → `fit_parameters_run`. Only without `--multi-point`.**

```python
# packages/midas_nf_pipeline/midas_nf_pipeline/cli.py:201-205  (CALL SITE)
        from midas_nf_fitorientation import fit_parameters_run
        fit_parameters_run(
            args.paramFN, row_nr=args.row_nr,
            n_cpus=args.nCPUs, device=args.device,
        )

# packages/midas_nf_fitorientation/midas_nf_fitorientation/fit_parameters.py:45-54  (CALLEE)
def fit_parameters_run(
    paramfile: str,
    voxel_idx: int,
    n_cpus: int = 1,
    *,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    lbfgs_config: Optional[LBFGSConfig] = None,
) -> dict:
```

`TypeError`: no `row_nr` parameter, and required positional `voxel_idx` is missing.

**Not broken: `--multi-point`.** `cli.py:198-199` calls
`fit_multipoint_run(args.paramFN, n_cpus=args.nCPUs, device=args.device)` against
`fit_multipoint.py:75-84` `def fit_multipoint_run(paramfile: str, n_cpus: int = 1, *,
device: str = "auto", ...)`. Matches.

**Workaround for #5:** the standalone CLI is correct — `midas-nf-fit-parameters` passes
`voxel_idx=row_nr` (`midas_nf_fitorientation/cli.py`, `fit_parameters_main`). Use
`midas-nf-fit-parameters params.txt <rowNr> [nCPUs]`.

Supporting evidence the `run` path is untested: every code cell in
`packages/midas_nf_pipeline/notebooks/00_quickstart_au.ipynb`,
`01_single_resolution.ipynb`, `02_multi_resolution.ipynb` and `05_multi_layer_batch.ipynb`
has `execution_count: null`. The preprocess and fitorientation notebooks are marked "runs
clean" (`packages/midas_nf_preprocess/notebooks/README.md`,
`packages/midas_nf_fitorientation/notebooks/README.md`).

### 8b. The route that works — nine commands

Every step below has an `argparse`-defined signature, so the flags are the flags. **Run all
of them from inside `OutputDirectory`** — the fitter resolves all inputs relative to it
(`fit_orientation.py:238-251`).

```bash
cd <OutputDirectory>

# 0. hkls.csv — no console script exists for the NF variant. Use the stage helper,
#    which is NOT one of the five broken call sites (stages.py:118-136).
/home/beams12/S1IDUSER/opt/envs/midas/bin/python - <<'PY'
from midas_nf_pipeline.params import parse_parameters
from midas_nf_pipeline import stages
p = parse_parameters('params.txt'); p['resultFolder'] = '.'
stages.run_get_hkls(p, 'params.txt')
PY

# 1. seed orientations
midas-nf-preprocess seed-orientations --method cache --space-group 225 \
    --output seedOrientations.csv
#    FF-seeded instead (writes the 11-column layout; diffr-spots reads the first
#    4 columns, hkls.py:107-113, so it is accepted directly):
# midas-nf-preprocess seed-orientations --method from-grains \
#     --grains-file Grains.csv --output seedOrientations.csv
#    then set  SeedOrientations seedOrientations.csv
#         and  NrOrientations <wc -l of that file>   in params.txt

# 2. voxel grid
midas-nf-preprocess hex-grid params.txt                 # -> grid.txt

# 3. optional grid mask (bypasses broken call site #3)
midas-nf-preprocess tomo-filter grid.txt grid_filt.txt --tomo tomo.bin --px-tomo 1.5
# midas-nf-preprocess tomo-filter grid.txt grid_filt.txt --bbox -500 500 -500 500

# 4. forward-simulate candidate spots
midas-nf-preprocess diffr-spots params.txt              # -> Key.bin OrientMat.bin DiffractionSpots.bin

# 5. raw TIFFs -> SpotsInfo.bin.  --all-layers IS MANDATORY.
#    The positional "1" is a DETECTOR DISTANCE index and is ignored with --all-layers.
midas-nf-preprocess process-images params.txt 1 --all-layers --device cuda --dtype fp32

# 6. fit orientations (positionals: paramfile blockNr nBlocks nCPUs)
midas-nf-fit-orientation params.txt 0 1 8 --device cuda --fp32

# 7. binary -> text .mic + .map/.map.kam/.map.grainId/.map.grod
midas-nf-pipeline parse-mic params.txt

# 8. cluster voxels into grains (doNeighborSearch=1 -> spatial BFS)
midas-nf-pipeline mic2grains params.txt out.mic Grains.csv 1 8

# 9. bundle into one readable HDF5
midas-nf-pipeline consolidate out.mic --paramFN params.txt --output out_consolidated.h5
```

Seed cache: `NF_HEDM/seedOrientations/`, overridable with `$MIDAS_NF_SEED_DIR` or
`--seed-dir` (`from_cache.py:36-49`). Fully populated in this checkout:
`seed_cubic_high.csv` 243129 rows, `seed_hexagonal_high.csv` 486755 rows. The
`orientations_master.bin` + `lookup_<type>.bin` fallback is present too
(`from_cache.py:64-82`).

Fit-orientation flags (`midas_nf_fitorientation/cli.py:33-63`): `--device {auto,cpu,cuda}`,
`--fp32`, `--screen-only`, `--verbose`, `--lbfgs-max-outer N` (20), `--lbfgs-max-iter N`
(20), `--refine {nm-batched,nm-serial,lbfgs+nm,lbfgs}` (default `nm-batched`),
`--nm-max-iter N` (200; the C used 5000), `--nm-batch-size N` (4096 — the GPU-memory
knob).

`nm-triton` is **not** a CLI choice (`midas_nf_fitorientation/cli.py:47-48`). It is
auto-selected when `--refine nm-batched` **and** device is CUDA **and** Triton is
importable **and** the obs volume is bit-packed (`fit_orientation.py:370-377`).

Manual sharding: block *b* of *nBlocks* covers voxels
`[ceil(N/nBlocks)*b, min(ceil(N/nBlocks)*(b+1), N-1)]` (`io.py:236-245`). **Multi-process
sharding is deliberately not wired** — `MicFileBinary` writes need a `pwrite`-safety audit
first (`packages/midas_nf_pipeline/USAGE.md:244-256`).

### 8c. Multi-layer by hand

`run_multi_layer` (`workflows.py:556`) is part of the broken `run` path. Reproduce it
manually: for each sample layer *n*, make `<result-folder>/LayerNr_<n>/`, copy the
paramfile in, and rewrite two keys (`workflows.py:586-594`):

```
OutputDirectory  <result-folder>/LayerNr_<n>
RawStartNr       RawStartNr0 + (n-1) * nDistances * NrFilesPerDistance
```

then run §8b in that directory. Per-layer grain list afterwards:
`midas-nf-pipeline mic2grains ... > <result-folder>/GrainsLayer<n>.csv`.

**Known inconsistency in the built-in version** (read, not executed): the per-layer grain
list is built from `<base>.<NumLoops>.mic` — the *seeded* pass of the last loop
(`workflows.py:608-612`) — not from `<base>_merged.<NumLoops>.mic`, which the loop itself
designated final (`workflows.py:532`). If you build grain lists yourself, use the merged
file.

**Multi-resolution**, if you want it: `GridRefactor StartingGridSize ScalingFactor NumLoops`;
absent ⇒ single resolution, `NumLoops = 0`, only loop 0 runs — there is no separate code
path (`workflows.py:253-261`). Loop *k* runs at `StartingGridSize / ScalingFactor**k`
(`workflows.py:373`). Each loop *k* ≥ 1 (`workflows.py:369-542`): rewrite `GridSize` into
the paramfile → `Mic2GrainsList` on the previous `.mic` → seeded pass → **bad-voxel
filter** (voxels with `Confidence < MinConfidence` collected, `grid.txt` *overwritten*
with only those lines, `workflows.py:437-470`; short-circuits if none) → unseeded pass on
just those voxels from `SeedOrientationsAll` → binary merge by `pwrite` overlay at
full-grid offsets, then `ParseMic` (`workflows.py:164-219`). `doImageProcessing` is forced
to 0 from loop 1 on (`workflows.py:371`).

Stage labels, in resume order (`workflows.py:54-68`): `loop_0_initial`, then
`loop_<k>_seeded`, `loop_<k>_unseeded`, `loop_<k>_merge`.

### 8d. Where files land

Everything is **flat** in `OutputDirectory` (fallback `DataDirectory`, then cwd —
`workflows.py:239-244`); the driver `chdir`s into it (`workflows.py:308`).

| File | Written by | Contents |
|---|---|---|
| `midas_log/` | driver (`workflows.py:242-243`) | log dir |
| `hkls.csv` | HKL gen | 11 cols `h k l d RingNr g1 g2 g3 θ 2θ R` (`midas_hkls/nf_hkls.py:1-21`) |
| `<SeedOrientations>` | seed stage | comma-separated `w,x,y,z` |
| `grid.txt` | hex grid | count line, then `dx dy x y edge_half` per voxel (`hex_grid/io.py:1-9`); the fitter reads the same 5 columns as `y1 y2 xs ys gs` (`fitorientation/io.py:273-305`) |
| `grid_unfilt.txt`, `grid_old.txt` | tomo / mask filter | pre-filter copies |
| `DiffractionSpots.bin`, `OrientMat.bin`, `Key.bin` | diffr-spots | `T×3` f64, `N×9` f64, `N×2` int32 (`fitorientation/io.py:56-86`) |
| `SpotsInfo.bin` | image processing | bit-packed int32 spot mask (`process_images/spots_io.py:90-96`: sized `nDistances * NrFilesPerDistance * NrPixelsY * NrPixelsZ` bits) |
| `<MicFileBinary>` | fitting | 11 f64/voxel at offset `voxel_idx*88` (`fitorientation/output.py:32-56`) |
| `<MicFileBinary>.AllMatches` | fitting | `7 + 4*SaveNSolutions` f64/voxel (`output.py:92`, `parse_mic.py:585`) |
| `screen_cpu.csv` | fitting with `--screen-only` | phase-1 dump (`fit_orientation.py:310`) |
| `<MicFileText>.mic` + `.map` `.map.kam` `.map.grainId` `.map.grod` | `ParseMic` | §9 |
| `<base>_pipeline.h5` | `PipelineH5` | provenance + completed stages |
| `<base>_consolidated.h5` | consolidator | §9c |

Backups the multi-resolution driver leaves: `DiffractionSpots.bin_unseeded_backup` and
friends (`workflows.py:81-97`), `<MicFileBinary>.seeded_backup`, `.unseeded_backup`,
`<SeedOrientationsAll>_Backup`.

### 8e. Multi-phase samples — reduce ONCE, fit once per phase

**The NF path fits one phase per run.** `NumPhases` and `PhaseNr` are forwarded
only to `parse_mic` (`stages.py:359-360`); `diffr-spots` and `fit-orientation`
each read a single `LatticeParameter`/`SpaceGroup`. A two-phase sample therefore
needs two paramfiles and two runs.

**But do not re-run image processing for the second phase.** `SpotsInfo.bin` is
**phase-independent** — verified by reading `ProcessParams`
(`process_images/params.py:29-58`), which parses only I/O, frame indexing and
reduction keys (`BlanketSubtraction`, `DoLoGFilter`, `MedFiltRadius`,
`LoGMaskRadius`, `GaussFiltRadius`, `RawStartNr`, `NrFilesPerDistance`,
`nDistances`). It reads **no** lattice parameter, space group, wavelength, `Lsd`,
`BC`, tilt or `MaxRingRad`. Nothing in the reduction depends on the crystal.

So:

```bash
# phase 1 — does the reduction
midas-nf-pipeline run params_phase1.txt --n-cpus 32 --device cuda

# phase 2 — reuse the SAME SpotsInfo.bin, skip the 1440-TIFF reduction entirely
ln -s <phase1>/LayerNr_1/SpotsInfo.bin <phase2>/LayerNr_1/SpotsInfo.bin
midas-nf-pipeline run params_phase2.txt --n-cpus 32 --device cuda \
    --no-image-processing
```

The two phase runs are otherwise independent and can go on separate GPUs
concurrently.

**What DOES invalidate a shared `SpotsInfo.bin`:** changing any reduction key
above. In particular `BlanketSubtraction` and `DoLoGFilter` are baked in at
reduction time, so changing either forces a regeneration for *both* phases.

**Wanted (not built): a proper multi-phase driver** that reduces once and then
loops the fit over N phases, merging the per-phase `.mic` by confidence into one
multi-phase map. Today that orchestration is manual, and `PhaseNr` in the `.mic`
is only whatever the paramfile declared — it is *not* evidence that a phase
assignment was fitted.

### 8f. Comparing two runs — never by checksum

`MicFileBinary` is **11 float64 per voxel** (`output.py:32-56`), in this order:

```
0 OrientRowNr  1 OrientID  2 RunTime  3 X  4 Y  5 TriEdge
6 UpDown  7 Eul1  8 Eul2  9 Eul3  10 Confidence
```

Column 2 is a **per-voxel wall-clock time**. It differs on every run, so `md5sum`
of two physically identical reconstructions never matches. Compare field by field:

```python
import numpy as np
NAMES = ["OrientRowNr","OrientID","RunTime","X","Y","TriEdge",
         "UpDown","Eul1","Eul2","Eul3","Confidence"]
a = np.fromfile("run_a.bin", dtype=np.float64).reshape(-1, 11)
b = np.fromfile("run_b.bin", dtype=np.float64).reshape(-1, 11)
for c, n in enumerate(NAMES):
    d = np.abs(a[:, c] - b[:, c])
    print(f"{n:12s} ndiff {int((d>0).sum()):6d}  maxabs {d.max():.6g}")
```

**Read it as float64, not float32.** A float32 view splits each field across two
columns and makes one changed `RunTime` look like two changed physical
quantities — including something that reads convincingly as a confidence shift
of 0.35. That misread cost real time; do not repeat it.

Reference: the `screen()` dtype rework (float intermediates → `bool`/`int32`) was
validated exactly this way on a 5046-voxel grid — every field bit-identical,
`RunTime` the only difference.

### 8g. `screen()` memory — chunking and its knob

The vectorised path builds `(V, T, 3)` tensors, where `T` is the **total**
simulated-spot count over every candidate orientation (~3×10⁷ for a cubic seed
list). One voxel already costs `T*3*itemsize`; a full grid is terabytes
(5046 × 3.02×10⁷ × 3 × 4 B = **1704 GiB**, the allocation that actually failed).
It only ever worked before because calibration runs have `V = 1`.

Voxels are therefore processed in chunks sized from free device memory. Override
with:

```bash
MIDAS_NF_SCREEN_VOXEL_CHUNK=<n>     # voxels per chunk; omit to auto-size
```

Auto-sizing is the right default. Forcing it **too large OOMs**: on a 47 GiB
A6000, `MIDAS_NF_SCREEN_VOXEL_CHUNK=64` died trying to allocate a further
21.62 GiB. Results are independent of the chunk size — verified by comparing runs
at a fixed forced chunk (identical in every field but `RunTime`).

Performance is *not* the problem and never was: 5046 voxels × 4 distances took
`screen=697.22s nm_batched=8.03s writeback=23.45s` on one A6000, against ~748 s
for the C reference scaled to the same grid. Memory was the only defect.

---

## 9. STEP 7 — Read the result

### 9a. The text `.mic`

Header: **three `%` lines plus one `%` column-name line** ⇒ `skip_header=4`
(`consolidate.py:47-49`). Column names are the literal string at `parse_mic.py:145-147`;
the *meanings* come from the writer's `MicRecord` (`fitorientation/output.py:37-56`).

| col | header name | what it actually holds | units |
|---|---|---|---|
| 0 | `OrientationRowNr` | row index of the winning candidate in the seed list (`best_row_nr`) | index |
| 1 | `OrientationID` | **number of phase-1 winners** carried into phase 2 (`n_winners`) — **not an ID** | count |
| 2 | `RunTime` | per-voxel fit wall time | s |
| 3 | `X` | voxel centre, sample frame | µm |
| 4 | `Y` | voxel centre, sample frame | µm |
| 5 | `TriEdgeSize` | voxel triangle size | µm |
| 6 | `UpDown` | `+1` if grid col0 ≤ col1 else `-1` (`fitorientation/io.py:298`, `:248-269`) | ±1 |
| 7–9 | `Eul1..3` | Bunge ZXZ | **radians** |
| 10 | `Confidence` | FracOverlap of the best solution | 0–1 |
| 11 | `PhaseNr` | copied from the `PhaseNr` key | int |

Radians confirmed by the bundled reference row `… 4.860570 0.768787 2.147169 0.114583 1`
in `NF_HEDM/Example/Au_txt_Reconstructed.mic`.

**Two things that will bite you:**

- **Rows with `Confidence == 0` are silently dropped** from the text file
  (`parse_mic.py:150-151`). The text `.mic` is *shorter* than the voxel grid. Only
  `MicFileBinary` is one-record-per-voxel. **Row *i* of the `.mic` is not voxel *i*.**
- **`%TriEdgeSize` is read off row 0 of the binary** (`parse_mic.py:140`), which may be an
  *invalid* voxel. The bundled Au reference has header `%TriEdgeSize 0.000000` while every
  data row carries `1.000000`. Downstream: `mic2grains` falls back from spatial to global
  merging when `TriEdgeSize <= 1e-6` (`mic2grains.py:365-373`) and the grain radius
  collapses to zero (`mic2grains.py:294`). **Read column 5 of a data row.**

### 9b. The companion binaries

Little-endian float64, **4-double header `[xSize, ySize, minX, minY]`**
(`parse_mic.py:1-22`).

| file | payload | units |
|---|---|---|
| `<MicFileText>.map` | 7 planes of `xSize*ySize`: Confidence, Eul1, Eul2, Eul3, OrientationRowNr, PhaseNr, distance-to-voxel-centre (`parse_mic.py:290-304`) | mixed; Eulers rad |
| `.map.kam` | 1 plane: KAM over the assigned 8-neighbours | **radians** |
| `.map.grainId` | 1 plane: connected-component grain label, BFS with edge threshold `GBAngle` | int, 1-based |
| `.map.grod` | 1 plane: misorientation to the highest-confidence pixel of the same grain | **radians** |

Unassigned pixels: `-15` in `.map` (`parse_mic.py:306`), `0` in the single-plane files.

**`.map.grainId` is the only real grain segmentation in the output set.**

### 9c. The consolidated HDF5 — and its four mislabelled datasets

Written by `generate_consolidated_hdf5` (`consolidate.py:185`) on `PipelineH5`
(`state.py:71`); arrays gzip-4 (`state.py:33`).

```
/provenance                 attrs: created, last_opened, parameter_file (full text),
                                   one attr per MIDAS package version
/pipeline_state             attrs: workflow_type, command_line_args (JSON), start,
                                   last_update, current_stage
    completed_stages/<i>    stage-name strings, in completion order
    timestamps/per_stage    attrs: stage -> ISO timestamp
/parameters/…               SpaceGroupNr, LatticeConstant, GridSize, GlobalPosition,
                            GBAngle, NumPhases, PhaseNr, nSaves (whichever were found)
/voxels/position            (N,2) X, Y um
/voxels/euler_angles        (N,3) radians
/voxels/confidence          (N,)  0-1
/voxels/{orientation_row_nr, orientation_id, tri_edge_size, up_down}   <- ALL FOUR WRONG
/voxels/phase_nr            (N,)
/grains/{grain_id, mean_euler_angles, mean_position, mean_confidence, num_voxels}  <- NOT GRAINS
/grains/strain              empty group, attrs["status"] = "reserved"
/maps/orientation           (ySize, xSize, 7)
/maps/extent                [minX, minX+xSize, minY, minY+ySize]
/maps/{kam, grod, grain_id} (ySize, xSize)
/all_matches/data           the .AllMatches text, parsed
/grid/{points, num_points}
/multi_resolution/<label>/  attrs: grid_size, pass_type; then voxels/… maps/…
```

Resolution labels: `loop_0_unseeded`, `loop_<k>_{seeded,unseeded,merged}`
(`workflows.py:348-352, 424-428, 509-514, 534-540`). `/grains/`, `/all_matches/`, `/grid/`
are written **only for the root pass** (`consolidate.py:253, 276, 283`).

**`/raw_data_ref/` does not exist here.** `packages/midas_nf_pipeline/USAGE.md:197-204`
advertises it; grepping `packages/` finds it only in
`midas_ff_pipeline/midas_ff_pipeline/stages/consolidation.py:508`.

**The four mislabelled datasets** (`consolidate.py:238-250`, repeated at `:330-340`):

| dataset name | column written | what that column actually is |
|---|---|---|
| `tri_edge_size` | 0 | `OrientationRowNr` |
| `up_down` | 1 | `OrientationID` / winner count |
| `orientation_row_nr` | 2 | `RunTime` |
| `orientation_id` | 6 | `UpDown` |
| `run_time` | 12 | column 12 does not exist (text `.mic` has 0–11) — the write is skipped |

`position` (3:5), `euler_angles` (7:10), `confidence` (10), `phase_nr` (11) are correct.
The four above are **name-shifted, not value-corrupted**: `nf_qt.py:1651-1662` reads each
dataset back into the column index it came from, so the viewer round-trips
self-consistently. Any other consumer that trusts the names reads the wrong quantity.

**`/grains/` is not grains — re-verified 2026-07-29.** `aggregate_grains`
(`consolidate.py:153-176`):

```python
valid = mic_data[:, 10] > 0          # consolidate.py:157   confidence filter
data = mic_data[valid]
gids = np.unique(data[:, 6])         # consolidate.py:161   <-- column 6
gids = gids[gids >= 0]               # consolidate.py:162
...
    mask = data[:, 6] == gid         # consolidate.py:172
```

Column 6 of the text `.mic` is `UpDown` (`parse_mic.py:145-146`: `%OrientationRowNr
OrientationID RunTime X Y TriEdgeSize **UpDown** Eul1 Eul2 Eul3 Confidence PhaseNr`),
which takes values ±1 (`fitorientation/io.py:298`). After the `gids >= 0` filter, `/grains/`
gets **one "grain" per distinct non-negative `UpDown` value — in practice a single row
covering every upward-pointing voxel.** `/grains/mean_euler_angles` is then the mean Euler
angle of half the map. Established by reading; **not executed against a real H5** — check
the row count before quoting this. **Use `/maps/grain_id`, or run `mic2grains`.**

### 9d. Viewer — `gui/nf_qt.py`

```bash
cd <DATA_FOLDER>                    # required for BeamPos auto-detect (§3e)
python /Users/hsharma/opt/MIDAS/gui/nf_qt.py &     # --dark for dark theme (nf_qt.py:2169)
```

What it is for, in priority order for an agent:

- **Confirm the frame layout.** *First File* populates folder, stem, pad width and start
  frame from one chosen file (`nf_qt.py:1199-1220`). Its frame formula is flatter than the
  pipeline's: `fnr = start_frame + frame + dist * n_files_per_dist` (`nf_qt.py:1248`).
- **Tell distances apart.** Step *Distance* with *Frame* fixed; spot spread changes
  visibly. This is the practical discriminator when the log is missing.
- **Beam-edge measurement for calibration.** *Box H* / *Box V* — click two opposite
  corners; the right panel shows the integrated profile with the two threshold crossings,
  their centre and width (`nf_qt.py:1337-1361, 1490-1514`). The walkthrough is
  `manuals/NF_Calibration.md:82-100`.
- **Lab axes** (`A`) — overlays the MIDAS lab frame at the current distance's beam centre
  and warns if BC is still `(0,0)` (`nf_qt.py:1834-1861`). Convention, from tooltip/help
  (`nf_qt.py:386-391, 649-656`): `X_Lab` (= `Y_MIDAS`, red) display-left, `Y_Lab`
  (= `Z_MIDAS`, green) up, `Z_Lab` (= `X_MIDAS`, blue) into the page; η = 0 toward
  `Y_Lab`/`Z_MIDAS`; NF display origin bottom-right, FF bottom-left.
- **Overlay predicted spots.** *Load Grain* → *Make Spots* → *Select Point*
  (`nf_qt.py:2045-2116`, `1931-2015`). Shells out to the C `GetHKLList`,
  `GenSeedOrientationsFF2NFHEDM`, `SimulateDiffractionSpots` — **these must be built**, or
  it silently prints a failure and returns. Radius is scaled by `this_lsd / sim_lsd` with
  `this_lsd = Lsd + dist * dist_diff` (`nf_qt.py:1986-1992`) — hence the *distance
  difference* field.
- **Calc Median** shells out to the C `MedianImageLibTiff` (`$MIDAS_NF_BIN_DIR` or
  `~/opt/MIDAS/NF_HEDM/bin/`), one thread group per distance (`nf_qt.py:1865-1914`).
  *Max/Frames* and *Sum/Frames* read the `.bin` sidecars from §3f, not the TIFFs
  (`nf_qt.py:1256-1281`).
- **Load Mic / Load H5.** `.mic` uses `skip_header=4` and scatters; `.map` reads the
  4-double header then planes (`nf_qt.py:1518-1541`). *Load H5* enumerates
  `multi_resolution/*` into a *Resolution* combo, appending `⚠ slow` where a resolution has
  no rasterised map, defaulting to the highest `_seeded` loop that has maps
  (`nf_qt.py:1543-1610`).

**Two labelling traps in the viewer**, both inherited from §9a/§9c:

- Colour mode **`GrainID`** paints `.mic` column 0 / `.map` plane 4, which is
  `OrientationRowNr` — **not a grain label** (`nf_qt.py:1783-1784`).
- The mode that shows real grains is **`GrainMap`**, reading `maps/grain_id` or the
  `.map.grainId` sidecar (`nf_qt.py:1787-1799`). `KAM` and `GROD` come from `.map.kam` /
  `.map.grod` and are in **radians**.

Shortcuts: `←`/`→` frame, `L` log scale, `A` lab axes, `Q` quit, `Ctrl+scroll` frame
(`nf_qt.py:665-670`).

---

## 10. Parameter-file reference

One whitespace-delimited `Key Value [Value…]` per line; `#` comments; blanks skipped.

### 10a. Parser behaviour that differs between packages — know this before debugging

- **Repeated keys.** `midas_nf_pipeline.parse_parameters` keeps the **last** occurrence
  (`midas_nf_pipeline/params.py:51-100`); `collect_multiline()` gets *all*
  (`:103-127`). `midas_nf_fitorientation` and `diffr_spots` instead **accumulate** `Lsd`,
  `BC`, `OmegaRange`, `BoxSize`, `RingsToUse` into per-distance lists
  (`fitorientation/params.py:221-226`, `diffr_spots/params.py:75-109`). The pipeline's HKL
  stage deliberately uses the **last** `Lsd` line (`stages.py:30-40`).
- **Multi-value keys** get a fixed float count and raise if short: `LatticeParameter`(6),
  `GridMask`(4), `BC`(2), `OmegaRange`(2), `BoxSize`(4), `BCTol`(2), `GridPoints`(12),
  `GridRefactor`(3) (`midas_nf_pipeline/params.py:32-41`).
- **Everything else is stored as the first token only, as a string**
  (`midas_nf_pipeline/params.py:98-99`). `midas_nf_fitorientation` **silently skips
  malformed lines** (`fitorientation/params.py:343-346`).
- **Cheapest sanity check:** the fitorientation parser asserts `len(Lsd) == nDistances` and
  `len(BC) == nDistances` and raises otherwise (`fitorientation/params.py:352-361`).

Annotated reference file: `NF_HEDM/Example/ps_au.txt` (2 distances, `Lsd 8289.154576` /
`10290.724494`, `BC 985.415831 17.510494` / `985.161497 24.511210`, `px 1.48`,
`NrPixels 2048`, `OmegaStart 180`, `OmegaStep -0.25`, `StartNr 0`, `EndNr 1439`,
`NrFilesPerDistance 1440`).

### 10b. Material / crystallography

| Key | Values / units | Read by |
|---|---|---|
| `LatticeParameter` | `a b c α β γ` — Å, deg | pipeline (`params.py:33`); HKL gen (`stages.py:123-125`); fitorientation (`params.py:265`); diffr-spots (`params.py:91`); mic2grains (`mic2grains.py:65-68`) |
| `LatticeConstant` | alias | **only** fitorientation, diffr-spots and the H5 consolidator (`consolidate.py:130-134`). **Use `LatticeParameter`.** The pipeline's multi-value list contains `LatticeParameter` only (`params.py:33`), so writing `LatticeConstant` leaves `p["LatticeParameter"]` absent and the HKL stage raises `KeyError` (`stages.py:123`). Cost of using `LatticeParameter`: the consolidator greps for `LatticeConstant` only, so `/parameters/` carries no lattice. Cosmetic. |
| `Wavelength` | Å | HKL gen, diffr-spots, fitorientation |
| `SpaceGroup` | 1–230 | HKL gen, seed cache, `ParseMic`, mic2grains, diffr-spots, fitorientation |
| `SGNr` | fallback alias | pipeline stages only (`stages.py:122, 365`) |

### 10c. Detector geometry

| Key | Values / units | Read by |
|---|---|---|
| `nDistances` | count | pipeline, image processing loop, fitorientation (`params.py:186-190`) |
| `Lsd` | µm — **one line per distance** | fitorientation (list), diffr-spots (list), HKL gen (**last** line) |
| `BC` | `ybc zbc` px — one line per distance | fitorientation (`params.py:224-226`) |
| `tx` `ty` `tz` | deg — shared across distances | fitorientation |
| `Wedge` | deg | fitorientation |
| `px` | µm, square | diffr-spots, fitorientation |
| `NrPixels` | px — sets both Y and Z | image processing, fitorientation |
| `NrPixelsY` `NrPixelsZ` | px — override `NrPixels`; 0-fallback chain at `process_images/params.py:60-68` | image processing, fitorientation |
| `MaxRingRad` | µm | diffr-spots, fitorientation |
| `RhoD` | µm — **preferred** alias of `MaxRingRad` for the HKL stage only (`stages.py:43-49`) | HKL gen |

### 10d. Scan

| Key | Values / units | Read by |
|---|---|---|
| `OmegaStart` | deg — ω of the first frame. **See §2 for the sign.** | fitorientation |
| `OmegaStep` | deg/frame (negative = CW). **See §2.** | fitorientation |
| `OmegaRange` | `min max` deg — one line per distance | fitorientation (list), diffr-spots (list), pipeline |
| `StartNr` `EndNr` | frame numbers; `EndNr-StartNr+1` is what the fitter uses for frames/distance (`fitorientation/params.py:163-166`) | fitorientation |
| `NrFilesPerDistance` | count | image processing, pipeline, multi-layer offset |
| `WFImages` | wide-field frames per layer, **excluded** from `NrFilesPerDistance` (`process_images/io.py:31-33`) | image processing |
| `RawStartNr` | first raw file number; rewritten per sample layer | image processing, pipeline |

Arithmetic consistency check (derived from the example, **not enforced by code**):
`NrFilesPerDistance ≈ ω sweep / |OmegaStep|`. In `ps_au.txt` that is `1440 × 0.25 = 360°`.

### 10e. Sample geometry and I/O

| Key | Values / units | Read by |
|---|---|---|
| `Rsample` | µm — radius the hex grid must cover | hex grid (`hex_grid/params.py:19`) |
| `GridSize` | µm — voxel spacing. **Overwritten on disk each multi-resolution loop** (`workflows.py:373-376`) | hex grid; fitorientation (multipoint only) |
| `EdgeLength` | µm — triangle edge; `0`/absent ⇒ equals `GridSize` (`hex_grid/params.py:24-28`). **Leave it out.** See the warning below. | hex grid |

**`EdgeLength` must equal `GridSize` — just omit the key.** The two quantities are
independent in the code and only the *default* ties them together:

- the **lattice pitch** comes from `grid_size` alone —
  `x = xstart + grid_size*j/2`, `ht_triangle = √3·grid_size/2` (`hex_grid/grid.py:97,118,143`);
- `edge_length` sets only the triangle itself — `edge_half = edge_length/2` and the
  sub-triangle offsets `xt1 = edge_length·√3/6`, `xt2 = 2·edge_length·√3/6`
  (`grid.py:105-106,153`).

Set them inconsistently and the grid stops tiling: with `EdgeLength 1` and
`GridSize 10` you get **1 µm triangles on a 10 µm lattice**, i.e. ~1 % areal
coverage — a sparse point sampling, not a reconstruction of the volume. Grains
smaller than the pitch can fall entirely between probes, and every area that
`mic2grains` reports is wrong by `(GridSize/EdgeLength)²`.

**This bites hardest in multi-resolution runs.** `GridRefactor` rewrites
`GridSize` every loop (10 → 5 → 2.5), but a hardcoded `EdgeLength` does *not*
follow, so the mismatch grows as the run proceeds. Omitting the key makes the
edge track `GridSize` at every level automatically.

Check it on the generated grid rather than trusting the paramfile — column 5 of
`grid.txt` is `edge_half`, which must be **half of `GridSize`**:

```bash
head -2 grid.txt | tail -1 | awk '{print "edge_half =", $5}'   # GridSize 10 -> 5.0
```
| `GridFileName` | default `grid.txt` | hex grid, fitorientation |
| `GridMask` | 4 floats. The code filters grid columns 2 and 3, i.e. **x and y in µm** (`stages.py:211-227`). `ps_au.txt:89` labels them `ymin ymax zmin zmax`; **the code's meaning wins.** | pipeline `run_grid_mask` |
| `GlobalPosition` | µm — written into the `.mic` header | `ParseMic`, consolidator |
| `TomoImage` | path to a **square `uint8`** mask; side inferred from file size (`tomo_filter/filter.py:33-52`) | pipeline `run_tomo_filter` — **broken, §8a #3; use the CLI** |
| `TomoPixelSize` | µm per tomo pixel | as above |
| `DataDirectory` | path — raw TIFFs | everything |
| `OutputDirectory` | path — falls back to `DataDirectory` | everything |

### 10f. Image processing

All read by `midas_nf_preprocess.process_images` (`process_images/params.py:83-105`).

| Key | Units | Meaning |
|---|---|---|
| `BlanketSubtraction` | counts | flat offset subtracted **after** the temporal median, then clamped at 0 (`process_images/pipeline.py:165-166`) |
| `MedFiltRadius` | px | spatial median radius: `0` = identity, `1` = 3×3, `2` = 5×5 (`params.py:47`) |
| `GaussFiltRadius` | px | maps to the LoG `sigma` field — the *name* is `GaussFiltRadius`, the field is `sigma` |
| `LoGMaskRadius` | px | LoG kernel half-width |
| `DoLoGFilter` | 0/1 | `0` labels connected components of `img > 0` directly (`pipeline.py:180-195`). **Not a simple "always 1"** — LoG can suppress genuine weak peaks, so weak-signal samples are run with `0` and tolerate the cosmics. See the decision table in §5b. Changing it requires regenerating `SpotsInfo.bin`. |
| `OrigFileName` / `ReducedFileName` | stem | input / reduced stems |
| `extOrig` / `extReduced` | e.g. `tif` / `bin` | extensions |
| `WriteFinImage` | 0/1 | forced to 1 when `Deblur != 0` (`params.py:69-71`) |
| `Deblur`, `WriteLegacyBin` | 0/1 | |
| `SoftTemperature` | float or `auto` | **Python extension, not in the C** — sigmoid temperature for the differentiable spot-probability surrogate (`params.py:14-18`) |

### 10g. Orientation search

| Key | Values / units | Read by |
|---|---|---|
| `MinFracAccept` | 0–1 | phase-1 screen threshold; also a `MinConfidence` fallback in `mic2grains` (`mic2grains.py:80-83`). `ps_au.txt:124` suggests **0.1 seeded / 0.04 unseeded / 0.01 deformed** |
| `OrientTol` | deg | phase-2 search box per seed (`fit_orientation.py:363-365`). Default 1.0 |
| `ExcludePoleAngle` | deg | diffr-spots, fitorientation |
| `BoxSize` | 4 floats µm, relative to beam centre — one line per distance | diffr-spots (list), fitorientation (list) |
| `MinConfidence` | 0–1 | `mic2grains`; fitorientation; the multi-resolution bad-voxel filter (`workflows.py:439`) |
| `NrOrientations` | count | diffr-spots. **The pipeline overwrites it** from the seed-file line count (`stages.py:256-262`). Cubic-high cache = 243129 lines, matching `ps_au.txt:140` |
| `SeedOrientations` | path to the comma-separated `w,x,y,z` CSV (`seed_orientations/io.py:24-38`) | diffr-spots, pipeline |
| `SeedOrientationsAll` | path — the full unseeded library. **Required for multi-resolution** (`workflows.py:360-364`) | pipeline |
| `GrainsFile` | FF `Grains.csv`; rewritten per refinement loop | pipeline FF-seed stage |
| `SaveNSolutions` | count | fitorientation; `.AllMatches` record width; binary-merge record size (`workflows.py:165-166`) |
| `MinMisoNSaves` | deg | separation between saved solutions |
| `NearestMisorientation` | 0/1 | fitorientation |
| `RingsToUse` | ring number, **repeatable** | diffr-spots, fitorientation |
| `MaxAngle` | deg | `mic2grains` clustering tolerance; default 1.0 (`mic2grains.py:52`) |
| `GBAngle` | deg | `ParseMic` grain-boundary threshold for `.map.grainId`; default 5.0 (`parse_mic.py:56`) |

### 10h. Phase, output, calibration tolerances, multi-resolution, denoise

`NumPhases` (count, into the `.mic` header), `PhaseNr` (int, into `.mic` col 11 and `.map`
plane 5), `MicFileBinary` (filename), `MicFileText` (basename; `ParseMic` appends `.mic`,
`.map`, …).

Calibration tolerances, all read by `midas_nf_fitorientation`
(`fitorientation/params.py:276-289, 328-340`). Each becomes an
`x = x0 + tol*tanh(u)` box so the refined value cannot leave the box
(`packages/midas_nf_fitorientation/README.md:40-42`):

| Key | Units | Default |
|---|---|---|
| `LsdTol` | µm | 1000.0 |
| `LsdRelativeTol` | µm (between distances) | 100.0 |
| `BCTol` | `a b` px | 1.0, 1.0 |
| `TiltsTol` | deg | 0.05 |
| `NumIterations` | multi-start trials in `fit_multipoint` | 1 |
| `WedgeTol` | deg — only if `RefineWedge 1` | 0.05 |
| `RefineWedge` | 0/1 — **new, not in the C** | 0 |
| `TikhonovCalibration` | λ; 0 disables — **new** | 0.0 |
| `TikhonovSigmaLsd` / `SigmaTilts` / `SigmaBC` / `SigmaWedge` | µm / deg / px / deg | 100.0 / 0.05 / 1.0 / 0.05 |
| `GaussianSplatSigmaPx` | px — override the auto soft-overlap σ — **new** | auto |
| `GridPoints` | 12 values; the fitter reads fields 4,5,7,8,9,10 as `xc yc ud eul1 eul2 eul3` (`params.py:306-316`) | — |

Without a `GridPoints` block, `fit_multipoint_run` derives its voxel set from the
reconstructed `MicFileText` `.mic` — highest-confidence voxels above `MinConfidence`
(`packages/midas_nf_fitorientation/notebooks/README.md:26-31`). Note `ps_au.txt:174-176`
uses tighter values than the defaults: `LsdTol 500`, `LsdRelativeTol 3`, `BCTol 2 0.2`.

`GridRefactor StartingGridSize ScalingFactor NumLoops` — µm, ×, count; absent ⇒ single
resolution (`workflows.py:253-261`).

Denoise (optional step 0, `stages.run_denoise`, `stages.py:56-111`) requires the separate
`MIDAS-NF-preProc` package. `DenoiseMethod n2v` **raises** without a CUDA GPU
(`stages.py:66-74`). On success it rewrites `DataDirectory` in memory **and on disk, by
appending a line to your parameter file** (`stages.py:108-111`). Keys: `Denoise` (0),
`DenoiseMethod` (`nlm`|`n2v`, default `nlm`), `DenoisedDirectory`
(`<DataDirectory>/denoised`), `DenoiseConfigFile`, `DenoiseCheckpoint`, `DenoisePattern`
(`*.tif`), `DenoiseTrainJointly` (0), `DenoiseFinetune` (0), `DenoiseMaskThreshold`
(unset ⇒ `None`), `DenoiseNoMedian` (0 — 1 disables the temporal median).

### 10i. Keys in `ps_au.txt` that no Python NF module reads

Verified by grepping every `.py` under `packages/midas_nf_*` and `packages/midas_hkls`:
**`OnlySpotsInfo`, `WriteImage`, `LayerThickness`, `GlobalPositionFirstLayer`** are never
read. `PrecomputedSpotsInfo` (added by the fitorientation integration test's patched
paramfile, `tests/integration/test_vs_c_fit_orientation.py:119`) is likewise unread.
`Ice9Input` is explicitly **deprecated and silently ignored**
(`fitorientation/params.py:319-322`). Leaving them in is harmless; expecting them to do
anything is not.

---

## 11. Validation status — put every number you report in one of these buckets

"Byte-parity port" in a docstring is an *intent*. The test file is the evidence.

### Has a real parity test against a C reference

| Component | Test | Gate |
|---|---|---|
| `parse_mic` — text `.mic` | `midas_nf_pipeline/tests/test_parse_mic.py:98-122` | header lines byte-equal; data tokens `abs=1e-6` vs `Au_txt_Reconstructed.mic` |
| `parse_mic` — `.map` | `test_parse_mic.py:137-151` | planes 0–5 **exactly equal**; plane 6 `rtol=atol=1e-12` |
| `parse_mic` — `.map.grainId` | `test_parse_mic.py:154-164` | **exactly equal** |
| `parse_mic` — `.map.kam`, `.map.grod` | `test_parse_mic.py:167-185` | `atol=rtol=1e-10` (radians) |
| `mic2grains` | `tests/test_mic2grains.py` — invokes the **live C `Mic2GrainsList`** at test time | grain **count** equal; header lines equal; each Python grain matches some C grain to **< 0.1°**. Position and radius parity **explicitly not asserted** (`test_mic2grains.py:152-156`) |
| `fit_orientation` | `midas_nf_fitorientation/tests/integration/test_vs_c_fit_orientation.py` — runs C `simulateNF` + `nf_MIDAS.py` + `FitOrientationOMP`, then re-fits in Python | **median misorientation < 0.5°** and **≥ 90 % of voxels < 0.5°**, on a **30-voxel stratified sample** of the bundled Au example (`:66-68`, `:396-409`) |

The `parse_mic` reference input is a **frozen** copy,
`NF_HEDM/Example/sim/Au_bin_Reconstructed.mic.c_ref`, because the live binary is
overwritten by every Python fit (`test_parse_mic.py:39-48`). Both integration tests **skip**
unless `MIDAS_RUN_INTEGRATION=1`; `test_mic2grains` also skips if the C binary is not built.

### Does NOT have a parity test against C

- **`midas_nf_fitorientation` end to end.** Its own README: *"the forward path is validated
  against `midas-diffract` (pixel-exact vs. the C simulators); the fit drivers have
  unit-test coverage at the module level. **End-to-end agreement against the C
  `MicFileBinary` on a real reconstruction dataset is the next milestone.**"*
  (`packages/midas_nf_fitorientation/README.md:137-142`). The 30-voxel stratified test is a
  *sample* on synthetic Au, not a dataset-level result. No C-parity tests among the 13
  module-level test files.
- **Everything in `midas_nf_preprocess`.** There is **no** `tests/integration/` directory in
  that package; tests are unit tests over synthetic data
  (`packages/midas_nf_preprocess/notebooks/README.md`: *"All notebooks run on CPU with
  synthetic data"*). So `hex_grid`, `diffr_spots`, `process_images`, `seed_orientations` and
  `tomo_filter` — **including `SpotsInfo.bin` itself** — carry **no byte-parity evidence
  against the C**. Their docstrings cite C line numbers: that is provenance, not
  verification.
- **`midas_nf_pipeline` end to end.**
  `tests/integration/test_au_end_to_end.py` is gated on `MIDAS_RUN_INTEGRATION=1` and asserts
  only that the consolidated H5 *contains* `voxels/position` and either `grains/grain_id` or
  `multi_resolution/loop_0_unseeded` (`:61-66`). No numerical comparison, and given §8a it
  cannot currently pass.

### Deliberate, documented departures from the C — do not report these as parity failures

(`packages/midas_nf_fitorientation/README.md:21-57`)

- Orientation optimiser: NLopt Nelder-Mead → vectorised PyTorch NM over all
  `(voxel × winner)` problems at once.
- Calibration optimiser: NLopt NM → L-BFGS over a **soft Gaussian-splat surrogate** with
  tanh-boxed bounds. This optimises a slightly smoothed objective, which is the stated
  reason C-vs-Python misorientation sits near 0.12° rather than at zero
  (`test_vs_c_fit_orientation.py:20-32`).
- Multi-start replaces the C's NM→CRS2→NM ladder; *"CRS2's true global behaviour is lost"*.
- `mic2grains` uses a stable sort where the C used `qsort`, so equal-confidence voxels can
  seed a grain differently — which is why the test asserts orientation but not position
  (`mic2grains.py:348-353`).
- `parse_mic` deliberately **reproduces a C macro bug** so the pixel→voxel assignment stays
  byte-identical: `CalcNorm2` is unparenthesised in the C, so
  `CalcNorm2(X, intX+j, Y, intY+k)` expands to `sqrt((X-intX+j)² + (Y-intY+k)²)`
  (`parse_mic.py:230-242`). "Fixing" it breaks the `.map` parity test by design.
- `midas_hkls`' NF writer is functionally identical but has **deterministic intra-ring row
  order** where the C's `qsort` was unstable (`midas_hkls/nf_hkls.py:16-21`).
- Dropped outright (`packages/midas_nf_pipeline/README.md:151-162`): Parsl multi-node
  dispatch, per-machine config modules, the `MMapImageInfo` fallback, the `FitOrientationGPU`
  C binary.

### Could not verify — do not upgrade these

1. **Rotation-stage names other than `aero`.** Only `aero` was observed (441397/441397 rows
   in `pokharel_jun25`). What any other value implies for the ω sign is **unknown**. Stop
   and ask.
2. **`*_nf_*` folder naming** is convention; nothing in this repo parses it.
3. **The five §8a defects were established by reading, not by execution.** Exception
   *types* follow from Python semantics; exception *text* is not quoted because it was not
   observed.
4. **`/grains/` being a single row** was established by reading `aggregate_grains`, not
   against a real H5. Check the row count before quoting it.
5. **The `.mic` header `%TriEdgeSize 0.000000` degradation path** was traced in code
   (`mic2grains.py:365-373, 294`); the downstream effect was not measured on a real run.
6. **Whether the C `ProcessImagesCombined` behaved differently when invoked per distance**
   — the C was not read. The `--all-layers` rule (hard rule 6) is established from the
   Python (`process_images/pipeline.py:229-243`, `cli.py:57-60`) only.
7. **`FileCount.txt` fields 13, 14 and 18–23** were not identified. f13 = 0.02, f14 = 735.42,
   f18 ≈ image count offset, f22/f23 = 721 for Au4. Only f10, f11, f12, f15, f16, f17 are
   established.
8. **`fastsweep_Emon.txt` fields other than f10** were not identified.
9. **`NF.par` fields other than f6–f11, f17 and f29** were not identified.
10. **Absolute `Lsd` for `pokharel_jun25`** is not established in this document — §6 is a
    placeholder.
11. **Which of geometry A / B is physically correct** (§7e/§7f). A was adopted on operator
    judgement after B scored slightly worse on the map; the two orientations agree to
    0.04°, so this is a **preference, not a measurement**. Do not report A as "the
    verified geometry".
12. **Why `LsdRelativeTol 5` stalled at confidence 0.27** while `LsdRelativeTol 1`
    succeeded on the same data (§7b). Observed once, not diagnosed.
13. **Whether dropping the redundant 3-vertex axis in `screen()`'s centroid branch is
    safe** with non-zero tilts. Affine commutation breaks; the error bound was never
    measured. Do not implement on the assumption that it is safe.
14. **Whether the `.mic` row shortfall is exactly the `Confidence == 0` rows.** A 5046-voxel
    grid produced 5012 text rows; the drop rule is documented (§9a) but the specific
    count was not reconciled against the writer.

### Verified in this tree — safe to rely on

1. **`BoxSize` semantics and effect** (§7d): 0.949153 → 1.000000 on one Au voxel, matching
   the C reference exactly; Triton fused kernel agrees with eager in both states.
2. **`screen()` dtype rework is answer-preserving** (§8f): every field bit-identical across
   a 5046-voxel grid, `RunTime` the only difference.
3. **`screen()` results are independent of `MIDAS_NF_SCREEN_VOXEL_CHUNK`** (§8g), checked at
   a fixed forced chunk size.
4. **The three calibration negatives** (§7b) — plateau, multipoint, iteration ratchet — were
   each observed directly on `pokharel_jul26` Au5, not inferred.

### Bottom line

- **Trustworthy, with a C reference behind it:** `.mic` → `.map`/`.kam`/`.grainId`/`.grod`
  rasterisation; the voxel→grain clustering **count**.
- **Trustworthy to ~0.1–0.5° on synthetic Au, sampled:** per-voxel orientations.
- **Unverified against C:** the image-reduction chain that produces `SpotsInfo.bin`, the
  candidate-spot simulation, the grid generation, the pipeline orchestration.
- **Known wrong:** the `/grains/` group and four dataset names in the consolidated H5
  (§9c); the five orchestrator call sites (§8a).

**Say which bucket each number you report falls into.** Every quantitative claim must name
the file and the command that produced it.
