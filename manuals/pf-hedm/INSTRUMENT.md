# pf-HEDM instrument reference — the two configurations, and the two code generations

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Sibling: `manuals/ff-hedm/README.md` §Scope, which carries the same split for far-field.

Every recipe in this doc set has been run on one of **two** scanning configurations. Where
they diverge it is called out inline as **"20-ID:"**. If your data is neither, **stop and
ask** rather than adapting — the ω sign, the dark handling and the entry point all differ.

## I1. The two configurations

| | **1-ID scanning** | **20-ID HT-HEDM Varex** |
|---|---|---|
| files | DM-converted `.h5`, or a per-frame `.tif`/`.tif.bz2` series | `.vrx.h5` |
| detector | Pilatus 1475 × 1679 @ 172 µm (also GE) | Varex 2880² @ 150 µm |
| `Lsd` | ~731 mm | ~895.4 mm |
| energy / λ | per run | 63.000 keV, λ 0.19680 Å |
| ω | 1440 frames × 0.25° = 360° | 1442 raw frames → **1441 processed**, 0.25° |
| ω sign | negate the `aero` field (see §I2) | `OmegaStart 180`, `OmegaStep -0.25` — **already negated in the param file** |
| frame 0 | throwaway, `SkipFrame 1` | same |
| `ImTransOpt` | establish per detector | **2** (flip-Z) |
| dark | a `Dark` file, `darkLoc` | **`darkLoc /exchange/bright`** — `/exchange/dark` exists and is all zeros |
| beam / scan step | 1.5 µm / 1.5 µm | 1 µm / 1 µm |
| reference campaigns | notebook §1–§6 (cracked AM FCC-Ni, 259 × 259); NMC811 `bt_1id_jun25b` (phase 2 §2.5) | notebook §7 (`nf709` set A, 51 × 51) |

**20-ID: the all-zero dark in the zarr is cosmetic, not a fault.** `exchange/dark` reads
all zeros while the data really is dark-subtracted at zip time (raw frame mean ~1850 →
zarr ~0.6). Do not chase it as a halt condition. This is shared with the far-field set.

## I2. The ω sign on each

A flipped ω mirrors the whole map and reflects every orientation, and nothing downstream
complains (spine halt condition; phase 1.1). Confirm it against the encoder, not the
parameter file.

**20-ID, worked through on the reference campaign.** The `aero` encoder runs
**−180.17 → +179.93 at +0.242°/frame**. The aero stage turns *against* the MIDAS
convention, so `ω_MIDAS = −ω_aero`, which gives **`OmegaStart 180` / `OmegaStep −0.25`** —
matching the parameter file. Frame 0 is the throwaway, so `SkipFrame 1`.

That reconciliation is the pattern to repeat, not a value to copy: read the encoder, decide
whether the stage turns with or against the convention, and check the param file agrees.

## I3. ⚠ Two code generations — check which one you are on

This matters more than any parameter. The maintained tree is **`midas_pipeline`**; a
beamline install may still be the **legacy v11 C `pf_MIDAS.py`** under a station's own
`MIDAS` checkout. The reference campaign in notebook §7 ran entirely against the legacy C
path — *because that is what was installed at the beamline*, not because it is current.

```bash
python -c "import midas_pipeline as m; print(m.__version__)"
```

| | legacy `pf_MIDAS.py` (v11 C) | `midas_pipeline` |
|---|---|---|
| invocation | `pf_MIDAS.py -resultDir … -paramFile … -nCPUs 64 -preProcThresh 60 -doTomo 0 -numFrameChunks 100` | `midas-pipeline run …` (phase 3) |
| sinogram code | `FF_HEDM/src/findSingleSolutionPFRefactored.c` | `midas_pipeline/find_grains/` — a native replacement, `__init__.py` names the C ranges it replaces |
| concentration filter, occupancy flag | **absent** | shipped (phase 6 §6.5, §6.6) |
| spot positions | **`spotPositions_*.bin`, 97.7 % unwritten** — phase 6 §6.9 | `spotPos_*.bin`, correct |
| FBP crop registration | — | fixed in **0.11.0**; below it every shape is one voxel low in both axes (phase 6 §6.2) |

**Version floors.** `midas_pipeline ≥ 0.11.0` for the crop fix; **≥ 0.12.0** in practice,
which also added "a run that finds nothing must not exit 0". The `find_grains` migration is
covered in phase 6 §6.9 — including the rule **not** to patch the C, because the Python
path is already right.

**Known cosmetic bug, 0.14.0 only:** `midas-pipeline run --help` raises
`ValueError: unsupported format character` — a literal `%` in one option's help text breaks
argparse's formatting for the whole subcommand. Fixed in 0.14.1
(`89e6589a`). It does not affect running anything.

## I4. `preProcThresh` sits on the *dark-subtracted* pedestal — check which way it leans

`-preProcThresh` is a threshold above the dark-subtracted pedestal, so its effective value
depends on where the dark sits relative to the data.

**20-ID, measured:** the dark frame pedestal sits about **70 counts ABOVE** the data
pedestal, so `preProcThresh 60` is effectively **~130 above the real data floor**. That is
the working value for the reference campaign — but if a new dataset returns far fewer
spots than expected, this is the first number to look at, and the sign of that 70 is the
thing to re-measure rather than assume.

## I5. Two parameters that are not what they look like

- **`RhoD` is in µm, and overflowing it costs you every grain.** A value of `2000000`
  overflowed the indexer's 500-ring array and produced **zero seeds** on this beamtime;
  the correct value was **309537.68 µm**. A `RhoD` copied out of a GUI-written seed file
  may be in *pixels*. (Shared with far-field, where it is a numbered hard rule.)
- **`Hbeam` is a search bound, not the beam.** On the reference campaign `Hbeam 2000` was
  20× the slit height and was left as-is deliberately, to match the neighbouring scan set.
  That is correct: tightening it to the true dimension plops solutions onto the bounding
  box (spine hard rule 5). In PF it is doubly moot — PF does not fit position at all.

## I6. The reference-campaign parameter values

For orientation only. **Do not copy these into a new run** — take the geometry from a
calibrant on the same detector position (phase 1.3).

```
# 20-ID Varex, nf709 set A (notebook §7)
Lsd            895388.6 um          # FF calibration on the matching CeO2 set
BC             1450.856  1467.372   # px
px             150 um
Wavelength     0.19680026 A
LatticeConstant 3.599 A  (FCC)
OmegaStart     180      OmegaStep  -0.25    (1440 steps, 180 -> -180)
SkipFrame      1
ImTransOpt     2
n_scans        51       scan step 1 um      beam 1 um
preProcThresh  60       (see I4)
Hbeam          2000     (a bound, see I5)
```

Run cost, as a scaling anchor: **51 × 51 = 2601 voxels in 9305 s (2.6 h) on 64 CPUs**,
legacy C path, `-doTomo 0`, `-numFrameChunks 100`. Scale roughly with voxel count.
`-numFrameChunks` controls memory only — raise it if the job is killed.
