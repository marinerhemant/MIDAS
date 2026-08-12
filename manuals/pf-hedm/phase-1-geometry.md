# Phase 1 — Geometry: the ω sign, the position convention, the calibration

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

Two conventions here are **halt conditions**: they cannot be checked after the fact because
getting them wrong mirrors the map without changing anything you would look at. Fix them
first, then calibrate.

## 1.1 The ω sign — confirm against the encoder, not the param file

The `paramstest` carries `OmegaStart` / `OmegaStep`. **Do not trust the sign blindly** — the
scan encoder is ground truth. From the raw file (phase 0):

```python
aero = f["measurement/instrument/SMS/aero"][:]      # actual ω per frame
print("ω start", aero[0], "end", aero[-1], "median step", np.median(np.diff(aero)))
```

Reconcile with the site convention (some stations negate the logged step; a throwaway first
frame is dropped every acquisition on some instruments). The rule you must end with:

- The frame→ω mapping used by the forward model must match `aero`. Confirm it by checking
  that a **bright, well-determined reflection lands on its predicted frame** (phase 4's
  anchor check does exactly this for the strain path; do a lightweight version here if in
  doubt).

A flipped ω reflects every orientation and mirrors the map. Nothing downstream complains.

## 1.2 The `positions.csv` convention — FILE ORDER, and the sign

`positions.csv` is the 1-D list of beam translation positions (µm), **one per scan file**.
Two traps:

1. **It is in FILE ORDER, not sorted.** The scanning indexer and the refiner index it by
   scan-file number. A common convention is *descending*: file 1 at `+halfspan`, the centre
   file at `0`, the last file at `−halfspan` — i.e. `position[file n] = centre − n·step`.
   The reconstruction sorts positions internally for the voxel grid, but the **per-scan σ
   index stays file-order**. If you feed a sorted list where file-order is expected, or flip
   the sign, the voxel map mirrors about a diagonal — a plausible-looking microstructure.
2. **The voxel grid is the Cartesian product** of the sorted positions with themselves:
   voxel `v = i·n_scans + j` sits at `(x,y) = (pos_sorted[i], pos_sorted[j])`. Voxel 0 is a
   grid corner (often outside the sample), not the centre.

Verify by reconstructing a **known feature** — a sample edge, a notch, a fiducial — and
checking it lands where the microscope/tomo says it should. If it is mirrored, flip the
order or sign and re-run from binning (cheap) — do **not** "fix" it downstream.

> If a byte-exact original `positions.csv` is unavailable, it can be rebuilt from the scan
> spec (n_scans, centre-file index, step, direction). Cross-check the rebuilt file's byte
> length against the original if you have it; an off-by-one in the centre index shifts the
> whole grid by one voxel.

## 1.3 Calibrate the detector geometry

pf-HEDM geometry is calibration-level and **reused across layers** of the same mount — you
usually do **not** re-calibrate per layer. If a calibrant scan exists, calibrate as in the
far-field doc set (`manuals/ff-hedm/phase-1-geometry.md`): a powder standard (CeO₂, LaB6)
gives `Lsd`, beam centre `BC`, tilts `ty/tz`, and the distortion coefficients.

Record, with provenance, into the runbook:

| quantity | source |
|---|---|
| Energy / wavelength | monochromator / beamline confirmation |
| `Lsd` (sample-detector) | calibrant fit — **not** the `DetZ` readback |
| `BC` (beam centre, px) | calibrant fit |
| tilts `ty`, `tz` | calibrant fit, 0/180 spread |
| distortion `p0…` | calibrant fit |
| pixel size, detector size | detector spec |
| `Hbeam` / beam height | **the true per-layer beam**, never the sample size (hard rule) |

> **A powder calibrant cannot constrain `tx`** (rotation about the beam) — hold it fixed
> during powder calibration, refine it from grains after. (Shared with FF; envelope §1.)

## 1.4 Match the layer to the far-field seed

You will need a far-field orientation seed (phase 2). The FF measurement is usually a stack
of layers at different heights. Match **this pf layer's `samY`** (phase 0) to the FF layer
list by height; the closest FF layer supplies the seed. The FF grain **positions** may be
unreliable, but the **orientations** are what seed indexing, and those transfer even when
the sample was repositioned in-plane between the FF and PF scans.

When the ω sign, the position convention, and the geometry are all recorded and confirmed,
go to [`phase-2-configure.md`](phase-2-configure.md).
