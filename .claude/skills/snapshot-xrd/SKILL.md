---
name: snapshot-xrd
description: >-
  Take a still-frame diffraction series (monochromatic beam, area detector, sample NOT rotated;
  one frame or a time series) from raw frames to per-frame matrix-scale and halo traces, a
  calibrated sparse-spot population, features linked across frames and tested on raw photons,
  a measured detection limit, and candidate-cell tests with nulls and negative controls, then a
  report with provenance. Use when asked to process, reduce or diagnose in situ / operando /
  single-shot / snapshot diffraction without rotation -- melting and solidification, heating or
  cooling ramps, compression shots, operando cells -- when handed a folder of per-frame
  detector images with spotty or sparse rings where radial lineouts lose the signal, when asked
  whether a minor phase is present, or when an in situ phase call rests on one or two peaks.
  Scope of what has been measured: per-frame TIFF images from two in situ laser-melting series
  only; the other cases listed (ramps, compression shots, operando cells) are untested, so read
  ENVELOPE.md first and measure the injection curve on every dataset.
  Rotation series redirect to ff-hedm / nf-hedm / pf-hedm; smooth powder rings to
  calibrate-integrate or xrd-ct; polychromatic data to laue; diffuse scattering to defect.
---

# Snapshot diffraction (still frames, no rotation)

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository so
it lives beside the `midas_snapshot` code it cites and stays usable without this skill.

## Start here

Read **`manuals/snapshot-xrd/README.md`** -- the spine: scope gate, install gate, order of
operations, hard rules, halt conditions, and an index of the other files.

Then give, or work out from the data:

```
Frames:          <ABSOLUTE PATH>   # per-frame images of the series
Calibrant:       <ABSOLUTE PATH>   # same geometry, or "find it"
Matrix phase:    <CIF>             # the dominant known phase (optional but recommended)
Candidates:      <CIF ...>         # phases to test for, all from ONE source under ONE rule
Goal:            traces | minor-phase search | both
```

## Things to know before you start

1. **A single peak identifies nothing.** Under a free thermal scale one line matches almost any
   candidate. The test here needs a population of distinct features on several lines, beating
   a null, with negative controls that fail.
2. **A pass names a cell and a pattern, not a chemistry.** Structures of one type differ only in
   cell size; report the cell.
3. **The detection limit is measured per dataset** by injecting synthetic spots into real frames.
   A search without it cannot say "absent".
4. **Repeated detections are one feature.** Counting them separately makes noise significant.
5. **"Absent after" must hold on raw photons**, not on the absence of a thresholded detection.
6. **Temperature is relative.** The matrix scale mixes thermal and uniform elastic dilatation.
7. **Choose windows from traces only**, before looking at spots, by the stated rule.
