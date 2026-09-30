# Snapshot diffraction -- measurement envelope

**Technique:** monochromatic still frames, sample stationary, area detector
**Last checked:** 2026-09 with `midas_snapshot` 0.1.0 · **Owner:** MIDAS maintainers

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

What the measurement can and cannot determine. Numbers below were measured on two in situ
laser-melting series of different alloys (100 keV, 172 um hybrid-pixel detector, 250 Hz / 3 ms
frames; detector distance about 0.9 m for series A and about 0.6 m for series B) with this
configuration (`margin_px` 21 and `fa_per_image` 0.05 are defaults; `tth_max` 12 deg was set explicitly). They are examples of what
to expect, not guarantees: **measure the injection curve on every dataset.**

---

## 1. Detection limit (image-level injection into real frames)

Synthetic spots on a cubic candidate pattern, placed only where detection is allowed, pushed
through the identical pipeline (`analysis.json` -> `injection`).

| Series | Frames | Recovered at 20 / 40 / 80 counts per frame | Called off-matrix at 40 / 80 |
|---|---|---|---|
| A | single | 0.77 / 0.98 / 1.00 | 0.72 / 0.81 |
| B | single | 0.62 / 0.99 / 1.00 | 0.80 / 0.87 |
| A | 25-frame sums | 0.83 at 20, 0.00 at 10 counts per frame | 0.84 at 40 |
| B | 25-frame sums | 0.88 already at 5 counts per frame | 0.82 at 5 |

- Single frames: a candidate spot of about 40 counts per frame is found almost always.
- Summing frames lowers the limit on series B about eightfold but not on series A. Unexplained;
  treat the summed-window limit as dataset-specific until measured.
- "Called off-matrix" plateaus at 0.8-0.9 of recovered spots: that fraction of the candidate
  pattern falls inside matrix bands and cannot be separated from the matrix by position.

## 2. Coverage

Only part of a candidate's rings lies where detection is allowed (valid pixels, margin from gaps,
below `tth_max`): **55 % (series A) and 29 % (series B)** of the pattern's ring pixels with this
configuration. A shorter distance puts more of the pattern beyond `tth_max`; a larger margin removes
more near module gaps. Report coverage next to every "absent".

## 3. Matrix scale and classification

- Matrix line width (robust sigma of the relative d-residual) 0.12-0.32 % across 16 series-window
  combinations; the off-matrix threshold is 6 sigma.
- Pre-event matrix lattice parameters from one geometry agree to +-0.2 % across sample positions.
  Absolute values carry the geometry uncertainty.

## 4. What cannot be determined

- **Absolute temperature.** The matrix scale mixes thermal and uniform elastic dilatation.
- **A strain tensor from one still.** One spot gives one projection.
- **A phase from one line.** Under a free scale a single line matches most candidates.
- **A chemistry from a cell.** Structures of one type differ only in cell size (partly absorbed by
  the scale window); report the cell family.
- **Whether a vanished population dissolved or left the beam.** Diffraction alone does not say.

## 5. Candidate-test sensitivity

A pass needs a population: >= 5 features on >= 2 (preferably >= 3) distinct lines, beating a null
that has the detector's coverage, with negative controls failing. Sparse minor phases concentrate
at a few sample positions; pool series and test leave-one-out. Single-feature coincidences are
leads.

## 6. Throughput

Per-window pass on one 96-core CPU node under shared load: 5000 single frames (1475 x 1679 px) in
about 200 s; 2000 frames in 90-120 s; 25-frame sums in 35-50 s per series.
