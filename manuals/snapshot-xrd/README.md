# Snapshot diffraction (still frames) -- spine

**Technique:** monochromatic diffraction on an area detector, sample stationary, one frame or a
time series. **Package:** `midas_snapshot` (primitives in `midas_peakfit`, `midas_integrate_v2`,
`midas_hkls`). **Owner:** MIDAS maintainers.

This file is the only one meant to stay loaded. It says whether the recipes apply, in what order
to run them, which rules are hard, and when to stop.

| File | Holds |
|---|---|
| [`ENVELOPE.md`](ENVELOPE.md) | what the measurement can and cannot determine, with measured limits |
| [`RUNBOOK.md`](RUNBOOK.md) | commands, configuration keys, outputs |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptoms -> causes -> fixes (every entry cost a wrong result once) |
| [`phase-0-survey.md`](phase-0-survey.md) | look at raw frames, traces, calibrant |
| [`phase-1-geometry.md`](phase-1-geometry.md) | orientation, calibration, the checks that must pass |
| [`phase-2-windows.md`](phase-2-windows.md) | time windows from traces, fixed before spots are opened |
| [`phase-3-spots-and-controls.md`](phase-3-spots-and-controls.md) | detection, classification, features, injection |
| [`phase-4-phase-test.md`](phase-4-phase-test.md) | candidate-cell test, nulls, negative controls, cell scan |
| [`phase-5-traces.md`](phase-5-traces.md) | matrix scale (relative thermometer), halo |
| [`phase-6-report.md`](phase-6-report.md) | what to state, how to word it, provenance |
| [`profiles/melt-solidify.md`](profiles/melt-solidify.md) | laser melting / solidification series |

## Scope gate

In scope: monochromatic beam; the sample does not rotate during a frame series; an area detector;
patterns that are spotty or sparse (few crystallites in the beam), possibly with a diffuse halo.

Redirect: rotation series -> `ff-hedm`, `nf-hedm`, `pf-hedm`; smooth rings where lineouts work ->
`calibrate-integrate`, `xrd-ct`; polychromatic -> `laue`; diffuse scattering -> `defect`.

## Install gate

`python -c "import midas_snapshot, midas_peakfit.snapshot_detect, midas_hkls.feature_phase,
midas_integrate_v2.streaming.snapshot_profile"` must succeed. On a shared filesystem, run from a
directory that does not contain the package source folders (a source folder named like the package
shadows the installed package).

## Order of operations

0. **Survey** raw frames: counts per frame, a few frames at key times, a max projection. Look before
   modelling.
1. **Geometry** from a calibrant at the same distance; confirm orientation with a radial profile
   (not with overlay contours).
2. **Setup and traces.** If the distance or alloy is uncertain, `midas-snapshot select` picks the
   (geometry, matrix) pair from the data. `midas-snapshot run`: matrix scale, halo, spot count per
   window. Decide the series kind: event, static, or map.
3. **Windows** from the traces only, by the stated rule; write them down before opening spots.
4. **Preregister** the test: candidates (one source, one rule), windows, thresholds, pass criteria.
5. **Analyse** (`midas-snapshot analyse --candidates ... --controls`): features, raw-photon test,
   candidate test with negative controls, injection curve.
6. **Report** with provenance; positive phase calls stay provisional until independently checked.

## Hard rules

- No phase call from one line. No chemistry from a cell.
- Negative controls must fail, or the run is unreadable.
- A best scale on the edge of its window is flagged and reported as such.
- "Absent" is always "absent above X counts per frame", with X from the injection curve.
- Merge repeats before any statistic. Test before/after on raw photons.
- Candidate cells from one kind of source under one selection rule, recorded.
- Temperature from the matrix scale is relative and confounded with uniform stress.

## Halt conditions

- The calibrant profile does not line up with the predicted lines in any orientation.
- The matrix scale cannot be fitted on the frames meant to be solid.
- A negative control passes (the statistic is contaminated: look for recurring artefacts).
- The injection curve never reaches 80 % recovery at any tested intensity.
