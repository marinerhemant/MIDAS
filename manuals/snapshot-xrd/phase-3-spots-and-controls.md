# Phase 3 -- spots, features, controls

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

## Detection (`midas_peakfit.snapshot_detect`)
- Local background (masked box mean after capping bright pixels), not a per-ring mean.
- Matched filter (`gauss`) or aperture counts (`poisson`) scored against that background.
- Threshold from pure-Poisson null images of the frame's own background: `fa_per_image` false
  peaks per image. No absolute count floor.
- Margin from invalid pixels; optional maximum angle to skip low-count corners.

## Classification
- Every spot gets its d-spacing and signed distance to the nearest matrix line at the window's
  fitted scale. The matrix width `sigma_matrix` is measured from the core of that distribution.
- Off-matrix: beyond `nsig_off * sigma_matrix`. Part of any candidate pattern is hidden inside the
  matrix bands; the injection curve measures how much.

## Features (`midas_peakfit.tracks`)
- Merge detections within `merge_px` into features (lifetime, detections, medians); keep features
  with at least `min_det` detections. Repeats are one feature.
- Raw-photon test between windows: aperture counts against the local background; "present before"
  and "absent after" are Poisson tests, not the absence of a detection.
- Check features against unrelated data (other positions, background runs): a feature at the same
  pixel there is a detector artefact.

## Detection limit (image-level injection)
- Synthetic spots on a candidate pattern are injected into real frames of the analysed window and
  pushed through the identical pass. The curve (recovered and called off-matrix vs counts per
  frame) is the dataset's detection limit. Quote "absent" only above the intensity where the
  off-matrix rate reaches its plateau (>= 80 % of the plateau).
