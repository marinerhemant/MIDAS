# Snapshot diffraction -- runbook

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

## Commands

```bash
midas-snapshot init --frames FRAMES_DIR --geometry params.txt --out OUT \
    --flip ud --mask mask.tif --matrix-cif matrix.cif \
    --halo-band 2.9 3.2 --base-band 1.9 2.2 --tth-max 12
# edit OUT/snapshot_config.json if needed (nproc, window_sizes, windows rule, detector)
midas-snapshot run OUT/snapshot_config.json                     # per-window pass, each window size
midas-snapshot analyse OUT/snapshot_config.json --candidates a.cif b.cif --controls
midas-snapshot report OUT/snapshot_config.json
```

### Three kinds of series

| Kind | Frames are | Command | What changes |
|---|---|---|---|
| event | one position, an event (melt, shot, ramp) in time | `analyse` | before/after windows from the halo trace; raw-photon test |
| static | one position, no event | `analyse --static` | one window; detections merged across frames; no before/after test |
| map | different sample positions | `analyse --map` | no windows; NO merging across frames; a pixel recurring in many frames is flagged detector-fixed |

### Choosing geometry and matrix per series

```bash
midas-snapshot select --frames FRAMES_DIR --geometries near=params_450.txt far=params_900.txt \
    --matrices fcc1=a.cif bcc1=b.cif hcp1=c.cif --out setup.json
```
Every (geometry, matrix) pair is fitted on frame blocks spread through the series. A fit counts
only if it is consistent (>= 3 sharp rings, each agreeing with the others within max(0.4 %, one
pixel in d), per-ring spread <= 0.3 %); among those the most COMPLETE wins (fraction of the
reference's lines present), not the one with most rings. A fit whose rings all lie on a more
complete pattern of the same block (bcc 110/220/400 on fcc 111/222/422) is never chosen, and
between identical ring sets the pattern predicting fewer lines wins. Several known phases per series are
allowed; one structure type at cells more than 3 % apart is two phases. The full ranking is
written. Matrices of one pattern fit alike: report the fitted cell, not the reference's element.

Read `unexplained` before trusting the labels: it lists, per block, strong sharp lines that no
known phase explains. The reference list is finite, and a phase outside it is reported as the
nearest reference that catches some of its lines. A non-empty list means the phase set is
incomplete; index those lines (or add a reference) before reading the phase column.

## Configuration keys that decide results

| Key | Meaning | Default |
|---|---|---|
| `flip` | raw -> geometry orientation (`ud`, `lr`, `udlr`, none); verify on the calibrant profile | `ud` |
| `mask`, `invalid_below` | raw-orientation mask; raw values below this are invalid | none, 0 |
| `matrix_cif`, `matrix_fit_lines` | dominant phase; lines used for the per-window scale | -, 8 |
| `fit_window_deg` | ring-centroid half-window | max(0.06, 6 px) |
| `halo_band`, `base_band` | diffuse band and quiet band (clear of every ring) | none |
| `sigma_px`, `local_box`, `statistic`, `fa_per_image`, `margin_px` | detector | auto (measured on bright spots), 31, gauss, 0.05, 21 |
| `tth_max` | ignore spots beyond this angle (low-count corners) | unset (no cut) |
| `window_sizes` | frames per window (1 = single frames) | [1, 25] |
| `rise`, `guard`, `after_len`, `baseline_until`, `min_before` | window rule on the halo trace (`rise` None = 6 x robust scatter of the baseline) | None, 50, 600, 200, 200 |
| `before`, `after` | explicit windows (override the rule) | none |
| `nsig_off`, `core_rel` | off-matrix classification | 6, 0.005 |
| `merge_px`, `min_det` | feature merging | 3, 3 |

## Outputs (all in OUT)

| File | Content |
|---|---|
| `snapshot_config.json` | the configuration used |
| `windows_W{W}.csv` | per window: first frame, matrix scale (coarse, fitted), rings used, halo, spot count |
| `spots_W{W}.npy`, `meta_W{W}.json` | spot table (columns in meta), threshold, calibration frames, wall time |
| `analysis.json` | windows and rule, sigma_matrix, features (with raw-photon test), candidate test, injection curve |
| `report.json`, `traces_W{W}.png` | summary citing the files above |

## Throughput (measured on one 96-core CPU node, 2026-09)

Single frames at 1475 x 1679 px: 5000 frames in about 200 s, 2000 frames in 90-120 s per node under
shared load, including reading; 25-frame sums 35-50 s per series. The injection control is serial.
