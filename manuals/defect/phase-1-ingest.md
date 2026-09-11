# Phase 1 — ingest: raw frames → mask → background → 3-D spots → voxel cloud

Skip this phase if you already have a `VoxelCloud`; go to `phase-2-index.md`. Until
2026-09-01 this phase did not exist in the package and was done by an out-of-tree script.

## The order, and why

```python
from midas_defect.ingest import (build_mask, choose_sectors, subtract_background,
                                 find_blobs_3d, detect_powder_rings, flag_powder)
```

**1. Mask.** `build_mask(frames, low_count_threshold=20.0, grow=2)`

Two components, separately reported: the vendor gap/defect convention (negative pixel), and
*persistently* low-count pixels — dead pixels and the beamstop-mount shadow, taken from the
per-pixel **median over frames**, never from one frame. A pixel low once is counting
statistics.

`grow` dilates the result. Pixels bordering a module gap have anomalous response and, after
background subtraction, produce coherent **negative** structures. Verify the growth with
`count_signed_blobs`, do not assume it.

*Sanity:* sweep `low_count_threshold` and check the added-pixel count is on a **plateau**. A
tuned value that is not on a plateau is a red flag.

**2. Background.** `choose_sectors(frames, tth, azi, mask, candidates=(1, 8, 24, 48, 96))`

Polar median in (2θ, azimuth-sector) cells: removes what is azimuthally uniform — smooth
background *and* powder rings — and keeps what is localised. The sector count is **chosen by
a control that can fail**: diffraction is positive-only, so every coherent *negative*
structure the subtraction leaves is an artifact of the model. Fewer negatives wins, and the
comparison table comes back with the choice.

`n_sectors = 1` is the assumption that the background is azimuthally uniform. It is not, for
anything that absorbs through an anvil, a gasket or a furnace.

> **The one limit to check.** The model is smoothed over `smooth_bins × tth_bin` in 2θ and
> **cannot follow a ring narrower than that window** — such a ring survives into the spot
> list as a train of false reflections. `detect_powder_rings` measures the widths actually
> present; check the window against them rather than guessing.

**3. Spots.** `find_blobs_3d(stack, mask, threshold=200, min_vol=10, split_ratio=3.0,
gap_bridge=21, return_counts=True)`

> **Read the floor off the data, never off the default.** 200 counts is a default, not a
> property of your cloud. The reference `all_labels_qvox` cloud bottoms out at **30**, with
> **71.5 %** of voxels below 200 — a write-up that assumed the default misreported a fraction
> that moves **2.5×** between the two. Check `stack.min()` and the intensity histogram first.
> `ENVELOPE.md` §3.

Threshold → bridge detector gaps → label with **26-connectivity in (ω, row, col)** → drop
blobs below `min_vol` *real* voxels → watershed-split by **scale-free** prominence.

* 3-D, not per-frame: a reflection sweeps several ω frames (rule 3 in the spine).
* `split_ratio` is a *ratio*, not a count, so a 10³-count peak and a 10⁶-count peak are judged
  on equal terms. Absolute prominence shatters bright streaks; prominence relative to the
  blob maximum annihilates weak-but-real lobes.
* Position comes from the high-intensity **core** only (`core_frac`, flat over 0.3–0.7). A
  centroid over the whole sub-region is dragged down a streak's faint tail.
* Shape is second **moments**, not a Gaussian fit — these are elongated streaks and a Gaussian
  is a misspecified model.
* `return_counts=True` gives `labelled / rejected_small / kept / blobs_split / sub_peaks`.
  **Quote them** (spine rule 5).

**4. Powder separation.** `detect_powder_rings` then `flag_powder`

Rings come from the **azimuthal median of the data**, so this transfers to a gasket,
substrate or capillary never seen before — no ring table, no phase list. The discriminant is
**two-sided** and both halves are required: on a detected ring **and** having ≥
`min_companions` neighbours at its own radius and a different azimuth. Either half alone
mislabels — a good single-crystal index genuinely puts several reflections at one `|G|`.

*Acceptance gate:* flagging may only remove contamination. Re-run the index on the flagged-out
list; if it loses reflections, the flag is too aggressive and the flag is wrong, not the index.

**5. To q-space.** `geometry.pixel_to_qlab` then `geometry.qlab_to_qsample`, or
`data_io.load_voxel_npz` for an existing cloud. `midas_defect` uses **q = 2π/d** throughout;
mixing conventions is silent, so check against `lattice.bragg_shells`.

> **Then let the data referee the orientation convention.**
> `bragg_diffuse.check_orientation_convention` decides whether the cloud wants `OM` or `OM.T`.
> This is a property of the **cloud**, not a rule you can look up: two products of the same
> experiment needed **opposite** conventions, and a docstring asserting the transpose
> universally was wrong for one of them. Note the check is **blind on a satellite-only
> cloud** — both conventions score near zero and it correctly returns `decisive=False`; use
> the axis test there instead.

**6. Remember what a label is.** A connected component is a segmentation unit, not a physical
object: 3-D connectivity merges a reflection, its asterism and any streak leaving it into one
label. Do not later sum over labels and call the result a budget — see `phase-3`.

## The calling contract — copy this, do not reconstruct it

Every signature below was read from `midas_defect/ingest.py`, not remembered. The
author of this section rewrote the docs above in the morning and then hit **six**
of the traps in this table the same afternoon building a new sample's front end.
Reconstructing the chain from prose does not work; copy the block.

```python
import numpy as np, tifffile, torch
from midas_integrate_v2.compat.pyfai import poni_file_to_row_col
from midas_defect.ingest import (live_frames, build_mask, subtract_background,
                                 find_blobs_3d, detect_powder_rings, flag_powder)
from midas_defect.geometry import (Geometry, detector_angle_maps, pixel_to_qlab,
                                   qlab_to_qsample)

raw    = np.stack([tifffile.imread(path(k)) for k in range(NFRAME)]).astype(np.float32)
live, _ = live_frames(raw)                          # dead / shutter-ramp frames out
fidx   = np.flatnonzero(live)                       # RAW index of every kept frame
frames = raw[live]

# A PONI gives a SEED, never the centre. At this boundary use the row/col function:
row0, col0 = poni_file_to_row_col(PONI)             # NOT poni_file_to_bc -- see the traps
# ...flip the row if your reader disagrees with the calibration's, then MEASURE the centre
# from Friedel pairs (midas_calibrate_v2.friedel) and re-run from here on it.
g = Geometry(lsd_um=LSD, bcy_px=COL_BC, bcz_px=ROW_BC, px_um=PX, wavelength_A=LAM,
             n_pix_y=NCOL, n_pix_z=NROW, omega_first_deg=OMEGA0,   # centre of RAW frame 0
             omega_step_deg=DOMEGA, n_frames=NFRAME)

tth, az = detector_angle_maps(g)                    # the maps every call below needs
m    = build_mask(frames)
mask = m.mask if hasattr(m, "mask") else m          # returns an OBJECT
sub  = subtract_background(frames, tth, az, mask)   # (frames, TTH, AZ, mask)

spots, counts = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                              split_ratio=3.0, gap_bridge=21,
                              return_counts=True)   # returns a TUPLE
spots = spots[spots.n_frames >= 2].reset_index(drop=True)

# spots.frame is a FRACTIONAL index into the LIVE stack. Map it back through fidx; never floor it.
omega = OMEGA0 + DOMEGA*np.interp(spots.frame.values, np.arange(len(fidx)), fidx)

qlab = pixel_to_qlab(spots.row.values, spots.col.values, g, device="cpu")
q    = qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(omega, dtype=qlab.dtype))
                       ).detach().cpu().numpy()     # sample frame, q = 2 pi / d
qn   = np.linalg.norm(qlab.detach().cpu().numpy(), axis=1)
stth = np.degrees(2*np.arcsin(np.clip(qn*LAM/(4*np.pi), -1, 1)))
rad  = np.hypot(spots.row.values - ROW_BC, spots.col.values - COL_BC)

rings = detect_powder_rings(sub.max(axis=0), tth, mask, azimuth_deg=az)
pw    = flag_powder(stth,
                    np.degrees(np.arctan2(spots.row.values - ROW_BC,
                                          spots.col.values - COL_BC)),
                    rad, rings)
keep = ~pw
```

| trap | symptom | fix |
|---|---|---|
| **`detect_powder_rings` without `azimuth_deg`** | occupancy comes back all-`NaN`, 3–6× too many "rings", **half the real reflections discarded as powder** | always pass `azimuth_deg=az`. Measured on one sample: 105–152 rings → 17–50, kept spots 1133 → 1776 |
| **`poni_file_to_bc` fed to `Geometry`** | the beam centre lands on the wrong AXES — 123/59 px on one Pilatus, every ring and q wrong, nothing raised | `midas_integrate_v2` calls the ROW axis `BC_y`; `midas_defect.Geometry` calls the COLUMN `bcy_px`. Use `poni_file_to_row_col` and pass `bcz_px=row, bcy_px=col` |
| **flooring `spots.frame`, or treating it as a RAW frame after `live_frames`** | half the transverse q residual (0.0199 → 0.0141 1/Å, anisotropy 2.92 → 2.01), or every ω off by the number of dropped leading frames | keep the fractional centroid and map it through the live indices, as in the block above. One project's long-standing "frame i == raw frame i+2" offset was exactly this |
| hand-built `tth` / `az` maps | η from untilted pixel offsets while 2θ is tilted; one project's local helper had 94 importers | `detector_angle_maps(g)` (added 2026-09-10) |
| `pixel_to_qlab` / `qlab_to_pixel` default device | `TypeError: can't convert cuda:0` (or `mps:0`) `device type tensor to numpy` | `device="cpu"` |
| `find_blobs_3d(..., return_counts=True)` | unpacking error, or a DataFrame where you expected one | it returns `(spots, counts)` |
| `build_mask` | `tth_deg shape () != frame` further down | it returns an object; take `.mask` |
| `flag_powder` argument order | wrong spots rejected, silently | `(spot_tth, spot_azimuth, spot_radius, rings)` — arrays, not the spots frame |
| `RingSet.radius_px` | `AttributeError` | fields are `centre_deg`, `width_deg`, `occupancy`, `profile*` |
| 2θ per spot by interpolating `tth[:, BCC]` against radius | errors to 21.7°, powder filter rejects the wrong spots | compute 2θ from each spot's own `|q|`, as above |

**Nothing in this chain fails loudly on a wrong argument order.** `subtract_background`
took a mask where 2θ belonged and raised only because a shape mismatch happened to
surface three calls later.

## Calibration, same discipline

`midas_calibrate_v2.calibrate(...)` needs `mask` (**nonzero = BAD**) — before
2026-08-29 it could not take one at all and bad pixels entered the cake as
genuine zeros. Seed with `make_seed` **from the image**, never from a copied
geometry or a delivered `.poni`.

| trap | note |
|---|---|
| **`make_seed(..., use_diplib=True)` SEGFAULTS** | do not opt in. It was the default until `midas-calibrate-v2` 0.14.0 and is now `False` on all three entry points; a segfault is not a Python exception, so the internal `try/except` cannot catch it and the process dies with no traceback |
| `make_seed` may return `n_measured=1` | one ring is not a calibration; check `n_measured` before believing `Lsd`. Its `rms_px` will be ~1e-14 precisely because nothing was fitted |
| `weight_by_radius=` on `calibrate()` | accepted, warns, **does nothing** — a v1 C-file key no Python code reads |
| **pyFAI `.poni` → MIDAS BC** | there is no documented conversion and the naive `Poni1/px, Poni2/px` is WRONG. Settle the centre from the DATA: the correct one maximises ring sharpness. On one frame the naive conversion was 46.5 px out and fragmented every ring into 2–3 radial peaks |
| **a delivered `.poni` may be ROW-FLIPPED against the image as your reader returns it** | every ring is at the wrong radius and nothing warns you; a Friedel beam centre disagrees with the PONI's `getFit2D` centre by tens of px in ROW only | flip the image once on load and work in one row order for the rest of the run. **NOTHING IN THE METADATA DECLARES THIS.** In particular `Detector_config: {"orientation": 3}` does NOT: orientation 3 (BottomRight) is pyFAI's DEFAULT and is geometrically identical to 0 — both give p1 increasing down rows, i.e. no flip. TIFFs usually carry no Orientation tag (274) either. It is knowable ONLY from the data: measured on La3Ni2O7, as-read was +57.44 px wrong in row and row-flipped agreed to 0.13 px |
| **detector count rate past the validated correction** | strong Bragg cores read low, and node WIDTHS inflate with brightness — a defect signal that is really a detector one | check the peak rate before quantifying any width. Measured on La3Ni2O7: cores at **6.8–7.2 Mcounts/s/pixel with 1256 px above 2 Mcps**, 3–7× past where Pilatus/CdTe rate correction is validated. Restricting to dim nodes helps but does not fix it — a cut on a background-subtracted local mean is NOT a cut on pixel rate |
| capping `max_ring_radius_px` | the WRONG remedy for high strain — reject clipped BINS (`min_cell_coverage`, on by default) and leave the outer rings, which carry the tilt leverage |

**Judging a calibration.** The `<100 µε` gate is for *strain* work. It is not the
right gate for indexing: 210 µε at |q| ≈ 4 Å⁻¹ is Δq ≈ 8e-4, about **0.1σ** of a
typical radial matching tolerance. If the deliverable is a lattice-parameter
*difference* (an a/b splitting, say), the radially symmetric part cancels and the
number that constrains you is the **azimuthal cos2η residual** — measure it, and
quote it as the floor. Always overlay the predicted rings on the image before
believing any of it.

## Healthy numbers

See `RUNBOOK.md`. Briefly: masked 10–15 %, kept/labelled ≈ 1/3, ~9 % of blobs split,
400–650 sub-peaks for a single crystal plus a gasket on 36 frames.
