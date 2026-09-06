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

## Healthy numbers

See `RUNBOOK.md`. Briefly: masked 10–15 %, kept/labelled ≈ 1/3, ~9 % of blobs split,
400–650 sub-peaks for a single crystal plus a gasket on 36 frames.
