# Phase 0 — survey: what is actually in these frames?

**Goal:** decide DCT vs TT, and find out what the acquisition really recorded — before any
number is computed. Everything here is cheap. Skipping it is how a whole reconstruction ends
up mirrored, half-discarded, or scaled by a factor of two.

## 0.1 Look at frames before reading any header

Open a handful spread across the rotation, plus a dark and a flat if they exist.

```python
import h5py, numpy as np
with h5py.File(master, "r") as f:
    print(list(f))                      # scan entries: the long one is your rotation series
    d = f[path_to_frames]
    print(d.shape, d.dtype)             # (n_frames, ny, nx), usually uint16
```

Ask, in this order:

1. **Discrete spots, or continuous rings?** Continuous rings ⇒ wrong doc set (`xrd-ct`).
2. **Does a given spot persist across many frames, or flash for a few?**
   * flashes for a handful of frames out of thousands ⇒ **DCT**
   * one blob persists through the entire sweep ⇒ **TT**, and the grain is already aligned
3. **Is there a direct beam / is the sample larger than the beam?** If the sample never
   clears the beam at any ω, the projections are truncated and **no absorption tomography is
   available from this scan** — you will not get a sample outline this way. On the DCT set
   here there was also neither a flat nor a dark, and FBP gave the classic truncation wedge.

## 0.2 The four acquisition facts that are not recoverable later

| Fact | How to check | Cost if wrong |
|---|---|---|
| **Is the frame already dark-corrected?** | Compare a frame's median against the dark's median. If the frame median is already at the dark floor, it is corrected | Double subtraction suppresses weak spots and reads as a *detection* limit |
| **Is the image flipped?** | Look for a flip flag in the frame header; and see §0.4 | A missed mirror reconstructs a mirrored grain that looks perfect |
| **Is ω recorded per frame?** | Read the motor value from several frames, not one | On one TT set the rotation motor read a **single constant value in all 2880 frames**, and the per-frame angle list in the config was empty. The angles simply were not recorded — they had to be reconstructed from the scan command |
| **What does the scan command say?** | The scan title usually encodes start, step, count, exposure | This is often the *only* honest source of ω. A `0 → 0.1 → 3600` command means 3600 projections over 360° at 0.1°, which is what you should trust over a static motor block |

## 0.3 Count what you have

```python
n_frames, ny, nx = d.shape
print(f"{n_frames} frames, {ny}x{nx}")
# frame-median image: the beam footprint and any slit box show up here
med = np.median(d[::max(1, n_frames//200)], axis=0)
```

The **frame-median image** is worth plotting on its own. It shows the beam box, any slits, and
dead regions — and the slit box is the single most useful thing in it, because it is a *known
physical length* and therefore the way to get the effective pixel size in phase 1.

## 0.4 Handedness: decide now that you cannot decide it

Diffraction alone fixes only the **product** `y_sign × ω_sign`. Flipping both mirrors every
grain in the reconstruction and changes **no residual anywhere**. On the DCT scan here the
product was pinned (to `+1`, via the fcc {111} interplanar angles 70.53°/109.47° recovered at
5σ), but the individual signs were not.

So handedness must come from outside the diffraction: a detector-orientation record for *this*
pipeline, a known chiral feature, or a fiducial. A configuration file from a *different*
acquisition on the same instrument is suggestive and **is not evidence**.

**Record it as undetermined and report the map as mirror-ambiguous.** Do not pick one to clear
the gate.

## 0.5 If a reference grain map exists

Useful as a format reference and, carefully, as ground truth. Two cautions:

* **DCT erodes grain boundaries.** A reconstruction here came out ~30 % larger than the
  reference grain, and *growing the reference* by 2 voxels improved the agreement. If you
  "correct" the reconstruction toward the reference you will be correcting the wrong one.
* **Rodrigues vectors carry a convention.** Grain maps written by the common Python
  microstructure toolchain use the negated convention, so read them with
  `midas_dct_tt.rodrigues_to_crystal_to_sample`. Separately, **below midas-stress 0.9.0**
  `rodrigues_to_orient_mat` was defective (right axis, wrong angle: 60°→80°, 90°→180°) — check
  your installed version, because the error is invisible at small misorientation.

## 0.6 Exit criteria

You may proceed to phase 1 when you can state:

- [ ] DCT or TT, and the evidence (flash vs persist)
- [ ] number of frames, ω range and step, and **where ω came from** (motor block vs scan command)
- [ ] whether frames are dark-corrected, and whether a flip is applied
- [ ] whether the sample clears the beam (⇒ whether any absorption tomography is possible)
- [ ] that handedness is pinned by something external, or explicitly is not

**Halt** if ω cannot be established at all. Everything downstream is a function of ω, and a
plausible wrong ω produces a plausible wrong grain map.
