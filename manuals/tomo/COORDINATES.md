# Coordinate systems at APS 1-ID — the reference for every MIDAS modality

> Part of the **tomo doc set**. Spine: [`README.md`](README.md).
> This file is **not tomo-specific**: FF, NF, pf and DFXM all use it, and it
> lives here because tomography is the modality that forced it to be written
> down — it is the one that has to register against all the others.

**Last checked:** 2026-08-23

The single source of truth in code is
[`midas_stress/frames.py`](../../packages/midas_stress/midas_stress/frames.py)
(`packages/midas_stress/midas_stress/frames.py:50` defines the matrix).
Anything here that disagrees with that module is wrong; the module is
unit-tested and `midas_dfxm` asserts against it.

---

## 1. The two lab frames

| | x | y | z |
|---|---|---|---|
| **MIDAS** (ESRF convention) | along the beam | outboard | up |
| **APS** (Park convention) | outboard | up | along the beam |

Written the way it is usually said out loud:

```
x_MIDAS = z_APS      (beam)
y_MIDAS = x_APS      (outboard, horizontal transverse)
z_MIDAS = y_APS      (up, and the ω rotation axis)
```

As a matrix, `v_APS = R_MIDAS_TO_APS @ v_MIDAS` with

```
R_MIDAS_TO_APS = [[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]]
```

This is a **cyclic permutation**, therefore a proper rotation: `det = +1`. That
matters more than it looks. A non-cyclic swap of two axes has `det = −1` and is
a *reflection* — it would mirror every reconstruction, and a mirrored
microstructure is self-consistent, reconstructs perfectly, and is invisible in
every quality metric. Handedness is checked in
`packages/midas_stress/tests/test_frames_tomo.py:47`, not assumed.

`R` is orthogonal, so the inverse is the transpose (`R_APS_TO_MIDAS`).

**What transforms how.** A vector rotates; a tensor conjugates; an orientation
matrix that takes crystal → lab is left-multiplied. Use the named helpers
rather than applying `R` by hand:

| quantity | helper |
|---|---|
| vectors `(..., 3)` | `vector_midas_to_aps` / `vector_aps_to_midas` |
| orientation matrices | `orient_midas_to_aps` / `orient_aps_to_midas` |
| tensors (strain, stress) | `tensor_midas_to_aps` / `tensor_aps_to_midas` |
| a whole `Grains.csv` | `grains_midas_to_sample` |

---

## 2. The sample frame

The sample frame is the lab frame rotated by ω about the rotation axis, and
coincides with the lab frame at ω = 0. The rotation axis is **z in MIDAS**, **y
in APS** — so `packages/midas_stress/midas_stress/frames.py:76` `lab_to_sample_rotation`
needs to be told which lab convention you are in. Getting that wrong rotates about the beam
instead of about the vertical, which is not subtle but is easy to type.

> **ω sign is a hard rule and lives elsewhere.** At 1-ID with the aero stage,
> ω must be negated. That is an acquisition fact, not a frame convention, and
> it is recorded in the FF and NF spines. Nothing in this file will save you
> from getting it wrong.

---

## 3. The tomography reconstruction grid

The third frame, and the one with no established convention — which is why it
needs the most care.

```
detector (tomo camera) : (row, col)   rows along the rotation axis
reconstruction grid    : (slice, iy, ix)
```

One slice per detector row, so **`slice` is the vertical axis**:

```
slice  ->  MIDAS z  =  APS y
```

`packages/midas_stress/midas_stress/frames.py:384` `tomo_grid_to_midas` converts
voxel indices to MIDAS lab µm. It requires, and **will not default**:

| input | why it cannot be guessed |
|---|---|
| `pixel_size_um` | recorded in the acquisition config, never in the reconstruction file. `midas_tomo` writes the cube shape into the *filename* and nothing else. It scales every path length and the illuminated volume. |
| `rot_axis_ix`, `rot_axis_iy` | an **output** of the reconstruction — the shift sweep finds it. `n/2` is a guess. A wrong axis translates the whole sample. |
| `in_plane` | the in-plane handedness depends on projection ordering, rotation direction, and whether the reconstructor flips its output. A wrong choice **mirrors** the sample. |
| `slice_pitch_um` | equals `pixel_size_um` only for an isotropic reconstruction; vertical detector binning breaks that. |

The eight legal `in_plane` values are in `packages/midas_stress/midas_stress/frames.py:372`; each is a signed axis
assignment and each is asserted orthonormal, so a choice can mirror or rotate
the reconstruction but never shear it.

---

## 4. How tomo, FF and NF are registered to each other

**By the sample-stage vertical position, which is recorded — so this is read,
not fitted.**

Every scan carries the stage vertical (APS y = MIDAS z) in its metadata. A
tomogram's slice 0 sits at some `slice0_z_um`; an FF or NF layer sits at its own
measured z. `packages/midas_stress/midas_stress/frames.py:449` `tomo_slice_for_z` turns the
second into the first.

That this is a *read* and not a *fit* is the whole point. Fitting a registration
and then validating the reconstruction with the same data is circular — the fit
absorbs exactly the error the check is looking for. Where a residual translation
does have to be fitted, fit it on one half of the objects and score it on the
other.

`tomo_slice_for_z` **raises** when the requested z falls outside the
reconstruction rather than clamping to the end slice. Clamping would extend the
sample mask past the tomographic field of view and fabricate path length, which
is worse than failing.

The in-plane registration is not given by a motor position and has to be
established per experiment. §5 of [`DIAGNOSIS.md`](DIAGNOSIS.md) lists the
checks and, importantly, when each has no power.

---

## 5. Detector frames

Distinct from all of the above, and the source of a recurring confusion.

* **Diffraction detector pixels** `(Y_pix, Z_pix)` become ideal lab µm through
  `packages/midas_transforms/midas_transforms/fit_setup/transform.py:96`, which ends
  with `Y = −R·sin(η)`, `Z = R·cos(η)`. The forward model's `CalcSpotPosition`
  (`packages/midas_ckernel/c_src/forward.c:21`) uses the identical expression — that
  agreement is what lets a detector mask be pushed forward into the frame where
  predicted spots live.
* **`Spots.bin` col 0/1 are wedge-corrected**, while the push-forward above is
  pre-wedge. At `Wedge = 0` these are identical by construction; across the
  parameter files sampled on 2026-08-22 the largest non-zero wedge was
  −0.0126°, displacing the outermost ring by ~0.28 px.
* **Masks are stored `[Z, Y]`** — first axis Z — per
  `packages/midas_peakfit/midas_peakfit/preprocess.py:132`, indexed at
  `packages/midas_peakfit/midas_peakfit/seeds.py:171`, which square-pads into
  `(NrPixels, NrPixels)` with the `[:Z, :Y]` block populated.

---

## 6. Traps, each of which has cost real time

1. **A mirrored reconstruction is self-consistent.** Nothing downstream flags
   it. Only an external check — a known fiducial, a transmission channel, or a
   modality measured independently — can catch it.
2. **`Rsample` / `Hbeam` / `BeamSize` are not sample or beam dimensions.** They
   are deliberately generous *search bounds*. Reading them as geometry is a
   documented trap; see the FF envelope.
3. **APS y is up; MIDAS y is outboard.** The letter `y` means different things
   in the two frames and both appear in the same conversation.
4. **`det = +1` is not decoration.** If you build an axis map by hand, check the
   determinant before trusting anything computed with it.
