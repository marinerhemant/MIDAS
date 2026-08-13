# The 2022 MPE U3O8 dataset: what has been established

Findings about the dataset `midas-dt` was developed against. Tracked
deliberately: `packages/*/dev` and `CHECKPOINT*.md` are both gitignored in this
repo, so anything recorded only there does not survive the working tree.

Every number here is reproducible with a named script.

## Where the data is

Reach: `ssh chiltepin` then `ssh haydn`, both as `s1iduser`. haydn has no
`~/.ssh/config` entry on the Mac and is not directly reachable.

| | |
|---|---|
| raw | `/scratch/s1iduser/mpe_nov22_midas2/mpe_nov22/` (2.2 TB, 190 files) |
| 600 A | `dm_dt_pf_U3O8_600A_000161..000215` (55 translations) |
| 700 A | `dm_dt_pf_U3O8_700A_000461..000515` (55) |
| 800 A | `dm_dt_pf_U3O8_800A_000831..000887` (57) |
| darks | 22 `dark_before_*.raw`; the nearest preceding one per scan |
| params | `/scratch/s1iduser/DTnewversion/ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt` |

The `43-97` range named in `DT/runDTrecon.py` is **not** on haydn. 161-215 is
the same 600 A sample and has 2023 reference output.

## File layout — verified by arithmetic, not inferred

One 8192-byte header **per file**, then contiguous int32 frames of 1475x1679:

```
data  14,274,698,292 B = 8192 + 1441 x (1475 x 1679 x 4)   -> 1441 frames
dark      99,069,192 B = 8192 +   10 x (...)               ->   10 frames
```

Both divide with no remainder, which is what makes the layout confirmed.

## Geometry — verified visually

λ = 0.136994 Å (**90.5 keV**, not the 55.618 keV in the file's own comment),
Lsd = 1071098.336 µm, BC = (790.3118888, 864.5394861) px, px = 172 µm,
`ImTransOpt 2`, RhoD = 150000 µm.

`dev/look_at_frame.py` renders a raw frame with ring radii overlaid on that
beam centre. **The marker lands on the beamstop and the circles land on the
rings.** BC and the radius scale are both correct.

*(Note for anyone re-deriving this: `DTScan.frame()` already applies
`ImTransOpt 2`, so `BC_z` is used AS-IS. Flipping it again puts the marker
51 px off and makes correct radii look wrong.)*

## The rings are CONTINUOUS

Smooth powder rings, not spots. **This is the assumption midas-dt rests on** —
XRD-CT requires continuous rings, and a coarse-grained sample belongs to
scanning-3DXRD instead. With the Pilatus module gaps masked, the azimuthal
coefficient of variation is 0.64 at R = 115 px and 0.35 at R = 205 px.

*(Counting the module gaps as azimuthal structure makes every ring look
"spotty". Mask them.)*

## Ring list — INCOMPLETE

`dev/inspect_u3o8_lineout.py`, translation 27, 24 frames averaged, 60-560 px:

| R (px) | d (Å) | height |
|---|---|---|
| 205.29 | 4.157 | 65.6 |
| 248.38 | 3.437 | 44.6 |
| 323.53 | 2.640 | 38.8 |
| 115.11 | 7.412 | 36.3 |
| 483.85 | 1.767 | 27.6 |
| 253.39 | 3.369 | 25.0 |

**The image shows roughly 15-20 rings; the peak finder reported 6.** This list
is a floor, not a census, and that matters for indexing (below).

Both 2023 channel choices are validated by it: `rad_105_125` contains the
115.11 px ring, `rad_470_490` contains the 483.85 px one. The strongest ring
(205.29 px) was not used by the 2023 runs at all.

## Phase assignment — UNRESOLVED

`dev/index_u3o8_rings.py`, via `midas_dt.index_rings`, at a loose 2% tolerance:

| phase | matched | rms residual |
|---|---|---|
| α-U3O8 (C2mm, 6.716/11.960/4.147) | 4/6 | 7971 ppm |
| γ-UO3 (Pbnm, 9.813/19.93/9.711) | 4/6 | 5805 ppm |
| CeO2 (the calibrant) | 0/6 | — |

**Neither candidate fits.** Residuals of 5800-8000 ppm are far too poor for a
correct phase, and the two rings neither indexes include the **strongest ring
in the pattern**. CeO2 at 0/6 is the control showing the matcher does not
simply match everything.

The geometry is verified, so this is **not** a scale error — an earlier
hypothesis to that effect (from γ-UO3's uniformly negative residuals) is
refuted. It is a phase problem, and the incomplete ring list is the more
likely weak input.

**No d-spacing or strain map from this dataset should be read as a lattice
measurement until this is resolved.**

### Next steps, in order

1. Extract a complete ring list — lower the threshold, deblend overlaps.
2. Re-index against a wider phase list (U4O9, UO2, other U3O8 polymorphs).
3. Only then treat a d-spacing map as a lattice measurement.

## The legacy code, and what is wrong with it

`DT/runDTrecon.py` is a 2022 session transcript, not a driver: hard-coded
paths, two mutually inconsistent methods run unconditionally, and a call to
`DetectorMapper` when the binary is `DetectorMapperDT` — with the return code
ignored, so it had been failing silently. The real surviving pipeline is
`/scratch/s1iduser/mpe_nov22_dt/recon_peak_all_mul.py`.

**The 12 fit-output channels are mislabelled in every legacy Python script.**
The C is canonical and self-consistent in two places
(`IntegratorPeakFitOMP.c` `valTypes[]`, `PeakFit.c` `Rfit[]`): slot 5 is
`MaxIntensityObs`. Both `runDTrecon.py` and `recon_peak_all_mul.py` omit it,
shifting every label from index 5 on — a file named `*_BGFit_*` holds
`MaxIntensityObs`. Indices 0-4 are unaffected, which is why nobody noticed.

**When reading 2023 output, index by position and take the name from
`midas_dt.conventions.FIT_OUTPUT_NAMES`.**

Also: `PeakFit.c` sets `SigmaG = SigmaL = x[2]`, so the legacy "pseudo-Voigt"
is a 5-parameter model with one shared width and its 12 outputs carry 11
distinct values. Constrain them equal before comparing against 2023 results.
