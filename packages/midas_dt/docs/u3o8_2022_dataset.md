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

## Ring list — 28 rings

`dev/inspect_u3o8_lineout.py` (translation 27, 24 frames averaged, 60-560 px),
via `midas_dt.rings.find_rings` with a **rolling-median baseline**:

**28 rings above 3 sigma.** The strongest are R = 483.85 (SNR 47), 437.76
(26), 411.70 (25), 205.29, 323.53, 248.38 px. The full list with d-spacings is
printed by the script in a form ready to paste into an indexing input.

An earlier pass reported **6**. It used a *global* median baseline, which sits
far above the background at low R and below it at high R: it found the strong
inner rings and lost the rest. `midas_dt.rings` exists to prevent that, and a
test asserts the rolling baseline finds strictly more rings than a global one
on a profile with a realistic falling background.

Both 2023 channel choices are validated: `rad_105_125` contains the 115.11 px
ring, `rad_470_490` contains the 483.85 px one (the strongest in the pattern).

## Phase assignment — UNRESOLVED, and now well characterised

`dev/index_u3o8_rings.py`, all 28 rings, two tolerances:

| phase | 2.0% | 0.5% | rms @ 0.5% |
|---|---|---|---|
| α-U3O8 (C2mm) | 17/28 | 11/28 | 2406 ppm |
| **γ-UO3 (Pbnm)** | 24/28 | **17/28** | **1819 ppm** |
| U4O9 (I23, a=21.77) | 24/28 | 17/28 | 2379 ppm |
| UO2 (Fm-3m) | 5/28 | 2/28 | 2362 ppm |
| CeO2 (calibrant) | 3/28 | 1/28 | 2374 ppm |

γ-UO3 leads, but **17/28 at ~1800 ppm is not an assignment** — a correct cell
should index nearly every ring at a few hundred ppm. U4O9's equal count is
almost certainly chance: a = 21.77 Å gives a very dense reflection list, which
is exactly the failure mode the module docstring warns about. CeO2 at 1/28
remains the control.

### The mixture hypothesis: partially supported, not sufficient

The 600A/700A/800A series is an oxidation study, so a mixture is the obvious
explanation for no single cell indexing everything. Testing α-U3O8 and γ-UO3
together at 0.5%:

| | rings |
|---|---|
| α-U3O8 only | 3 |
| γ-UO3 only | 9 |
| both | 8 |
| **neither** | **8** |

The two ARE partly complementary, which supports a mixture. But 8 rings remain
unexplained, and they are not a random subset:

```
  91.06 px   d 9.369 A
  95.07 px   d 8.974 A
 115.11 px   d 7.412 A     <- the 2023 rad_105_125 channel
 205.29 px   d 4.157 A     <- among the strongest rings
 238.36 px   d 3.581 A
 264.41 px   d 3.229 A
 270.42 px   d 3.157 A
 332.55 px   d 2.568 A
```

Three of them are at **large d (7.4-9.4 Å)**, i.e. low angle. That is the
signature of a phase with a larger unit cell than anything tried here. And one
of the strongest rings in the pattern is in this set.

**A phase assignment that cannot account for the strongest ring, or for the
channel the 2023 analysis actually used, is not an assignment.**

### Next steps, in order

1. Search for a phase with large d-spacings (9.4, 9.0, 7.4 Å) — uranium
   oxide hydrates and layered uranyl phases are the obvious family, but this
   should be a search, not a guess.
2. Re-run the coverage check with any new candidate; the criterion is
   accounting for the unexplained 8, not raising the total match count.
3. Only then treat a d-spacing map from this dataset as a lattice measurement.

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
