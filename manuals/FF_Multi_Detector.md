# FF-HEDM with several detector panels

`midas-pipeline run --scan-mode ff --detectors detectors.json` reconstructs one
far-field layer measured on several panels (a pinwheel of tiled detectors, or
panels at different standoffs).

## Inputs

* One `.MIDAS.zip` **per panel**, each carrying that panel's own geometry (Lsd,
  beam centre, tilts, distortion). Nothing is built from raw frames in this mode.
* `detectors.json`: a list with one entry per panel, e.g.

  ```json
  [{"det_id": 1, "zarr_path": "/data/p1.MIDAS.zip", "lsd": 3300000.0,
    "y_bc": 2300.0, "z_bc": 2150.0, "tx": 15.0, "ty": 0.0, "tz": 0.0},
   {"det_id": 2, "zarr_path": "/data/p2.MIDAS.zip", "lsd": 3420000.0, ...}]
  ```

* `--params Parameters.txt` for the sample / scan keys.

Preflight checks that every panel's zip exists **and that its stored geometry
(Lsd, beam centre, tilts) agrees with `detectors.json`**. The two feed different
consumers: `transforms` turns a panel's spots into lab coordinates with the
geometry in its zip, while the indexer's `DetParams` rows come from
`detectors.json`. A mismatch used to pass silently; it is now an error that
names the panel and the quantities.

### Parameter file against the zips

Each panel's zip is read as built and is **not** refreshed from `Parameters.txt`
(a single-panel run refreshes its zip; here that would overwrite the panel's own
Lsd / beam centre / tilts). The scan-level keys, `Wedge`, `OmegaStart`,
`OmegaStep`, `SkipFrame`, `Wavelength`, the lattice, `SpaceGroup` and `MinEta`,
are read from the zip too, and the same key in the parameter file used to be
ignored without a message (every generated paramstest read `Wedge 0.000000`).

Preflight now compares them, and **stops before any stage runs** when the file
sets one of them to a value a zip does not hold. The message lists, per zip,
the key, the zip's value and the file's, and gives the two ways out:

1. Update the zip to the file's value. Run it once per zip, `--dry-run` first;
   `--keys` limits it to the listed parameters so the file's other keys, such
   as a panel's Lsd, are not written into the panel:

   ```
   midas-refresh-zarr-params -fn panel1.MIDAS.zip -paramFN Parameters.txt --keys Wedge,OmegaStart --dry-run
   ```

2. Edit `Parameters.txt` so the key holds the zip's value.

`--accept-zip-params` runs on the zip values anyway and logs the differences as
a warning. `RingThresh` and `MaxRingRad` differ between panels by design and
only warn. `OmegaStart` is refreshed together with `scan_parameters/start`, the
dataset peakfit reads for omega (it is derived from `OmegaStart` and
`SkipFrame` when a zip is built).

### Per-detector omega offset

When one panel's omega record runs ahead of the others (the two scans of a
combined run, a trigger or readout lag), give that panel an
`omega_offset_deg` in `detectors.json`:

```json
{"det_id": 4, "zarr_path": "/data/p4.MIDAS.zip", "lsd": 3300000.0,
 "y_bc": 2300.0, "z_bc": 2150.0, "omega_offset_deg": 0.65}
```

True omega = recorded omega - offset, applied in peakfit to that panel only; the
zip is not touched. Only differences between panels matter: a common offset on
every panel is absorbed by the grain poses. Changing an offset after peakfit has
run is refused on resume (the old omegas would be reused), so delete the
`Det_<id>/` outputs or the result folder first.

## What the stages do

| stage | per-panel work |
|---|---|
| `hkl`, `peakfit`, `transforms` | run once per panel into `LayerNr_N/Det_<id>/` |
| `cross_det_merge` | concatenates the panels, renumbers SpotIDs globally, adds a `DetID` column, writes the merged `paramstest.txt` (`DetParams`, `RingRadii_Det<N>`, `EtaCoverage_Det<N>`, the union of the panels' rings), a layer `IDsHash.csv`, and `PanelOverlap.csv` |
| `global_powder` | per-ring eta coverage from every panel's geometry (non-square panels at their real shape) |
| `indexing`, `refinement` | the c-omp binaries, with per-panel geometry (below); seeds come from every ring by default (below) |

### Per-panel geometry in the c-omp indexer and refiner

Both read the panel of every observed spot from `Spots_det.bin` and the panel
table from the `DetParams` rows. In a multi-panel run:

* a seed's direction, ring radius and Friedel-pair geometry use the seed's own
  panel (distance and ring radius); a Friedel partner on another panel is moved
  into the seed's plane;
* a predicted spot whose (ring, eta) lies on no panel (the `EtaCoverage`
  arcs) cannot be observed, so it leaves the completeness denominator, the same
  way `PFCoverageAwareCompleteness` treats a voxel that was not lit;
* in the refiner each observed spot's grain-displacement correction and
  g-vector use its own panel's Lsd, and the corrected position is moved onto the
  reference plane before the comparison with theory.
* the refiner's position window follows the indexer's position grid. The FF
  indexer places a grain on a `StepsizePos` grid along the seed's line, so its start
  position can be off by a few steps, and the refiner fits position in two boxes of
  start +- `Rsample`/2, which reaches only `Rsample` from the start. In a
  multi-panel run the pipeline therefore raises the refiner's own `Rsample` to at
  least 3 x `StepsizePos` (the indexer keeps its value). `MIDAS_REFINER_POS_STEPS`
  changes the 3; 0 switches it off. A single detector is not touched.

A single-detector run takes exactly the arithmetic it always did (checked
bit-for-bit on `IndexBest_all.bin`, `OrientPosFit.bin` and `Grains.csv`).

Whether an installed binary has this is not visible from the package version
(C is compiled at install time). `midas_index.backend_c.supports_multidet()` and
`midas_fit_grain.backend_c.supports_multidet()` look for the marker the feature
adds; the pipeline warns when a multi-panel run meets an older binary.

### Seeds: why every ring

A seed is an indexed spot, and `process_grains` drops a cluster with fewer than
`MinNrSpots` (3) seeds. On a full detector a grain shows dozens of spots of the
one seed ring, but a panel layout sees only a few: two grains in the standoff
test had 32 observed spots each and only 2 spots of the seed ring, so 2 seeds.
Both were indexed at completeness 1.0 and then dropped. Across 13 single-ring
runs, 34 of the 35 lost grains were indexed but had at most 2 alive seeds; the
35th had no seed-ring spot in the omega window at all, so it was never tried.
By default a
multi-panel run therefore seeds from every spot of every ring inside the omega
window (`--seed-rings auto`). `--seed-rings transforms` restores the single ring
transforms chose; `--seed-rings 1,2,3` seeds those rings. More seeds cost
indexing time roughly in proportion. `MinNrSpots` is not lowered: three
reflections are the minimum that determine an orientation.

## Overlapping panels

Panels that overlap record one reflection twice. `cross_det_merge` always
measures the overlap and writes `PanelOverlap.csv`, one row per panel pair:

* `n_pairs` duplicates, and their fraction of each panel;
* the matching window the data support (`tol_pos_um`, `tol_ome_deg`), set from
  the fitted residual, not assumed;
* the **relative** alignment of panel b to panel a as a similarity transform on
  the lab plane: translation `ty_um`, `tz_um`, `scale`, `rot_mrad`, each with a
  standard error, plus the position scatter and the fit's condition number.

Read the transform in **lab** coordinates (`YLab`, `ZLab`), not pixel axes: a
beam-centre error in pixels is rotated by the panel's `tx`. Only the relative
misalignment is observable, so the file says two panels disagree, not which one
is wrong. A recorded-`Lsd` error shows up in `scale`, not in the translation.

Duplicates are kept by default. `--dedup-panels` (config `dedup_panels`) drops
one record of each pair, preferring the one farther from its panel's edge, and
moves a seed that sat on the dropped record to its kept partner.

## Validated, and what is not

**Read the position and orientation errors below with this in mind.** The simulator
puts each spot in the single nearest frame unless `omega_sigma_frames` is set, so
the omega a spot is recorded at is off by up to half a frame (rms 0.15 deg at
0.5 deg steps) whatever `pos_noise_px` says. That error, not the panel geometry,
produced the 1 to 3 um position floor and 0.05 deg orientation floor in the
figures below: with a realistic omega blur (`omega_sigma_frames=1`,
`frame_integrate=True`) the same pipeline gives 30 of 30 standoff grains (seed 5)
at 0.56 um and 0.0009 deg median, and 40 of 40 hydra grains (seed 2) at 0.46 um and
0.0009 deg. The grain COUNTS below are from the quantised-omega sets. With the
blur, 12 further seeds (6 per layout) recovered every grain and gave per-seed median
position errors of 0.25 to 0.54 um. Limits of that figure: spot positions were exact
(0.1 px = 20 um of position noise gives about 9 um; 0.3 px about 26 um), the 0.0009 deg
needs a blur of 0.5 frames or more (0.3 frames gives 0.008 deg, no blur 0.05 deg), so it
depends on the real rocking width, and the simulator and the pipeline share their
conventions.

Measured on simulated panels (`generate_pinwheel_synthetic_dataset`, no position noise,
c-omp binaries built from this tree). These are synthetic results, and the
seed-42 figures are single draws; the multi-seed ranges are the ones to quote.

* **Standoff layout** (four panels, two of them 120 mm further out), 30 grains,
  default completeness, seeds from every ring: 30 of 30 grains on each of 9
  seeds, no unmatched entries, per-seed median position error 1.4 to 3.4 um,
  median misorientation 0.04 to 0.06 deg. With the single seed ring it was 25 to
  29 of 30 (seeds 1 to 8, 42), and the binaries before this change gave 0
  grains. The median hides a tail: the worst grain per seed is 6 to 75 um off. The tail comes from members that
  refine badly and from the grain record being the member with the smallest
  internal angle, which sees only orientation. A member refines badly when the
  indexer's start position is more than about 100 um from the grain: the indexer
  places a grain on a `StepsizePos` grid along the seed's line (100 um here), and
  the refiner then fits position twice inside a box of start +- `Rsample`/2
  clamped to +- `Rsample`, so it cannot reach farther than 100 um (refined values
  end at start -+ 100 or on the +-100 wall). About 10% of multi-panel members
  start that far out (1% on one detector); on the omega-blurred sets 46 and 45% of such members end more than 15 um off, and 13 of 13 and 33 of 34 bad members are of this kind (the 34th started 57 um out, on the box wall, and stayed clamped there). Refining with `Rsample` 300 (what the pipeline now does by default, see above) left 0 of 2861 standoff and 2 of 3812 hydra members more than
  15 um off on the omega-blurred sets (from 62 and 276), and on the quantised sets cut them
  from 813 to 404 over 14 runs. With spot-position noise the wider window helps less
  (25 of 473 members still off at 0.1 px). A finer indexer `StepSizePos` (25 um) also cut
  the far starts from 28 to 9. Before the window change about 3% of hydra grains
  (11 of 400) were more than 15 um off. With the window in the pipeline, 14
  omega-blurred runs (7 per layout) recovered every grain with a worst delivered
  grain of 1.0 to 2.9 um (median position error 0.17 to 0.52 um), where the same
  kind of sets at the old window had grains 15 to 74 um off. Taking the median position of a grain's members instead cut the
  worst grain to 3.5 to 10 um (standoff) and 13 to 31 um (hydra). `Grains.csv`
  still carries the C `ProcessGrains` choice.
* **Hydra-geometry layout** (a different Lsd, beam centre and tilt on every
  panel), 40 grains: 40 of 40 grains on each of 6 seeds, per-seed median
  position error 3.2 to 5.8 um, worst grain 20 to 77 um. Single seed ring: 35 to
  39 of 40. Before this change: 0 grains.
* **Overlap:** on overlapping panels with a recorded-geometry error of known
  size, the simulated duplicates were found in 14 of 18 runs and 1 to 2 short in
  the other 4 (in the one run examined the spots were already missing from the
  per-panel tables before the merge). Beam-centre
  errors up to 1.5 px (rotated by the panel's `tx`) were recovered to 0.08 um,
  and a recorded Lsd error to 1e-7 in scale. The quoted standard errors are
  about 1.3 times too small (the truth fell inside 1, 2 and 3 SE in 52%, 85% and
  100% of 54 estimates).

Sensitivity, measured on the standoff set:

* **A wrong recorded Lsd.** The result assumes each panel's recorded Lsd is
  right. Written consistently into both the zip and `detectors.json` (so the
  preflight check cannot see it), an Lsd 0.03% too long on two panels still gave
  27 of 30 grains at 4.8 um; 0.15% too long gave 10 of 30. Grains were lost, not
  invented. `PanelOverlap.csv` reports exactly this kind of error, as a scale
  term, wherever panels overlap.
* **Noise.** With spot positions scattered by 0.3 px, the overlap matcher still
  found all 161 duplicates with no false pairs, and the shift and scale were
  within about one standard error; with 60 to 400 um of scatter recall fell from
  0.99 to 0.92 at unchanged precision. Noise was not added to the indexing and
  refinement runs.

Not tested:


* **Real panel data with grains.** The reporter's six-panel Eiger set (issue
  #75) confirmed the routing and merge; its indexing result predates the C
  support described above.
* **Two small panels.** A two-panel layout with about 17 spots per grain gave
  10 of 30 grains at a 32 um median position error: too few spots, not a
  panel-geometry effect.
* Noise. Spot positions in all simulations are exact.
* Whether removing duplicates changes the grains found. On two four-panel
  overlapping sets (10 to 12% of reflections seen on two panels, 30 grains)
  `--dedup-panels` made no difference: 30 of 30 with or without it, position
  median 0.67 vs 0.75 um and 0.67 vs 0.35 um (the maxima moved both ways).
  Noiseless, exact geometry; duplicates cost nothing there.
* Panels whose Lsd differ by a large factor: the indexer's reference-plane
  scaling is exact for a grain at the origin and approximate by (offset / Lsd)
  otherwise, and the refiner's is exact. Per spot the indexer error is a few um
  for 100 um offsets and tens of um for mm-scale ones.
