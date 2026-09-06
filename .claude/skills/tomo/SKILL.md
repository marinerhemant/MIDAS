---
name: tomo
description: >-
  Take an APS 1-ID tomography dataset from raw projections to a sample shape
  registered into the MIDAS lab frame: read the scan's own record for geometry
  and frame layout, ingest the TIFFs, measure transmission and mu*D before
  reconstructing, find the rotation-axis shift automatically, optionally
  phase-retrieve, reconstruct, measure the detector roll, threshold to a mask,
  and register it against an FF, NF or pf scan. Use when asked to reconstruct
  or diagnose a tomography scan, when handed a folder of projection TIFFs or a
  .raw stack with a sidecar, when a reconstruction looks wrong (mirrored,
  cupped, hollow, double-edged), when the rotation-axis centre or the detector
  tilt needs finding, or when a diffraction analysis needs a sample shape for
  the illuminated volume or an absorption path. Also the reference for
  COORDINATE SYSTEMS across every MIDAS modality -- the MIDAS/APS axis
  permutation and how tomo, FF and NF register to each other. Covers
  parallel-beam absorption and propagation phase-contrast tomography;
  diffraction-contrast tomography (DCT) is a different measurement and is gated.
---

# Tomography reconstruction, and the coordinate reference

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites, gets checked by the repo's own hooks,
and stays usable without this skill.

## Start here

Read **`manuals/tomo/README.md`** — the spine. Scope gate, install gate, the order of
operations, the hard rules, and the halt conditions. It carries an index saying which
file holds which section; open those as you reach them.

Then give, or work out from the data:

```
Scan record:      <ABSOLUTE PATH to <prefix>_TomoFastScan.dat>
Image root:       <ABSOLUTE PATH to the local dir holding the scan's image folder>
Paired scan:      <FF / NF / pf layer, or "none">
Sample material:  <e.g. Ce, NMC811, or "unknown, tell me from the data">
```

## The one command

```bash
midas-tomo-reconstruct <scan_record> --root <image root> --out <dir> \
    [--crop ROW0 ROW1 COL0 COL1] [--measure-tilt] [--delta-beta N]
```

It reads the record, ingests the frames, finds the rotation-axis shift coarse-then-fine
with two criteria that must agree, reconstructs, writes NXtomoproc with provenance, and
prints the `SampleShape` call to use. **It stops rather than reconstructing on an
uncertified shift** — pass `--no-strict` to override, and everything downstream is then
marked unverified.

## Read the scan record. Do not read `tomocupy_args.yml`.

`<prefix>_TomoFastScan.dat` is self-describing: pixel size, propagation distance,
energy, handedness, angles, and the exact white/dark/projection frame layout. It is
normally at `<expt>/metadata/<expt>/<scan>/`.

**`tomocupy_args.yml` carries a different camera's pixel size.** It says 1.17 µm for
both beamtimes surveyed here; both scans ran on a FLIR-GH1 at 5X, which is 0.708 µm
(bt_1id_jun25b) and 0.69 µm (bt_1id_jul26). 1.17 µm belongs to the PointGrey. **A 1.65× pixel
error is 4.5× in every volume.** Same class of trap as the stale `exp_setup.yml EDGE:`.

`midas_tomo.scanrecord.read_scan_record` parses it and cross-checks the block sizes
against the recorded first/last image numbers, refusing when they disagree — an
off-by-one boundary averages projections into the flat field, silently.

## Six things to know before you start

1. **The un-illuminated detector does not read as zero.** Outside the beam
   `white − dark ≈ 0`, so the transmission ratio is noise and a clip floor turns it
   into `−log(1e-6) = 13.8` — an unlit row scores as *the strongest absorber on the
   detector*. **Derive the illuminated region from the flat field, never from the
   attenuation**, in both rows and columns. This trap was met four times in one
   campaign: as a "furnace with two windows" that was not there, and three times inside
   one function.

2. **Never default the pixel size, the rotation-axis position, or the in-plane
   handedness.** `midas_stress.frames.tomo_grid_to_midas` refuses all three. A wrong
   pixel size rescales every downstream length; a wrong axis translates the sample; a
   wrong handedness **mirrors** it — and a mirrored reconstruction is self-consistent,
   reconstructs perfectly, and is invisible in every quality metric. The metastr does
   record `left handed`, but that names a *convention*, not an axis assignment.

3. **Measure μ·D on the projections before reconstructing.** It decides whether an
   absorption correction is testable at all. Measured: NMC811 at 52 keV gives **0.05**
   (null, at the noise floor); Ce at 95 keV gives **1.63** (testable). Below ~0.1 the
   honest answer is "no detectable effect", and that is a result.

4. **A propagation-contrast scan may not yield a mask at any threshold.** Both datasets
   here were taken at D≈100 mm. Paganin retrieval exists (`--delta-beta`) and is
   **off by default** because it is a strong low-pass whose parameter sets how large the
   specimen comes out. On bt_1id_jun25b it did **not** rescue the mask; that outcome was
   reported rather than tuned away.

5. **A specimen fiducial beats every automatic criterion — look for one first.**
   Three Au cubes in `park_dmi_sam5` were missed by two automated searches: one
   found a cube and rejected it because its radius from the volume centre was
   constant across slices ("ring artefact") — but a fixed real object keeps the same
   `(row, col)` in every slice, so constant radius is exactly what it should do; a
   ring artefact is an annulus spanning all azimuths. The next search eroded the
   specimen mask to "look inside" and deleted everything on the boundary, where all
   three sat. **Render a slice and look before trusting a blob finder**, and do not
   erode away the surface.

6. **Check that a check could have failed.** Several here cannot, by construction: the
   V1 sinogram check has zero power on a cylinder; centroid containment is blind to a
   pure translation; and a threshold sweep over *percentiles of the data* pins
   `radius_spread` at exactly `100**(1/3) = 4.642` whatever the input. `manuals/tomo/`
   records which check is powerless on which sample.

## Finding the centre and the tilt

* **Rotation-axis shift** — `midas_tomo.center.find_center_consensus` scores two
  criteria that fail differently and reports `trustworthy=False` when they disagree.
  Choose the slices with `slices_with_signal`: evenly spaced probes land on empty rows,
  whose sharpness curves have no interior optimum, so `argmax` returns a sweep edge.

  **When it refuses, read the two per-criterion picks before doing anything else.**
  If they straddle the answer by ~1.5 px with **total variation low**, that is the
  documented TV bias, not your data — variance has been right on both datasets where
  this was checked. If instead the score is *flat* ("best within 1 % of the median"),
  the criterion separated nothing and `argmax` returned a number regardless; strong
  ring artefacts do this, being concentric about the axis by construction.
  `LAB_NOTEBOOK.md` §3.5 and §3.12, `DIAGNOSIS.md`.

  **Two levers that do not go through a sharpness proxy**, in order of preference:
  1. **A dense compact inclusion**, if the specimen has one — a marker, a precipitate,
     an Au fiducial. It has curvature in every direction and is not swamped by rings.
     On `park_dmi_sam5` a 50 µm Au cube gave a sharp edge-gradient peak with **3.3×**
     dynamic range where the bulk sweep varied by 0.85 %. Just look at it across
     shifts; the eye is good at this and the metric agrees.
  2. **180° half-scan agreement**, for a 360° scan: reconstruct the first and second
     halves separately and minimise `RMS(A−B)/RMS(A)` over the specimen support. Both
     halves image the same object, so they coincide only at the true axis, and rings —
     common to both — largely cancel.

  **Compare only shifts within one interpolation class.** A fractional shift resamples
  the sinogram, and that low-pass improves any agreement or smoothness metric whether
  or not the axis is right — so a mixed integer/fractional sweep shows a spurious
  period-1 oscillation with minima at half-integers. Fit integer and half-integer
  classes separately; their spread is a fair uncertainty.
* **Detector roll** — `midas_tomo.detector_tilt`, three estimators against **two
  references**: the beam-box edges reference the *slits*, while per-slice best shift and
  rotation-axis drift reference the *rotation axis*. Prefer `tilt_from_slice_shifts`
  over the centre-of-mass route. `compare_tilt_estimates` adjudicates, and refuses to
  recommend a value that flagged itself invalid.

## The coordinate reference

**`manuals/tomo/COORDINATES.md` is not tomo-specific.** It is the reference for every
MIDAS modality, and it lives here because tomography is the one that has to register
against all the others.

```
x_MIDAS = z_APS   (beam)
y_MIDAS = x_APS   (outboard)
z_MIDAS = y_APS   (up, and the omega rotation axis)
```

A cyclic permutation, therefore a proper rotation (`det = +1`). The single source of
truth in code is `midas_stress/frames.py`; anything that disagrees with that module is
wrong.

**Registration between tomo, FF and NF is the sample-stage vertical position** — a
recorded motor value, so it is read, not fitted. Fitting a registration and then
validating the reconstruction with the same data is circular.

**`COORDINATES.md` §4a is the recipe for validating one modality against another**
without falling into that circle: match on the quantity that needs no registration
(orientation), then score the one that does (position), so the second is an independent
test; pick any unavoidable discrete choice on a random half and report on the held-out
half; settle frame conventions by **scatter**, not by mean offset — the wrong beam-centre
convention once gave the *better* mean offset and was caught only by its scatter. Two
numbers there are not what they look like: a cross-modality misorientation contains both
instruments' precision, and **the other modality's grain list is a segmentation, not
ground truth** — re-segmenting one EBSD map moved an FF precision figure 13 points with the
reconstruction untouched. A third depends on a question you must **ask, not assume**:
whether the two modalities sample the **same volume**. If the diffraction layer is a thin
slice through the plane the reference sectioned, the position residual is a real accuracy;
if not, it is inflated by ~the grain radius. Regress the residual on grain size — it falls
in the first case and grows in the second. And when the layer *is* the section, the
reconstruction's **Z spread is its Z error**, otherwise unmeasurable.

**The omega sign:** on the aero stage the recorded SPEC angles run opposite to the
sample rotation. `TomoScan.thetas()` negates them and records that it did.

## When something looks wrong

Go to **`manuals/tomo/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
keyed by symptom. Before re-investigating anything, read
**`manuals/tomo/LAB_NOTEBOOK.md`**, which records what has already been refuted and,
just as usefully, which checks were found to have no power on which samples.

## Sibling doc sets

`manuals/ff-hedm/`, `manuals/nf-hedm/`, `manuals/pf-hedm/`, `manuals/xrd-ct/`,
`manuals/defect/` (diffuse-scattering defect metrology), and `manuals/dct-tt/` —
**diffraction**-contrast tomography, a different measurement with different geometry; do
not apply this doc set to it.
