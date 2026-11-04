---
name: sx
description: >-
  Take a single-crystal ROTATION scan (monochromatic beam, area detector, one crystal turned
  about one axis, discrete Bragg spots, optionally in a diamond-anvil cell) from raw frames
  to a corrected, merged F2 table and a refined crystal structure with midas-sx: read the
  frames, resolve the image orientation and omega sense by indexing score, find the lattice
  blind and match it to a reference setting, refine orientation plus the geometry the header
  does not carry (beam centre, distance, tilts, rotation-axis wedge), integrate on a fixed
  support, correct (Lorentz, polarization, anvil absorption), scale several runs with a smooth
  per-run scale, merge robustly, refine positions and displacement parameters, and report with
  the limit on every number. Use when asked to reduce or refine a single-crystal structure
  from rotation frames, to merge several rotation runs, to get a SHELX .hkl, or when an
  R_int / R_F2 / GooF from such a scan looks wrong. NOT for: an unknown or disputed cell
  (solve-cell), many grains with a known cell (ff-hedm), diffuse scattering between the
  Bragg peaks (defect), continuous powder rings (xrd-ct), white-beam Laue (laue), or finding a
  twin law from scratch (this package takes a declared twin law and a declared cell; it
  does not discover them).
---

# Single-crystal rotation-method structure refinement (midas-sx)

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository,
beside the `midas_sx` code it cites, so it stays usable without this skill.

## Start here

Read **`manuals/sx/README.md`**: the spine (scope gate, order of operations, install gate, the
things that already cost a retraction, halt conditions). Open the phase files as you reach them:
`phase-1-ingest-and-setup.md`, `phase-2-reduce-and-merge.md`, `phase-3-refine.md`; package traps
and what is opt-in or unverified are in `PACKAGE_NOTES.md`.

Then give, or work out from the data:

```
Data:       <ABSOLUTE PATH>   # rotation frames: Pilatus/imgCIF .cbf, Bruker .sfrm, or an array
Runs:       <how many, omega range and step of each, phi/chi of each>
Reference:  <CIF with cell + space group + START model>  or  "none (then no model-free readout)"
Detector:   <distance, pixel, swing angle if any, count cutoff, exposure>
Goal:       F2 table | .hkl | structure refinement | what to change at the beamline
```

## Is this the right doc set?

| you have | you want | go to |
|---|---|---|
| rotation frames of ONE crystal, cell (or a reference CIF) known | merged F2, a refinement | **here** |
| spots, cell unknown or in question | the lattice | `solve-cell` |
| many grains, known cell | orientations, strain per grain | `ff-hedm` |
| grains indexed, diffuse scattering between peaks | defects | `defect` |
| continuous powder rings | phase / strain per voxel | `xrd-ct` |
| polychromatic beam | Laue indexing | `laue` |
| a twinned crystal, twin law and cell DECLARED | twin fraction | **here**, narrow scope (`PACKAGE_NOTES.md`) |
| a twinned crystal, twin law unknown | find it | **NOT IN SCOPE, say so** |

## What this package is, in four lines

Frames -> spots (`midas_defect` front end) -> orientation and geometry refined from the spots
(`orient.py`) -> fixed-support box integration, corrected at the calculated Ewald crossing
(`midas_hkls.corrections`) -> `workflow.merge_runs` (smooth per-run scale, robust merge) ->
`refine.py` (weighted least squares, gated verdict). Physics lives in `midas_hkls`; this package
holds the workflow only.

## Install gate

```
midas-sx 0.0.1     (this package: packages/midas_sx)
midas-hkls>=0.17.0   corrections (polarization, rotation Lorentz, anvil absorption), frames, ab_initio, SpaceGroup/Lattice
midas-defect>=0.9.1  mask, polar background, 3-D spots, pixel -> q geometry (itself needs midas-stress>=0.16.0 in the working tree)
numpy>=1.22  scipy>=1.10  pandas>=1.5  torch>=2.0
optional: fabio>=2023.6   (extra "io": CBF / Bruker readers use it)   gemmi>=0.6   (tests, CIF reading)
```

A `midas_hkls` without the `corrections` module cannot run the reduction (the 0.16.0 version string
in the working tree predates the release that carries it). Check
`python -c "import midas_hkls.corrections"` first.

## Four things that decide the result, before anything else

1. **The geometry of a real scan is not in its header.** The beam centre can be 4 px off, a header can
   list two distances, a swung detector's `Beam_xy` is the perpendicular foot and not the beam
   position, and the image orientation and sense of rotation come from the indexing score, not from
   the header. (`phase-1-ingest-and-setup.md`)
2. **Check for a smooth multiplicative drift of each run before you trust any merge.** On the
   L-alanine runs the systematic floor in F2 was 14 % against 2.7 % from counting statistics until a
   per-run smooth scale and a robust merge were added: R_int 15.67 -> 6.77 %. (`phase-2-reduce-and-merge.md`)
3. **Refinement results here are a LOCAL refinement.** The start model decides the hand and origin
   of a polar or non-centrosymmetric structure, the basin is a few tenths of an angstrom, and ten
   starts that end at one point are ONE result. (`phase-3-refine.md`)
4. **Read the envelope.** What is established, on what, and what is only simulated:
   `manuals/sx/README.md` section "Status in one paragraph" and `packages/midas_sx/README.md`.
   Everything in it was shown on three public real data sets and an independent simulator; the
   structure-quality claim rests on ONE crystal (L-alanine, three runs).

## Rules of the house that apply here

* **A result is provisional until a fresh context has tried to refute it** (`/verify`). The status
  words in the docs (ESTABLISHED / PROVISIONAL / INCONCLUSIVE / REFUTED) are the ones the log
  recorded; do not promote them in a summary.
* **Preregister the reading** (`/preregister`) before you look at the reference: the L-alanine
  campaign lost a day to a reference-readout definition (least-squares scale 18.7 % versus
  L1-optimal scale 14.15 % on the same table) that was fixed only after seeing both.
* **Never re-implement what a `midas_*` package already does**: corrections and symmetry are in
  `midas_hkls`, orientation sampling and symmetry operations in `midas_stress`, pixel -> q in
  `midas_defect`.
* **Mac: under 3 GB per process.** A 850 x 1679 x 1475 Pilatus run is 2.1 G voxels: spot finding
  belongs on copland (about 20 minutes per run).

## Log a halt

Stop and ask rather than guess when: the lattice will not index from any convention (the four
mirror-equivalent conventions tie exactly and are fine; a wrong class does not index); the
orientation refinement does not converge (`OrientationNotConverged`); a refined parameter rails at a
bound (`at_bound`, `null_params`); the refinement reaches GooF above 3; two files give one quantity
two values; or the answer changes under a different start, margin or polarization plane.
