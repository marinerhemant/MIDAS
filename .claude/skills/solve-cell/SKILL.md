---
name: solve-cell
description: >-
  Determine a crystal LATTICE from spot positions when the cell itself is in question:
  survey what the spots can and cannot constrain, fix the geometry first because a cell
  cannot be better than the geometry it was fitted on, find the lattice (ab initio with
  no cell supplied, or from reciprocal-lattice rows), refine it and refine several
  domains jointly, decide its symmetry and its distortion MODE rather than assuming
  a != b, identify which member of a structural series it is against candidates of equal
  provenance, invert an equation of state for pressure, audit completeness, and report
  every number with the identifiability limit that bounds it -- including when that limit
  says the quantity asked for is not obtainable at any precision. Use when asked to index
  or refine a unit cell, determine a space group or lattice symmetry, measure an
  orthorhombic splitting or a shear, identify a phase or a Ruddlesden-Popper member,
  get a pressure from a diffraction pattern, or when a refined cell looks wrong. Many
  grains with a KNOWN cell is ff-hedm; the diffuse scattering BETWEEN the Bragg peaks is
  defect; continuous powder rings are xrd-ct; geometry from a powder calibrant is
  calibrate-integrate. Atomic positions are NOT in scope -- this determines lattices,
  not structures.
---

# Lattice and cell determination

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository
so it lives beside the `midas_hkls` and `midas_defect` code it cites and stays usable without
this skill.

## Start here

Read **`manuals/solve-cell/README.md`** — the spine: scope gate, the order of operations, the
calling contracts, and the halt conditions. Then give, or work out from the data:

```
Data:      <ABSOLUTE PATH>   # rotation frames, or a spot list, or a q-space cloud
Geometry:  <ABSOLUTE PATH>   # a .poni / Parameters.txt, or "measure it from the data"
Known:     <e.g. "nothing", or "tetragonal, a~3.6", or a candidate CIF/series>
Goal:      cell | symmetry | a/b splitting | phase / series member | pressure | completeness
```

## Is this the right doc set?

| you have | you want | go to |
|---|---|---|
| spots, cell UNKNOWN or in question | the lattice, its symmetry, its distortion | **here** |
| spots, cell KNOWN, many grains | orientations + strain per grain | `ff-hedm` |
| grains already indexed | what the scattering BETWEEN peaks says | `defect` |
| continuous powder rings | per-voxel phase / strain / texture | `xrd-ct` |
| a powder calibrant | detector geometry | `calibrate-integrate` |
| a lattice, and you want atom positions | Rietveld / structure solution | **NOT IN SCOPE — say so** |

## Twelve things to know before you start

Each cost a real retraction on La3Ni2O7 (Ruddlesden-Popper n = 2, DAC) or on the samples
before it. Long form and evidence: `manuals/solve-cell/`.

1. **A cell is never better than the geometry it was fitted on — but scale errors and
   SHAPE errors are different animals.** Distance and wavelength are exactly common-mode:
   they scale every d equally and CANNOT change an axial ratio. Only beam centre and tilts
   are azimuth-dependent and can. So if `c/a` looks wrong, refining the distance will never
   fix it, and if `a` and `c` are both off by the same factor, suspect the scale.
   (`phase-1-geometry.md`)

2. **A gate or a seed that contains the answer is not a confirmation.** A packaged
   "re-derived" gate that hardcodes `CELL_A, CELL_C` from the hand index and refines from it
   measures nothing; an unseeded re-index of the same data landed 0.35 % away in both axes.
   Selecting a subset with `h+k+l even` and then reporting the symmetry returns CUBIC at
   97.5 % — the selector WAS the answer. Check what seeded the run before you quote agreement.

3. **The h↔k permutation null for an a/b splitting is INVALID — it cannot fail.**
   A null that cannot fail licenses nothing. Neither can a null a deliberately WRONG model
   also beats: always run a decoy alongside, and settle it with a structural test.

4. **A `return None` floor will silently answer for your null.** A refiner returning `None`
   below 6 matches made both the null and the decoy report 0, giving a bogus "p = 0.000".
   Uncensored, the null median was 3 and the decoy 4. Re-run every null with the floor lowered.

5. **Ab initio on a centred lattice returns the PRIMITIVE cell.** I4/mmm is body-centred, so
   the answer is 3.632/3.705/9.947 with angles ~79.6°, V = 129.4 — the third axis being the
   centring vector (a/2, a/2, c/2). Compared against the CONVENTIONAL cell (V = 251.1) a
   correct answer reads as a "×0.52 failure". Convert with `midas_hkls.conventional` before
   judging anything. This cost most of a session.

6. **Decide the distortion MODE before measuring a splitting.** In a Ruddlesden-Popper
   SUBCELL an Fmmm supercell distortion is a **γ shear** — γ = 89.613° vs 90.387° for the two
   variants — with a and b IDENTICAL. Fitting a diagonal cell to sheared data manufactures a
   splitting. Use `distortion_mode` / `shear_separable`, not `ab_separable`.

7. **The SIGN of an a/b splitting is a gauge, not a measurement.** A 90° rotation about c*
   swaps a and b and refits equally well; the sign is inherited from the seed. A splitting
   quoted with a consistent sign across domains is the signature of an **index asymmetry**
   artifact, not a physical result — run `index_asymmetry` and report it alongside.

8. **Set every search window from the model's own residual distribution**, and split it.
   An eyeballed 2° window (14× the median residual) manufactured a false positive retracted
   the same day. And a single scalar tolerance in G-space mixes a ~1 px RADIAL error with an
   ANGULAR error set by the ω step: a split radial/angular criterion took one sample from
   7 to 12 reflections without loosening the cell.

9. **Never ring-filter before a 3-D indexer** (it cost 17 of 45 reflections), and **never rank
   seed families by d-spacing** when c* lies near the beam — the largest-d families are (00L)
   with ZERO observed spots, while the brightest spots sat in a family ranked 18th of 23.

10. **Phase ID needs candidates of EQUAL provenance and a line count.** Giving the favourite a
    cell refined from THIS dataset at THIS pressure while rivals get ambient cells tests cells,
    not phases. And a d-matching comparison is decided by how many lines each candidate offers
    — La4Ni3O10 is Bmab with 97 lines, not F with 50. Both traps fired here, together.

11. **A quantity obtained by inverting through a fitted curve is a property of the FIT until
    you show otherwise.** Vary the functional form and leave-one-out every anchor before
    quoting a spread: a "37 GPa discordance" moved to 5.1 GPa when one provenance-mixed anchor
    was dropped, and the interpolant-choice range alone was 35.5 GPa. Never splice an anchor of
    different temperature, phase or instrument into a reference ladder. (`phase-4-phase-id.md`)

12. **Read the conditions off the record, never off a filename.** A directory named
    `<sample>_LT/wide_25K` records a TEMPERATURE. A verification lens read "ambient pressure,
    no cell" out of it; the frame log contained only exposure times and file paths, and the
    sample was in a diamond anvil cell at tens of GPa.

## Read the envelope before promising an answer

**`manuals/solve-cell/ENVELOPE.md`** lists what these data cannot determine at any precision —
including the two that recur: an a/b splitting when every reflection has h = k or h = k = 0
(|G| is symmetric under a↔b, and **no precision fixes this**), and which member of a
structural series, when the accessible reflections carry almost no c information and c is the
only parameter separating the series. If the ask is on that list, say so before running.

## Log a halt

Stop and ask rather than guess when: the lattice will not converge from any seed; two files
give one quantity two values (reconcile them, do not pick); a fitted parameter rails at a
bound; or the answer changes sign or order under a re-seed.
