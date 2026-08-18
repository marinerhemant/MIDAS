# DCT + topotomography — from a beamline scan to grain shapes and intragranular fields

**Use this doc to start a fresh session on a dataset this pipeline has never seen.**
Paste it in together with `LAB_NOTEBOOK.md`, then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # HDF5/EDF detector frames
Metadata / geom: <ABSOLUTE PATH>   # or "find it" -- usually there is nothing usable
Material:        <e.g. fcc / hcp / unknown -- tell me from the data>
Goal:            grain map | one grain's 3-D shape | intragranular orientation field
```

**Scope.** Two related techniques on the same instrument, sharing one geometry chain:

* **DCT** — box beam, sample rotated through 360°, each grain *flashes* at a few
  discrete ω as reflections cross the Ewald sphere. Many grains at once, one shape
  each. Pipeline: segment spots → Friedel-pair → index → assign → back-project.
* **TT (topotomography)** — the goniometer is tilted so **one** grain's **G** lies
  along the rotation axis, so that grain diffracts *continuously* through the whole
  360° sweep. One grain, hundreds of topographs, and the only route here to an
  **intragranular** field.

If your rings are continuous powder rings rather than discrete spots, this is the
wrong doc set — that is `xrd-ct`. If you have a far-field spot pattern with no
tomographic translation, that is `ff-hedm`.

> **On sources.** Every number in this doc set is a measurement made by this project,
> on data this project processed. Datasets, proposals, instruments and people are
> referred to generically throughout, by design. Where a figure comes from a specific
> run, the *script* that produced it is named — that is the provenance you can re-run.

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, the order, hard rules, halt conditions | always |
| `phase-0-survey.md` | DCT or TT? what is actually recorded; the ω trap | first |
| `phase-1-geometry.md` | **geometry from the data**, because the header has none | before anything quantitative |
| `phase-2-dct-index.md` | segment → Friedel pairs → self-calibrate → index grains | DCT |
| `phase-3-dct-shapes.md` | assignment → per-frame extraction → SIRT → grain map | DCT, if shapes are the goal |
| `phase-4-tt.md` | tilt solution, reachability, conditioning, masks, reconstruction, fields | TT |
| `phase-5-report.md` | what to state, what to label provisional, provenance | at the end |
| `ENVELOPE.md` | what these measurements **can** determine, and what they cannot | **before promising an answer** |
| `DIAGNOSIS.md` | symptom → discriminating test → cause → lever | **when something looks wrong** |
| `INSTRUMENT.md` | detector/stage conventions that cannot be recovered later | **before touching a new instrument's data** |
| `RUNBOOK.md` | a real multi-tens-of-GB scan taken end-to-end, 0 → 862 grains | on resume, or as the shape of a run |
| `LAB_NOTEBOOK.md` | evidence, and **seven retracted results** | before re-investigating anything |

## STOP — read this before touching anything

### The one fact that governs both techniques

**Nothing you need is in the file header, and what is there may be wrong.**

On the DCT scan taken end-to-end here, every geometric quantity — pixel size, detector
distance, rotation-axis column, lattice type, λ/2a — was **derived from the data**,
because the header carried none of it. The one geometric number that *was* in the
header was the **sensor** pixel, not the imaging pixel: the true effective pixel came
out a factor **6.65** away. On topotomography scans from the same instrument the
metadata file disagreed with the camera actually in use by a factor **2** in every
length, silently.

So: **derive the geometry, then check it against something you did not fit.** A wrong
pixel size or distance produces a complete, plausible, entirely wrong reconstruction.

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** All of these finish and look
right:

| Condition | Why you cannot decide it yourself |
|---|---|
| The **effective pixel size** is not confirmed against a known length | Header pixel was wrong by 6.65× on one scan and 2× on another. Scales every grain, every position, every field. |
| The **material** is unknown and any *absolute* length is wanted | Only `s = λ/2a` is measurable from the pattern. λ and `a` are individually undetermined. Indexing and the grain map do not care; d-spacing and strain do. |
| **Handedness** is not pinned by something outside the diffraction | Only `y_sign × ω_sign` is fixed by the data. Flipping both mirrors every grain and changes **no residual**. Report as *mirror-ambiguous*; do not pick one to clear the gate. |
| An indexing **tolerance was set by hand** rather than from a null | At a 4× looser margin the real data and an ω-scrambled null indexed **identically** (2761/2902 seeds, completeness 0.250 both). That result was believed, written down, and retracted. |
| A **grain shape** is reported without a null | Every threshold produces *some* compact body. SIRT cores separate **15.0×** in volume from a spot-swap null; hull assignment on contested voxels does not separate at all. |
| The TT grain's **two reflections are closer than ~60°** | The rotation field's third component is then not weakly constrained but nearly null: a 13.3° pair gives eigenvalues `[0.0067, 0.4933, 0.5]` — 75× worse in one direction. You can still solve; you cannot claim a tensor. |
| A per-voxel **field** is reported without a wrong-support control | A deliberately wrong support scored 0.810 against the true support's 0.860. The data determine the field, not the domain — "it converged on my mask" is not evidence the mask is right. |
| The **ω per frame** is assumed from the motor block | On one TT set the rotation motor read a single constant value in **all 2880 frames** and the per-frame angle list in the config was empty. Per-frame angles were simply not recorded. |

When you halt, say which row fired, what you measured, and what you would need. Finish
everything not blocked by it first.

### Hard rules

1. **Derive the geometry from the data and state how.** Effective pixel from a known
   physical length in the field of view (here a slit box, two axes agreeing to 1.4 %);
   rotation-axis column from whatever makes Friedel-pair ring radii sharpest; lattice and
   `λ/2a` from a multi-ring fit — 5 rings, **2 free parameters, 0.91 px rms**.

2. **Set every tolerance from a null, never by hand.** The ω-scrambled null is the
   workhorse: permute ω, re-run, require real ≫ null. The adopted margin (0.52° on the
   first ring, with a 0.09 minimum-match fraction) sits above the null's maximum
   completeness of 0.069. Four times looser and the null matched the real data exactly.

3. **Feed the indexer Friedel-pair *virtual* spots, not raw spots.** `(y+y')/2 − c` and
   `(z−z')/2` are exactly what a point grain on the axis would give, so the sample-radius
   and beam-height parameters go to the floor and ring assignment becomes unambiguous.
   Raw-spot ring assignment was impossible here — rings 84 px apart while grain position
   shifts a spot by up to 150 px.

4. **The grain position is fixed in the SAMPLE frame, not the lab.** Each Friedel pair
   flashes at its own ω, so one shared *lab* position across a grain's pairs is the wrong
   model. Including `Rz(σω)` in the design matrix cut the residual 52 → 41 µm and turned a
   smeared centre cloud into the sample cross-section.

5. **Misorientation symmetry goes on the RIGHT.** With `v_sample = U v_crystal`,
   equivalents are `U·S`, so the misorientation is `Uaᵀ Ub S`. `S` in the middle reported
   **29.8°** for a pair genuinely **0.33°** apart, and left seeds refusing to cluster
   (367 clusters from 488 seeds; 205 after the fix). Cross-check against
   `midas_stress.misorientation` — that is what caught it.

6. **Threshold reconstructions with Otsu, never a fraction of the max.** Streak artefacts
   put a few voxels far above the grain's own level; `0.5 × max` reported **60 µm grains
   as 6 µm**.

7. **One thread per worker.** `torch` grabs every core in every worker: 20 workers × 64
   threads did not finish a *single* grain in 32 minutes. With `torch.set_num_threads(1)`
   and `OMP_NUM_THREADS=1`, all 121 finished in ~3 min.

8. **Never report a grain map without saying how much of it is dilation.** In the adopted
   map **86 % of the labelled volume is dilation**, not measurement, and its clean
   appearance is that interpolation. ~22 % of the domain is uncontested at *any*
   threshold — the information ceiling. Lowering the threshold converts uncontested →
   contested, not uncontested → measured.

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `midas_diffract` places spots at the **antipode** of `bragg_flashes` — exact inversion about the beam centre, which `flip_y` cannot express | Every forward-simulated spot in the wrong place; comparisons look "nearly right" | phase-2, `LAB_NOTEBOOK` |
| `midas_diffract` converts ω to a frame **without wrapping**, so half a 360° scan is silently discarded | Reads as a *physics* limit on completeness. It is not | phase-2 |
| `IndexBest.bin` **col 13 is `n_t_spots`, col 14 is `n_matches`** — the module docstring says the opposite; `_seed_record` is authoritative | Every seed looks perfect | phase-2 |
| `midas-index` treats `OutputFolder` as the directory **holding** `Spots.bin`, not its parent | Silent no-op or wrong inputs | phase-2 |
| MIDAS FF margins are in **µm against the ring radius** | At an 880 µm ring radius a 150 µm margin is a ~10° window and everything matches | phase-2 |
| Frames stored **X-flipped** (a flip flag in the frame header) | Reconstructs a mirrored grain; looks perfect | phase-0 |
| **Double-subtracting the dark** when frames are already corrected | Suppressed weak spots; reads as a detection limit | phase-0 |
| DCT **erodes grain boundaries** | A reconstruction came out ~30 % larger than a reference grain map, and growing the *reference* by 2 voxels improved agreement. Do not "correct" the wrong one | phase-3 |
| **Below midas-stress 0.9.0**, `rodrigues_to_orient_mat` returned the right **axis** at the wrong **angle** (60°→80°, 90°→180°) | Any Rodrigues-stored grain map read through the old version is wrong at large misorientation, silently. Check the version; read deposits with `midas_dct_tt.rodrigues_to_crystal_to_sample`, which also handles the negated convention | phase-0, `LAB_NOTEBOOK` |

## 0. Verify the install

```bash
$PY -c "
import midas_dct_tt, midas_hkls, midas_invert, torch
print('midas_dct_tt', midas_dct_tt.__version__)
print('midas_hkls  ', midas_hkls.__version__, '(need >=0.7.2: Lattice.reciprocal_cartesian_vectors)')
print('midas_invert', midas_invert.__version__, '(need >=0.1.1: fit(lr_schedule=...))')
from midas_dct_tt import tt_alignment, topotomo_tilts, best_reachable_pair, sirt
print('ok')"
$PY -m pytest tests/ -q        # in packages/midas_dct_tt
```

Both floors are correctness, not features: `>=0.1.1` because `recon.py` passes
`lr_schedule=` (a `TypeError` on 0.1.0), and `>=0.7.2` because `goniometer.reciprocal_basis`
calls `Lattice.reciprocal_cartesian_vectors` (an `AttributeError` on 0.6.x).

## 0a. THE ORDER

```
phase 0   survey        DCT or TT? frames, flips, dark, and WHAT OMEGA IS ACTUALLY RECORDED
phase 1   geometry      pixel from a known length; axis column; lattice + lambda/2a; omega sign
          ---- confirm the pixel against something you did not fit. Do not skip. ----
DCT branch
phase 2   index         segment -> Friedel pairs -> self-calibrate -> virtual spots -> index
          ---- set every tolerance from an omega-scrambled null. ----
phase 3   shapes        forward-assign -> per-frame extraction -> SIRT -> threshold -> map
          ---- report the dilation fraction and the uncontested fraction. ----
TT branch
phase 4   tt            reachability + conditioning FIRST, then masks, volume, orientation field
          ---- a 13 deg pair cannot give a rotation tensor, however good the fit. ----
phase 5   report        with provenance, and with the provisional labels intact
```

The DCT and TT branches are **independent** and answer different questions. DCT gives many
grains and one shape each; TT gives one grain and its interior. A DCT map does not validate
a TT field, and a TT field does not need a DCT map — though a DCT map is the natural way to
*choose* which grain to scan (`midas_dct_tt.chain`).

## 1. Where things live

| Thing | Where |
|---|---|
| TT alignment, ψ scan, DCT ω scan conventions | `midas_dct_tt.conventions`, `.scan` |
| Goniometer tilts, stage reachability, best reachable pair | `midas_dct_tt.goniometer` |
| Rotation-field conditioning law and its eigenvalues | `midas_dct_tt.rotation_coverage` |
| Forward model: topographs, detector, PSF, noise | `midas_dct_tt.forward`, `.detector`, `.project` |
| Friedel pairing, spot assignment | `midas_dct_tt.pairing` |
| Reconstruction (SIRT, differentiable) | `midas_dct_tt.recon` |
| 12-D deformation-field inverse | `midas_dct_tt.field_inverse` |
| Instrument geometry import + the Rodrigues convention adapter | `midas_dct_tt.esrf` |
| Scan planning from an FF/NF grain map | `midas_dct_tt.chain`, `.planning` |
| Extinction / kinematical validity | `midas_dct_tt.extinction` |
| The real end-to-end DCT chain (~100 numbered scripts) | `packages/midas_dct_tt/dev/real_data/dct/` |
| The real TT chain | `packages/midas_dct_tt/dev/real_data/` |

## 2. Done means

* The **effective pixel** and **detector distance** are derived from the data, with the
  check that confirmed them stated.
* Every **tolerance** used in indexing has a null behind it, quoted.
* Any grain map carries its **dilation fraction** and **uncontested fraction**.
* Any shape carries a **null separation** (spot-swap or ω-scramble), not just a picture.
* Any TT field carries its **conditioning** (the two reflections' separation and the
  eigenvalues) and a **wrong-support control**.
* Anything that depends on the unsplit `λ/2a`, on handedness, or on an assumed material is
  **labelled as such in the text that leaves the session**, not only in the working notes.

## Sibling doc sets

`manuals/ff-hedm/` (far-field HEDM, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/pf-hedm/` (scanning 3DXRD, skill `pf-hedm`), `manuals/xrd-ct/`
(powder-like diffraction tomography, skill `xrd-ct`), `manuals/dfxm/` (dark-field X-ray
microscopy, skill `dfxm`), and in the LaueMatching repository the `laue` skill.
