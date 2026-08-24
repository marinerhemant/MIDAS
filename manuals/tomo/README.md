# Tomography — raw projections to a registered sample shape

**Use this doc to start a fresh session on a dataset this pipeline has never seen.**
Paste it in together with `LAB_NOTEBOOK.md`, then give:

```
Data folder:      <ABSOLUTE PATH>          # projections (TIFF stack or .raw)
Paired scan:      <FF / NF / pf layer, or "none">
Sample material:  <e.g. Ce, NMC811, or "unknown, tell me from the data">
```

**Scope.** Absorption or propagation-based (in-line phase contrast) parallel-beam
tomography at APS 1-ID, reconstructed with `midas_tomo` (FBP + shift sweep).
Measured on: a 128×128 `.raw` stack with a text sidecar (bt_1id_jun25b NMC811, 1.17 µm
px, phase contrast at 50 mm) and a 2048×2320 TIFF stack (bt_1id_jul26 Ce, 95 keV,
capillary). If your data is cone-beam, laminographic, uses a different
reconstructor, or is a diffraction-contrast tomography (DCT) scan, **stop and
ask** — DCT is `manuals/dct-tt/`, and the geometry below does not apply to it.

The output this doc set exists to produce is not a pretty reconstruction. It is a
**sample shape registered into the MIDAS lab frame**, usable as the illuminated
volume and the absorption medium for a diffraction experiment.

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order, hard rules, halt conditions | always |
| **`COORDINATES.md`** | the frames — MIDAS, APS, sample, reconstruction grid, detector | **before any registration**; it serves every modality, not just tomo |
| `DIAGNOSIS.md` | symptom → test → cause → lever | **when something looks wrong** |
| `ENVELOPE.md` | what a tomogram can and cannot determine here | before promising an answer |
| `RUNBOOK.md` | where it runs, what healthy looks like, pick-up point | on resume |
| `LAB_NOTEBOOK.md` | evidence, ledger, **retracted claims** | before re-investigating |

## STOP — read this before touching anything

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** Three failures in this
technique finish and look right:

* a **mirrored** reconstruction is self-consistent — it reconstructs perfectly, every
  quality metric is normal, and the microstructure is simply handed the wrong way;
* a **wrong pixel size** produces a sharp, plausible reconstruction of an object of the
  wrong size, and silently rescales every downstream path length and volume;
* a **truncated** scan reconstructs into a bowl-shaped cupping artefact that looks like
  a density gradient rather than like missing data.

So the trigger is not confusion. **Halt on these named conditions, whether or not
anything seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| The pixel size is not in the acquisition metadata | It is not recoverable from the reconstruction. Inferring it from the sample's apparent size assumes the density and composition you were going to measure. |
| Transmission does not return to ~1 either side of the sample at **every** angle | Truncated, or something else is in the beam. Which of those it is changes what you do, and both look like attenuation. |
| The reconstruction is to be registered against FF/NF/pf and the stage vertical is not recorded | Registration would have to be fitted, and a fitted registration cannot then validate the reconstruction — the fit absorbs the error the check looks for. |
| The scan is phase contrast (non-zero propagation distance) and a **mask** is wanted | Thresholding a propagation-based reconstruction without phase retrieval gives a hollow shell from edge enhancement, not a filled sample. |
| Sample composition or density is unknown and an **absorption** correction is wanted | μ cannot be computed, and μ·D decides whether the correction is measurable at all. |

When you halt, say which row fired, what you measured, and what you would need to
proceed. Finish everything not blocked by it first.

### Hard rules

1. **Establish the flat/dark layout by measuring frame means, never by assuming.**
   Layouts differ between beamtimes: one dataset here is `dark, white1, white2,
   projections`, another is `flats, projections, flats, darks`. Getting it wrong
   gives transmission > 1 or < 0 — and if you only look at the mean you will not
   notice. A negative transmission anywhere means the assignment is wrong; stop.
2. **The beam does not fill the detector.** Restrict every transmission statistic to
   the illuminated region. Outside it `flat − dark ≈ 0` and the ratio is noise that
   reads as a smooth, plausible plateau of partial attenuation.
3. **Never default the pixel size, the rotation-axis position, or the in-plane
   handedness.** `midas_stress.frames.tomo_grid_to_midas` refuses all three for this
   reason. `n/2` is a guess, not an axis.
4. **Registration to a diffraction scan is a read, not a fit** — the sample-stage
   vertical (APS y = MIDAS z). See `COORDINATES.md` §4.
5. **Check μ·D before promising an absorption correction.** Below ~0.1 the correction
   is at or under the per-spot noise and the honest answer is "no detectable effect".
   Measure it from the projections; do not assume it from the element.

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| Whole-frame mean used for transmission | unilluminated corners read as ~0.4 attenuation | rule 2, `DIAGNOSIS.md` |
| Flats assumed at one end only | transmission > 1, or negative | rule 1 |
| Shift sweep index chosen by eye and not recorded | double edges; mask fattens; every volume inflates | `DIAGNOSIS.md` |
| Pixel size taken from a template config | correct-looking recon, all lengths wrong by a constant | rule 3 |
| Phase-contrast recon thresholded directly | hollow-shell mask, sample interior excluded | scope gate |

## 0. Verify the install

```bash
python -c "import midas_tomo, midas_stress; print(midas_tomo.__version__)"
python -c "from midas_stress.frames import tomo_grid_to_midas, TOMO_IN_PLANE; \
           print(sorted(TOMO_IN_PLANE))"
```

The second import is the gate that matters: it is the frame conversion, and if it is
absent the installed `midas_stress` predates the tomo frame support and any
registration done with it will have been improvised.

## 0a. THE ORDER

| # | Step | Where | Why it is here and not later |
|---|---|---|---|
| 0 | Verify the install | §0 | invalidates everything downstream if skipped |
| 1 | Survey the stack: frame layout, dtype, dimensions | `DIAGNOSIS.md` | every later number depends on which frames are flats |
| 2 | Establish the geometry from metadata: pixel size, angles, energy | acquisition config | not recoverable later; halt if absent |
| 3 | Measure transmission and μ·D **on projections**, before reconstructing | `ENVELOPE.md` | decides whether an absorption correction is even testable, and costs minutes |
| 4 | Truncation check | `DIAGNOSIS.md` | a truncated scan cannot become a sample mask; find out before spending the reconstruction |
| 5 | Reconstruct with a shift sweep; **record the chosen shift** | `RUNBOOK.md` | the rotation axis is an output, and the choice must travel with the result |
| 6 | Phase retrieval, if propagation distance ≠ 0 | `RUNBOOK.md` | required before thresholding |
| 7 | Threshold to a mask, with a sensitivity band | `ENVELOPE.md` | the threshold multiplies the illuminated volume directly |
| 8 | Register into MIDAS lab | `COORDINATES.md` §4 | vertical from the stage; in-plane checked, not assumed |
| 9 | Verify the registration, and check the check had power | `DIAGNOSIS.md` §5 | a symmetric sample defeats most of these tests |

## Sibling doc sets

`manuals/ff-hedm/`, `manuals/nf-hedm/`, `manuals/pf-hedm/`, `manuals/dct-tt/`
(diffraction-contrast tomography — a different measurement), `manuals/xrd-ct/`.
