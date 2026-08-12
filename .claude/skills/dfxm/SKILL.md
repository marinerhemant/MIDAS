---
name: dfxm
description: >-
  Take a dark-field X-ray microscopy (DFXM) dataset from raw detector frames to
  per-pixel orientation and strain maps and a report: survey the scan, configure
  the material and reflection, subtract the pedestal, reduce a mosaicity/strain
  scan by moment analysis, attempt the multi-reflection full deformation-gradient
  tensor (gated by inter-reflection registration), check the kinematic validity
  boundary, and report with provenance. Use when asked to reduce, reconstruct or
  diagnose a DFXM / dark-field microscopy scan, when handed a folder of DFXM rock/
  roll/strain frames, or when a DFXM orientation or strain map looks wrong. Covers
  ESRF ID06-HXM style acquisition and the archived ID03 sets through the
  midas_dfxm chain; APS 6-ID-C is gated.
---

# DFXM reduction and analysis

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the `midas_dfxm` code it cites and stays usable without this
skill.

## Start here

Read **`manuals/dfxm/README.md`** — the spine. It is the only file meant to stay loaded:
scope gate, install gate, the order of operations, the hard rules, and the halt
conditions. It carries an index telling you which file holds which section; open those as
you reach them.

Then give, or work out:

```
Scan folder:     <ABSOLUTE PATH>     # the frame tree (a mosaicity / strain / multi-reflection scan)
Beamline:        <ESRF ID06-HXM | archived ID03 | other, tell me>
Sample material: <e.g. Cu / Al / unknown, tell me from the data>
Reflection(s):   <e.g. 002, 111 — or "find it from the geometry">
Question:        <orientation map | strain map | full-F tensor | dislocation typing>
```

## Eight things to know before you start

DFXM's silent failures are different from HEDM's. These are the ones a context-free session
gets wrong, each earned on real data or verified against the dynamical forward (Lab Notebook
cites the measurement for each). The spine carries **twenty** numbered hard rules; these are
the ones worth loading before you read anything else.

1. **Subtract the detector pedestal before you take the first moment.** On raw ID03 frames
   the naive centre-of-mass is **~67× too small** — the pedestal carries 98.5 % of the
   centroid weight. Reproduce the community reduction (`darling`) on background-subtracted
   frames, not raw ones (§2, Lab Notebook §1).

2. **Refraction is an absolute strain-scale gauge, not a per-pixel strain.** The mean
   refraction shifts the Bragg peak by a *constant* $\chi_{0r}/(2\sin^2\theta_B)$
   (≈ 144 µε for Cu 002 at 0.71 Å). On a **relative** intragranular strain map it is
   absorbed into the lattice reference — do **not** "correct" it per pixel. It becomes a
   real map bias only across a thickness or perfection **gradient** (§4, Lab Notebook §3).

3. **The kinematic read is exact for orientation and safe for defect contrast below
   ~0.3 Λ.** Centroid orientation is exact by symmetry in symmetric Laue; the kinematic
   defect/strain inverse holds until the crystal is thicker than ~0.3 extinction lengths,
   then it biases. Past that (thick / near-perfect / high-Z) you need the dynamical
   forward. Verified cross-model, not just inverse-crime (§4, Lab Notebook §4).

4. **Multi-reflection full-F is gated by inter-reflection REGISTRATION, not photon
   statistics.** Different reflections sit at different 2θ → different magnification and
   field of view; without co-registration metadata you cannot fuse them into a tensor.
   This is a **halt condition**, established on the real ID06 multi-Bragg set, not a
   tuning problem (§3, Lab Notebook §2).

5. **No ground truth on a real scan → validate by injection-recovery, not round-trip.** A
   forward/inverse round-trip to 1e-16 is a software-consistency check, not physical
   accuracy. Resample a *known* shift into the measured raw frames and recover it against
   the real noise/background (§2, Lab Notebook §1).

The last three come from re-analysing an archived deposit reduced by **another group's
pipeline** — the situation where the boring cause is almost always in a toolchain, theirs or
yours (rules 11–20, Lab Notebook §7):

6. **A published rocking FWHM is the *integrated* width — measure the per-pixel width on the
   frames.** It ran **2.6–2.7× wider** than the per-pixel median on one archived set, so
   dividing the step into a published width overstates points-per-FWHM by that factor. It
   invalidated the premise of an entire preregistration (§2, Lab Notebook §7a).

7. **Do not let the background subtraction follow the signal, and measure the detector gain.**
   A rolling-ball background whose kernel exceeds the ROI degenerates to a θ-dependent scalar
   (r ≈ +0.92…+0.97 with the rocking curve): per-pixel widths are then biased while the
   centroid is fine, so the map still looks right. And gain is often not 1 — one integrating
   sCMOS measured `var = 2.23·y + 149`, inflating every absolute χ²/dof ~2.2× and making an
   adequate model look rejected (§2, Lab Notebook §7b–c).

8. **Every control must be able to fail, and a correction must recover a planted bias.** Three
   "tests that could not fail" reached user-facing text in that campaign — a rule checked
   against a set containing no case that could break it, a null injecting the fitted model's
   own parameters, and a sweep that left the deciding threshold hardcoded. State what result
   would have refuted the control. A correction that moved 39.5 % of labels left a **24.2 %
   residual on a planted 6.4 % bias**, and that is what killed it (rules 17–20,
   Lab Notebook §5j).

## When something looks wrong

Go to **`manuals/dfxm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
indexed by symptom rather than by step. Every entry carries a test that can come back the
other way.

Before re-investigating anything, read **`manuals/dfxm/LAB_NOTEBOOK.md`** — **twelve**
attractive claims are recorded there as *refuted or softened*, each with the measurement or
reference that killed it (§5). None died of new physics: they died of a detrending choice, an
iid bootstrap on correlated pixels, a symmetric alternative that could not move a centroid, a
Gaussian fit on 7 points, a hardcoded constant, a wrong FWHM convention, and prior art —
twice over prior art that was cited in `midas_dfxm`'s own docstrings.

## Scope

**ESRF ID06-HXM style acquisition and the archived ID03 mosaicity sets**, reduced through
the `midas_dfxm` Python API (a library, not a CLI — the procedure is script-based). The
package's *capability* analyses (full-tensor inverse, Stroh dislocation typing, defect
model, dynamical Takagi–Taupin) are, as of this writing, **simulation-grounded**; the
README marks which steps are real-data-proven and which are demonstrated in simulation.
**APS 6-ID-C DFXM is gated** — it is a different instrument and a bilateral collaboration
(credit rules); confirm before touching that data.

## Sibling doc sets

`manuals/ff-hedm/` (far-field HEDM, skill `ff-hedm`), `manuals/nf-hedm/` (near-field,
skill `nf-hedm`), and, in the LaueMatching repository, `scripts/pipeline/laue/` (skill
`laue`). The paper that grounds this doc set is `packages/midas_dfxm/dev/paper/P_merged/`.
