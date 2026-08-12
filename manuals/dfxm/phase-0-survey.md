# Phase 0 — Environment and survey

> Part of the **DFXM doc set**. The spine — scope gate, install gate, hard rules, halt
> conditions and the order of operations — is [`README.md`](README.md). Section numbers
> (§n) are continuous across the set.

---

## 0b. Survey the scan folder — write `SURVEY.md` before promising anything

**Goal: a written `SURVEY.md` in your work directory answering *what is actually here*,
with every number read from the files, never from a folder or file name** (rule 10). Start
from [`SURVEY_TEMPLATE.md`](SURVEY_TEMPLATE.md) — copy it to `SURVEY.md` and fill it in.

Record, per acquisition:

| field | how to get it | why it matters |
|---|---|---|
| scan kind | frame count + which motor moves | decides the whole path: **mosaicity** (rock/roll, χ/φ) → orientation; **strain** (θ / energy) → d-spacing; **multi-reflection** set → tensor (and the §3 gate) |
| reflection(s) | the aligned Bragg setting in the metadata | fixes θ_B, Λ, χ (§1); **never** from the filename |
| energy / wavelength | the beamline metadata, cross-checked | sets θ_B and the extinction length; a wrong λ scales absolute strain |
| effective pixel size | the objective magnification record | µm/px on the *sample*; differs per reflection (the §3 wall) |
| frame count + step | the scan record | too few points across a rocking curve flattens the moment for a real reason |
| **is there a background / dark?** | the deposit | **you must subtract it before any moment (§2)** — if absent, stop and find one |
| co-registration metadata? | fiducials / shared sample frame in the deposit | **its absence is the §3 halt condition** — a multi-reflection tensor is unrecoverable without it |
| ground truth? | almost never present on public scans | if absent, physical accuracy comes from injection-recovery, not round-trip (§2) |
| **flux / monitor column?** | the motor metadata, read column by column | its **absence** blocks any intensity comparison between separately-acquired groups — halt (rule 16). The frame total is **not** a monitor on a rocking scan; it *is* the rocking curve |
| **magnification provenance** | the optical record, via ≥ 2 independent routes | it sets every µm/px in the result — never a script constant, and a factor ~2 error is the common one (rule 15) |
| **detector type and gain** | the detector record, then photon transfer on the frames | photon-counting vs integrating changes every absolute χ²/dof and σ; measure it, per detector (rule 13) |
| sampling **per channel** | step and pixel size for *each* reflection separately | a weak channel is often acquired far coarser than the strong one, so the highest-resolution map can be structurally blind to the physics of interest (Notebook §7g) |
| **does each scan's window bracket its own peak?** | per-scan argmax position vs the window edges | one fixed rocking window reused across a raster while θ_B drifts biases widths and integrals, and manufactures apparent two-population structure (Notebook §5l) |
| **was it reduced before, and by whom?** | the deposit's own analysis scripts | read them before comparing two channels or reporting any discrepancy — the correction you think is missing is often already there (rule 11) |

**Do not derive anything from a folder name.** The archived ID06 set's two reflections
(111, 002) sat at 2θ = 67.5° and 14.2° — different magnification and FOV — a fact you only
learn by reading the geometry, and the fact that kills the tensor (§3, Notebook §2).

**If someone else reduced this data, read their pipeline before you re-analyse it.** A single
preprocessing constant upstream of a comparison can be symmetric in pixels and wildly
asymmetric in signal — in one archive a column clip discarded 15× more of one channel's
in-aperture signal than the other's and moved ~20 % of the published labels. The same reading
prevents the opposite error: reporting a correction as absent when that pipeline already
applies it (rule 11, Notebook §7d). Where the reduction belongs to a collaborator, a
discrepancy is a **halt condition**, not just a technical finding.

**Is the scan still being written?** Count the files twice, ~120 s apart. Never reduce a
scan that is still growing.

---

## 0c. Already reduced? Check before recomputing

`midas_dfxm` is a library, so "already processed" means a previous script left artifacts in
your work directory — typically `.npz` files of per-pixel centre-of-mass maps (the ID06
campaign wrote `com_111.npz`, `com_002.npz`) or a saved orientation/strain field. Their
presence means a reduction ran **at some background/threshold, not necessarily yours**.

Before reusing one, confirm the two things that silently corrupt it:

| check | why |
|---|---|
| was the background subtracted before the moment? | a pedestal-dominated map is smooth and plausible but ~67× low (§2, Notebook §1a) |
| which reflection / geometry produced it? | maps from different reflections are on different grids and cannot be overlaid (§3) |

**After changing the background or threshold, regenerate the map** — do not trust an
inherited `.npz` whose reduction settings you cannot name.

---

## 1. Environment

See the spine §0 for the install gate and where to run. In short: `pip install
"midas-dfxm>=0.3.2"` (and `darling` for ESRF frame reduction); run the import gate and read
its output; on the Mac use the project env (`midas_env`), on an APS host the shared env by
full path. Outputs in a project/gdata directory you own — **never `/tmp`**.

DFXM reductions are not GPU-bound (moment analysis is cheap); the dynamical Takagi–Taupin
forward and the capability inverses (§4) benefit from a GPU but run on CPU. Nothing here
requires CUDA.
