# SURVEY — <dataset name>

> Fill this in **before promising anything** (phase-0-survey.md §0b). Every value read from
> the file/metadata, never from a folder or file name (README rule 10). Copy this file into
> your work directory, rename it `SURVEY.md`, and delete the guidance in angle brackets.

**Date surveyed:** <YYYY-MM-DD>  ·  **Surveyed by:** <you>
**Scan folder (absolute):** `<path>`
**Beamline:** <ESRF ID06-HXM | archived ID03 | other — if 6-ID-C, STOP: gated (README scope)>

---

## What is here (per acquisition)

| field | value | source file / key | notes |
|---|---|---|---|
| scan kind | <mosaicity (rock/roll) / strain (θ,energy) / multi-reflection set> | | decides the path |
| reflection(s) | <e.g. 002, 111> | <the aligned-Bragg metadata, NOT the filename> | |
| energy / wavelength | <keV / Å> | | cross-checked ≥2 ways |
| effective pixel size (sample) | <µm/px> | <objective magnification record> | differs per reflection |
| frame count + step | <N frames, Δ per step> | | too few points flattens the moment |
| background / dark present? | <yes → path / NO> | | **must subtract before any moment (§2); if NO, stop** |
| co-registration metadata? | <yes / NO> | | **absence = §3 halt for a tensor** |
| ground truth? | <almost always NO> | | if NO, accuracy = injection-recovery, not round-trip (§2d) |
| scan still being written? | <no — counted twice, 120 s apart> | | never reduce a growing scan |
| **flux / monitor column?** | <yes → column name / NO> | <the motor metadata, column by column> | **absence = halt** for any intensity comparison across separately-acquired groups (rule 16). The frame total is **not** a monitor on a rocking scan |
| **magnification provenance** | <value, and the ≥2 routes that agree> | <the optical record — NOT a script constant> | sets every µm/px; a factor ~2 error is the common one (rule 15). A focus check does **not** verify scale |
| **detector type + measured gain** | <photon-counting / integrating; `var = a·y + b`> | <detector record, then photon transfer on pedestal-subtracted frames> | absolute χ²/dof and σ scale as 1/gain (rule 13). Measure **per detector**; never reuse another's |
| **sampling, per channel** | <step and µm/px for *each* reflection> | | a weak channel is often far coarser, so the best map can be blind to the physics of interest (Notebook §7g) |
| **does each scan's window bracket its own peak?** | <yes / N of M truncated> | <per-scan argmax vs window edges> | one fixed window reused across a raster while θ_B drifts biases widths and integrals unequally between channels (Notebook §5l) |
| **prior reduction — whose, and read?** | <none / path to their scripts, read: yes> | | read them before comparing channels or reporting a discrepancy; the correction is often already there (rule 11) |

---

## Configuration (from phase-1-configure.md — fill after §1)

**One row per reflection.** Λ ∝ 1/|F|, so a weak satellite and its strong parent can sit in
different regimes in the same crystal — never classify once per sample (§1b, Notebook §7g).

| quantity | value | call |
|---|---|---|
| θ_B / 2θ | | `bragg_angle_deg` / `bragg_two_theta_deg` |
| extinction length Λ (µm) | <per reflection> | `extinction_length` |
| t / Λ regime | <thin ≲0.15 / marginal / thick ≳0.3, per reflection> | governs what §4 may claim |
| coherent block size used for t | <NOT the mosaic width> | mosaic spread ≠ coherent block size (Notebook §5k) |
| Im χ_h (absorption) | | `susceptibility_fourier` |
| refraction gauge ε_ref | <χ₀ᵣ/(2sin²θ_B)> | a reference offset, NOT a per-pixel strain (§4a) |
| resolution widths (anisotropic) | | `poulsen_resolution_widths` |

## Reduction health (from phase-2-reduce.md — fill after §2)

| quantity | value | healthy | why |
|---|---|---|---|
| pedestal share of centroid weight | | subtract regardless | ≳95 % dilutes the moment ~1/(1−f_ped) (§2a) |
| measured gain, this detector | <`var = a·y + b`> | quote it, don't assume 1 | absolute χ²/dof scales as 1/gain (§2a′) |
| r(background level, rocking curve) | | \|r\| ≲ 0.3 | ≳0.9 means a θ-dependent scalar → widths biased, centroid not (§2a) |
| kernel vs downsampled ROI | <kernel px / ROI px> | kernel < ROI | a kernel exceeding the ROI degenerates to a scalar (§2a) |
| **per-pixel** rocking FWHM | <argmax-local, contiguous crossings> | — | never infer sampling from an integrated/published width (§2f) |
| points per FWHM, per-pixel | | ≳12 for a model-selection test | below it, moment statistics measure broadness (§2f) |
| injection-recovery gain | | 0.9998–1.0000 observed | physical accuracy — not a round-trip (§2d) |

---

## Halt-condition check (README STOP table)

- [ ] not 6-ID-C (or confirmed OK to proceed on a gated instrument)
- [ ] background available to subtract before any moment
- [ ] if a tensor is asked for: co-registration metadata exists (else report the wall)
- [ ] any uniform ~100s-µε strain offset understood as refraction gauge, not a field
- [ ] any strain claim is inside t ≲ 0.3 Λ, or uses the dynamical forward
- [ ] if intensities are compared across separately-acquired groups: a flux monitor exists
- [ ] µm/px traced to an optical record, not to a script constant
- [ ] before reporting a discrepancy with someone else's reduction: their scripts read
- [ ] if resolution is worse than the instrument's best and a correction is wanted: a vibration
      **spectrum** exists (an amplitude cannot answer it)
- [ ] every control run can state what result would have refuted it

## Open questions / what would unblock the next step

<one line each>
