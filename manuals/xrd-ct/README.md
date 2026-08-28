# XRD-CT — per-voxel phase, strain and texture from a diffraction-tomography scan

**Use this doc to start a fresh session on a dataset this pipeline has never seen.**
Paste it in together with `LAB_NOTEBOOK.md`, then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # detector frames, or an integrated (R, eta) cake
Metadata / geom: <ABSOLUTE PATH>   # calibration + translations/omega, or "find it"
Sample material: <e.g. CeO2 / hcp Ti / unknown -- tell me from the data>
Goal:            phase map | per-voxel strain | per-voxel texture (ODF)
```

**Path conventions.** `$MIDAS` is the root of whichever MIDAS checkout you are working in
(on a beamline host, `~s1iduser/opt/MIDAS_canonical`). **`$ANALYSIS` is a campaign working
directory that is NOT in this repo** — the harnesses that produced numbers in
`LAB_NOTEBOOK.md` are local, deliberately unversioned analysis scripts, so a `$ANALYSIS/...`
path is *provenance, not a link*: it names the script a number came from, and promises
nothing about reaching it from where you are sitting.

**Scope.** Powder-like **XRD-CT**: the sample is translated across a beam at each of many
ω rotations, rings are continuous and integrated azimuthally, and the reconstruction is a
per-voxel diffraction pattern on a 2-D voxel grid. Handled through `midas_dt`. If the rings
break into **discrete spots** the sample is coarse-grained and this is the wrong tool — that
is scanning-3DXRD, doc set `pf-hedm`. The dividing line is operational: continuous at the
working (R, η) bin size, or not. **Check it in phase 0 before any recipe here applies.**

<!-- The scope gate is load-bearing. A spotty ring fed to an azimuthal-integration
     pipeline produces a per-voxel "texture" that is a crystallite-count fluctuation.
     Muerer's MAD filter (phase 2) suppresses the worst of it but cannot rescue a
     genuinely coarse-grained sample. -->

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order, hard rules, halt conditions | always |
| `phase-0-survey.md` | is it XRD-CT? translations, ω, format, ring continuity | first |
| `phase-1-geometry.md` | calibration, the distance trap, ω sign, the R↔2θ map | before extracting |
| `phase-1a-reduce.md` | **frames → cake**: unseeded calibration, integration, caching, axis order | if starting from raw frames |
| `phase-1b-reconstruct.md` | **cake → voxels**: sinogram, rotation axis, `RECON_SIGN`, the three branches | if starting from raw frames |
| `phase-2-extract.md` | background, ring vetting, per-azimuth area and centroid | always |
| `phase-3-strain.md` | per-voxel / per-azimuth strain from centroids — **the tractable deliverable** | if strain is the goal |
| `phase-4-texture.md` | per-voxel ODF: the operator, symmetry, the ladder, the control | if texture is the goal |
| `phase-5-report.md` | what to state, what to label provisional, provenance | at the end |
| `ENVELOPE.md` | what this measurement **can** determine, and what it cannot | **before promising an answer** |
| `DIAGNOSIS.md` | symptom → discriminating test → cause → lever | **when something looks wrong** |
| `BEAMLINES.md` | per-beamline reach, formats, conventions that cannot be recovered later | **before touching a new beamline's data** |
| `RUNBOOK.md` | a from-scratch run worked end-to-end on real 11-ID-C CeO₂ | on resume, or as the shape of a run |
| `LAB_NOTEBOOK.md` | evidence, the ledger, **four refuted, one DOWNGRADED, one SUPERSEDED (cause since found), three withdrawn** | before re-investigating |

## STOP — read this before touching anything

### The one fact that governs this technique

**Area is a difference; centroid is a ratio.**

Every azimuthal quantity in XRD-CT is one or the other, and at low peak-to-background they
behave completely differently:

* **Area** (→ texture) is a small difference of large numbers. It inherits the background
  model's error in full, and that error is independent per frame.
* **Centroid** (→ strain) is an intensity-weighted ratio. A slowly varying background largely
  cancels.

Measured on a synthetic at 2 % contrast with no planted azimuthal structure: **area scatter
36 %, centroid scatter 0.85 %** (`tests/test_azimuthal.py`). Measured on the real DAC Ti
scan, where peaks sat 0.5–17 % above background: **strain consistent across six independent
reflections; texture bounded near zero.** Same frames, same windows, same code.

So: **measure the contrast first, and let it decide what you promise.** A scan that cannot
support a texture map can still give an excellent strain map.

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** These failures all finish and look
right:

| Condition | Why you cannot decide it yourself |
|---|---|
| The **sample-to-detector distance** is not confirmed against the data itself | On an 11-ID-C CeO₂ scan the metadata said 1600 mm and the beamline calibration 1579.5 mm; the data required **1632 mm**. Both stored values were wrong. A wrong distance is an *absolute* strain scale error that leaves every relative map looking fine. |
| The **ω sign** is not confirmed against the encoder | Flips the reconstruction. At 1-ID the aero stage needs **every ω negated**; nothing downstream complains. **If the sample is near-symmetric the sign may be genuinely undeterminable** (CeO₂: both signs give identical diameter and CV) — then record it as *undetermined* and report the map as **mirror-ambiguous**. Do not pick one to clear this gate. |
| A ring's **peak-to-background** has not been measured | It decides whether texture is answerable at all. Reporting a texture map without it is the single most common way to publish extraction error. |
| Rings have not been **vetted for singlets** | A sub-pixel ring assignment matches a doublet as one line. hcp Ti (101) is a doublet (maxima 381.6 and 393.6 px) and was fitted as one ring for four analyses. |
| A texture claim has not been through the **positive control** | `scripts/odf_positive_control.py`. Without it, a null is uninterpretable — you cannot tell "no texture" from "cannot see texture". |
| Two ring centres are **closer than their windows** | They contaminate each other's background and area while each still passing a within-window singlet test. CeO₂ (331) and (420) sit 26.7 px apart and read a 29 px "FWHM" against 2–4 px for every clean ring. A gap check is a **separate** test from a multiplet check. |
| The **cake axis order** has not been verified by collapsing each axis | (η, R) at 11-ID-C, (R, η) at 1-ID. Both reshape cleanly, so a swap gives a transposed array and no error. Two lines to check. |
| The **DAC loading geometry** (axial vs radial) is unknown, and texture is the goal | The uniaxial model fixes the fibre along the rotation axis. If the cell was loaded radially, the model cannot fit it *by construction*, and a null means nothing. This is a fact about the experiment, not the data. |

When you halt, say which row fired, what you measured, and what you would need to proceed.
Finish everything not blocked by it first — **strain is usually not blocked by any of this.**

### Hard rules

1. **Look at a radial profile before anything else.** Four analyses on the DAC Ti scan were
   invalidated by not doing this. The profile immediately showed the background dominated
   (contrast 1.17×) and that α(101) was a doublet. One plot, five minutes, four wasted
   analyses.

2. **Background comes from ring-FREE radii, interpolated across the peaks.** A rolling low
   percentile over blocks comparable to the peak width sits on the peak *flank* and biases
   the background up exactly where the peak is. At 1 % contrast a 1 % background error is a
   ~20 % area error. `midas_dt.azimuthal.background_from_ring_free`, **not**
   `rings.rolling_baseline`, which is for ring *finding*.

3. **The background varies with both R and η**, so estimate it per azimuth. Compton from
   anvils or a furnace falls with angle; absorption varies with path length. A radial-only
   background leaves an η pattern that looks exactly like texture — and did.

4. **Gate on per-azimuth SNR, never on a ring's median.** Gating on the median lets dead
   azimuths through, and any peak-to-peak spread over η is a max-minus-min that a handful of
   them dominate. That is how one reflection reported 107,410 µε and another 638,282 µε
   (11 % and 64 % strain — nonsense).

5. **Only even harmonic orders are measurable, for every scan design.** `Y_l^n(-h) =
   (-1)^l Y_l^n(h)` and diffraction cannot separate `h` from `-h`. Odd `l` is the classical
   ghost subspace and **no amount of extra data recovers it** — only a positivity
   constraint does. `SymGSH.ghost_dimension()` reports its size; quote it rather than
   hiding it.

6. **Symmetry is the Laue group, not the point group.** Friedel makes the measurement
   centrosymmetric, so improper operations map to `-R` rather than being discarded.
   Discarding them under-symmetrises the **73** space groups that have improper operations
   but no inversion centre. Use `midas_hkls.proper_rotations_from_space_group`; do not
   hand-roll a table.

7. **Never report a per-voxel map without its polynomial `r²`.** A field a smooth low-order
   polynomial explains is an instrument, absorption or geometry signature wearing a map's
   clothes. `midas_dt.odf_uniaxial.explained_by_polynomial`. This has retracted a result in
   this project **three times**.

8. **Choose λ by the discrepancy principle against a measured noise floor, never by
   L-curve** — and know what that buys. It fixes the *weight*, not the *prior*: three
   different priors reached the same residual floor, and the run with the **best** residual
   (0.0606) gave among the **worst** reconstructions (MAE 0.250 against 0.056).

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| Radial windows written in **bins** when the cake is binned finer than a pixel (0.25 px/bin is typical) | Window ~4× too narrow; the lost part is the tails, which is where area lives | phase-2 |
| Under differential stress the peak **moves with azimuth** (`d(ψ) = d₀[1 + (1−3cos²ψ)Q]`, Singh) | A fixed window converts movement into azimuth-dependent intensity: **fake texture**. `azimuthal.radial_half_correlation` goes NEGATIVE when the peak moves (measured −0.72 on a CeO₂ standard) | phase-2, DIAGNOSIS |
| Truncating the harmonic expansion at the operator's `L` when using kernels as a positivity basis | Non-negativity constrains the *full* function. A **sharp** kernel loses most of its amplitude (8° at L=6 keeps 5.8 %); a wide one does not | phase-4 |
| Comparing a cap-**averaged** Monte-Carlo pole figure against a **pointwise** model | Slope biased low on sharp features (0.969 vs 1.008), looks like a broken operator | phase-4 |
| Closing a symmetry group on **quaternions** | `q` and `-q` are the same rotation and the tie-break is unstable at `w = 0`, exactly where the 2-folds sit: 432 closes to **28** elements | phase-4 |
| Unpinned BLAS threads on a CPU fan-out | 15 processes drove one 96-core host to **load 437** with nothing finishing in 40 min. Always `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` | any batch run |
| `pkill -f "foo.py"` | Matches its own ssh command line and kills the session (exit 255). Use `pkill -f "[f]oo"` | any batch run |

## 0. Verify the install

```bash
$PY -c "import midas_dt; print('midas_dt', midas_dt.__version__)"
# texture path only -- these floors are for correctness, not features:
$PY -c "
import midas_dt, midas_hkls, scipy, numpy
print('midas_hkls', midas_hkls.__version__, '(need >=0.7.3 for point_group)')
print('scipy', scipy.__version__, '(need >=1.15 for sph_harm_y)')
print('numpy', numpy.__version__, '(need >=2.0 for trapezoid)')
from midas_dt import SymGSH, fit_uniaxial_ladder      # lazy import, will raise if short
from midas_hkls import proper_rotations_from_space_group
print('texture extra OK')"
```

`scipy>=1.15` is not cosmetic: the predecessor of `sph_harm_y` takes its arguments in the
**opposite order** (azimuth before polar). Swapping them gives a transposed pole figure that
is still smooth, still symmetric, and wrong.

**Reproduce the operator gates before trusting a texture number on new data:**

```bash
$PY -m pytest tests/test_gsh.py tests/test_texture_kernel.py -q     # ~30 s
$PY -m pytest tests/test_texture_kernel.py -q -m slow               # ~20 s, corr > 0.999
```

## 0a. THE ORDER

```
phase 0   survey        is it XRD-CT? format, translations, ω, ring continuity
phase 1   geometry      calibration; CONFIRM the distance against the data; ω sign; R↔2θ map
phase 1a  reduce        frames → (η, R) cake, cached once. VERIFY THE AXIS ORDER.
phase 1b  reconstruct   sinogram; rotation axis; RECON_SIGN; choose a branch; index rings
          ---- look at a radial profile. Do not skip. ----
phase 2   extract       background from ring-free R; vet singlets; per-η area + centroid + SNR
          ---- measure peak/background. It decides what is answerable. ----
phase 3   strain        centroids → d-spacing → per-voxel/per-azimuth strain
phase 4   texture       ONLY if the contrast supports it, and only after the positive control
phase 5   report        with provenance, and with the provisional labels intact
```

Phase 3 and phase 4 are **independent**. Strain does not wait for texture, and a texture null
does not invalidate strain. On the one real dataset taken through both, strain was the
deliverable and texture was bounded near zero.

## 1. Where things live

| Thing | Where |
|---|---|
| Reduction, sinograms, reconstruction, branches | `midas_dt` — procedure in `phase-1a` / `phase-1b`; design rationale in the package README |
| Per-azimuth extraction | `midas_dt.azimuthal` |
| Per-voxel deviatoric strain tensor (direct inversion) | `midas_dt.tensor_strain` |
| The pole-figure operator, symmetry-adapted GSH | `midas_dt.gsh` |
| Orientation kernels (simulation / validation only) | `midas_dt.texture_kernel` |
| Uniaxial squared-modulus ODF + the model ladder | `midas_dt.odf_uniaxial` |
| Proper rotation groups, all 230 space groups | `midas_hkls.point_group` |
| **The positive control** | `packages/midas_dt/scripts/odf_positive_control.py` |
| Third-party operator validation | `packages/midas_dt/scripts/validate_gsh_vs_textom.py` |

## 2. Done means

* The **distance** is confirmed against the data, not taken from metadata.
* Every ring used is a **vetted singlet** with a stated peak/background and per-azimuth SNR.
* Strain is reported with its **spread** (inter-percentile range and MAD), not only a mean,
  and with the number of live azimuths.
* Any per-voxel map carries its **polynomial `r²`**.
* Any texture claim carries a **positive-control result at the measured contrast**, and the
  control's own softener (Poisson noise only) travels with it.
* Provisional results are **labelled provisional in the text that leaves the session**, not
  only in the working notes.

## Sibling doc sets

`manuals/ff-hedm/` (far-field HEDM, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/pf-hedm/` (scanning 3DXRD — **the right doc set if your rings are
spotty**, skill `pf-hedm`), `manuals/dfxm/` (dark-field X-ray microscopy, skill `dfxm`), and
in the LaueMatching repository the `laue` skill.
