# MIDAS Reconstruction Reports

**Purpose.** Turn any MIDAS reconstruction into a single self-contained HTML report —
grain/orientation maps, per-grain error maps, residual diagnostics, strain, and a d0
calibration — plus an *honest* interpretation that separates what is trustworthy from
what is a fixable systematic.

This document is written for an **agent** to follow. The heavy lifting is done by
`utils/midas_ff_report.py`; this file explains how to run it, how to read what it
produces, and how to extend it.

---

## 1. Quick start

```bash
# Run WHERE THE DATA LIVES (processgrains_diagnostics.h5 can be hundreds of MB).
source /path/to/conda/etc/profile.d/conda.sh && conda activate <midas_env>
python utils/midas_ff_report.py <RUN_DIR>/LayerNr_1 \
    --material "SrTiO3 (STO)" \
    --title    "SrTiO₃ — Far-Field HEDM reconstruction" \
    --out      <RUN_DIR>/report.html
# optional, enables the d0 → stress-bias number:
#   --c11 317.2 --c12 102.5      (GPa, single-crystal stiffness)
```

Then **publish** `report.html` with the Artifact tool (favicon `🔬`). Re-publishing the
same file path keeps the same URL, so reports can be updated in place.

`RUN_DIR/LayerNr_N` must contain `Grains.csv`; `processgrains_diagnostics.h5` is optional
but is where most of the value lives — without it you lose the residual plate and most
auto-findings.

---

## 2. Inputs the generator reads

### `Grains.csv`
Comment header carries `%NumGrains`, `%SpaceGroup`, `%Lattice Parameter` — the generator
**auto-detects** material symmetry and the nominal lattice from these, so you rarely pass
them by hand. Columns (47):

```
ID  O11..O33  X Y Z  a b c alpha beta gamma  DiffPos DiffOme DiffAngle
GrainRadius  Confidence  eFab11..eFab33  eKen11..eKen33  RMSErrorStrain  PhaseNr  Eul0-2
```

- Positions `X Y Z` in **µm**; Euler angles in **radians**.
- `eFab*` = sample-frame strain, `eKen*` = crystal-frame strain — **already in microstrain**.
  (Do *not* multiply by 1e6. A common bug: doing so yields absurd 1e8 µε.)
- `Confidence` is completeness (fraction of expected reflections observed).

### `processgrains_diagnostics.h5`
```
residuals/spot_table   (N,11)  grain_idx, spot_id, ring_nr, eta_deg, dy_um, dz_um,
                               drad_um, dtan_um, dome_deg, internal_angle_deg, r_exp_um
residuals/grain_*      per-grain medians (drad, dtan, dome, dy, dz, internal_angle, n_spots)
residuals/ring_*       per-ring    (ring_nr, n_spots, med_drad_um, mad_drad_um, drad_ppm)
residuals/eta_*        per-η-bin   (bin_lo_deg, med_drad_um, med_dtan_um, med_dome_deg, n_spots)
residuals/overall_*    scalars     (med/mad of drad, dtan, dome, dy, dz, internal_angle)
diagnostics/*          cluster_sizes, n_resolved_hkls, n_majority_hkls, …
```

---

## 3. What it produces

Five plates, a stat-tile summary, auto-derived findings, and an improvement roadmap:

| # | Plate | Content |
|---|---|---|
| 01 | Grain centroid maps | X–Y / X–Z / Y–Z scatter, IPF-Z colored (cubic), sized by radius |
| 02 | Per-grain error maps | completeness, spots/grain, internal angle, radial residual, DiffPos, RMS strain error |
| 03 | Residual diagnostics | per-ring Δr ppm, residual-vs-η, spot Δrad–Δtan density, internal-angle & Δω histograms |
| 04 | Strain / lattice / quality | hydrostatic map, strain-component boxplots, lattice-a, completeness, spots, DiffPos |
| 05 | d0 calibration | before/after hydrostatic strain, recovered a₀ (cubic only, needs `midas_stress`) |

Findings and roadmap items are **computed from the data**, not hardcoded — see §4.

---

## 4. Diagnosis reference — symptom → cause → lever

This is the analytical core. Apply it whenever reading a report.

### Δradial shows a sinusoid in azimuth η
Fit `Δrad ≈ A·cos(η) + B·sin(η) + C`. Then **discriminate by ring**:

- **Amplitude constant (in µm) across rings** → **beam-center offset**.
  A rigid detector-center shift produces the same µm displacement at every ring radius.
  Magnitude in pixels = amplitude / pixel size. Δtan shows the same amplitude 90° out of phase.
- **Amplitude grows with ring radius** → sample displacement / Lsd effect, not BC.

**Confirm it is global, not per-grain:** fit `(dy,dz)` per grain and compare mean vs
scatter. `mean ≫ std` ⇒ a common offset ⇒ detector geometry, fixable by recalibration.
`mean ≈ 0` with large scatter ⇒ genuine per-grain position spread, *not* a bug.

*Lever:* recalibrate BC/Lsd/tilts/distortion against a powder calibrant (see §5), re-index.

### Per-ring Δr/r (ppm) trend
- **Constant ppm across rings** → **Lsd** error (`δR/R = δLsd/Lsd`, ring-independent).
- **ppm grows with 2θ** → **reference lattice or wavelength** (d0) error.
- A large *range* across rings (≳200 ppm) ⇒ refine Lsd and/or the reference scale.

### Large DiffPos / |Δy| but small angular residuals
Not a defect. An extended beam (line or box) poorly constrains position *along the beam*.
Report it as a geometry property; orientation stays trustworthy.

### Grains placed outside the illuminated beam ("Z tail")
A Z distribution with a sharp core plus a broad tail. **Do not assume it is physical, and do
not assume it is divergence-to-bound** — test both:

1. **Bound test.** Positions are bounded by `Hbeam/2` (and `Rsample`). Divergence-to-bound
   leaves a *pile-up at the bound*. If the outer shell holds ~0% of grains, the bound is not
   being hit and divergence-to-bound is refuted.
2. **Residual test (decisive).** Correlate fitted `Z` against `residuals/grain_med_dz_um`.
   - `corr ≈ 0`, residual flat vs Z ⇒ the Z values are supported by the spots — physical.
   - **Strong negative corr** (e.g. −0.8), with core grains at ~0 residual and tail grains
     carrying residuals pointing *back toward the beam centre* ⇒ the spots contradict the
     assigned Z. The tail is a **fitting artifact**.
3. **Rule out geometry**: compare *ring composition* of core vs tail. If identical, a
   ring-dependent tilt/distortion error is not the cause. Also check η coverage (`frac_vert`).

*Lever:* set `Hbeam`/`BeamThickness` to the **true per-layer beam**, not the full sample
height (a 10-layer 100 µm scan often carries `Hbeam 1000`, letting Z roam ±500 µm). Grains
outside the beam cannot diffract, so this is a physical prior, not a fudge. Then re-check that
the dz residual stays flat vs Z.

### Bimodal DiffPos / DiffAngle — a discrete solver branch
Per-grain histograms (not maps) expose this. If `DiffPos` and `DiffAngle` are both bimodal and
the two splits agree (≈94% in one observed case), it is **one** population split.

**Discriminate smooth-geometric from algorithmic:** bin grains by radial distance
`r = hypot(X,Y)` from the rotation axis and histogram DiffPos *within each bin*.
- Modes move with r ⇒ a smooth geometric effect (position information ∝ ω-modulation
  amplitude, which vanishes on the axis).
- **Mode positions fixed across r, only the population fraction shifting** ⇒ a **discrete
  algorithmic branch**. Suspect the Friedel-pair position path (`UseFriedelPairs 1`)
  succeeding vs falling back.

Confirm by re-running a subset with `UseFriedelPairs 0`: if the bimodality collapses, the
Friedel path is the branch. Expect the bad branch to also carry inflated |Z|, internal angle,
`mad_dtan` and strain error — verify they co-move before blaming one cause.

### Bimodal completeness / spots-per-grain, split spatially
The illumination footprint (which part of the sample the beam actually covered), not a
reconstruction artifact. Check the spatial error maps to confirm the split is positional.

### Nonzero mean hydrostatic strain
A **d0 (strain-free reference)** error — see §6.

---

## 5. Geometry recalibration (powder)

`midas-autocalibrate` (package `midas_calibrate`) fits BC/Lsd/tilts/distortion to a powder
calibrant. Critical points:

- It needs a **2-D image array** — the CLI's `--image` only accepts `.tif/.tiff/.npy/.h5`
  (h5 = first key). For a multi-frame `.vrx.h5`, call the orchestrator from a driver:
  mean the frames, pass the embedded `exchange/data_dark` as `dark`, then
  `midas_calibrate.orchestrator.autocalibrate(params, image, dark=dark)`.
- The params file needs the **cake/integration keys** or it divides by zero:
  `RMin RMax RBinSize EtaMin EtaMax EtaBinSize` (plus `DoSmoothing DoPeakFit MultiplePeaks PeakFitMode`).
  Size `RMax` (px) to cover the outermost ring you actually index.
- **RhoD must match the reconstruction.** Distortion is `R_norm = rad_um / RhoD`, so the
  `p0..p10` coefficients are only meaningful with the RhoD they were fit against. Calibrate
  with the *same* RhoD your recon param uses, then transplant `{Lsd, BC, tx, ty, tz, p0..p10, RhoD}`
  as one block. Mixing a new `p` set with an old RhoD silently corrupts the distortion.
- Powder cannot constrain `tx` (rotation about the beam) — keep it fixed and refine it from
  grains if needed.
- Target: powder mean strain **< 100 µε** (hard threshold). Compare trimmed-to-trimmed.
- **Run detached** (`nohup setsid … > log 2>&1 < /dev/null &`) writing to a *file*. A
  verbose job piped to `tail` over ssh dies of SIGPIPE when the connection drops.

---

## 6. d0 (strain-free reference) with `midas_stress`

For a **cubic, free-standing** (unloaded) polycrystal the equilibrium condition reduces
exactly to ⟨ε_hydro⟩_V = 0 — no stiffness needed. Any nonzero volume-averaged hydrostatic
strain *is* the d0 error.

```python
import midas_stress as ms
r = ms.recover_d0_cubic_free_standing(lattice_params,      # (N,6) a b c α β γ
                                      assumed_reference,   # (6,) nominal
                                      volumes=radius**3,
                                      confidences=conf, min_confidence=0.5)
a0  = r["reference_recovered"][0]
eps = r["eps_iso"]                     # fractional isotropic error
strain = ms.lattice_params_to_strain(latc, new_ref)   # Green-Lagrange, matches C
```

Key properties to state in any report:
- The correction is **purely isotropic** → **deviatoric strain is unchanged**; only the
  baseline moves. It fixes *bias*, never *scatter*.
- Stress impact = `eps_iso × 3K`. Use `ms.d0_sensitivity(C11=…, C12=…)` →
  `sensitivity_MPa_per_ppm`. This is often large (hundreds of MPa) and is the headline.
- Non-cubic: use `ms.recover_d0` (needs stiffness + orientations) or `ms.correct_d0`.

---

## 7. Design system (for hand-built or extended reports)

- **Figures**: paper `#f7f6f3`, ink `#131619`, accents teal `#0e7c86` + copper `#c07a3e`,
  muted `#6b6862`. Perceptually-uniform colormaps: `cividis`/`viridis` sequential,
  `RdBu_r` diverging (always `TwoSlopeNorm` centred on 0 for strain), cubic-IPF for orientation.
- **Page**: theme-aware via `:root` CSS custom properties + `prefers-color-scheme` **and**
  `:root[data-theme=…]` overrides. System font stacks only — the Artifact CSP blocks font CDNs.
- **Figure cards are always-light "plates"** in both themes so plots read correctly.
- Wide content gets `overflow-x:auto`; numbers use `font-variant-numeric: tabular-nums`.
- Everything inlined (`data:image/png;base64,…`) — one file, no external requests.

---

## 8. Writing the interpretation

The framing is the point. Rules:

1. **Separate solid from fixable.** Name explicitly what is trustworthy (usually grain count,
   orientation, in-plane position, lattice) versus what is a systematic with a known lever.
2. **Every claim carries its number**, with units, straight from the data.
3. **Never dismiss a systematic as noise** without the discriminating test from §4.
4. **Distinguish bias from scatter.** d0 removes bias; only geometry/ring-coverage reduces scatter.
5. **State caveats plainly** — e.g. "per-grain strain is indicative on 4-ring line-beam data".
6. No fabricated numbers, ever. If a diagnostic is missing, say it is missing.

---

## 9. Extending

- **Non-cubic**: IPF coloring is cubic-only; grains render in a flat accent instead. Add a
  symmetry-aware colorer (via `midas_stress.orientation`) for hexagonal/tetragonal.
- **Multi-phase**: `PhaseNr` column exists; split per phase and emit one report per phase.
- **PF / NF reconstructions**: those produce `voxel_grid.csv` / `.mic` (voxel maps) rather
  than grain centroids — the spatial plates need reworking into true voxel maps, but the
  residual/strain/d0 machinery carries over unchanged.
- **Comparisons** (before/after a recalibration): run the generator on both result dirs and
  place the tiles side by side; keep the same colour scales.

---

## 10. Gotchas

- `eFab`/`eKen` are already µε (§2).
- numpy ≥2: `np.ptp(x)` not `x.ptp()`; use `np.nanmedian` (zero-spot grains carry NaN).
- `paramstest.txt` values may carry trailing `;` (C format) — strip before `float()`.
- The HTML template must not be built with `str.format` (CSS braces collide) — use
  `string.Template` (`$name`), which is what the generator does.
- Run remote work detached; never leave a long job depending on a live ssh pipe.
- Write outputs next to the run, not `/tmp`.
