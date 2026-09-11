# PDF / total-scattering runbook — area-detector frames to S(Q), G(r) and a model

**Use this to reduce high-energy total-scattering frames to a pair distribution function and
fit a model to it.** Give:

```
Data folder:   <ABSOLUTE PATH>      # sample, calibrant, empty container, air frames
Samples:       <composition, density or packing, container geometry — per frame>
Energy or λ:   <keV or Å — or "find it": the tabulated K edge of the mono foil>
```

Everything else is worked out. **Sample facts are inputs you state, not outputs you tune**
(hard rule 15). If they are unknown, say so in the report.

**Two companions.** This spine carries the gates and the order. The reduction code is in
[`phase-5-sq-gr.md`](phase-5-sq-gr.md), and the modelling recipes are in
[`phase-6-model.md`](phase-6-model.md). [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) holds the evidence,
including the claims from an earlier analysis of the same beamtime that did not survive.

Citations are `path:line` **relative to the repository root**. `$ANALYSIS/...` is provenance
(see the end of this file).

---

## Order of operations — not optional

§0 scope gate → §1 install gate → §2 survey → §3 geometry hand-off → §4 integrate with
corrections → [§5 reduce to S(Q), G(r)](phase-5-sq-gr.md) → §6 verify → [§7 model](phase-6-model.md).

Each gate invalidates everything after it, and each failure downstream of it **still produces a
plausible curve**:
- an uncorrected profile gives a G(r) with a clean-looking low-r spike;
- a missing `RhoD` gives an empty or absurd integration;
- an unverified Q scale shifts every peak by hundreds of ppm.

The halt conditions (§8) apply throughout, and [`HARD_RULES.md`](HARD_RULES.md) applies to every
phase.

## §0. Scope gate — read before touching data

Everything here was measured on **one beamtime**:
- **Detector and beam:** 1-ID-E, Varex 2880² at 150 µm, 180-frame sums, Ta K edge 67.4164 keV
  (λ 0.183908 Å), Lsd 470 mm.
- **Coverage:** the beam centre sits **at the detector's left edge**, so azimuthal coverage is
  175°.
- **Frames:** CeO2, Ni powder in carbon black, liquid IPA, empty Kapton capillary, air.

[`ENVELOPE.md`](ENVELOPE.md) records what has and has not been exercised.

| you have | do |
|---|---|
| high-energy (≳ 60 keV) area-detector frames: sample, empty container, air, and a calibrant at the same distance | continue |
| no calibrant frame at the sample distance | **stop** — a geometry cannot be invented; see `calibrate-integrate` |
| no empty-container or air frame | **stop and ask** — container and background subtraction are impossible, and every S(Q) check will fail for that reason |
| a different detector or energy | continue, but re-derive every number: the *procedure* transfers, the *values* do not |
| spotty or single-crystal rings | **stop** — `ff-hedm` or `pf-hedm` |
| a time series (Δ-PDF against a baseline), SAXS/SANS joint refinement, anomalous or neutron data | **not exercised — stop and ask** |

## §1. Install gate

```bash
python -c "import midas_pdf, midas_integrate_v2, midas_calibrate_v2, midas_hkls; \
print(midas_pdf.__version__, midas_integrate_v2.__version__, midas_calibrate_v2.__version__, midas_hkls.__version__)"
```

Exercised with midas-pdf 0.3.0, midas-integrate-v2 0.7.1, midas-calibrate-v2 0.15.0 and
midas-hkls 0.11.0. **There is no behavioural floor check for this doc set yet.** The defects in
`HARD_RULES.md` rules 1, 2, 11, 12 and 13 were present in those versions. Re-check each one
against the cited line before assuming it is fixed, or still broken.

## §2. Survey — what is in the folder

Work out, do not ask:

- **What a frame is.** A sum of how many exposures, from the detector's frame counter. On the
  reference data every file held a 180-frame sum; the pixel ceiling was 180 × 65534.
- **Darks.** Match each dark to its own file. A dark taken right after an exposure carries lag
  of that exposure: ~0.7–0.9 % on the reference Varex (`$ANALYSIS/out/00c_dark_lag.json`).
- **Incident flux.** Take it from the ion-chamber scalers (net counts), **not** from the
  attenuator PV (hard rule 5).
- **Energy.** The tabulated K edge of the monochromator foil
  (`manuals/calibrate-integrate/HARD_RULES.md` rule 9).
- **The detector's noise, measured, not assumed.** On the reference Varex the per-pixel σ of
  data − dark was ≈ 2400–2700 ADU, nearly independent of signal, so √S on ADU is wrong by an
  order of magnitude (hard rule 6).
- **A mask.** Dead pixels (exact zeros present in every frame), pixels whose dark is at the
  ceiling, line defects and border columns. Add an opaque-shadow mask: pixel ÷ median of its own
  R bin on the corrected calibrant frame, median-filtered, below 0.6, dilated. The shadow mask
  needs the geometry, so it is built in §4.
- **Sample facts, as stated inputs:** composition, bulk density, packing fraction, container
  material and radii. They go into absorption, multiple scattering and the low-r check. If a
  powder is dispersed in a matrix, the bulk density is wrong for the low-r slope (hard rule 15).

## §3. Geometry hand-off — from `calibrate-integrate`, carried across whole

Calibrate with the `calibrate-integrate` doc set: from scratch, λ fixed, verified against the raw
rings. On a dense calibrant at high energy, the recipe that passed was:

```python
import math
from midas_calibrate_v2 import calibrate
from midas_distortion.rhod import resolve_rho_d_um

res = calibrate(img, wavelength=LAMBDA, pxY=PX, dark=dark, mask=mask, calibrant="CeO2",
                output_dir=OUT, min_ring_separation_px=11.0, refine_distortion="none", n_iter=10)
assert res.seed_method != "fallback"                     # halt H2

# to_integration_spec() carries neither RhoD nor the residual map (hard rule 2)
by, bz = float(res.seed_BC_y), float(res.seed_BC_z)      # the SEED centre, as calibrate() used
ny, nz = int(res.NrPixelsY), int(res.NrPixelsZ)
rho_px = math.hypot(max(by, ny - 1 - by), max(bz, nz - 1 - bz))
rho_um, _how = resolve_rho_d_um(rho_px, ny, nz, by, bz, float(res.pxY))
spec = res.to_integration_spec(RMin=20.0, RMax=R_CORNER_PX, RBinSize=1.0,
                               EtaMin=-180.0, EtaMax=180.0, EtaBinSize=1.0, RhoD=rho_um)
spec.ResidualCorrectionMap = str(res.residual_corr_bin_path)
```

**Why those arguments.**
- At 470 mm and 67.4 keV, CeO2 rings beyond R ≈ 1100 px sit closer than the E-step window.
- Beyond ≈ 1300 px they carry almost no intensity: the E-step SNR stays below 1 at every η bin
  from 5° to 20°.
- The package defaults (n_iter 4, full distortion) oscillated and left 606 µε in-loop.
- The accepted run left 50.3 µε ring-mean RMS to Q 17.2 **with** the map and 264.6 without
  (Lab Notebook §3).

**Then verify the Q axis to the Q_max you will transform to (hard rule 4):**
- Use crest positions of **isolated** rings (≥ 12 px from both neighbours), off the raw profile.
- Build the ring list with a large `n_rings`: `_ring_table(n_rings=120)` stops at Q ≈ 17.2 on
  CeO2 at 67 keV.
- If the calibrant has no measurable rings between the last one and Q_max, the Q scale there is
  **unmeasured**. Write that into every downstream number.

## §4. Integrate with the corrections the integrator does not apply

```python
import numpy as np, torch
from midas_calibrate_v2.io.readers import read_image
from midas_calibrate_v2.forward.geometry import build_tilt_matrix
from midas_integrate_v2.forward.pixels import pixel_to_REta_from_spec
from midas_integrate_v2.corrections.intensity import PolarizationCorrection, SolidAngleCorrection
from midas_integrate_v2.binning import HardBinGeometry, integrate_hard_with_variance
from midas_integrate_v2.pdf import R_px_to_Q

dt = torch.float64
Z, Y = torch.meshgrid(torch.arange(spec.NrPixelsZ, dtype=dt), torch.arange(spec.NrPixelsY, dtype=dt),
                      indexing="ij")
px = torch.as_tensor(spec.pxY, dtype=dt)
with torch.no_grad():
    pix = pixel_to_REta_from_spec(Y, Z, spec)
    sa = SolidAngleCorrection()(Y.reshape(-1), Z.reshape(-1), Ycen=spec.BC_y, Zcen=spec.BC_z,
                                TRs=build_tilt_matrix(spec.tx, spec.ty, spec.tz), Lsd=spec.Lsd,
                                pxY=px, pxZ=torch.as_tensor(spec.pxZ, dtype=dt))
    pf = PolarizationCorrection(pol_fraction=float(spec.PolarizationFraction),
                                pol_plane_eta_deg=float(spec.PolarizationPlaneEtaDeg))(
        pix.R_px.reshape(-1), pix.eta_deg.reshape(-1), Lsd=spec.Lsd, px=px)
corr = (sa * pf).reshape(spec.NrPixelsZ, spec.NrPixelsY)      # divided out: I/c, var/c²

geom = HardBinGeometry.from_spec(spec, mask=mask)               # True = excluded
frame = read_image(FILE, data_loc="exchange/data") - read_image(FILE, data_loc="exchange/data_dark")
mean2d, sigma2d = integrate_hard_with_variance(torch.as_tensor(frame), geom,
                                               variance_image=VAR, correction=corr,
                                               error_model="poisson")
ones = torch.ones_like(corr)
_, s_one = integrate_hard_with_variance(ones, geom, variance_image=ones)
npix2d = 1.0 / s_one ** 2                                       # pixel count per (eta, R) bin
```

- **Binning.** Hard binning at 1° × 1 px. Collapse η later, weighting by pixel count, so that an
  η wedge or an interleaved sliver set can be cut from the same cake.
- **`VAR`.** A measured per-pixel variance model (offset + gain × signal), passed in with
  `error_model="poisson"` so the integrator propagates it rather than √S. `"azimuthal"` uses
  in-bin scatter instead.
- **Before quoting any σ,** calibrate it with interleaved 1° slivers (hard rule 6).
- **Polarization plane.** Check it once on the uncorrected cake: the ring-intensity minimum
  must sit at `PolarizationPlaneEtaDeg`. It was 83°/93° on the reference data against a
  default of 90.

## §5. Reduce: I(Q) → S(Q) → G(r)

**The recipe is [`phase-5-sq-gr.md`](phase-5-sq-gr.md).** In order:
1. 1-D I(Q), pixel-weighted.
2. A **uniform** Q grid (hard rule 3).
3. Flux from the ion chambers.
4. Paalman-Pings container subtraction.
5. Multiple scattering.
6. Scale anchored on the high-Q tail only.
7. Faber-Ziman S(Q) with Compton.
8. Sine transform.

Two things belong in the spine because skipping them invalidates the run:
- **Keep the low-r term out of the normalisation** (`w_lowr=0`, anchor on the tail). The
  package default (`w_lowr=1.0`, `packages/midas_pdf/midas_pdf/refine.py:64`) fits the −4πρ₀r
  line, and then the §6 low-r check cannot test it.
- **Run the uncorrected arm beside the corrected one.** It is the discriminating test for
  rule 1.

## §6. Verify — before fitting anything

Physical consistency, on every sample:

| check | statistic | reference-beamtime result |
|---|---|---|
| S(Q) → 1 **outside** the anchor window | \|⟨S⟩[10,16] − 1\| | Ni 0.28, IPA 1.38, Kapton 0.49, CeO2 1.07 — **failed** |
| low-r line | max \|G + 4πρ₀r\| below the first peak ÷ first-peak height | Ni 0.36, IPA 4.3, Kapton 1.35 — **failed** |
| window / Q_max | the same fit at {Lorch, none} × {FT Q_max 18, 21} | a_Ni spread 3.5e-4 Å, U_iso ×1.8 |
| independent reduction, same wedge and settings | Pearson, first-peak Δr | vs GSAS-II: 0.9974, −0.0009 Å — consistency only (rule 16) |

When the physical checks fail, as they did on the reference data, every model number downstream
is **conditional on the reduction**. Say that, and carry the chain uncertainty (phase-6 §6.2).
The one lever that repaired the liquids was an oblique-incidence detector efficiency of unknown
thickness (hard rule 14; DIAGNOSIS).

## §7. Model

**The recipes are in [`phase-6-model.md`](phase-6-model.md):**
- small box, with the uncertainty recipe;
- a raw-data phase check before multiphase, and the multiphase decoy;
- strain-PDF, read in the Fisher eigenbasis;
- RMC;
- Bayesian posterior and WAIC/LOO.

Each has a package limitation that silently changes what the number means (hard rules 7–12).

---

## §8. Halt conditions — stop on these whether or not anything looks wrong

- **H1** The geometry was not verified against the raw rings out to Q_max, **and** the
  unverified range is not declared.
- **H2** `calibrate()` returned `seed_method == "fallback"`.
- **H3** `RhoD` or the residual map was not carried into the integration spec (median pixel R
  far beyond the detector diagonal, or `ResidualCorrectionMap` empty while `residual_corr.bin`
  exists).
- **H4** The corrected ÷ uncorrected 1-D profile is ≡ 1, i.e. the intensity corrections did not
  reach the integrator.
- **H5** There is no empty-container or air frame, or the flux normalisation came from an
  attenuator PV instead of an ion chamber.
- **H6** The array handed to the transform has a non-uniform Q step.
- **H7** A σ, a significance or an information criterion is quoted without a sliver calibration
  of the per-pixel error model. Label it uncalibrated instead.
- **H8** S(Q)/G(r) fails the §6 checks and a model parameter (a, U_iso, a phase fraction) is
  reported without saying it is conditional on that reduction.
- **H9** A minority phase is modelled without a raw-data detection and a decoy (hard rules 9, 10).
- **H10** A strain component is quoted from `recover_strain` without the Fisher eigen reading
  (hard rule 11).
- **H11** An RMC configuration carries species `X` (hard rule 12).
- **H12** The reduction ran through `midas-integrate-v2-pdf` (hard rule 13).

## §9. Hard rules

**In [`HARD_RULES.md`](HARD_RULES.md)** — 18 rules, each written after a silent wrong answer.

## §10. Traps that silently corrupt results

| trap | symptom | guard |
|---|---|---|
| variance integrator without `correction=` | liquid G(r) spike of +400 at r ≈ 0.1 Å; S(Q) off by ×2–5 at high Q | rule 1; H4 |
| `to_integration_spec()` without `RhoD` | R ~1e33 px, empty integration | rule 2; H3 |
| residual map not set on the spec | outer-ring crests 5× worse (264.6 vs 50.3 µε) | rule 2 |
| ring list capped by `n_rings` | the Q-scale check looks complete and stops at Q 17 | rule 4 |
| R-binned profile into the transform | plausible, slightly wrong G(r) | rule 3; H6 |
| attenuator PV trusted for flux | 10× scale error on the attenuated samples | rule 5 |
| √S on ADU as σ | every significance 15–19× inflated | rule 6; H7 |
| Hessian σ quoted raw | σ(a) 1.7e-6 Å where 4.5e-4 is honest | rule 7 |
| Lorch data, windowless model | U_iso +84 %, a moves hundreds of ppm | rule 8 |
| multiphase weight read as a fraction | a weight with no σ and a free scale | rule 9 |
| strain from one frame | +5 % e11 on an unloaded powder | rule 11; H10 |
| `Supercell.from_crystal` species | every atom `X`; swaps and CIFs wrong | rule 12; H11 |
| `midas-integrate-v2-pdf` | mask, Compton and absorption silently not applied | rule 13; H12 |
| thickness chosen to flatten S(Q) | a correction fitted to its own test | rule 14 |
| bulk density for a dispersed powder | low-r check fails in every arm | rule 15 |
| two reductions agreeing | read as validation | rule 16 |
| `contour(..., extent=(0, W, H, 0))` over `imshow` | overlay mirrored top to bottom by 2·(BC_z − (H−1)/2) | explicit coordinate arrays (DIAGNOSIS) |

---

## Provenance paths

A path written `$ANALYSIS/...` names the analysis campaign directory for the reference beamtime
(`midas_pdf_rerun`, with `scripts/`, `out/`, `logs/`, `PREREGISTER.md` and `RESULTS.md`). It is
deliberately **not** in this repository. It is *provenance, not a link*: it names the file a
number came from, and promises nothing about reaching it from another machine.
