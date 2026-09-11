# Envelope — what has actually been exercised

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).

**Owner:** Hemant Sharma. **Last reviewed 2026-09-10.**

The spine reads as procedure. This records how much of it has been *run*, and on what, so an
untested path is not promoted to a recommendation.

## Tiers — which limits can move, and which cannot

A *configured* limit can be proposed as a change. An *intrinsic* one must be reported as
unobtainable, not tuned at.

| tier | meaning | what falls here |
|---|---|---|
| **Fixed** | the detector and beamline for this beamtime | pixel size; the scintillator and its **unknown thickness** (hence no detector-efficiency correction); the per-pixel noise; the beam centre at the detector edge, which limits azimuthal coverage to 175° and keeps near-vertical wedges from reaching Q 21 |
| **Configured** | chosen per run | Q range and dQ; the window; the normalisation anchor; the r range of every fit; background order; wedge or sliver selection; the ring-separation cut and distortion model of the calibration |
| **Intrinsic** | no parameter recovers it | λ–Lsd degeneracy (inherited from `calibrate-integrate`); the Q scale beyond the last measurable calibrant ring; e11, e12, e13 from a single frame; phase weight × scale; Fourier-correlated G(r) points (N_eff = Δr·Q_max/π) |

## Exercised end to end, on real data — one beamtime (2026-09-10)

**The data.** 1-ID-E, Varex 2880² at 150 µm, 67.4164 keV, Lsd ≈ 470 mm. Frames: CeO2, Ni powder
in carbon black, liquid IPA, an empty Kapton capillary, air; 180-frame sums.
**Provenance:** `$ANALYSIS/RESULTS.md`, against `$ANALYSIS/PREREGISTER.md`.

| path | evidence | outcome |
|---|---|---|
| `calibrate()` with no hint → `make_seed` | `out/cal_edge/summary.json` | seeded from the image; the defaults did not converge (§3 of the notebook) |
| six calibration variants, isolated-ring crests, E-step ring SNR, η-bin scan | `out/cal_*/verify*.json`, `out/01g`–`01i*.json` | outer rings unmeasurable beyond R ≈ 1300 px; accepted at 50.3 µε to Q 17.2 |
| geometry hand-off with `RhoD` and the residual map | `out/cal_sep11_none/verify_map.json` | works when set by hand; `to_integration_spec()` alone does not |
| per-pixel polarization × solid angle into `integrate_hard_with_variance`, 5 frames, 4 σ models | `out/int_sep11_none_map/meta.json` | 19 s wall |
| interleaved-sliver σ calibration with a planted control | `out/03c_sigma_edge.json` | every model refuted |
| I(Q) → S(Q) → G(r): Paalman-Pings, multiple scattering, tail-anchored normalisation, Compton, sine FT | `out/pdf_sep11_none_map_q21_lorch/summary.json` | ~30 s for four samples; **physical checks fail** |
| report-only arms: uncorrected, Compton k 3 and it94, no window, FT Q 18, GSAS-II wedge, detector efficiency at 200 and 600 µm | `out/pdf_sep11_none_map_*` | detector efficiency is the only lever that repaired the liquids |
| comparison with a GSAS-II PDF read from its `.gpx` (restricted unpickler) | `out/05e_h5_*.json` | consistent (Pearson 0.9974, Δr −0.0009 Å) |
| `refine_structure` on Ni and CeO2, 8 arms each, uncertainty recipe | `out/06a_smallbox_sep11_none_map.json` | CeO2 −1107 ppm; a_Ni ± 3.9e-3 Å |
| Δ-PDF between interleaved halves, with +100 and +500 µε plants | `out/08a_delta_pdf_sep11_none_map.json` | power at 100 µε; σ 1.5× too small |
| `strain_crlb` / `recover_strain` on four wedges, plus the Fisher eigen reading | `out/09a_*.json`, `out/09b_*.json` | package output unusable as-is; in-plane components ≤ 123 µε |
| raw-I(Q) matched filter for a minority phase | `out/10a_nio_raw_sep11_none_map.json` | NiO absent; the controls work |
| `refine_multi_phase` decoy from two starts; `refine_core_shell` null | `out/10b_decoy_coreshell_sep11_none_map.json` | the free decoy absorbs misfit; the core-shell null is non-zero with σ = 0 |
| RMC ensemble: 4 chains × 10 000 moves, 256-atom Ni | `out/11a_rmc_sep11_none_map.json` | chains agree (χ² spread 4 %); first shell +0.006 Å from a/√2, which is the s²/d disorder bias; 48 min on the Mac |
| SVI for 3 background models, WAIC/LOO; NUTS on bg 0 | `out/12a_bayes_sep11_none_map.json`, `out/12a_nuts_sep11_none_map.json` | NUTS matches the MAP (\|z\| ≤ 0.06) and the rescaled Laplace width, after two package crashes were worked around (59 min); SVI lands 1.8–4.1σ from its MAPs; ranking not decisive after √(N/N_eff) |
| CIF round trip; every `midas-pdf-*` CLI on the real G(r) | `out/13a_cli_roundtrip_sep11_none_map.json` | U_iso dropped; `--pin-geometry` no-op; `X` atoms |

## Not exercised — stop and ask rather than improvise

| path | why it is not covered |
|---|---|
| any other detector, energy or distance | every number above is from one beamtime |
| detector efficiency at a **known** sensor thickness | the thickness was unknown; the two arms run are sensitivity only. No public, panel-specific number exists for the Varex 4343CT (checked: its own datasheet, and the two published high-energy beamlines using this model) — the best documented analogue is a sibling Varex product's 600 µm CsI:Tl, not a measurement of this panel |
| refining ρ₀ (`refine_normalization(..., fit_number_density=True)`) | not run; it is the obvious next test of the Ni low-r failure |
| per-site or anisotropic ADPs (`refine_aniso_occupancy`) | registered as report-only, not run |
| polygon vs hard vs subpixel binning | registered as report-only, not run; hard binning only |
| neutral vs ionic form factors on CeO2 | Ce⁴⁺ is refused by midas_hkls in this version |
| Monte-Carlo multiple scattering (`multiple_scattering_mc_cylinder`) against the transport model | not run |
| a time-series Δ-PDF (`sequence_delta_pdf`) | no series in this beamtime |
| SAXS / SANS joint refinement (`midas-pdf-joint`, `joint_refine`) | no SAXS data |
| anomalous scattering (`form_factor_averages(anomalous=True)`) | never enabled by the pipeline |
| fluorescence subtraction | the package only reports expected lines |
| `midas_pdf.image_to_iq` / `image_to_Gr` | they do not correct intensity (hard rule 1); not used |
| `midas-integrate-v2-pdf` | unusable as it stands (hard rule 13) |
| GPU | `midas_pdf` has no device option |
| this doc set handed to a context-free model | not run |
| `/verify` on any positive result | not run |

## Numbers that are beamtime-specific, not constants

- Polarization × solid-angle factor: ×2.3 at Q 20 Å⁻¹, ×4.8 at 26, at Lsd 470 mm.
- Per-pixel σ of data − dark: 2400–2700 ADU; dark lag 0.7–0.9 % of its own exposure.
- Last measurable CeO2 ring for the E-step: R ≈ 1200–1300 px at this distance and energy.
- Per-sliver fractional systematic f: 0.05–0.12 %.
- Every H4 value, every G(r) number, and the Ni and CeO2 lattice constants.

## Versions

midas-pdf 0.3.0, midas-integrate-v2 0.7.1 and midas-hkls 0.11.0 were editable installs of this
repository.

**midas-calibrate-v2 was bumped during the work.**
- 0.15.0 (commit `f1e65c57`, 08:49) → 0.16.0 (`37493d6a`, 11:55) on 2026-09-10.
- The default `edge` calibration (written 11:39) ran on 0.15.0 source.
- The accepted `sep11_none` calibration (written 13:14, after 24 min) ran on 0.16.0 source.
- The working tree also carried an uncommitted 10-line change in `pipelines/auto.py` from
  outside this work.
