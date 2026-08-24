# Tomography — measurement envelope

**Instrument:** APS 1-ID, parallel beam, `tomoC` camera
**Last checked:** 2026-08-23 · **Owner:** Hemant Sharma (hsharma@anl.gov)

> Part of the **tomo doc set**. Spine: [`README.md`](README.md). Frames:
> [`COORDINATES.md`](COORDINATES.md).

What a tomogram can and cannot determine *for the purpose this doc set exists
for* — supplying a registered sample shape to a diffraction experiment. Read it
before promising a correction.

> **Not the scope gate.** The scope gate says whether these recipes apply. This
> says whether the measurement can answer the question. A scan can be squarely
> in scope and still unable to support what is asked of it — the two datasets
> below differ by 33× in exactly that way.

---

## 1. Fixed — cannot change this cycle

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Contrast mechanism | absorption **and/or** propagation phase | `tomocupy_args.yml: propagationDistance` | With a non-zero propagation distance the reconstruction is edge-enhanced. Thresholding it directly gives a **hollow shell**, not a filled sample: the interior has no contrast. | Phase retrieval before thresholding. Without it, use the mask only for the outer boundary and say so. |
| Field of view | `n_px × pixel_size` | acquisition | A sample wider than the FOV is **truncated**; the reconstruction cups and cannot be a sample mask at any threshold. | Re-acquire at lower magnification, or restrict every claim to the interior region that is fully sampled at all angles. |
| μ of the specimen | set by composition and energy | `midas_hkls.absorption` (NIST) | **Decides whether an absorption correction exists to be measured.** Below μ·D ≈ 0.1 the correction is under the per-spot noise. | None at fixed energy and composition. Report "no detectable effect" — that is a result. |
| Registration in-plane | **recorded in `tomo_metastr`** | per-scan `*_TomoFastScan.dat` | Nothing in the *reconstruction* records it, so a tomogram read off disk carries no handedness. But the acquisition does: every 1-ID tomo scan writes `aero axis, left handed` (or right) into its metadata string. | Read it from `<expt>/metadata/<expt>/<scan>/<scan>_TomoFastScan.dat`, not from the reconstruction. Mapping `left handed` onto one of the eight `TOMO_IN_PLANE` signed permutations is **still unresolved** — the string names the convention, not the axis assignment — so verify with the N3 meta-null regardless. |
| Pixel size | **NOT in `tomocupy_args.yml`** | per-scan `tomo_metastr` | `tomocupy_args.yml` carries a *detector-configuration* value that need not match the scan. Measured: it says `pixelSize: 1.17` for both bt_1id_jun25b and bt_1id_jul26, but both scans ran on **FLIR-GH1 at 5X** — 0.708 µm/px (bt_1id_jun25b `nmc811s5tomo1`) and 0.69 µm/px (bt_1id_jul26 `tomo_Ce_ht525_s2`). 1.17 µm is the PointGrey value from `ad_settings.csv`, which also lists gh1 0.69, gh2/pg6 2.95, pg1/pg5 1.17. A 1.65× error in pixel size is a **4.5× error in every volume**. | Always read `tomo_metastr` from the scan's own `*_TomoFastScan.dat`. Same class of trap as the stale `exp_setup.yml EDGE:` field. |

**Measured, on the two datasets this doc set was written against:**

| dataset | energy | μ (cm⁻¹) | **measured μ·D** | verdict for an absorption correction |
|---|---|---|---|---|
| bt_1id_jun25b NMC811 s5 | 51.9 keV | 6.53 | **0.05** | **null** — at the ±2.5 % flat-field noise floor |
| bt_1id_jul26 Ce `ht525_s2` | 95 keV | 18.94 | **1.63** ± 0.02 over 8 angles | **testable** — 33× the above, transmission 18 % |

**Measured 2026-08-23 — the NMC811 tomogram cannot produce a sample mask.**
Run on the cleaned best-shift reconstruction (`s5_tomo1_cleaned`,
`BEST_SHIFT_+13.00`, 128³ at 0.708 µm):

| diagnostic | result |
|---|---|
| threshold stationarity | **FAIL** — volume swings 372 134 → 3 721 µm³ over a p50–p99.5 sweep; fractional spread 9.2 (**4.64× in radius RETRACTED**, §3.7) |
| mask extent | 81.4 × 90.6 µm in a **90.6 µm** field of view, all 128 slices occupied — the "sample" fills the reconstruction |
| against the projections | the sample's projected width is ~29 µm at 0.708 µm/px, so the mask is **3× too wide** |

The mask is thresholded background and edge-enhancement, not specimen. This is
the phase-contrast row of §1 confirmed on real data: with μ·D 0.05 there is no
absorption contrast to threshold, and an FBP of propagation-contrast
projections has no interior to find. Phase retrieval first, or use a different
specimen.

The NMC811 scan is phase contrast precisely *because* its absorption contrast
is a few percent. That is the tell: if the experiment needed propagation to see
the sample, absorption will not correct the diffraction either.

## 2. Configured — set per run, changeable next time

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| Pixel size / magnification | per run | the lens set installed | available objectives and the scintillator's own resolution | Resolution against field of view. The binding constraint is that the sample must fit at **every** angle, not on average. |
| Number of projections | per run | up to the Crowther limit, ≈ π/2 × in-plane pixels | acquisition time and dose | Angular sampling. Under-sampling shows as streaks radiating from high-contrast edges, which threshold into spurious mask fingers. |
| Propagation distance | per run | 0 to a few hundred mm | detector stage travel, and fringe overlap between neighbouring features at large distance | Edge contrast for a weakly absorbing sample, at the cost of needing phase retrieval before the reconstruction is a density map. |
| Shift-sweep range/step | per reconstruction | range ≳ the expected axis error; step ≳ 0.1 px | compute time for the range; **the reconstruction's own resolution** for the step — a step well below a pixel resolves nothing further | How finely the rotation axis is located. Too coarse leaves a double edge that fattens the mask and inflates every derived volume. |
| Energy | per run | 1-ID's usable range | undulator harmonic and monochromator | μ, and therefore whether an absorption correction is measurable. Also penetration: μ·D ≳ 3 is effectively opaque. |

**Rows deliberately blank.** Detector frame rate, stage travel limits, and dose
tolerance are not recorded in this doc set. Until filled in, a report **will
not** propose changing exposure or total dwell.

## 3. Intrinsic — the sample or the physics forbids it

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Absolute density from a phase-contrast reconstruction | Propagation contrast is a function of the *gradient* of the refractive index, not its value. The reconstruction is not a density map without phase retrieval and a calibration. | An absorption tomogram at the same energy **is** proportional to μ, hence to density given composition. |
| Grain boundaries in a single-phase metal | There is no absorption or phase contrast between grains of the same phase and density. | This is DCT's job — `manuals/dct-tt/` — which uses diffraction, not attenuation. |
| The sample's composition | A single-energy tomogram gives μ, which is one number confounding density and composition. | Two energies bracketing an absorption edge separate them; that is a different experiment. |
| Anything outside the illuminated region | No measurement. | An unilluminated detector region produces a *smooth plausible plateau* in a naively computed transmission — it looks like data. See spine rule 2. |

## 4. Derived limits

| Quantity | Limit | From |
|---|---|---|
| Smallest resolvable feature | ≈ 2 × pixel size, worse with phase contrast | §2 |
| Usable sample diameter | < FOV at **every** angle, with margin | §1 |
| Absorption correction detectable | μ·D ≳ 0.1, comfortably at ≳ 0.3 | §1 measured rows |
| Illuminated-volume accuracy | set by the **threshold**, not by resolution | §5 below |

## 5. Did not versus cannot

- **Threshold sensitivity not reported.** The binarisation threshold multiplies
  the illuminated volume directly, and therefore every grain volume derived
  from it. Always report `V_illum` with a band from sweeping the threshold; if
  the volume is not stationary across a reasonable band, the mask is not usable.
  Skipping this is a choice, not a limit.
- **Shift index not recorded.** The reconstruction is a *sweep* over candidate
  rotation-axis shifts; picking one by eye and not writing it down makes the
  result unreproducible. A choice, not a limit.
- **Registration verified on a symmetric sample.** Most in-plane checks have no
  power on a cylinder. That is a property of the *sample*, so it belongs in §1
  for that specimen — but it is fixable by using a fiducial, which makes it a
  "did not" for the experiment as a whole.
- **Phase retrieval not run.** A pipeline-version question, not a measurement
  limit. Report as "not retrieved", never "not available".

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
