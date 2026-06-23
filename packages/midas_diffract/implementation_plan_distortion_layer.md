# Implementation plan — gated ideal→raw detector-response (distortion + tilts) in midas_diffract

**Goal:** let raw-pixel-patch consumers (pf_odf, grain_odf) predict spot positions in the
**raw detector frame** (where their measured pixels live), using the **canonical
`midas_distortion` v2 model**, WITHOUT changing the default ideal-frame behaviour that the
indexer / fit-grain / transforms pipeline rely on.

## 1. Background / why
- `midas_transforms` corrects **observations** raw→ideal (undistort + untilt + BC); the
  indexer and `midas_fit_grain` match those ideal observations against the **ideal**
  `midas_diffract` forward. midas_diffract already documents this: FF/PF mode *ignores*
  tilts/distortion "to avoid double-correcting" (`HEDMGeometry`, forward.py:65–90, 1127).
- **pf_odf / grain_odf are the exception:** they consume **raw pixel patches** (they need
  the full peak *shape*, which can't be reduced to a corrected centroid), so they never go
  through `transforms`. Predicting with the ideal forward leaves a detector-position-
  dependent offset (measured 0.77px → 1.7px with radius on Bucsek Pilatus 2M) = the
  ~1500µε false-strain floor.
- Fix = map the **prediction** ideal→raw (NOT warp the patch, which would smear the very
  peak shape we measure). Mirrors calibrate_v2's own forward
  (`bragg.py` ideal radius → `geometry.py` `apply_distortion` → detector radius).

## 2. Decision (agreed with user)
Add distortion to **midas_diffract** (shared infra → pf_odf, grain_odf, future raw consumers
all benefit), **gated and default-off**, alongside the existing `apply_tilts` flag. Use the
new canonical **`midas_distortion`** package as the single source (import, don't re-port).
Standardize on the **v2** coefficient model everywhere.

## 3. Canonical model (midas_distortion, on Mac repo, v0.2.0)
- `apply_distortion(R, eta_deg, p_coeffs, rho_d, terms=v2_term_layout())` = `R · D(R/rho_d, η)`,
  multiplicative, **backend-agnostic (numpy AND torch → autograd-safe)**.
- `D = 1 + Σ_t amp·ρ^power·[cos(fold·η_T + phase) | 1 if fold==0]`, η_T=(90−η)·π/180.
- v2 layout: iso_R2/R4/R6 (folds 0) + a1..a6/phi1..phi6. `v1_to_v2_coeffs` /
  `v2_coeffs_from_named` bridge legacy p0–p3(..14). `resolve_rho_d_um` for ρ normalization.
- **Direction:** calibrate_v2 forward uses `apply_distortion` as **ideal→raw**
  (bragg ideal radius → detector radius) ⇒ pf_odf uses it directly, same direction.

## 4. Changes
### 4a. midas_diffract/pyproject.toml — add `midas-distortion>=0.2.0` dependency.
### 4b. HEDMGeometry (forward.py) — new optional fields, **defaults = no-op**:
- `p_distortion: list[float] | None = None`  (15 v2 coeffs, `P_COEF_NAMES` order; None ⇒ off)
- `rho_d: float | None = None`                (µm; None ⇒ `midas_distortion.resolve_rho_d_um`)
- `apply_distortion: bool = False`            (`apply_tilts: bool = False` already exists)
### 4c. forward radius→pixel step (calc_bragg_geometry / predict_spot_coords) — gated, mirrors
calibrate_v2.forward.geometry:
```
# after existing apply_tilts path + projection to rad_um, eta_deg:
if geom.apply_distortion and p_distortion is not None:
    rad_um = midas_distortion.apply_distortion(rad_um, eta_deg, p_v2_tensor, rho_d,
                                               terms=v2_term_layout())
# then rad_um -> pixels as today
```
Default off ⇒ rad_um unchanged ⇒ **byte-identical** to current (indexer/fit-grain safe).
Coeffs as tensor ⇒ differentiable; midas_distortion handles device/backend.
### 4d. Consumers — pf_odf (`pfodf_realdata_fast.py`) + grain_odf opt in: set
`apply_tilts=True, apply_distortion=True`, feed calibrated v2 coeffs + rho_d from
calibrate_v2 `AutoCalibrationResult`.

## 5. Risks to verify DURING implementation
1. **Tilt convention:** earlier `apply_tilts=True` made centering *worse* (0.69→0.90px).
   Confirm midas_diffract tilt sense == calibrate_v2 `build_tilt_matrix`; fix sign/order if not.
2. **Frame / η convention:** calibrate_v2 native (1679×1475) vs pf_odf square-pad+transpose.
   η and BC must be in pf_odf's frame (transpose/flip rotates η). Map BC + phases (phi_k) to
   the pf_odf frame OR evaluate natively & map back. VALIDATE by the offset test, don't assume.
3. **rho_d** must match calibrate_v2's (detector corner, µm) — use the same resolver.

## 6. Tests (differentiable + CPU/CUDA/MPS per repo standard)
- **Regression (critical):** `apply_distortion=False` ⇒ forward bit-identical to pre-change.
- **Round-trip:** transforms raw→ideal ∘ forward ideal→raw = identity (±1e-6 px).
- **Autograd:** gradient flows to `p_distortion`.
- **Device portability:** CPU vs CUDA agree ~1e-10.
- **Real-data validation:** predicted-raw vs measured CeO2 peak offset collapses 0.77→~0 px
  across radius (overlay test, predictions in raw frame).

## 7. Rollout
1. Implement 4a–4c on Mac; run midas_diffract tests (esp. bit-identical regression).
2. Deploy to s1iduser env. 3. pf_odf grain-1 with layer on + calibrated coeffs → peak offset
~0, undeformed floor → <100µε. 4. Re-run recon + deformed 220N vs 0N contrast.

## 8. Non-goals
- Not changing transforms/indexer/fit-grain (stay ideal). Not migrating legacy paramstest
  p0–p3 writers now (v1↔v2 bridge covers reading). Not warping measured patches.
