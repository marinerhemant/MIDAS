# Grain-Based `tx` Refinement — the Geometry a Powder Calibration Cannot See

**Package:** `midas_joint_ff_calibrate`
**Applies to:** FF-HEDM reconstructions calibrated from a powder standard
**Units:** micrometres, degrees, Ångstroms (MIDAS convention)

---

## 1. The problem

A powder / calibrant calibration (`midas-calibrate-v2`, `AutoCalibrateZarr`)
determines `Lsd`, the beam centre, the detector tilts `ty` and `tz`, and the
distortion harmonics. It **cannot determine `tx`** — the in-plane detector
rotation about the beam axis.

The reason is structural, not a limitation of the fitting: Debye–Scherrer rings
are rotationally symmetric about the beam. Rotating the detector in its own
plane maps every ring onto itself, so every value of `tx` fits a powder pattern
equally well. No amount of calibrant data, exposure, or ring statistics changes
this.

`tx` is only observable in **single-crystal spots**, whose azimuthal positions
are not rotationally symmetric.

**Consequence:** every FF reconstruction seeded from a powder calibration runs
with `tx = 0`. If the true `tx` is not 0, every spot is displaced azimuthally by
`R·sin(tx)`. On a real Ni dataset a `tx` of 0.049° at `R ≈ 150 mm` is a ~130 µm
shift — enough to corrupt indexing and to leak into refined grain positions and
strains.

## 2. What `tx` actually does

Measured, not asserted (notebook 05, part 1):

| Quantity | Sensitivity to `tx` |
|---|---|
| azimuth `η` | ~1:1 — a 0.4° `tx` error moves `η` by 0.4° |
| radius `R` | zero to numerical precision |
| `2θ` | invariant |
| `ω` | invariant |

Two things follow.

**The loss must be η-sensitive.** A radial or pixel-distance residual is blind
to `tx`, so `kind="pixel"` is *disabled* in the API. Use `angular` (the default
3-D `2θ, η, ω` residual) or `internal_angle`.

**`tx` cannot fix a wrong `Lsd` or a hydrostatic strain offset.** Those live in
`2θ`, which `tx` does not touch. For the absolute-strain problem use `d0`
recovery (`midas_stress.recover_d0_cubic_free_standing`) instead.

## 3. Why the grain pose is frozen during the fit

`refine_geometry_from_grains` holds grain orientation, position and strain
**fixed** at their reconstructed values while fitting `tx`. The usual
explanation is a degeneracy — `tx` rotates the observed pattern about the beam,
a grain's orientation rotates the predicted pattern about the beam, so the
grains could absorb any `tx`.

That explanation is **not quite right**, and notebook 05 part 2 measures it.
Counter-rotating every grain about the beam by the same angle as a `tx` offset
does *not* restore the fit — it makes the cost roughly 1.5× **worse**. The
degeneracy is not exact, because rotating a grain about the beam also shifts the
`ω` at which each of its spots diffracts, and `tx` leaves `ω` untouched. That
`ω` mismatch, accumulated across differently-oriented grains, is exactly the
multi-grain coupling that makes `tx` identifiable at all.

The practical conclusion is unchanged: with the pose thawed, the refinement
**fails anyway** — it stalls, leaving `tx` near 0. Near-degenerate directions
plus dozens of weakly-determined nuisance parameters destroy the conditioning
even though the problem is formally identifiable. Frozen-pose, `tx` is a
one-parameter fit with a clean minimum. Pose and strain are refined downstream
in `process_grains`, where they belong.

This also means **`tx` needs several grains**. One grain cannot separate `tx`
from its own orientation. The default `max_grains = 50` is a sensible floor.

## 4. Running it

### Command line

```bash
midas-joint-ff-calibrate grain-tx \
    --paramstest  master_ff_params.txt \
    --layer-dir   recon/LayerNr_1 \
    --refine tx --kind angular --max-grains 50 \
    --out recon/LayerNr_1/paramstest_graintx.txt
```

Reads `Grains.csv`, `SpotMatrix.csv` and `hkls.csv` from the layer directory.
Grains are selected by **best fit** (smallest mean g-vector angle at the seed
pose), not by highest confidence — confidence admits badly-fit grains whose
residuals swamp `tx`'s sub-degree signal.

Add `Wedge` to `--refine` to co-refine the rotation axis.

### Library

```python
from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains

res = refine_geometry_from_grains(
    paramstest="master_ff_params.txt",
    layer_dir="recon/LayerNr_1",
    refine_params=("tx",), kind="angular",
    max_grains=50, out_paramstest="recon/LayerNr_1/paramstest_graintx.txt")
print(res.refined, res.cost_init, res.cost_final, res.n_spots_matched)
```

### As a pipeline stage

```bash
midas-pipeline run --scan-mode ff ... \
    --grain-geometry-run --grain-geometry-refine tx
```

FF-only, off by default, runs after `process_grains`. No-ops cleanly when
disabled, in PF mode, or when the grain files are missing.

## 5. ⚠ Pass the *master* parameter file

The single most common failure. Give `grain-tx` the **master** FF parameter file
— the one `build_paramstest` / `FitSetupParams` wrote — not the stripped
per-layer `paramstest.txt` that the refiner consumes.

The stripped file drops the ω-scan and detector keys (`OmegaStep`,
`NrFilesPerSweep`, `NrPixelsY/Z`). Without them `OmegaStep` defaults to 0, which
is a degenerate zero-width ω scan, so the forward model marks **every** predicted
spot invalid. The symptom is `matched spots = 0`, cost `0 → 0`, and `tx = 0` —
a silent, entirely misleading success.

There is now a guard that raises `ValueError` on `OmegaStep == 0` rather than
returning a bogus zero.

## 6. Completing the second pass

`tx` enters the analysis at the **transforms** step, so a refined `tx` only
helps if you re-run the reconstruction with it. Fold the value into your master
parameter file and re-run:

```python
import re
tx_ref = res.refined["tx"]
txt = master.read_text()
txt = (re.sub(r"(?m)^tx\b.*$", f"tx {tx_ref:.6f}", txt)
       if re.search(r"(?m)^tx\b", txt) else txt + f"\ntx {tx_ref:.6f}\n")
master.with_name(master.stem + "_tx.txt").write_text(txt)
```

Peak fitting is geometry-independent, so `midas-pipeline resume --from
transforms` reuses it and only re-runs transforms → binning → index → refine →
grains. A fresh result directory instead gives a clean pass-1 vs pass-2
comparison (grain count, median confidence, strain spread).

The corrected-parameter writer **text-edits** the original file rather than
round-tripping it through `V1Params`, which would drop non-v1 keys such as
`LatticeParameter` and produce a "lattice constants must be positive" failure
downstream.

## 7. Limitations

- **Only `tx` and `Wedge` are refinable.** `refine_geometry_from_grains` builds
  its parameter spec with those two; `Lsd`, `BC_y`, `BC_z`, `ty`, `tz` and the
  distortion coefficients are supplied as fixed geometry, so requesting them
  raises `KeyError`. The archived C `FitGrain` fit ten geometry parameters;
  that breadth has not been ported, because `tx` is the one a powder
  calibration genuinely cannot supply.
- **Not a substitute for a good powder calibration.** It corrects one parameter
  the powder fit could never have given you, not a bad powder fit.
- **`with_powder=True` (full joint powder + grains) raises `NotImplementedError`**
  at this entry point; use
  `midas_joint_ff_calibrate.runners.run_real_phase3_joint`.

## 8. See also

- `packages/midas_joint_ff_calibrate/notebooks/05_grain_tx_from_grains.ipynb` —
  the measurements behind sections 2 and 3, and a runnable version of section 4.
- `packages/midas_pipeline/notebooks/06_ff_cbf_real_data_calibrate_and_reconstruct.ipynb`
  §5 — the two-pass flow inside a full raw-CBF → grains reconstruction.
- [FF_Calibration.md](FF_Calibration.md) — the powder calibration that comes first.
- [FF_Parameters_Reference.md](FF_Parameters_Reference.md) — `tx`, `ty`, `tz`, `Wedge`.
