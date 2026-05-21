# midas-calibrate

Native Python/Torch detector geometry calibration for MIDAS. Replaces
`AutoCalibrateZarr → CalibrantIntegratorOMP → CalibrationCore`. Same input
parameter file format, byte-compatible output, runs on CPU or GPU.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_calibrate/notebooks).

## Quick start

```python
import tifffile
from midas_calibrate import CalibrationParams, autocalibrate

params = CalibrationParams.from_file("calib.txt")
image = tifffile.imread("ceo2_calibrant.tif")
result = autocalibrate(params, image)

result.params.write("calib_refined.txt")
print(f"final mean strain: {result.history[-1].mean_strain_uE:.1f} μϵ")
```

CLI:

```bash
midas-autocalibrate calib.txt --image ceo2.tif --output calib_refined.txt
```

## How it works

* **E-step** — `midas-integrate` builds a CSR pixel→bin map from the current
  geometry and integrates the image into a 2D (R, η) cake. Per (ring × η-bin)
  the radial peak position is extracted via weighted centroid.
* **M-step** — fit detector geometry to the (Y_pix, Z_pix, ring) data using a
  custom batched Levenberg-Marquardt solver (`midas_peakfit.lm_solve_generic`)
  with sigmoid-bounded reparameterisation, Cholesky_ex, and optional Huber
  loss reshaping.
* **Orchestrator** — alternating E↔M iterations with optional σ-clip outlier
  rejection between iterations.

The geometry forward model in [`geometry_torch.py`](midas_calibrate/geometry_torch.py)
is a byte-for-byte port of `midas_integrate.geometry.pixel_to_REta` — verified
to fp64 epsilon by parity tests.

## Dependencies

- [`midas-hkls`](../midas_hkls) — pure-Python crystallography (sginfo replacement)
- [`midas-integrate`](../midas_integrate) — CSR pixel→bin mapper + integration
- [`midas-peakfit`](../midas_peakfit) ≥ 0.2.0 — generic LM solver

## Synthetic-data parity test

The end-to-end synthetic test forward-simulates a CeO₂ calibrant image at
known geometry, perturbs the seed (Lsd ±300μm, BC ±1.5px, tilts ±0.06°), and
verifies recovery:

```
[iter 0] n_fits= 176  rc=0  strain=  105.2μϵ  Lsd=1000219.4  BC=(512.20,511.91)  ty=0.343  tz=0.180
[iter 1] n_fits= 176  rc=0  strain=   25.7μϵ  Lsd= 999973.5  BC=(512.01,512.00)  ty=0.403  tz=0.250
[iter 2] n_fits= 176  rc=0  strain=   19.4μϵ  Lsd= 999946.1  BC=(511.99,512.00)  ty=0.400  tz=0.267
[iter 3] n_fits= 176  rc=0  strain=   21.6μϵ  Lsd= 999918.1  BC=(511.99,512.00)  ty=0.392  tz=0.285
```

Final recovery: Lsd within 82μm of truth, BC within 0.01 px, tilts within
0.04°. Mean strain 21.6μϵ, well under the 50μϵ MIDAS calibration target.

## Engines

`autocalibrate` is the alternating E↔M engine (default).

`autocalibrate_joint` will be the fully differentiable engine — geometry +
per-(ring × η-bin) peak-shape parameters jointly refined in one batched
Schur-complement-reduced LM (see §13 of the design doc). v0.1 ships with a
working stub that delegates to the alternating engine; the arrowhead-LM
infrastructure (`midas_peakfit.lm_solve_arrowhead`) is in place and tested.

## Status

v0.2.x — alternating engine production-ready, joint engine scaffolded.
For the torch-native, fully differentiable next-generation calibration
stack see [`midas-calibrate-v2`](../midas_calibrate_v2/).

See [`AutoCalibrate.md`](../../manuals/AutoCalibrate.md) for the manual.
