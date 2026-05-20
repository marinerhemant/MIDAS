# midas-diffract examples

Runnable notebooks demonstrating the differentiable HEDM forward model. All
run on CPU with small synthetic data — no C binaries, no real datasets. The
reflection lists come from the optional `midas-hkls` companion package:

```bash
pip install midas-diffract[hkls]
```

| Notebook | Topic |
|----------|-------|
| [01_ff_pixel_exact.ipynb](01_ff_pixel_exact.ipynb) | FF-HEDM spot-coordinate mode: forward-simulate a grain, check gradients, recover orientation + lattice from a perturbed start via L-BFGS. |
| [02_nf_forward_render.ipynb](02_nf_forward_render.ipynb) | NF-HEDM image mode: render a grain to a `(frame, y, z)` detector volume with `predict_images()` (3D Gaussian splatting) and confirm gradients flow back through the splatting. |

Re-execute any notebook with:

```bash
jupyter nbconvert --to notebook --execute --inplace 02_nf_forward_render.ipynb
```
