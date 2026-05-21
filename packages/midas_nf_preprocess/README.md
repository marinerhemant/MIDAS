# midas-nf-process-images

Differentiable PyTorch port of `NF_HEDM/src/ProcessImagesCombined.c` for CPU, CUDA, and MPS.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_nf_preprocess/notebooks).

## Pipeline

For one layer of a near-field HEDM scan:

1. **Load** `NrFilesPerLayer` raw TIFF frames into a `[N, Z, Y]` tensor.
2. **Temporal median** across frames → per-pixel background `[Z, Y]`.
3. **Per-frame**:
   - Subtract median + `BlanketSubtraction`, clamp at 0.
   - Spatial median (3x3 or 5x5).
   - LoG response at one or more scales.
   - Threshold and label connected components → integer spot labels (detached).
   - Sigmoid surrogate → continuous spot-probability map (autograd path).
4. **Accumulate** spot pixels into a bit-packed `SpotsInfo.bin` mmap (drop-in compatible
   with `FitOrientationOMP` / `simulateNF`).

Phases 1, 2, and 3 (filtered, log_response, soft spot probability) are differentiable
end-to-end. Connected components and the binary `SpotsInfo.bin` output are computed on
detached tensors and do not participate in autograd.

## Quickstart

```python
from midas_nf_process_images import ProcessParams, ProcessImagesPipeline

params = ProcessParams.from_paramfile("ps.txt")
pipe = ProcessImagesPipeline(params, device="cuda")  # or "mps", "cpu"
spots = pipe.process_layer(layer_nr=1)
spots.write("/path/to/SpotsInfo.bin")

# Or all layers in one call:
spots = pipe.process_all(layers=range(1, params.n_distances + 1))
```

## CLI

```bash
midas-nf-process-images <ParameterFile> <LayerNr> [--device cuda] [--dtype fp32] [--n-cpus 8]
```

Behaves like the C `ProcessImagesCombined` executable.

## Backend selection

Same contract as `midas-transforms`:

- Default device: `cuda` -> `mps` -> `cpu`.
- Default dtype: `float64` on CPU, `float32` on CUDA/MPS.
- Override with `device=` / `dtype=` kwargs, or env vars
  `MIDAS_NF_PROCESS_IMAGES_DEVICE` / `MIDAS_NF_PROCESS_IMAGES_DTYPE`.

## Differentiability

The end-to-end pipeline produces three tensors:

- `filtered`: median-subtracted, spatial-median-filtered image (autograd).
- `log_response`: LoG convolution output (autograd).
- `spot_prob`: soft, continuous spot-probability map via `sigmoid(L / temperature)` (autograd).

Plus one detached output:

- `labels`: integer connected-component IDs from hard-thresholded LoG (no gradient).

Optimization through the discrete spot mask uses the soft `spot_prob` surrogate.
