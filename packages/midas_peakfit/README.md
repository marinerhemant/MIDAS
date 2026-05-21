# midas-peakfit

Differentiable PyTorch peak-fitting for FF-HEDM Zarr archives. A drop-in
replacement for `PeaksFittingOMPZarrRefactor` (C/OpenMP/NLopt) with the
following changes:

- **Optimizer:** Nelder-Mead → batched **Levenberg-Marquardt** on CPU/CUDA
- **Backend:** OpenMP → PyTorch (autograd, batched linear algebra)
- **Precision:** fp64 default; `--dtype float32` available for speed
- **Output:** identical binary format (`AllPeaks_PS.bin`, `AllPeaks_PX.bin`)
- **CLI:** drop-in compatible

The C tool is kept as the validation oracle. Output is **scientifically
equivalent** to the C tool, not bit-exact (LM and Nelder-Mead converge to
slightly different minima within the same basin).

## Installation

```bash
pip install -e packages/midas_peakfit[dev]
```

or once published:

```bash
pip install midas-peakfit
```

PyTorch with CUDA support must be installed separately if GPU acceleration is
desired. Follow the [PyTorch install guide](https://pytorch.org/get-started/locally/)
for the right wheel for your CUDA version.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_peakfit/notebooks).

## Usage

```bash
peakfit_torch DataFile.MIDAS.zip 0 1 8                    # CPU/CUDA auto
peakfit_torch DataFile.MIDAS.zip 0 1 8 --device cuda --dtype float32
peakfit_torch DataFile.MIDAS.zip 0 1 8 OutputFolder 1     # explicit ResultFolder + fitPeaks
peakfit_torch DataFile.MIDAS.zip 0 1 8 \
    --validate-against /path/to/c_AllPeaks_PS.bin         # parity check
```

Positional args mirror the C tool exactly: `DataFile blockNr nBlocks numProcs [ResultFolder] [fitPeaks]`.

New flags:

| Flag | Default | Meaning |
|---|---|---|
| `--device {cpu,cuda}` | `cuda` if available else `cpu` | Compute device |
| `--dtype {float32,float64}` | `float64` | Numeric precision |
| `--batch-size N` | `4096` | Cross-frame region batch threshold |
| `--validate-against PATH` | — | Compare to C-produced `AllPeaks_PS.bin` and emit parity report |
| `--deterministic` | off | Force deterministic algorithms (fp64 only) |

## Parity tolerances

| Field | Tolerance |
|---|---|
| `nPeaks` per frame, pixel sets, `maxY/maxZ`, `returnCode`, `maskTouched` | exact |
| `YCen, ZCen, Radius, diffY, diffZ` | ≤ 0.05 px |
| `Eta` | ≤ 0.02° |
| `IMax, IntegratedIntensity, RawSumIntensity` | ≤ 1% relative |
| `BG, SigmaR, SigmaEta, σGR, σLR, σGEta, σLEta, MU, FitRMSE` | ≤ 5% relative |

Downstream gate: indexer (`IndexerOMP`) on both outputs must produce identical
grain orientations within 0.05° misorientation.

## Output

Two binary files in `{ResultFolder}/Temp/`:

- `AllPeaks_PS.bin` — peak summary (29 columns × nPeaks per frame; see
  `FF_HEDM/src/PeaksFittingConsolidatedIO.h` for layout).
- `AllPeaks_PX.bin` — pixel coordinates for each peak.

These files are byte-compatible with the C tool's output and readable by
`ConsolidatedPeakReader`/`ConsolidatedPixelReader` in that header.

## Tests

```bash
cd packages/midas_peakfit
pytest tests/ -v                      # unit tests (fast)
pytest tests/ -v -m slow              # full pipeline parity (~3 min)
pytest tests/ -v -m gpu               # CUDA-only tests (skip on CPU)
```

## License

BSD-3-Clause.
