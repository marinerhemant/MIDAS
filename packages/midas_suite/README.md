# midas-suite

A meta-package that installs the MIDAS Python pipeline in a single command.

```bash
pip install midas-suite
```

This pulls in the currently published MIDAS sub-packages — the FF/NF
HEDM analysis chain, calibration, peak fitting, radial integration,
forward model, indexing, transforms, grain processing, and stress/strain
analysis.

`midas-suite` itself contains no scientific code. It's a thin
[meta-package](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package)
whose only job is to declare the sub-packages as dependencies so users
don't have to install them one at a time.

## What you get

`pip install midas-suite` installs **19 sub-packages** (as of v0.3.1):

**Top-level orchestrators (the entry points most users want):**

| Sub-package | Role |
|---|---|
| `midas-pipeline` | Unified FF + PF orchestrator (`--scan-mode {ff,pf,auto}`). End-to-end from raw data through grain reconstruction; single source for both scan modes. |
| `midas-ff-pipeline` | Independent FF-HEDM workflow orchestrator (1-N detectors). Co-exists with `midas-pipeline`; same kernels under the hood. |
| `midas-nf-pipeline` | Pure-Python NF-HEDM pipeline orchestrator (single + multi-resolution, multi-layer). Drop-in for `nf_MIDAS.py` / `nf_MIDAS_Multiple_Resolutions.py`. |
| `midas-parsl-configs` | Bundled + user-extensible Parsl configs for running MIDAS pipelines on laptops, workstations, clusters. |

**FF-HEDM building blocks:**

| Sub-package | Role |
|---|---|
| `midas-peakfit` | Differentiable PyTorch peak fitting for FF-HEDM Zarr |
| `midas-transforms` | FF-HEDM peak transforms (merge / radius / fit-setup / save-bin) |
| `midas-index` | Pure-Python/PyTorch FF-HEDM indexer (drop-in for `IndexerOMP`) |
| `midas-fit-grain` | Single/multi-grain refiner |
| `midas-process-grains` | FF-HEDM grain-determination + strain pipeline |

**NF-HEDM building blocks:**

| Sub-package | Role |
|---|---|
| `midas-nf-preprocess` | NF-HEDM preprocessing (hex grid, tomo filter, spot prediction) |
| `midas-nf-fitorientation` | NF-HEDM orientation/calibration fitter |

**Shared foundations:**

| Sub-package | Role |
|---|---|
| `midas-stress` | Crystallographic stress/strain analysis (Voigt-Mandel, Cij inversion, slip/Schmid/Taylor) |
| `midas-params` | Parameter-file registry, validator, wizard for FF/NF/PF/RI |
| `midas-hkls` | Pure-Python crystallography & HKL list generator (sginfo-equivalent) |
| `midas-diffract` | End-to-end differentiable HEDM forward model (FF + NF + pf-HEDM) |
| `midas-integrate` | Pure-Python radial integration (`DetectorMapper` + CSR + streaming server) |
| `midas-integrate-v2` | Differentiable, autograd-clean integration kernels (torch); companion to v1 |
| `midas-calibrate` | Native Python/Torch detector geometry calibration (LM-based) |
| `midas-calibrate-v2` | Torch-native Bayesian/Laplace calibration (LM + L-BFGS); companion to v1 |

You then `import midas_stress`, `import midas_diffract`, etc. directly —
each sub-package retains its own API. `midas-suite` does not re-export
them.

To check what was installed:

```python
import midas_suite
print(midas_suite.installed())
```

## Modality bundles

If you don't want everything, the optional extras let you pick a workflow:

```bash
pip install "midas-suite[ff]"        # FF-HEDM stack
pip install "midas-suite[pf]"        # PF-HEDM stack (scanning / point-focus)
pip install "midas-suite[nf]"        # NF-HEDM stack
pip install "midas-suite[calib]"     # v1 calibration + integration
pip install "midas-suite[calib-v2]"  # v2 (torch differentiable) calibration + integration
pip install "midas-suite[ff,plots]"
```

| Extra | What it pulls |
|---|---|
| `ff` | `midas-ff-pipeline` (transitively pulls hkls, peakfit, transforms, index, fit-grain, process-grains, diffract, parsl-configs) + stress, params, calibrate, integrate |
| `pf` | `midas-pipeline[fast]` (numba) + stress, params, calibrate, integrate (scan-mode pf pulls index + fit-grain + transforms + stress transitively) |
| `nf` | `midas-nf-pipeline` (transitively pulls hkls, stress, nf-preprocess, nf-fitorientation) + params |
| `calib` | hkls, integrate, peakfit, calibrate (v1 C-backed stack) |
| `calib-v2` | hkls, calibrate-v2, integrate-v2, peakfit (torch differentiable stack) |
| `plots` | matplotlib (for sub-package plotting helpers) |

## What `pip install midas-suite` does NOT include

Be aware:

- **The MIDAS C executables** (`IndexerOMP`, `ProcessGrains`, `MakeDiffrSpots`, …)
  still need to be built from source via `cmake --build .` from the MIDAS
  monorepo. The pure-Python pipeline (calibration → integration → indexing
  → grain processing) is now end-to-end in PyTorch and does not require
  them.
- **The PyQt FF viewer GUI** needs `PyQt5` or `PySide6` installed
  separately. Not declared here because it's optional and platform-sensitive.
- **Optional crystallography backends** for `midas-hkls`: install
  `gemmi` or `pycifrw` separately for CIF I/O via `pip install
  midas-hkls[cif]`.
- **GPU acceleration** is a runtime backend selected by PyTorch device
  string. CUDA/MPS just work if your `torch` install supports them; no
  separate `*-gpu` package needed.
- **Unpublished packages**: `midas-ckernel` (its C is vendored into
  `midas-index` / `midas-fit-grain`, so nothing needs it from PyPI) and
  `midas-dct-tt` (in development). Everything else the suite names is on
  PyPI, including the research packages `midas-grain-odf`, `midas-pf-odf`,
  `midas-pink`, `midas-propagate`, `midas-uq` and
  `midas-joint-ff-calibrate`.

## The c-omp binaries — check you actually got them

`midas-index` and `midas-fit-grain` each bundle a C/OpenMP executable
(`midas_indexer`, `midas_fitgrain`). These are the fast path: the pure-Python
backends do the same job far more slowly.

Both publish as **sdist only**, so pip *compiles them on your machine* during
install. CMake and ninja arrive automatically as build requirements — you do
not need them preinstalled. What you do need is **a C compiler and OpenMP**:

| platform | what to install |
|---|---|
| macOS | `brew install libomp gcc` |
| Linux | your distro's `gcc` (usually pulls in `libgomp`) |
| Windows | Visual Studio Build Tools, "Desktop development with C++" |

**If either is missing the install still succeeds** — `CMakeLists.txt` probes
with `check_language(C)` and returns cleanly rather than failing the wheel
build, leaving you on the Python-only path. That is deliberate, but it means
the degradation is easy to miss: pip's build isolation buries the CMake
warning, and `pip install -q` hides it entirely. A slow pipeline with no error
message is the usual symptom.

So check explicitly after installing:

```python
import midas_index.backend_c as b
print(b.available())    # True  -> c-omp indexer present
print(b.binary_path())  # where it looked

import midas_fit_grain.backend_c as f
print(f.available())    # same for the refiner
```

`available() == False` means you are on the Python path. Install the compiler
and OpenMP for your platform, then
`pip install --force-reinstall --no-deps midas-index midas-fit-grain` to
rebuild.

## Cross-platform

Most MIDAS sub-packages are pure Python or PyTorch and ship as `py3-none-any`
wheels. The exceptions are `midas-index` and `midas-fit-grain`, which are
sdist-only and compile a C/OpenMP binary at install time (see above) — that is
why there are no per-platform binary wheels to maintain. Tested install paths:
Linux, macOS, Windows. See
[`packages/RELEASE_READINESS.md`](../RELEASE_READINESS.md) for the detailed
cross-platform readiness matrix.

## Versioning

`midas-suite` versions are independent of the sub-package versions.
The rule:

| Change | Bump |
|---|---|
| Floors tightened (no new sub-package added) | patch (`0.1.0` → `0.1.1`) |
| New sub-package added to the dep list | minor (`0.1.0` → `0.2.0`) |
| Backwards-incompatible reorganisation of bundles | major (`0.x.y` → `1.0.0`) |

Floors are pinned with `>=`, never `==`, so a sub-package patch release
doesn't break `midas-suite`.

## Releasing a new version

See [`RELEASING.md`](RELEASING.md) for the full release flow. TL;DR:

```bash
cd packages/midas_suite
./release.sh 0.3.2 --publish
```

## License

BSD-3-Clause, same as the sub-packages.
