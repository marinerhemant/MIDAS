# midas-integrate-gpu

CUDA streaming backend for [`midas-integrate`](https://pypi.org/project/midas-integrate/).

## Install

```bash
pip install 'midas-integrate[gpu]'
```

(which pulls both `midas-integrate` and `midas-integrate-gpu`)

Wheels are published for **Linux x86_64 only**, compute capability 7.5+
(Turing onwards). Build matrix: CUDA 12.4 runtime, CPython 3.9–3.12.

## Runtime requirements

- NVIDIA GPU, compute capability ≥ 7.5 (Turing, Ampere, Ada, Hopper).
- NVIDIA driver ≥ 525 (CUDA 12 forward-compat guarantee).
- Linux x86_64. macOS has no CUDA. Windows is on the v0.2 roadmap.

Verify before running:

```python
import midas_integrate_gpu as mig
env = mig.check_environment()
if not env:
    for problem in env.problems:
        print(f"  ✗ {problem}")
```

## How it works

Installing this wheel ships `MIDASIntegratorGPU` (the CUDA kernel
binary) into a `_bin/` folder that
`midas_integrate._binaries.midas_bin` already knows to check. There's
no new API; just pass `backend="gpu"` to the usual entry points:

```python
import midas_integrate as mi

integrator = mi.Integrator(cfg, artefacts, backend="gpu")
profiles = integrator.integrate(zarr_zip)       # runs on GPU

with mi.stream.Server(cfg, artefacts, backend="gpu") as srv:
    # Paper's "1700+ FPS on H100" path
    srv.serve_forever()
```

## Why a separate wheel

- Keeps CPU-only installs lean (no CUDA toolchain, no NVIDIA driver needed).
- Matches the CUDA 12 runtime distribution model — GPU wheels are
  Linux-only with tight driver compat; bundling with CPU forces every
  user to opt into it.
- Lets us bump CUDA versions independently of the main `midas-integrate`
  cadence.

## Limitations in v0.1.0

- Single-frame-at-a-time dispatch from Python — batch pipelining on GPU
  is v0.2.
- No Windows CUDA wheels (planned v0.3).
- Only the streaming-protocol kernel. The CPU-equivalent
  `IntegratorZarrOMP` GPU port is tracked for v0.2.

## License

BSD-3-Clause.
