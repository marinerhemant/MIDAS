# midas-tomo

Gridrec filtered back-projection CT reconstruction, with a NumPy API.

Gridrec is a Fourier-domain FBP that uses prolate spheroidal wave function
(PSWF) interpolation onto the polar grid. This package wraps the MIDAS
implementation: a `MIDAS_TOMO` binary compiled at install time from the C in
`c_src/`, plus a CUDA variant (`MIDAS_TOMO_GPU`) where a CUDA toolkit is
present.

```python
from midas_tomo import run_tomo
recon = run_tomo(data, dark, whites, "/path/to/work", thetas, shifts=1.0)
```

## Installing

```bash
pip install midas-tomo
```

The binary needs a C/C++ compiler and **OpenMP**. That is all — the FFT comes
from **pocketfft**, which is BSD-3-Clause and vendored in `c_src/vendor/`, so
no external FFT library is required.

```bash
brew install libomp                        # macOS only; Linux gcc has libgomp
```

**pocketfft is the default engine.** FFTW3f is optional and selectable with
`--fft-engine=fftw`; use it to reproduce historical runs bit-for-bit. It is
not shipped, because FFTW is GPL-2.0-or-later and this package is BSD-3-Clause
— a wheel containing FFTW would have to be distributed under the GPL.

Measured on one machine (128×180 phantom): pocketfft agrees with FFTW to
2.9e-07 relative, is **reproducible run-to-run where FFTW is not**, and ran
**0.86× the time** of FFTW — slightly faster, not slower.

**HDF5 is not required.** This package reads HDF5 in Python with `h5py` and hands
the engine a staged binary, so the C never needs an HDF5 reader — and h5py copes
with more layouts, compression filters and chunkings than the C one did. Use
`stage_exchange_to_binary()` for scans too large to hold in memory; it streams
the conversion a chunk of frames at a time. If HDF5 *is* present at build time the
C reader is compiled in as well, for hand-written parameter files that set
`HDF5FileName`.

then reinstall. To check what you have:

```python
import midas_tomo
midas_tomo.backend_c.available()            # CPU engine present?
midas_tomo.backend_c.available(gpu=True)    # CUDA engine present?
print(midas_tomo.backend_c.why_unavailable())
```

There are no binary wheels. The sdist compiles from source at install time,
which is deliberate: FFTW is GPL-2.0-or-later, so a wheel with FFTW linked
into it would have to be distributed under the GPL, while this package is
BSD-3-Clause. Building against *your* FFTW keeps that boundary clean.

## Relationship to `TOMO/`

`c_src/` is a byte-identical mirror of `TOMO/src/` at commit `0a426739`, and
is **canonical** from that point on — future fixes land here, and `TOMO/`
stays frozen. See `c_src/FORK.txt`. A CI job diffs the two and warns on any
divergence, so the mirror is kept clean on purpose: an unexplained diff is a
signal, not noise.

Two things recorded there are worth knowing before you touch the build:

- The binary is compiled `-fPIC -O3 -w -g` **unconditionally**, not because
  of a build type. `Cnvlvnt()` is a C99 `inline` with no extern definition, so
  the link only succeeds when the optimiser inlines every call — at `-O0` it
  fails outright.
- Those flags are part of the bit-parity pin. Optimisation level changes
  floating-point instruction selection in the gridrec inner loops, so changing
  them changes the reconstruction's low-order bits.

## Reproducibility

By default the CPU engine plans its FFTs with `FFTW_MEASURE` and caches the
result in `fftwf_wisdom_{1,2}d_<N>.txt` **in the current working directory**.
`FFTW_MEASURE` chooses a plan by *timing* candidates, so the choice depends on
the machine and on transient load.

Measured on one machine, three trials: **two identical runs in two fresh
directories disagreed in 2 of 3 trials** — at the ~1e-7 relative level, i.e. a
few float32 ulps. So the default path is not reproducible run to run, let alone
across machines. Reconstructions are unaffected scientifically; byte-comparisons
are not.

For reproducibility use the deterministic planner (`FFTW_ESTIMATE`, no timing,
no wisdom side-effect):

```python
run_tomo(..., deterministic=True)
```

Be precise about what that buys. Measured: repeated runs of the *same* backend
are then bitwise identical. It does **not** make different contexts agree — the
in-process library and the CLI binary still differ by ~3e-7 relative, and
consistently so. Reproducibility is scoped to one build and one process
context, not absolute.

## License

BSD-3-Clause. See `LICENSE`.
