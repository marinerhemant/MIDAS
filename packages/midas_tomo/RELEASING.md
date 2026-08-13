# Releasing midas-tomo

```bash
cd packages/midas_tomo
./release.sh 0.1.0 --dry-run     # build + test, no commit or tag
./release.sh 0.1.0               # + commit and tag locally
./release.sh 0.1.0 --publish     # + push, GitHub release, CI publishes to PyPI
```

The tag is `midas-tomo-v<version>`; `.github/workflows/python-packages.yml`
maps it to this package and publishes to PyPI via trusted publishing.

## Release from a machine that can build the engine

This is the one thing about this package that differs from the pure-Python
ones, and it is easy to get wrong.

The engine tests **skip** when `MIDAS_TOMO` is not built, so `pytest` goes
green on a laptop without FFTW while never once exercising the C. Expected
counts:

| environment | result |
|---|---|
| engine + `MIDAS_TOMO_REFERENCE_BIN` set | **112 passed** |
| no engine at all (e.g. a Mac with no FFTW) | 89 passed, 3 skipped |

`release.sh` passes `-rs` so the skip reasons are printed. If you see the
89-passed line, you tested the wrapper and nothing else — release from a host
with FFTW, or run the suite there first.

## The parity gate

`tests/test_parity_vs_fork.py` is the evidence that this package still
computes what `TOMO/` computed. It needs a reference binary built from the
pre-fork source:

```bash
./scripts/build_reference_binary.sh /path/to/a/MIDAS/checkout
export MIDAS_TOMO_REFERENCE_BIN=$PWD/dev/refbin/MIDAS_TOMO_REF
```

Set `FFTW_PREFIX` if FFTW is not on the default search path. Run this before
any release that touched `c_src/`.

## Reproducibility caveat when comparing releases

Do not diff reconstructions byte-for-byte across builds or machines and treat
a difference as a regression. Measured: the default `FFTW_MEASURE` planner
picks by timing, so two identical runs disagreed in 2 of 3 trials at ~1e-7
relative. `deterministic=True` fixes that *within* one backend and one build,
not across them. See the Reproducibility section of `README.md`.

## Before a release that touched `c_src/`

- [ ] `scripts/build_reference_binary.sh` + parity test pass
- [ ] builds **with and without** HDF5 (it is optional; the Python reader is
      the supported path)
- [ ] any new divergence from the fork is recorded in `c_src/FORK.txt` with
      its reason
- [ ] `tests/test_library.py` passes — it covers the in-process path, where
      an error the CLI merely printed becomes a silent wrong answer
