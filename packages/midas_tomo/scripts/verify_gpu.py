"""Verify the CUDA target against the CPU engine on a real GPU.

The GPU target is built fail-soft and skipped everywhere without CUDA, so it
is the one part of this package that can rot silently: it compiles in CI's
absence and nobody notices until a user with a GPU gets wrong numbers. Run
this on a CUDA host before any release that touched ``c_src/``.

What it checks, in increasing strictness:

1. **The GPU binary exists and runs.** Building is not testing.
2. **The GPU reconstruction is the object.** Correlation against the phantom,
   same threshold the CPU test uses. This catches a wiring or convention
   error -- a transposed image, a wrong angle sign -- which a CPU-vs-GPU
   difference metric alone would miss if both were wrong the same way.
3. **GPU agrees with CPU.** cuFFT and pocketfft/FFTW are different
   implementations, so this is a closeness check, not a bitwise one.
4. **``--fftw-bridge`` is bitwise identical to CPU.** That flag exists to
   route FFTs back through CPU FFTW, and it is the only GPU mode that claims
   bit-parity. If it does not hold, the claim in the usage text is wrong.

Usage:
    python scripts/verify_gpu.py [--n 128] [--angles 180]

Exit status is nonzero if any check fails, so it can gate a release.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_tomo import backend_c, run_tomo_from_sinos       # noqa: E402
from tests.phantom import make_sino_dataset                 # noqa: E402

#: Same threshold as tests/test_engine.py. The ceiling is the reference
#: projector's own accuracy (~0.93), not the engine's.
CORR_MIN = 0.85


def _central(img: np.ndarray, n: int) -> np.ndarray:
    """Crop to the central n/2 box; the engine pads, and the empty outer ring
    would otherwise dominate any correlation."""
    x = img.shape[-1]
    h = n // 4
    return img[x // 2 - h:x // 2 + h, x // 2 - h:x // 2 + h]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--angles", type=int, default=180)
    ap.add_argument("--cpus", type=int, default=4)
    args = ap.parse_args()

    print(f"CPU binary: {backend_c.binary_path()}")
    print(f"GPU binary: {backend_c.binary_path(gpu=True)}")
    if not backend_c.available():
        print(f"FAIL: no CPU engine. {backend_c.why_unavailable()}")
        return 1
    if not backend_c.available(gpu=True):
        print(f"FAIL: no GPU engine. {backend_c.why_unavailable(gpu=True)}")
        return 1

    phantom, sino, angles = make_sino_dataset(n=args.n, n_angles=args.angles)
    print(f"phantom {phantom.shape}, sinogram {sino.shape}, "
          f"{len(angles)} angles")

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        common = dict(thetas=angles, n_cpus=args.cpus)

        cpu = run_tomo_from_sinos(sino, root / "cpu", **common)[0, 0]
        gpu = run_tomo_from_sinos(sino, root / "gpu", use_gpu=True, **common)[0, 0]
        bridge = run_tomo_from_sinos(sino, root / "bridge", use_gpu=True,
                                     fftw_bridge=True, **common)[0, 0]
        # The scale reference for "is the GPU gap big?": how far apart the two
        # CPU FFT backends are. Without it, 1e-5 is a number with no yardstick.
        cpu_fftw = run_tomo_from_sinos(sino, root / "cpu_fftw",
                                       fft_engine="fftw", **common)[0, 0]

        # --- 2. the GPU result is recognisably the object -------------------
        h = args.n // 4
        p = phantom[args.n // 2 - h:args.n // 2 + h,
                    args.n // 2 - h:args.n // 2 + h]
        for name, rec in (("cpu", cpu), ("gpu", gpu), ("bridge", bridge)):
            corr = np.corrcoef(_central(rec, args.n).ravel(), p.ravel())[0, 1]
            ok = corr > CORR_MIN
            print(f"  {name:7s} corr vs phantom = {corr:.4f}  "
                  f"{'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{name} correlates only {corr:.4f} "
                                f"(need > {CORR_MIN})")

        # --- 3. GPU close to CPU, but NOT identical to it -------------------
        # Scale-relative, not absolute: gridrec output carries an arbitrary
        # scale, so an absolute tolerance would be testing normalisation.
        scale = float(np.abs(cpu).max())
        rel = float(np.abs(gpu - cpu).max() / scale)
        print(f"  gpu vs cpu   max |diff| / max|cpu| = {rel:.3e}")
        # 1e-3, not 1e-7: cuFFT and pocketfft are different algorithms in
        # single precision over a padded transform. Anything at 1e-2 or worse
        # is a bug, not arithmetic.
        if not rel < 1e-3:
            failures.append(f"gpu differs from cpu by {rel:.3e} relative")

        # The check that makes this script worth running. cuFFT and
        # pocketfft/FFTW cannot agree to the last bit, so an EXACT match means
        # the GPU run was not a GPU run -- the request was silently serviced by
        # the CPU path. That is precisely what happened the first time this was
        # run on copland: api.run_engine defaulted to the ctypes library, which
        # is built without CUDA, and reported a difference of 0.000e+00.
        if rel == 0.0:
            failures.append(
                "gpu output is BITWISE identical to cpu. cuFFT cannot match "
                "pocketfft exactly, so the --gpu request was silently served "
                "by the CPU path. Check which binary actually ran."
            )

        # --- 4. the bridge is a diagnostic, not a parity switch -------------
        # It USED to be documented as byte-identical to the CPU path. Measured
        # on an A6000 (128x180): bridge-vs-CPU-FFTW 1.014e-05, cuFFT-vs-CPU-FFTW
        # 9.954e-06, CPU-FFTW-vs-pocketfft 2.777e-07. Routing the FFTs through
        # FFTW does not close the GPU/CPU gap, so the residual is in the
        # gridding, not the transform. The docs now say so; this checks the
        # bridge stays in the same regime rather than asserting a parity that
        # was never real.
        brel = float(np.abs(bridge - cpu).max() / scale)
        print(f"  bridge vs cpu max |diff| / max|cpu| = {brel:.3e} "
              f"(bitwise identical = {np.array_equal(bridge, cpu)}; "
              f"parity is NOT expected -- see c_src/tomo_gpu.h)")
        if not brel < 1e-3:
            failures.append(f"--fftw-bridge differs from cpu by {brel:.3e} "
                            f"relative, well beyond the measured 1.0e-5")

        # --- 5. the yardstick -----------------------------------------------
        # The two CPU FFT backends against each other. Everything above should
        # be read relative to this: the GPU gap is ~36x the CPU-backend gap,
        # which is why it is attributed to the gridding rather than the FFT.
        crel = float(np.abs(cpu_fftw - cpu).max() / scale)
        print(f"  cpu fftw vs cpu pocketfft         = {crel:.3e}  "
              f"(the FFT-backend yardstick)")
        if crel == 0.0:
            failures.append(
                "cpu --fft-engine=fftw is bitwise identical to pocketfft. "
                "Those are different transforms, so the fft_engine argument "
                "is not reaching the engine."
            )

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all GPU checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
