"""Measure the pocketfft backend against FFTW: correctness, agreement, speed.

Run on a host with both builds present:

    cd <tomo_buildtest>
    LD_LIBRARY_PATH=<fftw>/lib PYTHONPATH=. python dev/verify_fft_engines.py

Expects:
  bt/MIDAS_TOMO          built WITH FFTW  (both engines selectable)
  bt_nofftw/MIDAS_TOMO   built WITHOUT it (pocketfft only)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from midas_tomo import TomoConfig, backend_c          # noqa: E402
from midas_tomo.api import read_recon_cube, run_tomo_from_sinos  # noqa: E402
from tests.phantom import make_sino_dataset            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FFTW_BIN = str(backend_c.binary_path())
NOFFTW_BIN = str(ROOT / "bt_nofftw" / "MIDAS_TOMO")


def prep(sino, angles, wd: Path) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    run_tomo_from_sinos(sino, wd, angles, n_cpus=4, backend="subprocess",
                        do_cleanup=False)
    return wd


def run(binary: str, wd: Path, extra: list[str], n: int = 1):
    t0 = time.perf_counter()
    for _ in range(n):
        p = subprocess.run([binary, str(wd / "midastomo.par"), "4", *extra],
                           cwd=str(wd), capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  FAIL rc={p.returncode} {extra}: {p.stderr.strip()[-200:]}")
            return None, 0.0
    dt = (time.perf_counter() - t0) / n
    cube, _ = read_recon_cube(TomoConfig.from_param_file(wd / "midastomo.par"), 2)
    return cube, dt


def main() -> int:
    phantom, sino, angles = make_sino_dataset(n=128, n_angles=180)
    td = Path(tempfile.mkdtemp(prefix="fftcmp_"))

    print("1. the FFTW-free build actually reconstructs")
    a, _ = run(NOFFTW_BIN, prep(sino, angles, td / "nofftw"), [])
    if a is None:
        return 1
    x, h = a.shape[-1], 32
    # read_recon_cube gives (n_shifts, n_slices, X, X): index BOTH.
    crop = a[0, 0][x // 2 - h:x // 2 + h, x // 2 - h:x // 2 + h]
    corr = np.corrcoef(crop.ravel(), phantom[64 - h:64 + h, 64 - h:64 + h].ravel())[0, 1]
    print(f"   correlation with the phantom: {corr:+.4f}")

    print("2. FFTW-free build vs FFTW build")
    b, _ = run(FFTW_BIN, prep(sino, angles, td / "fftwref"),
               ["--deterministic", "--fft-engine=fftw"])
    if b is not None:
        scale = float(np.abs(b).max())
        rel = float(np.abs(a - b).max()) / scale
        print(f"   max relative difference: {rel:.2e}  (bitwise: {np.array_equal(a, b)})")

    print("3. is pocketfft reproducible without any --deterministic flag?")
    p1, _ = run(FFTW_BIN, prep(sino, angles, td / "p1"), ["--fft-engine=pocketfft"])
    p2, _ = run(FFTW_BIN, prep(sino, angles, td / "p2"), ["--fft-engine=pocketfft"])
    if p1 is not None and p2 is not None:
        print(f"   two fresh runs bitwise identical: {np.array_equal(p1, p2)}")

    print("4. and is FFTW, for comparison?")
    f1, _ = run(FFTW_BIN, prep(sino, angles, td / "f1"), ["--fft-engine=fftw"])
    f2, _ = run(FFTW_BIN, prep(sino, angles, td / "f2"), ["--fft-engine=fftw"])
    if f1 is not None and f2 is not None:
        print(f"   two fresh runs bitwise identical: {np.array_equal(f1, f2)}")

    print("5. speed")
    _, tf = run(FFTW_BIN, prep(sino, angles, td / "bf"), ["--fft-engine=fftw"], n=5)
    _, tp = run(FFTW_BIN, prep(sino, angles, td / "bp"), ["--fft-engine=pocketfft"], n=5)
    if tf and tp:
        print(f"   fftw {tf * 1000:.1f} ms   pocketfft {tp * 1000:.1f} ms   "
              f"ratio {tp / tf:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
