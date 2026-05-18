"""Performance benchmark — compare v1, v2 binning kernels, and (if
installed) pyFAI on a Pilatus-sized detector.

Run with:

    python bench/bench_integrate.py

Prints a table of build + integrate times for a fixed image size +
geometry. Useful for spotting performance regressions.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@contextmanager
def _timer(name, results: Dict[str, float]):
    t0 = time.perf_counter()
    yield
    results[name] = time.perf_counter() - t0


def benchmark(
    NY: int = 1475, NZ: int = 1679,
    *, RBinSize: float = 1.0, EtaBinSize: float = 5.0,
    n_integrate_trials: int = 5,
):
    print(f"Benchmark on {NY}×{NZ} = {NY*NZ:,} pixels detector")
    print(f"Bin sizes: RBinSize={RBinSize} px, EtaBinSize={EtaBinSize} deg\n")

    results: Dict[str, Dict[str, float]] = {}

    # Build the spec
    from midas_integrate.params import IntegrationParams
    from midas_integrate_v2 import spec_from_v1_params
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=172.0, pxZ=172.0, Lsd=657_437.0,
        BC_y=NY/2 + 0.37, BC_z=NZ/2 - 0.41,
        RhoD=float(NY),
        RMin=10.0, RMax=float(min(NY, NZ) // 2 - 5), RBinSize=RBinSize,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBinSize,
        Wavelength=0.172979,
    )
    spec = spec_from_v1_params(p, requires_grad=False)
    print(f"Bins: {spec.n_eta_bins} η × {spec.n_r_bins} R = "
          f"{spec.n_eta_bins * spec.n_r_bins:,} bins\n")

    # Synthetic image
    rng = np.random.default_rng(0)
    image = rng.uniform(1.0, 100.0, size=(NZ, NY)).astype(np.float64)
    image_t = torch.from_numpy(image)

    # === v2: HardBinGeometry ===
    from midas_integrate_v2 import HardBinGeometry, integrate_hard
    r = {}
    with _timer("build", r):
        geom_h = HardBinGeometry.from_spec(spec)
    integrate_hard(image_t, geom_h)   # warm up
    with _timer("integrate", r):
        for _ in range(n_integrate_trials):
            integrate_hard(image_t, geom_h)
    r["integrate"] /= n_integrate_trials
    results["v2 hard"] = r

    # === v2: SubpixelBinGeometry K=2 ===
    from midas_integrate_v2 import SubpixelBinGeometry, integrate_subpixel
    r = {}
    with _timer("build", r):
        geom_s = SubpixelBinGeometry.from_spec(spec, K=2)
    integrate_subpixel(image_t, geom_s)
    with _timer("integrate", r):
        for _ in range(n_integrate_trials):
            integrate_subpixel(image_t, geom_s)
    r["integrate"] /= n_integrate_trials
    results["v2 subpixel K=2"] = r

    # === v2: PolygonBinGeometry (parallel) ===
    from midas_integrate_v2 import PolygonBinGeometry, integrate_polygon
    r = {}
    with _timer("build", r):
        geom_p = PolygonBinGeometry.from_spec(spec, n_jobs=-1)
    integrate_polygon(image_t, geom_p)
    with _timer("integrate", r):
        for _ in range(n_integrate_trials):
            integrate_polygon(image_t, geom_p)
    r["integrate"] /= n_integrate_trials
    results["v2 polygon"] = r

    # === v1 hot path (numba) ===
    try:
        from midas_integrate.detector_mapper import build_map as v1_build_map
        from midas_integrate.bin_io import PixelMap
        from midas_integrate.kernels import build_csr, integrate as v1_integrate
        from midas_integrate_v2 import v1_params_from_spec
        p1 = v1_params_from_spec(spec)
        r = {}
        with _timer("build", r):
            res = v1_build_map(p1, auto_load=False, verbose=False)
        pm = PixelMap(pxList=res.pxList, counts=res.counts,
                       offsets=res.offsets, map_header=None, nmap_header=None)
        csr = build_csr(pm, n_r=p1.n_r_bins, n_eta=p1.n_eta_bins,
                          n_pixels_y=NY, n_pixels_z=NZ,
                          device="cpu", dtype=torch.float64,
                          bc_y=p1.BC_y, bc_z=p1.BC_z)
        v1_integrate(image_t, csr, mode="floor", normalize=True)
        with _timer("integrate", r):
            for _ in range(n_integrate_trials):
                v1_integrate(image_t, csr, mode="floor", normalize=True)
        r["integrate"] /= n_integrate_trials
        results["v1 hot path (numba)"] = r
    except Exception as e:
        print(f"v1 path skipped: {e}")

    # === pyFAI (if installed) ===
    try:
        import pyFAI
        # pyFAI uses pixel-corner convention; MIDAS uses pixel-centre.
        # Convert via PONI = (BC + 0.5) · pixel_size. Skipping the +0.5
        # gives a half-pixel BC shift that biases all peak positions.
        ai = pyFAI.AzimuthalIntegrator(
            dist=p.Lsd * 1e-6,                 # m
            poni1=(p.BC_y + 0.5) * p.pxY * 1e-6,
            poni2=(p.BC_z + 0.5) * p.pxZ * 1e-6,
            pixel1=p.pxY * 1e-6, pixel2=p.pxZ * 1e-6,
            wavelength=p.Wavelength * 1e-10,    # m
        )
        r = {}
        with _timer("build", r):
            ai.integrate1d(image, spec.n_r_bins, method="splitpixel")
        with _timer("integrate", r):
            for _ in range(n_integrate_trials):
                ai.integrate1d(image, spec.n_r_bins, method="splitpixel")
        r["integrate"] /= n_integrate_trials
        results["pyFAI splitpixel"] = r
    except ImportError:
        print("pyFAI not installed; skipping that comparison\n")
    except Exception as e:
        print(f"pyFAI failed: {e}\n")

    # Print results
    print("\n" + "=" * 65)
    print(f"{'Method':<28}  {'Build (s)':>10}  {'Integrate (ms)':>15}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<28}  {r['build']:>10.2f}  "
              f"{r['integrate']*1000:>15.2f}")
    print("=" * 65)

    # Compare v2 polygon vs v1 hot path (the headline comparison)
    if "v2 polygon" in results and "v1 hot path (numba)" in results:
        v2 = results["v2 polygon"]; v1 = results["v1 hot path (numba)"]
        print(f"\nv2 polygon vs v1 hot path:")
        print(f"  build: v2 is {v2['build']/v1['build']:.1f}× of v1's")
        print(f"  integrate: v2 is {v2['integrate']/v1['integrate']:.1f}× of v1's")
        print("  (build amortises across many frames; integrate is the hot path)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=1475, help="detector NY")
    ap.add_argument("--nz", type=int, default=1679, help="detector NZ")
    ap.add_argument("--rbin", type=float, default=1.0)
    ap.add_argument("--etabin", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    benchmark(
        NY=args.ny, NZ=args.nz,
        RBinSize=args.rbin, EtaBinSize=args.etabin,
        n_integrate_trials=args.trials,
    )
