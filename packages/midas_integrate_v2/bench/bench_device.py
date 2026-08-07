"""Cross-device hot-path benchmark for MIDAS radial integration.

Times the v1 CSR sparse-matmul integration kernel (the *fastest* path —
a single SpMV, no transpose/copy) on a given device (cpu / cuda / mps).
The hot path is a sparse-CSR SpMV, so this is essentially a
memory-bandwidth benchmark — useful for comparing GPUs (A100 / H200 /
A6000 / GB10) and CPUs on the same workload.

Only depends on `midas-integrate` (v1). Run:
    python bench_device.py --device cuda --dtype float32
    python bench_device.py --device cuda --dtype float64
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import platform
import time

import numpy as np
import torch


def device_label(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.machine()}, {os.cpu_count()} cores)"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run(device: str, dtype: torch.dtype,
        NY: int, NZ: int, RBinSize: float, EtaBinSize: float,
        trials: int, warmup: int):
    from midas_integrate.params import IntegrationParams
    from midas_integrate.detector_mapper import build_map
    from midas_integrate.bin_io import PixelMap
    from midas_integrate.kernels import build_csr, integrate

    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=172.0, pxZ=172.0, Lsd=657_437.0,
        BC_y=NY / 2 + 0.37, BC_z=NZ / 2 - 0.41,
        RhoD=float(NY),
        RMin=10.0, RMax=float(min(NY, NZ) // 2 - 5), RBinSize=RBinSize,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBinSize,
        Wavelength=0.172979,
    )
    n_r = p.n_r_bins
    n_eta = p.n_eta_bins
    n_bins = n_r * n_eta

    # Build the pixel->bin map on CPU (numba), then the CSR on the target device.
    t0 = time.perf_counter()
    res = build_map(p, auto_load=False, verbose=False)
    pm = PixelMap(pxList=res.pxList, counts=res.counts, offsets=res.offsets,
                  map_header=None, nmap_header=None)
    csr = build_csr(pm, n_r=n_r, n_eta=n_eta,
                    n_pixels_y=NY, n_pixels_z=NZ,
                    device=device, dtype=dtype,
                    bc_y=p.BC_y, bc_z=p.BC_z)
    build_s = time.perf_counter() - t0
    nnz = csr.csr_floor._nnz()

    rng = np.random.default_rng(0)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    image = torch.from_numpy(
        rng.uniform(1.0, 100.0, size=(NZ, NY)).astype(np_dtype)
    ).to(device=device, dtype=dtype)

    for _ in range(warmup):
        integrate(image, csr, mode="floor", normalize=True)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(trials):
        integrate(image, csr, mode="floor", normalize=True)
    sync(device)
    integrate_s = (time.perf_counter() - t0) / trials

    fps = 1.0 / integrate_s
    print(f"{'='*70}")
    print(f"Device      : {device_label(device)}  [{device}]")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"dtype       : {str(dtype).replace('torch.','')}")
    print(f"Detector    : {NY}x{NZ} = {NY*NZ:,} px   "
          f"Bins: {n_eta} eta x {n_r} R = {n_bins:,}")
    print(f"CSR nnz     : {nnz:,}")
    print(f"{'-'*70}")
    print(f"build       : {build_s*1000:9.2f} ms  (one-time, amortised)")
    print(f"integrate   : {integrate_s*1000:9.3f} ms/frame   "
          f"({fps:,.0f} frames/s)")
    print(f"{'='*70}")
    return dict(device=device, label=device_label(device),
                torch=torch.__version__, cuda=torch.version.cuda,
                dtype=str(dtype), build_ms=build_s * 1000,
                integrate_ms=integrate_s * 1000, fps=fps,
                ny=NY, nz=NZ, n_bins=n_bins, nnz=int(nnz))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--ny", type=int, default=2048)
    ap.add_argument("--nz", type=int, default=2048)
    ap.add_argument("--rbin", type=float, default=0.25)
    ap.add_argument("--etabin", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")

    res = run(args.device, dt, args.ny, args.nz, args.rbin, args.etabin,
              args.trials, args.warmup)
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
