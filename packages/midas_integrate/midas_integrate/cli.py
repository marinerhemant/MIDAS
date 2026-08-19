"""Command-line entry points for midas-integrate.

Four console scripts (registered by pyproject.toml):

  midas-detector-mapper      PARAMS [--out DIR] [-j N]
  midas-integrate            PARAMS --image FILE [--device cuda] [--mode floor|bilinear|gradient] ...
  midas-integrate-server     PARAMS [--device cuda] [--port 60439] [--out DIR] ...
  midas-integrate-export-csv ZARR --out DIR [--frames 0-99] [--include-cake]

These mirror the C/CUDA binaries (DetectorMapper, IntegratorZarrOMP,
IntegratorFitPeaksGPUStream); ``export-csv`` is the Python-side answer to
issue #23 (clean CSV dump of integrated zarr output).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from midas_integrate.bin_io import load_map

# Local import of __version__ avoids the package's eager import chain so
# the CLI can start fast and report errors clearly.
def _version() -> str:
    try:
        from midas_integrate import __version__
        return __version__
    except Exception:
        return "unknown"
__version__ = _version()
from midas_integrate.detector_mapper import build_and_write_map
from midas_integrate.image import average_dark_frames, load_image
from midas_integrate.kernels import build_csr, integrate, profile_1d, r_axis
from midas_integrate.params import parse_params
from midas_integrate.server import FrameServer

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-integrate"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("params", type=Path, help="MIDAS parameter file (.txt)")
    p.add_argument("--map-dir", type=Path, default=None,
                   help="Directory containing Map.bin/nMap.bin "
                        "(default: same dir as params file)")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate {__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# midas-detector-mapper
# ─────────────────────────────────────────────────────────────────────────────
def detector_mapper_main(argv=None) -> int:
    p = _midas_make_parser(
        prog="midas-detector-mapper",
        description="Build Map.bin / nMap.bin from a MIDAS parameter file.",
    )
    p.add_argument("params", type=Path, help="MIDAS parameter file (.txt)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: directory of params file)")
    p.add_argument("--flat", type=Path, default=None,
                   help="Optional per-pixel sensitivity map (TIFF or raw). "
                        "Each pxList weight is divided by flat[pix] at map-build "
                        "time; integration cost is unchanged. Overrides "
                        "FlatFile in the parameter file if provided.")
    p.add_argument("--mask", type=Path, default=None,
                   help="Optional binary pixel mask (TIFF or raw); 1 = masked.")
    p.add_argument("-j", "--jobs", type=int, default=-1,
                   help="Number of joblib workers (default: -1 = all CPUs)")
    p.add_argument("--per-row-max", type=int, default=None,
                   help="Entries a single detector row may contribute to the "
                        "map. Default is derived from RBinSize/EtaBinSize/"
                        "SubPixelLevel. Raise it if you get a map-truncation "
                        "warning; lower it to cap memory.")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate {__version__}")
    args = p.parse_args(argv)

    params = parse_params(args.params)
    out_dir = args.out or args.params.parent

    flat_path = args.flat or (Path(params.FlatFile) if params.FlatFile else None)
    mask_path = args.mask or (Path(params.MaskFile) if params.MaskFile else None)
    flat_arr = (load_image(flat_path,
                           n_pixels_y=params.NrPixelsY,
                           n_pixels_z=params.NrPixelsZ).astype(np.float64)
                if flat_path is not None else None)
    mask_arr = (load_image(mask_path,
                           n_pixels_y=params.NrPixelsY,
                           n_pixels_z=params.NrPixelsZ).astype(np.float64)
                if mask_path is not None else None)

    map_path, nmap_path = build_and_write_map(
        params, output_dir=out_dir,
        mask=mask_arr, flat=flat_arr,
        n_jobs=args.jobs,
        per_row_max_entries=args.per_row_max,
        verbose=not args.quiet,
    )
    print(f"wrote: {map_path}")
    print(f"       {nmap_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# midas-integrate (one-shot)
# ─────────────────────────────────────────────────────────────────────────────
def integrate_main(argv=None) -> int:
    p = _midas_make_parser(
        prog="midas-integrate",
        description="Integrate one detector frame against an existing Map.bin.",
    )
    _add_common_args(p)
    p.add_argument("--image", type=Path, required=True, help="Image (TIFF or raw)")
    p.add_argument("--raw-dtype", default=None,
                   help="numpy dtype string for raw binary input "
                        "(e.g. 'uint16'); not needed for TIFF")
    p.add_argument("--dark", type=Path, default=None,
                   help="Optional multi-frame dark file (averaged)")
    p.add_argument("--dark-dtype", default="int64",
                   help="dtype of dark file (default int64)")
    p.add_argument("--mode", default="floor",
                   choices=["floor", "bilinear", "gradient"])
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | cuda:0 | mps | …")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--out", type=Path, default=Path("."),
                   help="Where to write output files")
    args = p.parse_args(argv)

    params = parse_params(args.params)
    params.validate()
    map_dir = args.map_dir or args.params.parent

    pixmap = load_map(map_dir / "Map.bin", map_dir / "nMap.bin")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    geom = build_csr(
        pixmap, n_r=params.n_r_bins, n_eta=params.n_eta_bins,
        n_pixels_y=params.NrPixelsY, n_pixels_z=params.NrPixelsZ,
        device=args.device, dtype=torch_dtype,
        bc_y=params.BC_y, bc_z=params.BC_z,
    )
    image = load_image(args.image,
                       n_pixels_y=params.NrPixelsY,
                       n_pixels_z=params.NrPixelsZ,
                       raw_dtype=args.raw_dtype)
    img_t = torch.from_numpy(image.astype(np.float64)).to(
        device=args.device, dtype=torch_dtype
    )
    if args.dark is not None:
        dark = average_dark_frames(args.dark,
                                   n_pixels_y=params.NrPixelsY,
                                   n_pixels_z=params.NrPixelsZ,
                                   dtype=args.dark_dtype)
        dark_t = torch.from_numpy(dark.astype(np.float64)).to(
            device=args.device, dtype=torch_dtype
        )
        img_t = img_t - dark_t

    t0 = time.perf_counter()
    int2d = integrate(img_t, geom, mode=args.mode,
                      normalize=bool(params.Normalize))
    prof_aw = profile_1d(int2d, geom, mode="area_weighted")
    prof_sm = profile_1d(int2d, geom, mode="simple_mean")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"integrate+profile: {elapsed_ms:.3f} ms (mode={args.mode}, "
          f"device={args.device})")

    args.out.mkdir(parents=True, exist_ok=True)
    r_vals = r_axis(n_r=params.n_r_bins,
                    RMin=params.RMin, RBinSize=params.RBinSize)
    pairs = np.empty(params.n_r_bins * 2, dtype=np.float64)
    pairs[0::2] = r_vals
    pairs[1::2] = prof_aw.detach().cpu().numpy().astype(np.float64)
    (args.out / "lineout.bin").write_bytes(pairs.tobytes())
    pairs[1::2] = prof_sm.detach().cpu().numpy().astype(np.float64)
    (args.out / "lineout_simple_mean.bin").write_bytes(pairs.tobytes())
    if params.Write2D:
        (args.out / "Int2D.bin").write_bytes(
            int2d.detach().cpu().numpy().astype(np.float64).tobytes()
        )
    print(f"wrote: {args.out / 'lineout.bin'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# midas-integrate-server (streaming)
# ─────────────────────────────────────────────────────────────────────────────
def server_main(argv=None) -> int:
    p = _midas_make_parser(
        prog="midas-integrate-server",
        description="Streaming integrator server (TCP, port 60439 by default).",
    )
    _add_common_args(p)
    p.add_argument("--dark", type=Path, default=None)
    p.add_argument("--dark-dtype", default="int64")
    p.add_argument("--mode", default="floor",
                   choices=["floor", "bilinear", "gradient"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--port", type=int, default=60439)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--no-simple-mean", action="store_true")
    args = p.parse_args(argv)

    params = parse_params(args.params)
    params.validate()
    map_dir = args.map_dir or args.params.parent
    pixmap = load_map(map_dir / "Map.bin", map_dir / "nMap.bin")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    geom = build_csr(
        pixmap, n_r=params.n_r_bins, n_eta=params.n_eta_bins,
        n_pixels_y=params.NrPixelsY, n_pixels_z=params.NrPixelsZ,
        device=args.device, dtype=torch_dtype,
        bc_y=params.BC_y, bc_z=params.BC_z,
    )

    dark = None
    if args.dark is not None:
        dark_arr = average_dark_frames(args.dark,
                                       n_pixels_y=params.NrPixelsY,
                                       n_pixels_z=params.NrPixelsZ,
                                       dtype=args.dark_dtype)
        dark = torch.from_numpy(dark_arr.astype(np.float64)).to(
            device=args.device, dtype=torch_dtype
        )

    server = FrameServer(
        geom=geom, params=params,
        out_dir=args.out, integration_mode=args.mode,
        host=args.host, port=args.port,
        queue_size=args.queue_size,
        write_2d=bool(params.Write2D),
        write_simple_mean=not args.no_simple_mean,
        dark=dark,
    )
    server.start()
    print(f"midas-integrate-server listening on {args.host}:{args.port} "
          f"(device={args.device}, mode={args.mode})")
    print("Ctrl-C to stop.")

    stop = False
    def _handler(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (AttributeError, ValueError):
        pass
    while not stop:
        time.sleep(0.5)
    print("shutting down...")
    server.stop()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# midas-integrate-export-csv
# ─────────────────────────────────────────────────────────────────────────────
def export_csv_main(argv=None) -> int:
    p = _midas_make_parser(
        prog="midas-integrate-export-csv",
        description="Export per-frame lineouts and the REtaMap from an "
                    "integrated zarr (.zarr.zip) as plain CSVs.",
    )
    p.add_argument("zarr", type=Path,
                   help="Path to integrated zarr.zip (output of "
                        "IntegratorZarrOMP or midas-integrate)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (created if needed)")
    p.add_argument("--frames", type=str, default="all",
                   help="Frame selector: 'all', '0-99', '0,5,10', etc.")
    p.add_argument("--include-cake", action="store_true",
                   help="Also dump the full 2D cake for each selected frame")
    p.add_argument("--no-summed", action="store_true",
                   help="Skip the all-frame sum lineout")
    p.add_argument("--no-metadata", action="store_true",
                   help="Skip the metadata sidecar")
    p.add_argument("--no-retamap", action="store_true",
                   help="Skip the REtaMap (R/2θ/η/area/Q) sidecar")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate {__version__}")
    args = p.parse_args(argv)

    if not args.zarr.exists():
        p.error(f"input does not exist: {args.zarr}")

    from midas_integrate.exporters import export
    written = export(
        args.zarr, args.out,
        frames=args.frames,
        include_cake=args.include_cake,
        include_summed=not args.no_summed,
        include_metadata=not args.no_metadata,
        include_retamap=not args.no_retamap,
    )
    for p_out in written:
        print(f"wrote: {p_out}")
    if not written:
        print("nothing written — input zarr had no recognised datasets")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(detector_mapper_main())
