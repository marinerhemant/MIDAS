"""Command-line entry points.

Most are stubs pending v0.1.0's remaining weeks:
    - main_server / main_client   — Week 12 (this release)
    - main_integrate / main_batch / main_correct / main_benchmark — pending

All stubs return exit code 2 with a pointer to the release plan.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import numpy as np


def _not_implemented(name: str) -> int:
    sys.stderr.write(
        f"{name}: implementation pending. See "
        "packages/midas_integrate/ release plan for timeline.\n"
    )
    return 2


def main_integrate() -> int:
    return _not_implemented("midas-integrate")


def main_correct() -> int:
    return _not_implemented("midas-correct-image")


def main_batch() -> int:
    return _not_implemented("midas-integrate-batch")


def main_benchmark() -> int:
    return _not_implemented("midas-integrate-benchmark")


# ---------------------------------------------------------------------------
# Week 12: stream server + client CLIs
# ---------------------------------------------------------------------------

def main_server(argv: list[str] | None = None) -> int:
    """``midas-integrate-server`` — run a live-feed integration server."""
    parser = argparse.ArgumentParser(
        prog="midas-integrate-server",
        description="Start an in-process integration server for live frames.",
    )
    parser.add_argument("params_file", type=Path,
                        help="MIDAS Parameters.txt used to build the map and "
                             "seed each frame's zarr.zip.")
    parser.add_argument("--map-dir", type=Path, default=None,
                        help="Directory containing Map.bin + nMap.bin. "
                             "Default: same dir as the params file.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind interface (default: loopback).")
    parser.add_argument("--port", type=int, default=60439,
                        help="TCP port (default: 60439).")
    parser.add_argument("--listen-all", action="store_true",
                        help="Allow binding to non-loopback hosts. "
                             "v0.1.0 has no auth — only pass behind "
                             "a firewall.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)

    try:
        from ._config import IntegrationConfig
        from .mapper import MapArtifacts
        from .stream import Server
    except ImportError as e:
        sys.stderr.write(f"import error: {e}\n")
        return 2

    # For MVP, users supply a pre-built Map.bin + nMap.bin and a Parameters.txt
    # matching the config used to build them. A full "from-parameters"
    # bootstrap (run Mapper automatically) is a v0.2 convenience.
    if not args.params_file.exists():
        sys.stderr.write(f"params file missing: {args.params_file}\n")
        return 1
    map_dir = args.map_dir or args.params_file.parent
    map_bin = map_dir / "Map.bin"
    n_map_bin = map_dir / "nMap.bin"
    for p in (map_bin, n_map_bin):
        if not p.exists():
            sys.stderr.write(
                f"{p.name} not found in {map_dir}. Run "
                "`mi.Mapper(config).build(...)` first, or pass --map-dir.\n"
            )
            return 1

    # Parse the params file into an IntegrationConfig. MVP: user must set
    # all relevant fields via the text file — we just re-read geometry
    # so the Integrator knows what to pass into make_zarr_zip.
    cfg = _parse_config_from_txt(args.params_file)
    artifacts = MapArtifacts(
        work_dir=map_dir, map_bin=map_bin, n_map_bin=n_map_bin,
    )

    host = args.host
    if host != "127.0.0.1" and not args.listen_all:
        sys.stderr.write(
            f"--host {host} is non-loopback; pass --listen-all to confirm "
            "(v0.1.0 has no auth).\n"
        )
        return 1

    sys.stdout.write(
        f"midas-integrate-server starting on {host}:{args.port} "
        f"(backend={args.backend}, Ctrl-C to stop)\n"
    )
    sys.stdout.flush()
    server = Server(
        cfg, artifacts, host=host, port=args.port, backend=args.backend,
        listen_all_allowed=args.listen_all,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        sys.stdout.write("\nserver stopped.\n")
    return 0


def main_client(argv: list[str] | None = None) -> int:
    """``midas-integrate-client`` — feed TIFFs to a running server."""
    parser = argparse.ArgumentParser(
        prog="midas-integrate-client",
        description="Feed a sequence of TIFFs into a running integration server.",
    )
    parser.add_argument("endpoint",
                        help="host:port of the running server (e.g. 127.0.0.1:60439).")
    parser.add_argument("tiffs", type=Path, nargs="+",
                        help="TIFF frame(s) to integrate.")
    parser.add_argument("--out", type=Path, default=Path("."),
                        help="Output directory for cake .npy files (default: .)")
    args = parser.parse_args(argv)

    try:
        import tifffile
        from .stream import Client
    except ImportError as e:
        sys.stderr.write(f"import error: {e}\n")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    with Client(args.endpoint) as c:
        for tif in args.tiffs:
            frame = tifffile.imread(tif)
            cake = c.send_frame(frame)
            dest = args.out / f"{tif.stem}.cake.npy"
            np.save(dest, cake)
            sys.stdout.write(f"  {tif.name} -> {dest.name} (shape={cake.shape})\n")
    return 0


def _parse_config_from_txt(path: Path):
    """Minimal Parameters.txt → IntegrationConfig for the server CLI.

    Only reads the keys make_zarr_zip needs: Lsd, BC, tx/ty/tz, px,
    NrPixels*, Wavelength, RMin/RMax/RBinSize, EtaMin/EtaMax/EtaBinSize,
    RhoD, p0..p14. Any keys not listed stay at IntegrationConfig defaults.
    """
    from ._config import IntegrationConfig

    cfg = IntegrationConfig()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        key = tokens[0]
        vals = tokens[1:]
        try:
            if key == "Lsd":        cfg.lsd = float(vals[0])
            elif key == "BC":       cfg.ybc, cfg.zbc = float(vals[0]), float(vals[1])
            elif key == "tx":       cfg.tx = float(vals[0])
            elif key == "ty":       cfg.ty = float(vals[0])
            elif key == "tz":       cfg.tz = float(vals[0])
            elif key == "Wavelength": cfg.wavelength = float(vals[0])
            elif key == "px":       cfg.pixel_size = float(vals[0])
            elif key == "NrPixelsY": cfg.nr_pixels_y = int(vals[0])
            elif key == "NrPixelsZ": cfg.nr_pixels_z = int(vals[0])
            elif key == "RhoD":     cfg.rho_d = float(vals[0])
            elif key == "RMin":     cfg.r_min = float(vals[0])
            elif key == "RMax":     cfg.r_max = float(vals[0])
            elif key == "RBinSize": cfg.r_bin_size = float(vals[0])
            elif key == "EtaMin":   cfg.eta_min = float(vals[0])
            elif key == "EtaMax":   cfg.eta_max = float(vals[0])
            elif key == "EtaBinSize": cfg.eta_bin_size = float(vals[0])
            elif key.startswith("p") and key[1:].isdigit():
                i = int(key[1:])
                if 0 <= i <= 14:
                    setattr(cfg, f"p{i}", float(vals[0]))
        except (ValueError, IndexError):
            continue
    return cfg
