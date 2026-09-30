"""``midas-nf-preprocess process-images`` subcommand.

Mirrors the C ``ProcessImagesCombined`` executable's positional argv:

    midas-nf-preprocess process-images <ParameterFile> <LayerNr>
        [--device cuda] [--dtype fp32] [--n-cpus 8] [--all-layers]

If ``--all-layers`` is passed, ``LayerNr`` is ignored and every distance is
processed in a single invocation (Python-native API).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from .params import ProcessParams
from .pipeline import ProcessImagesPipeline


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("parameter_file", help="MIDAS parameter file (C-format keys).")
    parser.add_argument("layer_nr", type=int, help="1-indexed layer number.")
    parser.add_argument(
        "--device", default=None, help="cpu | cuda | mps (default: auto-detect)."
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="fp32 | fp64 (default: fp64 on cpu, fp32 on cuda/mps).",
    )
    parser.add_argument(
        "--n-cpus", type=int, default=0, help="torch.set_num_threads on CPU."
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Process all distances in one go (Python-native).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override SpotsInfo.bin output path (default: <output_directory>/SpotsInfo.bin).",
    )


def run(args: argparse.Namespace) -> int:
    params = ProcessParams.from_paramfile(args.parameter_file)
    pipe = ProcessImagesPipeline(
        params, device=args.device, dtype=args.dtype, n_cpus=args.n_cpus
    )
    if args.all_layers:
        bitmask = pipe.process_all()
    else:
        bitmask = pipe.process_layer(args.layer_nr)

    out = args.output or str(Path(params.output_directory) / "SpotsInfo.bin")
    bitmask.write(out)
    print(f"Wrote {out} ({bitmask.count_bits()} set bits, {bitmask.n_words} words)")
    if pipe.grey is not None:
        gpath = pipe.grey.write(Path(out).with_name("SpotsGrey.npz"))
        print(f"Wrote {gpath} ({pipe.grey.n_pixels} grey pixels; must equal the set bits)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Direct entry point: ``python -m midas_nf_preprocess.process_images``."""
    parser = argparse.ArgumentParser(
        prog="midas-nf-preprocess process-images",
        description="Differentiable PyTorch port of NF_HEDM ProcessImagesCombined.",
    )
    add_arguments(parser)
    parser.add_argument(
        "--version", action="version", version=f"midas-nf-preprocess {__version__}"
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
