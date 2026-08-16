"""``midas-parsl-configs`` console script.

Subcommands:
  list        — show every visible config (builtin + user) and where it lives
  generate    — convert a SLURM/PBS submit script into a Parsl config module
  show        — print the resolved Config repr for a given name
  path        — print the user config dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .generate import (
    build_config_module,
    parse_submit_script,
    write_user_config,
)
from .registry import (
    AVAILABLE_BUILTIN,
    available_configs,
    load_config,
    user_configs_dir,
)

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-parsl-configs"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _build_parser() -> argparse.ArgumentParser:
    p = _midas_make_parser(
        prog="midas-parsl-configs",
        description="Bundled + user-extensible Parsl configs for MIDAS pipelines.",
    )
    p.add_argument("--version", action="version",
                   version=f"midas-parsl-configs {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show every visible config")
    p_list.set_defaults(func=_cmd_list)

    p_gen = sub.add_parser(
        "generate",
        help="convert a SLURM/PBS submit script into a parsl config module",
    )
    p_gen.add_argument("script", help="path to the submit script")
    p_gen.add_argument("--name", required=True,
                       help="short name (becomes <Name>Config.py)")
    p_gen.add_argument("--out-dir", default=None,
                       help="override target dir (default: ~/.midas/parsl_configs)")
    p_gen.add_argument("--print", dest="print_only", action="store_true",
                       help="print module to stdout instead of writing it")
    p_gen.set_defaults(func=_cmd_generate)

    p_show = sub.add_parser("show", help="print the resolved Config for a name")
    p_show.add_argument("name")
    p_show.add_argument("--n-nodes", type=int, default=1)
    p_show.add_argument("--n-cpus", type=int, default=8)
    p_show.set_defaults(func=_cmd_show)

    p_path = sub.add_parser("path", help="print the user config dir")
    p_path.set_defaults(func=_cmd_path)

    return p


def _cmd_list(args: argparse.Namespace) -> int:
    cfgs = available_configs()
    if not cfgs:
        print("(no configs visible — try `midas-parsl-configs generate ...`)")
        return 0
    width = max(len(n) for n in cfgs)
    for name in sorted(cfgs):
        print(f"  {name:<{width}}  {cfgs[name]}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    info = parse_submit_script(args.script)
    src = build_config_module(name=args.name, info=info)
    if args.print_only:
        print(src)
        return 0
    target_dir: Optional[Path] = (
        Path(args.out_dir).expanduser() if args.out_dir else None
    )
    out = write_user_config(args.name, src, target_dir=target_dir)
    print(f"wrote {out}")
    print(f"detected scheduler: {info.scheduler}")
    print(f"  partition/queue : {info.partition or info.queue or '(none)'}")
    print(f"  walltime        : {info.walltime or '(default)'}")
    print(f"  nodes_per_block : {info.nodes_per_block}")
    print(f"  cores_per_worker: {info.cores_per_worker}")
    if info.n_gpus_per_node:
        print(f"  gpus per node   : {info.n_gpus_per_node}")
    print(f"\nload it with:  midas_parsl_configs.load_config({args.name!r})")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.name, n_nodes=args.n_nodes, n_cpus=args.n_cpus)
    except Exception as e:
        print(f"failed to load {args.name!r}: {e}", file=sys.stderr)
        return 1
    print(repr(cfg))
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    print(user_configs_dir())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
