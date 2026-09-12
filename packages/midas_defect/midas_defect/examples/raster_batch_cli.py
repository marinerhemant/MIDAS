r"""Command-line driver for midas_defect.raster.reduce_raster_block.

Scheduler-agnostic on purpose: this is a plain script with --block-nr/--n-blocks, not a
.sbatch template. Nothing in this codebase commits to one scheduler's array-job syntax, so
wire it into whichever one you have -- one line, for SLURM:

    sbatch --array=0-19 --wrap="python raster_batch_cli.py --config my_raster.py \
        --n-blocks 20 --block-nr $SLURM_ARRAY_TASK_ID --out-dir results/"

or a PBS job array ($PBS_ARRAYID), or a plain shell loop with no scheduler at all:

    for i in $(seq 0 19); do
        python raster_batch_cli.py --config my_raster.py --n-blocks 20 --block-nr $i \
            --out-dir results/ &
    done; wait

--config is a plain Python file (imported, not parsed) because the one thing this module
cannot be generic about is how YOUR beamline stores frames. It must define:

    from midas_defect.geometry import Geometry
    GEOM = Geometry(...)                       # one geometry for the WHOLE raster
    MATERIAL = dict(a=..., c=..., space_group_number=..., sigma_rtn=(..., ..., ...))
    POINT_INDICES = list(range(621))           # every point in the raster, in your own order
    def loader(p):                             # returns point p's raw
        ...                                    # (n_frames, n_rows, n_cols) frame stack,
        ...                                    # ALREADY dead/shutter-ramp frames dropped

``GEOM`` is shared across every point in ``POINT_INDICES`` -- in particular ONE
``omega_first_deg``. If a raw position has dead/shutter-ramp frames at the ends (a real 2604
raster does), drop them with ``midas_defect.ingest.live_frames`` inside ``loader`` before
returning the stack, same as ``05_raster_lattice_single_point.ipynb``'s CONFIG cell shows -- but
check that every position you batch drops the SAME number of leading frames first (verified true
for five real 2604 positions, 2026-09-12; not guaranteed for yours). If it varies by position,
``loader`` needs to pad/trim to one fixed frame count itself, or you need more than one ``GEOM``
and more than one call to :func:`reduce_raster_block`.

See midas_defect/notebooks/06_raster_lattice_batch.ipynb for a runnable example (including a
synthetic loader that needs no real data), and reduce_one_position's docstring for what every
MATERIAL key means and why none of them has a default.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from midas_defect.raster import reduce_raster_block


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("raster_batch_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load a Python module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("GEOM", "MATERIAL", "POINT_INDICES", "loader"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} must define {name!r}")
    return module


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a Python file defining "
                  "GEOM, MATERIAL, POINT_INDICES and loader(p) -- see this file's docstring")
    p.add_argument("--out-dir", required=True, help="one position_{p}.json is written per point")
    p.add_argument("--block-nr", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=1)
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    done = reduce_raster_block(cfg.loader, cfg.GEOM, cfg.POINT_INDICES, args.out_dir,
                               block_nr=args.block_nr, n_blocks=args.n_blocks, **cfg.MATERIAL)
    print(f"block {args.block_nr}/{args.n_blocks}: wrote {len(done)} point(s) to {args.out_dir}: "
         f"{done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
