"""Subprocess worker for the sr-midas branch of the FF `peakfit` stage.

Runs `sr_midas.pipeline.sr_process.run_sr_process` in a throwaway process
so its CUDA context is fully released on exit, before indexing/refinement
claim the GPU. See stages/peakfit.py::_run_sr_subprocess for why this needs
to be a subprocess rather than an in-process call.

Not a public CLI — invoked internally as:
    python -m midas_pipeline._sr_worker <result_dir> [options]
Exits 0 on success, 2 if sr-midas isn't importable, 1 on any other failure
(with a traceback on stderr for the caller's log file to capture).
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", help="Directory containing the *.MIDAS.zip")
    p.add_argument("--srfac", type=int, default=8)
    p.add_argument("--sr-config", default=None,
                    help="Path to a custom sr_config.json; omit for the bundled default")
    p.add_argument("--save-sr-patches", type=int, default=0, choices=[0, 1])
    p.add_argument("--save-frame-good-coords", type=int, default=0, choices=[0, 1])
    p.add_argument("--use-gpu", type=int, default=1, choices=[0, 1])
    p.add_argument("--peak-fit-method", default=None,
                    help="Override sr_config's peak_fit_method (e.g. midas_lm, gpu_adam)")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(argv)

    try:
        from sr_midas.pipeline.sr_process import run_sr_process
    except ImportError as e:
        print(f"sr-midas is not importable: {e}", file=sys.stderr)
        return 2

    kwargs = dict(
        midasZarrDir=args.result_dir,
        srfac=args.srfac,
        saveSRpatches=args.save_sr_patches,
        saveFrameGoodCoords=args.save_frame_good_coords,
        use_gpu=args.use_gpu,
    )
    if args.sr_config is not None:
        kwargs["SRconfig_path"] = args.sr_config
    if args.peak_fit_method is not None:
        kwargs["peak_fit_method"] = args.peak_fit_method
    if args.max_frames is not None:
        kwargs["max_frames"] = args.max_frames

    run_sr_process(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
