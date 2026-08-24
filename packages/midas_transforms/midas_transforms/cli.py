"""CLI dispatch for ``midas-transforms``.

The umbrella command ``midas-transforms <stage> [args]`` plus four
sub-CLIs that mirror the C-binary argv contracts:

  ``midas-merge-peaks <zarr_path>``
  ``midas-calc-radius <zarr_path>``
  ``midas-fit-setup <zarr_path>``
  ``midas-bin-data``    (no positional args, reads from the cwd)

Each sub-CLI is a thin wrapper around the corresponding library function.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-transforms"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _common_argparser(prog: str, description: str) -> argparse.ArgumentParser:
    p = _midas_make_parser(prog=prog, description=description)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default=None)
    p.add_argument("--version", action="version", version=f"midas-transforms {__version__}")
    return p


def merge_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-merge-peaks", "Frame-by-frame mutual-nearest merge of consolidated peakfit output.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None,
                   help="Override the result folder (default: directory of zarr_path).")
    p.add_argument("--allpeaks-ps-bin", default=None,
                   help="Override the AllPeaks_PS.bin path "
                        "(default: <result-folder>/Temp/AllPeaks_PS.bin).")
    p.add_argument("--allpeaks-px-bin", default=None,
                   help="Override the AllPeaks_PX.bin path "
                        "(default: <result-folder>/Temp/AllPeaks_PX.bin). "
                        "Used only when UsePixelOverlap=1.")
    p.add_argument("--overlap-length", type=float, default=None,
                   help="Centroid distance threshold in px (default: from Zarr params, fallback 2.0).")
    p.add_argument("--use-pixel-overlap", type=int, choices=[0, 1], default=None,
                   help="Override Zarr's UsePixelOverlap flag (0=centroid, 1=pixel-overlap).")
    args = p.parse_args(argv)

    from .merge import merge_overlapping_peaks
    from .params import read_zarr_params
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    overlap = args.overlap_length if args.overlap_length is not None else zp.OverlapLength
    use_px = bool(args.use_pixel_overlap) if args.use_pixel_overlap is not None else None
    merge_overlapping_peaks(
        zarr_path=args.zarr_path,
        allpeaks_ps_bin=args.allpeaks_ps_bin,
        allpeaks_px_bin=args.allpeaks_px_bin,
        result_folder=rf,
        overlap_length=overlap,
        skip_frame=zp.SkipFrame,
        use_maxima_positions=bool(zp.UseMaximaPositions),
        use_pixel_overlap=use_px,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        device=args.device, dtype=args.dtype,
        write=True,
    )
    print(f"midas-merge-peaks {__version__}: wrote Result_*.csv and MergeMap.csv to {rf}", file=sys.stderr)
    return 0


def radius_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-calc-radius", "Per-spot ring/Bragg/grain-volume calculation.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None)
    args = p.parse_args(argv)

    from .params import read_zarr_params
    from .radius import calc_radius
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    calc_radius(
        result_folder=rf, zarr_params=zp,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-calc-radius {__version__}: wrote Radius_*.csv to {rf}", file=sys.stderr)
    return 0


def fit_setup_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-fit-setup", "Per-spot tilt+distortion+wedge correction, filtering, and paramstest.txt writer.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None)
    p.add_argument("--no-fit", action="store_true", help="Force DoFit=0 (skip the geometry refine).")
    args = p.parse_args(argv)

    from .fit_setup import fit_setup
    from .params import read_zarr_params
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    do_fit = False if args.no_fit else (zp.DoFit == 1)
    fit_setup(
        result_folder=rf, zarr_params=zp,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        do_fit=do_fit,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-fit-setup {__version__}: wrote InputAll.csv et al to {rf}", file=sys.stderr)
    return 0


def bin_data_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-bin-data", "Bin spots into Spots.bin / ExtraInfo.bin / Data.bin / nData.bin.")
    p.add_argument("--result-folder", default=".")
    args = p.parse_args(argv)

    from .bin_data import bin_data
    bin_data(
        result_folder=args.result_folder,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-bin-data {__version__}: wrote Spots.bin / ExtraInfo.bin / Data.bin / nData.bin to {args.result_folder}", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Umbrella command: ``midas-transforms <stage> [args]``."""
    parser = _midas_make_parser(
        prog="midas-transforms",
        description="Pure-Python/PyTorch FF-HEDM transforms (merge / radius / fit-setup / bin-data).",
    )
    parser.add_argument("--version", action="version", version=f"midas-transforms {__version__}")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("merge-peaks", add_help=False)
    sub.add_parser("calc-radius", add_help=False)
    sub.add_parser("fit-setup", add_help=False)
    sub.add_parser("bin-data", add_help=False)
    sub.add_parser("pipeline", add_help=False)
    sub.add_parser("detector-mask", add_help=False)
    sub.add_parser("grain-size-correct", add_help=False)
    sub.add_parser("vsample", add_help=False)

    # Parse only the first positional, dispatch the rest.
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    if argv[0] in ("--version", "-V"):
        parser.parse_args(argv)
        return 0
    # An explicit --help is a REQUEST, not a usage error: it belongs on stdout
    # with exit 0. Without this it fell through to the unknown-stage branch
    # below, which prints to stderr and returns 2 -- so `midas-transforms
    # --help > x` wrote an empty file and any script checking the exit status
    # saw a failure. The no-argv and unknown-stage cases below are genuine
    # usage errors and correctly stay on stderr with 2.
    if argv[0] in ("-h", "--help"):
        parser.print_help(sys.stdout)
        return 0
    stage, rest = argv[0], argv[1:]
    if stage == "merge-peaks":
        return merge_main(rest)
    if stage == "calc-radius":
        return radius_main(rest)
    if stage == "fit-setup":
        return fit_setup_main(rest)
    if stage == "bin-data":
        return bin_data_main(rest)
    if stage == "pipeline":
        return pipeline_main(rest)
    if stage == "detector-mask":
        return detector_mask_main(rest)
    if stage == "grain-size-correct":
        return grain_size_correct_main(rest)
    if stage == "vsample":
        return vsample_main(rest)

    parser.print_help(sys.stderr)
    return 2


def vsample_main(argv: Optional[List[str]] = None) -> int:
    """Measure Vsample from a tomogram and, on request, write it in.

    Vsample is the gauge volume that divides into every reported grain volume,
    and nothing has ever produced it -- runs either omit it (falling back to
    Hbeam * pi * Rsample^2, two search bounds) or inherit a template constant.
    A tomogram of the same specimen measures it.
    """
    import argparse

    import numpy as np

    p = argparse.ArgumentParser(
        "midas-transforms vsample",
        description="Set Vsample in an FF parameter file from a tomographic "
                    "reconstruction of the same specimen.",
        epilog="The beam HEIGHT is operator-supplied: it is a slit setting, "
               "not a property of the specimen, and Hbeam is a search bound. "
               "Everything else comes from the tomogram.",
    )
    p.add_argument("recon", help="NXtomoproc .h5 or midas-tomo .bin")
    p.add_argument("--beam-height-um", type=float, required=True)
    p.add_argument("--beam-width-um", type=float, default=None,
                   help="omit when the beam is wider than the specimen")
    p.add_argument("--pixel-size-um", type=float, required=True,
                   help="from the scan record's tomo_metastr, NOT tomocupy_args.yml")
    p.add_argument("--rot-axis", nargs=2, type=float, required=True,
                   metavar=("IX", "IY"))
    p.add_argument("--in-plane", default="xy",
                   help="volume is handedness-invariant, so the default is "
                        "safe here even though it is not for path lengths")
    p.add_argument("--det-xdim", type=int, default=None)
    p.add_argument("--shift-index", type=int, default=None)
    p.add_argument("--slice-range", nargs=2, type=int, default=None)
    p.add_argument("--params", default=None,
                   help="FF parameter file: read to report what is replaced, "
                        "and patched when --write is given")
    p.add_argument("--write", action="store_true",
                   help="patch --params in place (a backup is kept)")
    p.add_argument("--force", action="store_true",
                   help="write even an unusable value; it is then not a "
                        "measurement and the file says so")
    args = p.parse_args(argv)

    from .geometry import from_midas_tomo_bin, from_nxtomoproc
    from .geometry.tomo import otsu_threshold, threshold_sensitivity
    from .radius.vsample import vsample_from_shape, write_vsample

    reader = (from_nxtomoproc if str(args.recon).endswith((".h5", ".hdf5", ".nxs"))
              else from_midas_tomo_bin)
    common = dict(pixel_size_um=args.pixel_size_um,
                  rot_axis_ix=args.rot_axis[0], rot_axis_iy=args.rot_axis[1],
                  in_plane=args.in_plane, det_xdim=args.det_xdim,
                  shift_index=args.shift_index,
                  slice_range=tuple(args.slice_range) if args.slice_range else None)

    # The greyscale is needed before a threshold can be chosen, and the readers
    # threshold on the way in, so load the values first.
    grey = _load_recon_values(
        args.recon, shift_index=args.shift_index,
        slice_range=tuple(args.slice_range) if args.slice_range else None,
    )
    otsu = otsu_threshold(grey)
    ths = np.linspace(0.6 * otsu, 1.4 * otsu, 12)
    report = threshold_sensitivity(grey, ths,
                                   voxel_volume_um3=args.pixel_size_um ** 3)
    print(f"threshold  otsu {otsu:.6g}, swept +/-40 %: fractional spread "
          f"{report['fractional_spread']:.4f}, "
          f"stationary={report['stationary']}", file=sys.stderr)

    shape = reader(args.recon, threshold=otsu, **common)
    res = vsample_from_shape(
        shape, beam_height_um=args.beam_height_um,
        beam_width_um=args.beam_width_um, threshold_report=report,
        param_file=args.params,
    )
    print(res.summary(), file=sys.stderr)

    if not args.write:
        print("\n(nothing written; pass --write with --params to patch the "
              "parameter file)", file=sys.stderr)
        return 0 if res.usable else 1
    if args.params is None:
        print("--write needs --params", file=sys.stderr)
        return 2
    try:
        out = write_vsample(args.params, res, force=args.force)
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(f"\npatched {out}", file=sys.stderr)
    return 0


def _load_recon_values(path, *, shift_index=None, slice_range=None):
    """Raw reconstruction values, for choosing a threshold."""
    import numpy as np

    from .geometry.tomo import parse_recon_filename

    if str(path).endswith((".h5", ".hdf5", ".nxs")):
        import h5py

        with h5py.File(path, "r") as hf:
            ds = hf["entry/reconstruction/data"]
            i = 0 if shift_index is None else int(shift_index)
            s0, s1 = slice_range or (0, ds.shape[1])
            return np.asarray(ds[i, s0:s1])

    meta = parse_recon_filename(path)
    shape = (meta["n_shifts"], meta["n_slices"], meta["ydim"], meta["xdim"])
    cube = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    i = 0 if shift_index is None else int(shift_index)
    s0, s1 = slice_range or (0, shape[1])
    return np.asarray(cube[i, s0:s1])


def grain_size_correct_main(argv: Optional[List[str]] = None) -> int:
    """Rescale Grains.csv radii onto a measured illuminated volume.

    A separate command for the same reason as ``detector-mask``: it changes a
    reported number, so it is opt-in and it writes a new file rather than
    editing ``Grains.csv`` in place. The output carries ``GrainRadius_shape``
    **alongside** ``GrainRadius``, never over it.

    The shape comes either from an analytic specimen (``--cylinder`` /
    ``--box``, no tomography needed) or from a reconstruction (``--tomo``).
    Nothing here guesses a registration: the tomo route requires the pixel
    size, the rotation-axis position and the in-plane handedness, and says so.
    """
    import argparse

    import numpy as np

    p = argparse.ArgumentParser(
        "midas-transforms grain-size-correct",
        description=(
            "Replace the search-bound gauge volume with a measured "
            "illuminated volume. V_gauge = Hbeam*pi*Rsample^2 is built from "
            "two SEARCH BOUNDS, so absolute grain size currently carries a "
            "canned constant."
        ),
    )
    p.add_argument("grains_csv", help="Grains.csv from process_grains.")
    p.add_argument("--params", required=True,
                   help="The run's parameter file, for Hbeam/Rsample/Vsample.")
    p.add_argument("--beam-height-um", type=float, required=True,
                   help="Illuminated slab height. NOT Hbeam or BeamSize -- "
                        "those are search bounds. Measure it.")
    p.add_argument("--beam-width-um", type=float, default=None,
                   help="Horizontal beam extent; omit if wider than the sample.")
    p.add_argument("--beam-centre-z-um", type=float, default=0.0)
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <grains dir>/Grains_shape.csv).")

    g = p.add_argument_group("sample shape (choose exactly one)")
    g.add_argument("--cylinder", nargs=2, type=float, metavar=("DIAM_UM", "HEIGHT_UM"))
    g.add_argument("--box", nargs=3, type=float,
                   metavar=("SIZE_X_UM", "SIZE_Y_UM", "HEIGHT_UM"))
    g.add_argument("--tomo", metavar="PATH",
                   help="Reconstruction: midas-tomo .bin or NXtomoproc .h5.")

    t = p.add_argument_group("tomo registration (required with --tomo)")
    t.add_argument("--pixel-size-um", type=float)
    t.add_argument("--slice-pitch-um", type=float, default=None)
    t.add_argument("--rot-axis", nargs=2, type=float, metavar=("IX", "IY"))
    t.add_argument("--in-plane", default=None,
                   help="One of xy, yx, -xy, x-y, -x-y, -yx, y-x, -y-x. "
                        "No default: the wrong choice mirrors the sample.")
    t.add_argument("--threshold", type=float, default=None)
    t.add_argument("--shift-index", type=int, default=None)
    t.add_argument("--slice0-z-um", type=float, default=0.0)
    t.add_argument("--det-xdim", type=int, default=None,
                   help="Detector width, for the reconstruction-pad check.")
    args = p.parse_args(argv)

    chosen = [n for n, v in (("--cylinder", args.cylinder), ("--box", args.box),
                             ("--tomo", args.tomo)) if v]
    if len(chosen) != 1:
        p.error(f"give exactly one of --cylinder / --box / --tomo; got {chosen or 'none'}")

    from .geometry import SampleShape, from_midas_tomo_bin, from_nxtomoproc
    from .radius.shape_correction import GaugeVolume, correct_grain_volumes

    if args.cylinder:
        shape = SampleShape.cylinder(
            diameter_um=args.cylinder[0], height_um=args.cylinder[1],
            pixel_size_um=args.pixel_size_um or args.cylinder[0] / 100.0,
            centre_z_um=args.beam_centre_z_um,
        )
    elif args.box:
        shape = SampleShape.box(
            size_x_um=args.box[0], size_y_um=args.box[1], height_um=args.box[2],
            pixel_size_um=args.pixel_size_um or min(args.box[:2]) / 50.0,
            centre_z_um=args.beam_centre_z_um,
        )
    else:
        missing = [f for f, v in (("--pixel-size-um", args.pixel_size_um),
                                  ("--rot-axis", args.rot_axis),
                                  ("--in-plane", args.in_plane),
                                  ("--threshold", args.threshold)) if v is None]
        if missing:
            p.error(
                f"--tomo needs {', '.join(missing)}. None of these is in any "
                "reconstruction file format, and each is a silent wrong "
                "answer if guessed: a wrong pixel size gives a sharp "
                "reconstruction of the wrong-sized object, and the wrong "
                "handedness mirrors the sample."
            )
        reader = from_nxtomoproc if str(args.tomo).endswith((".h5", ".hdf5", ".nxs")) \
            else from_midas_tomo_bin
        shape = reader(
            args.tomo, pixel_size_um=args.pixel_size_um,
            slice_pitch_um=args.slice_pitch_um,
            rot_axis_ix=args.rot_axis[0], rot_axis_iy=args.rot_axis[1],
            in_plane=args.in_plane, threshold=args.threshold,
            shift_index=args.shift_index, slice0_z_um=args.slice0_z_um,
            det_xdim=args.det_xdim,
        )

    v_illum = shape.illuminated_volume_um3(
        beam_height_um=args.beam_height_um, beam_width_um=args.beam_width_um,
        beam_centre_z_um=args.beam_centre_z_um,
    )
    gauge = GaugeVolume.from_param_file(args.params)

    rows = Path(args.grains_csv).read_text().splitlines()
    hdr_i = next((i for i, r in enumerate(rows) if "GrainRadius" in r), None)
    if hdr_i is None:
        p.error(f"{args.grains_csv} has no GrainRadius column in any header line")
    header = rows[hdr_i].lstrip("%").split()
    col = header.index("GrainRadius")
    body = [r for r in rows[hdr_i + 1:] if r.strip()]
    radii = np.array([float(r.split()[col]) for r in body])

    volumes = (4.0 / 3.0) * math.pi * radii ** 3
    _, r_new, rep = correct_grain_volumes(
        volumes, gauge=gauge, illuminated_volume_um3=v_illum,
    )

    out = Path(args.out) if args.out else Path(args.grains_csv).with_name("Grains_shape.csv")
    with out.open("w") as f:
        f.write("% " + " ".join(header) + " GrainRadius_shape\n")
        for line, rv in zip(body, r_new):
            f.write(f"{line} {rv:.6f}\n")

    print(rep.summary(), file=sys.stderr)
    print(f"\n{len(body)} grains: median GrainRadius "
          f"{np.median(radii):.3f} -> {np.median(r_new):.3f} um",
          file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


def detector_mask_main(argv: Optional[List[str]] = None) -> int:
    """Build BigDetectorMask.bin from a zarr's detector mask.

    Deliberately a separate command rather than a pipeline stage. Enabling the
    mask changes ``Completeness`` -- the number that gates which grains exist --
    so it is opt-in, and the ``BigDetSize`` this prints has to be pasted into
    the parameter file by hand. Silently switching it on for every run that
    happens to carry a MaskFile would move every historical grain count with no
    record of why.
    """
    import argparse

    p = argparse.ArgumentParser(
        "midas-transforms detector-mask",
        description=(
            "Push a detector mask forward into the ideal-lab active-area "
            "bitset the C indexer and refiner read as BigDetectorMask.bin."
        ),
    )
    p.add_argument("zarr_path", help="MIDAS Zarr archive (.zip) carrying exchange/mask.")
    p.add_argument("--out", default=None,
                   help="Output path (default: <zarr dir>/BigDetectorMask.bin).")
    p.add_argument("--off-detector", choices=("drop", "keep"), default="drop",
                   help="drop (default): a reflection predicted off the panel "
                        "also leaves the completeness ratio. keep: only "
                        "explicitly masked cells are removed, isolating the "
                        "mask's effect from the detector-extent effect.")
    p.add_argument("--dilate", type=int, default=1,
                   help="Grow the masked set by N cells (default 1). "
                        "Conservative: errs toward excluding a spot.")
    p.add_argument("--big-det-size", type=int, default=None,
                   help="Override the auto-sized grid (must be even).")
    args = p.parse_args(argv)

    from .geometry.detector_mask import (
        build_active_area_bitset_from_zarr, write_big_detector_mask,
    )

    bits, size, stats = build_active_area_bitset_from_zarr(
        args.zarr_path,
        off_detector=args.off_detector,
        dilate_masked=args.dilate,
        big_det_size=args.big_det_size,
    )
    out = Path(args.out) if args.out else Path(args.zarr_path).parent / "BigDetectorMask.bin"
    write_big_detector_mask(out, bits)

    active_pct = 100.0 * stats["n_cells_keep"] / max(stats["n_cells"], 1)
    print(f"wrote {out}", file=sys.stderr)
    print(f"  bad pixels          : {stats['n_bad_pixels']} of "
          f"{stats['n_detector_pixels']}", file=sys.stderr)
    print(f"  masked cells        : {stats['n_cells_masked']} "
          f"(-> {stats['n_cells_masked_after_dilation']} after dilation)",
          file=sys.stderr)
    print(f"  active cells        : {stats['n_cells_keep']} of "
          f"{stats['n_cells']} ({active_pct:.2f}%)", file=sys.stderr)
    print(f"  widest pixel span   : {stats['max_pixel_cell_span'] + 1} cells",
          file=sys.stderr)
    print(f"  off-detector policy : {stats['off_detector']} "
          f"(keep would give {stats['n_cells_keep_if_keep']} active cells, "
          f"drop gives {stats['n_cells_keep_if_drop']})", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"Add this to the parameter file to switch the mask ON:",
          file=sys.stderr)
    print(f"    BigDetSize {size}", file=sys.stderr)
    print("Leave it out (or 0) and nothing changes.", file=sys.stderr)
    return 0


def pipeline_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-transforms pipeline", "Run all four stages on-device with no disk round-trips between them.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--out-dir", default=None, help="Output directory (default: dir of zarr_path).")
    p.add_argument("--allpeaks-ps-bin", default=None)
    args = p.parse_args(argv)

    from .pipeline import Pipeline
    pipe = Pipeline.from_zarr(
        args.zarr_path,
        allpeaks_ps_bin=args.allpeaks_ps_bin,
        device=args.device, dtype=args.dtype,
    )
    pipe.run()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.zarr_path).parent
    pipe.dump(out_dir)
    print(f"midas-transforms pipeline {__version__}: wrote 9 files to {out_dir}", file=sys.stderr)
    return 0
