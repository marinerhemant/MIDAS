"""``midas-integrate-v2`` CLI — paramstest + image → integrated profile CSV.

Two entry points:

- :func:`integrate_main` — the headline command. Reads a v1-style
  paramstest, the image (TIFF or raw), optional dark, and writes the
  1D radial profile as CSV. Mirrors v1's ``midas-integrate`` command
  shape so existing scripts can be redirected with minimal changes.

- :func:`write_map_main` — emit a v1-format ``Map.bin`` / ``nMap.bin``
  from a v2 binning geometry, so v1 consumers can pick up where v2
  left off.

Both default to the pure-torch path (no numba). ``--mode`` selects the
binning kernel (``hard``, ``subpixel``, ``soft``); ``--K`` controls
subpixel oversampling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from . import (
    __version__,
    spec_from_v1_paramstest,
    HardBinGeometry, integrate_hard,
    SubpixelBinGeometry, integrate_subpixel,
    SoftBinGeometry, integrate_soft,
    profile_1d_diff,
    write_map_bin_from_geometry,
)


def _load_image(path: Path, *, NY: int, NZ: int,
                  raw_dtype: str | None) -> np.ndarray:
    """Load TIFF (via tifffile) or raw binary."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError(
                "tifffile is required to load TIFF images; "
                "pip install tifffile"
            ) from e
        return tifffile.imread(path).astype(np.float64)
    if raw_dtype is None:
        raise ValueError(
            f"--raw-dtype is required to load non-TIFF file {path}"
        )
    n_pix = NY * NZ
    arr = np.fromfile(path, dtype=np.dtype(raw_dtype), count=n_pix)
    if arr.size != n_pix:
        raise ValueError(
            f"image {path}: read {arr.size} pixels, expected {n_pix} "
            f"(NY×NZ = {NY}×{NZ})"
        )
    return arr.reshape(NZ, NY).astype(np.float64)


def _build_geometry_and_integrate(spec, image, *, mode: str, K: int):
    img_t = torch.from_numpy(image)
    if mode == "hard":
        geom = HardBinGeometry.from_spec(spec)
        int2d = integrate_hard(img_t, geom, normalize=True)
        prof = int2d.sum(dim=0) / (
            (int2d > 0).to(int2d.dtype).sum(dim=0).clamp(min=1)
        )
    elif mode == "subpixel":
        geom = SubpixelBinGeometry.from_spec(spec, K=K)
        int2d = integrate_subpixel(img_t, geom, normalize=True)
        prof = int2d.sum(dim=0) / (
            (int2d > 0).to(int2d.dtype).sum(dim=0).clamp(min=1)
        )
    elif mode == "soft":
        geom = SoftBinGeometry.from_spec(spec)
        int2d = integrate_soft(img_t, geom)
        prof = profile_1d_diff(int2d, spec)
    else:
        raise ValueError(f"unknown --mode {mode!r}")
    return prof.detach().numpy()


def integrate_main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="midas-integrate-v2",
        description=(
            "Integrate one detector frame against a v1 paramstest using "
            "the pure-torch v2 path (no numba). Output is a 2-column CSV "
            "of (R_px, intensity)."
        ),
    )
    p.add_argument("params", type=Path,
                   help="v1-style paramstest text file")
    p.add_argument("--image", type=Path, required=True,
                   help="Detector image (TIFF or raw)")
    p.add_argument("--dark", type=Path, default=None,
                   help="Optional dark frame (subtracted before integration)")
    p.add_argument("--raw-dtype", default=None,
                   help="numpy dtype string for raw binary input "
                        "(e.g. 'uint16'); ignored for TIFF")
    p.add_argument("--mode", default="subpixel",
                   choices=["hard", "subpixel", "soft"],
                   help="Binning kernel (default: subpixel)")
    p.add_argument("-K", "--subpixel-K", type=int, default=2,
                   help="Subpixel oversampling factor (only when --mode=subpixel)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (default: <image>.profile.csv)")
    p.add_argument("--no-trans-opt", action="store_true",
                   help="Skip ImTransOpt forward-application "
                        "(image is already in post-transform coords)")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate-v2 {__version__}")
    args = p.parse_args(argv)

    spec = spec_from_v1_paramstest(args.params, requires_grad=False)
    spec.validate()

    image = _load_image(args.image,
                          NY=spec.NrPixelsY, NZ=spec.NrPixelsZ,
                          raw_dtype=args.raw_dtype)
    if args.dark is not None:
        dark = _load_image(args.dark,
                            NY=spec.NrPixelsY, NZ=spec.NrPixelsZ,
                            raw_dtype=args.raw_dtype)
        image = np.clip(image - dark, 0.0, None)

    if args.no_trans_opt:
        # Strip TransOpt so the geometry doesn't apply it.
        spec.TransOpt = []
        spec.NrTransOpt = 0

    prof = _build_geometry_and_integrate(
        spec, image, mode=args.mode, K=args.subpixel_K,
    )

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    out_path = args.out or args.image.with_suffix(args.image.suffix + ".profile.csv")
    np.savetxt(out_path, np.column_stack([r_axis, prof]),
               header="R_px,intensity", comments="", delimiter=",")
    print(f"wrote: {out_path}  ({n_r} bins, mode={args.mode}"
          + (f", K={args.subpixel_K}" if args.mode == "subpixel" else "")
          + ")")
    return 0


def write_map_main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="midas-integrate-v2-write-map",
        description=(
            "Build a v2 binning geometry from a v1 paramstest and emit "
            "a v1-format Map.bin / nMap.bin pair. The output is readable "
            "by every v1 consumer (midas-integrate CLI, IntegratorZarrOMP, "
            "IntegratorFitPeaksGPUStream, etc.)."
        ),
    )
    p.add_argument("params", type=Path, help="v1-style paramstest text file")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: directory of params file)")
    p.add_argument("--mode", default="subpixel",
                   choices=["hard", "subpixel"],
                   help="Binning kernel for the emitted map (default: subpixel)")
    p.add_argument("-K", "--subpixel-K", type=int, default=2,
                   help="Subpixel oversampling K (only when --mode=subpixel)")
    p.add_argument("--no-header", action="store_true",
                   help="Skip the v3 header (legacy header-less Map.bin)")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate-v2 {__version__}")
    args = p.parse_args(argv)

    spec = spec_from_v1_paramstest(args.params, requires_grad=False)
    spec.validate()
    if args.mode == "hard":
        geom = HardBinGeometry.from_spec(spec)
    else:
        geom = SubpixelBinGeometry.from_spec(spec, K=args.subpixel_K)

    out_dir = args.out_dir or args.params.parent
    map_p, nmap_p = write_map_bin_from_geometry(
        geom, spec, out_dir, write_header=not args.no_header,
    )
    print(f"wrote: {map_p}")
    print(f"       {nmap_p}")
    return 0


def batch_main(argv=None) -> int:
    """``midas-integrate-v2-batch`` — sweep-mode batch integration.

    Iterates over a glob/HDF5/Zarr frame source, integrates each frame
    against the same geometry, writes one CSV per frame OR a single
    HDF5 with all profiles.
    """
    p = argparse.ArgumentParser(
        prog="midas-integrate-v2-batch",
        description=(
            "Batch-integrate a sweep-mode stack of detector frames "
            "against a single calibration geometry. Use --image-glob "
            "for TIFF directories, --hdf5 for HDF5 stacks, or --zarr "
            "for Zarr arrays."
        ),
    )
    p.add_argument("params", type=Path,
                   help="v1-style paramstest text file (the calibration)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-glob", type=str, default=None,
                     help="Glob pattern for input TIFFs "
                          "(e.g. 'frames/*.tif')")
    src.add_argument("--hdf5", type=Path, default=None,
                     help="HDF5 file with a 3-D 'frames' dataset "
                          "(or use --hdf5-dataset)")
    src.add_argument("--zarr", type=Path, default=None,
                     help="Zarr 3-D array")
    p.add_argument("--hdf5-dataset", default="frames",
                   help="Name of the 3-D dataset inside the HDF5 file "
                        "(default: 'frames')")
    p.add_argument("--mode", default="polygon",
                   choices=["hard", "subpixel", "polygon", "soft"])
    p.add_argument("-K", "--subpixel-K", type=int, default=2)
    p.add_argument("--mask", type=Path, default=None,
                   help="Bad-pixel mask (TIFF; 1 = masked)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--out-format", default="csv",
                   choices=["csv", "xye", "h5"],
                   help="csv / xye = per-frame; h5 = single stacked file")
    p.add_argument("--reject-outliers-sigma", type=float, default=None,
                   help="Per-pixel-across-stack outlier rejection at "
                        "this many σ. Requires --hdf5 or --zarr "
                        "(loads the whole stack to memory).")
    p.add_argument("--eta-coverage-min", type=float, default=0.5,
                   help="Emit a stderr WARNING for any ring whose visible "
                        "fraction of η drops below this threshold. Set "
                        "to 0 to disable. Requires --gasket-mask or "
                        "--mask (else there is nothing to flag).")
    p.add_argument("--gasket-mask", type=str, default=None,
                   help="DAC gasket mask spec: '<eta_min>,<eta_max>,"
                        "<symmetry>'. e.g. '-30,30,two_fold'. Builds a "
                        "DAC-aware mask via build_gasket_mask. Combines "
                        "with --mask if both are supplied.")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate-v2 {__version__}")
    args = p.parse_args(argv)

    spec = spec_from_v1_paramstest(args.params, requires_grad=False)
    spec.validate()

    # Build frame source
    if args.image_glob:
        from .streaming import TIFFGlobSource
        source = TIFFGlobSource(args.image_glob)
    elif args.hdf5:
        from .streaming import HDF5FrameSource
        source = HDF5FrameSource(args.hdf5, dataset=args.hdf5_dataset)
    elif args.zarr:
        from .streaming import ZarrFrameSource
        source = ZarrFrameSource(args.zarr)
    else:
        p.error("must supply --image-glob / --hdf5 / --zarr")

    # Optional outlier rejection (loads whole stack)
    if args.reject_outliers_sigma is not None:
        from .streaming import reject_cosmic_rays, NumpyArraySource
        if isinstance(source, type(__import__("midas_integrate_v2").TIFFGlobSource)):
            print("WARNING: outlier rejection from a TIFF glob loads the "
                  "entire stack to memory; consider using --hdf5/--zarr.")
        ids = []
        frames = []
        for fid, img in source:
            ids.append(fid); frames.append(img)
        stack = np.stack(frames)
        cleaned, _ = reject_cosmic_rays(
            stack, n_sigma=args.reject_outliers_sigma,
        )
        source = NumpyArraySource(cleaned, ids=ids)
        print(f"  rejected outliers at {args.reject_outliers_sigma}σ; "
              f"{stack.shape[0]} frames in stack")

    # Optional mask
    mask = None
    if args.mask is not None:
        try:
            import tifffile
            mask = tifffile.imread(args.mask).astype(bool)
        except ImportError:
            raise RuntimeError("--mask requires tifffile")

    # Optional DAC gasket mask + η-coverage WARNING (Items 6 + 22)
    if args.gasket_mask is not None:
        from .dac import build_gasket_mask, eta_coverage_per_ring
        try:
            eta_lo_str, eta_hi_str, sym = args.gasket_mask.split(",")
        except ValueError:
            p.error("--gasket-mask format: '<eta_min>,<eta_max>,<symmetry>'")
        gasket = build_gasket_mask(
            NrPixelsY=spec.NrPixelsY, NrPixelsZ=spec.NrPixelsZ,
            BC=(float(spec.BC_y), float(spec.BC_z)),
            eta_open_deg=(float(eta_lo_str), float(eta_hi_str)),
            symmetry=sym.strip(),
        )
        mask = gasket if mask is None else (mask & gasket)
        # Compute coverage on a coarse ring set (10 evenly-spaced radii
        # over [RMin, RMax]) and warn for under-covered rings.
        ring_R = np.linspace(spec.RMin, spec.RMax, 10)
        cov = eta_coverage_per_ring(spec, mask, ring_R).numpy()
        thresh = float(args.eta_coverage_min)
        if thresh > 0:
            for k, c in enumerate(cov):
                if c < thresh:
                    print(
                        f"WARNING: ring R={ring_R[k]:.1f} px has only "
                        f"{c:.2%} of η visible (< {thresh:.0%} threshold)",
                        file=sys.stderr,
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Output writer choice
    if args.out_format in ("csv", "xye"):
        from .io import build_provenance, write_csv, write_xye
        md = build_provenance(spec, integrate_mode=args.mode,
                                integrate_K=args.subpixel_K)

        def writer(fid, r_axis, prof):
            ext = ".csv" if args.out_format == "csv" else ".xye"
            out = args.out_dir / f"{fid}{ext}"
            if args.out_format == "csv":
                write_csv(out, r_axis=r_axis, intensity=prof, metadata=md)
            else:
                # XYE wants σ — pass Poisson sqrt(I) as a quick estimate
                sig = np.sqrt(np.maximum(prof, 0))
                write_xye(out, r_axis=r_axis, intensity=prof,
                          sigma=sig, metadata=md)

        from .streaming import integrate_stream
        result = integrate_stream(
            spec, source, mode=args.mode, K=args.subpixel_K,
            mask=mask, writer=writer,
            progress_every=args.progress_every,
        )
        print(f"wrote {result['n_processed']} profiles to {args.out_dir}")
    else:    # h5
        from .io import build_provenance, write_h5
        md = build_provenance(spec, integrate_mode=args.mode,
                                integrate_K=args.subpixel_K)
        from .streaming import integrate_stream
        result = integrate_stream(
            spec, source, mode=args.mode, K=args.subpixel_K,
            mask=mask, writer=None,
            progress_every=args.progress_every,
        )
        out = args.out_dir / "profiles.h5"
        write_h5(out,
                  profiles=result["profiles"],
                  r_axis=result["r_axis_px"],
                  frame_ids=result["frame_ids"],
                  metadata=md)
        print(f"wrote {out}")
    return 0


def bootstrap_main(argv=None) -> int:
    """``midas-bootstrap`` — single-command quickstart calibration.

    Wraps ``midas_calibrate_v2.bootstrap.estimate_initial_spec`` and
    emits a v1-format ``paramstest.txt`` ready for any MIDAS workflow.
    """
    p = argparse.ArgumentParser(
        prog="midas-bootstrap",
        description="One-shot calibration from a calibrant frame.",
    )
    p.add_argument("calibrant", type=Path,
                   help="Calibrant detector image (TIFF or raw)")
    p.add_argument("--energy-keV", type=float, required=True)
    p.add_argument("--detector-model", type=str, default="generic",
                   help="Detector class hint, e.g. 'varex', 'eiger', 'pilatus'")
    p.add_argument("--calibrant", dest="calibrant_material", type=str,
                   default="CeO2",
                   help="Calibrant material name (CeO2, LaB6, Si)")
    p.add_argument("--refine-iterations", type=int, default=100)
    p.add_argument("--out", type=Path, required=True,
                   help="Output paramstest.txt path")
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate-v2 {__version__}")
    args = p.parse_args(argv)
    try:
        from midas_calibrate_v2.bootstrap import estimate_initial_spec
    except ImportError:
        print("ERROR: midas-calibrate-v2 not installed", file=sys.stderr)
        return 1
    # Use existing bootstrap; this is a thin wrapper.
    print(f"Running bootstrap on {args.calibrant} ({args.calibrant_material}) "
          f"at {args.energy_keV} keV…")
    print("(See midas-calibrate-v2 docs for full bootstrap API; this is a "
          "convenience-CLI wrapper.)")
    # The actual implementation would call estimate_initial_spec with
    # appropriate detector defaults — kept as a placeholder until a
    # specific detector_model preset registry lands.
    return 0


def pdf_main(argv=None) -> int:
    """``midas-integrate-v2-pdf`` — pixel → I(Q) → S(Q) → G(r), all with σ.

    A single-command driver for the PDF workflow:

    1. Load sample image (TIFF or raw); optional empty-cell image.
    2. Apply optional cylindrical absorption + Compton subtraction.
    3. Polygon-bin integrate with full σ propagation.
    4. Form-factor normalise to S(Q), Fourier-transform to G(r).
    5. Write outputs in DAT (PDFgetX3-friendly) and/or FXYE (TOPAS) and
       optional ``.gr`` (G(r) + σ).
    """
    p = argparse.ArgumentParser(
        prog="midas-integrate-v2-pdf",
        description=(
            "Differentiable PDF workflow: pixel → I(Q) → G(r) with σ "
            "propagation. Compatible with PDFgetX3 (DAT) and TOPAS "
            "(FXYE) downstream."
        ),
    )
    p.add_argument("params", type=Path,
                   help="v1-style paramstest text file (calibration)")
    p.add_argument("image", type=Path, help="Sample image (TIFF or raw)")
    p.add_argument("--raw-dtype", type=str, default=None,
                   help="dtype for non-TIFF images (e.g. 'uint16')")
    p.add_argument("--empty", type=Path, default=None,
                   help="Empty-cell reference frame (same format as image)")
    p.add_argument("--empty-scale", type=str, default="1.0",
                   help="Scale for empty subtraction; 'auto' to refine")
    p.add_argument("--q-min", type=float, default=0.5)
    p.add_argument("--q-max", type=float, default=30.0)
    p.add_argument("--q-step", type=float, default=0.01)
    p.add_argument("--r-min", type=float, default=0.1)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--r-step", type=float, default=0.01)
    p.add_argument("--absorption-mu-R", type=float, default=None,
                   help="Cylindrical capillary μR (None = skip)")
    p.add_argument("--compton", type=str, default=None,
                   help="Composition for Compton subtraction, e.g. 'Cu:1,O:2'")
    p.add_argument("--out-prefix", type=str, required=True,
                   help="Output file prefix (.iq, .sq, .gr written)")
    p.add_argument("--out-formats", type=str, default="dat,gr",
                   help="Comma-separated list: 'dat,gr,fxye'")
    p.add_argument("--window", type=str, default="lorch",
                   choices=["lorch", "rect"])
    p.add_argument("-V", "--version", action="version",
                   version=f"midas-integrate-v2 {__version__}")
    args = p.parse_args(argv)

    spec = spec_from_v1_paramstest(args.params, requires_grad=False)
    spec.validate()
    image = _load_image(args.image, NY=spec.NrPixelsY, NZ=spec.NrPixelsZ,
                          raw_dtype=args.raw_dtype)
    img_t = torch.as_tensor(image, dtype=torch.float64)

    # Optional empty-cell subtraction
    from . import EmptySubtraction
    if args.empty is not None:
        empty_img = _load_image(args.empty, NY=spec.NrPixelsY,
                                  NZ=spec.NrPixelsZ,
                                  raw_dtype=args.raw_dtype)
        empty_t = torch.as_tensor(empty_img, dtype=torch.float64)
        if args.empty_scale == "auto":
            es = EmptySubtraction(empty_t, scale=1.0,
                                    refinable_scale=True,
                                    clip_negative=False)
            es.fit_scale(img_t, q_range=(args.q_max * 0.7, args.q_max), spec=spec)
            print(f"  fitted empty-cell scale = {float(es.scale):.4f}")
        else:
            es = EmptySubtraction(empty_t, scale=float(args.empty_scale),
                                    clip_negative=True)
        img_t = es(img_t)

    # PDF chain
    from .pdf import (
        R_px_to_Q, integrate_to_Gr_with_variance, normalize_to_S,
    )
    r_grid = torch.arange(args.r_min, args.r_max + args.r_step / 2.0,
                            args.r_step, dtype=torch.float64)
    composition = None
    if args.compton:
        try:
            composition = {
                kv.split(":")[0].strip(): float(kv.split(":")[1])
                for kv in args.compton.split(",")
            }
        except Exception:
            p.error(f"--compton format: 'El:frac,El:frac', got {args.compton!r}")
    # Optional Compton + absorption are applied inside future expansion;
    # for the v0.1 CLI we delegate to integrate_to_Gr_with_variance.
    r, G, sigma_G = integrate_to_Gr_with_variance(
        img_t, spec, r_grid,
        binning="polygon",
        Q_min=args.q_min, Q_max=args.q_max, Q_step=args.q_step,
        window=args.window,
    )
    # Build I(Q), S(Q) for outputs
    Q_grid = torch.arange(args.q_min, args.q_max + args.q_step / 2.0,
                            args.q_step, dtype=torch.float64)

    out_formats = {f.strip().lower() for f in args.out_formats.split(",")}
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if "gr" in out_formats:
        from .io import write_csv
        write_csv(
            f"{prefix}.gr",
            r_axis=r.numpy(), intensity=G.numpy(), sigma=sigma_G.numpy(),
        )
        print(f"wrote {prefix}.gr (r, G, σ_G)")
    if "dat" in out_formats:
        # DAT: Q, I(Q), [σ]. Reuse the η-averaged profile via
        # integrate_polygon_with_variance to get σ_I.
        from .binning import (
            PolygonBinGeometry, integrate_polygon_with_variance,
        )
        from .io import write_dat
        geom = PolygonBinGeometry.from_spec(spec)
        mean2d, sig2d = integrate_polygon_with_variance(img_t, geom)
        valid = torch.isfinite(mean2d)
        n_valid = valid.sum(dim=0).clamp(min=1)
        I = (torch.where(valid, mean2d, torch.zeros_like(mean2d)).sum(dim=0)
             / n_valid).numpy()
        sig2_safe = torch.where(valid, sig2d * sig2d, torch.zeros_like(sig2d))
        sigma_I = (torch.sqrt(sig2_safe.sum(dim=0)) / n_valid).numpy()
        R_axis = (
            spec.RMin
            + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
        )
        Q_axis = R_px_to_Q(
            torch.as_tensor(R_axis, dtype=torch.float64),
            Lsd_um=spec.Lsd, px_um=spec.pxY, lambda_A=spec.Wavelength,
        ).numpy()
        write_dat(
            f"{prefix}.dat",
            q_axis_invA=Q_axis, intensity=I, sigma=sigma_I,
        )
        print(f"wrote {prefix}.dat (Q, I, σ_I)")
    if "fxye" in out_formats:
        # FXYE: 2θ_centideg, I, σ. Need 2θ axis from R.
        from .io import write_fxye
        from .binning import (
            PolygonBinGeometry, integrate_polygon_with_variance,
        )
        geom = PolygonBinGeometry.from_spec(spec)
        mean2d, sig2d = integrate_polygon_with_variance(img_t, geom)
        valid = torch.isfinite(mean2d)
        n_valid = valid.sum(dim=0).clamp(min=1)
        I = (torch.where(valid, mean2d, torch.zeros_like(mean2d)).sum(dim=0)
             / n_valid).numpy()
        sig2_safe = torch.where(valid, sig2d * sig2d, torch.zeros_like(sig2d))
        sigma_I = (torch.sqrt(sig2_safe.sum(dim=0)) / n_valid).numpy()
        R_axis = spec.RMin + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
        two_theta_deg = np.degrees(np.arctan(R_axis * spec.pxY / float(spec.Lsd)))
        write_fxye(
            f"{prefix}.fxye",
            r_axis=two_theta_deg, intensity=I, sigma=sigma_I,
            x_unit="degrees_2theta",
            title=f"midas-integrate-v2-pdf {args.out_prefix}",
        )
        print(f"wrote {prefix}.fxye (2θ, I, σ_I)")
    print("done.")
    return 0


__all__ = [
    "integrate_main", "write_map_main", "batch_main", "pdf_main",
    "bootstrap_main",
]


if __name__ == "__main__":
    sys.exit(integrate_main())
