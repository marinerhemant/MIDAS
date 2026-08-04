"""Pipeline stages — thin Python wrappers around existing package APIs.

Each stage is a one-shot function: it parses the parameter file (or
takes the already-parsed dict), calls into the relevant package's
public Python API, and writes its outputs to ``params['resultFolder']``.

No subprocess shells, no C binaries — every step is pure Python.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .params import collect_multiline, parse_parameters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers shared by stages
# ---------------------------------------------------------------------------

def _resolve_lsd(p: Dict[str, Any], param_file: str | Path) -> float:
    """Read the LAST ``Lsd`` value from the param file.

    NF param files typically have one ``Lsd`` line per detector
    distance; the C parser ``cfg->Lsd = ...`` overwrites on each call,
    so the LAST one wins. We mirror that.
    """
    lsds = collect_multiline(param_file, "Lsd")
    if not lsds:
        raise ValueError(f"No Lsd line found in {param_file}")
    return float(lsds[-1].split()[0])


def _resolve_max_ring_rad(p: Dict[str, Any]) -> float:
    """Pick ``RhoD`` (preferred) or ``MaxRingRad`` (alias)."""
    if "RhoD" in p:
        return float(p["RhoD"])
    if "MaxRingRad" in p:
        return float(p["MaxRingRad"])
    raise ValueError("Param file missing RhoD/MaxRingRad")


# ---------------------------------------------------------------------------
#  Stage 0: denoise (opt-in, requires MIDAS-NF-preProc)
# ---------------------------------------------------------------------------

def run_denoise(p: Dict[str, Any], param_file: str | Path) -> None:
    """Optional step 0: denoise raw TIFFs and re-point ``DataDirectory``."""
    if int(p.get("Denoise", 0)) != 1:
        logger.info("Denoise disabled (Denoise=0); skipping.")
        return

    method = str(p.get("DenoiseMethod", "nlm")).lower()
    if method not in ("nlm", "n2v"):
        raise ValueError(f"DenoiseMethod must be 'nlm' or 'n2v', got {method!r}")
    if method == "n2v":
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("DenoiseMethod=n2v requires torch") from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DenoiseMethod=n2v requires a CUDA GPU; none detected. "
                "Use DenoiseMethod=nlm for CPU-only."
            )
    try:
        from MIDAS_NF_preProc import denoise_directory
    except ImportError as e:
        raise RuntimeError(
            "Denoise=1 requires MIDAS-NF-preProc; "
            "install with: pip install MIDAS-NF-preProc"
        ) from e

    raw_dir = p.get("DataDirectory")
    if not raw_dir:
        raise RuntimeError("Denoise=1 requires DataDirectory in param file")
    denoised_dir = p.get("DenoisedDirectory") or os.path.join(raw_dir, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    work_dir = os.path.join(p["resultFolder"], "_denoise_work")
    mask_threshold = (float(p["DenoiseMaskThreshold"])
                      if "DenoiseMaskThreshold" in p else None)

    logger.info(f"Denoise (method={method}): {raw_dir} -> {denoised_dir}")
    denoise_directory(
        input_dir=raw_dir, output_dir=denoised_dir,
        method=method,
        config_path=p.get("DenoiseConfigFile") or None,
        checkpoint_path=p.get("DenoiseCheckpoint") or None,
        pattern=str(p.get("DenoisePattern", "*.tif")),
        work_dir=work_dir,
        train_jointly=bool(int(p.get("DenoiseTrainJointly", 0))),
        finetune=bool(int(p.get("DenoiseFinetune", 0))),
        mask_threshold=mask_threshold,
        median=(int(p.get("DenoiseNoMedian", 0)) == 0),
    )

    # Re-point DataDirectory in-memory + on-disk so downstream stages
    # (ProcessImagesCombined, etc.) read the denoised TIFFs.
    p["DataDirectory"] = denoised_dir
    from .params import append_param_line
    append_param_line(param_file, "DataDirectory", denoised_dir)
    logger.info(f"Denoise complete; DataDirectory → {denoised_dir}")


# ---------------------------------------------------------------------------
#  Stage 1: preprocessing — hkls + seeds + hex grid + tomo filter + diffr spots
# ---------------------------------------------------------------------------

def _parse_phase_atoms(param_file: str | Path):
    """Build the unit-cell basis from repeated ``PhaseAtom`` lines.

        PhaseAtom <element> <x> <y> <z> [occupancy] [B_iso]

    e.g. a DHCP cell (hP4, sites 2a + 2c)::

        PhaseAtom Ce 0.0 0.0 0.0
        PhaseAtom Ce 0.3333333 0.6666667 0.25

    Read via ``collect_multiline``, not ``parse_parameters``: the latter keeps
    only the FIRST token of an unrecognised key, so the coordinates would be
    silently discarded and every atom would land at the origin.

    Returns None when no atoms are declared, which keeps ``hkls.csv``
    byte-identical to the historical output.
    """
    from midas_hkls import Atom
    from .params import collect_multiline

    rows = collect_multiline(param_file, "PhaseAtom")
    if not rows:
        return None
    atoms = []
    for r in rows:
        toks = r.split()
        if len(toks) < 4:
            raise ValueError(
                f"PhaseAtom needs '<element> <x> <y> <z>', got {r!r}")
        atoms.append(Atom(
            str(toks[0]), (float(toks[1]), float(toks[2]), float(toks[3])),
            occupancy=float(toks[4]) if len(toks) > 4 else 1.0,
            B_iso=float(toks[5]) if len(toks) > 5 else 0.0,
        ))
    return atoms


def run_get_hkls(p: Dict[str, Any], param_file: str | Path) -> str:
    """Generate ``hkls.csv`` via :func:`midas_hkls.write_nf_hkls_csv`.

    With a ``PhaseAtom`` basis declared, an ``F2`` column (|F|^2 normalised) is
    appended, and ``DropForbiddenReflections 1`` additionally removes rows with
    |F|^2 = 0 before anything downstream sees the file.

    Why dropping matters: space-group extinction rules cannot see
    basis-dependent extinctions, so the list can contain reflections no crystal
    can produce. They are predicted, never matched, and sit in the confidence
    denominator -- 126 of 736 for a DHCP polytype, capping FracOverlap at 0.829,
    while fcc (one atom at the origin) has none and reaches 1.000.

    Filtering HERE rather than in the fraction calculation is deliberate: every
    downstream stage (diffr-spots, the orientation search in ``screen``, the
    refine) reads this one file, so removing the rows fixes the SEARCH as well
    as the reported number, with no change to those code paths.
    """
    from midas_hkls import Lattice, SpaceGroup, write_nf_hkls_csv

    sg_nr = int(p.get("SpaceGroup", p.get("SGNr", 225)))
    lp = p["LatticeParameter"]
    sg = SpaceGroup.from_number(sg_nr)
    lat = Lattice(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5])
    wl = float(p["Wavelength"])
    lsd = _resolve_lsd(p, param_file)
    max_r = _resolve_max_ring_rad(p)
    atoms = _parse_phase_atoms(param_file)

    out = Path(p["resultFolder"]) / "hkls.csv"
    n = write_nf_hkls_csv(
        out, sg, lat,
        wavelength_A=wl, lsd_um=lsd, max_ring_rad_um=max_r, atoms=atoms,
    )
    logger.info(f"Generated hkls.csv with {n} reflections at {out}")

    if atoms is not None and int(p.get("DropForbiddenReflections", 0)) == 1:
        eps = float(p.get("ForbiddenF2Threshold", 1e-6))
        lines = out.read_text().splitlines()
        header, body = lines[0], lines[1:]
        kept = [ln for ln in body
                if ln.strip() and float(ln.split()[11]) > eps]
        dropped = len(body) - len(kept)
        out.write_text("\n".join([header] + kept) + "\n")
        logger.info(
            f"DropForbiddenReflections: removed {dropped} of {len(body)} "
            f"reflections with |F|^2 <= {eps:g} (kept {len(kept)}); "
            f"max achievable FracOverlap was {len(kept)/max(len(body),1):.3f}"
        )
    return str(out)


def run_seed_orientations_from_ff(p: Dict[str, Any]) -> str:
    """Far-field → near-field seed conversion (port of
    GenSeedOrientationsFF2NFHEDM via midas_nf_preprocess.seed_orientations).
    """
    import torch

    from midas_nf_preprocess.seed_orientations.from_grains import (
        read_grains_orientations,
    )
    from midas_nf_preprocess.seed_orientations.io import write_seeds_csv

    grains_path = p["GrainsFile"]
    out_path = p["SeedOrientations"]
    grains = read_grains_orientations(grains_path)
    if not grains:
        raise ValueError(f"No grain orientations parsed from {grains_path}")
    # read_grains_orientations returns list[GrainOrientation], but write_seeds_csv
    # wants an (N, 4) quaternion tensor -- stack the .quat fields.
    seeds = torch.stack([g.quat for g in grains])
    # NOTE argument order: write_seeds_csv(quats, path), not (path, quats).
    write_seeds_csv(seeds, out_path)
    logger.info(f"Wrote {len(grains)} seed orientations: {grains_path} → {out_path}")
    return out_path


#: Seed resolutions that reproduce the shipped cache's density per lookup type.
#: Cubic 2.8 deg -> 251,545 against the cache's 243,129; hexagonal 2.25 deg ->
#: 486,946 against 486,755. Generating at the sampler default (1.5 deg) would
#: give ~2.8x as many seeds and cost ~2.8x the fitting time for no gain.
#: Keys are ``space_group_to_lookup_type`` values ("cubic_high", not "cubic") --
#: matched by PREFIX so cubic_low/hexagonal_low resolve too.
_SCRATCH_RESOLUTION_DEG = {"cubic": 2.8, "hexagonal": 2.25}
_SCRATCH_RESOLUTION_DEFAULT = 2.5


def _scratch_resolution(lookup: str) -> float:
    for prefix, res in _SCRATCH_RESOLUTION_DEG.items():
        if lookup.startswith(prefix):
            return res
    return _SCRATCH_RESOLUTION_DEFAULT


def run_seed_orientations_from_cache(p: Dict[str, Any], install_dir: Optional[str] = None) -> str:
    """Seed orientations from the MIDAS lookup cache, or generated if absent.

    The cache lives in the source tree (``NF_HEDM/seedOrientations``) and is NOT
    shipped in the wheel, so a plain ``pip install midas-suite`` has no cache at
    all and this used to raise -- leaving a pip-only user with no way to run NF
    without hand-generating seeds. The orientations are derivable, so derive
    them rather than fail.
    """
    from midas_nf_preprocess.seed_orientations.from_cache import (
        load_seeds_for_space_group, DEFAULT_SEED_DIR, SeedCacheNotFound,
        space_group_to_lookup_type,
    )
    from midas_nf_preprocess.seed_orientations.io import write_seeds_csv

    sg = int(p.get("SpaceGroup", p.get("SGNr", 225)))
    seed_dir = DEFAULT_SEED_DIR
    if install_dir:
        candidate = Path(install_dir) / "NF_HEDM" / "seedOrientations"
        if candidate.is_dir():
            seed_dir = candidate
    out_path = p["SeedOrientations"]

    try:
        seeds = load_seeds_for_space_group(sg, seed_dir=seed_dir)
        source = f"cache ({seed_dir})"
    except (SeedCacheNotFound, FileNotFoundError, OSError):
        from midas_nf_preprocess.seed_orientations.from_scratch import (
            generate_uniform_seeds,
        )
        try:
            lookup = space_group_to_lookup_type(sg)
        except Exception:
            lookup = ""
        res = _scratch_resolution(lookup)
        logger.info(
            "Seed cache not found at %s; generating seeds for SG %d from "
            "scratch at %.2f deg resolution (this is a one-off, ~1 min)",
            seed_dir, sg, res)
        seeds = generate_uniform_seeds(sg, resolution_deg=res)
        source = f"generated, {res:g} deg"

    # NOTE argument order: write_seeds_csv(quats, path), not (path, quats).
    write_seeds_csv(seeds, out_path)
    logger.info(f"Seed orientations for SG {sg}: {len(seeds)} [{source}] -> {out_path}")
    return out_path


def run_hex_grid(p: Dict[str, Any], param_file: str | Path) -> str:
    """Generate ``grid.txt`` via :class:`midas_nf_preprocess.hex_grid.HexGrid`."""
    from midas_nf_preprocess.hex_grid.params import HexGridParams
    from midas_nf_preprocess.hex_grid.grid import HexGrid

    grid_params = HexGridParams.from_paramfile(str(param_file))
    grid = HexGrid.from_params(grid_params)
    out_path = Path(p["resultFolder"]) / grid_params.grid_filename
    grid.write(str(out_path))
    logger.info(f"Generated hex grid with {grid.n_points} voxels at {out_path}")
    return str(out_path)


def run_tomo_filter(p: Dict[str, Any]) -> None:
    """Mask grid points by a tomography image (port of filterGridfromTomo)."""
    if not p.get("TomoImage") or len(str(p["TomoImage"])) < 1:
        return
    import torch

    from midas_nf_preprocess.tomo_filter.filter import (
        filter_grid_by_tomo, load_square_tomo,
    )

    tomo_path = p["TomoImage"]
    tomo_pixel_size = float(p.get("TomoPixelSize", 1.0))
    tomo = load_square_tomo(tomo_path)
    grid_path = Path(p["resultFolder"]) / "grid.txt"
    if not grid_path.exists():
        return

    # filter_grid_by_tomo takes an (N, 5) TENSOR and returns
    # (filtered_points, mask) -- it does no file I/O at all
    # (tomo_filter/filter.py:121-150). The previous call passed a path string,
    # used the keyword `tomo_pixel_size` (the parameter is `px_tomo_um`), and
    # treated the return value as an output path.
    pts = np.genfromtxt(str(grid_path), skip_header=1, delimiter=" ")
    pts = np.atleast_2d(pts)
    kept, _mask = filter_grid_by_tomo(
        torch.as_tensor(pts, dtype=torch.float64), tomo, tomo_pixel_size,
    )
    kept_np = kept.cpu().numpy()

    # Match the legacy script: original becomes grid_unfilt.txt, filtered → grid.txt.
    shutil.move(str(grid_path), str(grid_path.with_name("grid_unfilt.txt")))
    np.savetxt(
        str(grid_path), kept_np, fmt="%.6f", delimiter=" ",
        header=str(kept_np.shape[0]), comments="",
    )
    logger.info(
        f"Tomo-filtered grid: {pts.shape[0]} → {kept_np.shape[0]} points at {grid_path}"
    )


def run_grid_mask(p: Dict[str, Any]) -> None:
    """Apply a rectangular ``GridMask`` ``[xmin, xmax, ymin, ymax]`` filter."""
    if not p.get("GridMask") or not isinstance(p["GridMask"], list) or len(p["GridMask"]) != 4:
        return
    grid_path = Path(p["resultFolder"]) / "grid.txt"
    if not grid_path.exists():
        return
    pts = np.genfromtxt(str(grid_path), skip_header=1, delimiter=" ")
    m = p["GridMask"]
    keep = (pts[:, 2] >= m[0]) & (pts[:, 2] <= m[1]) & (pts[:, 3] >= m[2]) & (pts[:, 3] <= m[3])
    pts = pts[keep]
    shutil.move(str(grid_path), str(grid_path.with_name("grid_old.txt")))
    np.savetxt(
        str(grid_path), pts, fmt="%.6f", delimiter=" ",
        header=str(pts.shape[0]), comments="",
    )
    logger.info(f"GridMask kept {pts.shape[0]} grid points")


def run_diffr_spots(p: Dict[str, Any], param_file: str | Path, *,
                    device: Optional[str] = None,
                    dtype: Optional[str] = None) -> None:
    """Run :class:`midas_nf_preprocess.diffr_spots.DiffrSpotsPipeline`."""
    from midas_nf_preprocess.diffr_spots.cli import run as diffr_run
    # hkls_csv / seeds are read unconditionally by diffr_spots.cli.run
    # (cli.py:41-48); omitting them raises AttributeError.
    #
    # They must be passed EXPLICITLY, not left None. DiffrSpotsParams resolves
    # its defaults against `data_directory` (diffr_spots/params.py:20,43), but
    # the orchestrator writes hkls.csv into the per-layer `resultFolder`
    # (run_get_hkls / workflows.py LayerNr_<n>). Leaving them None therefore
    # sends the callee looking for hkls.csv inside the raw DATA directory and it
    # raises FileNotFoundError.
    result_folder = Path(p.get("resultFolder", ".")).resolve()
    hkls_path = result_folder / "hkls.csv"

    seed_raw = p.get("SeedOrientations") or ""
    seed_path = Path(seed_raw)
    if seed_raw and not seed_path.is_absolute():
        seed_path = result_folder / seed_path

    # device/dtype are threaded in, NOT left None. Left None the callee
    # auto-detects and grabs CUDA even when the user asked for --device cpu,
    # which on a shared GPU means an OOM (or an NVML assert) in a stage that had
    # no reason to be on the GPU at all.
    args = Namespace(
        parameter_file=str(param_file),
        device=device, dtype=dtype,
        output_dir=str(result_folder),
        hkls_csv=str(hkls_path) if hkls_path.exists() else None,
        seeds=str(seed_path) if seed_raw and seed_path.exists() else None,
    )
    diffr_run(args)
    logger.info("Diffraction spots simulated")


def run_preprocessing(
    p: Dict[str, Any], param_file: str | Path, *,
    skip_hex_grid: bool = False,
    skip_diffr_spots: bool = False,
    ff_seed_orientations: bool = False,
    install_dir: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    """Run the full preprocessing block (HKLs → seeds → grid → diffr spots)."""
    run_get_hkls(p, param_file)

    if ff_seed_orientations:
        run_seed_orientations_from_ff(p)
    elif not p.get("SeedOrientations") or not Path(p["SeedOrientations"]).exists():
        run_seed_orientations_from_cache(p, install_dir=install_dir)

    # Update param file with the actual orientation count (NF C codes need it).
    seed_path = p.get("SeedOrientations", "")
    if seed_path and Path(seed_path).exists():
        with open(seed_path) as f:
            n_orient = sum(1 for _ in f)
        from .params import update_param_file
        update_param_file(param_file, {"NrOrientations": str(n_orient)})

    if not skip_hex_grid:
        run_hex_grid(p, param_file)
        run_tomo_filter(p)
        run_grid_mask(p)

    if not skip_diffr_spots:
        run_diffr_spots(p, param_file, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
#  Stage 2: image processing — ProcessImagesCombined per detector distance
# ---------------------------------------------------------------------------

def run_image_processing(p: Dict[str, Any], param_file: str | Path, *,
                         device: Optional[str] = None,
                         dtype: Optional[str] = None) -> None:
    """Loop over detector distances and run the combined median + peak
    pipeline (port of ``ProcessImagesCombined`` in
    :mod:`midas_nf_preprocess.process_images`).
    """
    from midas_nf_preprocess.process_images.cli import run as proc_run

    # ONE call with all_layers=True, NOT a loop over distances.
    #
    # Two reasons. (1) process_images.cli.run reads args.all_layers, args.layer_nr
    # and args.output (cli.py:49-60); the old Namespace supplied `distance_nr`,
    # which no callee reads, and omitted all three -> AttributeError.
    # (2) Even with the names fixed, looping would be WRONG: each call writes the
    # whole SpotsInfo.bin, so distance d+1 overwrites distance d and only the last
    # distance's bits survive (pipeline.py:229-243). SpotsInfo.bin is sized for
    # nDistances * NrFilesPerDistance * NrPixelsY * NrPixelsZ bits and must be
    # produced in a single pass.
    n_distances = int(p.get("nDistances", 1))
    logger.info(f"ProcessImages: all {n_distances} distance(s) in one pass")
    args = Namespace(
        parameter_file=str(param_file),
        n_cpus=int(p.get("nCPUs", 1)),
        device=device, dtype=dtype,      # honour --device; see run_diffr_spots
        all_layers=True, layer_nr=1, output=None,
    )
    proc_run(args)


# ---------------------------------------------------------------------------
#  Stage 3: orientation fitting (FitOrientationOMP equivalent)
# ---------------------------------------------------------------------------

def _fit_sharded(
    p: Dict[str, Any], param_file: str | Path, gpus, *,
    n_cpus: int, dtype: str, refine: str,
) -> None:
    """Fan the voxel range out across ``gpus``, one process per GPU.

    ``grid.slice_block(block_nr, n_blocks)`` gives each block a contiguous,
    disjoint voxel range, and MicWriter indexes by ABSOLUTE voxel index, so the
    workers write non-overlapping rows of the same files.

    The parent allocates those files first and the workers open them with
    ``--no-create-output``: MicWriter's default zeroes the whole file on open,
    so without this every worker would wipe the others' records.
    """
    import subprocess
    import sys

    from midas_nf_fitorientation.output import MicWriter
    from midas_nf_fitorientation.params import parse_paramfile as _pp

    rf = Path(p["resultFolder"])
    mic_bin = p.get("MicFileBinary")
    if not mic_bin:
        raise ValueError("MicFileBinary is required for sharded fitting")
    mic_path = rf / mic_bin

    # Voxel count comes from the grid file's header line.
    grid_path = rf / p.get("GridFileName", "grid.txt")
    with open(grid_path) as fh:
        n_voxels = int(fh.readline().split()[0])

    try:
        n_saves = int(_pp(str(param_file)).save_n_solutions)
    except Exception:
        n_saves = int(p.get("SaveNSolutions", 1))

    MicWriter.allocate(mic_path, n_voxels=n_voxels, n_saves=n_saves)
    logger.info(
        f"Sharded fit: {n_voxels} voxels over {len(gpus)} GPU(s) {list(gpus)}; "
        f"outputs pre-allocated at {mic_path}"
    )

    # Resolve the worker entry point. shutil.which() is NOT enough: under a
    # non-interactive ssh the env's bin/ is often off PATH. Fall back to the
    # sibling of the running interpreter, then to calling the console-script
    # function directly ('midas_nf_fitorientation' has no __main__, so
    # `python -m midas_nf_fitorientation` does not work).
    exe = shutil.which("midas-nf-fit-orientation")
    if not exe:
        sibling = Path(sys.executable).parent / "midas-nf-fit-orientation"
        if sibling.exists():
            exe = str(sibling)
    if exe:
        base = [exe]
    else:
        base = [sys.executable, "-c",
                "import sys;from midas_nf_fitorientation.cli import "
                "fit_orientation_main as m;sys.exit(m())"]

    procs = []
    for i, gpu in enumerate(gpus):
        cmd = base + [
            str(param_file), str(i), str(len(gpus)), str(max(1, n_cpus // len(gpus))),
            "--device", "cuda", "--no-create-output", "--refine", refine,
        ]
        if str(dtype) in ("float32", "fp32"):
            cmd.append("--fp32")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logfile = open(rf / f"fit_block{i}.log", "w")
        procs.append((i, gpu, subprocess.Popen(cmd, env=env, stdout=logfile,
                                               stderr=subprocess.STDOUT), logfile))

    failed = []
    for i, gpu, proc, logfile in procs:
        rc = proc.wait()
        logfile.close()
        if rc != 0:
            failed.append((i, gpu, rc))
    if failed:
        raise RuntimeError(
            "sharded fit failed for block(s): "
            + ", ".join(f"block {i} on GPU {g} (rc={rc})" for i, g, rc in failed)
            + f" -- see {rf}/fit_block*.log"
        )
    logger.info(f"Sharded fit complete across {len(gpus)} GPU(s)")


def run_fitting(
    p: Dict[str, Any], param_file: str | Path, *,
    n_blocks: int = 1, block_nr: int = 0,
    n_cpus: int = 1, device: str = "auto",
    dtype: str = "float64",
    refine: str = "nm-batched",
    gpus=None,
) -> None:
    """Run :func:`midas_nf_fitorientation.fit_orientation_run`.

    ``dtype`` is the new (post-auto-detect) plumbing: when the pipeline
    CLI resolves ``--dtype auto`` it lands here as ``'float32'`` on
    cuda/mps and ``'float64'`` on cpu. ``refine`` selects the phase-2
    strategy — ``nm-batched`` is the safe default; ``nm-triton`` is
    auto-selected by midas-nf-fitorientation when device=cuda + Triton
    is available, and is the production-fastest path.
    """
    import torch
    from midas_nf_fitorientation import fit_orientation_run

    # Delete stale MicFileBinary so fitting starts from a clean slate.
    mic_bin = p.get("MicFileBinary")
    if mic_bin:
        bin_path = Path(p["resultFolder"]) / mic_bin
        if bin_path.exists():
            bin_path.unlink()
            logger.info(f"Removed stale MicFileBinary: {bin_path}")

    # Multi-GPU: fan disjoint voxel blocks out, one process per GPU.
    if gpus and len(list(gpus)) > 1:
        _fit_sharded(p, param_file, list(gpus),
                     n_cpus=n_cpus, dtype=dtype, refine=refine)
        return

    # Map "float32" / "float64" / "fp32" / "fp64" → torch.dtype.
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float64": torch.float64, "fp64": torch.float64,
    }
    torch_dtype = dtype_map.get(str(dtype), torch.float64)

    fit_orientation_run(
        str(param_file), block_nr=block_nr, n_blocks=n_blocks,
        n_cpus=n_cpus, device=device, dtype=torch_dtype,
        refine=refine, verbose=False,
    )


# ---------------------------------------------------------------------------
#  Stage 4: post-process (ParseMic) + consolidation
# ---------------------------------------------------------------------------

def run_parse_mic(p: Dict[str, Any]) -> dict:
    """Run :func:`midas_nf_pipeline.parse_mic.parse_mic`."""
    from .parse_mic import ParseMicParams, parse_mic

    rf = p["resultFolder"]
    mic_bin = p.get("MicFileBinary", "")
    mic_text = p.get("MicFileText", "")
    if not mic_bin or not mic_text:
        raise ValueError("MicFileBinary and MicFileText required for parse_mic")

    bin_path = Path(rf) / mic_bin
    out_path = Path(rf) / (mic_text + ".mic")

    out = parse_mic(ParseMicParams(
        PhaseNr=int(p.get("PhaseNr", 1)),
        NumPhases=int(p.get("NumPhases", 1)),
        GlobalPosition=float(p.get("GlobalPosition", 0.0)),
        inputfile=str(bin_path),
        outputfile=str(out_path),
        nSaves=int(p.get("SaveNSolutions", 1)),
        SGNr=int(p.get("SpaceGroup", p.get("SGNr", 225))),
        GBAngle=float(p.get("GBAngle", 5.0)),
    ))
    logger.info(f"ParseMic produced {len(out)} output files")
    return out


def run_consolidate(p: Dict[str, Any], param_text: str, args_namespace=None,
                    output_path: Optional[str] = None,
                    resolution_label: Optional[str] = None) -> str:
    """Bundle outputs into a consolidated H5."""
    from .consolidate import generate_consolidated_hdf5

    rf = p["resultFolder"]
    mic_text_name = p.get("MicFileText", "")
    if not mic_text_name:
        raise ValueError("MicFileText not set")
    mic_text_path = Path(rf) / (mic_text_name + ".mic")
    return generate_consolidated_hdf5(
        str(mic_text_path),
        param_text=param_text,
        args_namespace=args_namespace,
        output_path=output_path,
        resolution_label=resolution_label,
    )


# ---------------------------------------------------------------------------
#  Multi-resolution helpers
# ---------------------------------------------------------------------------

def run_mic_to_grains(
    param_file: str | Path, mic_file: str | Path, out_file: str | Path,
    *, do_neighbor_search: int = 0, n_cpus: int = 1,
    min_conf_override: Optional[float] = None,
) -> int:
    """Run :func:`midas_nf_pipeline.mic2grains.mic_to_grains`."""
    from .mic2grains import Mic2GrainsParams, mic_to_grains
    return mic_to_grains(Mic2GrainsParams(
        param_file=param_file,
        mic_file=mic_file,
        out_file=out_file,
        do_neighbor_search=int(do_neighbor_search),
        n_cpus=int(n_cpus),
        min_conf_override=min_conf_override,
    ))
