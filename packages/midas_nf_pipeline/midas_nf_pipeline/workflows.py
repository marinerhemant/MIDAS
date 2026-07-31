"""Pipeline orchestrators.

Two public entry points:

  - :func:`run_layer_pipeline` — process a *single* layer via the
    multi-resolution loop (which collapses to single-resolution when
    ``NumLoops == 0``).
  - :func:`run_multi_layer` — outer driver that loops
    ``startLayerNr..endLayerNr``, calling :func:`run_layer_pipeline`
    for each. Per-layer outputs go into
    ``<resultFolder>/LayerNr_<n>/`` with the param file's
    ``RawStartNr`` offset by ``(layer_nr - 1) * nDistances *
    NrFilesPerDistance``.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import stages
from .params import (
    append_param_line,
    parse_parameters,
    update_param_file,
)
from .state import PipelineH5, get_completed_stages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _change_directory(new_dir: str | Path):
    old = os.getcwd()
    try:
        os.chdir(str(new_dir))
        yield
    finally:
        os.chdir(old)


def _stage_label(loop: int, pass_name: str) -> str:
    """Match the labels used by the legacy nf_MIDAS_Multiple_Resolutions.py."""
    return f"loop_{loop}_{pass_name}"


def _all_stage_labels(num_loops: int) -> List[str]:
    """All ordered stage labels for resume bookkeeping."""
    out = [_stage_label(0, "initial")]
    for li in range(1, num_loops + 1):
        out.extend([
            _stage_label(li, "seeded"),
            _stage_label(li, "unseeded"),
            _stage_label(li, "merge"),
        ])
    return out


def _strip_loop_suffix(name: str) -> str:
    """Strip ``.<n>`` / ``_merged.<n>`` / ``_all_solutions.<n>`` accumulated
    by previous interrupted multi-resolution runs."""
    return re.sub(r"(_all_solutions|_merged)?(\.\d+)+$", "", name)


# ---------------------------------------------------------------------------
#  Diffraction-spot file backup/restore (loop 0 result reused in unseeded passes)
# ---------------------------------------------------------------------------

_DIFFR_FILES = ("DiffractionSpots.bin", "OrientMat.bin", "Key.bin")


def _backup_diffr_spots(rf: str | Path, suffix: str = "_unseeded_backup") -> None:
    rf = Path(rf)
    for fn in _DIFFR_FILES:
        src = rf / fn
        if src.exists():
            shutil.copy2(src, rf / (fn + suffix))


def _restore_diffr_spots(rf: str | Path, suffix: str = "_unseeded_backup") -> None:
    rf = Path(rf)
    for fn in _DIFFR_FILES:
        src = rf / (fn + suffix)
        if src.exists():
            shutil.copy2(src, rf / fn)


# ---------------------------------------------------------------------------
#  Bad-voxel filter (loops ≥ 1)
# ---------------------------------------------------------------------------

def _read_grid_xy_index(grid_path: str | Path) -> tuple[Dict[tuple, str], Dict[tuple, int]]:
    """Map ``(x, y) → (grid line, voxel index)`` from a ``grid.txt``."""
    line_map: Dict[tuple, str] = {}
    idx_map: Dict[tuple, int] = {}
    voxel_idx = 0
    with open(grid_path, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) < 5:
                continue
            try:
                kx, ky = float(tokens[2]), float(tokens[3])
            except ValueError:
                continue
            line_map[(kx, ky)] = line.strip()
            idx_map[(kx, ky)] = voxel_idx
            voxel_idx += 1
    return line_map, idx_map


def _filter_bad_voxels(
    mic_pass1_path: str | Path,
    grid_pass1_map: Dict[tuple, str],
    grid_pass1_idx: Dict[tuple, int],
    min_conf: float,
) -> tuple[List[str], List[int]]:
    """Identify voxels with confidence < ``min_conf`` and return their grid
    lines + full-grid indices."""
    bad_lines: List[str] = []
    bad_indices: List[int] = []
    with open(mic_pass1_path, "r") as f:
        for line in f:
            if line.startswith("%"):
                continue
            tokens = line.split()
            if len(tokens) < 11:
                continue
            try:
                conf = float(tokens[10])
                x_val = float(tokens[3])
                y_val = float(tokens[4])
            except ValueError:
                continue
            if conf >= min_conf:
                continue
            line_str = grid_pass1_map.get((x_val, y_val))
            idx = grid_pass1_idx.get((x_val, y_val))
            if line_str is None or idx is None:
                # Fall back to tolerance match
                for (gx, gy), gline in grid_pass1_map.items():
                    if abs(gx - x_val) < 1e-5 and abs(gy - y_val) < 1e-5:
                        line_str = gline
                        idx = grid_pass1_idx[(gx, gy)]
                        break
            if line_str is not None and idx is not None:
                bad_lines.append(line_str)
                bad_indices.append(idx)
    return bad_lines, bad_indices


def _merge_seeded_unseeded_binaries(
    rf: str | Path, mic_file_binary: str, n_saves: int,
    bad_indices: List[int],
) -> None:
    """Overlay unseeded-binary records (only for ``bad_indices``) onto the
    seeded binary at the correct full-grid offsets.

    Mirrors the binary-merge block of nf_MIDAS_Multiple_Resolutions.py.
    """
    rf = Path(rf)
    record_size_mic = 11 * 8
    record_size_all = (7 + n_saves * 4) * 8

    seeded_bin = rf / f"{mic_file_binary}.seeded_backup"
    unseeded_bin = rf / mic_file_binary
    merged_bin = rf / mic_file_binary
    if not seeded_bin.exists() or not unseeded_bin.exists():
        logger.warning("Missing seeded/unseeded backup; skipping binary merge.")
        return

    unseeded_save = rf / f"{mic_file_binary}.unseeded_backup"
    shutil.copy2(unseeded_bin, unseeded_save)
    if (unseeded_bin.with_suffix(".AllMatches")).exists():
        shutil.copy2(
            f"{unseeded_bin}.AllMatches",
            f"{unseeded_save}.AllMatches",
        )

    # Start from seeded backup; overlay unseeded at correct offsets.
    shutil.copy2(seeded_bin, merged_bin)
    with open(unseeded_save, "rb") as f_uns:
        with open(merged_bin, "r+b") as f_merged:
            for unseeded_idx, full_idx in enumerate(bad_indices):
                f_uns.seek(unseeded_idx * record_size_mic)
                rec = f_uns.read(record_size_mic)
                if len(rec) != record_size_mic:
                    continue
                f_merged.seek(full_idx * record_size_mic)
                f_merged.write(rec)
    logger.info(f"Merged {len(bad_indices)} unseeded records into seeded binary")

    # Same for AllMatches if present.
    seeded_all = Path(f"{seeded_bin}.AllMatches")
    merged_all = Path(f"{merged_bin}.AllMatches")
    unseeded_all_save = Path(f"{unseeded_save}.AllMatches")
    if seeded_all.exists() and unseeded_all_save.exists():
        shutil.copy2(seeded_all, merged_all)
        with open(unseeded_all_save, "rb") as f_uns:
            with open(merged_all, "r+b") as f_merged:
                for unseeded_idx, full_idx in enumerate(bad_indices):
                    f_uns.seek(unseeded_idx * record_size_all)
                    rec = f_uns.read(record_size_all)
                    if len(rec) != record_size_all:
                        continue
                    f_merged.seek(full_idx * record_size_all)
                    f_merged.write(rec)


# ---------------------------------------------------------------------------
#  Main per-layer driver (multi-res with NumLoops=0 == single-res)
# ---------------------------------------------------------------------------

def run_layer_pipeline(
    args: Namespace,
    *,
    install_dir: Optional[str] = None,
) -> str:
    """Run the full NF pipeline for ONE layer (``args.paramFN`` already
    points at the layer-specific param file copy).

    Returns the path of the consolidated H5.
    """
    t0 = time.time()
    p = parse_parameters(args.paramFN)

    rf = p.get("OutputDirectory") or p.get("DataDirectory") or os.getcwd()
    rf = os.path.abspath(rf)
    os.makedirs(rf, exist_ok=True)
    log_dir = os.path.join(rf, "midas_log")
    os.makedirs(log_dir, exist_ok=True)
    p["resultFolder"] = rf
    p["logDir"] = log_dir
    p["nCPUs"] = int(getattr(args, "nCPUs", p.get("nCPUs", 1)))

    # Optional multi-GPU fan-out for the fitting stage. A comma-separated list
    # ("0,1") splits the voxel range into one disjoint block per GPU, each run
    # in its own process against a shared, pre-allocated output file.
    _raw_gpus = getattr(args, "fitGpus", None) or ""
    _fit_gpus = [g.strip() for g in str(_raw_gpus).split(",") if g.strip()]
    if len(_fit_gpus) > 1:
        logger.info(f"Fitting will be sharded across GPUs: {_fit_gpus}")

    mic_text_raw = p.get("MicFileText", "nf_output")
    mic_base = _strip_loop_suffix(mic_text_raw)
    seed_raw = p.get("SeedOrientations", "nf_seeds.csv")
    seed_base = _strip_loop_suffix(seed_raw)

    grid_refactor = p.get("GridRefactor")
    if grid_refactor:
        starting_grid = float(grid_refactor[0])
        scaling_factor = float(grid_refactor[1])
        num_loops = int(grid_refactor[2])
    else:
        starting_grid = float(p.get("GridSize", 1.0))
        scaling_factor = 1.0
        num_loops = 0
    logger.info(
        f"NF pipeline: rf={rf}  num_loops={num_loops}  "
        f"starting_grid={starting_grid:.4f}  scale={scaling_factor:.2f}"
    )

    # Pipeline H5 (resume / restart support)
    with open(args.paramFN) as pf:
        param_text = pf.read()
    h5_path = os.path.join(rf, f"{mic_base}_pipeline.h5")
    workflow_type = "nf_multi_res" if num_loops > 0 else "nf_midas"
    ph5 = PipelineH5(h5_path, workflow_type, vars(args), param_text)
    ph5.__enter__()
    ph5.write_dataset("parameters/resultFolder", rf)
    ph5.write_dataset("parameters/MicFileText", mic_base)
    ph5.write_dataset("parameters/paramFN", os.path.abspath(args.paramFN))

    resume_from = ""
    if getattr(args, "resume", ""):
        completed = set(get_completed_stages(args.resume))
        if completed:
            for s in _all_stage_labels(num_loops):
                if s not in completed:
                    resume_from = s
                    break
            if not resume_from:
                resume_from = _all_stage_labels(num_loops)[-1]
            logger.info(f"Resuming from stage '{resume_from}' (completed={completed})")
    elif getattr(args, "restartFrom", ""):
        resume_from = args.restartFrom

    skip_idx = -1
    if resume_from and resume_from in _all_stage_labels(num_loops):
        skip_idx = _all_stage_labels(num_loops).index(resume_from)

    def _should_run(loop: int, pass_name: str) -> bool:
        if skip_idx < 0:
            return True
        all_stages = _all_stage_labels(num_loops)
        target = _stage_label(loop, pass_name)
        if target not in all_stages:
            return True
        return all_stages.index(target) >= skip_idx

    consolidated_h5: Optional[str] = None

    try:
        with _change_directory(rf):
            # Optional denoise (one-time; before any loop).
            stages.run_denoise(p, args.paramFN)

            # ----- LOOP 0 (initial unseeded pass) ----------------------------
            update_param_file(args.paramFN, {
                "MicFileText": f"{mic_base}.0",
                "GridSize": f"{starting_grid:.6f}",
            })
            if _should_run(0, "initial"):
                stages.run_preprocessing(
                    p, args.paramFN,
                    ff_seed_orientations=bool(int(getattr(args, "ffSeedOrientations", 0))),
                    install_dir=install_dir,
                )
                if int(getattr(args, "doImageProcessing", 1)) == 1:
                    stages.run_image_processing(p, args.paramFN)
                stages.run_fitting(
                    p, args.paramFN,
                    n_cpus=p["nCPUs"],
                    device=getattr(args, "device", "auto"),
                    dtype=getattr(args, "dtype", "float64"),
                    refine=getattr(args, "refine", "nm-batched"),
                    gpus=_fit_gpus,
                )
                # MicFileText was updated ON DISK above, but `p` is the
                # in-memory copy parsed before that, and run_parse_mic reads
                # p["MicFileText"] (stages.py:502). Without this override
                # loop 0 writes "<base>.mic" while everything downstream --
                # the consolidated H5 below and run_mic_to_grains in loop 1 --
                # looks for "<base>.0.mic", which then FileNotFoundErrors.
                stages.run_parse_mic({**p, "MicFileText": f"{mic_base}.0"})
                if num_loops > 0:
                    _backup_diffr_spots(rf)
                ph5.mark(_stage_label(0, "initial"))
            else:
                logger.info("Skipping loop 0 (resume).")

            # Build / update consolidated H5 for loop 0.
            mic_loop0 = Path(rf) / f"{mic_base}.0.mic"
            if mic_loop0.exists():
                consolidated_h5 = stages.run_consolidate(
                    {**p, "MicFileText": f"{mic_base}.0"},
                    param_text=param_text, args_namespace=args,
                    output_path=os.path.join(rf, f"{mic_base}_consolidated.h5"),
                )
                # Also stash as loop_0_unseeded resolution.
                from .consolidate import add_resolution_to_h5
                add_resolution_to_h5(
                    consolidated_h5, str(mic_loop0),
                    resolution_label="loop_0_unseeded",
                    grid_size=starting_grid, pass_type="unseeded",
                )

            current_mic = f"{mic_base}.0"

            # ----- REFINEMENT LOOPS -----------------------------------------
            seed_all_backup: Optional[str] = None
            if num_loops > 0:
                seed_all = p.get("SeedOrientationsAll")
                if not seed_all:
                    raise ValueError(
                        "Multi-resolution requires SeedOrientationsAll in the param file"
                    )
                seed_all_p = Path(seed_all)
                if not seed_all_p.is_absolute():
                    seed_all_p = Path(rf) / seed_all_p

                # Nothing in loop 0 creates this file, so on a fresh multi-res run
                # it does not exist and copy2 raised FileNotFoundError.
                #
                # Materialise it from the loop-0 SeedOrientations. That is exactly
                # what it must contain: the FULL candidate list. `SeedOrientations`
                # is rewritten with grain-derived seeds inside every refinement loop
                # (update_param_file below), so the complete list has to be preserved
                # separately for the unseeded pass on bad voxels, which reads it back
                # via seed_all_backup.
                if not seed_all_p.exists():
                    src = Path(p.get("SeedOrientations", ""))
                    if not src.is_absolute():
                        src = Path(rf) / src
                    if not src.exists():
                        raise FileNotFoundError(
                            f"SeedOrientationsAll ({seed_all_p}) does not exist and the "
                            f"loop-0 SeedOrientations ({src}) is missing too, so it "
                            f"cannot be reconstructed."
                        )
                    seed_all_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, seed_all_p)
                    logger.info(
                        f"SeedOrientationsAll did not exist; seeded it from the "
                        f"loop-0 candidate list {src} → {seed_all_p}"
                    )

                seed_all_backup = f"{seed_all_p}_Backup"
                shutil.copy2(seed_all_p, seed_all_backup)
                logger.info(f"SeedOrientationsAll backed up: {seed_all_backup}")

            for loop_idx in range(1, num_loops + 1):
                logger.info(f">>> Loop {loop_idx}/{num_loops}")
                args.doImageProcessing = 0  # images reused across loops

                new_grid_size = starting_grid / (scaling_factor ** loop_idx)
                update_param_file(args.paramFN, {
                    "GridSize": f"{new_grid_size:.6f}",
                })

                # (a) cluster previous mic into grains for the seeded pass.
                grains_file = f"Grains.csv.{loop_idx}"
                grains_path = os.path.join(rf, grains_file)
                stages.run_mic_to_grains(
                    args.paramFN,
                    os.path.join(rf, current_mic + ".mic"),
                    grains_path,
                    do_neighbor_search=0, n_cpus=p["nCPUs"],
                    min_conf_override=float(getattr(args, "minConfidence", 0.6)),
                )

                # (b) seeded pass.
                seed_loop = f"{seed_base}.{loop_idx}"
                target_mic_seeded = f"{mic_base}.{loop_idx}"
                update_param_file(args.paramFN, {
                    "GrainsFile": grains_path,
                    "MicFileText": target_mic_seeded,
                    "SeedOrientations": seed_loop,
                })
                args.ffSeedOrientations = 1
                p_seeded = parse_parameters(args.paramFN)
                p_seeded["resultFolder"] = rf
                p_seeded["logDir"] = log_dir
                p_seeded["nCPUs"] = p["nCPUs"]

                grid_pass1_map: Dict[tuple, str] = {}
                grid_pass1_idx: Dict[tuple, int] = {}
                if _should_run(loop_idx, "seeded"):
                    stages.run_preprocessing(
                        p_seeded, args.paramFN,
                        ff_seed_orientations=True,
                        install_dir=install_dir,
                    )
                    grid_pass1_map, grid_pass1_idx = _read_grid_xy_index(
                        os.path.join(rf, "grid.txt")
                    )
                    stages.run_fitting(
                        p_seeded, args.paramFN,
                        n_cpus=p["nCPUs"],
                        device=getattr(args, "device", "auto"),
                        dtype=getattr(args, "dtype", "float64"),
                        refine=getattr(args, "refine", "nm-batched"),
                    gpus=_fit_gpus,
                    )
                    stages.run_parse_mic(p_seeded)
                    mic_seeded_path = Path(rf) / f"{target_mic_seeded}.mic"
                    if consolidated_h5 and mic_seeded_path.exists():
                        from .consolidate import add_resolution_to_h5
                        add_resolution_to_h5(
                            consolidated_h5, str(mic_seeded_path),
                            resolution_label=f"loop_{loop_idx}_seeded",
                            grid_size=new_grid_size, pass_type="seeded",
                        )
                    ph5.mark(_stage_label(loop_idx, "seeded"))
                else:
                    logger.info(f"Skipping loop {loop_idx} seeded (resume)")
                    grid_pass1_map, grid_pass1_idx = _read_grid_xy_index(
                        os.path.join(rf, "grid.txt")
                    )

                # (c) bad-voxel filter on the seeded mic.
                mic_seeded_path = os.path.join(rf, f"{target_mic_seeded}.mic")
                min_conf = float(p_seeded.get("MinConfidence", 0.5))
                bad_lines, bad_indices = _filter_bad_voxels(
                    mic_seeded_path, grid_pass1_map, grid_pass1_idx, min_conf,
                )
                logger.info(f"Loop {loop_idx}: {len(bad_lines)} bad voxels (conf < {min_conf})")

                if not bad_lines:
                    # No bad voxels → seeded result is final for this loop.
                    merged_name = f"{mic_base}_merged.{loop_idx}"
                    update_param_file(args.paramFN, {"MicFileText": merged_name})
                    p_m = parse_parameters(args.paramFN)
                    p_m["resultFolder"] = rf
                    p_m["logDir"] = log_dir
                    p_m["nCPUs"] = p["nCPUs"]
                    stages.run_parse_mic(p_m)
                    current_mic = merged_name
                    if consolidated_h5:
                        from .consolidate import add_resolution_to_h5
                        add_resolution_to_h5(
                            consolidated_h5,
                            os.path.join(rf, f"{merged_name}.mic"),
                            resolution_label=f"loop_{loop_idx}_merged",
                            grid_size=new_grid_size, pass_type="merged",
                        )
                    ph5.mark(_stage_label(loop_idx, "merge"))
                    continue

                # (d) write bad-voxel grid + run unseeded pass on those.
                with open(os.path.join(rf, "grid.txt"), "w") as f:
                    f.write(f"{len(bad_lines)}\n")
                    for line in bad_lines:
                        f.write(f"{line}\n")

                target_mic_unseeded = f"{mic_base}_all_solutions.{loop_idx}"
                update_param_file(args.paramFN, {
                    "MicFileText": target_mic_unseeded,
                    "SeedOrientations": seed_all_backup,
                })
                args.ffSeedOrientations = 0
                p_uns = parse_parameters(args.paramFN)
                p_uns["resultFolder"] = rf
                p_uns["logDir"] = log_dir
                p_uns["nCPUs"] = p["nCPUs"]

                if _should_run(loop_idx, "unseeded"):
                    stages.run_preprocessing(
                        p_uns, args.paramFN,
                        skip_hex_grid=True,           # bad-voxel grid was just written
                        skip_diffr_spots=True,        # reuse loop-0 spots
                        install_dir=install_dir,
                    )
                    _restore_diffr_spots(rf)
                    # Backup the seeded binary before unseeded overwrites it.
                    mfb = p_uns.get("MicFileBinary", "")
                    if mfb:
                        seeded_src = Path(rf) / mfb
                        if seeded_src.exists():
                            shutil.copy2(seeded_src, f"{seeded_src}.seeded_backup")
                            am = Path(f"{seeded_src}.AllMatches")
                            if am.exists():
                                shutil.copy2(am, f"{seeded_src}.AllMatches.seeded_backup")
                    stages.run_fitting(
                        p_uns, args.paramFN,
                        n_cpus=p["nCPUs"],
                        device=getattr(args, "device", "auto"),
                        dtype=getattr(args, "dtype", "float64"),
                        refine=getattr(args, "refine", "nm-batched"),
                    gpus=_fit_gpus,
                    )
                    stages.run_parse_mic(p_uns)
                    if consolidated_h5:
                        from .consolidate import add_resolution_to_h5
                        add_resolution_to_h5(
                            consolidated_h5,
                            os.path.join(rf, f"{target_mic_unseeded}.mic"),
                            resolution_label=f"loop_{loop_idx}_unseeded",
                            grid_size=new_grid_size, pass_type="unseeded",
                        )
                    ph5.mark(_stage_label(loop_idx, "unseeded"))

                # (e) merge seeded + unseeded binaries; ParseMic on the merged.
                _merge_seeded_unseeded_binaries(
                    rf,
                    mic_file_binary=p_uns.get("MicFileBinary", ""),
                    n_saves=int(p_uns.get("SaveNSolutions", 1)),
                    bad_indices=bad_indices,
                )
                merged_name = f"{mic_base}_merged.{loop_idx}"
                update_param_file(args.paramFN, {"MicFileText": merged_name})
                p_m = parse_parameters(args.paramFN)
                p_m["resultFolder"] = rf
                p_m["logDir"] = log_dir
                p_m["nCPUs"] = p["nCPUs"]
                stages.run_parse_mic(p_m)
                current_mic = merged_name

                if consolidated_h5:
                    from .consolidate import add_resolution_to_h5
                    add_resolution_to_h5(
                        consolidated_h5,
                        os.path.join(rf, f"{merged_name}.mic"),
                        resolution_label=f"loop_{loop_idx}_merged",
                        grid_size=new_grid_size, pass_type="merged",
                    )
                ph5.mark(_stage_label(loop_idx, "merge"))

    finally:
        ph5.__exit__(None, None, None)

    elapsed = time.time() - t0
    logger.info(f"Layer pipeline finished in {elapsed:.1f}s")
    return consolidated_h5 or h5_path


# ---------------------------------------------------------------------------
#  Multi-layer outer driver
# ---------------------------------------------------------------------------

def run_multi_layer(args: Namespace, *, install_dir: Optional[str] = None) -> List[str]:
    """Process layers ``args.startLayerNr .. args.endLayerNr`` in order.

    Each layer gets its own ``<base_result_folder>/LayerNr_<n>/`` with
    ``OutputDirectory`` and ``RawStartNr`` updated in a per-layer copy
    of the param file.
    """
    base_p = parse_parameters(args.paramFN)
    base_rf = (
        getattr(args, "resultFolder", "")
        or base_p.get("OutputDirectory")
        or base_p.get("DataDirectory")
        or os.getcwd()
    )
    base_rf = os.path.abspath(base_rf)
    os.makedirs(base_rf, exist_ok=True)

    original_raw_start = int(base_p.get("RawStartNr", 0))
    n_distances = int(base_p.get("nDistances", 1))
    n_files_per_distance = int(base_p.get("NrFilesPerDistance", 1))

    start_layer = int(getattr(args, "startLayerNr", 1))
    end_layer = int(getattr(args, "endLayerNr", start_layer))
    if end_layer < start_layer:
        raise ValueError(f"endLayerNr ({end_layer}) < startLayerNr ({start_layer})")
    n_layers = end_layer - start_layer + 1
    logger.info(f"Processing {n_layers} layer(s): {start_layer}..{end_layer}")

    out_h5: List[str] = []
    for layer_nr in range(start_layer, end_layer + 1):
        layer_folder = os.path.join(base_rf, f"LayerNr_{layer_nr}")
        os.makedirs(layer_folder, exist_ok=True)
        layer_param = os.path.join(layer_folder, os.path.basename(args.paramFN))
        shutil.copy2(args.paramFN, layer_param)
        layer_raw_start = original_raw_start + (layer_nr - 1) * n_distances * n_files_per_distance
        update_param_file(layer_param, {
            "OutputDirectory": layer_folder,
            "RawStartNr": str(layer_raw_start),
        })
        layer_args = Namespace(**vars(args))
        layer_args.paramFN = layer_param
        logger.info(
            f"Layer {layer_nr}/{end_layer}: rf={layer_folder} "
            f"RawStartNr={layer_raw_start}"
        )
        h5 = run_layer_pipeline(layer_args, install_dir=install_dir)
        out_h5.append(h5)

        # Mic2GrainsList on the last seeded mic → GrainsLayer<n>.csv.
        layer_p = parse_parameters(layer_param)
        mic_text_raw = layer_p.get("MicFileText", "nf_output")
        mic_base = _strip_loop_suffix(mic_text_raw)
        if "GridRefactor" in layer_p:
            num_loops = int(layer_p["GridRefactor"][2])
            last_mic = os.path.join(layer_folder, f"{mic_base}.{num_loops}.mic")
        else:
            last_mic = os.path.join(layer_folder, f"{mic_base}.0.mic")
        grains_out = os.path.join(base_rf, f"GrainsLayer{layer_nr}.csv")
        if os.path.exists(last_mic):
            stages.run_mic_to_grains(
                layer_param, last_mic, grains_out,
                do_neighbor_search=0,
                n_cpus=int(getattr(args, "nCPUs", 1)),
                min_conf_override=float(getattr(args, "minConfidence", 0.6)),
            )

    logger.info(f"All {n_layers} layers done")
    return out_h5
