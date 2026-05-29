"""Batch / multi-layer raw-file auto-discovery.

Mirrors ``ff_MIDAS.py``'s ``-batchMode``: scans ``RawFolder`` for files
matching ``{FileStem}_{NNNNNN}{Ext}``, skips ``dark_*``, returns a sorted
``list[(file_nr, filestem)]``. Used by ``cli._cmd_run`` when ``--batch``
is set.

For each discovered file the pipeline:

  1. patches ``FileStem`` in the parameter file,
  2. resolves NF→FF seed grains for that layer (if ``nf_result_dir``),
  3. runs ``Pipeline._run_layer(layer_nr)`` exactly the way the
     non-batch path does.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from ._logging import LOG
from .config import LayerSelection
from .ff_seeding import patch_params_with_grains, resolve_grains_file_for_layer


def discover_layer_files(
    raw_folder: str,
    ext: str,
    padding: int,
    start_file_nr: int,
    end_file_nr: int,
) -> List[Tuple[int, str]]:
    """Scan ``raw_folder`` for ``{stem}_{NNNNNN}{ext}`` files in range.

    Returns a sorted ``[(file_nr, file_stem), …]``. ``dark_*`` files are
    skipped. Files outside ``[start_file_nr, end_file_nr]`` are skipped.
    """
    if not os.path.isdir(raw_folder):
        LOG.error("RawFolder does not exist: %s", raw_folder)
        return []

    if not ext.startswith("."):
        ext = "." + ext
    pattern = re.compile(rf"^(.+?)_(\d{{{padding}}}){re.escape(ext)}$")

    found: list[tuple[int, str]] = []
    n_darks = 0
    for fname in os.listdir(raw_folder):
        m = pattern.match(fname)
        if not m:
            continue
        stem, num = m.group(1), int(m.group(2))
        if num < start_file_nr or num > end_file_nr:
            continue
        if stem.lower().startswith("dark_"):
            n_darks += 1
            continue
        found.append((num, stem))

    found.sort(key=lambda x: x[0])
    n_missing = (end_file_nr - start_file_nr + 1) - len(found) - n_darks
    LOG.info(
        "Batch discovery in %s: %d data files, %d darks skipped, %d missing",
        raw_folder, len(found), n_darks, max(0, n_missing),
    )
    if found:
        stems = sorted({s for _, s in found})
        LOG.info("  file numbers: %d..%d", found[0][0], found[-1][0])
        if len(stems) <= 5:
            LOG.info("  unique stems: %s", stems)
        else:
            LOG.info("  unique stems: %d different stems", len(stems))
    return found


def _patch_filestem(params_file: Path, file_stem: str) -> None:
    """Replace ``FileStem`` in-place in the parameter file."""
    text = params_file.read_text()
    new_lines: list[str] = []
    seen = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("FileStem"):
            new_lines.append(f"FileStem {file_stem}")
            seen = True
        else:
            new_lines.append(raw)
    if not seen:
        new_lines.append(f"FileStem {file_stem}")
    params_file.write_text("\n".join(new_lines).rstrip() + "\n")


def _read_param(path: Path, key: str, default: str | None) -> str | None:
    if not path.exists():
        return default
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
        if not line:
            continue
        toks = line.split()
        if toks[0] == key and len(toks) >= 2:
            return toks[1].rstrip(";")
    return default


def run_batch(pipe, args) -> None:
    """Drive a batch run. ``pipe`` is a constructed ``Pipeline``; ``args``
    is the parsed argparse namespace from the CLI (we only read layer
    range, params, nf_result_dir, grains_file from the config).
    """
    config = pipe.config
    params_file = Path(config.params_file)

    raw_folder = _read_param(params_file, "RawFolder", ".") or "."
    if config.raw_dir:
        raw_folder = config.raw_dir
    ext = _read_param(params_file, "Ext", ".ge3") or ".ge3"
    if not ext.startswith("."):
        ext = "." + ext
    padding = int(_read_param(params_file, "Padding", "6") or "6")
    start_fn = int(_read_param(params_file, "StartFileNrFirstLayer", "1") or "1")

    layer_sel: LayerSelection = config.layer_selection
    start_file_nr = start_fn + (layer_sel.start - 1)
    end_file_nr = start_fn + (layer_sel.end - 1)

    discovered = discover_layer_files(
        raw_folder, ext, padding, start_file_nr, end_file_nr,
    )
    if not discovered:
        raise RuntimeError("Batch mode: no valid raw files found in range.")

    for file_nr, file_stem in discovered:
        layer_nr = file_nr - start_fn + 1
        LOG.info("=== batch: layer %d (file_nr=%d, stem=%s) ===",
                 layer_nr, file_nr, file_stem)

        _patch_filestem(params_file, file_stem)

        seed = resolve_grains_file_for_layer(
            layer_nr=layer_nr,
            grains_file=config.grains_file,
            nf_result_dir=config.nf_result_dir,
        )
        if seed:
            patch_params_with_grains(params_file, seed)

        pipe._run_layer(layer_nr)
