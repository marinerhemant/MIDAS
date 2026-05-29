"""Resolve per-layer ``GrainsFile`` seeding.

Two seeding paths, in priority order:

1. ``config.nf_result_dir`` — for layer N look for ``GrainsLayer{N}.csv``
   inside the NF result directory and use that as the seed.
2. ``config.grains_file`` — explicit one-grains-file-fits-all-layers.

The chosen seed is written into the layer's ``Parameters.txt`` (so that
``midas-fit-setup`` picks it up) **and** into the merged ``paramstest.txt``
when that already exists. ``MinNrSpots 1`` is appended so the indexer
doesn't reject sparsely-observed seed grains.

The resolver is idempotent: running it twice with the same seed produces
the same files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._logging import LOG


def resolve_grains_file_for_layer(
    *,
    layer_nr: int,
    grains_file: Optional[str],
    nf_result_dir: Optional[str],
) -> Optional[str]:
    """Pick the right seed file for ``layer_nr`` per the resolver rules."""
    if nf_result_dir:
        cand = Path(nf_result_dir) / f"GrainsLayer{layer_nr}.csv"
        if cand.exists():
            LOG.info("  layer %d: seeding from NF grains %s", layer_nr, cand)
            return str(cand.resolve())
        LOG.warning(
            "  layer %d: nf_result_dir set but %s missing — falling back",
            layer_nr, cand,
        )
    if grains_file:
        return grains_file
    return None


def patch_params_with_grains(params_file: Path, grains_path: str) -> None:
    """Add/replace ``GrainsFile`` and ``MinNrSpots 1`` in ``params_file``.

    Acts on the on-disk parameter file in-place. Lines whose key matches
    are replaced; otherwise the key is appended.
    """
    if not params_file.exists():
        raise FileNotFoundError(f"params file not found: {params_file}")
    text = params_file.read_text()
    new_lines: list[str] = []
    seen_grains = False
    seen_minnr = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("GrainsFile"):
            new_lines.append(f"GrainsFile {grains_path}")
            seen_grains = True
        elif stripped.startswith("MinNrSpots"):
            new_lines.append("MinNrSpots 1")
            seen_minnr = True
        else:
            new_lines.append(raw)
    if not seen_grains:
        new_lines.append(f"GrainsFile {grains_path}")
    if not seen_minnr:
        new_lines.append("MinNrSpots 1")
    params_file.write_text("\n".join(new_lines).rstrip() + "\n")


def apply_raw_dir_override(params_file: Path, raw_dir: str) -> None:
    """Patch ``RawFolder`` and ``Dark`` (preserving the dark filename)
    in ``params_file`` to point at ``raw_dir``. Mirrors ff_MIDAS.py's
    ``-rawDir`` behaviour.
    """
    if not params_file.exists():
        raise FileNotFoundError(f"params file not found: {params_file}")
    text = params_file.read_text()
    raw_dir = str(Path(raw_dir).resolve())

    new_lines: list[str] = []
    seen_raw = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("RawFolder"):
            new_lines.append(f"RawFolder {raw_dir}")
            seen_raw = True
        elif stripped.startswith("Dark"):
            toks = stripped.split(None, 1)
            if len(toks) == 2:
                dark_name = Path(toks[1]).name
                new_lines.append(f"Dark {raw_dir}/{dark_name}")
            else:
                new_lines.append(raw)
        else:
            new_lines.append(raw)
    if not seen_raw:
        new_lines.append(f"RawFolder {raw_dir}")
    params_file.write_text("\n".join(new_lines).rstrip() + "\n")
