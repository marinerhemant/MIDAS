"""Stage: consolidation.

FF mode: no-op finalizer. FF grain consolidation is handled upstream by
the :mod:`midas_pipeline.stages.process_grains` stage which invokes
``midas-process-grains``. We return a skipped StageResult here so the
pipeline ledger has a uniform row for both modes.

PF mode: invokes :func:`consolidation_pf.consolidate_pf`, the pure-Python
port of ``pf_MIDAS.py:2429-2519``. PF mode never uses
``midas-process-grains`` — the consolidation logic is inlined as ~90
lines of Python in the legacy workflow and is now lifted here.

The dispatcher is intentionally thin so the parity test can target
:func:`consolidation_pf.consolidate_pf` directly without going through a
``StageContext``.
"""

from __future__ import annotations

from pathlib import Path

from ._base import StageContext
from ._stub import stub_run
from ..results import ConsolidationResult, StageResult


_DEFAULT_SPACE_GROUP = 225            # FCC; matches midas-stress fallback


def _read_space_group(layer_dir: Path) -> int:
    """Best-effort space-group resolver.

    Reads the first ``SpaceGroup <int>`` directive from
    ``<layer_dir>/paramstest.txt`` and falls back to 225 (FCC) if the
    file is missing or doesn't carry the directive. This mirrors the
    pf_MIDAS.py behavior of defaulting silently — the consolidator only
    uses the SG for symmetry-zone reduction inside
    :func:`consolidation_pf.consolidate_pf`.
    """
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return _DEFAULT_SPACE_GROUP
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "SpaceGroup":
            # Strip trailing punctuation: legacy paramstest.txt uses C
            # param-file syntax ``SpaceGroup 194;`` which would raise
            # ValueError and silently fall back to FCC = 225.
            digits = "".join(c for c in toks[1] if c.isdigit())
            if digits:
                return int(digits)
    return _DEFAULT_SPACE_GROUP


def run(ctx: StageContext) -> StageResult:
    """FF → stub; PF → :func:`consolidation_pf.consolidate_pf`."""
    if ctx.is_ff:
        return stub_run("consolidation", ctx)
    # PF mode — dispatch to the pure-Python port.
    from .consolidation_pf import consolidate_pf
    space_group = _read_space_group(ctx.layer_dir)
    n_scans = int(ctx.config.scan.n_scans)
    # n_grains is computed inside consolidate_pf from the per-voxel
    # CSV stack; we pass 0 here as a signal to "auto-detect from
    # disk", which is the legacy behaviour at pf_MIDAS.py:2429.
    result = consolidate_pf(
        layer_dir=Path(ctx.layer_dir),
        n_grains=0,
        n_scans=n_scans,
        space_group=space_group,
    )
    # Always surface the space-group choice in metrics so callers can
    # verify the dispatcher's defaulting behaviour.
    if not isinstance(result.metrics, dict):
        result.metrics = {}
    result.metrics.setdefault("space_group", space_group)
    result.metrics.setdefault("n_scans", n_scans)
    result.skipped = False
    return result
