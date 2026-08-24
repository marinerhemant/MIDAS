"""The vendored C copies must stay byte-identical to the canonical one.

``midas-index`` and ``midas-fit-grain`` compile their C from their OWN sdists,
so they cannot reach into ``midas_ckernel/c_src`` at build time and each carries
a physical copy.  That is deliberate (see ``utils/sync_vendored_c.py``), and it
costs drift: a fix applied to one copy leaves the three packages computing
different forward models while every other test still passes.

Nothing caught that before this test.  ``test_forward_parity.py`` compares
ckernel's ``forward.c`` against the LEGACY indexer and refiner bodies -- a
different axis entirely -- and would stay green with the mirrors diverged.

The 2026-08-22 BigDetector bounds fix is the case in point: an out-of-bounds
read past an mmap'd bitset, which had to land in all three copies to be fixed
anywhere it runs.
"""
from __future__ import annotations

import filecmp
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_UTILS = _REPO / "utils"

pytestmark = pytest.mark.skipif(
    not (_UTILS / "sync_vendored_c.py").exists(),
    reason="not a source checkout (utils/ absent in an installed wheel)",
)


def _load_syncer():
    sys.path.insert(0, str(_UTILS))
    try:
        import sync_vendored_c  # type: ignore
    finally:
        sys.path.pop(0)
    return sync_vendored_c


def test_every_shared_file_exists_in_the_canonical_dir():
    """The list must describe reality, not a wish."""
    m = _load_syncer()
    missing = [n for n in m.SHARED if not (m.c_src(m.CANONICAL_PKG) / n).exists()]
    assert not missing, f"named in SHARED but absent from the canonical dir: {missing}"


@pytest.mark.parametrize("pkg", ["midas_fit_grain", "midas_index"])
def test_mirror_matches_canonical(pkg):
    m = _load_syncer()
    src_dir, dst_dir = m.c_src(m.CANONICAL_PKG), m.c_src(pkg)
    if not dst_dir.exists():
        pytest.skip(f"{pkg} not present in this checkout")
    drifted = [
        n for n in m.SHARED
        if not (dst_dir / n).exists()
        or not filecmp.cmp(src_dir / n, dst_dir / n, shallow=False)
    ]
    assert not drifted, (
        f"{pkg}/c_src has drifted from {m.CANONICAL_PKG}: {drifted}\n"
        f"Edit the canonical copy, then run: python utils/sync_vendored_c.py"
    )


def test_check_mode_agrees_with_the_per_file_comparison():
    """The tool's own --check must not disagree with this test."""
    m = _load_syncer()
    assert m.check() == [], "sync_vendored_c.check() reports drift"


def test_sync_is_idempotent_when_already_in_sync(tmp_path):
    """A no-op run must write nothing -- otherwise `--check` and `sync` differ
    and the tool would churn the tree on every invocation."""
    m = _load_syncer()
    if m.check():
        pytest.skip("tree already drifted; the drift tests above cover it")
    assert m.sync() == [], "sync() rewrote files that were already identical"
