"""A saturated region is deleted whole. That has to at least be countable.

``find_regional_maxima`` returns ``None`` for any region containing a pixel
above ``IntSat`` (``UpperBoundThreshold``), so the entire region -- every peak
in it -- disappears. There is no flag column, and until 2026-08-22 no count
either, which makes the loss invisible in two directions at once:

* a saturated reflection is a STRONG one, so its absence is later scored as
  incompleteness; the grain is penalised for having produced too much signal;
* it is also the brightest contributor to its ring's ``powder_int``
  normalisation (``midas_transforms/radius/core.py:153``), so dropping it
  biases every grain volume on that ring upward.

These tests pin the two properties the counting relies on: saturation is the
*only* reason ``seed_region`` returns ``None`` (otherwise the count means
something else), and every producer return path carries the count slot
(otherwise the consumer raises on unpack -- and the error paths are exactly
the ones no test exercises).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from midas_peakfit.connected import Region
from midas_peakfit.seeds import find_regional_maxima, seed_region


def _region(vals, r0=50, c0=50):
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.size
    return Region(
        id=1,
        intensities=vals,
        pixel_rows=np.arange(r0, r0 + n, dtype=np.int64),
        pixel_cols=np.full(n, c0, dtype=np.int64),
    )


@pytest.fixture
def img():
    return np.zeros((256, 256), dtype=np.float64)


def test_saturated_region_returns_none(img):
    reg = _region([10.0, 900.0, 20.0])
    assert find_regional_maxima(reg, img, np.zeros((256, 256)), 500.0, 10) is None


def test_unsaturated_region_does_not(img):
    reg = _region([10.0, 400.0, 20.0])
    assert find_regional_maxima(reg, img, np.zeros((256, 256)), 500.0, 10) is not None


def test_one_pixel_over_the_line_kills_the_whole_region(img):
    """The cut is per-pixel but the consequence is per-region."""
    below = _region([10.0, 499.0, 20.0, 480.0, 30.0])
    above = _region([10.0, 501.0, 20.0, 480.0, 30.0])
    m = np.zeros((256, 256))
    assert find_regional_maxima(below, img, m, 500.0, 10) is not None
    assert find_regional_maxima(above, img, m, 500.0, 10) is None, (
        "a single pixel 2 counts higher deletes 5 pixels' worth of peaks"
    )


def test_saturation_is_the_only_none_from_seed_region():
    """The counters say ``sr is None`` means saturated. Hold that true.

    If another early ``return None`` is ever added to ``seed_region``, the
    per-frame "Saturated (dropped)" number silently starts counting something
    else -- the kind of drift that makes a diagnostic worse than none.
    """
    src = inspect.getsource(seed_region)
    body = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    n_returns_none = body.count("return None")
    assert n_returns_none == 1, (
        f"seed_region has {n_returns_none} `return None` paths; the saturation "
        "counter assumes exactly one. Give the new one its own counter."
    )


def test_consumer_tolerates_a_four_tuple_from_an_older_producer():
    """A half-applied deploy must degrade the counter, not kill the scan.

    ``orchestrator.py`` and ``_producer_worker.py`` are separate files in a
    site-packages env that a live multi-layer campaign runs off, and the
    saturation counter widened the contract between them. A partial deploy
    breaks in BOTH directions — a 5-tuple unpack against a 4-tuple return, or
    the reverse.

    ``n_saturated`` is a diagnostic: losing it costs a log line, whereas
    raising costs the layer, and midas_pipeline logs a failed scan as a
    WARNING and finishes from whatever survived — so the failure would be
    nearly invisible. Hence the consumer unpacks ``*rest``.
    """
    import inspect

    from midas_peakfit import orchestrator

    src = inspect.getsource(orchestrator)
    # Every consumer of a producer result must use a starred unpack.
    strict = [
        l.strip() for l in src.splitlines()
        if "seeded_list, n_sat = result" in l
    ]
    assert not strict, (
        f"strict 5-tuple unpack(s) remain, which will raise against an older "
        f"_producer_worker.py: {strict}"
    )
    assert src.count("seeded_list, *_sat = result") == 3, (
        "expected all three unpack sites (ingest, process pool, threaded) to "
        "tolerate both arities"
    )

    # And the tolerant form actually behaves: simulate both shapes.
    for result, want in ((("f", 0.0, 7, [], 4), 4), (("f", 0.0, 7, []), 0)):
        frame_nr, omega, n_regs, seeded_list, *_sat = result
        assert (_sat[0] if _sat else 0) == want


@pytest.mark.parametrize(
    "func_name, module_name",
    [
        ("process_frame_in_worker", "midas_peakfit._producer_worker"),
    ],
)
def test_every_producer_return_path_has_the_count_slot(func_name, module_name):
    """Arity must match on the ERROR paths too.

    The read-failure branches are the ones no happy-path test reaches, so a
    4-tuple left behind there would surface only on a corrupt frame -- i.e.
    during the run you least want a new exception in.
    """
    import importlib

    mod = importlib.import_module(module_name)
    src = inspect.getsource(getattr(mod, func_name))
    returns = [
        l.strip() for l in src.splitlines()
        if l.strip().startswith("return ") and "Tuple" not in l
    ]
    assert returns, "no return statements found — test is looking at the wrong thing"
    for r in returns:
        # count top-level commas in the returned tuple
        inner = r[len("return "):].strip().strip("()")
        depth, commas = 0, 0
        for ch in inner:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                commas += 1
        assert commas == 4, f"{r!r} returns {commas + 1} values, expected 5"
