"""Reading text .mic files."""
from __future__ import annotations

import numpy as np
import pytest

from midas_plotting.mic import read_mic

HEADER = "\n".join(["# header line %d" % i for i in range(4)])


def _write(tmp_path, rows):
    p = tmp_path / "t.mic"
    with open(p, "w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.9g}" for v in r) + "\n")
    return p


def _row(x, y, e1, e2, e3, conf, runtime=1234.5):
    # 11 columns: 3=x 4=y 7:10=euler 10=conf, 2=RunTime
    return [0, 1, runtime, x, y, 5, 6, e1, e2, e3, conf]


def test_reads_columns_and_derives_pitch(tmp_path):
    rows = [_row(x, y, 0.1, 0.2, 0.3, 0.75)
            for x in (0.0, 5.0, 10.0) for y in (0.0, 5.0)]
    m = read_mic(_write(tmp_path, rows))
    assert len(m) == 6
    np.testing.assert_allclose(sorted(set(m.x)), [0.0, 5.0, 10.0])
    np.testing.assert_allclose(m.confidence, 0.75)
    assert m.pitch == pytest.approx(10.0)      # 2 * median spacing


def test_pitch_is_derived_not_read_from_a_column(tmp_path):
    """Multi-resolution output does not update the grid-size column, so pitch
    has to come from the positions."""
    rows = [_row(x, 0.0, 0.0, 0.0, 0.0, 0.5) for x in (0.0, 1.25, 2.5)]
    m = read_mic(_write(tmp_path, rows))
    assert m.pitch == pytest.approx(2.5)


def test_mask_and_summary(tmp_path):
    rows = [_row(0.0, 0.0, 0, 0, 0, c) for c in (0.05, 0.2, 0.45, 0.95)]
    m = read_mic(_write(tmp_path, rows))
    assert m.mask(0.3).sum() == 2
    s = m.summary()
    assert "maxC 0.9500" in s and ">=0.3: 2" in s


def test_rejects_too_few_columns(tmp_path):
    p = tmp_path / "bad.mic"
    p.write_text(HEADER + "\n1 2 3\n")
    with pytest.raises(ValueError, match="columns"):
        read_mic(p)
