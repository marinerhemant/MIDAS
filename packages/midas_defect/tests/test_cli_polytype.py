"""CLI integration test for midas-defect-polytype.

Monkeypatches the voxel loader and the seed indexer so the orchestration is
exercised end-to-end on the small g1592 fixture without the raw-voxel format or a
slow seed search. Asserts the JSON summary carries the headline ladder/doublet
results and no parent/twin identity.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midas_defect.cli import polytype as cli

_FIX = Path(__file__).parent / "fixtures" / "demk_g1592_9r.npz"


@pytest.mark.skipif(not _FIX.exists(), reason="g1592 fixture absent")
def test_cli_polytype_end_to_end(tmp_path, monkeypatch):
    d = np.load(_FIX, allow_pickle=True)
    q = np.asarray(d["q"], float)
    OM = np.asarray(d["OM"], float)

    cloud = SimpleNamespace(
        qx=q[:, 0], qy=q[:, 1], qz=q[:, 2],
        intensity=np.asarray(d["intensity"], float),
        omega_deg=np.asarray(d["omega"], float),
        n_voxels=lambda: q.shape[0],
    )
    monkeypatch.setattr(cli, "load_voxel_npz", lambda p: cloud)
    # seed indexer returns the package-convention U (= raw OM transposed)
    monkeypatch.setattr(cli, "find_seed_orientation",
                        lambda *a, **k: SimpleNamespace(U=OM.T, score=20, a=3.6356, c=3.6356))

    out = tmp_path / "g1592_polytype.json"
    rc = cli.main(["--voxels", "dummy.npz", "--out", str(out)])
    assert rc == 0
    summary = json.loads(out.read_text())

    assert summary["ladder"]["n_fundamentals"] == 4
    assert summary["ladder"]["n_satellites"] >= 6
    assert summary["doublets"]["G/3"]["verdict"] == "two-reflections"
    assert summary["doublets"]["G/3"]["is_twin_polarity"]
    assert "parent" not in json.dumps(summary["ladder"]).lower()
    assert "attribution_note" in summary
