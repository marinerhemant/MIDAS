"""``Pipeline.dump()`` must write the ring tables, like fit_setup does.

Found by an end-to-end validation run, not by a unit test.

``fit_setup(write=True)`` wrote InputAll.csv, InputAllExtraInfoFittingAll.csv,
**IDRings.csv, IDsHash.csv**, SpotsToIndex.csv and paramstest.txt.
``Pipeline.dump()`` — the path ``midas-pipeline run --scan-mode ff`` takes —
wrote everything on that list EXCEPT the two ring tables.

IDsHash.csv's fourth column is the reference d-spacing d₀ per ring, and it is
the only source midas-process-grains has for the Kenesei per-spot strain gauge
``ε = (d_obs − d₀)/d₀``. With the file absent the consumer substituted zeros,
so on both datasetA and shade_LSHR every grain's strain pegged at the ±0.01
bound (1.000e+04 µε) and RMSErrorStrain came out ~1e36 — in runs whose grain
count, positions, orientations, sizes and completeness were all correct and
inside their acceptance bands. Nothing warned.

Two guards, because either alone leaves the hole open:
  * one writer (``write_ring_tables``) used by both paths, so they cannot
    diverge again;
  * the consumer refuses a missing IDsHash.csv instead of fabricating d₀ = 0.
"""

from __future__ import annotations

import inspect
from pathlib import Path


def test_one_writer_for_both_paths():
    from midas_transforms.fit_setup import core
    assert hasattr(core, "write_ring_tables")
    src = inspect.getsource(core.fit_setup)
    assert "write_ring_tables(" in src, "fit_setup must use the shared writer"


def test_dump_writes_the_ring_tables():
    from midas_transforms import pipeline
    src = inspect.getsource(pipeline)
    assert "write_ring_tables(" in src, (
        "Pipeline.dump() must emit IDRings.csv + IDsHash.csv; the FF pipeline "
        "path goes through dump(), not fit_setup(write=True)"
    )


def test_the_result_carries_what_dump_needs():
    from midas_transforms.fit_setup.core import FitSetupResult
    for f in ("ring_numbers", "per_ring_count", "ds_per_ring", "id_rings_rows"):
        assert f in FitSetupResult.__dataclass_fields__, (
            f"FitSetupResult.{f} is required for dump() to write IDsHash.csv"
        )


def test_d_spacing_is_computed_outside_the_write_branch():
    """It used to live inside ``if write:``, so dump() could never see it."""
    from midas_transforms.fit_setup import core
    src = inspect.getsource(core.fit_setup)
    i_ds = src.index("ds_per_ring = [")
    i_write = src.index("if write:")
    assert i_ds < i_write, (
        "ds_per_ring must be computed unconditionally, before the write branch"
    )


def test_write_ring_tables_emits_both_files(tmp_path: Path):
    from midas_transforms.fit_setup.core import write_ring_tables
    write_ring_tables(tmp_path, [1, 2], [10, 20], [2.0784, 1.8000],
                      [(1, 5, 1), (2, 6, 2)])
    idr = (tmp_path / "IDRings.csv").read_text().splitlines()
    idh = (tmp_path / "IDsHash.csv").read_text().splitlines()
    assert idr[0].startswith("RingNumber")
    assert len(idr) == 3
    assert len(idh) == 2
    # ring, start, end, d — d must be the real spacing, never 0
    first = idh[0].split()
    assert int(first[0]) == 1
    assert float(first[3]) > 0.0


def _code_only(src: str) -> str:
    """Strip comments — the explanatory notes quote the very expression the
    assertions forbid, and a naive substring check matches the comment."""
    out = []
    for line in src.splitlines():
        stripped = line.split("#", 1)[0]
        out.append(stripped)
    return "\n".join(out)


def test_d_zero_is_never_silently_substituted():
    from midas_process_grains.compute import c_parity_emit, c_parity_run
    emit = _code_only(inspect.getsource(c_parity_emit.gather_per_grain_spot_data))
    assert "np.zeros_like(y)" not in emit, (
        "a fabricated d0 of 0 turns Kenesei strain into (d_obs-0)/0"
    )
    run = inspect.getsource(c_parity_run.run_c_parity_pipeline_from_disk)
    assert "raise FileNotFoundError" in run
    assert "IDsHash" in run
