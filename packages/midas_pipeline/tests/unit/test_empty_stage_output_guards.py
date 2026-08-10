"""An empty artefact must be caught where it is produced, not three stages on.

Reported as github.com/marinerhemant/MIDAS issues/68. A c-omp FF run indexed
nothing, refinement wrote a 0-byte OrientPosFit.bin, and the run died in
midas_process_grains:

    File ".../midas_process_grains/io/binary.py", line 248, in read_orient_pos_fit
        arr = np.memmap(p, dtype=np.float64, mode="r")
    ValueError: cannot mmap an empty file

— a traceback in a package with no visibility of the real fault, which was
upstream in indexing. Three guards, one per stage:

  * indexing  RAISES when the indexer exits 0 having written no recognisable
              seed file (a broken contract), and only WARNS when the file
              exists and honestly reports zero seeds (a real, if disappointing,
              scientific outcome). The reporter drew exactly this distinction.
  * refinement WARNS when its own OrientPosFit.bin is absent or 0 bytes.
  * process_grains skips on a 0-byte OrientPosFit.bin rather than handing an
    empty file to np.memmap.
"""

from __future__ import annotations

import inspect


def test_indexing_raises_when_no_seed_file_was_written():
    from midas_pipeline.stages import indexing
    src = inspect.getsource(indexing._run_ff)          # noqa: SLF001
    # the branch itself, not merely "the function contains a raise" -- _run_ff
    # already raises elsewhere for unrelated reasons.
    branch = src[src.index("if counted_from is None:"):
                 src.index("if n_indexed == 0:")]
    assert "raise RuntimeError(" in branch
    assert "no recognisable" in branch


def test_indexing_only_warns_when_the_count_is_honestly_zero():
    """0 seeds is a result. A missing file is a bug. Do not conflate them."""
    from midas_pipeline.stages import indexing
    src = inspect.getsource(indexing._run_ff)          # noqa: SLF001
    tail = src[src.index("if n_indexed == 0:"):]
    assert "LOG.warning" in tail
    assert "raise" not in tail.split("else:")[0], (
        "a genuine zero-seed result must not abort the run"
    )


def test_refinement_reports_an_empty_orientposfit():
    from midas_pipeline.stages import refinement
    src = inspect.getsource(refinement)
    assert "if n_grains_refined == 0:" in src
    branch = src[src.index("if n_grains_refined == 0:"):]
    assert "LOG.warning" in branch
    assert "NOTHING" in branch, "the message must say the refiner refined nothing"
    assert "empty (0 bytes)" in branch, "absent and empty must read differently"


def test_process_grains_checks_size_not_just_existence():
    from midas_pipeline.stages import process_grains
    src = inspect.getsource(process_grains)
    assert "st_size == 0" in src, (
        "a 0-byte OrientPosFit.bin passes .exists() and reaches np.memmap"
    )
    # and it must bail rather than proceed
    tail = src[src.index("st_size == 0"):]
    assert "stub_run" in tail.split("pg_paramstest")[0]


def test_the_guards_name_the_upstream_stage():
    """The whole point is that the message points at the real fault."""
    from midas_pipeline.stages import process_grains, refinement
    pg = inspect.getsource(process_grains)
    rf = inspect.getsource(refinement)
    assert "upstream" in pg
    assert "indexing" in pg
    assert "indexing stage produced seeds" in rf
