"""progress.txt must not freeze, and the throttle must not lose counts.

Both failures below were seen live on a PF indexing stage (chutoro L7): the
file read ``[1/361 voxels, 0%]`` with ``RUNNING: indexing (0.3s)`` and stayed
that way for nine minutes while the c-omp indexer burned 37 cores.

The cause is that the c-omp indexer counts voxels on ENTRY: with 64 OpenMP
threads, entries 1..64 land in the same millisecond. ``update()`` records each
one but only *calls* ``on_change`` when ``min_interval_s`` has passed -- and it
DROPS the suppressed ones rather than deferring them, so the 63 that followed
the first never reached the file.
"""

from __future__ import annotations

import threading
import time

from midas_pipeline.progress import StageProgress


def _sink(**kw):
    """A StageProgress whose every write is recorded, snapshot and all."""
    writes: list = []
    ref: list = []

    def on_change():
        if ref:
            writes.append(ref[0].snapshot())

    sp = StageProgress(on_change=on_change, **kw)
    ref.append(sp)
    return sp, writes


def test_throttle_drops_updates_and_the_heartbeat_recovers_them():
    sp, writes = _sink(min_interval_s=5.0, heartbeat_s=0.2)
    try:
        sp.reset("indexing")
        for i in range(1, 65):          # 64 threads entering at once
            sp.update(i, 361, "voxels")

        # Only the first got through: this is the bug, reproduced.
        assert [w["done"] for w in writes] == [1]
        assert sp.snapshot()["done"] == 64, "the sink itself knows the truth"

        # The heartbeat writes the real count without any further update().
        deadline = time.time() + 3.0
        while time.time() < deadline and writes[-1]["done"] != 64:
            time.sleep(0.05)
        assert writes[-1]["done"] == 64, (
            "progress.txt would still claim 1/361 while 64 voxels are in flight"
        )
    finally:
        sp.close()


def test_heartbeat_fires_with_no_updates_at_all():
    # `elapsed` is computed when the file is written, so a stage with no
    # reporter needs the file rewritten just to keep its clock honest.
    sp, writes = _sink(heartbeat_s=0.2)
    try:
        sp.reset("find_grains")
        deadline = time.time() + 3.0
        while time.time() < deadline and len(writes) < 2:
            time.sleep(0.05)
        assert len(writes) >= 2, "no heartbeat: the elapsed clock would freeze"
    finally:
        sp.close()


def test_close_stops_the_thread():
    # One sink is built per LAYER; leaking a thread per layer would mean
    # finished layers rewriting their own progress.txt for the rest of the run.
    before = threading.active_count()
    sp, _ = _sink(heartbeat_s=0.1)
    sp.close()
    deadline = time.time() + 3.0
    while time.time() < deadline and threading.active_count() > before:
        time.sleep(0.05)
    assert threading.active_count() <= before


def test_heartbeat_can_be_disabled():
    sp, writes = _sink(heartbeat_s=0.0)
    try:
        sp.reset("peakfit")
        time.sleep(0.4)
        assert writes == []
    finally:
        sp.close()


def test_updates_still_write_immediately_when_not_throttled():
    sp, writes = _sink(min_interval_s=0.0, heartbeat_s=0.0)
    try:
        sp.reset("peakfit")
        sp.update(3, 19, "scans")
        sp.update(4, 19, "scans")
        assert [w["done"] for w in writes] == [3, 4]
    finally:
        sp.close()


def test_final_tick_always_writes():
    # Pre-existing guarantee: the file must not freeze one report short of 100%.
    sp, writes = _sink(min_interval_s=999.0, heartbeat_s=0.0)
    try:
        sp.reset("peakfit")
        sp.update(1, 19, "scans")
        sp.update(19, 19, "scans")
        assert writes[-1]["done"] == 19
    finally:
        sp.close()
