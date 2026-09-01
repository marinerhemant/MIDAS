"""``midas_state.h5`` is written by the main thread and read by a heartbeat.

``Pipeline._run_layer`` records each stage into the ledger as it completes,
while ``StageProgress``'s heartbeat thread calls ``_refresh_progress`` — and so
``ProvenanceStore.all_stages()`` — every couple of seconds to rewrite
``progress.txt``. Those are a read-write open and a read-only open of the same
file from the same process, which HDF5 refuses.

Unguarded, a stage that finished inside the heartbeat's window killed the run:

    OSError: Unable to synchronously open file (file is already open for
    read-only)

Seen twice on 1-ID FF layers (2026-09-01), both at the end of a ~22 min
peakfit — so the cost of the race is hours, not seconds. Note
``HDF5_USE_FILE_LOCKING=FALSE`` does not help, because this is HDF5's own
in-process registry rather than an OS lock; reaching for the usual NFS
workaround is a dead end here.
"""
from __future__ import annotations

import threading

import h5py
import pytest

from midas_pipeline.provenance import ProvenanceStore


def test_hdf5_refuses_rw_while_the_same_process_holds_it_read_only(tmp_path):
    """The hazard is real — without this, the test below proves nothing.

    Deterministic and thread-free: it is the plain HDF5 rule that the lock in
    ``provenance`` exists to work around.
    """
    p = tmp_path / "ledger.h5"
    with h5py.File(p, "w"):
        pass
    with h5py.File(p, "r"):
        with pytest.raises(OSError):
            h5py.File(p, "a")


def test_record_survives_a_concurrent_heartbeat_reader(tmp_path):
    """Recording stages must not fail while the heartbeat reads the ledger."""
    store = ProvenanceStore(tmp_path)
    store.record("seed")                      # create the file

    errors: list[BaseException] = []
    stop = threading.Event()

    def heartbeat() -> None:
        # What _refresh_progress does, as fast as it will go.
        while not stop.is_set():
            try:
                store.all_stages()
            except BaseException as exc:      # noqa: BLE001 — recording it is the point
                errors.append(exc)
                return

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        for i in range(150):
            store.record(f"stage_{i}", status="complete")
    except BaseException as exc:              # noqa: BLE001
        errors.append(exc)
    finally:
        stop.set()
        t.join(timeout=10)

    assert not errors, f"ledger access raced: {errors[0]!r}"
    assert len(store.all_stages()) == 151     # seed + 150


def test_invalidate_is_also_guarded(tmp_path):
    """``invalidate`` opens read-write too, so it races the same way."""
    store = ProvenanceStore(tmp_path)
    for i in range(20):
        store.record(f"stage_{i}")

    errors: list[BaseException] = []
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            try:
                store.all_stages()
            except BaseException as exc:      # noqa: BLE001
                errors.append(exc)
                return

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        for i in range(20):
            store.invalidate(f"stage_{i}")
    except BaseException as exc:              # noqa: BLE001
        errors.append(exc)
    finally:
        stop.set()
        t.join(timeout=10)

    assert not errors, f"invalidate raced: {errors[0]!r}"
    assert store.all_stages() == {}
