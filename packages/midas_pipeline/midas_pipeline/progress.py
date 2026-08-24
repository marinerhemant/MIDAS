"""Intra-stage progress: how far through the *current* stage a run is.

``progress.txt`` already says which stage is live and for how long. That is
enough to tell a finished stage from a running one, but not a slow run from a
hung one -- and one stage dominates a long FF run (peakfit was 88.5 % of a
2652-grain gamma reconstruction, 62 minutes of a 70-minute run) so "peakfit,
running, 41m" is most of what the file ever says.

Sources, one per long stage:

* **peakfit** runs in-process, so it pushes counts through a callback
  (``midas_peakfit.orchestrator.run(progress_cb=...)``, every 10 frames).
* **indexing** and **refinement** shell out to the c-omp binaries, which now
  print ``  progress: N/M voxels|seeds, R u/s, elapsed Es`` on a pipe. The
  subprocess runner streams stdout and feeds matching lines back here.
* **peakfit (PF)** counts completed scans, and **find_grains** completed voxels.

Writes are throttled: the C reporters fire ~200 times per stage and peakfit
every 10 frames, which would otherwise mean thousands of rewrites of a file
nobody reads that often. The throttle *drops* the suppressed updates rather
than deferring them, so a separate heartbeat rewrites the file every
``heartbeat_s`` -- see :meth:`StageProgress._heartbeat` for the two ways that
bit.

Known coarseness (PF indexing): the C counts voxels on ENTRY, and its comment
calls the overstatement "at most numProcs voxels in flight (64 of ~36000),
immaterial for a progress bar". That holds for an FF-sized grid. A PF layer is
``n_scans**2`` voxels -- 361 for a 19-scan layer -- so 64 threads put **18 %**
of the grid in flight at once and the bar can only move ~6 times. Finer
reporting needs a counter inside the per-voxel loop, i.e. a C change.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, Dict, Optional

__all__ = ["StageProgress", "parse_progress_line"]

# Emitted by IndexerUnified.c and FitUnified.c. Both packages pin the format;
# if it changes there, this regex and their comments must change together.
_PROGRESS_RE = re.compile(
    r"progress:\s+(\d+)\s*/\s*(\d+)\s+(\w+)"
    r"(?:,\s*([\d.]+)\s*[\w/]*\s*/?s)?"
)


def parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one c-omp progress line, or return None if it is not one."""
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    done, total, unit = int(m.group(1)), int(m.group(2)), m.group(3)
    if total <= 0:
        return None
    rate = float(m.group(4)) if m.group(4) else None
    return {"done": done, "total": total, "unit": unit, "rate": rate}


class StageProgress:
    """Thread-safe holder for the live stage's sub-progress.

    ``on_change`` is invoked (already throttled) whenever the snapshot moves,
    and is expected to rewrite progress.txt. It must never raise.
    """

    def __init__(self, on_change: Optional[Callable[[], None]] = None,
                 *, min_interval_s: float = 2.0,
                 heartbeat_s: float = 20.0) -> None:
        self._lock = threading.Lock()
        self._on_change = on_change
        self._min_interval = float(min_interval_s)
        self._stage: Optional[str] = None
        self._snap: Optional[Dict[str, Any]] = None
        self._last_write = 0.0
        self._stop = threading.Event()
        self._hb: Optional[threading.Thread] = None
        if on_change is not None and heartbeat_s > 0:
            self._hb = threading.Thread(
                target=self._heartbeat, args=(float(heartbeat_s),),
                name="midas-progress-heartbeat", daemon=True,
            )
            self._hb.start()

    def _heartbeat(self, period: float) -> None:
        """Rewrite progress.txt on a timer, not only when the count moves.

        Two failures this fixes, both observed on a live PF indexing stage:

        * **The throttle DROPS updates instead of deferring them.** The c-omp
          indexer counts voxels on *entry*, so with 64 OpenMP threads entries
          1..64 land in the same millisecond: the first is written and the
          other 63 fall inside ``min_interval_s`` and never reach the file.
          progress.txt reported ``1/361 voxels`` while this object held 64.
        * **``elapsed`` is computed when the file is written**, so between
          updates the clock freezes. A stage that reports rarely -- or has no
          reporter at all -- looked like it had been running 0.0s for hours.

        A stage with no reporter now still gets a live clock, which is most of
        what "is it alive?" needs.
        """
        while not self._stop.wait(period):
            cb = self._on_change
            if cb is None:
                continue
            try:
                cb()
            except Exception:
                pass                   # reporting must never fail a run
            with self._lock:
                self._last_write = time.time()

    def close(self) -> None:
        """Stop the heartbeat. One StageProgress is built per LAYER, so
        without this each finished layer would keep a thread alive rewriting
        its own progress.txt for the rest of the run."""
        self._stop.set()

    def reset(self, stage: Optional[str]) -> None:
        """Called at each stage boundary; clears any previous stage's counts."""
        with self._lock:
            self._stage = stage
            self._snap = None
            self._last_write = 0.0

    def update(self, done: int, total: int, unit: str = "",
               rate: Optional[float] = None) -> None:
        """Record progress. Safe to call from a worker thread."""
        now = time.time()
        fire = False
        with self._lock:
            if total and total > 0:
                self._snap = {
                    "stage": self._stage, "done": int(done), "total": int(total),
                    "unit": unit, "rate": rate, "at": now,
                }
                # Always fire on the final tick so the file does not freeze
                # one report short of 100%.
                if (now - self._last_write >= self._min_interval
                        or int(done) >= int(total)):
                    self._last_write = now
                    fire = True
        if fire and self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass                       # reporting must not fail a run

    def feed_line(self, line: str) -> None:
        """Consume one line of a subprocess's stdout."""
        p = parse_progress_line(line)
        if p:
            self.update(p["done"], p["total"], p["unit"], p["rate"])

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._snap) if self._snap else None


def format_sub(snap: Optional[Dict[str, Any]]) -> str:
    """Render a snapshot as ``[1200/3600 frames 33%, 5.2/s, eta 7.7m]``."""
    if not snap or not snap.get("total"):
        return ""
    done, total = snap["done"], snap["total"]
    pct = 100.0 * done / total
    bits = [f"{done}/{total} {snap.get('unit') or ''}".strip(), f"{pct:.0f}%"]
    rate = snap.get("rate")
    if rate and rate > 0:
        bits.append(f"{rate:.1f}/s")
        remain = (total - done) / rate
        if remain > 0:
            bits.append("eta " + (f"{remain:.0f}s" if remain < 90 else
                                  f"{remain/60:.1f}m" if remain < 5400 else
                                  f"{remain/3600:.1f}h"))
    return "  [" + ", ".join(bits) + "]"
