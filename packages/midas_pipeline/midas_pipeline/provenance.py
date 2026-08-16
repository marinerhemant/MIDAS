"""Per-layer / per-run provenance ledger ``midas_state.h5``.

One ``midas_state.h5`` per run directory (FF: ``LayerNr_N/``, PF: the
top-level result_dir). Each h5 file has a ``stages/<name>`` group per
stage with status, timestamps, file hashes, and metrics.

The pipeline writes after each successful stage and reads at startup to
pick up where it left off when ``resume="auto"``.

Lifted from ``midas_ff_pipeline.provenance`` with no semantic changes
(the schema is shared).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py


PROVENANCE_FILENAME = "midas_state.h5"


# ----- Hash helpers ---------------------------------------------------


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file into a hex digest. Empty/missing → 'missing'."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: Iterable[str | Path]) -> Dict[str, str]:
    return {str(Path(p)): file_sha256(p) for p in paths}


# ----- Provenance store -----------------------------------------------


class ProvenanceStore:
    """Run-scoped ledger of completed stages."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / PROVENANCE_FILENAME

    def record(self, stage_name: str, *,
               status: str = "complete",
               started_at: Optional[float] = None,
               finished_at: Optional[float] = None,
               duration_s: Optional[float] = None,
               inputs: Optional[Dict[str, str]] = None,
               outputs: Optional[Dict[str, str]] = None,
               metrics: Optional[Dict[str, Any]] = None) -> None:
        if started_at is None:
            started_at = time.time()
        if finished_at is None:
            finished_at = started_at
        if duration_s is None:
            duration_s = finished_at - started_at
        with h5py.File(self.path, "a") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name in f:
                del f[grp_name]
            grp = f.create_group(grp_name)
            grp.attrs["status"] = status
            grp.attrs["started_at"] = float(started_at)
            grp.attrs["finished_at"] = float(finished_at)
            grp.attrs["duration_s"] = float(duration_s)
            grp.create_dataset("inputs",
                               data=json.dumps(inputs or {}, default=_json_default))
            grp.create_dataset("outputs",
                               data=json.dumps(outputs or {}, default=_json_default))
            grp.create_dataset("metrics",
                               data=json.dumps(metrics or {}, default=_json_default))

    def read(self, stage_name: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with h5py.File(self.path, "r") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name not in f:
                return None
            grp = f[grp_name]
            return {
                "status": _decode(grp.attrs.get("status")),
                "started_at": float(grp.attrs.get("started_at", 0.0)),
                "finished_at": float(grp.attrs.get("finished_at", 0.0)),
                "duration_s": float(grp.attrs.get("duration_s", 0.0)),
                "inputs": _safe_loads(grp["inputs"][()]),
                "outputs": _safe_loads(grp["outputs"][()]),
                "metrics": _safe_loads(grp["metrics"][()]),
            }

    def all_stages(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        with h5py.File(self.path, "r") as f:
            if "stages" not in f:
                return {}
            for name in f["stages"]:
                grp = f[f"stages/{name}"]
                out[name] = {
                    "status": _decode(grp.attrs.get("status")),
                    "started_at": float(grp.attrs.get("started_at", 0.0)),
                    "finished_at": float(grp.attrs.get("finished_at", 0.0)),
                    "duration_s": float(grp.attrs.get("duration_s", 0.0)),
                    "inputs": _safe_loads(grp["inputs"][()]),
                    "outputs": _safe_loads(grp["outputs"][()]),
                    "metrics": _safe_loads(grp["metrics"][()]),
                }
        return out

    def is_complete(self, stage_name: str,
                    *, expected_outputs: Optional[List[str | Path]] = None) -> bool:
        rec = self.read(stage_name)
        if rec is None or rec["status"] != "complete":
            return False
        recorded = rec.get("outputs") or {}
        if not expected_outputs:
            for p, h in recorded.items():
                if h == "missing":
                    continue
                if file_sha256(p) != h:
                    return False
            return True
        for p in expected_outputs:
            p_str = str(Path(p))
            if file_sha256(p) != recorded.get(p_str):
                return False
        return True

    def invalidate(self, stage_name: str) -> None:
        if not self.path.exists():
            return
        with h5py.File(self.path, "a") as f:
            grp_name = f"stages/{stage_name}"
            if grp_name in f:
                del f[grp_name]


# ----- helpers -------------------------------------------------------


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _safe_loads(blob: Any) -> Any:
    if isinstance(blob, bytes):
        blob = blob.decode()
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {}


def _decode(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    return v


# ---------------------------------------------------------------------------
# Human-readable progress file
# ---------------------------------------------------------------------------

PROGRESS_FILENAME = "progress.txt"


def write_progress(run_dir: str | Path, *,
                   layer_nr: int,
                   scan_mode: str,
                   stage_names: List[str],
                   stages: Dict[str, Dict[str, Any]],
                   pid: Optional[int] = None) -> None:
    """Refresh ``progress.txt`` -- what a user actually cats over ssh.

    ``midas_state.h5`` already holds this, but it needs h5py to read, which is
    friction when the question is just "is it still going, and where is it?".
    A long FF run spends most of its wall time inside ONE stage (peakfit is
    ~83% of a beta run), so knowing which stage is live and how long it has
    been live is the difference between waiting and debugging.

    Written atomically (tmp + replace) so a concurrent ``cat`` never sees a
    half-written file.
    """
    now = time.time()
    lines = [
        f"MIDAS pipeline progress -- LayerNr_{layer_nr}",
        f"  scan mode   {scan_mode}",
        f"  stages      {len(stage_names)}",
        f"  pid         {pid if pid is not None else os.getpid()}",
        f"  updated     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        "",
    ]
    done = sum(1 for n in stage_names
               if (stages.get(n) or {}).get("status") == "complete")
    running = [n for n in stage_names
               if (stages.get(n) or {}).get("status") == "running"]
    failed = [n for n in stage_names
              if (stages.get(n) or {}).get("status") == "failed"]
    head = f"  {done}/{len(stage_names)} stages complete"
    if running:
        rec = stages.get(running[0]) or {}
        el = now - float(rec.get("started_at", now))
        head += f"   |   RUNNING: {running[0]} ({_hms(el)})"
    # A failure has to be in the header, not only in the per-stage list. This
    # file exists to be read after a run stops, and "3/12 stages complete" with
    # nothing else is indistinguishable from a run that is still going.
    if failed:
        head += f"   |   *** FAILED: {', '.join(failed)} ***"
    lines += [head, ""]

    for i, name in enumerate(stage_names, 1):
        rec = stages.get(name) or {}
        st = rec.get("status") or "pending"
        if st == "running":
            el = now - float(rec.get("started_at", now))
            lines.append(f"  [{i:2d}/{len(stage_names)}] {name:<22} RUNNING   "
                         f"{_hms(el)}  <-- in progress")
        elif st == "complete":
            lines.append(f"  [{i:2d}/{len(stage_names)}] {name:<22} complete  "
                         f"{_hms(float(rec.get('duration_s', 0.0)))}")
        elif st == "skipped":
            lines.append(f"  [{i:2d}/{len(stage_names)}] {name:<22} skipped")
        elif st == "failed":
            # Without its own branch this fell through to "pending", so a stage
            # that died read as one that had not started yet -- in the file a
            # user cats precisely to find out what went wrong.
            el = now - float(rec.get("started_at", now))
            lines.append(f"  [{i:2d}/{len(stage_names)}] {name:<22} FAILED    "
                         f"{_hms(el)}  <-- see the log")
        else:
            lines.append(f"  [{i:2d}/{len(stage_names)}] {name:<22} pending")

    outs = [(n, (stages.get(n) or {}).get("outputs") or {}) for n in stage_names]
    outs = [(n, o) for n, o in outs if o]
    if outs:
        lines += ["", "  outputs so far:"]
        for n, o in outs:
            for k, v in list(o.items())[:4]:
                lines.append(f"    {n}: {k} = {v}")

    p = Path(run_dir) / PROGRESS_FILENAME
    tmp = p.with_suffix(".txt.tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(p)
    except OSError:                      # progress reporting must never fail a run
        pass


def _hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:6.1f}s"
    if sec < 3600:
        return f"{sec/60:6.1f}m"
    return f"{sec/3600:6.2f}h"
