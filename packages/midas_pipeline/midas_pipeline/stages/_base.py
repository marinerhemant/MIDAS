"""Common helpers for stage modules.

The ``StageContext`` is a mutable execution context handed to every
stage. It carries the immutable ``PipelineConfig`` plus the per-layer /
per-run path bookkeeping the stage needs.

Multi-detector FF runs populate ``detectors`` with >1 panel; single-det
runs (FF or PF) carry a one-element list. The ``is_multi_detector`` /
``detector_dir()`` / ``stage_dir()`` helpers let downstream stages stay
oblivious to which mode they're in.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .._logging import LOG
from ..config import PipelineConfig
from ..detector import DetectorConfig


@dataclass
class StageContext:
    """Mutable execution context handed to each stage."""

    config: PipelineConfig
    layer_nr: int
    layer_dir: Path
    log_dir: Path
    detectors: List[DetectorConfig] = field(default_factory=list)
    merged_paramstest: Optional[Path] = None
    # Intra-stage progress sink; set by the Pipeline. Stages may leave it None
    # (library callers construct a StageContext directly).
    progress: Optional[Any] = None

    @property
    def scan_mode(self) -> str:
        return self.config.scan.scan_mode

    @property
    def is_ff(self) -> bool:
        return self.config.is_ff

    @property
    def is_pf(self) -> bool:
        return self.config.is_pf

    @property
    def is_multi_detector(self) -> bool:
        return len(self.detectors) > 1

    def detector_dir(self, det: DetectorConfig) -> Path:
        """Per-detector working dir (only used for multi-det runs)."""
        return self.layer_dir / f"Det_{det.det_id}"

    def stage_dir(self, det: Optional[DetectorConfig] = None) -> Path:
        """Where a stage writes its outputs.

        Single-det runs write directly into ``layer_dir``. Multi-det
        per-detector stages write into ``layer_dir/Det_<id>/``.
        """
        if det is None or not self.is_multi_detector:
            return self.layer_dir
        return self.detector_dir(det)


def resolve_layer_dir(result_dir: Path, layer_nr: int) -> Path:
    """Return the working directory for a given layer.

    Default layout is ``<result_dir>/LayerNr_<n>/``. For back-compat with
    legacy pf-HEDM workdirs (produced by `FF_HEDM/workflows/pf_MIDAS.py`)
    where all per-layer files live directly under ``<result_dir>/``, return
    ``<result_dir>`` instead.

    Resolution order:
      1. If ``LayerNr_<n>`` exists AND contains any payload subdir with at
         least one file (Recons/, Output/, Results/, or a top-level CSV/bin),
         use it.
      2. Else if ``<result_dir>`` looks like a legacy flat layout
         (Recons/recon_grNr_*.tif OR Output/IndexBest_all.bin OR
         paramstest.txt at top level), use it.
      3. Else default to ``LayerNr_<n>`` (fresh-run default; will be created).
    """
    result_path = Path(result_dir)
    candidate = result_path / f"LayerNr_{layer_nr}"

    def has_payload(d: Path) -> bool:
        """True if d has real data, not just empty subdirs."""
        if not d.exists():
            return False
        markers = [
            d / "Recons",
            d / "Output",
            d / "Results",
        ]
        for m in markers:
            if m.exists() and any(m.iterdir()):
                return True
        # Top-level legacy files
        for f in ("paramstest.txt", "positions.csv", "Spots.bin"):
            if (d / f).exists():
                return True
        return False

    if has_payload(candidate):
        return candidate
    if has_payload(result_path):
        return result_path
    return candidate


def run_subprocess(cmd: Sequence[str], *,
                   cwd: str | Path,
                   stdout_path: str | Path,
                   stderr_path: str | Path,
                   env: Optional[dict] = None,
                   timeout: Optional[int] = None,
                   line_cb=None) -> int:
    """Run a subprocess and tee stdout/stderr to per-stage log files.

    ``line_cb`` (optional) receives each stdout line as it arrives, so a
    long-running c-omp binary can report intra-stage progress live. Without
    it stdout is redirected straight to the file (the original path), which
    stays the cheapest option for the many short subprocesses.
    """
    cmd_list = [str(c) for c in cmd]
    log_cmd = " ".join(shlex.quote(c) for c in cmd_list)
    LOG.debug("$ %s   (cwd=%s)", log_cmd, cwd)

    Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)

    if line_cb is None:
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            proc = subprocess.run(
                cmd_list, cwd=str(cwd),
                stdout=out, stderr=err,
                env={**os.environ, **(env or {})},
                timeout=timeout,
                check=False,
            )
    else:
        # Stream stdout so progress lines arrive while the stage runs, while
        # still writing the same log file byte-for-byte.
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            popen = subprocess.Popen(
                cmd_list, cwd=str(cwd),
                stdout=subprocess.PIPE, stderr=err,
                env={**os.environ, **(env or {})},
                text=True, bufsize=1,
            )
            try:
                for line in popen.stdout:            # type: ignore[union-attr]
                    out.write(line)
                    try:
                        line_cb(line)
                    except Exception:
                        pass                          # never fail a run
            finally:
                if popen.stdout is not None:
                    popen.stdout.close()
                popen.wait(timeout=timeout)
        proc = subprocess.CompletedProcess(cmd_list, popen.returncode)

    if proc.returncode != 0:
        LOG.error("subprocess failed (rc=%d): %s", proc.returncode, log_cmd)
        try:
            tail = Path(stderr_path).read_text().strip().splitlines()[-10:]
            LOG.error("stderr tail:\n  %s", "\n  ".join(tail))
        except Exception:
            pass
        raise subprocess.CalledProcessError(proc.returncode, cmd_list)
    return proc.returncode


def env_for_index_refine(config: PipelineConfig) -> dict:
    """Common env for indexer + refiner: dtype, device, group size."""
    return {
        "MIDAS_INDEX_DTYPE": config.dtype,
        "MIDAS_INDEX_DEVICE": config.device,
        "MIDAS_INDEX_GROUP_SIZE": str(config.indexer_group_size),
        "MIDAS_FIT_GRAIN_DTYPE": config.dtype,
        "MIDAS_FIT_GRAIN_DEVICE": config.device,
    }


def hash_inputs(paths: Iterable[Path]) -> dict[str, str]:
    from ..provenance import file_sha256
    return {str(p): file_sha256(p) for p in paths}


def run_checked_streamed(cmd: Sequence[str], *, cwd: str | Path,
                         out_fp, err_fp, line_cb=None) -> None:
    """``subprocess.run(check=True)`` that can stream stdout line by line.

    The c-omp indexer and refiner print intra-stage progress on stdout. With
    stdout redirected straight to a file the pipeline cannot see those lines
    until the process exits, which is exactly when they stop being useful.
    ``line_cb=None`` keeps the original redirect (cheaper, no pump thread).
    """
    cmd_list = [str(c) for c in cmd]
    if line_cb is None:
        subprocess.run(cmd_list, cwd=str(cwd), check=True,
                       stdout=out_fp, stderr=err_fp)
        return
    popen = subprocess.Popen(cmd_list, cwd=str(cwd), stdout=subprocess.PIPE,
                             stderr=err_fp, text=True, bufsize=1)
    try:
        for line in popen.stdout:                 # type: ignore[union-attr]
            out_fp.write(line)
            try:
                line_cb(line)
            except Exception:
                pass                              # reporting never fails a run
    finally:
        if popen.stdout is not None:
            popen.stdout.close()
        rc = popen.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd_list)
