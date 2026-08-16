"""Pre-run input validation: fail loudly, before any stage does work.

Why this exists
---------------
``zip_convert`` shells out to ``midas_zipper.ff_zip``. When that subprocess
cannot find its inputs it exits non-zero, and every downstream FF stage then
skipped with "no zarr/zip available at None" -- so the run printed
``done. layers processed: 1`` and **exited 0** having produced nothing but
``midas_log/`` and ``midas_state.h5``. Under ``nohup`` that is
indistinguishable from success, and two users lost days to it: one mistyped
``--params Paramers_...txt``, and the failure looked like a parameter problem
rather than a missing file.

The PF path already learned this lesson (see the ``P0-2`` note in
``stages/zip_convert.py``); FF never did. This module closes it from the other
end: check the inputs *before* the first stage runs, and report every problem
found in one message rather than the first one only.

Deliberately narrow. Only checks that cannot produce a false positive --
existence and readability of files the run genuinely requires. It does not
validate geometry, rings, or physics; a run that passes preflight can still be
scientifically wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ._logging import LOG

__all__ = ["PreflightError", "check_inputs", "preflight"]

# Keys the zipper needs to resolve a raw filename. Mirrors
# midas_zipper.ff_zip:
#   fNr    = StartFileNrFirstLayer + (layer-1) * ScanStep|NrFilesPerSweep
#   dataFN = RawFolder / f"{FileStem}_{fNr:0{Padding}d}{Ext}"
_RAW_KEYS = ("RawFolder", "FileStem", "Padding", "Ext")


class PreflightError(RuntimeError):
    """Raised when a required input is missing, before any stage runs."""


def _resolve_raw_path(kv: dict, layer_nr: int) -> Optional[Path]:
    """Reproduce midas_zipper's raw-filename construction for one layer."""
    try:
        start = int(float(kv["StartFileNrFirstLayer"]))
        step = int(float(kv.get("ScanStep", kv.get("NrFilesPerSweep", "1")) or 1))
        pad = int(float(kv["Padding"]))
        f_nr = start + (layer_nr - 1) * step
        return Path(kv["RawFolder"]) / f"{kv['FileStem']}_{str(f_nr).zfill(pad)}{kv['Ext']}"
    except (KeyError, ValueError, TypeError):
        return None


def check_inputs(cfg, layers: Optional[List[int]] = None) -> List[str]:
    """Return a list of human-readable problems. Empty list == good to go."""
    problems: List[str] = []

    # ---- 1. the parameter file itself -----------------------------------
    pf = Path(cfg.params_file)
    if not pf.exists():
        problems.append(
            f"parameter file not found: {pf}\n"
            f"    (check the spelling of --params; the run would otherwise "
            f"skip every stage and still exit 0)")
        return problems                      # nothing further is parseable
    if not pf.is_file():
        problems.append(f"--params is not a file: {pf}")
        return problems
    try:
        if pf.stat().st_size == 0:
            problems.append(f"parameter file is empty: {pf}")
            return problems
        # read BYTES: read_text() applies universal-newline translation, so
        # "\r\n" would never survive to be detected.
        raw_bytes = pf.read_bytes()
    except OSError as e:
        problems.append(f"parameter file unreadable: {pf} ({e})")
        return problems

    # CRLF line endings survive the C parsers unevenly and are a real
    # cross-platform footgun when a file is edited on Windows.
    if b"\r\n" in raw_bytes:
        problems.append(
            f"parameter file has CRLF (Windows) line endings: {pf}\n"
            f"    fix with:  sed -i 's/\\r$//' {pf}")

    # ---- 2. a pre-built zarr/zip means raw data is not needed -----------
    if getattr(cfg, "zarr_path", None):
        z = Path(cfg.zarr_path)
        if not z.exists():
            problems.append(f"--zarr given but not found: {z}")
        return problems

    layers = layers or [1]
    layer_dirs = [Path(cfg.result_dir) / f"LayerNr_{n}" for n in layers]
    if all(d.exists() and any(d.glob("*.MIDAS.zip")) for d in layer_dirs):
        return problems                      # resume: zips already built

    if not getattr(cfg, "convert_files", True):
        return problems                      # user opted out of conversion

    # ---- 3. raw data + dark ---------------------------------------------
    from ._pf_scans import parse_params_kv
    try:
        kv = parse_params_kv(pf)
    except Exception as e:                   # pragma: no cover - defensive
        problems.append(f"parameter file could not be parsed: {pf} ({e})")
        return problems

    missing_keys = [k for k in _RAW_KEYS if k not in kv]
    if "StartFileNrFirstLayer" not in kv and "StartNr" not in kv:
        missing_keys.append("StartFileNrFirstLayer")
    if missing_keys:
        problems.append(
            "parameter file is missing keys needed to locate the raw data: "
            + ", ".join(sorted(set(missing_keys))))
        return problems
    kv.setdefault("StartFileNrFirstLayer", kv.get("StartNr", "1"))

    raw_dir = Path(kv["RawFolder"])
    if not raw_dir.is_dir():
        problems.append(f"RawFolder is not a directory: {raw_dir}")
    else:
        for n in layers:
            raw = _resolve_raw_path(kv, n)
            if raw is None:
                problems.append(
                    "could not build the raw filename from RawFolder / FileStem "
                    "/ Padding / StartFileNrFirstLayer / Ext -- check their values")
                break
            if not raw.exists():
                near = sorted(p.name for p in raw_dir.glob(f"{kv['FileStem']}*"))[:4]
                hint = ("\n    files present with that stem: " + ", ".join(near)
                        if near else
                        f"\n    no file in {raw_dir} starts with "
                        f"{kv['FileStem']!r} -- check FileStem and RawFolder")
                problems.append(f"raw data file for layer {n} not found: {raw}{hint}")

    dark = kv.get("Dark", "").strip()
    if dark and not Path(dark).exists():
        problems.append(f"Dark file not found: {dark}")

    return problems


def preflight(cfg, layers: Optional[List[int]] = None) -> None:
    """Validate inputs; raise :class:`PreflightError` listing every problem."""
    problems = check_inputs(cfg, layers)
    if not problems:
        LOG.debug("preflight: inputs OK")
        return
    body = "\n".join(f"  [{i}] {p}" for i, p in enumerate(problems, 1))
    raise PreflightError(
        f"preflight failed -- {len(problems)} problem(s) with the run inputs; "
        f"nothing was executed:\n{body}")
