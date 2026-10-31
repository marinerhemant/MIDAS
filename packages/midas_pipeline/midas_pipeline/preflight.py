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

    # ---- 2a. multi-detector: one pre-built zip per panel ----------------
    # ``--detectors`` carries a .MIDAS.zip per panel (zarr_path), so the raw
    # frames are not needed either; without this the raw-key check below
    # rejects a run whose inputs are complete.
    if getattr(cfg, "detectors_json", None):
        dj = Path(cfg.detectors_json)
        try:
            from .detector import DetectorConfig
            dets = DetectorConfig.load_many(dj)
        except Exception as e:
            problems.append(f"--detectors could not be read: {dj} ({e})")
            return problems
        if len(dets) > 1:
            for d in dets:
                if not d.zarr_path or not Path(d.zarr_path).exists():
                    problems.append(
                        f"detector {d.det_id}: zarr_path not found: "
                        f"{d.zarr_path or '(empty)'}")
            if not problems:
                problems.extend(_geometry_mismatches(dets))
                problems.extend(_zip_param_mismatches(
                    pf, dets, accept=bool(getattr(cfg, "accept_zip_params", False))))
            return problems

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
    try:
        raw_is_dir = raw_dir.is_dir()
    except OSError as e:                     # e.g. an ACL that denies this account: report, don't crash
        problems.append(f"RawFolder is not readable by this account: {raw_dir} ({e.strerror or e}); "
                        "if the raw frames are not needed (resuming past conversion), pass --no-convert")
        raw_is_dir = None
    if raw_is_dir is False:
        problems.append(f"RawFolder is not a directory: {raw_dir}")
    elif raw_is_dir:
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
    try:
        dark_ok = not dark or Path(dark).exists()
    except OSError as e:
        problems.append(f"Dark file is not readable by this account: {dark} ({e.strerror or e})")
        dark_ok = True
    if not dark_ok:
        problems.append(f"Dark file not found: {dark}")

    return problems


# Tolerances for "the zip and detectors.json describe the same panel".
# Looser than any real refinement step, tighter than any real difference.
_GEOM_TOL = {"lsd": 10.0, "bc": 0.05, "tilt": 0.01}   # um, px, deg


def _geometry_mismatches(dets) -> List[str]:
    """Panels whose zip and ``detectors.json`` disagree on Lsd / BC / tilts.

    The two feed different consumers: ``transforms`` turns each panel's spots
    into lab coordinates with the geometry stored in ITS ZIP, while the merged
    paramstest's ``DetParams`` rows (what the indexer and the coverage arcs
    read) come from ``detectors.json``. When they differ nothing errors and
    the spots are simply placed with one geometry and interpreted with
    another. A zip that cannot be read here is skipped, not an error.
    """
    try:
        import zarr
    except ImportError:                                  # pragma: no cover
        return []
    out: List[str] = []
    for d in dets:
        try:
            ap = zarr.open(str(d.zarr_path), mode="r")[
                "analysis/process/analysis_parameters"]

            def val(k, default=None):
                return float(ap[k][...].ravel()[0]) if k in ap else default

            got = {"Lsd": val("Lsd"), "y_bc": val("YCen"), "z_bc": val("ZCen"),
                   "tx": val("tx", 0.0), "ty": val("ty"), "tz": val("tz")}
        except Exception:
            continue
        want = {"Lsd": d.lsd, "y_bc": d.y_bc, "z_bc": d.z_bc,
                "tx": d.tx, "ty": d.ty, "tz": d.tz}
        tol = {"Lsd": _GEOM_TOL["lsd"], "y_bc": _GEOM_TOL["bc"],
               "z_bc": _GEOM_TOL["bc"], "tx": _GEOM_TOL["tilt"],
               "ty": _GEOM_TOL["tilt"], "tz": _GEOM_TOL["tilt"]}
        bad = [f"{k}: zip {got[k]:.6g} vs detectors.json {want[k]:.6g}"
               for k in want if got[k] is not None
               and abs(got[k] - want[k]) > tol[k]]
        if bad:
            out.append(
                f"detector {d.det_id}: the zip and detectors.json disagree on "
                f"the panel geometry ({'; '.join(bad)}). transforms uses the "
                f"zip, the indexer's DetParams use detectors.json; make them "
                f"match (fix the json, or rebuild the zip from the right "
                f"parameter file)")
    return out


def _zip_param_mismatches(params_file, dets, *, accept: bool = False) -> List[str]:
    """Scan-level keys the parameter file sets that a panel zip does not hold.

    With ``--detectors`` every panel zip is read as built and is not refreshed
    from the parameter file, so a Wedge or OmegaStart in the file used to be
    ignored with no message (every generated paramstest then read
    ``Wedge 0.000000``). Returned as problems, so the run stops before any
    stage works; ``accept`` (``--accept-zip-params``) logs them instead. Only
    ``midas_zipper.param_refresh.SCAN_LEVEL_KEYS`` stop the run: a panel's own
    Lsd / BC / tilts differ from the file by design, and ``PANEL_LEVEL_KEYS``
    (RingThresh, MaxRingRad) only warn. A zip that cannot be read
    is skipped.
    """
    try:
        from midas_zipper.param_refresh import (PANEL_LEVEL_KEYS,
                                                check_zip_vs_params,
                                                mismatch_message)
    except ImportError:                                  # pragma: no cover
        LOG.warning("preflight: midas_zipper.param_refresh unavailable; zip "
                    "parameters NOT compared with %s", params_file)
        return []
    found = {}
    per_panel = {}
    for d in dets:
        try:
            changes = check_zip_vs_params(d.zarr_path, params_file)
            soft = check_zip_vs_params(d.zarr_path, params_file, keys=PANEL_LEVEL_KEYS)
        except Exception as e:                           # noqa: BLE001 - unreadable zip
            LOG.debug("preflight: zip parameters of %s not compared: %s", d.zarr_path, e)
            continue
        if changes:
            found[str(d.zarr_path)] = changes
        if soft:
            per_panel[d.det_id] = sorted({c.source_key for c in soft})
    if per_panel:
        LOG.warning("preflight: %s differs between %s and the zips of detector(s) %s; "
                    "the zip values are used (expected when thresholds or ring radii "
                    "are set per panel)",
                    "/".join(sorted({k for v in per_panel.values() for k in v})),
                    Path(params_file).name, sorted(per_panel))
    if not found:
        return []
    msg = mismatch_message(
        found, params_file,
        accept_hint="To run on the zip values anyway: --accept-zip-params.")
    if accept:
        LOG.warning("preflight: --accept-zip-params: running on the zip values.\n%s", msg)
        return []
    return [msg]


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
