"""ESRF Bliss raw data -> the per-scan ``.MIDAS.zip`` files a pf-HEDM run reads.

``zip_convert`` calls :func:`build_zips_from_bliss` when ``--bliss-master`` is given. The frames are read from the
master's Lima files and written unchanged (apart from ``--preproc-thresh``); what this module adds is every check
that a Bliss file needs before its numbers can be trusted, because a wrong choice here mirrors or reverses the map
without any error:

1. **Scan count.** The master holds exactly as many scans as ``positions.csv`` has rows, all with the same frame
   count.
2. **dty.** Every scan has a recorded ``dty`` and the step between scans is the ``--scan-step`` the run was given.
   Whether ``positions.csv`` is ``+dty`` or ``-dty`` is a convention that is reported, not decided here.
3. **Omega.** Per-frame omega is ``sign * rot_center + offset``, wrapped into [-180, 180) (MIDAS takes no other range;
   a scan running THROUGH +-180 is refused). The sign is **never guessed**: it is given (``--bliss-omega-sign``) or
   pinned from zips of another layer of the same experiment (``--bliss-omega-ref-raw``), where exactly one sign must
   match frame by frame and the offset must be ~0. Snake scans (omega rising on one scan, falling on the next) need
   nothing special: ``omegaCenter`` carries each frame's own angle.

The record of what was decided goes to ``<layer_dir>/midas_log/bliss_ingest.json``.
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ._logging import LOG
from ._pf_scans import PFScanInfo, fan_out_scans

__all__ = ["BlissIngestError", "build_zips_from_bliss"]


class BlissIngestError(RuntimeError):
    """A gate failed; nothing was written."""


def _trailing_number(p: Path) -> int:
    m = re.search(r"(\d+)\.MIDAS\.zip$", p.name)
    if not m:
        raise BlissIngestError(f"cannot number reference zip {p}")
    return int(m.group(1))


def _reference_omegas(ref_raw: Path, n_scans: int) -> List[np.ndarray]:
    import zarr
    from midas_zipper.bliss import OMEGA_CENTER_KEY

    zips = sorted(Path(ref_raw).rglob("*.MIDAS.zip"), key=_trailing_number)
    if not zips:
        raise BlissIngestError(f"--bliss-omega-ref-raw {ref_raw}: no *.MIDAS.zip found")
    nums = [_trailing_number(z) for z in zips]
    if len(set(nums)) != len(nums):
        raise BlissIngestError(f"--bliss-omega-ref-raw {ref_raw}: duplicate scan numbers among the reference zips")
    if len(zips) < n_scans:
        raise BlissIngestError(f"--bliss-omega-ref-raw {ref_raw}: {len(zips)} reference zips for {n_scans} scans")
    return [np.asarray(zarr.open(str(z), "r")[OMEGA_CENTER_KEY][...]) for z in zips[:n_scans]]


@dataclass
class _Plan:
    sign: float
    offset: float
    pin: Optional[dict]


def _choose_omega(rots: Sequence[np.ndarray], *, sign: Optional[int], offset: float, ref_raw: Optional[str],
                  allow_offset: bool) -> _Plan:
    from midas_zipper.bliss import omega_map_from_reference

    if sign not in (None, 1, -1):
        raise BlissIngestError(f"--bliss-omega-sign must be +1 or -1, got {sign}")
    if sign is None and ref_raw is None:
        raise BlissIngestError(
            "the omega sign of a Bliss master is not guessed: pass --bliss-omega-sign +1/-1, or "
            "--bliss-omega-ref-raw DIR (per-scan .MIDAS.zip of another layer of the same experiment) to pin it")
    pin = None
    if ref_raw is not None:
        refs = _reference_omegas(Path(ref_raw), len(rots))
        try:
            pin = omega_map_from_reference(rots, refs, max_abs_offset=None if allow_offset else 0.05)
        except ValueError as e:
            raise BlissIngestError(f"omega pin against {ref_raw} failed: {e}") from e
        if sign is not None and float(sign) != pin["sign"]:
            raise BlissIngestError(f"--bliss-omega-sign {sign:+d} disagrees with the sign {pin['sign']:+.0f} the "
                                   f"reference zips pin (residual p99 {pin['residual_p99']:.4f} deg vs "
                                   f"{pin['other_residual_p99']:.4f} deg for the other sign)")
        if offset and abs(offset - pin["offset"]) > 0.05:
            raise BlissIngestError(f"--bliss-omega-offset {offset} disagrees with the pinned offset {pin['offset']:+.4f}")
        return _Plan(pin["sign"], pin["offset"] if allow_offset else 0.0, pin)
    return _Plan(float(sign), float(offset), None)


def _convert_in_subprocess(sc, out_zip, param_file, omega_sign, omega_offset, mask, preproc_thresh, work_dir) -> str:
    """One scan, one fresh interpreter. The zipper is driven through sys.argv and cannot be called concurrently from
    threads of one process; a plain ``python -m`` child (not multiprocessing) also never re-imports the caller's
    ``__main__``, so callers need no ``if __name__ == "__main__"`` guard."""
    payload = Path(work_dir) / f".bliss_payload_{Path(out_zip).stem}.pkl"
    payload.write_bytes(pickle.dumps((sc, str(out_zip), str(param_file), omega_sign, omega_offset, mask,
                                      int(preproc_thresh), str(work_dir))))
    try:
        r = subprocess.run([sys.executable, "-m", "midas_pipeline.bliss_ingest", str(payload)],
                           capture_output=True, text=True, check=False)
    finally:
        payload.unlink(missing_ok=True)
    if r.returncode != 0:
        tail = "\n".join([ln for ln in (r.stderr or "").splitlines() if ln.strip()][-8:])
        raise RuntimeError(f"zipper child for {Path(out_zip).name} exited {r.returncode}:\n{tail}")
    return "built"


def _child_main(payload: str) -> int:
    from midas_zipper.bliss import convert_scan

    sc, out_zip, param_file, sign, offset, mask, thresh, work_dir = pickle.loads(Path(payload).read_bytes())
    convert_scan(sc, out_zip, param_file, omega_sign=sign, omega_offset=offset, mask=mask, preproc_thresh=thresh,
                 work_dir=work_dir)
    return 0


def build_zips_from_bliss(
    scans: Sequence[PFScanInfo],
    *,
    master: str | Path,
    param_file: str | Path,
    layer_dir: Path,
    scan_step_um: float,
    omega_sign: Optional[int] = None,
    omega_offset: float = 0.0,
    omega_ref_raw: Optional[str] = None,
    allow_offset: bool = False,
    mask_from: Optional[str] = None,
    detector: str = "eiger",
    preproc_thresh: int = -1,
    zip_workers: int = 1,
    dty_step_tol: float = 0.05,
) -> dict:
    """Write ``scan.zip_path`` for every scan whose zip does not exist yet. Scan ``s`` of the run is entry ``s`` of
    the master in acquisition order. All gates run before the first zip is written. Returns the record that is also
    saved as ``<layer_dir>/midas_log/bliss_ingest.json``."""
    import zarr
    from midas_zipper.bliss import MASK_KEY, survey_master, wrap_omega_180

    master = Path(master)
    if not master.is_file():
        raise BlissIngestError(f"--bliss-master {master}: not a file")
    sc = survey_master(master, detector=detector)
    n = len(scans)
    frames = sorted({s.n_frames for s in sc})
    if len(sc) != n or len(frames) != 1:
        raise BlissIngestError(f"{master.name}: {len(sc)} scans with frame counts {frames}, but the run has {n} "
                               "scan positions (positions.csv) and needs one frame count")
    dty = np.array([np.nan if s.dty is None else s.dty for s in sc], dtype=float)
    if np.isnan(dty).any():
        raise BlissIngestError(f"{master.name}: scans {list(np.flatnonzero(np.isnan(dty)) + 1)[:10]} have no dty")
    step = np.diff(dty)
    if n > 1 and (np.abs(np.abs(step) - abs(scan_step_um)) > dty_step_tol).any():
        k = int(np.argmax(np.abs(np.abs(step) - abs(scan_step_um))))
        raise BlissIngestError(f"dty steps are not the --scan-step {scan_step_um}: between scans {k + 1} and {k + 2} "
                               f"they are {step[k]:+.4f} um (median {np.median(step):+.4f})")
    pos = np.array([s.y_position_um for s in scans], dtype=float)
    if np.allclose(pos, dty, atol=dty_step_tol):
        pos_vs_dty = "+dty"
    elif np.allclose(pos, -dty, atol=dty_step_tol):
        pos_vs_dty = "-dty"
    else:
        pos_vs_dty = "neither"
        LOG.warning("bliss ingest: positions.csv matches neither +dty nor -dty of the master (positions %.3f..%.3f, "
                    "dty %.3f..%.3f); check the scan origin before trusting any map", pos[0], pos[-1], dty[0], dty[-1])
    rots = [s.rot_center for s in sc]
    plan = _choose_omega(rots, sign=omega_sign, offset=omega_offset, ref_raw=omega_ref_raw, allow_offset=allow_offset)
    omegas = []
    for k, r in enumerate(rots, 1):                       # refuse a bad scan before anything is written
        try:
            omegas.append(wrap_omega_180(plan.sign * r + plan.offset))
        except ValueError as e:
            raise BlissIngestError(f"scan {k}: {e}") from e
    n_shifted = sum(1 for _, sh in omegas if sh)
    rising = np.array([r[-1] > r[0] for r in rots])
    mask = None
    if mask_from is not None:
        mask = np.asarray(zarr.open(str(mask_from), "r")[MASK_KEY][...])
    else:
        # Bliss/Lima frames carry the detector gaps as a huge pixel value, not as a mask. Without exchange/mask the
        # peak search treats gap edges as saturated (seen on a real Eiger pf scan series: nMasks 0, about 64 pct of regions
        # saturated, about 2 pct of raw peaks lost, lower PBP completeness than a masked control). Say so loudly, and record it.
        LOG.warning("bliss ingest: NO --bliss-mask-from: the zips will have no exchange/mask, so detector gaps are "
                    "unmasked in the peak search. Pass a zip of the same detector that carries exchange/mask.")

    todo = [s for s in scans if not s.zip_path.exists()]

    def _one(s: PFScanInfo) -> str:
        s.scan_dir.mkdir(parents=True, exist_ok=True)
        return _convert_in_subprocess(sc[s.scan_nr - 1], s.zip_path, param_file, plan.sign, plan.offset, mask,
                                      preproc_thresh, s.scan_dir)

    outcomes = fan_out_scans(todo, _one, layer_dir=layer_dir, stage="bliss_ingest", n_workers=max(1, int(zip_workers)))
    failed = [(s.scan_nr, o) for s, o in outcomes if isinstance(o, Exception)]
    record = {
        "master": str(master.resolve()), "detector": detector, "n_scans": n, "frames_per_scan": frames[0],
        "dty_first": float(dty[0]), "dty_last": float(dty[-1]), "dty_median_step": float(np.median(step)) if n > 1 else 0.0,
        "positions_vs_dty": pos_vs_dty,
        "omega_sign": plan.sign, "omega_offset": plan.offset, "omega_pin": plan.pin,
        "omega_min": float(min(o.min() for o, _ in omegas)), "omega_max": float(max(o.max() for o, _ in omegas)),
        "n_scans_wrapped_into_pm180": n_shifted,
        "n_scans_rising": int(rising.sum()), "n_scans_falling": int((~rising).sum()),
        "snake_odd_rising_even_falling": bool(rising[0::2].all() and (~rising[1::2]).all()),
        "mask_from": None if mask_from is None else str(mask_from),
        "preproc_thresh": int(preproc_thresh), "n_built": len(todo) - len(failed), "n_present": n - len(todo),
        "n_failed": len(failed),
    }
    out = Path(layer_dir) / "midas_log" / "bliss_ingest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    LOG.info("bliss ingest: %d built, %d present, %d failed; omega = %+.0f*rot%+.4f in [%.3f, %.3f]; dty %.3f..%.3f "
             "(positions.csv = %s); snake %s -> %s", record["n_built"], record["n_present"], record["n_failed"],
             plan.sign, plan.offset, record["omega_min"], record["omega_max"], dty[0], dty[-1], pos_vs_dty,
             record["snake_odd_rising_even_falling"], out)
    if failed:
        raise BlissIngestError(f"{len(failed)} scans failed, first: scan {failed[0][0]}: {failed[0][1]!r}")
    return record


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1]))
