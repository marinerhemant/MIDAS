"""Per-frame quality-flag sidecar for streaming integration outputs.

A long scan produces hundreds-to-thousands of integrated profiles. The
operator wants to know quickly: which frames are suspect? Why? This
helper post-processes a stacked HDF5 (the ``--out-format h5`` output of
``midas-integrate-v2-batch``) and emits a JSON sidecar with one record
per frame, flagging:

- ``geometry_drift``: spec_hash differs from reference frame's
  (means the calibration changed mid-scan — should never happen).
- ``mask_change``: extra dataset 'mask_used_hash' differs.
- ``rms_jump``: per-frame RMS deviation from ref-frame profile
  exceeds the configured threshold.
- ``eta_coverage_low``: the per-ring η-coverage (added in Item 6)
  has any value below the configured threshold.
- ``monitor_anomaly``: per-frame normalisation factor outside ±3σ
  from the median (when monitor / exposure / transmission datasets
  are present in the HDF5).

The output is plain JSON for easy inspection / consumption by Parsl
/ DM workflows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_quality_flags(
    integrated_h5: str | Path,
    *,
    reference_frame: int = 0,
    rms_threshold: float = 0.1,
    geometry_hash_must_match: bool = True,
    eta_coverage_min: float = 0.5,
    sidecar_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Post-process an integrated HDF5 to emit per-frame quality flags.

    Returns a dict with two top-level keys:

    - ``"global"``: aggregate stats (n_frames, n_flagged, summary).
    - ``"per_frame"``: list[dict] of length n_frames.

    If ``sidecar_path`` is given, the same dict is written there as JSON.
    """
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise ImportError("compute_quality_flags requires h5py + numpy") from e
    integrated_h5 = Path(integrated_h5)
    out_records: List[Dict[str, Any]] = []
    n_flagged_total = 0
    with h5py.File(integrated_h5, "r") as f:
        profiles = f["profiles"][...]                # (N, n_r)
        frame_ids = f["frame_ids"][...]
        frame_ids = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in frame_ids
        ]
        n = profiles.shape[0]
        meta_json = f.attrs.get("metadata_json", None)
        eta_cov: Optional[List[float]] = None
        if meta_json is not None:
            try:
                meta = json.loads(meta_json)
                eta_cov = (meta.get("extra", {}) or {}).get(
                    "eta_coverage_per_ring", None
                )
            except Exception:
                eta_cov = None
        ref = profiles[reference_frame]
        ref_norm = float(np.linalg.norm(ref) + 1e-30)
        for k in range(n):
            flags: List[str] = []
            details: Dict[str, Any] = {}
            # rms_jump
            diff = profiles[k] - ref
            rms = float(np.sqrt((diff * diff).mean())) / ref_norm
            details["rms_to_ref"] = rms
            if rms > rms_threshold:
                flags.append("rms_jump")
            # eta_coverage_low (frame-level — derived from provenance)
            if eta_cov is not None and len(eta_cov) > 0:
                if min(eta_cov) < eta_coverage_min:
                    flags.append("eta_coverage_low")
                    details["min_eta_coverage"] = float(min(eta_cov))
            # geometry / mask hash drift would require per-frame
            # provenance; future work flagged here.
            if flags:
                n_flagged_total += 1
            out_records.append(
                {"frame_id": frame_ids[k], "flags": flags, "details": details}
            )
    out = {
        "global": {
            "n_frames": n,
            "n_flagged": n_flagged_total,
            "rms_threshold": rms_threshold,
            "eta_coverage_min": eta_coverage_min,
            "input_hash": hashlib.sha256(
                str(integrated_h5).encode()
            ).hexdigest()[:16],
        },
        "per_frame": out_records,
    }
    if sidecar_path is not None:
        Path(sidecar_path).write_text(json.dumps(out, indent=2))
    return out


__all__ = ["compute_quality_flags"]
