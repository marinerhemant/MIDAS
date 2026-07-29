"""Master-inventory CSV: write / load every :class:`AnalysisResult` in one table.

Each row is one AnalysisResult, with the (full) bootstrap sample array stored
as a single JSON-serialised column. This keeps the CSV human-readable for the
headline numbers while preserving the full distribution for downstream UQ
composition (see plan decision #2: retain all samples).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..types import AnalysisResult, BootUnit

_CSV_COLUMNS = [
    "name",
    "units",
    "boot_unit",
    "n_boot",
    "median",
    "ci_low",
    "ci_high",
    "bootstrap_samples_json",
    "per_grain_json",
    "per_pair_json",
    "per_reflection_json",
    "metadata_json",
]


def write_master_inventory_csv(
    results: Iterable[AnalysisResult],
    output_path: str | Path,
) -> None:
    """Write a list of :class:`AnalysisResult` to a single CSV.

    The bootstrap-sample array (and any per-grain / per-pair / per-reflection
    arrays) are serialised as JSON list strings so a downstream reader can
    reconstruct full :class:`AnalysisResult` instances without information loss.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {
                "name": r.name,
                "units": r.units,
                "boot_unit": r.boot_unit.value,
                "n_boot": int(r.n_boot),
                "median": float(r.population_median),
                "ci_low": float(r.population_ci[0]),
                "ci_high": float(r.population_ci[1]),
                "bootstrap_samples_json": json.dumps(np.asarray(r.bootstrap_samples).tolist()),
                "per_grain_json": _arr_to_json(r.per_grain),
                "per_pair_json": _arr_to_json(r.per_pair),
                "per_reflection_json": _arr_to_json(r.per_reflection),
                "metadata_json": json.dumps(_jsonable(r.metadata)),
            }
            writer.writerow(row)


def load_master_inventory_csv(input_path: str | Path) -> list[AnalysisResult]:
    """Reverse of :func:`write_master_inventory_csv`."""
    in_path = Path(input_path)
    out: list[AnalysisResult] = []
    with in_path.open("r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(
                AnalysisResult(
                    name=row["name"],
                    units=row["units"],
                    boot_unit=BootUnit(row["boot_unit"]),
                    n_boot=int(row["n_boot"]),
                    population_median=float(row["median"]),
                    population_ci=(float(row["ci_low"]), float(row["ci_high"])),
                    bootstrap_samples=np.asarray(
                        json.loads(row["bootstrap_samples_json"]), dtype=float
                    ),
                    per_grain=_json_to_arr(row["per_grain_json"]),
                    per_pair=_json_to_arr(row["per_pair_json"]),
                    per_reflection=_json_to_arr(row["per_reflection_json"]),
                    metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                )
            )
    return out


def _arr_to_json(a) -> str:
    if a is None:
        return ""
    return json.dumps(np.asarray(a).tolist())


def _json_to_arr(s: str):
    if not s:
        return None
    return np.asarray(json.loads(s), dtype=float)


def _jsonable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


__all__ = ["load_master_inventory_csv", "write_master_inventory_csv"]
