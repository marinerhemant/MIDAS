"""Canonical return types and enums for midas_defect analyses.

Every public analysis in midas_defect returns an :class:`AnalysisResult`. The
type is shaped to compose downstream: bootstrap samples are retained in full
so callers can re-aggregate (e.g. propagate UQ through Mecking-Kocks closure
or stratify the population CI by a covariate computed post-hoc).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CrystalPhase(Enum):
    FCC = "FCC"
    BCC = "BCC"
    HCP = "HCP"


class BootUnit(Enum):
    """Statistical resampling unit for bootstrap UQ."""

    VOXEL = "voxel"
    GRAIN = "grain"
    PAIR = "pair"
    REFLECTION = "reflection"
    LAYER = "layer"


_OPTIONAL_ARRAYS = ("per_grain", "per_pair", "per_reflection")


@dataclass
class AnalysisResult:
    """Canonical return type for every analysis in midas_defect.

    Conventions
    -----------
    per_grain / per_pair / per_reflection
        NaN encodes failed-fit entries; downstream callers must mask via
        :func:`numpy.isfinite`. Never raise inside a per-element compute.
    bootstrap_samples
        Full ``(n_boot,)`` array of the bootstrap statistic (typically the
        median, but ``stat_fn`` is analysis-defined). Retained on purpose
        so composite UQ can be recomputed without rerunning the bootstrap.
    population_ci
        ``(p16, p84)`` from ``bootstrap_samples`` = +/- 1 sigma. Always
        ordered ``low <= high``.
    metadata
        Free-form, analysis-specific extras (q_Ungar grid, k_NN, RNG seed,
        active slip system index, etc.).
    """

    name: str
    units: str
    boot_unit: BootUnit
    n_boot: int

    population_median: float
    population_ci: tuple[float, float]
    bootstrap_samples: NDArray[np.floating]

    per_grain: NDArray[np.floating] | None = None
    per_pair: NDArray[np.floating] | None = None
    per_reflection: NDArray[np.floating] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lo, hi = self.population_ci
        if lo > hi:
            raise ValueError(
                f"population_ci must be ordered (low <= high); got ({lo}, {hi})"
            )
        if self.n_boot < 0:
            raise ValueError(f"n_boot must be non-negative; got {self.n_boot}")
        self.bootstrap_samples = np.asarray(self.bootstrap_samples)
        if self.bootstrap_samples.ndim != 1:
            raise ValueError(
                "bootstrap_samples must be 1-D; got shape "
                f"{self.bootstrap_samples.shape}"
            )
        if self.bootstrap_samples.size != self.n_boot:
            raise ValueError(
                f"bootstrap_samples length {self.bootstrap_samples.size} "
                f"does not match n_boot={self.n_boot}"
            )
        for name in _OPTIONAL_ARRAYS:
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation; numpy arrays become lists."""
        out: dict[str, Any] = {
            "name": self.name,
            "units": self.units,
            "boot_unit": self.boot_unit.value,
            "n_boot": int(self.n_boot),
            "population_median": float(self.population_median),
            "population_ci": [float(self.population_ci[0]), float(self.population_ci[1])],
            "bootstrap_samples": np.asarray(self.bootstrap_samples).tolist(),
            "metadata": _metadata_to_jsonable(self.metadata),
        }
        for name in _OPTIONAL_ARRAYS:
            arr = getattr(self, name)
            out[name] = None if arr is None else np.asarray(arr).tolist()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisResult":
        return cls(
            name=d["name"],
            units=d["units"],
            boot_unit=BootUnit(d["boot_unit"]),
            n_boot=int(d["n_boot"]),
            population_median=float(d["population_median"]),
            population_ci=(float(d["population_ci"][0]), float(d["population_ci"][1])),
            bootstrap_samples=np.asarray(d["bootstrap_samples"], dtype=float),
            per_grain=None if d.get("per_grain") is None else np.asarray(d["per_grain"], dtype=float),
            per_pair=None if d.get("per_pair") is None else np.asarray(d["per_pair"], dtype=float),
            per_reflection=None
            if d.get("per_reflection") is None
            else np.asarray(d["per_reflection"], dtype=float),
            metadata=dict(d.get("metadata", {})),
        )


def _metadata_to_jsonable(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = _metadata_to_jsonable(v)
        else:
            out[k] = v
    return out


__all__ = ["AnalysisResult", "BootUnit", "CrystalPhase"]
