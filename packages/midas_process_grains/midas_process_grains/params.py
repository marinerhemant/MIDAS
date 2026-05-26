"""ProcessGrains parameters.

Extends ``midas_transforms.params.ParamsTest`` with the new keys this package
needs:

  * ``MisoriTol``        — Phase 1 misorientation cutoff (deg). Default 0.25.
  * ``MinNrSpots``       — minimum cluster size to qualify as a grain. Floor 2.
  * ``JaccardTol``       — Phase 2 SpotID-set Jaccard threshold.
  * ``AgreementTol``     — Phase 2 per-hkl SpotID-agreement threshold.
  * ``MergeAlpha``       — Phase 2 weight blending Jaccard vs. per-hkl agreement.
  * ``PixelTol``         — Phase 2 informative-hkl filter (in detector pixels).
  * ``StrainMethod``     — "lstsq" (default), "lattice", or "both".
  * ``ConflictPolicy``   — "vote_then_residual" (default) or "forward_sim".
  * ``MaterialName``     — for stress output via Hooke's law.
  * ``StiffnessFile``    — alternative path to a 6×6 stiffness matrix.
  * ``TwinRelations``    — path to user twin-orientation-relationships file.
  * ``ConfidenceTol``    — minimum Confidence to keep a grain.
  * ``Twin``             — 0/1, enable twin post-processor.

Per the implementation plan §6, defaults for ``MisoriTol`` and ``MinNrSpots``
also depend on ``mode``. The reader accepts the explicit value if set; otherwise
the mode-dispatch in ``modes.py`` substitutes the per-mode default at run time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from midas_transforms.params import ParamsTest, read_paramstest as _read_paramstest_base


# Sentinel: "not set in paramstest" — distinguishes default from explicit 0.
_UNSET = float("nan")


@dataclass
class ProcessGrainsParams:
    """Typed view of paramstest.txt + ProcessGrains-specific keys.

    Wraps ``ParamsTest`` so we get the full upstream key set for free
    (Lsd, Wavelength, RingNumbers, ...). Adds the new merge/strain/stress
    knobs as first-class fields.
    """

    # --- the upstream ParamsTest, fully populated --------------------------
    base: ParamsTest = field(default_factory=ParamsTest)

    # --- ProcessGrains-specific keys ---------------------------------------
    MisoriTol: Optional[float] = None      # deg; mode-default if None
    MinNrSpots: int = 2                    # floor; clamped if user sets <2
    JaccardTol: float = 0.5
    AgreementTol: float = 0.7
    MergeAlpha: float = 0.6
    PixelTol: float = 1.0
    # Strain method:
    #   "kenesei"        — bounded per-spot lstsq (paper Eq. 8-11). DEFAULT.
    #   "fable_beaudoin" — lattice-parameter mapping (paper Eq. 5-7).
    #   "kenesei_unbounded" — autograd-friendly closed-form lstsq (FF-HEDM
    #                          ε_xx blow-up; use only when gradients matter).
    #   "both"           — emit kenesei in Grains.csv plus fable_beaudoin in
    #                      data_consolidated.h5/grains/strain_fable_beaudoin.
    #   Backwards-compat aliases (still accepted): "lstsq" → "kenesei",
    #                                              "lattice" → "fable_beaudoin".
    StrainMethod: str = "kenesei"
    ConflictPolicy: str = "vote_then_residual"
    MaterialName: Optional[str] = None
    StiffnessFile: Optional[str] = None
    TwinRelations: Optional[str] = None
    ConfidenceTol: float = 0.0
    Twin: int = 0
    PhaseNr: int = 1

    # --- Pass A (post-Phase-1 spot-overlap merge of cluster reps) -----------
    # Replaces C ProcessGrains' position-based Pass A (5 µm + 0.1°) with a
    # spot-overlap-based merge. Two Phase-1 cluster reps are merged into one
    # super-cluster if their misorientation < PassAMisoriTol AND their
    # resolved-SpotID Jaccard ≥ PassAJaccardTol.
    EnablePassA: bool = True
    PassAMisoriTol: float = 1.0    # deg
    # Empirical (peakfit_hard, IBF col 0): within-cluster Jaccard median 0.16,
    # p5 0.02; cross-cluster Jaccard max 0.0041. So 0.02 catches 94.6 % of
    # true same-grain pairs with 0 % false positives. See compute/pass_a.py
    # docstring; rerun /tmp/probe_jaccard.py if your data has a different
    # IBF noise floor.
    PassAJaccardTol: float = 0.02

    # ---- Joint-NNLS grain-volume correction (compute/volume_nnls.py) -----
    # Replace the standard ``GrainRadius = mean(R_per_spot)`` formula with a
    # sparse non-negative least squares solution that accounts for shared
    # spots between twin partners, sub-grains, and crowded neighbours.
    # Twins (Σ3, Σ9, Σ27) share reflections exactly; the naive formula
    # attributes the combined intensity to BOTH partners, inflating both
    # radii by 20–40 %. NNLS attributes shared intensity correctly.
    # Off by default for byte-level compatibility with the C reference.
    NnlsVolume: bool = False

    # When True (and NnlsVolume is True), compute K(ring) from the physical
    # formula |F|²·LP·DWF·multiplicity using Cromer-Mann atomic scattering
    # factors. When False, use the empirical median observed intensity per
    # ring. On dense datasets the two agree to ~0.3 % at the population
    # level; physical K is preferred when the dataset is sparse (the
    # empirical median is unreliable). Requires NnlsVolume = True.
    PhysicalK: bool = False

    # raw passthrough for any key we don't model explicitly
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors mirroring ParamsTest ------------------------
    @property
    def SGNr(self) -> int:                                 # noqa: N802
        return self.base.SpaceGroup

    @property
    def Lsd(self) -> float:                                # noqa: N802
        return self.base.Lsd

    @property
    def Wavelength(self) -> float:                         # noqa: N802
        return self.base.Wavelength

    @property
    def px(self) -> float:                                 # noqa: N802
        return self.base.px

    @property
    def OutputFolder(self) -> str:                         # noqa: N802
        return self.base.OutputFolder

    @property
    def ResultFolder(self) -> str:                         # noqa: N802
        return self.base.ResultFolder

    @property
    def LatticeConstant(self) -> Tuple[float, ...]:        # noqa: N802
        return self.base.LatticeConstant

    @property
    def RingNumbers(self) -> List[int]:                    # noqa: N802
        return self.base.RingNumbers

    @property
    def RingRadii(self) -> List[float]:                    # noqa: N802
        return self.base.RingRadii

    def to_dict(self) -> Dict[str, Any]:
        """Flatten for logging / HDF5 attrs."""
        d = asdict(self)
        d["base"] = asdict(self.base)
        return d

    # ---- validation --------------------------------------------------------
    def validated(self) -> "ProcessGrainsParams":
        """Apply input clamps and warnings; returns self for chaining."""
        if self.MinNrSpots < 2:
            warnings.warn(
                f"MinNrSpots={self.MinNrSpots} is below the minimum of 2; "
                "clamping to 2.",
                stacklevel=2,
            )
            self.MinNrSpots = 2
        if not (0.0 <= self.JaccardTol <= 1.0):
            raise ValueError(f"JaccardTol must be in [0, 1]; got {self.JaccardTol}")
        if not (0.0 <= self.AgreementTol <= 1.0):
            raise ValueError(f"AgreementTol must be in [0, 1]; got {self.AgreementTol}")
        if not (0.0 <= self.MergeAlpha <= 1.0):
            raise ValueError(f"MergeAlpha must be in [0, 1]; got {self.MergeAlpha}")
        if self.PixelTol < 0:
            raise ValueError(f"PixelTol must be >= 0; got {self.PixelTol}")
        # Map backwards-compat aliases to the canonical paper names.
        _aliases = {
            "lstsq": "kenesei",
            "lattice": "fable_beaudoin",
        }
        if self.StrainMethod in _aliases:
            self.StrainMethod = _aliases[self.StrainMethod]
        valid = {"kenesei", "kenesei_unbounded", "fable_beaudoin", "both"}
        if self.StrainMethod not in valid:
            raise ValueError(
                f"StrainMethod must be one of {sorted(valid)} "
                f"(or alias lstsq/lattice); got {self.StrainMethod}"
            )
        if self.ConflictPolicy not in {"vote_then_residual", "forward_sim"}:
            raise ValueError(
                f"ConflictPolicy must be vote_then_residual or forward_sim; "
                f"got {self.ConflictPolicy}"
            )
        return self


# Keys we recognize beyond what ParamsTest knows.
_PG_FLOAT_KEYS = {
    "MisoriTol", "JaccardTol", "AgreementTol", "MergeAlpha",
    "PixelTol", "ConfidenceTol",
}
_PG_INT_KEYS = {"MinNrSpots", "Twin", "PhaseNr"}
_PG_STR_KEYS = {
    "StrainMethod", "ConflictPolicy", "MaterialName",
    "StiffnessFile", "TwinRelations",
}


def read_paramstest_pg(path: Union[str, Path]) -> ProcessGrainsParams:
    """Parse a paramstest.txt file into a ``ProcessGrainsParams``.

    Reads the upstream ``ParamsTest`` keys via
    :func:`midas_transforms.params.read_paramstest`, then walks the file once
    more to pick up the PG-specific keys.
    """
    base = _read_paramstest_base(path)
    p = ProcessGrainsParams(base=base)
    with open(path, "r") as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip().rstrip(";")
            if not line:
                continue
            tokens = [t.rstrip(";") for t in line.split()]
            if not tokens:
                continue
            key, args = tokens[0], tokens[1:]
            if not args:
                continue
            p.raw[key] = args
            if key in _PG_FLOAT_KEYS:
                setattr(p, key, float(args[0]))
            elif key in _PG_INT_KEYS:
                setattr(p, key, int(args[0]))
            elif key in _PG_STR_KEYS:
                setattr(p, key, args[0])
    return p.validated()
