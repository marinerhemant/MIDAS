"""MIDAS FF/NF grain map -> grain selection -> TT scan plan.

Phase 6 of ``implementation_plan.md``. This closes the loop that makes the
package usable at APS: the c-omp far-field indexer and refiner already produce a
``Grains.csv`` for every dataset we take, and topotomography needs exactly what
that file contains -- an orientation, a position, a refined unit cell, and a
quality figure. So the scan plan is derivable from work already done, with no
extra measurement.

The convention chain, pinned rather than assumed
------------------------------------------------
Everything here rests on one composition, ``G_sample = OM @ B @ hkl``, and both
halves of it were checked against something external rather than reasoned about:

* **``OM`` is used directly, not transposed.** Measured on 300 real grains from
  ``bt_1id_nov25/Grains.csv``: the ``O11..O33`` block reproduces
  ``midas_stress.orientation.euler_to_orient_mat(Eul0, Eul1, Eul2)`` (radians) to
  **1.1e-06**, which is the file's own print precision, while the transpose
  disagrees by **2.0**. So ``Grains.csv``'s OM *is* a ``midas_stress``
  orientation matrix and needs no adjustment.
* **``OM`` maps crystal -> sample.** Taken from the validated in-house forward
  model rather than re-derived: ``midas_diffract.forward`` computes
  ``G_C = einsum("nij,mj->nmi", orientation_matrices, hkls_cart)``, i.e. it
  left-multiplies the Cartesian G by the orientation matrix with no transpose.

``B`` comes from :func:`midas_dfxm.field.reciprocal_basis` (the ``2*pi``
convention, ``|G| = 2*pi/d``), fed the refined ``a b c alpha beta gamma``
columns.

Why reading the refined cell matters
------------------------------------
:func:`~midas_dct_tt.conventions.tt_alignment` must be solved on the grain's
**actual** lattice; aligning on an undeformed reference leaves the scan off the
Bragg condition and manufactures a spurious orientation dependence (measured
3.7x spread, collapsing to 1.0x when aligned properly -- see the warning on
``tt_alignment``). Using ``Grains.csv`` closes that trap for free, because FF
refinement already fits ``a b c alpha beta gamma`` per grain, and this module
feeds those refined values straight into ``B``. A plan built here is aligned on
the real cell by construction.

Why this file has its own CSV reader
------------------------------------
``midas_pipeline.seeding.handoff._parse_grains_csv`` exists, but it is private,
returns only ``(orientation matrices, ids)``, and grain *selection* needs the
position, radius and confidence columns it discards. Depending on
``midas_pipeline`` -- a full pipeline package -- from this leaf would also be a
large dependency for one parser. The header handling here deliberately copies
its hard-won robustness: key off the ``O11`` column rather than the ``%GrainID``
literal, because ``ProcessGrains`` has spelled the ID column both ``GrainID`` and
``ID`` and anchoring on the literal silently rejects valid files.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from midas_dfxm.field import reciprocal_basis

from .conventions import tt_alignment
from .geometry import missing_cone_half_angle_deg
from .planning import accessible_reflections, rank_reflection_sets

__all__ = [
    "FF_RADIUS_FLOOR_UM",
    "GrainRecord",
    "TTScanPlan",
    "read_grains_csv",
    "radius_is_suspect",
    "select_tt_candidates",
    "tt_scan_plan",
]

#: Below this, an FF-HEDM ``GrainRadius`` is not a measurement. Far-field HEDM
#: cannot resolve sub-micron grains at all, so a median radius under 1 um means
#: the column is reporting something other than grain size.
FF_RADIUS_FLOOR_UM = 1.0

_OM_COLS = ("O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33")
_CELL_COLS = ("a", "b", "c", "alpha", "beta", "gamma")


@dataclass
class GrainRecord:
    """One grain from a MIDAS ``Grains.csv``, in MIDAS units (um, degrees, A).

    ``orientation`` is the crystal->sample matrix exactly as stored; ``lattice``
    is the **refined** cell for this grain, which is what makes the derived TT
    alignment correct rather than nominal.
    """

    grain_id: int
    orientation: torch.Tensor          # (3, 3), crystal -> sample
    position_um: torch.Tensor          # (3,)
    lattice: torch.Tensor              # (6,) a, b, c (A), alpha, beta, gamma (deg)
    radius_um: float = float("nan")
    confidence: float = float("nan")
    phase_nr: int = 1
    rms_error_strain: float = float("nan")

    def reciprocal_basis(self) -> torch.Tensor:
        """``B`` for this grain's refined cell, ``2*pi`` convention."""
        return reciprocal_basis(self.lattice)

    def G_sample(self, hkl) -> torch.Tensor:
        """``OM @ B @ hkl`` -- the sample-frame reciprocal vector. See module docs."""
        h = torch.as_tensor(hkl, dtype=self.lattice.dtype)
        return self.orientation.to(self.lattice.dtype) @ (self.reciprocal_basis() @ h)

    def offcenter_um(self) -> float:
        """Distance from the sample origin.

        This is *the* TT position criterion, not the off-axis radius: a TT scan
        rotates about an axis through the origin, so a grain at distance ``r``
        sweeps a circle of radius up to ``r`` and must stay in the beam and on
        the detector for the whole 360 deg. A grain at the origin never moves,
        whatever the axis.
        """
        return float(torch.linalg.vector_norm(self.position_um))


def read_grains_csv(path) -> tuple:
    """Read a ``ProcessGrains`` ``Grains.csv`` -> ``(list[GrainRecord], metadata)``.

    ``metadata`` holds the ``%key value`` preamble lines (``NumGrains``,
    ``BeamCenter``, ...) as strings.

    Columns beyond the required orientation/position/cell block are optional:
    ``GrainRadius``, ``Confidence``, ``PhaseNr`` and ``RMSErrorStrain`` are
    filled with NaN (or 1 for the phase) when absent, so a trimmed file still
    loads and only the selection filters that need them will complain.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    metadata, header_idx, cols = {}, None, None
    for i, line in enumerate(lines):
        if not line.startswith("%"):
            continue
        toks = line[1:].split()
        if "O11" in toks:
            header_idx, cols = i, toks
            break
        if len(toks) >= 2:
            metadata[toks[0]] = " ".join(toks[1:])

    if header_idx is None:
        raise ValueError(
            f"{path}: no '%' header line containing the orientation columns "
            f"{list(_OM_COLS)} -- not a ProcessGrains output file?"
        )

    idx = {name: k for k, name in enumerate(cols)}
    missing = [c for c in _OM_COLS + _CELL_COLS + ("X", "Y", "Z") if c not in idx]
    if missing:
        raise ValueError(f"{path}: header is missing required columns {missing}")

    gid_col = next((c for c in ("GrainID", "ID") if c in idx), None)

    # A row only has to carry the columns we actually need, NOT the full header
    # width. Both directions occur in real files: mpe_dec24/Grains.csv declares 47
    # columns but writes 19 (exactly GrainID + OM + XYZ + cell, a run that stopped
    # before the strain block), while bt_1id_nov25/Grains.csv writes 48 against a
    # 47-name header (an unnamed trailing column). Requiring len(row) >= len(header)
    # threw away the whole of the first file; left-anchored positional indexing
    # handles the second for free. Optional columns past the end degrade to NaN via
    # _get, and select_tt_candidates refuses to filter on a NaN column.
    need_upto = max([idx[c] for c in _OM_COLS + _CELL_COLS + ("X", "Y", "Z")]
                    + ([idx[gid_col]] if gid_col is not None else []))

    def _get(toks, name, default=float("nan")):
        k = idx.get(name)
        if k is None or k >= len(toks):
            return default
        try:
            return float(toks[k])
        except ValueError:
            return default

    grains = []
    for raw in lines[header_idx + 1:]:
        if not raw.strip() or raw.startswith("%"):
            continue
        toks = raw.split()
        if len(toks) <= need_upto:
            continue
        try:
            om = torch.tensor([float(toks[idx[c]]) for c in _OM_COLS],
                              dtype=torch.float64).reshape(3, 3)
            pos = torch.tensor([float(toks[idx[c]]) for c in ("X", "Y", "Z")],
                               dtype=torch.float64)
            cell = torch.tensor([float(toks[idx[c]]) for c in _CELL_COLS],
                                dtype=torch.float64)
        except (ValueError, IndexError):
            continue
        gid = int(float(toks[idx[gid_col]])) if gid_col is not None else len(grains)
        grains.append(GrainRecord(
            grain_id=gid, orientation=om, position_um=pos, lattice=cell,
            radius_um=_get(toks, "GrainRadius"),
            confidence=_get(toks, "Confidence"),
            phase_nr=int(_get(toks, "PhaseNr", 1)),
            rms_error_strain=_get(toks, "RMSErrorStrain"),
        ))

    if not grains:
        raise ValueError(f"{path}: header found but no parsable data rows")
    return grains, metadata


def radius_is_suspect(grains) -> tuple:
    """Is this file's ``GrainRadius`` column trustworthy? ``(suspect, reason)``.

    ``GrainRadius`` is the natural size criterion for choosing a TT grain, and it
    is also the column with a known failure mode. ``midas_process_grains`` used to
    build its per-spot radius lookup from ``Radius_*.csv`` while everything
    downstream of the binner numbers spots in ``ExtraInfo.bin`` space -- the two
    order the same spots differently, so the join averaged ~112 arbitrary spots
    and **every grain came out near the global mean**, about 5.5x too small
    (20.775/17.161 um reported against 114.621/99.963 um true). That is fixed
    (``midas_process_grains/tests/test_spot_radius_id_space.py``), but *any
    Grains.csv written before the fix carries wrong sizes*, and nothing in the
    file records which version produced it.

    The discriminator used here is absolute magnitude, not spread. Measured
    across three real files: ``bt_1id_nov25`` (796 grains) and
    ``bt_20id_feb26/gpu_analysis`` (22003 grains) both report a median radius of
    **0.83 um** -- two unrelated samples landing on the same sub-micron value --
    while ``bt_1id_jun25b/compressed_lf`` reports 80.95 um. Coefficient of
    variation does *not* separate them (0.409, 0.131, 0.124 respectively), so
    the "compressed distribution" signature is not usable on its own;
    sub-micron-ness is, because far-field HEDM cannot resolve such grains.
    """
    r = [g.radius_um for g in grains if not math.isnan(g.radius_um)]
    if not r:
        return True, "no GrainRadius column in this file"
    r.sort()
    median = r[len(r) // 2]
    if median < FF_RADIUS_FLOOR_UM:
        return True, (
            f"median GrainRadius is {median:.2f} um, below the {FF_RADIUS_FLOOR_UM} um "
            "floor for far-field HEDM. This is the signature of the pre-fix "
            "Radius/ExtraInfo ID-space bug; treat sizes from this file as invalid "
            "and re-run ProcessGrains."
        )
    return False, f"median GrainRadius {median:.2f} um"


def select_tt_candidates(grains, *, min_radius_um=None, max_radius_um=None,
                         min_confidence=None, max_offcenter_um=None,
                         phase_nr=None, top_n=None):
    """Filter and rank grains for a topotomography scan. Biggest first.

    TT spends a whole scan on **one** grain, so the choice is expensive and the
    criteria are physical, not statistical:

    ``min_radius_um``
        The grain must be resolvable. Below a few detector pixels there is no
        intragranular map to reconstruct.
    ``max_radius_um``
        It must also fit: an oversized grain overruns the detector at some
        ``psi`` and the projections are truncated, which a reconstruction will
        happily turn into artefacts.
    ``min_confidence``
        A poorly-determined orientation means the goniometer will not actually
        land on the Bragg condition, so the scan returns nothing.
    ``max_offcenter_um``
        See :meth:`GrainRecord.offcenter_um` -- distance from the origin, since
        the grain orbits it during the scan.
    ``phase_nr``
        Restrict to one phase in a multiphase sample.

    Filters that would need an absent column raise rather than silently passing
    everything: a NaN comparison is False, so an un-guarded filter on a trimmed
    file would quietly reject *all* grains and look like "no candidates".
    """
    def _need(col, value, getter):
        if value is None:
            return
        if any(math.isnan(getter(g)) for g in grains):
            raise ValueError(
                f"cannot filter on {col}: it is absent (NaN) for at least one "
                "grain in this file"
            )

    if min_radius_um is not None or max_radius_um is not None:
        suspect, reason = radius_is_suspect(grains)
        if suspect:
            warnings.warn(
                f"filtering TT candidates on GrainRadius, but {reason}",
                RuntimeWarning, stacklevel=2,
            )

    _need("GrainRadius", min_radius_um, lambda g: g.radius_um)
    _need("GrainRadius", max_radius_um, lambda g: g.radius_um)
    _need("Confidence", min_confidence, lambda g: g.confidence)

    out = list(grains)
    if phase_nr is not None:
        out = [g for g in out if g.phase_nr == phase_nr]
    if min_radius_um is not None:
        out = [g for g in out if g.radius_um >= min_radius_um]
    if max_radius_um is not None:
        out = [g for g in out if g.radius_um <= max_radius_um]
    if min_confidence is not None:
        out = [g for g in out if g.confidence >= min_confidence]
    if max_offcenter_um is not None:
        out = [g for g in out if g.offcenter_um() <= max_offcenter_um]

    out.sort(key=lambda g: (-(g.radius_um if not math.isnan(g.radius_um) else 0.0),
                            g.offcenter_um()))
    return out[:top_n] if top_n is not None else out


@dataclass
class TTScanPlan:
    """A concrete TT scan proposal for one grain.

    ``alignments`` pairs each reflection of the chosen set with its solved
    :class:`~midas_dct_tt.conventions.TTAlignment`, so the plan carries the
    actual goniometer setting, not just a recommendation of which peaks to use.
    """

    grain_id: int
    wavelength_A: float
    report: object                     # ReflectionSetReport for the chosen set
    alignments: list                   # [(hkl, TTAlignment), ...]
    alternatives: list                 # further ReflectionSetReports, next-best first
    n_accessible: int = 0

    def summary(self) -> str:
        lines = [
            f"grain {self.grain_id}  lambda={self.wavelength_A:.4f} A  "
            f"{self.n_accessible} accessible reflections",
            "  chosen: " + self.report.summary(),
        ]
        for hkl, al in self.alignments:
            lines.append(
                f"    {str(tuple(hkl)):12s} theta {float(al.theta_deg):6.3f} deg  "
                f"missing cone {float(al.missing_cone_deg()):6.3f} deg"
            )
        return "\n".join(lines)


def tt_scan_plan(grain: GrainRecord, wavelength_A, *, hkl_max: int = 2,
                 max_theta_deg: float = 3.0, n_reflections: int = 3,
                 n_alternatives: int = 5, azimuth_deg: float = 90.0,
                 crystal=None) -> TTScanPlan:
    """Plan a TT scan for one grain: pick the reflection set, solve the alignments.

    Reflections are enumerated in this grain's own **sample** frame from its
    refined cell, ranked by :func:`~midas_dct_tt.planning.rank_reflection_sets`
    (zero strain-to-orientation leakage first, then missing cone, then
    conditioning), and the winning set is turned into goniometer settings.

    ``max_theta_deg`` is a genuine pre-filter, not a nicety: at HEXM energies
    nearly every low-index reflection is accessible, and the number of candidate
    triplets grows as the cube of the count. It also selects *good* reflections,
    since the TT missing cone has half-angle ``theta``.

    The 3.0 deg default is picked so the guard is not tripped by an ordinary
    case *and* so that the recommendation this module exists to make is
    reachable. The earlier 2.5 deg default was calibrated against a candidate
    pool that still contained systematic absences; once those are filtered, fcc
    at 71.7 keV admits only the eight {111} inside 2.5 deg, and the
    symmetry-closed {200} family -- the zero-leakage set, the whole point of
    ranking on leakage -- sits at theta = 2.726 deg, just outside. At 3.0 deg
    both families are in play and the leakage-optimal set is selectable.

    ``crystal`` is what makes the plan physical. Geometry admits ``d >=
    lambda/2``; it does not know about systematic absences. For fcc, 18 of those
    26 are **forbidden** ({100} and {110} both have mixed parity, so ``|F| = 0``)
    and only the 8 {111} reflections survive. This matters more than a filter
    normally would: the missing cone has half-angle ``theta``, so the *forbidden*
    low-angle reflections are exactly the ones a cone-based ranking prefers, and
    they win. Passing the crystal is therefore strongly recommended -- without it
    this function will happily return a beautifully-ranked plan for a scan that
    produces no diffraction at all.

    Raises
    ------
    ValueError
        If no full-rank set exists within the filter -- widen ``max_theta_deg``
        or ``hkl_max``. Reported rather than returned empty, because an empty
        plan is easy to mistake for a plan.
    """
    B = grain.reciprocal_basis()
    refl = accessible_reflections(
        B, wavelength_A, hkl_max=hkl_max, orientation=grain.orientation,
        max_theta_deg=max_theta_deg, crystal=crystal,
    )
    if len(refl) < n_reflections:
        raise ValueError(
            f"grain {grain.grain_id}: only {len(refl)} reflections within "
            f"theta <= {max_theta_deg} deg at lambda = {wavelength_A} A; need "
            f"{n_reflections}. Widen max_theta_deg or hkl_max."
        )

    # The CRLB must be evaluated on THIS grain's cell, not the midas_dfxm default
    # (a = 3.6356 A). For a cubic grain the error is a uniform scale and cancels in
    # ratios; for a non-cubic one it is simply the wrong matrix.
    reports = rank_reflection_sets(
        refl, n_reflections=n_reflections, top=n_alternatives + 1,
        crlb_kw={"lattice_params": tuple(float(v) for v in grain.lattice),
                 "orientation": grain.orientation},
    )
    if not reports:
        raise ValueError(
            f"grain {grain.grain_id}: no full-rank set of {n_reflections} "
            f"reflections within theta <= {max_theta_deg} deg. A full-rank set "
            "needs non-coplanar reflections; widen the filter."
        )

    best = reports[0]
    by_hkl = {tuple(h): g for h, g, _ in refl}
    alignments = [
        (hkl, tt_alignment(by_hkl[tuple(hkl)], wavelength_A, azimuth_deg=azimuth_deg))
        for hkl in best.hkls
    ]
    return TTScanPlan(
        grain_id=grain.grain_id, wavelength_A=float(wavelength_A), report=best,
        alignments=alignments, alternatives=reports[1:], n_accessible=len(refl),
    )


def _missing_cone_deg(theta_deg):
    """Convenience passthrough kept next to the planner for symmetry."""
    return missing_cone_half_angle_deg(theta_deg)
