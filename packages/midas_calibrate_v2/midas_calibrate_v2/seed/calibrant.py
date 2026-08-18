"""Single source of truth for calibrant lattice resolution.

Used by both :mod:`midas_calibrate_v2.seed.auto_seed` (the robust seeder's
ring-table generation) and :mod:`midas_calibrate_v2.pipelines.auto`
(``calibrate()``'s own ring predictions).  Having one shared
``CALIBRANTS`` dict and one ``resolve_calibrant`` means:

* Registering a custom material once (``CALIBRANTS['MyMaterial'] = {...}``)
  makes it available to ``calibrate()``, ``make_seed()``, and every code
  path in between — there used to be two independent registries that could
  (and did) drift apart.
* A custom calibrant dict resolves to the *same* lattice everywhere it is
  used, instead of ``make_seed`` honoring ``b``/``beta`` while
  ``calibrate()``'s own ring predictions silently forced ``b=a``/
  ``beta=alpha``.
* Defaulting/validation is aware of the crystal system implied by the
  space-group number, instead of blindly defaulting ``b=a``, ``c=a``,
  ``beta=alpha``, ``gamma=90`` regardless of symmetry.  A hexagonal/
  trigonal space group with a defaulted ``gamma=90`` silently builds a
  cubic metric tensor and returns a wrong-but-plausible ring table with no
  error — the crystal-system-aware defaults below make that impossible for
  the standard axis settings they support.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

# NIST SRM lattice constants.  Add new entries here — any material
# midas_hkls supports works, and registering here makes it available to
# both make_seed() and calibrate() (they share this dict object).
#
# WHY THE LATTICE CONSTANT DESERVES A SECOND LOOK
# -----------------------------------------------
# A relative error da/a in the assumed lattice constant is EXACTLY degenerate
# with a relative error dL/L in the sample-to-detector distance: both scale
# every predicted ring radius by the same factor.  Calibrating against a
# calibrant whose 'a' is wrong by 1e-4 hands you an Lsd wrong by 100 ppm and
# nothing in the residual will complain.  With two calibrants on one exposure
# the degeneracy is per-phase: an error in one phase's 'a' is indistinguishable
# from that powder sitting at a different position along the beam.
#
# ``srm`` records which certificate the value should be traced to.  ``a_sigma``
# is the 1-sigma uncertainty on ``a`` in Angstrom; it is left None here rather
# than guessed — fill it from the certificate for your lot, then
# :func:`lattice_uncertainty_lsd_ppm` converts it into the Lsd error it implies.
CALIBRANTS: Dict[str, Dict] = {
    "CeO2":  {"sg": 225, "a": 5.4116, "b": 5.4116, "c": 5.4116,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
               "srm": "NIST SRM 674b", "a_sigma": None},   # Fm-3m
    "LaB6":  {"sg": 221, "a": 4.1569, "b": 4.1569, "c": 4.1569,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
               "srm": "NIST SRM 660c", "a_sigma": None},   # Pm-3m
    "Si":    {"sg": 227, "a": 5.4310, "b": 5.4310, "c": 5.4310,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
               "srm": "NIST SRM 640d", "a_sigma": None},   # Fd-3m
    "Al2O3": {"sg": 167, "a": 4.7589, "b": 4.7589, "c": 12.9920,
               "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
               "srm": "NIST SRM 676a", "a_sigma": None},   # R-3c
}

#: Keys that describe provenance rather than the lattice itself.
_META_KEYS = ("srm", "a_sigma")


def lattice_uncertainty_lsd_ppm(a: float, a_sigma: float) -> float:
    """Lsd error (ppm) implied by a lattice-constant uncertainty.

    ``da/a`` and ``dLsd/Lsd`` enter the predicted ring radius identically, so
    the conversion is just ``1e6 * a_sigma / a``.  Quote this next to any Lsd
    you report: a calibration can be far more *precise* than the calibrant it
    is traceable to.
    """
    if a <= 0:
        raise ValueError(f"lattice constant must be positive; got {a}")
    return 1.0e6 * float(a_sigma) / float(a)

# Per-crystal-system constraint table: which lattice parameters are free
# (the caller must supply them; no safe default exists) and which are
# forced by symmetry (any value the caller supplies must match, else the
# dict describes an inconsistent lattice).  Mirrors
# ``midas_hkls.Lattice.for_system()``'s own constraint table so the two
# stay consistent.  Trigonal/hexagonal here means the standard
# hexagonal-axes setting (a=b, alpha=beta=90, gamma=120); rhombohedral-axes
# trigonal is not supported.
_SYSTEM_CONSTRAINTS = {
    "cubic":        {"free": ("a",), "forced": {"b": "a", "c": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "tetragonal":   {"free": ("a", "c"), "forced": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "trigonal":     {"free": ("a", "c"), "forced": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0}},
    "hexagonal":    {"free": ("a", "c"), "forced": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0}},
    "orthorhombic": {"free": ("a", "b", "c"), "forced": {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}},
    "monoclinic":   {"free": ("a", "b", "c", "beta"), "forced": {"alpha": 90.0, "gamma": 90.0}},
    "triclinic":    {"free": ("a", "b", "c", "alpha", "beta", "gamma"), "forced": {}},
}

_REL_TOL = 1e-4  # relative tolerance for "does the supplied value match the forced one"


def _canonical_name(calibrant: str) -> str:
    """Case-insensitive lookup: 'ceo2' / 'CeO2' / 'CEO2' → 'CeO2'."""
    lc = calibrant.lower().replace("-", "").replace(" ", "")
    for key in CALIBRANTS:
        if key.lower().replace("-", "") == lc:
            return key
    raise KeyError(f"Unknown calibrant {calibrant!r}. Known: {list(CALIBRANTS)}")


def _crystal_system(sg: int) -> str:
    from midas_hkls.tables import crystal_system_for
    return crystal_system_for(sg)


def resolve_calibrant(calibrant: Union[str, Dict]) -> dict:
    """Normalize a calibrant to a full, symmetry-consistent lattice spec.

    Accepts either a registered name (see ``CALIBRANTS``) or a dict
    describing an arbitrary powder calibrant.  Mirrors the
    ``Union[str, Dict]`` contract of :func:`midas_calibrate_v2.calibrate`.

    Dict form: ``a`` and ``sg`` are required.  Which of ``b``, ``c``,
    ``alpha``, ``beta``, ``gamma`` are required or optional depends on the
    crystal system implied by ``sg`` (e.g. hexagonal/trigonal requires
    ``c`` explicitly — it cannot be safely defaulted from ``a`` — and
    forces ``gamma=120``; cubic only requires ``a`` and forces everything
    else). Supplying a value that conflicts with what the crystal system
    forces (e.g. ``gamma=90`` for a hexagonal space group) raises
    ``ValueError`` rather than silently building the wrong lattice.

    Returns
    -------
    dict with keys ``name, sg, a, b, c, alpha, beta, gamma`` (``name`` is
    the canonical name for registered calibrants, else ``"<custom>"``).
    """
    if isinstance(calibrant, str):
        name = _canonical_name(calibrant)
        return {"name": name, **CALIBRANTS[name]}
    if isinstance(calibrant, (list, tuple)):
        raise TypeError(
            "resolve_calibrant() takes ONE calibrant; for a mixed-calibrant "
            "exposure (e.g. CeO2 + LaB6) use resolve_calibrants(), which "
            "returns one spec per phase.")
    if not isinstance(calibrant, dict):
        raise TypeError(
            f"calibrant must be a str name or a lattice dict, got {type(calibrant)}"
        )
    try:
        a = float(calibrant["a"])
        sg_raw = calibrant["sg"]
    except KeyError as e:
        raise ValueError(
            f"custom calibrant dict is missing required key {e}; "
            "required: 'a' (Å), 'sg' (space-group number). "
            "which of 'b', 'c', 'alpha', 'beta', 'gamma' are required "
            "depends on the crystal system implied by 'sg'."
        ) from None
    sg = int(sg_raw)
    if isinstance(sg_raw, float) and sg_raw != sg:
        raise ValueError(
            f"custom calibrant dict: space-group number must be an integer, "
            f"got sg={sg_raw!r}."
        )

    system = _crystal_system(sg)
    if system not in _SYSTEM_CONSTRAINTS:
        raise ValueError(f"unsupported crystal system {system!r} for sg={sg}")
    spec = _SYSTEM_CONSTRAINTS[system]

    values = {"a": a}
    for key in spec["free"]:
        if key == "a":
            continue
        if key not in calibrant:
            raise ValueError(
                f"custom calibrant dict: space group {sg} is {system}, which "
                f"requires {sorted(k for k in spec['free'] if k != 'a')!r} to "
                f"be given explicitly (no safe default exists) — missing {key!r}."
            )
        values[key] = float(calibrant[key])

    for key, forced in spec["forced"].items():
        forced_value = values[forced] if isinstance(forced, str) else forced
        if key in calibrant:
            supplied = float(calibrant[key])
            if abs(supplied - forced_value) > _REL_TOL * max(abs(forced_value), 1.0):
                raise ValueError(
                    f"custom calibrant dict: space group {sg} is {system}, which "
                    f"requires {key}={forced_value:g}; got {key}={supplied:g}. "
                    "Either omit it or set it to the required value. "
                    "(Rhombohedral-axes trigonal settings are not supported.)"
                )
        values[key] = forced_value

    # Boundary validation: catch physically impossible lattices here with a
    # clear message rather than letting them reach midas_hkls.Lattice, which
    # would raise a less contextual error (or, for a wrong metric tensor,
    # silently produce a bad d-spacing list).
    for _k in ("a", "b", "c"):
        if values[_k] <= 0.0:
            raise ValueError(
                f"custom calibrant dict: lattice length {_k}={values[_k]:g} Å "
                "must be positive."
            )
    for _k in ("alpha", "beta", "gamma"):
        if not 0.0 < values[_k] < 180.0:
            raise ValueError(
                f"custom calibrant dict: lattice angle {_k}={values[_k]:g}° "
                "must be in (0, 180)."
            )

    return {
        "name": str(calibrant.get("name", "<custom>")), "sg": sg,
        "a": values["a"], "b": values["b"], "c": values["c"],
        "alpha": values["alpha"], "beta": values["beta"], "gamma": values["gamma"],
        "srm": calibrant.get("srm"), "a_sigma": calibrant.get("a_sigma"),
    }


CalibrantSpec = Union[str, Dict]


def resolve_calibrants(
    calibrants: Union[CalibrantSpec, Sequence[CalibrantSpec]],
) -> List[dict]:
    """Normalise one calibrant, or several, to a list of lattice specs.

    Accepts anything :func:`resolve_calibrant` accepts, or a sequence of them
    for a mixed-calibrant exposure::

        resolve_calibrants("CeO2")                      # -> 1 spec
        resolve_calibrants(["CeO2", "LaB6"])            # -> 2 specs
        resolve_calibrants(["CeO2", {"a": 4.15, "sg": 221, "name": "LaB6ish"}])

    Names are made unique so the per-phase residual breakdown can label the
    rows unambiguously; duplicates get a ``#2``, ``#3`` suffix.
    """
    if isinstance(calibrants, (str, dict)):
        calibrants = [calibrants]
    specs = [resolve_calibrant(c) for c in calibrants]
    if not specs:
        raise ValueError("no calibrant given")
    seen: Dict[str, int] = {}
    for s in specs:
        base = s["name"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            s["name"] = f"{base}#{seen[base]}"
    return specs


def phases_from_calibrants(
    calibrants: Union[CalibrantSpec, Sequence[CalibrantSpec]],
) -> List[dict]:
    """Build the ``CalibrationParams.Phases`` list from calibrant specs."""
    return [{"name": s["name"], "sg": int(s["sg"]),
             "lattice": (s["a"], s["b"], s["c"],
                         s["alpha"], s["beta"], s["gamma"])}
            for s in resolve_calibrants(calibrants)]


__all__ = ["CALIBRANTS", "resolve_calibrant", "resolve_calibrants",
           "phases_from_calibrants", "lattice_uncertainty_lsd_ppm"]
