"""Declared conventions for voxel products, and a loader that refuses without one.

Two demk products of the *same experiment* disagree about how to read them:
``all_labels_qvox.npz`` needs the raw ``Grains.csv`` matrix as given, the
``demk_g1592_9r`` ladder fixture needs its transpose, and ``cyl_n9R.npz`` uses the
opposite omega sign to the scripts that consume it. Nothing in any of those files
says so. A docstring stated one rule universally and was wrong for the others.

That is the error class behind the retracted "secondary {111}" result
(``LAB_NOTEBOOK.md`` R5): every contradictory result in that campaign came from an
ad-hoc script carrying its own private convention.

The fix is not a better docstring. It is that **a voxel product must declare how to
read it, in a file that travels with it**, and the loader must refuse to guess.

Conventions live in a sidecar ``<product>.convention.json`` so nothing has to be
rewritten and the declaration is git-trackable next to the data it describes.

    from midas_defect.provenance import QConvention, stamp, load_cloud

    stamp("cloud.npz", QConvention(
        omega_map="180 - 0.25*frame", omega_rotation="Rz(-omega)",
        orientation="OM", q_units="2pi/d", source="extract_all_label_qvox.py"))

    cloud, conv = load_cloud("cloud.npz")        # raises if undeclared

`assert_matches_data` then checks the declaration against the data rather than
trusting it, using the same on-lattice test as
:func:`midas_defect.bragg_diffuse.check_orientation_convention`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["QConvention", "UndeclaredConventionError", "ConventionMismatch",
           "stamp", "read_convention", "load_cloud", "assert_matches_data",
           "sidecar_path", "file_digest"]

SCHEMA = 1


class UndeclaredConventionError(RuntimeError):
    """A voxel product was loaded with no declared convention."""


class ConventionMismatch(AssertionError):
    """The declared convention disagrees with what the data says."""


@dataclass(frozen=True)
class QConvention:
    """How to read a q-space voxel product.

    Every field is required and none has a default that could silently be wrong.

    Parameters
    ----------
    omega_map
        How omega is computed from the frame index, as a literal expression,
        e.g. ``"180 - 0.25*frame"``. Two demk products differ here.
    omega_rotation
        Which rotation takes lab to sample, e.g. ``"Rz(-omega)"``.
    orientation
        Which sense of a MIDAS ``Grains.csv`` matrix reproduces this cloud's
        lattice: ``"OM"`` (as given) or ``"OM.T"``. **Not a universal property** —
        it belongs to the cloud, and the two demk products disagree.
    q_units
        ``"2pi/d"`` or ``"1/d"``.
    source
        The script or command that produced the file.
    notes
        Anything a reader would otherwise have to reverse-engineer.
    """
    omega_map: str
    omega_rotation: str
    orientation: str
    q_units: str
    source: str
    notes: str = ""
    schema: int = field(default=SCHEMA)

    def __post_init__(self) -> None:
        if self.orientation not in ("OM", "OM.T", "n/a"):
            raise ValueError(f"orientation must be 'OM', 'OM.T' or 'n/a', "
                             f"got {self.orientation!r}")
        if self.q_units not in ("2pi/d", "1/d"):
            raise ValueError(f"q_units must be '2pi/d' or '1/d', got {self.q_units!r}")


def sidecar_path(product: str | Path) -> Path:
    p = Path(product)
    return p.with_suffix(p.suffix + ".convention.json")


def file_digest(product: str | Path, *, limit_mb: float = 64.0) -> str:
    """md5 of the file, or of a directory's sorted (name, size) listing.

    Zarr products are directories of many chunks; hashing all of them is slow and
    buys nothing over the structural listing, so directories get the listing.
    Large files are hashed over their first ``limit_mb`` plus their size, which is
    enough to catch a swapped or truncated input.
    """
    p = Path(product)
    h = hashlib.md5()
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f"{f.relative_to(p)}:{f.stat().st_size}\n".encode())
        return "dir-" + h.hexdigest()
    n = p.stat().st_size
    h.update(str(n).encode())
    with p.open("rb") as fh:
        h.update(fh.read(int(limit_mb * 1024 * 1024)))
    return h.hexdigest()


def stamp(product: str | Path, convention: QConvention, *,
          overwrite: bool = False) -> Path:
    """Write the sidecar declaring how to read ``product``."""
    p = Path(product)
    if not p.exists():
        raise FileNotFoundError(p)
    sc = sidecar_path(p)
    if sc.exists() and not overwrite:
        raise FileExistsError(f"{sc} exists; pass overwrite=True to replace it")
    payload = asdict(convention)
    payload["product"] = p.name
    payload["digest"] = file_digest(p)
    sc.write_text(json.dumps(payload, indent=2) + "\n")
    return sc


def read_convention(product: str | Path, *, check_digest: bool = True) -> QConvention:
    """Return the declared convention, or raise."""
    p = Path(product)
    sc = sidecar_path(p)
    if not sc.exists():
        raise UndeclaredConventionError(
            f"{p.name} has no declared convention ({sc.name} missing).\n"
            f"Two demk products of the same experiment need OPPOSITE conventions, "
            f"so this cannot be guessed. Stamp it with provenance.stamp(), or use "
            f"bragg_diffuse.check_orientation_convention to have the data decide.")
    d = json.loads(sc.read_text())
    if check_digest and "digest" in d:
        now = file_digest(p)
        if now != d["digest"]:
            raise ConventionMismatch(
                f"{p.name} has changed since it was stamped "
                f"(digest {now} != {d['digest']}). The declared convention may no "
                f"longer describe it; re-stamp deliberately.")
    return QConvention(**{k: v for k, v in d.items()
                          if k in QConvention.__dataclass_fields__})


def load_cloud(product: str | Path, *, check_digest: bool = True):
    """Load an ``.npz`` voxel product together with its declared convention."""
    p = Path(product)
    conv = read_convention(p, check_digest=check_digest)
    return np.load(p), conv


def assert_matches_data(q, intensity, orientations, crystal,
                        convention: QConvention, *, min_margin: float = 3.0):
    """Check the DECLARED orientation sense against what the data says.

    A declaration is only worth having if it is checked. Raises
    `ConventionMismatch` when the data decisively contradicts it; returns the
    verdict otherwise (including when the data cannot decide, e.g. a
    satellite-only cloud, where a declaration is all you have).
    """
    from .bragg_diffuse import check_orientation_convention

    v = check_orientation_convention(q, intensity, orientations, crystal,
                                     min_margin=min_margin)
    if v.decisive and convention.orientation in ("OM", "OM.T") \
            and v.convention != convention.orientation:
        raise ConventionMismatch(
            f"declared orientation={convention.orientation} but the data says "
            f"{v.convention}: {v.note}")
    return v
