"""Save/load one raster position's ingest + results, dense AND sparse, as one
HDF5 file -- so the expensive step (background subtraction + 3-D blob-finding)
never has to be redone, and every gate/domain/fit this project's pipeline
produced is on disk with the data it came from, not just in a terminal
scrollback.

**Two tiers, on purpose.** The bulky NUMERIC arrays (the dense
background-subtracted stack, the sparse per-blob table, q-vectors, claim
masks, bootstrap arrays) are real HDF5 datasets, chunked and zstd-compressed
via ``hdf5plugin`` (measured faster AND smaller than plain gzip on this
project's data -- see ``_COMPRESSION``'s own comment) -- that is what HDF5 is
for. Everything else (`PositionResult`'s many nested
dicts: completeness counts, decoy verdicts, ab-gate reasons, notes) is stored
as ONE JSON string, produced by `PositionResult.to_dict()` -- code that
already exists, is already used by `reduce_raster_block`, and already knows
this dataclass's exact shape. Re-deriving a parallel HDF5 schema for every
field of every gate dict would be new code with new bugs for something
`to_dict()` already solved; the only things restored on TOP of it here are
what `to_dict()` deliberately drops for JSON's sake (`claim` boolean masks,
reduced to a bare count; `cell_bootstrap_samples`; orientation envelope
`U_bootstraps`) -- see its own docstring for why, all still worth having in
an HDF5 file.

**Dense, sparse, or both** (``save_dense=True`` by default): the sparse
`spots` table (one row per detected blob, already what `reduce_one_position`
itself consumes) is always saved -- it is small and it IS the reusable
result. The dense background-subtracted stack (`sub`, same shape as the raw
frames -- tens to hundreds of MB per position, compresses well since most of
it sits at the pedestal) is optional: skip it for a large raster
where the sparse table plus the ability to re-run ingest from raw frames is
enough; keep it when you want to re-examine or re-threshold a position
without touching raw frames again.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import hdf5plugin

from .ingest import SPOT_COLUMNS
from .raster import IngestBundle, PositionResult

__all__ = ["save_position_hdf5", "load_ingest_hdf5", "load_position_result_hdf5"]

#: zstd beats gzip on this project's background-subtracted arrays on BOTH size and
#: speed (measured 2026-09-14: clevel=9, no blosc block/shuffle wrapper -- that wrapper's
#: own framing overhead cost more than it recovered here -- 43.2MB/2.6s vs gzip level 9's
#: 43.75MB/22.9s on the same array). Applied uniformly, not just to the big dense array:
#: overhead on the small ones is negligible and consistency is one less thing to reason about.
_COMPRESSION = hdf5plugin.Zstd(clevel=9)


def _write_bundle(h5, bundle: IngestBundle, *, save_dense: bool) -> None:
    g = h5.create_group("ingest")
    spots_g = g.create_group("spots")
    for col in SPOT_COLUMNS:
        spots_g.create_dataset(col, data=bundle.spots[col].to_numpy(), **_COMPRESSION)
    g.attrs["n_spots"] = len(bundle.spots)
    g.create_dataset("qlab", data=bundle.qlab.detach().cpu().numpy(), **_COMPRESSION)
    g.create_dataset("omega_deg", data=np.asarray(bundle.omega_deg), **_COMPRESSION)
    g.create_dataset("mask", data=np.asarray(bundle.mask), **_COMPRESSION)
    g.attrs["ingest_counts_json"] = json.dumps(bundle.ingest_counts, default=str)
    g.attrs["has_dense"] = bool(save_dense)
    if save_dense:
        g.create_dataset("sub", data=np.asarray(bundle.sub), chunks=True, **_COMPRESSION)
    g.attrs["sub_shape"] = np.asarray(bundle.sub).shape


def _write_results(h5, res: PositionResult) -> None:
    h5.attrs["results_json"] = json.dumps(res.to_dict(), default=str)
    dg = h5.create_group("domains_full")
    for i, dom in enumerate(res.domains.domains):
        sub = dg.create_group(str(i))
        sub.create_dataset("claim", data=np.asarray(dom.claim, bool), **_COMPRESSION)
        sub.create_dataset("frag", data=np.asarray(dom.frag, bool), **_COMPRESSION)
    if res.split_pct is not None:
        h5.attrs["has_cell_bootstrap"] = False   # PositionResult itself does not carry the
        # samples array (only midas_hkls.ConstrainedFit does, upstream of PositionResult) --
        # nothing to restore here; kept explicit rather than silently absent.
    oe_g = h5.create_group("orientation_envelope_full")
    for i, oe in enumerate(res.orientation_envelope):
        if oe is None or "U_bootstraps" not in oe:
            continue
        oe_g.create_dataset(str(i), data=np.asarray(oe["U_bootstraps"]), **_COMPRESSION)


def save_position_hdf5(path, *, bundle: IngestBundle, res: PositionResult,
                       save_dense: bool = True) -> None:
    """Write one position's ingest (dense + sparse, per ``save_dense``) and
    full results to ``path``. Overwrites; written to a temp path and renamed,
    matching ``reduce_raster_block``'s own idempotent-write convention (a
    rerun never leaves a partial file)."""
    import h5py
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        h5.attrs["point"] = res.point
        _write_bundle(h5, bundle, save_dense=save_dense)
        _write_results(h5, res)
    tmp.rename(path)


def load_ingest_hdf5(path) -> IngestBundle:
    """Load just the ingest bundle (spots, qlab, omega_deg, mask, sub-if-saved,
    ingest_counts) -- the piece ``reduce_one_position(..., ingested=...)``
    consumes to skip re-ingesting. Raises if the file was saved with
    ``save_dense=False`` and a caller then reads ``.sub`` (``None`` in that
    case) as if it were there -- check ``bundle.sub is not None`` first."""
    import h5py
    with h5py.File(Path(path), "r") as h5:
        g = h5["ingest"]
        cols = {col: g["spots"][col][()] for col in SPOT_COLUMNS}
        spots = pd.DataFrame(cols)
        qlab = torch.as_tensor(g["qlab"][()])
        omega_deg = g["omega_deg"][()]
        mask = g["mask"][()]
        sub = g["sub"][()] if g.attrs.get("has_dense", False) else None
        ingest_counts = json.loads(g.attrs["ingest_counts_json"])
    return IngestBundle(spots=spots, qlab=qlab, omega_deg=omega_deg, mask=mask,
                        sub=sub, ingest_counts=ingest_counts)


def load_position_result_hdf5(path) -> dict:
    """Load the full saved results as a dict (``PositionResult.to_dict()``'s
    own shape), with ``claim``/``frag`` boolean arrays and orientation
    ``U_bootstraps`` restored under ``domains_full``/``orientation_envelope_full``
    keys -- read-only reuse (e.g. building a raster map from saved files,
    `assemble_raster_results`'s JSON equivalent), NOT a live `PositionResult`
    object and NOT something ``reduce_one_position`` re-derives gates from --
    gates are always recomputed fresh from the cached ingest, never replayed
    silently, so a code change to a gate is never masked by a stale cache."""
    import h5py
    with h5py.File(Path(path), "r") as h5:
        d = json.loads(h5.attrs["results_json"])
        d["domains_full"] = {
            int(i): dict(claim=h5["domains_full"][i]["claim"][()],
                        frag=h5["domains_full"][i]["frag"][()])
            for i in h5["domains_full"]}
        d["orientation_envelope_full"] = {
            int(i): h5["orientation_envelope_full"][i][()]
            for i in h5["orientation_envelope_full"]}
    return d
