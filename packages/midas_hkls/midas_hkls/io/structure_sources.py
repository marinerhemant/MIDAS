"""Find candidate crystal structures, with provenance, from a local CIF directory
or the Crystallography Open Database (COD).

A phase comparison is only fair if every candidate's cell comes from the same
kind of source under the same selection rule. This module therefore returns
:class:`StructureRecord` objects that carry *where* a structure came from and
*why* it was chosen, and applies one rule (:func:`select_entry`) to all.

Local directories (e.g. a licensed database export) are indexed once by
:func:`index_cif_directory` (header fields only) and searched with
:func:`search_index`. COD is queried over HTTPS by :func:`cod_search` and
entries are fetched with :func:`cod_download`; both take an injectable
``fetch`` callable so they can be tested without a network.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["StructureRecord", "parse_cif_header", "index_cif_directory", "search_index",
           "cod_search", "cod_download", "select_entry"]

_KEYS = {
    "_chemical_formula_sum": "formula",
    "_space_group_IT_number": "sg", "_symmetry_Int_Tables_number": "sg",
    "_space_group_name_H-M_alt": "hm", "_symmetry_space_group_name_H-M": "hm",
    "_cell_length_a": "a", "_cell_length_b": "b", "_cell_length_c": "c",
    "_cell_angle_alpha": "alpha", "_cell_angle_beta": "beta", "_cell_angle_gamma": "gamma",
    "_diffrn_ambient_temperature": "T", "_cell_measurement_temperature": "T",
    "_diffrn_ambient_pressure": "P", "_cell_measurement_pressure": "P",
    "_journal_year": "year", "_citation_year": "year",
}
_COLS = ["source", "id", "path", "formula", "sg", "hm", "a", "b", "c", "alpha", "beta", "gamma",
         "T", "P", "year"]


@dataclass
class StructureRecord:
    source: str            # "local:<label>" or "cod"
    id: str
    path: str = ""
    formula: str = ""
    sg: str = ""
    hm: str = ""
    a: str = ""
    b: str = ""
    c: str = ""
    alpha: str = ""
    beta: str = ""
    gamma: str = ""
    T: str = ""
    P: str = ""
    year: str = ""
    selection_note: str = ""

    def elements(self) -> set:
        return set(re.findall(r"[A-Z][a-z]?", self.formula or ""))

    def to_dict(self) -> dict:
        return asdict(self)


def _strip(v: str) -> str:
    return re.sub(r"\(.*?\)", "", v.strip().strip("'\""))


def parse_cif_header(path: str) -> dict:
    """Header fields of one CIF (stops at the atom-site loop)."""
    r: dict = {}
    pending = None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("_atom_site_"):
                    break
                if pending:
                    if s and not s.startswith("_") and s != ";":
                        r.setdefault(pending, _strip(s))
                    pending = None
                    continue
                for k, v in _KEYS.items():
                    if s == k or s.startswith(k + " ") or s.startswith(k + "\t"):
                        val = s[len(k):].strip()
                        if val:
                            r.setdefault(v, _strip(val))
                        else:
                            pending = v
                        break
    except OSError:
        pass
    return r


def _row(args):
    label, path = args
    h = parse_cif_header(path)
    m = re.search(r"(\d+)", os.path.basename(path))
    return {"source": f"local:{label}", "id": m.group(1) if m else os.path.basename(path),
            "path": path, **h}


def index_cif_directory(cif_dir: str, out_csv: str, label: str = "local", nproc: int = 8) -> int:
    """Index every ``*.cif`` in ``cif_dir`` into ``out_csv``; returns the count."""
    files = sorted(os.path.join(cif_dir, x) for x in os.listdir(cif_dir) if x.lower().endswith(".cif"))
    jobs = [(label, f) for f in files]
    if nproc > 1 and len(jobs) > 1000:
        with Pool(nproc) as p:
            rows = p.map(_row, jobs, chunksize=500)
    else:
        rows = [_row(j) for j in jobs]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def search_index(index_csv: str, *, elements_required: Iterable[str] = (),
                 elements_allowed: Optional[Iterable[str]] = None, space_group: Optional[int] = None,
                 max_elements: Optional[int] = None, a_range=None, c_range=None
                 ) -> List[StructureRecord]:
    """Filter an index: all required elements present, no element outside
    ``elements_allowed`` (if given), optional space group, element count, cell ranges."""
    req = set(elements_required)
    allowed = set(elements_allowed) if elements_allowed is not None else None
    out = []
    with open(index_csv) as f:
        for r in csv.DictReader(f):
            rec = StructureRecord(**{k: (r.get(k) or "") for k in _COLS})
            els = rec.elements()
            if not req <= els:
                continue
            if allowed is not None and not els <= allowed:
                continue
            if max_elements is not None and len(els) > max_elements:
                continue
            if space_group is not None and rec.sg.strip() != str(space_group):
                continue
            a, c = _f(rec.a), _f(rec.c)
            if a_range and (a is None or not a_range[0] <= a <= a_range[1]):
                continue
            if c_range and (c is None or not c_range[0] <= c <= c_range[1]):
                continue
            out.append(rec)
    return out


COD_URL = "https://www.crystallography.net/cod"


def _http_get(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes:
    """GET with retries (transient TLS / connection drops are common on public APIs)."""
    import time
    import urllib.error
    import urllib.request
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if k == attempts - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise RuntimeError("unreachable")


def cod_search(elements: Sequence[str], *, exact: bool = True, space_group: Optional[int] = None,
               fetch: Callable[[str], bytes] = _http_get) -> List[StructureRecord]:
    """Query COD for entries containing ``elements`` (exactly these if ``exact``)."""
    q = "&".join(f"el{i + 1}={e}" for i, e in enumerate(elements))
    if exact:
        q += f"&strictmin={len(elements)}&strictmax={len(elements)}"
    data = json.loads(fetch(f"{COD_URL}/result?{q}&format=json").decode())
    out = []
    for e in data:
        rec = StructureRecord(source="cod", id=str(e.get("file", "")),
                              formula=(e.get("formula") or "").strip("- "),
                              sg=str(e.get("sgNumber") or ""), hm=e.get("sg") or "",
                              a=str(e.get("a") or ""), b=str(e.get("b") or ""), c=str(e.get("c") or ""),
                              alpha=str(e.get("alpha") or ""), beta=str(e.get("beta") or ""),
                              gamma=str(e.get("gamma") or ""),
                              T=str(e.get("celltemp") or e.get("diffrtemp") or ""),
                              P=str(e.get("cellpressure") or e.get("diffrpressure") or ""),
                              year=str(e.get("year") or ""))
        if space_group is None or rec.sg == str(space_group):
            out.append(rec)
    return out


def cod_download(rec: StructureRecord, dest_dir: str,
                 fetch: Callable[[str], bytes] = _http_get) -> StructureRecord:
    """Download a COD entry's CIF into ``dest_dir``; returns the record with ``path`` set."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"cod_{rec.id}.cif")
    with open(path, "wb") as f:
        f.write(fetch(f"{COD_URL}/{rec.id}.cif"))
    rec.path = path
    return rec


def _sig_digits(x: str) -> int:
    x = (x or "").split("(")[0]
    return len(x.split(".")[1]) if "." in x else 0


def select_entry(records: Sequence[StructureRecord], *, t_range=(273.0, 313.0),
                 max_pressure_kpa: float = 200.0, min_year: int = 1947,
                 consensus_tol: Optional[float] = 0.003) -> Optional[StructureRecord]:
    """One rule for every candidate: ambient (temperature within ``t_range`` or not
    stated; pressure not above ``max_pressure_kpa`` or not stated), not older than
    ``min_year`` when a year is stated (older cells may be in kX units), then the
    most significant digits in ``a``, then the most recent year. The reason is
    written into ``selection_note``."""
    ok = []
    for r in records:
        T, P, y = _f(r.T), _f(r.P), _f(r.year)
        if T is not None and not t_range[0] <= T <= t_range[1]:
            continue
        if P is not None and P > max_pressure_kpa:
            continue
        if y is not None and y < min_year:
            continue
        ok.append(r)
    if not ok:
        return None
    n_elig = len(ok)
    # Consensus on the cell: headers often omit temperature and pressure, so high-T,
    # high-P and computed entries survive the filters above. The centre is the DENSEST
    # cluster of a-values (most entries within ``consensus_tol``), not the median: a
    # spread of off-ambient entries drags the median away from the ambient cluster
    # (seen: an hcp element whose median sat 0.3 % below its tight ambient cluster).
    if consensus_tol is not None:
        a_vals = np.array([_f(r.a) for r in ok if _f(r.a) is not None])
        if a_vals.size >= 3:
            counts = np.array([np.sum(np.abs(a_vals / x - 1) <= consensus_tol) for x in a_vals])
            members = a_vals[np.abs(a_vals / a_vals[int(np.argmax(counts))] - 1) <= consensus_tol]
            centre = float(np.median(members))
            near = [r for r in ok if _f(r.a) is not None and abs(_f(r.a) / centre - 1) <= consensus_tol]
            ok = near or ok
    # Entries that STATE an ambient temperature rank first: an unstated temperature or
    # pressure is common in database headers and hides high-pressure, high-temperature
    # and computed entries (seen: a bcc cell 1.1 % below ambient won on digits alone).
    best = max(ok, key=lambda r: (_f(r.T) is not None, _sig_digits(r.a), _f(r.year) or 0))
    stated = sum(_f(r.T) is not None for r in ok)
    best.selection_note = (f"rule: ambient T {t_range} K or unstated, P <= {max_pressure_kpa} kPa or "
                           f"unstated, year >= {min_year} or unstated; stated ambient T first, then most "
                           f"digits in a, then newest; within {consensus_tol} of the densest a-cluster; "
                           f"{n_elig} of {len(records)} eligible, {len(ok)} in consensus, "
                           f"{stated} with stated T")
    return best
