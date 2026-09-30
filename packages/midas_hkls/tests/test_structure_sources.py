"""Tests for midas_hkls.io.structure_sources (offline; synthetic CIFs and a fake fetch)."""
import json

from midas_hkls.io.structure_sources import (
    StructureRecord,
    cod_download,
    cod_search,
    index_cif_directory,
    search_index,
    select_entry,
)

CIF = """data_{id}
_chemical_formula_sum '{formula}'
_space_group_IT_number {sg}
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {c}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma {gamma}
{extra}
loop_
_atom_site_label
_atom_site_fract_x
X1 0.0
"""


def _write(tmp_path, i, formula, sg, a, c, gamma=90, extra=""):
    (tmp_path / f"data_{i}-X.cif").write_text(CIF.format(id=i, formula=formula, sg=sg, a=a, c=c,
                                                         gamma=gamma, extra=extra))


def test_index_and_search(tmp_path):
    _write(tmp_path, 1, "Aa1 Bb2", 194, "4.83", "7.86", 120)
    _write(tmp_path, 2, "Aa1 Bb2 Cc1", 194, "4.90", "8.00", 120)
    _write(tmp_path, 3, "Aa1 Dd1", 225, "4.4(1)", "4.4")
    out = tmp_path / "idx.csv"
    assert index_cif_directory(str(tmp_path), str(out), label="t", nproc=1) == 3
    hits = search_index(str(out), elements_required=["Aa"], space_group=194, max_elements=2)
    assert [h.id for h in hits] == ["1"]
    hits = search_index(str(out), elements_required=["Aa"], elements_allowed=["Aa", "Dd"])
    assert [h.id for h in hits] == ["3"] and hits[0].a == "4.4"          # esd stripped
    assert search_index(str(out), elements_required=["Aa"], a_range=(4.85, 5.0))[0].id == "2"


def test_select_entry_rule_is_applied_and_recorded():
    recs = [StructureRecord("x", "1", a="4.47", T="295", year="1990"),
            StructureRecord("x", "2", a="4.4691", year="1963"),
            StructureRecord("x", "3", a="4.46912", T="1200", year="2010"),       # not ambient
            StructureRecord("x", "4", a="4.469123", year="1930")]               # too old
    best = select_entry(recs)
    # the entry with a STATED ambient temperature wins over a more precise unstated one
    assert best.id == "1" and "rule:" in best.selection_note and "2 of 4" in best.selection_note
    assert select_entry([recs[1]]).id == "2"
    assert select_entry([StructureRecord("x", "9", T="900")]) is None
    # consensus: an outlier cell (e.g. under pressure, pressure unstated) is not chosen on digits
    cons = [StructureRecord("x", str(i), a=a) for i, a in enumerate(["2.866", "2.8665", "2.8664", "2.833513"])]
    assert select_entry(cons).id == "1"
    # densest cluster, not median: off-ambient spread drags the median below the ambient cluster
    spread = ["2.83", "2.8596", "2.8619", "2.887", "2.9064", "2.915", "2.916", "2.9239", "2.924",
              "2.93664", "2.9503", "2.9504", "2.9506", "2.9508", "2.951", "2.95111"]
    pick = select_entry([StructureRecord("x", str(i), a=a) for i, a in enumerate(spread)])
    assert abs(float(pick.a) - 2.9507) < 0.002


def test_cod_search_and_download_with_fake_fetch(tmp_path):
    payload = [{"file": 111, "formula": "- Aa Bb -", "sgNumber": 225, "a": "4.4", "year": 2001},
               {"file": 222, "formula": "- Aa Bb -", "sgNumber": 221, "a": "3.1"}]
    seen = []

    def fetch(url):
        seen.append(url)
        return json.dumps(payload).encode() if "format=json" in url else b"data_x\n"

    recs = cod_search(["Aa", "Bb"], space_group=225, fetch=fetch)
    assert [r.id for r in recs] == ["111"] and recs[0].formula == "Aa Bb"
    assert "strictmin=2" in seen[0]
    r = cod_download(recs[0], str(tmp_path), fetch=fetch)
    assert r.path.endswith("cod_111.cif") and open(r.path).read().startswith("data_x")
