"""Seed ring diagnostic: midas_pipeline.diagnostics.seed_ring.

The numbers are the Li6PS5Cl run of 2026-10-29: per-ring mean |F|^2 from ICSD 418490 (Deiseroth 2008), relative to the
strongest searched ring, and the ring spot counts of S2 70wt 8kg layer 575. That run seeded from ring 8, a 1 % reflection.
"""
import json
import numpy as np

from midas_pipeline.diagnostics.seed_ring import advise_seed_ring, report_seed_ring, ring_seed_table

F2 = {1: .128, 2: .099, 3: .433, 4: .368, 5: .846, 7: .0456, 8: .0098, 9: .2986, 10: .2043, 11: 1.0}
SPOTS = {1: 84011, 2: 20575, 3: 152507, 4: 202611, 5: 145176, 7: 6380, 8: 1836, 9: 392603, 10: 83010, 11: 159713}


def _table():
    return [{"ring": r, "n_reflections": 8, "f2_rel": F2[r], "n_spots": SPOTS[r], "n_seeds_est": SPOTS[r] // 2} for r in F2]


def _write(layer, f2=True, spots=None, rings=None):
    rings = rings or {1: 40, 2: 10, 3: 60, 8: 8}
    rows = ["h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius" + (" F2" if f2 else "")]
    for r in sorted(rings):
        for k in range(2):
            rows.append(f"{r} 0 0 1.0 {r} 0 0 0 1 2 3" + (f" {0.01 if r == 8 else 0.5 + 0.1 * k}" if f2 else ""))
    (layer / "hkls.csv").write_text("\n".join(rows) + "\n")
    ia = ["YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta DetID"]
    sid = 0
    for r, n in sorted(rings.items()):
        for k in range(n):
            sid += 1
            ia.append(f"1 1 {-170 + 340 * k / max(n, 1)} 5 {sid} {r} 0 1 1")
    ia += ["0 0 0 0 0 0 0 0 1"] * 7                                     # ring-0 placeholder rows
    (layer / "InputAll.csv").write_text("\n".join(ia) + "\n")


def test_table_counts_and_relative_f2(tmp_path):
    _write(tmp_path)
    t = {r["ring"]: r for r in ring_seed_table(tmp_path / "hkls.csv", tmp_path / "InputAll.csv")}
    assert set(t) == {1, 2, 3, 8}                                        # ring 0 placeholders are ignored
    assert t[3]["n_spots"] == 60 and t[3]["n_seeds_est"] == 30
    assert t[3]["f2_rel"] == 1.0 or t[3]["f2_rel"] > t[1]["f2_rel"]      # strongest ring has the largest relative F2
    assert t[8]["f2_rel"] < 0.05


def test_weak_seed_ring_is_flagged_and_ring_2_recommended():
    a = advise_seed_ring(_table(), current=8)
    assert a["warn"] and a["recommended"] == 2                           # the ring that gave 14x more real grains
    assert "OverAllRingToIndex 2" in a["message"] and "heuristic" in a["message"]


def test_a_strong_seed_ring_is_not_flagged():
    a = advise_seed_ring(_table(), current=2)
    assert not a["warn"] and a["message"] == ""


def test_seed_budget_excludes_crowded_rings():
    # ring 5 has the second largest F2 but ~72k seeds: over the budget, never recommended
    assert advise_seed_ring(_table(), current=8, max_seeds=20000)["recommended"] != 5
    assert advise_seed_ring(_table(), current=8, max_seeds=100000)["recommended"] in (11, 5)


def test_no_f2_column_recommends_nothing_and_says_how(tmp_path):
    _write(tmp_path, f2=False)
    t = ring_seed_table(tmp_path / "hkls.csv", tmp_path / "InputAll.csv")
    assert all(r["f2_rel"] is None for r in t)
    a = advise_seed_ring(t, current=8)
    assert a["recommended"] is None and not a["warn"] and "PhaseCIF" in a["message"]


def test_report_writes_json_calibrates_seed_fraction_and_never_raises(tmp_path):
    _write(tmp_path)
    (tmp_path / "SpotsToIndex.csv").write_text("\n".join(str(i) for i in range(4)) + "\n")   # 4 of the ring-8 spots are seeds
    adv = report_seed_ring(tmp_path, 8)
    assert adv["warn"] and json.loads((tmp_path / "seed_ring.json").read_text())["current"] == 8
    t = {r["ring"]: r for r in adv["table"]}
    assert t[8]["n_seeds_est"] == 4 and t[3]["n_seeds_est"] == 30        # fraction 4/8 = 0.5 measured on the ring used
    assert "error" in report_seed_ring(tmp_path / "missing", 8)          # a missing layer is a log line, not an exception
