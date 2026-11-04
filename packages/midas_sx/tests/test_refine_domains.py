"""One scale per crystal domain (PREREGISTER A23): exact recovery of the domain ratio and the structure."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_sx.refine import RefineModel, perturbed_crystal, refine  # noqa: E402
from test_refine import crystal_of, unique_hkl                      # noqa: E402


@pytest.fixture(scope="module")
def blocks():
    st = sim.structure_s2_bilayer()
    hkl, f2 = unique_hkl(st, 0.8)
    rng = np.random.default_rng(0)
    pick1 = rng.random(len(hkl)) < 0.7                            # domain 1 sees 70 % of the unique reflections, domain 2 a different 60 %
    pick2 = rng.random(len(hkl)) < 0.6
    H = np.vstack([hkl[pick1], hkl[pick2]])
    F2 = np.concatenate([3.0 * f2[pick1], 3.0 * 0.4286 * f2[pick2]])
    dom = np.concatenate([np.zeros(pick1.sum(), int), np.ones(pick2.sum(), int)])
    return st, H, F2, dom


def test_domain_ratio_and_structure_are_recovered_from_exact_f2(blocks):
    st, H, F2, dom = blocks
    sigma = 0.01 * F2.max() * np.ones(len(F2))
    truth = crystal_of(sim.structure_s2_bilayer())
    start = perturbed_crystal(truth, np.random.default_rng(1), shift_A=0.05)
    m = RefineModel(start, n_domains=2)
    assert m.names[-1] == "scale.d2" and m.scale_idx == [0, m.n_par - 1]
    res = refine(m, H, F2, sigma, domain=dom)
    assert res.converged, res.reason
    k1, k2 = res.x[0], res.x[-1]
    assert abs(k2 / k1 - 0.4286) < 2e-3 and abs(k1 - 3.0) < 0.02
    cell = np.array(st.cell[:3])
    for atom, row in zip(truth.atoms, res.absolute.itertuples()):
        assert (np.abs((np.array([row.x, row.y, row.z]) - np.array(atom.fract)) * cell)).max() < 2e-3
    assert res.goof < 0.5


def test_single_domain_model_is_unchanged_by_the_new_argument(blocks):
    st, H, F2, dom = blocks
    sel = dom == 0
    truth = crystal_of(sim.structure_s2_bilayer())
    start = perturbed_crystal(truth, np.random.default_rng(1), shift_A=0.05)
    sig = 0.01 * F2.max() * np.ones(sel.sum())
    a = refine(RefineModel(start), H[sel], F2[sel], sig)
    b = refine(RefineModel(start), H[sel], F2[sel], sig, domain=np.zeros(sel.sum(), int))
    assert np.allclose(a.x, b.x, atol=1e-10) and abs(a.chi2 - b.chi2) < 1e-9


def test_domain_labels_are_validated(blocks):
    st, H, F2, dom = blocks
    truth = crystal_of(sim.structure_s2_bilayer())
    sig = 0.01 * F2.max() * np.ones(len(F2))
    with pytest.raises(ValueError, match="needs a domain label"):
        refine(RefineModel(truth, n_domains=2), H, F2, sig)
    with pytest.raises(ValueError, match="domain must be one label"):
        refine(RefineModel(truth, n_domains=2), H, F2, sig, domain=dom + 2)
