"""A low residual strain must not read as "ok" when it rests on no data.

The strain gate is the number a beamline judges a calibration by (the working
rule is <100 ustrain). It was computed from the residual alone, with no regard
for how many rings produced that residual — and a fit with as many free
parameters as constraints can drive its residual to zero at an arbitrary
geometry.

MEASURED: a synthetic CeO2 image whose auto-seeder locked onto a SINGLE ring
converged at 11.92 ustrain with Lsd wrong by a factor of 7.9 (300 000 µm true,
2 362 886 µm fitted). The old gate returned "ok". The ring count was already
known — ``cross_validation`` said "too few rings (1)" in the same run — but the
headline number did not inherit that caution.
"""

from __future__ import annotations

import pytest

from midas_calibrate_v2.pipelines.diagnostics import (
    strain_cap_check, run_all_gates, worst_severity, n_rings_from_fits,
)


class _Iter:
    def __init__(self, strain):
        self.mean_strain_uE = strain


@pytest.mark.parametrize("n_rings", [1, 2])
def test_low_strain_on_too_few_rings_is_not_ok(n_rings):
    d = strain_cap_check([_Iter(11.92)], n_rings=n_rings)
    assert d.severity == "warn", (
        f"{n_rings} ring(s) at 11.92 ustrain reported {d.severity!r}; a low "
        f"strain on this little support is not evidence of a good geometry"
    )
    assert "ring" in d.message
    assert d.metrics["n_rings"] == float(n_rings)


@pytest.mark.parametrize("n_rings", [3, 6, 20])
def test_low_strain_on_enough_rings_is_ok(n_rings):
    d = strain_cap_check([_Iter(11.92)], n_rings=n_rings)
    assert d.severity == "ok"
    assert f"{n_rings} rings" in d.message


def test_ring_count_is_optional_and_preserves_old_behaviour():
    """Callers that cannot supply a ring count still get the old verdict."""
    assert strain_cap_check([_Iter(11.92)]).severity == "ok"
    assert strain_cap_check([_Iter(75.0)]).severity == "warn"
    assert strain_cap_check([_Iter(900.0)]).severity == "fail"


def test_a_high_strain_still_fails_regardless_of_ring_count():
    """The support check must not be able to downgrade a real failure."""
    for n in (1, 2, 3, 50):
        assert strain_cap_check([_Iter(900.0)], n_rings=n).severity == "fail"


def test_ring_count_is_derived_from_the_fits():
    """The wiring, not just the gate.

    This failed silently once already: ``numpy`` is imported per-function in
    this module, not at module scope, so a bare ``np.unique`` raised NameError
    inside a blanket ``except`` and the count came back None — the gate kept
    reporting "ok" while looking correct in isolation.
    """
    import numpy as np
    import torch

    class _Fits:
        def __init__(self, idx):
            self.ring_idx = idx

    assert n_rings_from_fits(None) is None
    assert n_rings_from_fits(object()) is None
    assert n_rings_from_fits(_Fits(torch.tensor([0, 0, 0, 0]))) == 1
    assert n_rings_from_fits(_Fits(torch.tensor([0, 1, 1, 2, 2, 2]))) == 3
    # must not depend on it being a torch tensor
    assert n_rings_from_fits(_Fits(np.array([3, 3, 4]))) == 2


def test_the_gate_and_the_count_compose():
    """A one-ring dataset must come out of the pair as a warning."""
    import torch

    class _Fits:
        ring_idx = torch.tensor([0, 0, 0, 0])

    n = n_rings_from_fits(_Fits())
    d = strain_cap_check([_Iter(11.92)], n_rings=n)
    assert n == 1
    assert d.severity == "warn"
    assert worst_severity([d]) == "warn"
