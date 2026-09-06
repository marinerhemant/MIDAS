"""Split-tolerance and one-to-one matching in SpotAssigner.

Three defects in the isotropic path are pinned here as the reason the split
exists: it conflates physically different errors, it is mis-weighted between
channels, and it does not wrap the periodic ones.
"""
from __future__ import annotations

import math
import torch
import pytest

from midas_diffract.losses import SpotAssigner

VALID = torch.ones(1, 3)


def _obs(rows):
    return torch.tensor(rows, dtype=torch.float64)


def _pred(rows):
    t = torch.tensor(rows, dtype=torch.float64)
    return t.reshape(1, -1, 3), torch.ones(1, t.shape[0], dtype=torch.float64)


def test_isotropic_path_is_unchanged():
    """Backwards compatibility: no per-channel tolerance = old behaviour."""
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.10, 0.20, 0.31]])
    p, o, idx = a.assign(pc, pv, max_distance=0.1)
    assert len(p) == 1 and len(idx) == 1


def test_split_rejects_what_the_sphere_accepts():
    """A pair inside the sphere but outside one channel must be rejected."""
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.10, 0.20, 0.38]])          # omega off by 0.08 rad
    assert len(a.assign(pc, pv, max_distance=0.1)[0]) == 1        # sphere: in
    got = a.assign(pc, pv, max_two_theta=0.01, max_omega=0.02)[0]
    assert len(got) == 0, "split accepted an omega error it should reject"


def test_split_accepts_what_the_sphere_rejects():
    """The other direction: a large but ALLOWED omega error, tight in 2theta."""
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.1001, 0.2001, 0.45]])      # omega 0.15 rad = one big step
    assert len(a.assign(pc, pv, max_distance=0.1)[0]) == 0        # sphere: out
    got = a.assign(pc, pv, max_two_theta=0.001, max_eta=0.001, max_omega=0.2)[0]
    assert len(got) == 1, "split rejected a pair that is tight where it matters"


def test_periodic_channels_wrap():
    """eta at +179 deg and -179 deg are 2 deg apart, not 358."""
    a = SpotAssigner(_obs([[0.10, math.pi - 0.01, 0.30]]))
    pc, pv = _pred([[0.10, -math.pi + 0.01, 0.30]])
    # isotropic cdist does NOT wrap -> it sees ~2pi and rejects
    assert len(a.assign(pc, pv, max_distance=0.5)[0]) == 0
    # the split path wraps -> 0.02 rad apart, accepted
    got = a.assign(pc, pv, max_two_theta=0.01, max_eta=0.05, max_omega=0.01)[0]
    assert len(got) == 1, "the split path failed to wrap a periodic channel"


def test_unconstrained_channels_are_ignored():
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.10, 3.0, 0.30]])           # wild eta
    assert len(a.assign(pc, pv, max_two_theta=0.01, max_omega=0.01)[0]) == 1
    assert len(a.assign(pc, pv, max_two_theta=0.01, max_eta=0.01)[0]) == 0


def test_one_to_one_stops_two_predictions_claiming_one_observation():
    """Without it a match count inflates."""
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.1000, 0.2000, 0.3000],
                    [0.1002, 0.2002, 0.3002]])
    many = a.assign(pc, pv, max_two_theta=0.01, max_eta=0.01, max_omega=0.01)
    one = a.assign(pc, pv, max_two_theta=0.01, max_eta=0.01, max_omega=0.01,
                   one_to_one=True)
    assert len(many[0]) == 2
    assert len(one[0]) == 1, "one_to_one let an observation be claimed twice"


def test_one_to_one_keeps_the_BEST_claimant():
    a = SpotAssigner(_obs([[0.10, 0.20, 0.30]]))
    pc, pv = _pred([[0.1050, 0.2000, 0.3000],     # worse
                    [0.1001, 0.2000, 0.3000]])    # better
    p, o, idx = a.assign(pc, pv, max_two_theta=0.01, max_eta=0.01,
                         max_omega=0.01, one_to_one=True)
    assert len(idx) == 1 and int(idx[0]) == 1


def test_nonpositive_tolerance_is_refused():
    a = SpotAssigner(_obs([[0.1, 0.2, 0.3]]))
    pc, pv = _pred([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="must be positive"):
        a.assign(pc, pv, max_two_theta=0.0)


def test_empty_and_no_match_return_cleanly():
    a = SpotAssigner(_obs([[0.1, 0.2, 0.3]]))
    pc, pv = _pred([[9.0, 9.0, 9.0]])
    p, o, idx = a.assign(pc, pv, max_two_theta=1e-6)
    assert len(p) == 0 and len(o) == 0 and len(idx) == 0
