"""Space-group level tests: the centring rule and its space-group awareness."""
from __future__ import annotations

import numpy as np
import pytest

class TestCentringAllowed:
    """The centring rule, derived from LATTICE_TRANSLATIONS rather than tabulated.

    midas_defect.rows carried a hand-written version keyed on space-group number
    lists and applied the I rule to an F cell. On La3Ni2O7 S5 that credited
    33.6 % forbidden reflections AND was blind to every all-odd family F allows.
    These tests pin both directions.
    """

    def test_f_and_i_are_not_the_same_rule(self):
        import numpy as np
        from midas_hkls import centring_allowed
        h = np.array([[1, 1, 1], [1, 1, 0], [2, 0, 0], [3, 1, 1]])
        F = centring_allowed(h, 69)            # Fmmm
        I = centring_allowed(h, 139)           # I4/mmm
        # all-odd: F allows, I forbids -- the BLINDNESS half of the bug
        assert F[0] and F[3] and not I[0] and not I[3]
        # h+k+l even but mixed parity: I allows, F forbids -- the PERMISSIVE half
        assert I[1] and not F[1]
        assert (F != I).sum() == 3

    @pytest.mark.parametrize("sg,rule", [
        (225, lambda h, k, l: (h % 2 == k % 2) and (k % 2 == l % 2)),   # F
        (69,  lambda h, k, l: (h % 2 == k % 2) and (k % 2 == l % 2)),   # F
        (229, lambda h, k, l: (h + k + l) % 2 == 0),                    # I
        (139, lambda h, k, l: (h + k + l) % 2 == 0),                    # I
        (63,  lambda h, k, l: (h + k) % 2 == 0),                        # C
        (38,  lambda h, k, l: (k + l) % 2 == 0),                        # A
        (166, lambda h, k, l: (-h + k + l) % 3 == 0),                   # R obverse
        (221, lambda h, k, l: True),                                    # P
    ])
    def test_matches_the_textbook_condition(self, sg, rule):
        """Derived rule must reproduce the standard condition for every centring."""
        import numpy as np
        from midas_hkls import centring_allowed
        h = np.array([(a, b, c) for a in range(-3, 4) for b in range(-3, 4)
                      for c in range(-3, 4)])
        got = centring_allowed(h, sg)
        want = np.array([rule(*r) for r in h])
        assert (got == want).all(), f"SG {sg}: {int((got != want).sum())} disagree"

    def test_accepts_number_symbol_and_object(self):
        import numpy as np
        from midas_hkls import centring_allowed, SpaceGroup
        h = np.array([[1, 1, 1], [1, 1, 0]])
        a = centring_allowed(h, 225)
        b = centring_allowed(h, SpaceGroup.from_number(225))
        assert (a == b).all()

    def test_rejects_malformed_hkl(self):
        import numpy as np
        from midas_hkls import centring_allowed
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            centring_allowed(np.array([1, 1, 1]), 225)
