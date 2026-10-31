"""``Width`` default must be 1500 um, the value every other stage uses.

Peak search used 1000 um when an archive carried no ``Width`` while
``calc_radius`` (via the registry/C default), ``global_powder`` and
``eta_coverage`` used 1500 um, so the stages disagreed about the ring band.
"""

from midas_peakfit import params as pf_params
from midas_peakfit.params import ZarrParams


def test_default_width_is_1500_um():
    assert pf_params.DEFAULT_WIDTH == 1500.0
    assert ZarrParams().Width == 1500.0


def test_width_in_um_is_converted_to_pixels_once():
    p = ZarrParams()
    p.px = 75.0
    p.Width = 600.0
    p.finalize()
    assert p.Width == 600.0 / 75.0
