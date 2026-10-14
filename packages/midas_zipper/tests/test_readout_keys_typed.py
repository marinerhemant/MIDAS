"""The readout-timing keys reach the zarr with the right types."""
from midas_zipper.param_refresh import FORCE_DOUBLE_PARAMS, FORCE_INT_PARAMS


def test_readout_keys_typed():
    assert {"OmegaReadoutShift", "OmegaReadoutCentre"} <= FORCE_DOUBLE_PARAMS
    assert {"OmegaReadoutAxis", "OmegaReadoutShape"} <= FORCE_INT_PARAMS
