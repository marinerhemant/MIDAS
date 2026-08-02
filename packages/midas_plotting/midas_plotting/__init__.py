"""Standard plots for MIDAS reconstructions.

    from midas_plotting import read_mic, orientation_map
    orientation_map("Ce5Y.0.mic", space_group=225, cmin=0.3)

or from the shell::

    midas-plot Ce5Y.0.mic --kind orientation --cmin 0.3 --sg 225

Written after the same IPF colouring, .mic parsing and map plotting were
re-implemented several times in one-off analysis scripts, each time with its own
conventions.
"""
from .ipf import CUBIC, HEXAGONAL, ipf_rgb, laue_class, sym_matrices
from .maps import (
    TRUST_FLOOR, compare_maps, confidence_map, grain_labels, grain_map,
    orientation_map,
)
from .mic import MicMap, read_mic

__version__ = "0.1.0"
__all__ = [
    "MicMap", "read_mic", "ipf_rgb", "sym_matrices", "laue_class",
    "CUBIC", "HEXAGONAL", "orientation_map", "confidence_map", "grain_map",
    "grain_labels", "compare_maps", "TRUST_FLOOR", "__version__",
]
