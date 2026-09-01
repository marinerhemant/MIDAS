"""Standard plots for MIDAS reconstructions.

    from midas_plotting import read_mic, orientation_map
    orientation_map("sampleC.0.mic", space_group=225, cmin=0.3)

Far-field ``Grains.csv`` lives in the ``ff`` submodule::

    from midas_plotting import ff, read_grains
    g = read_grains("Grains.csv")
    ff.summary(g)                      # one-page overview
    ff.grain_map(g, color="ipf")       # IPF-coloured grain scatter
    ff.ipf_legend(g.space_group)       # the colour key

FF plots are namespaced rather than exported flat because both modalities have
a ``grain_map`` and they mean different things: ``maps.grain_map`` labels a
near-field voxel grid, ``ff.grain_map`` scatters far-field grain centres.

Laue microdiffraction lives in ``laue``, and reads the indexer's text output::

    from midas_plotting import laue, read_solutions
    sol = read_solutions("solutions.txt")          # one row per frame, not per grain
    sol = sol.gate(11)                             # the measured random-orientation null
    c = laue.cluster(sol, 1.0, space_group=194)    # grains, with full-field objects flagged
    laue.tilt_histogram(c.representatives(sol.orient_mat))   # vs the random reference
    laue.summary(sol)

or from the shell::

    midas-plot sampleC.0.mic --kind orientation --cmin 0.3 --sg 225
    midas-plot Grains.csv --kind summary

Written after the same IPF colouring, .mic parsing and map plotting were
re-implemented several times in one-off analysis scripts, each time with its own
conventions.
"""
from .ipf import (
    CUBIC, HEXAGONAL, direction_rgb, ipf_rgb, ipf_rgb_from_matrix,
    laue_class, sym_matrices,
)
from .maps import (
    TRUST_FLOOR, compare_maps, confidence_map, grain_labels, grain_map,
    orientation_map,
)
from . import ff, laue
from .grains import GrainList, read_grains
from .solutions import (
    LaueSolutions, LaueSpots, read_solutions, read_spots, read_validated,
)
from .mic import MicMap, read_mic

__version__ = "0.4.0"
__all__ = [
    "MicMap", "read_mic", "GrainList", "read_grains", "ff", "laue",
    "LaueSolutions", "LaueSpots", "read_solutions", "read_spots",
    "read_validated", "ipf_rgb", "ipf_rgb_from_matrix", "direction_rgb",
    "sym_matrices", "laue_class",
    "CUBIC", "HEXAGONAL", "orientation_map", "confidence_map", "grain_map",
    "grain_labels", "compare_maps", "TRUST_FLOOR", "__version__",
]
