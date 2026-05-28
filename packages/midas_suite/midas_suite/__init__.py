"""midas-suite: meta-package that installs the MIDAS Python pipeline.

This package contains no scientific code of its own. It declares the
MIDAS sub-packages as dependencies so that ``pip install midas-suite``
gives a user the full pipeline in one step.

The actual modules live in their own packages and are imported under
their own names (``midas_stress``, ``midas_diffract``, ``midas_index``,
etc.) — this package does not re-export them.

To inspect what was installed:

    >>> import midas_suite
    >>> print(midas_suite.installed())

"""

__version__ = "0.3.10"

# The sub-packages this meta-package pulls in (in publish-order).
# Kept in sync with pyproject.toml's ``dependencies`` list.
SUBPACKAGES = (
    # leaves
    "midas_stress",
    "midas_params",
    "midas_hkls",
    "midas_diffract",
    "midas_peakfit",
    "midas_integrate",
    "midas_integrate_v2",
    "midas_calibrate",
    "midas_calibrate_v2",
    "midas_index",
    "midas_transforms",
    "midas_process_grains",
    "midas_fit_grain",
    "midas_nf_preprocess",
    "midas_nf_fitorientation",
    "midas_zipper",
    # parsl + pipeline orchestrators
    "midas_parsl_configs",
    "midas_ff_pipeline",
    "midas_nf_pipeline",
    "midas_pipeline",
)


def installed():
    """Return a dict {sub_package_name: version_or_None} for every
    MIDAS sub-package this meta-package declares as a dependency.

    Useful for ``midas_suite.installed()`` from a notebook or REPL to
    confirm what got pulled in by ``pip install midas-suite``.
    """
    import importlib

    out = {}
    for name in SUBPACKAGES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            out[name] = None
            continue
        out[name] = getattr(mod, "__version__", "unknown")
    return out


__all__ = ["__version__", "SUBPACKAGES", "installed"]
