"""midas-parsl-configs — bundled + user-extensible Parsl configs.

Two ways to get a ``parsl.config.Config`` out of this package:

  - ``load_config(name)`` — picks a bundled or user-registered config by
    short name (``local``, ``umich``, ``polaris`` …). Sets the env vars
    each config expects (``MIDAS_SCRIPT_DIR``, ``nNodes`` …) before
    importing the module so the user gets a config that's actually
    runnable.
  - ``available_configs()`` — returns the dict of ``{name: source}`` for
    everything currently discoverable.

User-defined configs go in ``~/.midas/parsl_configs/<name>Config.py``
(override path with ``$MIDAS_PARSL_CONFIGS_DIR``). They're loaded first
so users can override a bundled config of the same name.

The generator (``midas_parsl_configs.generate``) reads a SLURM/PBS
submit script and writes a config module into the user dir.
"""
from __future__ import annotations

from .registry import (
    AVAILABLE_BUILTIN,
    available_configs,
    load_config,
    register_config,
    user_configs_dir,
)
from .generate import (
    build_config_module,
    parse_submit_script,
    write_user_config,
)

__all__ = [
    "AVAILABLE_BUILTIN",
    "available_configs",
    "load_config",
    "register_config",
    "user_configs_dir",
    "build_config_module",
    "parse_submit_script",
    "write_user_config",
]

__version__ = "0.1.1"
