"""BLAS-thread-guarded multiprocessing helpers.

The 96-worker pool x 64-thread BLAS pool failure mode at ``fork()`` is the
operating motivation: large per-grain analyses are embarrassingly parallel
across grains, but each worker's BLAS pool will blow ``ulimit -u`` on a fat
node if BLAS threading is not pinned to 1 per worker.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

_BLAS_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def init_blas_single_thread() -> None:
    """Pin every BLAS-family thread pool to 1 thread.

    Sets the standard NUM_THREADS env vars and -- if ``threadpoolctl`` is
    installed -- also caps already-loaded pools at runtime.

    The env-var assignment is only effective for child processes that import
    numpy *after* the assignment, so the canonical entry point is to pass
    this function as the ``initializer`` to a :class:`multiprocessing.Pool`.
    """
    for var in _BLAS_ENV_VARS:
        os.environ[var] = "1"
    try:  # best-effort runtime cap if available
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass


def _pool_initializer(init_globals: dict[str, Any] | None) -> None:
    init_blas_single_thread()
    if init_globals:
        import builtins

        for k, v in init_globals.items():
            setattr(builtins, k, v)


def bootstrap_pool(
    n_workers: int,
    init_globals: dict[str, Any] | None = None,
    context: str = "spawn",
) -> "mp.pool.Pool":
    """A multiprocessing Pool with BLAS pinned to 1 thread per worker.

    Parameters
    ----------
    n_workers
        Worker count. Use the *physical* core count; oversubscription past
        that loses to BLAS-pin overhead.
    init_globals
        Optional mapping injected into each worker's ``builtins`` so large
        read-only arrays can be shared without pickling per task.
    context
        Multiprocessing start method. ``"spawn"`` is the safe default (no
        inherited BLAS pool from the parent); ``"fork"`` is faster on Linux
        but inherits the parent's already-loaded BLAS state.
    """
    ctx = mp.get_context(context)
    return ctx.Pool(
        processes=n_workers,
        initializer=_pool_initializer,
        initargs=(init_globals,),
    )


__all__ = ["init_blas_single_thread", "bootstrap_pool"]
