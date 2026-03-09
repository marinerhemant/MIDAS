"""
MIDAS Pipeline State — Incremental HDF5 checkpoint tracking.

Provides utilities for all MIDAS workflows to:
  - Initialize a consolidated HDF5 with provenance + parameter snapshot
  - Mark pipeline stages as complete (with optional intermediate data)
  - Query which stages have been completed (for future restart support)

Usage in workflows:
    from pipeline_state import PipelineH5

    with PipelineH5(h5_path, "nf_midas", args, param_text) as ph5:
        run_preprocessing(...)
        ph5.mark("hkl")

        run_image_processing(...)
        ph5.mark("image_processing")

        run_fitting(...)
        ph5.mark("fitting", data={"voxels/position": pos_array})
"""

import datetime
import json
import logging
import os

import h5py
import numpy as np

from version import get_midas_version, stamp_h5

logger = logging.getLogger(__name__)

# Default gzip compression for all datasets
COMPRESSION = {"compression": "gzip", "compression_opts": 4}


class PipelineH5:
    """Context manager for incremental pipeline state tracking in HDF5.

    On enter:
      - Creates or opens the H5 file
      - Writes /provenance/ and /pipeline_state/ if this is a fresh file
      - Stores the parameter file text and CLI args for reproducibility

    Methods:
      - mark(stage_name, data=None): record a completed stage, optionally write datasets
      - write_dataset(path, array, attrs=None): convenience for writing compressed datasets
      - completed: property returning the list of completed stage names
      - is_complete(stage_name): check if a stage was already completed
    """

    def __init__(self, h5_path: str, workflow_type: str,
                 args_namespace=None, param_text: str = ""):
        """
        Args:
            h5_path:        Path for the consolidated HDF5 file.
            workflow_type:  One of "ff_midas", "ff_dual", "pf_midas",
                            "nf_midas", "nf_multi_res".
            args_namespace: argparse.Namespace (or dict) — serialized as JSON.
            param_text:     Full text of the parameter file.
        """
        self.h5_path = h5_path
        self.workflow_type = workflow_type
        self.param_text = param_text
        self._h5 = None

        # Serialize argparse namespace
        if args_namespace is None:
            self.args_json = "{}"
        elif isinstance(args_namespace, dict):
            self.args_json = json.dumps(args_namespace, default=str)
        else:
            self.args_json = json.dumps(vars(args_namespace), default=str)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.h5_path)), exist_ok=True)
        is_new = not os.path.exists(self.h5_path)
        self._h5 = h5py.File(self.h5_path, "a")

        if is_new:
            self._init_fresh()
        else:
            # Update timestamp on re-open
            if "provenance" in self._h5:
                self._h5["provenance"].attrs["last_opened"] = (
                    datetime.datetime.now().isoformat()
                )
            logger.info(
                f"Resumed pipeline H5: {self.h5_path} "
                f"({len(self.completed)} stages complete)"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5 is not None:
            try:
                # Update final timestamp
                if "pipeline_state" in self._h5:
                    self._h5["pipeline_state"].attrs["last_update"] = (
                        datetime.datetime.now().isoformat()
                    )
                self._h5.flush()
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark(self, stage_name: str, data: dict = None):
        """Record a pipeline stage as complete.

        Args:
            stage_name: Human-readable stage identifier (e.g. "peak_search").
            data:       Optional dict of {h5_path: array_or_scalar} to write
                        into the H5 at the same time. Numpy arrays get gzip
                        compression; scalars are stored as-is.
        """
        ps = self._h5.require_group("pipeline_state")

        # Append to completed_stages
        stages_grp = ps.require_group("completed_stages")
        idx = len(stages_grp)
        stages_grp.create_dataset(str(idx), data=stage_name)

        # Update current_stage marker
        ps.attrs["current_stage"] = stage_name

        # Record per-stage timestamp
        ts_grp = ps.require_group("timestamps/per_stage")
        ts_grp.attrs[stage_name] = datetime.datetime.now().isoformat()

        # Write optional data
        if data:
            for path, value in data.items():
                self.write_dataset(path, value)

        self._h5.flush()
        logger.info(f"Pipeline stage complete: {stage_name}")

    def write_dataset(self, path: str, value, attrs: dict = None,
                      overwrite: bool = True):
        """Write a dataset or scalar to the H5 file with compression.

        Args:
            path:      H5 internal path (e.g. "voxels/position").
            value:     numpy array, scalar, or string.
            attrs:     Optional dict of attributes to attach.
            overwrite: If True, delete existing dataset before writing.
        """
        if overwrite and path in self._h5:
            del self._h5[path]

        if isinstance(value, np.ndarray):
            ds = self._h5.create_dataset(path, data=value, **COMPRESSION)
        elif isinstance(value, str):
            ds = self._h5.create_dataset(path, data=value)
        elif isinstance(value, (int, float, bool)):
            ds = self._h5.create_dataset(path, data=value)
        else:
            # Try as numpy array
            arr = np.asarray(value)
            ds = self._h5.create_dataset(path, data=arr, **COMPRESSION)

        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = v

    def write_group_attrs(self, group_path: str, attrs: dict):
        """Write attributes to an H5 group (creating it if needed)."""
        grp = self._h5.require_group(group_path)
        for k, v in attrs.items():
            grp.attrs[k] = v

    @property
    def completed(self) -> list:
        """Return ordered list of completed stage names."""
        if self._h5 is None or "pipeline_state/completed_stages" not in self._h5:
            return []
        grp = self._h5["pipeline_state/completed_stages"]
        stages = []
        for i in range(len(grp)):
            key = str(i)
            if key in grp:
                val = grp[key][()]
                stages.append(val.decode() if isinstance(val, bytes) else val)
        return stages

    def is_complete(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return stage_name in self.completed

    @property
    def h5(self) -> h5py.File:
        """Direct access to the underlying h5py.File for custom writes."""
        return self._h5

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_fresh(self):
        """Initialize a fresh H5 file with provenance and pipeline state."""
        # Provenance (uses existing stamp_h5 for version info)
        prov = self._h5.require_group("provenance")
        stamp_h5(prov)
        prov.attrs["parameter_file"] = self.param_text

        # Pipeline state
        ps = self._h5.require_group("pipeline_state")
        ps.attrs["workflow_type"] = self.workflow_type
        ps.attrs["command_line_args"] = self.args_json
        ps.attrs["start"] = datetime.datetime.now().isoformat()
        ps.attrs["last_update"] = ps.attrs["start"]
        ps.attrs["current_stage"] = ""
        ps.require_group("completed_stages")
        ps.require_group("timestamps/per_stage")

        self._h5.flush()
        logger.info(
            f"Initialized pipeline H5: {self.h5_path} "
            f"(workflow={self.workflow_type})"
        )


# ------------------------------------------------------------------
# Standalone convenience functions (for simple usage without context mgr)
# ------------------------------------------------------------------

def get_completed_stages(h5_path: str) -> list:
    """Read completed stages from an existing pipeline H5."""
    if not os.path.exists(h5_path):
        return []
    with h5py.File(h5_path, "r") as h5:
        if "pipeline_state/completed_stages" not in h5:
            return []
        grp = h5["pipeline_state/completed_stages"]
        stages = []
        for i in range(len(grp)):
            key = str(i)
            if key in grp:
                val = grp[key][()]
                stages.append(val.decode() if isinstance(val, bytes) else val)
        return stages


def can_skip_to(h5_path: str, target_stage: str, stage_order: list) -> bool:
    """Check if we can skip to a target stage based on completed stages.

    Returns True if all stages before target_stage in stage_order are complete.

    Args:
        h5_path:      Path to pipeline H5.
        target_stage: Stage we want to skip to.
        stage_order:  Ordered list of stage names for this workflow.
    """
    if target_stage not in stage_order:
        return False
    completed = set(get_completed_stages(h5_path))
    target_idx = stage_order.index(target_stage)
    for i in range(target_idx):
        if stage_order[i] not in completed:
            return False
    return True
