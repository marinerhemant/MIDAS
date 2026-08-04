from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes"))

# Site-specific credentials. Deliberately NOT hardcoded: an allocation code and a
# user home directory identify the group whose allocation this is, and this file
# is public. Set both in the environment before launching.
#   MIDAS_SLURM_ACCOUNT   e.g. the value passed to "#SBATCH --account="
#   MIDAS_CONDA_ACTIVATE  absolute path to the conda "activate" script
SLURM_ACCOUNT = os.environ.get("MIDAS_SLURM_ACCOUNT", "")
CONDA_ACTIVATE = os.environ.get("MIDAS_CONDA_ACTIVATE", "")

uMichConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='UMGreatLakes',
            cores_per_worker=36,
            max_workers_per_node=1,
            provider=SlurmProvider(
                nodes_per_block=1,
                init_blocks=1,
                min_blocks=1,
                max_blocks=nNodes,
                partition='standard',
                scheduler_options=(
                    f'#SBATCH --account={SLURM_ACCOUNT}' if SLURM_ACCOUNT else ''
                ),
                worker_init=(
                    f'source {CONDA_ACTIVATE}' if CONDA_ACTIVATE else ''
                ),
                walltime='90:00:00',
                cmd_timeout=120,
            ),
        )
    ]
)
