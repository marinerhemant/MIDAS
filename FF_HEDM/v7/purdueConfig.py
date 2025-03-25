from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes"))

purdueConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='Purdue',
            cores_per_worker=128,
            max_workers_per_node=1,
            provider=SlurmProvider(
                nodes_per_block=1,
                init_blocks=1,
                min_blocks=1,
                max_blocks=nNodes,
                # partition='msangid',
                scheduler_options='#SBATCH -A msangid',
                worker_init=f"module load anaconda/2024.02-py311; conda activate MIDAS_Env ",
                walltime='90:00:00',
                cmd_timeout=120,
            ),
        )
    ]
)
