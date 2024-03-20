from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes"))

marquetteConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='Marquette',
            cores_per_worker=36,
            max_workers_per_node=1,
            provider=SlurmProvider(
                nodes_per_block=nNodes,
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                partition='defq',
                worker_init='module load python; source activate parsl',
                walltime='90:00:00',
                cmd_timeout=120,
            ),
        )
    ]
)
