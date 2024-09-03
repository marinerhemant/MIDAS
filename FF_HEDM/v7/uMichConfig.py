from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")
nNodes = int(os.environ.get("nNodes"))

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
                scheduler_options='#SBATCH -A abucsek1',
                worker_init='source /home/wenxli/miniconda3/bin/activate',
                walltime='90:00:00',
                cmd_timeout=120,
            ),
        )
    ]
)
