from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import GridEngineProvider
from parsl.executors import HighThroughputExecutor
import os,sys
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
v7Dir = os.path.expanduser('~/opt/MIDAS/FF_HEDM/v7/')
sys.path.insert(0,v7Dir)
from ff_apps import *

orthrosAllConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='orthros_new',
            max_workers_per_node=1,
            provider=GridEngineProvider(
                channel=LocalChannel(),
                nodes_per_block=1,
                init_blocks=1,
                max_blocks=1,
                # walltime="150:00:00",
                scheduler_options='''#$ -q sec1new7.q
#$ -pe sec1new7 11
#$ -l h_rt=175:59:00
#$ -l s_rt=175:58:50''',     # Input your scheduler_options if needed
                worker_init='source /clhome/TOMO1/opt/midasconda3/bin/activate',     # Input your worker_init if needed
            ),
        )
    ],
)
