from parsl.providers import AdHocProvider
from parsl.channels import SSHChannel
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from typing import Any, Dict
from parsl.channels import LocalChannel
from parsl.providers import GridEngineProvider
from parsl.executors import HighThroughputExecutor
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")

user_opts: Dict[str, Dict[str, Any]]
user_opts = {'adhoc':
             {'username': 'tomo1',
              'script_dir': SCRIPTDIR,
              'remote_hostnames': ['puppy80','puppy81','puppy82','puppy83','puppy84','puppy85','puppy86','puppy87','puppy88','puppy89','puppy90']
              }
             }

print(SCRIPTDIR)
orthrosAllConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='orthrosall',
            max_workers_per_node=1,
            worker_logdir_root=user_opts['adhoc']['script_dir'],
            provider=AdHocProvider(
                worker_init='source /clhome/TOMO1/opt/midasconda3/bin/activate',
                channels=[SSHChannel(hostname=m,
                                     username=user_opts['adhoc']['username'],
                                     script_dir=user_opts['adhoc']['script_dir'],
                                     ) for m in user_opts['adhoc']['remote_hostnames']]
            )
        ),
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
    strategy='none',
)