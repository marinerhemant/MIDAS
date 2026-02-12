from parsl.providers import AdHocProvider
from parsl.channels import SSHChannel
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from typing import Any, Dict
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

orthrosNewConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='orthrosnew',
            max_workers_per_node=1,
            worker_logdir_root=user_opts['adhoc']['script_dir'],
            provider=AdHocProvider(
                worker_init=os.environ.get('MIDAS_CONDA_INIT', 'source /clhome/TOMO1/opt/midasconda3/bin/activate'),
                channels=[SSHChannel(hostname=m,
                                     username=user_opts['adhoc']['username'],
                                     script_dir=user_opts['adhoc']['script_dir'],
                                     ) for m in user_opts['adhoc']['remote_hostnames']]
            )
        )
    ],
    strategy='none',
)

user_optsAll: Dict[str, Dict[str, Any]]
user_optsAll = {'adhoc':
             {'username': 'tomo1',
              'script_dir': SCRIPTDIR,
              'remote_hostnames': ['pup0100','pup0101','pup0102','pup0103','pup0105']
              }
             }

orthrosAllConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='orthrosall',
            max_workers_per_node=1,
            worker_logdir_root=user_optsAll['adhoc']['script_dir'],
            provider=AdHocProvider(
                worker_init=os.environ.get('MIDAS_CONDA_INIT', 'source /clhome/TOMO1/opt/midasconda3/bin/activate'),
                channels=[SSHChannel(hostname=m,
                                     username=user_optsAll['adhoc']['username'],
                                     script_dir=user_optsAll['adhoc']['script_dir'],
                                     ) for m in user_optsAll['adhoc']['remote_hostnames']]
            )
        )
    ],
    strategy='none',
)

user_optsCombined: Dict[str, Dict[str, Any]]
user_optsCombined = {'adhoc':
             {'username': 'tomo1',
              'script_dir': SCRIPTDIR,
              'remote_hostnames': ['puppy80','puppy81','puppy82','puppy83','puppy84','puppy85','puppy86','puppy87','puppy88','puppy89','puppy90',
                                   'pup0100','pup0101','pup0102','pup0103','pup0105']
              }
             }

orthrosCombinedConfig = Config(
    executors=[
        HighThroughputExecutor(
            label='orthroscombined',
            max_workers_per_node=1,
            worker_logdir_root=user_optsCombined['adhoc']['script_dir'],
            provider=AdHocProvider(
                worker_init=os.environ.get('MIDAS_CONDA_INIT', 'source /clhome/TOMO1/opt/midasconda3/bin/activate'),
                channels=[SSHChannel(hostname=m,
                                     username=user_optsCombined['adhoc']['username'],
                                     script_dir=user_optsCombined['adhoc']['script_dir'],
                                     ) for m in user_optsCombined['adhoc']['remote_hostnames']]
            )
        )
    ],
    strategy='none',
)
