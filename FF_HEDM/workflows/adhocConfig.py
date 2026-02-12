from parsl.providers import AdHocProvider
from parsl.channels import SSHChannel
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from typing import Any, Dict
import os

SCRIPTDIR = os.environ.get("MIDAS_SCRIPT_DIR")

user_opts: Dict[str, Dict[str, Any]]
user_opts = {'adhoc':
             {'username': 's1iduser',
              'script_dir': SCRIPTDIR,
              'remote_hostnames': ['califone.xray.aps.anl.gov', 'sekami.xray.aps.anl.gov', 
                                   'chiltepin.xray.aps.anl.gov']
                                #    , 'toro.xray.aps.anl.gov', 
                                #    'chutoro.xray.aps.anl.gov', 'copland.xray.aps.anl.gov',
                                #    'pinback.xray.aps.anl.gov'
                                #    ]
              }
             }


config = Config(
    executors=[
        HighThroughputExecutor(
            label='remote_htex',
            max_workers_per_node=1,
            worker_logdir_root=user_opts['adhoc']['script_dir'],
            provider=AdHocProvider(
                worker_init=os.environ.get('MIDAS_CONDA_INIT', 'source /home/beams/S1IDUSER/opt/midasconda3/midasconda/bin/activate'),
                channels=[SSHChannel(hostname=m,
                                     username=user_opts['adhoc']['username'],
                                     script_dir=user_opts['adhoc']['script_dir'],
                                     ) for m in user_opts['adhoc']['remote_hostnames']]
            )
        )
    ],
    strategy='none',
)
