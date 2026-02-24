from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

localConfig = Config(
    executors=(ThreadPoolExecutor(
        label='threads', 
        max_threads=1, 
        storage_access=None, 
        thread_name_prefix='', 
        working_dir=None
    ),), 
    initialize_logging=True, 
    internal_tasks_max_threads=1, 
    max_idletime=120.0, 
    monitoring=None, 
    retries=0, 
    retry_handler=None, 
    run_dir='runinfo', 
    strategy='simple', 
    strategy_period=5, 
    usage_tracking=False
)