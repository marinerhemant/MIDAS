#!/bin/bash
# ensure_cuda_ready.sh

# Check if the daemon is running
if [ ! -f /tmp/cuda_daemon_running ]; then
    # Start the daemon
    $HOME/opt/MIDAS/FF_HEDM/bin/cuda_daemon &
    # Wait for it to initialize
    sleep 2
    echo "CUDA daemon started"
else
    pid=$(cat /tmp/cuda_daemon_running)
    if ps -p $pid > /dev/null; then
        echo "CUDA daemon already running with PID $pid"
    else
        # Daemon crashed, restart it
        rm /tmp/cuda_daemon_running
        $HOME/opt/MIDAS/FF_HEDM/bin/cuda_daemon &
        sleep 2
        echo "CUDA daemon restarted"
    fi
fi

# Set environment variables that might help with faster initialization
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_MAXSIZE=1073741824  # 1GB cache
export __GL_SHADER_DISK_CACHE=1