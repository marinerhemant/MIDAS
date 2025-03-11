// cuda_daemon.c
// ~/opt/midascuda/cuda/bin/nvcc -o bin/cuda_daemon src/cuda_daemon.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>

int main(char *argv[], int argc) {
    if (argc == 2) {
        // If the user passes an argument, use it as the CUDA device
        int device = atoi(argv[1]);
        cudaSetDevice(device);
    } else{
        // Otherwise, use the default device
        cudaSetDevice(0);
    }
    
    // Create a small file to indicate the daemon is running
    FILE *f = fopen("/tmp/cuda_daemon_running", "w");
    if (f) {
        fprintf(f, "%d", getpid());
        fclose(f);
    }
    
    printf("CUDA device initialized. Daemon running with PID %d\n", getpid());
    
    // Sleep forever, keeping the CUDA context alive
    while(1) {
        sleep(3600);  // Sleep for an hour and repeat
    }
    
    return 0;
}