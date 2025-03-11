// MIDAS cuda_server.c
// ~/opt/midascuda/cuda/bin/nvcc -o bin/cuda_server src/cuda_server.c -lrt # -lcudart
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cuda_runtime.h>

#define SOCKET_PATH "/tmp/cuda_server_socket"

int main() {
    // Initialize CUDA once
    cudaSetDevice(0);
    printf("CUDA device initialized and ready\n");
    
    // Create socket
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
        return 1;
    }
    
    // Setup socket
    struct sockaddr_un address;
    memset(&address, 0, sizeof(struct sockaddr_un));
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, SOCKET_PATH, sizeof(address.sun_path) - 1);
    
    // Remove socket file if it exists
    unlink(SOCKET_PATH);
    
    // Bind socket
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        return 1;
    }
    
    // Listen for connections
    if (listen(server_fd, 5) < 0) {
        perror("Listen failed");
        return 1;
    }
    
    printf("CUDA server ready at %s\n", SOCKET_PATH);
    
    // Accept connections and process requests
    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        char buffer[256];
        read(client_fd, buffer, sizeof(buffer));
        
        // Process the CUDA operation here
        // For example, you could interpret commands to run specific CUDA functions
        
        char response[] = "CUDA operation completed";
        write(client_fd, response, strlen(response));
        
        close(client_fd);
    }
    
    close(server_fd);
    return 0;
}