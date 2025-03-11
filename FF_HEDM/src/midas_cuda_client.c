// cuda_client.c
#include "cuda_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#define SOCKET_PATH "/tmp/cuda_server_socket"

static int client_fd = -1;

int cuda_client_init() {
    client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (client_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    struct sockaddr_un address;
    memset(&address, 0, sizeof(struct sockaddr_un));
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, SOCKET_PATH, sizeof(address.sun_path) - 1);
    
    if (connect(client_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Connection failed");
        close(client_fd);
        client_fd = -1;
        return -1;
    }
    
    return 0;
}

int cuda_execute(const char* command) {
    if (client_fd < 0) {
        fprintf(stderr, "Not connected to CUDA server\n");
        return -1;
    }
    
    write(client_fd, command, strlen(command));
    
    char buffer[256];
    int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        printf("Server response: %s\n", buffer);
        return 0;
    }
    
    return -1;
}

void cuda_client_close() {
    if (client_fd >= 0) {
        close(client_fd);
        client_fd = -1;
    }
}