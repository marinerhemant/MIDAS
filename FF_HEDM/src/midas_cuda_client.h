// cuda_client.h
#ifndef CUDA_CLIENT_H
#define CUDA_CLIENT_H

// Initialize connection to CUDA server
int cuda_client_init();

// Perform CUDA operation through server
int cuda_execute(const char* command);

// Close connection
void cuda_client_close();

#endif