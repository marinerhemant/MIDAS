/**
 * midas_paths.h - Utilities for handling MIDAS executable paths
 * 
 * This is a self-contained header file for detecting and using MIDAS
 * executable paths across different platforms.
 */

#ifndef MIDAS_PATHS_H
#define MIDAS_PATHS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <libgen.h>
#else
#include <unistd.h>
#include <libgen.h>
#endif

/**
 * Get the directory containing the current executable
 * 
 * @param buffer Buffer to store the path
 * @param buffer_size Size of the buffer
 * @return 0 on success, -1 on failure
 */
static int get_executable_directory(char* buffer, size_t buffer_size) {
    if (buffer == NULL || buffer_size == 0) {
        return -1;
    }

#ifdef _WIN32
    // Windows implementation
    HMODULE hModule = GetModuleHandle(NULL);
    if (hModule == NULL) {
        return -1;
    }

    DWORD path_len = GetModuleFileName(hModule, buffer, buffer_size);
    if (path_len == 0 || path_len >= buffer_size) {
        return -1;
    }

    // Convert backslashes to forward slashes for consistency
    for (DWORD i = 0; i < path_len; i++) {
        if (buffer[i] == '\\') {
            buffer[i] = '/';
        }
    }

    // Remove the executable name to get the directory
    char* last_slash = strrchr(buffer, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
    }
#elif defined(__APPLE__)
    // macOS implementation
    uint32_t size = buffer_size;
    if (_NSGetExecutablePath(buffer, &size) != 0) {
        return -1;
    }
    
    // Get the directory component
    char* dir = dirname(buffer);
    if (dir == NULL) {
        return -1;
    }
    
    strncpy(buffer, dir, buffer_size);
    buffer[buffer_size - 1] = '\0';
#else
    // Linux/Unix implementation
    ssize_t len = readlink("/proc/self/exe", buffer, buffer_size - 1);
    if (len == -1) {
        // Try using argv[0] from main as fallback
        char* fallback_path = getenv("_");
        if (fallback_path != NULL) {
            strncpy(buffer, fallback_path, buffer_size - 1);
            buffer[buffer_size - 1] = '\0';
        } else {
            return -1;
        }
    } else {
        buffer[len] = '\0';
    }
    
    // Get the directory component
    char* dir = dirname(buffer);
    if (dir == NULL) {
        return -1;
    }
    
    strncpy(buffer, dir, buffer_size);
    buffer[buffer_size - 1] = '\0';
#endif

    return 0;
}

/**
 * Get the directory containing MIDAS binaries
 * 
 * @param buffer Buffer to store the path
 * @param buffer_size Size of the buffer
 * @return 0 on success, -1 on failure
 */
static int get_midas_bin_directory(char* buffer, size_t buffer_size) {
    static char cached_path[PATH_MAX] = {0};
    
    // Return cached path if available
    if (cached_path[0] != '\0') {
        strncpy(buffer, cached_path, buffer_size);
        buffer[buffer_size - 1] = '\0';
        return 0;
    }
    
    // Check if MIDAS_HOME environment variable is set
    const char* midas_home = getenv("MIDAS_HOME");
    if (midas_home != NULL) {
        snprintf(buffer, buffer_size, "%s/bin", midas_home);
        
        // Cache the result
        strncpy(cached_path, buffer, sizeof(cached_path));
        cached_path[sizeof(cached_path) - 1] = '\0';
        
        return 0;
    }
    
    // Get the directory containing the current executable
    char exe_dir[PATH_MAX];
    if (get_executable_directory(exe_dir, sizeof(exe_dir)) != 0) {
        // Fallback to default path
        strncpy(buffer, "~/opt/MIDAS/FF_HEDM/bin", buffer_size);
        buffer[buffer_size - 1] = '\0';
        
        fprintf(stderr, "Warning: Could not determine executable path, using default MIDAS path\n");
        return 0;
    }
    
    // Assume the bin directory is either:
    // 1. The same as the executable directory
    // 2. The parent directory of the executable + "/bin"
    
    // Try option 1 first
    strncpy(buffer, exe_dir, buffer_size);
    buffer[buffer_size - 1] = '\0';
    
    // Cache the result
    strncpy(cached_path, buffer, sizeof(cached_path));
    cached_path[sizeof(cached_path) - 1] = '\0';
    
    return 0;
}

/**
 * Run a MIDAS binary with the given parameters
 * 
 * @param binary_name Name of the binary executable
 * @param params Parameters to pass to the binary
 * @return Return value from system() call
 */
static int run_midas_binary(const char* binary_name, const char* params) {
    char bin_dir[PATH_MAX];
    char cmd[PATH_MAX * 2];
    
    if (get_midas_bin_directory(bin_dir, sizeof(bin_dir)) != 0) {
        fprintf(stderr, "Error: Could not determine MIDAS bin directory\n");
        return -1;
    }
    
    snprintf(cmd, sizeof(cmd), "%s/%s %s", bin_dir, binary_name, params);
    return system(cmd);
}

#endif /* MIDAS_PATHS_H */