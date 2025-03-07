//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  Created by Hemant Sharma on 2017/07/10.
//
//
// TODO: Add option to give QbinSize instead of RbinSize
// MUST HAVE SQUARE PIXELS

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-6
double *distortionMapY;
double *distortionMapZ;
int distortionFile;

// Define the struct that will hold our parameters
typedef struct {
    double tx;
    double ty;
    double tz;
    double px;
    double pxY;
    double pxZ;
    double yCen;  // First value from BC array
    double zCen;  // Second value from BC array
    double Lsd;
    double RhoD;
    double p0;
    double p1;
    double p2;
    double p3;
    double EtaBinSize;
    double EtaMin;
    double EtaMax;
    double RBinSize;
    double RMin;
    double RMax;
    int NrPixels;
    int NrPixelsY;
    int NrPixelsZ;
    int ImTransOpt[10];  // Array for multiple int values, max 10
    int NrTransOpt;  // Number of elements actually used
    char* DistortionFile;  // String for file path
} ConfigParams;

// Function declarations
const char* skipWhitespace(const char* str);
const char* skipUntil(const char* str, char target);
char* parseJsonString(const char** json);
bool startsWith(const char* str, const char* prefix);
double parseJsonNumber(const char** json);
bool parseJsonBoolean(const char** json);
bool parseJsonNull(const char** json);
bool parseJsonNumberArray(const char** json, double* array, int array_size);
int parseJsonIntArray(const char** json, int* array, int max_size);
const char* findJsonField(const char* json, const char* field);
ConfigParams* parseTextFile(const char* content, const char* original_filename);
bool isJsonFile(const char* content);

// Function to free memory allocated for the struct
void freeConfigParams(ConfigParams* params) {
    if (params == NULL) return;
    
    // Free string fields
    free(params->DistortionFile);
    
    // Free the struct itself
    free(params);
}

// Function to check if the file is a JSON file
bool isJsonFile(const char* content) {
    const char* p = skipWhitespace(content);
    return *p == '{';
}

// Skip whitespace characters
const char* skipWhitespace(const char* str) {
    while (isspace(*str)) str++;
    return str;
}

// Skip characters until reaching the target character
const char* skipUntil(const char* str, char target) {
    while (*str != '\0' && *str != target) str++;
    return str;
}

// Parse a JSON string value
char* parseJsonString(const char** json) {
    if (**json != '"') return NULL;
    
    (*json)++; // Skip opening quote
    const char* start = *json;
    
    // Find closing quote
    while (**json != '\0' && **json != '"') {
        // Handle escaped quotes
        if (**json == '\\' && *(*json + 1) == '"') {
            (*json) += 2;
        } else {
            (*json)++;
        }
    }
    
    if (**json != '"') return NULL; // Unterminated string
    
    size_t len = *json - start;
    char* result = (char*)malloc(len + 1);
    if (result == NULL) return NULL;
    
    // Copy string without quotes
    strncpy(result, start, len);
    result[len] = '\0';
    
    (*json)++; // Skip closing quote
    return result;
}

// Check if string starts with a specific word
bool startsWith(const char* str, const char* prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

// Parse a JSON number value
double parseJsonNumber(const char** json) {
    char* endptr;
    double value = strtod(*json, &endptr);
    *json = endptr;
    return value;
}

// Parse a JSON boolean value
bool parseJsonBoolean(const char** json) {
    if (startsWith(*json, "true")) {
        *json += 4;
        return true;
    } else if (startsWith(*json, "false")) {
        *json += 5;
        return false;
    }
    return false;
}

// Parse a JSON null value
bool parseJsonNull(const char** json) {
    if (startsWith(*json, "null")) {
        *json += 4;
        return true;
    }
    return false;
}

// Parse a JSON array of numbers
bool parseJsonNumberArray(const char** json, double* array, int array_size) {
    if (**json != '[') return false;
    
    (*json)++; // Skip opening bracket
    *json = skipWhitespace(*json);
    
    for (int i = 0; i < array_size; i++) {
        // Parse the number
        array[i] = parseJsonNumber(json);
        
        // Skip whitespace after the number
        *json = skipWhitespace(*json);
        
        // Check for comma or end of array
        if (**json == ']') {
            if (i == array_size - 1) {
                (*json)++; // Skip closing bracket
                return true;
            } else {
                return false; // Not enough elements
            }
        } else if (**json == ',') {
            (*json)++; // Skip comma
            *json = skipWhitespace(*json);
        } else {
            return false; // Invalid format
        }
    }
    
    // Skip any remaining elements
    while (**json != ']' && **json != '\0') {
        // Skip to next comma or closing bracket
        *json = skipUntil(*json, ',');
        if (**json == ',') {
            (*json)++;
            *json = skipWhitespace(*json);
        }
    }
    
    if (**json == ']') {
        (*json)++; // Skip closing bracket
        return true;
    }
    
    return false; // Unterminated array
}

// Parse a JSON array of integers
int parseJsonIntArray(const char** json, int* array, int max_size) {
    if (**json != '[') {
        // Handle single integer case (not an array)
        array[0] = (int)parseJsonNumber(json);
        return 1;
    }
    
    (*json)++; // Skip opening bracket
    *json = skipWhitespace(*json);
    
    int count = 0;
    while (**json != ']' && **json != '\0' && count < max_size) {
        // Parse the number
        array[count++] = (int)parseJsonNumber(json);
        
        // Skip whitespace after the number
        *json = skipWhitespace(*json);
        
        // Check for comma or end of array
        if (**json == ']') {
            break;
        } else if (**json == ',') {
            (*json)++; // Skip comma
            *json = skipWhitespace(*json);
        } else {
            break; // Invalid format
        }
    }
    
    // Skip to the closing bracket
    while (**json != ']' && **json != '\0') {
        (*json)++;
    }
    
    if (**json == ']') {
        (*json)++; // Skip closing bracket
    }
    
    return count;
}

// Find a field in the JSON object and return pointer to its value
const char* findJsonField(const char* json, const char* field) {
    size_t field_len = strlen(field);
    
    // Make sure we're in an object
    json = skipWhitespace(json);
    if (*json != '{') return NULL;
    json++;
    
    while (*json != '\0') {
        json = skipWhitespace(json);
        if (*json == '}') break;
        
        // Look for field name
        if (*json != '"') {
            json = skipUntil(json, ',');
            if (*json == ',') json++;
            continue;
        }
        
        // Compare field name
        json++; // Skip opening quote
        if (strncmp(json, field, field_len) == 0 && json[field_len] == '"') {
            json += field_len + 1; // Skip field name and closing quote
            json = skipWhitespace(json);
            
            // Check for colon separator
            if (*json != ':') return NULL;
            json++;
            
            return skipWhitespace(json);
        }
        
        // Field didn't match, skip to next field
        json = skipUntil(json, ',');
        if (*json == ',') json++;
    }
    
    return NULL;
}

// Parse a text file with "Key Value" pairs
ConfigParams* parseTextFile(const char* content, const char* original_filename) {
    // Allocate memory for the struct
    ConfigParams* params = (ConfigParams*)malloc(sizeof(ConfigParams));
    if (params == NULL) {
        fprintf(stderr, "Memory allocation failed for ConfigParams\n");
        return NULL;
    }
    
    // Initialize with default values
    memset(params, 0, sizeof(ConfigParams));
    
    // Buffer for creating JSON
    char* json_str = NULL;
    size_t json_capacity = 0;
    size_t json_length = 0;
    
    // Initialize JSON string
    json_capacity = 1024;  // Start with 1 KB
    json_str = (char*)malloc(json_capacity);
    if (json_str == NULL) {
        fprintf(stderr, "Memory allocation failed for JSON buffer\n");
        free(params);
        return NULL;
    }
    
    // Start the JSON object
    strcpy(json_str, "{\n");
    json_length = 2;
    
    // Parse line by line
    const char* line_start = content;
    const char* line_end;
    bool first_pair = true;
    
    while (*line_start) {
        // Skip leading whitespace
        while (isspace(*line_start)) line_start++;
        
        // Skip empty lines and comments
        if (*line_start == '\0' || *line_start == '#') {
            // Find the next line
            line_start = strchr(line_start, '\n');
            if (line_start == NULL) break;
            line_start++;  // Skip the newline
            continue;
        }
        
        // Find the end of the line
        line_end = strchr(line_start, '\n');
        if (line_end == NULL) line_end = line_start + strlen(line_start);
        
        // Get the key
        const char* key_start = line_start;
        const char* key_end = key_start;
        
        // Find the end of the key (first whitespace)
        while (key_end < line_end && !isspace(*key_end)) key_end++;
        
        // Skip whitespace between key and value
        const char* value_start = key_end;
        while (value_start < line_end && isspace(*value_start)) value_start++;
        
        // The value goes to the end of the line
        const char* value_end = line_end;
        
        // Skip trailing whitespace in value
        while (value_end > value_start && isspace(*(value_end - 1))) value_end--;
        
        // Process only if we have both key and value
        if (key_end > key_start && value_end > value_start) {
            // Extract key and value strings
            int key_len = key_end - key_start;
            int value_len = value_end - value_start;
            
            char* key = (char*)malloc(key_len + 1);
            char* value = (char*)malloc(value_len + 1);
            
            if (key == NULL || value == NULL) {
                fprintf(stderr, "Memory allocation failed for key or value\n");
                free(key);
                free(value);
                free(json_str);
                free(params);
                return NULL;
            }
            
            strncpy(key, key_start, key_len);
            key[key_len] = '\0';
            
            strncpy(value, value_start, value_len);
            value[value_len] = '\0';
            
            // Determine the value type and add to JSON
            bool is_string = false;
            bool is_array = false;
            bool is_bool = false;
            bool is_null = false;
            
            // Special handling for BC parameter
            if (strcmp(key, "BC") == 0) {
                // Check if BC is already in array format
                if (value[0] == '[' && value[value_len - 1] == ']') {
                    is_array = true;
                } else {
                    // Try to parse as space-separated values for BC
                    char* modified_value = NULL;
                    char* token;
                    char* saveptr;
                    double bc_values[2] = {0.0, 0.0};
                    int count = 0;
                    
                    // Make a copy to use with strtok_r
                    char* value_copy = strdup(value);
                    if (value_copy == NULL) {
                        fprintf(stderr, "Memory allocation failed for value copy\n");
                        free(key);
                        free(value);
                        free(json_str);
                        free(params);
                        return NULL;
                    }
                    
                    // Try to parse values
                    token = strtok_r(value_copy, " ,\t", &saveptr);
                    while (token != NULL && count < 2) {
                        bc_values[count++] = atof(token);
                        token = strtok_r(NULL, " ,\t", &saveptr);
                    }
                    
                    free(value_copy);
                    
                    // If we found two values, create a proper JSON array
                    if (count == 2) {
                        modified_value = (char*)malloc(50);  // Should be enough for two doubles
                        if (modified_value == NULL) {
                            fprintf(stderr, "Memory allocation failed for modified value\n");
                            free(key);
                            free(value);
                            free(json_str);
                            free(params);
                            return NULL;
                        }
                        
                        sprintf(modified_value, "[%g, %g]", bc_values[0], bc_values[1]);
                        free(value);
                        value = modified_value;
                        value_len = strlen(value);
                        is_array = true;
                    } else if (count == 1) {
                        // For backward compatibility, use single value and it will be handled later
                        is_array = false;
                    }
                }
            } else {
                // Check if it's an array (starts with '[' and ends with ']')
                if (value[0] == '[' && value[value_len - 1] == ']') {
                    is_array = true;
                }
                // Check if it's a boolean
                else if (strcmp(value, "true") == 0 || strcmp(value, "false") == 0) {
                    is_bool = true;
                }
                // Check if it's null
                else if (strcmp(value, "null") == 0) {
                    is_null = true;
                }
                // Check if it's a string (no numeric characters)
                else {
                    is_string = true;
                    for (int i = 0; i < value_len; i++) {
                        if (isdigit(value[i]) || value[i] == '-' || value[i] == '.') {
                            is_string = false;
                            break;
                        }
                    }
                }
            }
            
            // Ensure we have enough space in the JSON buffer
            size_t required_space = json_length + key_len + value_len + 20;  // Extra for quotes, comma, etc.
            if (required_space >= json_capacity) {
                json_capacity = required_space * 2;
                char* new_json = (char*)realloc(json_str, json_capacity);
                if (new_json == NULL) {
                    fprintf(stderr, "Memory reallocation failed for JSON buffer\n");
                    free(key);
                    free(value);
                    free(json_str);
                    free(params);
                    return NULL;
                }
                json_str = new_json;
            }
            
            // Add the key-value pair to the JSON
            if (!first_pair) {
                json_length += sprintf(json_str + json_length, ",\n");
            }
            
            json_length += sprintf(json_str + json_length, "  \"%s\": ", key);
            
            if (is_string) {
                json_length += sprintf(json_str + json_length, "\"%s\"", value);
            } else if (is_array || is_bool || is_null) {
                json_length += sprintf(json_str + json_length, "%s", value);
            } else {
                json_length += sprintf(json_str + json_length, "%s", value);
            }
            
            first_pair = false;
            
            free(key);
            free(value);
        }
        
        // Move to the next line
        line_start = line_end;
        if (*line_start == '\n') line_start++;
    }
    
    // Close the JSON object
    json_length += sprintf(json_str + json_length, "\n}");
    
    // Save the JSON file
    char json_filename[256];
    snprintf(json_filename, sizeof(json_filename), "%s.json", original_filename);
    FILE* json_file = fopen(json_filename, "w");
    if (json_file) {
        fprintf(json_file, "%s", json_str);
        fclose(json_file);
        printf("Converted text file to JSON: %s\n", json_filename);
    } else {
        fprintf(stderr, "Warning: Could not save JSON file: %s\n", json_filename);
    }
    
    // Now parse the JSON into the struct
    const char* field_value;
    
    // Parse JSON for each field
    // Numeric values
    if ((field_value = findJsonField(json_str, "tx")) != NULL) {
        params->tx = parseJsonNumber(&field_value);
    }
    
    if ((field_value = findJsonField(json_str, "ty")) != NULL) {
        params->ty = parseJsonNumber(&field_value);
    }
    
    if ((field_value = findJsonField(json_str, "tz")) != NULL) {
        params->tz = parseJsonNumber(&field_value);
    }
    
    if ((field_value = findJsonField(json_str, "px")) != NULL) {
        params->px = parseJsonNumber(&field_value);
        
        // Set pxY and pxZ to the same value as px by default
        params->pxY = params->px;
        params->pxZ = params->px;
    }
    
    // These will override the default if they exist in the JSON
    if ((field_value = findJsonField(json_str, "pxY")) != NULL) {
        params->pxY = parseJsonNumber(&field_value);
    }
    
    if ((field_value = findJsonField(json_str, "pxZ")) != NULL) {
        params->pxZ = parseJsonNumber(&field_value);
    }
    
    // Handle BC array and convert to yCen and zCen
    if ((field_value = findJsonField(json_str, "BC")) != NULL) {
        double bc_values[2] = {0.0, 0.0};
        
        if (*field_value == '[') {
            parseJsonNumberArray(&field_value, bc_values, 2);
        } else {
            // For backward compatibility, if it's a single number
            bc_values[0] = parseJsonNumber(&field_value);
        }
        
        // Assign values to yCen and zCen
        params->yCen = bc_values[0];
        params->zCen = bc_values[1];
    }
    
    // ... rest of the function remains the same ...
    
    // Free the JSON string
    free(json_str);
    
    return params;
}
// Function to read and parse the config file (either JSON or text)
ConfigParams* readConfigFile(const char* filename) {
    // Allocate memory for the struct
    ConfigParams* params = NULL;
    
    // Read the file
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read file content
    char* file_content = (char*)malloc(file_size + 1);
    if (file_content == NULL) {
        fprintf(stderr, "Memory allocation failed for file content\n");
        fclose(file);
        return NULL;
    }
    
    size_t read_size = fread(file_content, 1, file_size, file);
    fclose(file);
    
    if (read_size != file_size) {
        fprintf(stderr, "Failed to read the entire file\n");
        free(file_content);
        return NULL;
    }
    
    file_content[file_size] = '\0';
    
    // Check if it's a JSON file or a text file
    if (isJsonFile(file_content)) {
        // Parse as JSON
        params = (ConfigParams*)malloc(sizeof(ConfigParams));
        if (params == NULL) {
            fprintf(stderr, "Memory allocation failed for ConfigParams\n");
            free(file_content);
            return NULL;
        }
        
        // Initialize with default values
        memset(params, 0, sizeof(ConfigParams));
        
        // Parse JSON for each field
        const char* field_value;
        
        // Numeric values
        if ((field_value = findJsonField(file_content, "tx")) != NULL) {
            params->tx = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "ty")) != NULL) {
            params->ty = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "tz")) != NULL) {
            params->tz = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "px")) != NULL) {
            params->px = parseJsonNumber(&field_value);
            
            // Set pxY and pxZ to the same value as px by default
            params->pxY = params->px;
            params->pxZ = params->px;
        }
        
        // These will override the default if they exist in the JSON
        if ((field_value = findJsonField(file_content, "pxY")) != NULL) {
            params->pxY = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "pxZ")) != NULL) {
            params->pxZ = parseJsonNumber(&field_value);
        }
        
        // Handle BC array and convert to yCen and zCen
        if ((field_value = findJsonField(file_content, "BC")) != NULL) {
            double bc_values[2] = {0.0, 0.0};
            
            if (*field_value == '[') {
                parseJsonNumberArray(&field_value, bc_values, 2);
            } else {
                // For backward compatibility, if it's a single number
                bc_values[0] = parseJsonNumber(&field_value);
            }
            
            // Assign values to yCen and zCen
            params->yCen = bc_values[0];
            params->zCen = bc_values[1];
        }
        
        if ((field_value = findJsonField(file_content, "Lsd")) != NULL) {
            params->Lsd = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "RhoD")) != NULL) {
            params->RhoD = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "p0")) != NULL) {
            params->p0 = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "p1")) != NULL) {
            params->p1 = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "p2")) != NULL) {
            params->p2 = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "p3")) != NULL) {
            params->p3 = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "EtaBinSize")) != NULL) {
            params->EtaBinSize = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "EtaMin")) != NULL) {
            params->EtaMin = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "EtaMax")) != NULL) {
            params->EtaMax = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "RBinSize")) != NULL) {
            params->RBinSize = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "RMin")) != NULL) {
            params->RMin = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "RMax")) != NULL) {
            params->RMax = parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "NrPixels")) != NULL) {
            params->NrPixels = (int)parseJsonNumber(&field_value);
            
            // Set NrPixelsY and NrPixelsZ to the same value as NrPixels by default
            params->NrPixelsY = params->NrPixels;
            params->NrPixelsZ = params->NrPixels;
        }
        
        // These will override the default if they exist in the JSON
        if ((field_value = findJsonField(file_content, "NrPixelsY")) != NULL) {
            params->NrPixelsY = (int)parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "NrPixelsZ")) != NULL) {
            params->NrPixelsZ = (int)parseJsonNumber(&field_value);
        }
        
        if ((field_value = findJsonField(file_content, "ImTransOpt")) != NULL) {
            params->NrTransOpt = parseJsonIntArray(&field_value, params->ImTransOpt, 10);
        } else {
            // Default - no options
            params->NrTransOpt = 0;
        }
        
        // Handle string values
        if ((field_value = findJsonField(file_content, "DistortionFile")) != NULL) {
            if (*field_value == '"') {
                params->DistortionFile = parseJsonString(&field_value);
            } else if (parseJsonNull(&field_value)) {
                params->DistortionFile = NULL;
            }
        } else {
            params->DistortionFile = NULL;
        }
    } else {
        // Parse as text file with "Key Value" pairs
        printf("Detected text file format. Converting to JSON...\n");
        params = parseTextFile(file_content, filename);
    }
    
    // Clean up
    free(file_content);
    
    return params;
}

static inline
int BETWEEN(double val, double min, double max)
{
	return ((val-EPS <= max && val+EPS >= min) ? 1 : 0 );
}

static inline
double**
allocMatrix(int nrows, int ncols)
{
    double** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}


static inline double signVal(double x){
	if (x == 0) return 1.0;
	else return x/fabs(x);
}

static inline
void
MatrixMult(
           double m[3][3],
           double  v[3],
           double r[3])
{
    int i;
    for (i=0; i<3; i++) {
        r[i] = m[i][0]*v[0] +
        m[i][1]*v[1] +
        m[i][2]*v[2];
    }
}

static inline
void
MatrixMultF33(
    double m[3][3],
    double n[3][3],
    double res[3][3])
{
    int r;
    for (r=0; r<3; r++) {
        res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
        res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
        res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
    }
}

static inline
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline
void
REta4MYZ(
	double Y,
	double Z,
	double Ycen,
	double Zcen,
	double TRs[3][3],
	double Lsd,
	double RhoD,
	double p0,
	double p1,
	double p2,
	double p3,
	double n0,
	double n1,
	double n2,
	double px,
	double *RetVals)
{
	double Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT, Rt;
	Yc = (-Y + Ycen)*px;
	Zc = ( Z - Zcen)*px;
	ABC[0] = 0;
	ABC[1] = Yc;
	ABC[2] = Zc;
	MatrixMult(TRs,ABC,ABCPr);
	XYZ[0] = Lsd+ABCPr[0];
	XYZ[1] = ABCPr[1];
	XYZ[2] = ABCPr[2];
	Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
	Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
	RNorm = Rad/RhoD;
	EtaT = 90 - Eta;
	DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT+p3)))) + (p2*(pow(RNorm,n2))) + 1;
	Rt = Rad * DistortFunc / px; // in pixels
	RetVals[0] = Eta;
	RetVals[1] = Rt;
}

static inline
void YZ4mREta(double R, double Eta, double *YZ){
	YZ[0] = -R*sin(Eta*deg2rad);
	YZ[1] = R*cos(Eta*deg2rad);
}

const double dy[2] = {-0.5, +0.5};
const double dz[2] = {-0.5, +0.5};

static inline
void
REtaMapper(
	double Rmin,
	double EtaMin,
	int nEtaBins,
	int nRBins,
	double EtaBinSize,
	double RBinSize,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh)
{
	int i;
	for (i=0;i<nEtaBins;i++){
		EtaBinsLow[i] = EtaBinSize*i      + EtaMin;
		EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
	}
	for (i=0;i<nRBins;i++){
		RBinsLow[i] = RBinSize * i      + Rmin;
		RBinsHigh[i] = RBinSize * (i+1) + Rmin;
	}
}

struct Point {
	double x;
	double y;
};

struct Point center;

static int cmpfunc (const void * ia, const void *ib){
	struct Point *a = (struct Point *)ia;
	struct Point *b = (struct Point *)ib;
	if (a->x - center.x >= 0 && b->x - center.x < 0) return 1;
	if (a->x - center.x < 0 && b->x - center.x >= 0) return -1;
	if (a->x - center.x == 0 && b->x - center.x == 0) {
		if (a->y - center.y >= 0 || b->y - center.y >= 0){
			return a->y > b->y ? 1 : -1;
		}
        return b->y > a->y ? 1 : -1;
    }
	double det = (a->x - center.x) * (b->y - center.y) - (b->x - center.x) * (a->y - center.y);
	if (det < 0) return 1;
    if (det > 0) return -1;
    int d1 = (a->x - center.x) * (a->x - center.x) + (a->y - center.y) * (a->y - center.y);
    int d2 = (b->x - center.x) * (b->x - center.x) + (b->y - center.y) * (b->y - center.y);
    return d1 > d2 ? 1 : -1;
}

double PosMatrix[4][2]={{-0.5, -0.5},
						{-0.5,  0.5},
						{ 0.5,  0.5},
						{ 0.5, -0.5}};

static inline
double CalcAreaPolygon(double **Edges, int nEdges){
	int i;
	struct Point *MyData;
	MyData = malloc(nEdges*sizeof(*MyData));
	center.x = 0;
	center.y = 0;
	for (i=0;i<nEdges;i++){
		center.x += Edges[i][0];
		center.y += Edges[i][1];
		MyData[i].x = Edges[i][0];
		MyData[i].y = Edges[i][1];
	}
	center.x /= nEdges;
	center.y /= nEdges;

	qsort(MyData, nEdges, sizeof(struct Point), cmpfunc);
	double **SortedEdges;
	SortedEdges = allocMatrix(nEdges+1,2);
	for (i=0;i<nEdges;i++){
		SortedEdges[i][0] = MyData[i].x;
		SortedEdges[i][1] = MyData[i].y;
	}
	SortedEdges[nEdges][0] = MyData[0].x;
	SortedEdges[nEdges][1] = MyData[0].y;

	double Area=0;
	for (i=0;i<nEdges;i++){
		Area += 0.5*((SortedEdges[i][0]*SortedEdges[i+1][1])-(SortedEdges[i+1][0]*SortedEdges[i][1]));
	}
	free(MyData);
	FreeMemMatrix(SortedEdges,nEdges+1);
	return Area;
}

static inline
int FindUniques (double **EdgesIn, double **EdgesOut, int nEdgesIn, double RMin, double RMax, double EtaMin, double EtaMax){
	int i,j, nEdgesOut=0, duplicate;
	double Len, RT,ET;
	for (i=0;i<nEdgesIn;i++){
		duplicate = 0;
		for (j=i+1;j<nEdgesIn;j++){
			Len = sqrt((EdgesIn[i][0]-EdgesIn[j][0])*(EdgesIn[i][0]-EdgesIn[j][0])+(EdgesIn[i][1]-EdgesIn[j][1])*(EdgesIn[i][1]-EdgesIn[j][1]));
			if (Len ==0){
				duplicate = 1;
			}
		}
		RT = sqrt(EdgesIn[i][0]*EdgesIn[i][0] + EdgesIn[i][1]*EdgesIn[i][1]);
		ET = CalcEtaAngle(EdgesIn[i][0],EdgesIn[i][1]);
		if (fabs(ET - EtaMin) > 180 || fabs(ET - EtaMax) > 180){
			if (EtaMin < 0) ET = ET - 360;
			else ET = 360 + ET;
		}
		if (BETWEEN(RT,RMin,RMax) == 0){
			duplicate = 1;
		}
		if (BETWEEN(ET,EtaMin,EtaMax) == 0){
			duplicate = 1;
		}
		if (duplicate == 0){
			EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
			EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
			nEdgesOut++;
		}
	}
	return nEdgesOut;
}

struct data {
	int y;
	int z;
	double frac;
};

static inline
long long int
mapperfcn(
	double tx,
	double ty,
	double tz,
	int NrPixelsY,
	int NrPixelsZ,
	double pxY,
	double Ycen,
	double Zcen,
	double Lsd,
	double RhoD,
	double p0,
	double p1,
	double p2,
	double p3,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh,
	int nRBins,
	int nEtaBins,
	struct data ***pxList,
	int **nPxList,
	int **maxnPx)
{
	double txr, tyr, tzr;
	txr = deg2rad*tx;
	tyr = deg2rad*ty;
	tzr = deg2rad*tz;
	double Rx[3][3] = {{1,0,0},{0,cos(txr),-sin(txr)},{0,sin(txr),cos(txr)}};
	double Ry[3][3] = {{cos(tyr),0,sin(tyr)},{0,1,0},{-sin(tyr),0,cos(tyr)}};
	double Rz[3][3] = {{cos(tzr),-sin(tzr),0},{sin(tzr),cos(tzr),0},{0,0,1}};
	double TRint[3][3], TRs[3][3];
	MatrixMultF33(Ry,Rz,TRint);
	MatrixMultF33(Rx,TRint,TRs);
	double n0=2.0, n1=4.0, n2=2.0;
	double *RetVals, *RetVals2;
	RetVals = malloc(2*sizeof(*RetVals));
	RetVals2 = malloc(2*sizeof(*RetVals2));
	double Y, Z, Eta, Rt;
	int i,j,k,l,m;
	double EtaMi, EtaMa, RMi, RMa;
	int RChosen[500], EtaChosen[500];
	int nrRChosen, nrEtaChosen;
	double EtaMiTr, EtaMaTr;
	double YZ[2];
	double **Edges;
	Edges = allocMatrix(50,2);
	double **EdgesOut;
	EdgesOut = allocMatrix(50,2);
	int nEdges;
	double RMin, RMax, EtaMin, EtaMax;
	double yMin, yMax, zMin, zMax;
	double boxEdge[4][2];
	double Area;
	double RThis, EtaThis;
	double yTemp, zTemp, yTempMin, yTempMax, zTempMin, zTempMax;
	int maxnVal, nVal;
	struct data *oldarr, *newarr;
	long long int TotNrOfBins = 0;
	// long long int sumNrBins = 0;
	// long long int nrContinued=0;
	long long int testPos;
	double ypr,zpr;
	// double RT, ET;
	for (i=0;i<NrPixelsY;i++){
		for (j=0;j<NrPixelsZ;j++){
			EtaMi = 1800;
			EtaMa = -1800;
			RMi = 1E8; // In pixels
			RMa = -1000;
			// Calculate RMi, RMa, EtaMi, EtaMa
			testPos = j;
			testPos *= NrPixelsY;
			testPos += i;
			ypr = (double)i + distortionMapY[testPos];
			zpr = (double)j + distortionMapZ[testPos];
			for (k = 0; k < 2; k++){
				for (l = 0; l < 2; l++){
					Y = ypr + dy[k];
					Z = zpr + dz[l];
					REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
					Eta = RetVals[0];
					Rt = RetVals[1]; // in pixels
					if (Eta < EtaMi) EtaMi = Eta;
					if (Eta > EtaMa) EtaMa = Eta;
					if (Rt < RMi) RMi = Rt;
					if (Rt > RMa) RMa = Rt;
				}
			}
			// Get corrected Y, Z for this position.
			REta4MYZ(ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
			Eta = RetVals[0];
			Rt = RetVals[1]; // in pixels
			YZ4mREta(Rt,Eta,RetVals2);
			YZ[0] = RetVals2[0]; // Corrected Y position according to R, Eta, center at 0,0
			YZ[1] = RetVals2[1]; // Corrected Z position according to R, Eta, center at 0,0
			// Now check which eta, R ranges should have this pixel
			nrRChosen = 0;
			nrEtaChosen = 0;
			for (k=0;k<nRBins;k++){
				if (  RBinsHigh[k] >=   RMi &&   RBinsLow[k] <=   RMa){
					RChosen[nrRChosen] = k;
					nrRChosen ++;
				}
			}
			for (k=0;k<nEtaBins;k++){ // If Eta is smaller than 0, check for eta, eta+360, if eta is greater than 0, check for eta, eta-360
				// First check if the pixel is a special case
				if (EtaMa - EtaMi > 180){
					EtaMiTr = EtaMa;
					EtaMaTr = 360 + EtaMi;
					EtaMa = EtaMaTr;
					EtaMi = EtaMiTr;
				}
				if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)){
					EtaChosen[nrEtaChosen] = k;
					nrEtaChosen++;
					continue;
				}
				if (EtaMi < 0){
					EtaMi += 360;
					EtaMa += 360;
				} else {
					EtaMi -= 360;
					EtaMa -= 360;
				}
				if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)){
					EtaChosen[nrEtaChosen] = k;
					nrEtaChosen++;
					continue;
				}
			}
			yMin = YZ[0] - 0.5;
			yMax = YZ[0] + 0.5;
			zMin = YZ[1] - 0.5;
			zMax = YZ[1] + 0.5;
			// sumNrBins += nrRChosen * nrEtaChosen;
			// double totPxArea = 0;
			// Line Intercepts ordering: RMin: ymin, ymax, zmin, zmax. RMax: ymin, ymax, zmin, zmax
			//							 EtaMin: ymin, ymax, zmin, zmax. EtaMax: ymin, ymax, zmin, zmax.
			for (k=0;k<nrRChosen;k++){
				RMin = RBinsLow[RChosen[k]];
				RMax = RBinsHigh[RChosen[k]];
				for (l=0;l<nrEtaChosen;l++){
					EtaMin = EtaBinsLow[EtaChosen[l]];
					EtaMax = EtaBinsHigh[EtaChosen[l]];
					// Find YZ of the polar mask.
					YZ4mREta(RMin,EtaMin,RetVals);
					boxEdge[0][0] = RetVals[0];
					boxEdge[0][1] = RetVals[1];
					YZ4mREta(RMin,EtaMax,RetVals);
					boxEdge[1][0] = RetVals[0];
					boxEdge[1][1] = RetVals[1];
					YZ4mREta(RMax,EtaMin,RetVals);
					boxEdge[2][0] = RetVals[0];
					boxEdge[2][1] = RetVals[1];
					YZ4mREta(RMax,EtaMax,RetVals);
					boxEdge[3][0] = RetVals[0];
					boxEdge[3][1] = RetVals[1];
					nEdges = 0;
					// Now check if any edge of the pixel is within the polar mask
					for (m=0;m<4;m++){
						RThis = sqrt((YZ[0]+PosMatrix[m][0])*(YZ[0]+PosMatrix[m][0])+(YZ[1]+PosMatrix[m][1])*(YZ[1]+PosMatrix[m][1]));
						EtaThis = CalcEtaAngle(YZ[0]+PosMatrix[m][0],YZ[1]+PosMatrix[m][1]);
						if (EtaMin < -180 && signVal(EtaThis) != signVal(EtaMin)) EtaThis -= 360;
						if (EtaMax >  180 && signVal(EtaThis) != signVal(EtaMax)) EtaThis += 360;
						if (RThis   >= RMin   && RThis   <= RMax &&
							EtaThis >= EtaMin && EtaThis <= EtaMax){
							Edges[nEdges][0] = YZ[0]+PosMatrix[m][0];
							Edges[nEdges][1] = YZ[1]+PosMatrix[m][1];
							nEdges++;
						}
					}
					for (m=0;m<4;m++){ // Check if any edge of the polar mask is within the pixel edges.
						if (boxEdge[m][0] >= yMin && boxEdge[m][0] <= yMax &&
							boxEdge[m][1] >= zMin && boxEdge[m][1] <= zMax){
								Edges[nEdges][0] = boxEdge[m][0];
								Edges[nEdges][1] = boxEdge[m][1];
								nEdges ++;
							}
					}
					if (nEdges < 4){
						// Now go through Rmin, Rmax, EtaMin, EtaMax and calculate intercepts and check if within the pixel.
						//RMin,Max and yMin,Max
						if (RMin >= yMin) {
							zTemp = signVal(YZ[1])*sqrt(RMin*RMin - yMin*yMin);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMin;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMin >= yMax) {
							zTemp = signVal(YZ[1])*sqrt(RMin*RMin - yMax*yMax);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMax;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMax >= yMin) {
							zTemp = signVal(YZ[1])*sqrt(RMax*RMax - yMin*yMin);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMin;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMax >= yMax) {
							zTemp = signVal(YZ[1])*sqrt(RMax*RMax - yMax*yMax);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMax;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						//RMin,Max and zMin,Max
						if (RMin >= zMin) {
							yTemp = signVal(YZ[0])*sqrt(RMin*RMin - zMin*zMin);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMin;
								nEdges++;
							}
						}
						if (RMin >= zMax) {
							yTemp = signVal(YZ[0])*sqrt(RMin*RMin - zMax*zMax);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMax;
								nEdges++;
							}
						}
						if (RMax >= zMin) {
							yTemp = signVal(YZ[0])*sqrt(RMax*RMax - zMin*zMin);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMin;
								nEdges++;
							}
						}
						if (RMax >= zMax) {
							yTemp = signVal(YZ[0])*sqrt(RMax*RMax - zMax*zMax);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMax;
								nEdges++;
							}
						}
						//EtaMin,Max and yMin,Max
						if (fabs(EtaMin) < 1E-5 || fabs(fabs(EtaMin)-180) < 1E-5){
							zTempMin = 0;
							zTempMax = 0;
						}else{
							zTempMin = -yMin/tan(EtaMin*deg2rad);
							zTempMax = -yMax/tan(EtaMin*deg2rad);
						}
						if (BETWEEN(zTempMin,zMin,zMax) == 1){
							Edges[nEdges][0] = yMin;
							Edges[nEdges][1] = zTempMin;
							nEdges++;
						}
						if (BETWEEN(zTempMax,zMin,zMax) == 1){
							Edges[nEdges][0] = yMax;
							Edges[nEdges][1] = zTempMax;
							nEdges++;
						}
						if (fabs(EtaMax) < 1E-5 || fabs(fabs(EtaMax)-180) < 1E-5){
							zTempMin = 0;
							zTempMax = 0;
						}else{
							zTempMin = -yMin/tan(EtaMax*deg2rad);
							zTempMax = -yMax/tan(EtaMax*deg2rad);
						}
						if (BETWEEN(zTempMin,zMin,zMax) == 1){
							Edges[nEdges][0] = yMin;
							Edges[nEdges][1] = zTempMin;
							nEdges++;
						}
						if (BETWEEN(zTempMax,zMin,zMax) == 1){
							Edges[nEdges][0] = yMax;
							Edges[nEdges][1] = zTempMax;
							nEdges++;
						}
						//EtaMin,Max and zMin,Max
						if (fabs(fabs(EtaMin)-90) < 1E-5){
							yTempMin = 0;
							yTempMax = 0;
						}else{
							yTempMin = -zMin*tan(EtaMin*deg2rad);
							yTempMax = -zMax*tan(EtaMin*deg2rad);
						}
						if (BETWEEN(yTempMin,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMin;
							Edges[nEdges][1] = zMin;
							nEdges++;
						}
						if (BETWEEN(yTempMax,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMax;
							Edges[nEdges][1] = zMax;
							nEdges++;
						}
						if (fabs(fabs(EtaMax)-90) < 1E-5){
							yTempMin = 0;
							yTempMax = 0;
						}else{
							yTempMin = -zMin*tan(EtaMax*deg2rad);
							yTempMax = -zMax*tan(EtaMax*deg2rad);
						}
						if (BETWEEN(yTempMin,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMin;
							Edges[nEdges][1] = zMin;
							nEdges++;
						}
						if (BETWEEN(yTempMax,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMax;
							Edges[nEdges][1] = zMax;
							nEdges++;
						}
					}
					if (nEdges < 3){
						// nrContinued++;
						continue;
					}
					nEdges = FindUniques(Edges,EdgesOut,nEdges,RMin,RMax,EtaMin,EtaMax);
					if (nEdges < 3){
						// nrContinued++;
						continue;
					}
					// Now we have all the edges, let's calculate the area.
					Area = CalcAreaPolygon(EdgesOut,nEdges);
					if (Area < 1E-5){
						// nrContinued++;
						continue;
					}
					// Populate the arrays
					maxnVal = maxnPx[RChosen[k]][EtaChosen[l]];
					nVal = nPxList[RChosen[k]][EtaChosen[l]];
					if (nVal >= maxnVal){
						maxnVal += 2;
						oldarr = pxList[RChosen[k]][EtaChosen[l]];
						newarr = realloc(oldarr, maxnVal*sizeof(*newarr));
						if (newarr == NULL){
							return 0;
						}
						pxList[RChosen[k]][EtaChosen[l]] = newarr;
						maxnPx[RChosen[k]][EtaChosen[l]] = maxnVal;
					}
					pxList[RChosen[k]][EtaChosen[l]][nVal].y = i;
					pxList[RChosen[k]][EtaChosen[l]][nVal].z = j;
					pxList[RChosen[k]][EtaChosen[l]][nVal].frac = Area;
					// totPxArea += Area;
					(nPxList[RChosen[k]][EtaChosen[l]])++;
					TotNrOfBins++;
				}
			}
		}
	}
	return TotNrOfBins;
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], double *ImageIn, double *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	int i,k,l;
	if (NrTransOpt == 0){
		memcpy(ImageOut,ImageIn,NrPixelsY*NrPixelsZ*sizeof(*ImageIn)); // Nothing to do
		return;
	}
    for (i=0;i<NrTransOpt;i++){
		if (TransOpt[i] == 1){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[l*NrPixelsY+(NrPixelsY-k-1)]; // Invert Y
				}
			}
		}else if (TransOpt[i] == 2){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[(NrPixelsZ-l-1)*NrPixelsY+k]; // Invert Z
				}
			}
		}
	}
}

// Function to print usage information and help
void print_usage() {
    printf("\nDetectorMapper - A tool for detector mapping and distortion correction\n");
    printf("Copyright (c) 2014, UChicago Argonne, LLC\n\n");
    printf("Usage: detectormapper parameter_file\n\n");
    printf("Parameter file format:\n");
    printf("  Text format (key value) or JSON format are supported.\n\n");
    printf("Required parameters:\n");
    printf("  tx           - Rotation around x-axis (degrees)\n");
    printf("  ty           - Rotation around y-axis (degrees)\n");
    printf("  tz           - Rotation around z-axis (degrees)\n");
    printf("  px           - Pixel size (microns) - will be used for both Y and Z if pxY/pxZ not specified\n");
    printf("  BC           - Beam center [Y, Z] coordinates in pixels\n");
    printf("  Lsd          - Sample to detector distance (mm)\n");
    printf("  RhoD         - Radius of distortion (pixels)\n");
    printf("  p0, p1, p2   - Distortion correction polynomial coefficients\n");
    printf("  p3           - Phase shift parameter (degrees)\n");
    printf("  NrPixels     - Number of pixels (will be used for both Y and Z if not specified separately)\n");
    printf("  EtaBinSize   - Size of eta bins (degrees)\n");
    printf("  EtaMin       - Minimum eta value (degrees)\n");
    printf("  EtaMax       - Maximum eta value (degrees)\n");
    printf("  RBinSize     - Size of R bins (pixels)\n");
    printf("  RMin         - Minimum R value (pixels)\n");
    printf("  RMax         - Maximum R value (pixels)\n\n");
    
    printf("Optional parameters:\n");
    printf("  pxY          - Pixel size in Y direction (microns)\n");
    printf("  pxZ          - Pixel size in Z direction (microns)\n");
    printf("  NrPixelsY    - Number of pixels in Y direction\n");
    printf("  NrPixelsZ    - Number of pixels in Z direction\n");
    printf("  ImTransOpt   - Image transformation options [array of integers]\n");
    printf("                 1: Invert Y, 2: Invert Z\n");
    printf("  DistortionFile - Path to distortion correction file\n\n");
    
    printf("Example parameter file (text format):\n");
    printf("  tx 0.0\n");
    printf("  ty 0.0\n");
    printf("  tz 0.0\n");
    printf("  px 200.0\n");
    printf("  BC [1024, 1024]\n");
    printf("  Lsd 1000.0\n");
    printf("  RhoD 1000.0\n");
    printf("  p0 0.1\n");
    printf("  p1 0.1\n");
    printf("  p2 0.1\n");
    printf("  p3 0.0\n");
    printf("  NrPixels 2048\n");
    printf("  EtaBinSize 1.0\n");
    printf("  EtaMin -180.0\n");
    printf("  EtaMax 180.0\n");
    printf("  RBinSize 1.0\n");
    printf("  RMin 0.0\n");
    printf("  RMax 1000.0\n\n");
    
    printf("Output files:\n");
    printf("  Map.bin  - Binary mapping data\n");
    printf("  nMap.bin - Binary mapping index data\n");
    printf("  <parameter_file>.json - JSON representation of input parameters\n\n");
}

int main(int argc, char *argv[])
{
    // Add the help function
    if (argc != 2){
        print_usage();
        return 1;
    }
    
    clock_t start0, end0;
    start0 = clock();
    double diftotal;
    char *ParamFN;
    ParamFN = argv[1];

    // Check if help was requested
    if (strcmp(ParamFN, "-h") == 0 || strcmp(ParamFN, "--help") == 0) {
        print_usage();
        return 0;
    }

    ConfigParams* config = readConfigFile(argv[1]);
    if (config == NULL) {
        fprintf(stderr, "Error reading configuration file: %s\n", ParamFN);
        return 1;
    }

    double tx=config->tx; 
    double ty=config->ty;
    double tz=config->tz;
    double pxY=config->pxY;
    double yCen=config->yCen;
    double zCen=config->zCen;
    double Lsd=config->Lsd;
    double RhoD=config->RhoD;
    double p0=config->p0;
    double p1=config->p1;
    double p2=config->p2;
    double p3=config->p3;
    double EtaBinSize=config->EtaBinSize;
    double RBinSize=config->RBinSize;
    double RMax=config->RMax;
    double RMin=config->RMin;
    double EtaMax=config->EtaMax;
    double EtaMin=config->EtaMin;
    int NrPixelsY=config->NrPixelsY;
    int NrPixelsZ=config->NrPixelsZ;
    char* distortionFN = config->DistortionFile;
    distortionFile = (config->DistortionFile != NULL) ? 1 : 0;
    int NrTransOpt = config->NrTransOpt;
    int TransOpt[10];
    for (int i=0;i<NrTransOpt;i++) TransOpt[i] = config->ImTransOpt[i];
    printf("Configuration loaded:\n");
    printf("tx: %.6f\n", config->tx);
    printf("ty: %.6f\n", config->ty);
    printf("tz: %.6f\n", config->tz);
    printf("px: %.6f\n", config->px);
    printf("pxY: %.6f\n", config->pxY);
    printf("pxZ: %.6f\n", config->pxZ);
    printf("yCen: %.6f\n", config->yCen);
    printf("zCen: %.6f\n", config->zCen);
    printf("Lsd: %.6f\n", config->Lsd);
    printf("RhoD: %.6f\n", config->RhoD);
    printf("p0: %.6f\n", config->p0);
    printf("p1: %.6f\n", config->p1);
    printf("p2: %.6f\n", config->p2);
    printf("p3: %.6f\n", config->p3);
    printf("EtaBinSize: %.6f\n", config->EtaBinSize);
    printf("EtaMin: %.6f\n", config->EtaMin);
    printf("EtaMax: %.6f\n", config->EtaMax);
    printf("RBinSize: %.6f\n", config->RBinSize);
    printf("RMin: %.6f\n", config->RMin);
    printf("RMax: %.6f\n", config->RMax);
    printf("NrPixels: %d\n", config->NrPixels);
    printf("NrPixelsY: %d\n", config->NrPixelsY);
    printf("NrPixelsZ: %d\n", config->NrPixelsZ);
    
    // Print ImTransOpt as an array
    printf("ImTransOpt: [");
    for (int i = 0; i < config->NrTransOpt; i++) {
        printf("%d", config->ImTransOpt[i]);
        if (i < config->NrTransOpt - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("DistortionFile: %s\n", config->DistortionFile ? config->DistortionFile : "NULL");
    freeConfigParams(config);

    distortionMapY = calloc(NrPixelsY*NrPixelsZ,sizeof(double));
	distortionMapZ = calloc(NrPixelsY*NrPixelsZ,sizeof(double));
	if (distortionFile == 1){
		FILE *distortionFileHandle = fopen(distortionFN,"rb");
		double *distortionMapTemp;
		distortionMapTemp = malloc(NrPixelsY*NrPixelsZ*sizeof(double));
		fread(distortionMapTemp,NrPixelsY*NrPixelsZ*sizeof(double),1,distortionFileHandle);
		DoImageTransformations(NrTransOpt,TransOpt,distortionMapTemp,distortionMapY,NrPixelsY,NrPixelsZ);
		fread(distortionMapTemp,NrPixelsY*NrPixelsZ*sizeof(double),1,distortionFileHandle);
		DoImageTransformations(NrTransOpt,TransOpt,distortionMapTemp,distortionMapZ,NrPixelsY,NrPixelsZ);
		printf("Distortion file %s was provided and read correctly.\n",distortionFN);
	}
    // Parameters needed: Rmax RMin RBinSize (px) EtaMax EtaMin EtaBinSize (degrees)
	int nEtaBins, nRBins;
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int)ceil((EtaMax - EtaMin)/EtaBinSize);
	printf("Creating a mapper for integration.\nNumber of eta bins: %d, number of R bins: %d.\n",nEtaBins,nRBins);
	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = malloc(nRBins*sizeof(*RBinsHigh));
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);
	// Initialize arrays, need fraction array
	struct data ***pxList;
	int **nPxList;
	int **maxnPx;
	pxList = malloc(nRBins * sizeof(pxList));
	nPxList = malloc(nRBins * sizeof(nPxList));
	maxnPx = malloc(nRBins * sizeof(maxnPx));
	int i,j,k;
	for (i=0;i<nRBins;i++){
		pxList[i] = malloc(nEtaBins*sizeof(pxList[i]));
		nPxList[i] = malloc(nEtaBins*sizeof(nPxList[i]));
		maxnPx[i] = malloc(nEtaBins*sizeof(maxnPx[i]));
		for (j=0;j<nEtaBins;j++){
			pxList[i][j] = NULL;
			nPxList[i][j] = 0;
			maxnPx[i][j] = 0;
		}
	}
    // Parameters needed: tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen, Lsd, RhoD, p0, p1, p2
    long long int TotNrOfBins = mapperfcn(tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, yCen,
								zCen, Lsd, RhoD, p0, p1, p2, p3, EtaBinsLow,
								EtaBinsHigh, RBinsLow, RBinsHigh, nRBins,
								nEtaBins, pxList, nPxList, maxnPx);
	printf("Total Number of bins %lld\n",TotNrOfBins); fflush(stdout);
	long long int LengthNPxList = nRBins * nEtaBins;
	struct data *pxListStore;
	int *nPxListStore;
	pxListStore = malloc(TotNrOfBins*sizeof(*pxListStore));
	nPxListStore = malloc(LengthNPxList*2*sizeof(*nPxListStore));
	long long int Pos;
	int localNPxVal, localCounter = 0;
	for (i=0;i<nRBins;i++){
		for (j=0;j<nEtaBins;j++){
			localNPxVal = nPxList[i][j];
			Pos = i*nEtaBins;
			Pos += j;
			nPxListStore[(Pos*2)+0] = localNPxVal;
			nPxListStore[(Pos*2)+1] = localCounter;
			for (k=0;k<localNPxVal;k++){
				pxListStore[localCounter+k].y = pxList[i][j][k].y;
				pxListStore[localCounter+k].z = pxList[i][j][k].z;
				pxListStore[localCounter+k].frac = pxList[i][j][k].frac;
			}
			localCounter += localNPxVal;
		}
	}

	// Write out
	char *mapfn = "Map.bin";
	char *nmapfn = "nMap.bin";
	FILE *mapfile = fopen(mapfn,"wb");
	FILE *nmapfile = fopen(nmapfn,"wb");
	fwrite(pxListStore,TotNrOfBins*sizeof(*pxListStore),1,mapfile);
	fwrite(nPxListStore,LengthNPxList*2*sizeof(*nPxListStore),1,nmapfile);

	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
}
