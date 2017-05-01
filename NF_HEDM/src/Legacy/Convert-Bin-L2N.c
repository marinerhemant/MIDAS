//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

/*
 * Convert-Bin-L2N.c
 *
 *  Created on: Apr 29, 2014
 *      Author: justin
 *  Modified as standalone program by Hemant Sharma on 2014/08/07.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <string.h>

#define float32_t float

#define CHECK(condition, msg) \
{ if (!(condition)) { printf("%s\n", msg); \
                      printf("\t At: %s:%i\n", __FILE__, __LINE__); \
                    return false; } }

#define READ(data, size, count, fp)           \
 { int actual = fread(data, size, count, fp); \
   CHECK(actual == count, "short read!");     \
 }

#define FGETS(string, count, fp)           \
    { char* t = fgets(string, count, fp);  \
      CHECK(t != NULL, "fgets failed!");   \
    }

struct Theader {
    uint32_t uBlockHeader;
    uint16_t BlockType;
    uint16_t DataFormat;
    uint16_t NumChildren;
    uint16_t NameSize;
    char BlockName[4096];
    uint32_t DataSize;
    uint16_t ChunkNumber;
    uint16_t TotalChunks;
};

static void
file_not_found(const char* filename)
{
    printf("Could not find file: %s\n", filename);
    fflush(NULL);
    exit(EXIT_FAILURE);
}

static void
file_not_writable(const char* filename)
{
    printf("Could not write to file: %s\n", filename);
    fflush(NULL);
    exit(EXIT_FAILURE);
}

void
hton_Uint16s(uint16_t  *data, int count) {
    int i;
    for (i = 0; i < count; i++)
        data[i] = htons(data[i]);
}
void
hton_Uint32s(uint32_t  *data, int count) {
    int i;
    for (i = 0; i < count; i++)
        data[i] = htonl(data[i]);
}

void
hton_Float32s(float32_t *data, int count) {
    int32_t t;
    int i;
    for (i = 0; i < count; i++) {
        // These memcpy's are required for correctness
        memcpy(&t, &data[i], 4);
        int32_t v = htonl(t);
        memcpy(&data[i], &v, 4);
    }
}

bool ReadHeader(
    FILE *fp,
    struct Theader * head)
{
    // printf("ReadHeader\n");
    READ(&head->uBlockHeader,sizeof(uint32_t),1,fp);
    READ(&head->BlockType,sizeof(uint16_t),1,fp);
    READ(&head->DataFormat,sizeof(uint16_t),1,fp);
    READ(&head->NumChildren,sizeof(uint16_t),1,fp);
    READ(&head->NameSize,sizeof(uint16_t),1,fp);
    READ(&head->DataSize,sizeof(uint32_t),1,fp);
    READ(&head->ChunkNumber,sizeof(uint16_t),1,fp);
    READ(&head->TotalChunks,sizeof(uint16_t),1,fp);
    READ(&head->BlockName,(sizeof(char)*(head->NameSize)),1,fp);
    head->BlockName[head->NameSize] = '\0';
    return true;
}

void NWriteHeader(
    FILE *fp,
    struct Theader * head)
{
    uint16_t t16;
    uint32_t t32;
    t32 = htonl(head->uBlockHeader);
    fwrite(&t32,sizeof(uint32_t),1,fp);
    t16 = htons(head->BlockType);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    t16 = htons(head->DataFormat);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    t16 = htons(head->NumChildren);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    t16 = htons(head->NameSize);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    t32 = htonl(head->DataSize);
    fwrite(&t32,sizeof(uint32_t),1,fp);
    t16 = htons(head->ChunkNumber);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    t16 = htons(head->TotalChunks);
    fwrite(&t16,sizeof(uint16_t),1,fp);
    fwrite(&head->BlockName,(sizeof(char)*(head->NameSize)),1,fp);
}

bool
L2N(const char *input, const char *output)
{
    FILE* fp_in = fopen(input, "r");
    if (fp_in == NULL) file_not_found(input);
    FILE* fp_out = fopen(output, "w");
    if (fp_out == NULL) file_not_writable(output);
    // setvbuf(stdout, NULL, _IOFBF, 4*1024*1024);

    struct Theader head;
    int nElements;

    bool b;

    // 1 dummy
    printf("Dummies\n");
    uint32_t dummy;
    READ(&dummy, sizeof(uint32_t), 1, fp_in);
    hton_Uint32s(&dummy, 1);
    fwrite(&dummy, sizeof(uint32_t), 1, fp_out);

    // 5 dummies
    b = ReadHeader(fp_in, &head);
    CHECK(b, "ReadHeader failed (dummies)!");
    NWriteHeader(fp_out, &head);
    uint32_t *t_ui32=NULL;
    t_ui32 = malloc(sizeof(uint32_t) * 5);
    READ(t_ui32, sizeof(uint32_t), 5, fp_in);
    hton_Uint32s(t_ui32, 5);
    fwrite(t_ui32, sizeof(uint32_t), 5, fp_out);

    // Y positions
    printf("Y positions\n");
    b = ReadHeader(fp_in, &head);
    CHECK(b, "ReadHeader failed (Y)!");
    NWriteHeader(fp_out, &head);
    nElements = (head.DataSize - head.NameSize) / sizeof(uint16_t);
    uint16_t *t_ui16 = malloc(nElements*sizeof(uint16_t));
    READ(t_ui16,sizeof(uint16_t),nElements,fp_in);
    hton_Uint16s(t_ui16, nElements);
    fwrite(t_ui16, sizeof(uint16_t), nElements, fp_out);

    // Z positions
    printf("Z positions\n");
    b = ReadHeader(fp_in,&head);
    CHECK(b, "ReadHeader failed (Z)!");
    NWriteHeader(fp_out, &head);
    int nCheck = (head.DataSize - head.NameSize) / sizeof(uint16_t);
    CHECK(nCheck == nElements, "size mismatch (Z)!");
    READ(t_ui16,sizeof(uint16_t),nElements,fp_in);
    hton_Uint16s(t_ui16, nElements);
    fwrite(t_ui16, sizeof(uint16_t), nElements, fp_out);

    // Intensities
    printf("Intensities\n");
    float32_t *t_f32 = malloc(sizeof(float32_t)*nElements);
    b = ReadHeader(fp_in,&head);
    CHECK(b, "ReadHeader failed (intensities)!");
    NWriteHeader(fp_out, &head);
    nCheck = (head.DataSize - head.NameSize) / sizeof(float32_t);
    CHECK(nCheck == nElements, "size mismatch (intensities)!");
    READ(t_f32,sizeof(float32_t),nElements,fp_in);
    hton_Float32s(t_f32, nElements);
    fwrite(t_f32, sizeof(float32_t), nElements, fp_out);

    // Peak IDs
    printf("Peak IDs\n");
    b = ReadHeader(fp_in,&head);
    CHECK(b, "ReadHeader failed (peak IDs)!");
    NWriteHeader(fp_out,&head);
    nCheck = (head.DataSize - head.NameSize) / sizeof(uint16_t);
    CHECK(nCheck == nElements, "size mismatch (peak IDs)!");
    READ(t_ui16,sizeof(uint16_t),nElements,fp_in);
    hton_Uint16s(t_ui16, nElements);
    fwrite(t_ui16, sizeof(uint16_t), nElements, fp_out);

    // Finish up
    free(t_ui16);
    free(t_ui32);
    free(t_f32);
    fclose(fp_in);
    fclose(fp_out);
    return true;
}

int
main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("requires 2 filenames: input and output filenames!\n");
        exit(1);
    }
    char* input  = argv[1];
    char* output = argv[2];

    bool result = L2N(input, output);
    if (!result) {
        printf("%s failed!\n", argv[0]);
        return 1;
    }
    return 0;
 }
