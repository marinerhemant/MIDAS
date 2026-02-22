//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// HDF implementation.
// Save output to a bin file and then would have to put it in the hdf later.

#include <ctype.h>
#include <fcntl.h>
#include <hdf5.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define float32_t float
#define MAX_N_OVERLAPS 355000
typedef uint16_t pixelvalue;

#define PIX_SORT(a, b)                                                         \
  {                                                                            \
    if ((a) > (b))                                                             \
      PIX_SWAP((a), (b));                                                      \
  }
#define PIX_SWAP(a, b)                                                         \
  {                                                                            \
    pixelvalue temp = (a);                                                     \
    (a) = (b);                                                                 \
    (b) = temp;                                                                \
  }
pixelvalue opt_med9(pixelvalue *p) {
  PIX_SORT(p[1], p[2]);
  PIX_SORT(p[4], p[5]);
  PIX_SORT(p[7], p[8]);
  PIX_SORT(p[0], p[1]);
  PIX_SORT(p[3], p[4]);
  PIX_SORT(p[6], p[7]);
  PIX_SORT(p[1], p[2]);
  PIX_SORT(p[4], p[5]);
  PIX_SORT(p[7], p[8]);
  PIX_SORT(p[0], p[3]);
  PIX_SORT(p[5], p[8]);
  PIX_SORT(p[4], p[7]);
  PIX_SORT(p[3], p[6]);
  PIX_SORT(p[1], p[4]);
  PIX_SORT(p[2], p[5]);
  PIX_SORT(p[4], p[7]);
  PIX_SORT(p[4], p[2]);
  PIX_SORT(p[6], p[4]);
  PIX_SORT(p[4], p[2]);
  return (p[4]);
}
pixelvalue opt_med25(pixelvalue *p) {
  PIX_SORT(p[0], p[1]);
  PIX_SORT(p[3], p[4]);
  PIX_SORT(p[2], p[4]);
  PIX_SORT(p[2], p[3]);
  PIX_SORT(p[6], p[7]);
  PIX_SORT(p[5], p[7]);
  PIX_SORT(p[5], p[6]);
  PIX_SORT(p[9], p[10]);
  PIX_SORT(p[8], p[10]);
  PIX_SORT(p[8], p[9]);
  PIX_SORT(p[12], p[13]);
  PIX_SORT(p[11], p[13]);
  PIX_SORT(p[11], p[12]);
  PIX_SORT(p[15], p[16]);
  PIX_SORT(p[14], p[16]);
  PIX_SORT(p[14], p[15]);
  PIX_SORT(p[18], p[19]);
  PIX_SORT(p[17], p[19]);
  PIX_SORT(p[17], p[18]);
  PIX_SORT(p[21], p[22]);
  PIX_SORT(p[20], p[22]);
  PIX_SORT(p[20], p[21]);
  PIX_SORT(p[23], p[24]);
  PIX_SORT(p[2], p[5]);
  PIX_SORT(p[3], p[6]);
  PIX_SORT(p[0], p[6]);
  PIX_SORT(p[0], p[3]);
  PIX_SORT(p[4], p[7]);
  PIX_SORT(p[1], p[7]);
  PIX_SORT(p[1], p[4]);
  PIX_SORT(p[11], p[14]);
  PIX_SORT(p[8], p[14]);
  PIX_SORT(p[8], p[11]);
  PIX_SORT(p[12], p[15]);
  PIX_SORT(p[9], p[15]);
  PIX_SORT(p[9], p[12]);
  PIX_SORT(p[13], p[16]);
  PIX_SORT(p[10], p[16]);
  PIX_SORT(p[10], p[13]);
  PIX_SORT(p[20], p[23]);
  PIX_SORT(p[17], p[23]);
  PIX_SORT(p[17], p[20]);
  PIX_SORT(p[21], p[24]);
  PIX_SORT(p[18], p[24]);
  PIX_SORT(p[18], p[21]);
  PIX_SORT(p[19], p[22]);
  PIX_SORT(p[8], p[17]);
  PIX_SORT(p[9], p[18]);
  PIX_SORT(p[0], p[18]);
  PIX_SORT(p[0], p[9]);
  PIX_SORT(p[10], p[19]);
  PIX_SORT(p[1], p[19]);
  PIX_SORT(p[1], p[10]);
  PIX_SORT(p[11], p[20]);
  PIX_SORT(p[2], p[20]);
  PIX_SORT(p[2], p[11]);
  PIX_SORT(p[12], p[21]);
  PIX_SORT(p[3], p[21]);
  PIX_SORT(p[3], p[12]);
  PIX_SORT(p[13], p[22]);
  PIX_SORT(p[4], p[22]);
  PIX_SORT(p[4], p[13]);
  PIX_SORT(p[14], p[23]);
  PIX_SORT(p[5], p[23]);
  PIX_SORT(p[5], p[14]);
  PIX_SORT(p[15], p[24]);
  PIX_SORT(p[6], p[24]);
  PIX_SORT(p[6], p[15]);
  PIX_SORT(p[7], p[16]);
  PIX_SORT(p[7], p[19]);
  PIX_SORT(p[13], p[21]);
  PIX_SORT(p[15], p[23]);
  PIX_SORT(p[7], p[13]);
  PIX_SORT(p[7], p[15]);
  PIX_SORT(p[1], p[9]);
  PIX_SORT(p[3], p[11]);
  PIX_SORT(p[5], p[17]);
  PIX_SORT(p[11], p[17]);
  PIX_SORT(p[9], p[17]);
  PIX_SORT(p[4], p[10]);
  PIX_SORT(p[6], p[12]);
  PIX_SORT(p[7], p[14]);
  PIX_SORT(p[4], p[6]);
  PIX_SORT(p[4], p[7]);
  PIX_SORT(p[12], p[14]);
  PIX_SORT(p[10], p[14]);
  PIX_SORT(p[6], p[7]);
  PIX_SORT(p[10], p[12]);
  PIX_SORT(p[6], p[10]);
  PIX_SORT(p[6], p[17]);
  PIX_SORT(p[12], p[17]);
  PIX_SORT(p[7], p[17]);
  PIX_SORT(p[7], p[10]);
  PIX_SORT(p[12], p[18]);
  PIX_SORT(p[7], p[12]);
  PIX_SORT(p[10], p[18]);
  PIX_SORT(p[12], p[20]);
  PIX_SORT(p[10], p[20]);
  PIX_SORT(p[10], p[12]);
  return (p[12]);
}
#undef PIX_SORT
#undef PIX_SWAP

int **allocMatrixInt(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
  }
  return arr;
}
void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

void FindPeakPositions(int LoGMaskRadius, double sigma, pixelvalue *Image2,
                       int NrPixelsY, int NrPixelsZ, int *Image4) {
  int nTotalPixels = NrPixelsY * NrPixelsZ;
  int i, j, k;
  int NrElsLoGMask = ((2 * LoGMaskRadius) + 1) * ((2 * LoGMaskRadius) + 1);
  long LoGFilt[NrElsLoGMask];
  double FiltXYs[NrElsLoGMask][2];
  int cr;
  cr = 0;
  for (i = -LoGMaskRadius; i <= LoGMaskRadius; i++) {
    for (j = -LoGMaskRadius; j <= LoGMaskRadius; j++) {
      FiltXYs[cr][0] = j;
      FiltXYs[cr][1] = i;
      cr++;
    }
  }
  for (i = 0; i < NrElsLoGMask; i++) {
    LoGFilt[i] = (int)(79720 * (-1 / (M_PI * (sigma * sigma * sigma * sigma))) *
                       (1 - (((FiltXYs[i][0] * FiltXYs[i][0]) +
                              (FiltXYs[i][1] * FiltXYs[i][1])) /
                             (2 * sigma * sigma))) *
                       (exp(-((FiltXYs[i][0] * FiltXYs[i][0]) +
                              (FiltXYs[i][1] * FiltXYs[i][1])) /
                            (2 * sigma * sigma))));
  }
  printf("Applying Laplacian of Gaussian Filter with Gauss Radius = %2.2f and "
         "LoG mask radius = %d.\n",
         sigma, LoGMaskRadius);
  long *Image3; // LoG filtered image.
  Image3 = malloc(nTotalPixels * sizeof(*Image3));
  long arrayLoG[NrElsLoGMask];
  int counter;
  int LoGFiltSize = LoGMaskRadius;
  for (i = 0; i < nTotalPixels; i++) {
    if (((i + 1) % NrPixelsY <= LoGFiltSize) ||
        ((NrPixelsY - ((i + 1) % NrPixelsY)) < LoGFiltSize) ||
        (i < (NrPixelsY * LoGFiltSize)) ||
        (i > (nTotalPixels - (NrPixelsY * LoGFiltSize)))) {
      Image3[i] = 0;
    } else {
      counter = 0;
      for (j = -LoGFiltSize; j <= LoGFiltSize; j++) {
        for (k = -LoGFiltSize; k <= LoGFiltSize; k++) {
          arrayLoG[counter] = (long)Image2[i + (NrPixelsY * j) + k];
          counter++;
        }
      }
      Image3[i] = 0;
      for (j = 0; j < NrElsLoGMask; j++) {
        Image3[i] += (LoGFilt[j] * arrayLoG[j]);
      }
    }
  }
  printf("Detecting edges.\n");
  int *ImageEdges; // Edges in LoG filtered image, but just binarized!
  ImageEdges = malloc((nTotalPixels * sizeof(*ImageEdges)) / 32);
  memset(ImageEdges, 0, (nTotalPixels * sizeof(*ImageEdges)) / 32);
  for (i = NrPixelsY + 1; i < (nTotalPixels - (NrPixelsY + 1)); i++) {
    if (Image2[i] != 0 &&
        ((Image3[i] < 0 && Image3[i - 1] >= 0) ||
         (Image3[i] >= 0 && Image3[i - 1] < 0) ||
         (Image3[i] == 0 && Image3[i - 1] < 0 && Image3[i + 1] > 0) ||
         (Image3[i] == 0 && Image3[i - 1] > 0 && Image3[i + 1] < 0) ||
         (Image3[i] < 0 && Image3[i - NrPixelsY] >= 0) ||
         (Image3[i] >= 0 && Image3[i - NrPixelsY] < 0) ||
         (Image3[i] == 0 && Image3[i - NrPixelsY] < 0 &&
          Image3[i + NrPixelsY] > 0) ||
         (Image3[i] == 0 && Image3[i - NrPixelsY] > 0 &&
          Image3[i + NrPixelsY] < 0))) {
      SetBit(ImageEdges, i);
    }
  }
  printf("Carrying out connected components labeling.\n");
  int PeakID = 1;
  int AlreadyFound = 0, Pos;
  int LargerVal, SmallerVal;
  int *ImagePeakIDs; // Edges from LoG filtered images, but not unique IDs!
  ImagePeakIDs = malloc(nTotalPixels * sizeof(*ImagePeakIDs));
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i)) {
      Pos = i - NrPixelsY - 1;
      if (TestBit(ImageEdges, Pos)) {
        ImagePeakIDs[i] = ImagePeakIDs[Pos];
        AlreadyFound = 1;
      }
      Pos = i - NrPixelsY;
      if (TestBit(ImageEdges, Pos)) {
        if (AlreadyFound == 0) {
          ImagePeakIDs[i] = ImagePeakIDs[Pos];
          AlreadyFound = 1;
        }
      }
      Pos = i - NrPixelsY + 1;
      if (TestBit(ImageEdges, Pos)) {
        if (AlreadyFound == 0) {
          ImagePeakIDs[i] = ImagePeakIDs[Pos];
          AlreadyFound = 1;
        }
      }
      Pos = i - 1;
      if (TestBit(ImageEdges, Pos)) {
        if (AlreadyFound == 0) {
          ImagePeakIDs[i] = ImagePeakIDs[Pos];
          AlreadyFound = 1;
        }
      }
      if (AlreadyFound == 0) {
        ImagePeakIDs[i] = PeakID;
        PeakID++;
      }
    } else {
      ImagePeakIDs[i] = 0;
    }
    AlreadyFound = 0;
  }
  int **PeakConnectedIDs; // Connected EdgeIDs!
  int *PeakConnectedIDsMap;
  int *PeakConnectedIDsMin;
  int *PeakConnectedIDsTracker;
  PeakConnectedIDs = allocMatrixInt(PeakID, MAX_N_OVERLAPS);
  PeakConnectedIDsMap = malloc(PeakID * sizeof(*PeakConnectedIDsMap));
  PeakConnectedIDsMin = malloc(PeakID * sizeof(*PeakConnectedIDsMin));
  PeakConnectedIDsTracker = malloc(PeakID * sizeof(*PeakConnectedIDsTracker));
  for (i = 0; i < PeakID; i++) {
    PeakConnectedIDsMap[i] = 0;
    PeakConnectedIDsTracker[i] = 0;
    PeakConnectedIDsMin[i] = 0;
  }
  int Smaller, Larger, ID1, ID2, RowNr, RowNr2, Number1, Number2, Min, TotNr;
  int LastRowNr = 1;
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i)) {
      Pos = i - NrPixelsY;
      if (TestBit(ImageEdges, Pos)) {
        if (ImagePeakIDs[i] != ImagePeakIDs[Pos]) {
          ID1 = ImagePeakIDs[i];
          ID2 = ImagePeakIDs[Pos];
          Smaller = ID1 < ID2 ? ID1 : ID2;
          Larger = ID1 > ID2 ? ID1 : ID2;
          RowNr = PeakConnectedIDsMap[ID1];
          RowNr2 = PeakConnectedIDsMap[ID2];
          if (RowNr != 0 && RowNr2 != 0 && RowNr == RowNr2) {
            continue;
          } else if (RowNr == 0 && RowNr2 != 0) {
            PeakConnectedIDsMap[ID1] = RowNr2;
            Number2 = PeakConnectedIDsTracker[RowNr2];
            PeakConnectedIDs[RowNr2][Number2] = ID1;
            Min = ID1;
            for (j = 0; j < Number2; j++) {
              if (PeakConnectedIDs[RowNr2][j] < Min) {
                Min = PeakConnectedIDs[RowNr2][j];
              }
            }
            PeakConnectedIDsMin[RowNr2] = Min;
            PeakConnectedIDsTracker[RowNr2] += 1;
          } else if (RowNr != 0 && RowNr2 == 0) {
            PeakConnectedIDsMap[ID2] = RowNr;
            Number1 = PeakConnectedIDsTracker[RowNr];
            PeakConnectedIDs[RowNr][Number1] = ID2;
            Min = ID2;
            for (j = 0; j < Number1; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] += 1;
          } else if (RowNr != 0 && RowNr2 != 0) {
            Number1 = PeakConnectedIDsTracker[RowNr];
            Number2 = PeakConnectedIDsTracker[RowNr2];
            TotNr = Number1 + Number2;
            for (j = 0; j < Number2; j++) {
              PeakConnectedIDs[RowNr][Number1 + j] =
                  PeakConnectedIDs[RowNr2][j];
              PeakConnectedIDsMap[PeakConnectedIDs[RowNr2][j]] = RowNr;
              PeakConnectedIDs[RowNr2][j] = 0;
            }
            Min = ID1;
            for (j = 0; j < TotNr; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] = TotNr;
          } else {
            PeakConnectedIDsMap[ID1] = LastRowNr;
            PeakConnectedIDsMap[ID2] = LastRowNr;
            PeakConnectedIDsMin[LastRowNr] = Smaller;
            PeakConnectedIDsTracker[LastRowNr] = 2;
            PeakConnectedIDs[LastRowNr][0] = Smaller;
            PeakConnectedIDs[LastRowNr][1] = Larger;
            LastRowNr++;
          }
        }
      }
      Pos = i - NrPixelsY + 1;
      if (TestBit(ImageEdges, Pos)) {
        if (ImagePeakIDs[i] != ImagePeakIDs[Pos]) {
          ID1 = ImagePeakIDs[i];
          ID2 = ImagePeakIDs[Pos];
          Smaller = ID1 < ID2 ? ID1 : ID2;
          Larger = ID1 > ID2 ? ID1 : ID2;
          RowNr = PeakConnectedIDsMap[ID1];
          RowNr2 = PeakConnectedIDsMap[ID2];
          if (RowNr != 0 && RowNr2 != 0 && RowNr == RowNr2) {
            continue;
          } else if (RowNr == 0 && RowNr2 != 0) {
            PeakConnectedIDsMap[ID1] = RowNr2;
            Number2 = PeakConnectedIDsTracker[RowNr2];
            PeakConnectedIDs[RowNr2][Number2] = ID1;
            Min = ID1;
            for (j = 0; j < Number2; j++) {
              if (PeakConnectedIDs[RowNr2][j] < Min) {
                Min = PeakConnectedIDs[RowNr2][j];
              }
            }
            PeakConnectedIDsMin[RowNr2] = Min;
            PeakConnectedIDsTracker[RowNr2] += 1;
          } else if (RowNr != 0 && RowNr2 == 0) {
            PeakConnectedIDsMap[ID2] = RowNr;
            Number1 = PeakConnectedIDsTracker[RowNr];
            PeakConnectedIDs[RowNr][Number1] = ID2;
            Min = ID2;
            for (j = 0; j < Number1; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] += 1;
          } else if (RowNr != 0 && RowNr2 != 0) {
            Number1 = PeakConnectedIDsTracker[RowNr];
            Number2 = PeakConnectedIDsTracker[RowNr2];
            TotNr = Number1 + Number2;
            for (j = 0; j < Number2; j++) {
              PeakConnectedIDs[RowNr][Number1 + j] =
                  PeakConnectedIDs[RowNr2][j];
              PeakConnectedIDsMap[PeakConnectedIDs[RowNr2][j]] = RowNr;
              PeakConnectedIDs[RowNr2][j] = 0;
            }
            Min = ID1;
            for (j = 0; j < TotNr; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] = TotNr;
          } else {
            PeakConnectedIDsMap[ID1] = LastRowNr;
            PeakConnectedIDsMap[ID2] = LastRowNr;
            PeakConnectedIDsMin[LastRowNr] = Smaller;
            PeakConnectedIDsTracker[LastRowNr] = 2;
            PeakConnectedIDs[LastRowNr][0] = Smaller;
            PeakConnectedIDs[LastRowNr][1] = Larger;
            LastRowNr++;
          }
        }
      }
      Pos = i - 1;
      if (TestBit(ImageEdges, Pos)) {
        if (ImagePeakIDs[i] != ImagePeakIDs[Pos]) {
          ID1 = ImagePeakIDs[i];
          ID2 = ImagePeakIDs[Pos];
          Smaller = ID1 < ID2 ? ID1 : ID2;
          Larger = ID1 > ID2 ? ID1 : ID2;
          RowNr = PeakConnectedIDsMap[ID1];
          RowNr2 = PeakConnectedIDsMap[ID2];
          if (RowNr != 0 && RowNr2 != 0 && RowNr == RowNr2) {
            continue;
          } else if (RowNr == 0 && RowNr2 != 0) {
            PeakConnectedIDsMap[ID1] = RowNr2;
            Number2 = PeakConnectedIDsTracker[RowNr2];
            PeakConnectedIDs[RowNr2][Number2] = ID1;
            Min = ID1;
            for (j = 0; j < Number2; j++) {
              if (PeakConnectedIDs[RowNr2][j] < Min) {
                Min = PeakConnectedIDs[RowNr2][j];
              }
            }
            PeakConnectedIDsMin[RowNr2] = Min;
            PeakConnectedIDsTracker[RowNr2] += 1;
          } else if (RowNr != 0 && RowNr2 == 0) {
            PeakConnectedIDsMap[ID2] = RowNr;
            Number1 = PeakConnectedIDsTracker[RowNr];
            PeakConnectedIDs[RowNr][Number1] = ID2;
            Min = ID2;
            for (j = 0; j < Number1; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] += 1;
          } else if (RowNr != 0 && RowNr2 != 0) {
            Number1 = PeakConnectedIDsTracker[RowNr];
            Number2 = PeakConnectedIDsTracker[RowNr2];
            TotNr = Number1 + Number2;
            for (j = 0; j < Number2; j++) {
              PeakConnectedIDs[RowNr][Number1 + j] =
                  PeakConnectedIDs[RowNr2][j];
              PeakConnectedIDsMap[PeakConnectedIDs[RowNr2][j]] = RowNr;
              PeakConnectedIDs[RowNr2][j] = 0;
            }
            Min = ID1;
            for (j = 0; j < TotNr; j++) {
              if (PeakConnectedIDs[RowNr][j] < Min) {
                Min = PeakConnectedIDs[RowNr][j];
              }
            }
            PeakConnectedIDsMin[RowNr] = Min;
            PeakConnectedIDsTracker[RowNr] = TotNr;
          } else {
            PeakConnectedIDsMap[ID1] = LastRowNr;
            PeakConnectedIDsMap[ID2] = LastRowNr;
            PeakConnectedIDsMin[LastRowNr] = Smaller;
            PeakConnectedIDsTracker[LastRowNr] = 2;
            PeakConnectedIDs[LastRowNr][0] = Smaller;
            PeakConnectedIDs[LastRowNr][1] = Larger;
            LastRowNr++;
          }
        }
      }
    }
  }
  int *ImagePeakIDsCorrected; // Corrected unique EdgeIDs!
  ImagePeakIDsCorrected =
      malloc(nTotalPixels * sizeof(*ImagePeakIDsCorrected));
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i)) {
      if (PeakConnectedIDsMap[ImagePeakIDs[i]] != 0) {
        ImagePeakIDsCorrected[i] =
            PeakConnectedIDsMin[PeakConnectedIDsMap[ImagePeakIDs[i]]];
      } else {
        ImagePeakIDsCorrected[i] = ImagePeakIDs[i];
      }
    }
  }
  memset(Image4, 0, nTotalPixels * sizeof(int));
  int *FilledEdges; // Edges in LoG filtered image, but just binarized!
  FilledEdges = malloc((nTotalPixels * sizeof(*FilledEdges)) / 32);
  memset(FilledEdges, 0, nTotalPixels * sizeof(int) / 32);
  printf("Filling peaks.\n");
  int FoundInsidePx = 0;
  int RowAbove = 1;
  int InsidePxPos = 0;
  int FoundEdgePx = 0;
  int PxPositionsToFlood[900];
  int NrPositionsToFlood;
  int PosI, PosInter, LeftOutPxFound, EdgePxMet, PeakNumber, StartInsidePxPos,
      RightEdgeMet, PosFill, PosFillI;
  PeakNumber = 1;
  int Flip = 0;
  int PosNext = 0;
  int PosTr = 0;
  int DummyCntr = 1, OrigImagePeakID, PosF2 = 0;
  int Flip2 = 0;
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i) &&
        !TestBit(FilledEdges,
                 i)) { // We have found a pixel belonging to a peak eadge.
      PosI = i;
      while (FoundInsidePx == 0) { // Until we have found an inside pixel.
        FoundEdgePx = 0;
        Pos = PosI + NrPixelsY - 1;
        if (TestBit(ImageEdges, Pos) &&
            !TestBit(FilledEdges, i)) { // If the pixel above on the left is an
                                        // edge, check for an inside pixel.
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges,
                       PosNext)) { // If the next pixel is not an edge.
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        Pos = PosI + NrPixelsY;
        if (TestBit(ImageEdges, Pos) &&
            !TestBit(FilledEdges, i)) { // If the pixel above is an edge, check
                                        // for an inside pixel.
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges,
                       PosNext)) { // If the next pixel is not an edge.
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        Pos = PosI + NrPixelsY + 1;
        if (TestBit(ImageEdges, Pos) &&
            !TestBit(FilledEdges, i)) { // If the pixel above on the right is an
                                        // edge, check for an inside pixel.
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges,
                       PosNext)) { // If the next pixel is not an edge.
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        if (FoundEdgePx == 0) { // If no edge was found, do not continue.
          break;
        } else {
          if (FoundInsidePx == 0) { // If inside pixel was not found, but edge
                                    // pixel was, find the leftmost edge pixel.
            PosI = PosInter;
            LeftOutPxFound = 0;
            while (LeftOutPxFound == 0) {
              PosTr = PosI - 1;
              if (TestBit(ImageEdges, PosTr)) {
                PosI--;
              } else {
                LeftOutPxFound = 1;
              }
            }
          }
        }
      }
      if (FoundEdgePx == 0 || FoundInsidePx == 0) {
        continue;
      }
      // Run flood fill with the seed as InsidePxPos.
      for (j = 0; j < 900; j++) {
        PxPositionsToFlood[j] = 0;
      }
      OrigImagePeakID = ImagePeakIDsCorrected[i];
      PxPositionsToFlood[0] = InsidePxPos;
      NrPositionsToFlood = 1;
      while (NrPositionsToFlood > 0) {
        StartInsidePxPos = PxPositionsToFlood[0];
        NrPositionsToFlood--;
        for (j = 1; j < 900; j++) {
          if (PxPositionsToFlood[j] == 0) {
            break;
          } else {
            PxPositionsToFlood[j - 1] = PxPositionsToFlood[j];
          }
        }
        if (StartInsidePxPos == 0) {
          continue;
        }
        PosFill = StartInsidePxPos;
        Flip = 0;
        while (DummyCntr == 1) { // Go towards left until edge is met!
          PosFill = PosFill - 1;
          if (TestBit(ImageEdges, PosFill)) {
            Flip = 1;
          }
          if (!TestBit(ImageEdges, PosFill) && Flip == 1) {
            PosFill++;
            break;
          }
        }
        RightEdgeMet = 0;
        PosFillI = PosFill;
        Flip2 = 0;
        while (RightEdgeMet == 0) { // Go towards right until edge is met!
          if (Image2[PosFill] < 1) {
            break;
          }
          Image4[PosFill] = PeakNumber;        // Now fill the pixel.
          if (!TestBit(ImageEdges, PosFill)) { // To make sure we have at least
                                               // found one inside pixel.
            Flip2 = 1;
          }
          if (TestBit(ImageEdges,
                      PosFill)) { // If the current pixel was an edge, record
                                  // that it has been processed.
            SetBit(FilledEdges, PosFill);
          }
          Pos = PosFill - NrPixelsY - 1; // check if the pixel below to the left
                                        // is an edge, if it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          Pos = PosFill - NrPixelsY; // check if the pixel below is an edge, if
                                    // it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          Pos = PosFill - NrPixelsY + 1; // check if the pixel below to the right
                                        // is an edge, if it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          Pos = PosFill + NrPixelsY - 1; // check if the pixel above to the left
                                        // is an edge, if it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          Pos = PosFill + NrPixelsY; // check if the pixel above is an edge, if
                                    // it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          Pos = PosFill + NrPixelsY + 1; // check if the pixel above to the right
                                        // is an edge, if it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          PosF2 = PosFill + 1;
          if ((PosFill + 1) % NrPixelsY == 0 || (PosFill + 1) % NrPixelsY == 1 ||
              (TestBit(ImageEdges, PosF2) &&
               ImagePeakIDsCorrected[PosFill + 1] !=
                   OrigImagePeakID)) { // If we are filling the edges of the
                                       // image, should not continue!
            while (!TestBit(ImageEdges, PosFill) &&
                   (Image4[PosFill - NrPixelsY] == 0) &&
                   (Image4[PosFill + NrPixelsY] == 0)) {
              Image4[PosFill] = 0;
              PosFill--;
            }
            break;
          }
          if (TestBit(ImageEdges, PosFill) &&
              Flip2 == 1) { // This means we have gone thru at least one inside
                            // pixel and we have found an edge on the right.
            RightEdgeMet = 1;
          }
          if (TestBit(ImageEdges, (PosFill + NrPixelsY - 1)) &&
              !TestBit(ImageEdges, (PosFill + NrPixelsY)) &&
              Image2[PosFill + NrPixelsY - 1] < Image2[PosFill + NrPixelsY]) {
            PxPositionsToFlood[NrPositionsToFlood] = PosFill + NrPixelsY;
            NrPositionsToFlood++;
          }
          PosFill = PosFill + 1;
        }
        while (TestBit(ImageEdges, PosFill)) {

          Image4[PosFill] = PeakNumber;
          SetBit(FilledEdges, PosFill);
          PosFill++;
        }
      }
      PeakNumber++;
    }
    FoundInsidePx = 0;
    RowAbove = 1;
    FoundEdgePx = 0;
  }
  printf("Total Number of Peaks = %d\n", PeakNumber - 1);
  free(Image3);
  free(ImageEdges);
  free(ImagePeakIDs);
  FreeMemMatrixInt(PeakConnectedIDs, PeakID);
  free(PeakConnectedIDsMap);
  free(PeakConnectedIDsMin);
  free(PeakConnectedIDsTracker);
  free(ImagePeakIDsCorrected);
  free(FilledEdges);
}

const int dx[] = {+1, 0, -1, 0, +1, -1, +1, -1};
const int dy[] = {0, +1, 0, -1, +1, +1, -1, -1};

static inline void DepthFirstSearch(int x, int y, int current_label,
                                    int NrPixelsY, int NrPixelsZ,
                                    int **BoolImage,
                                    int **ConnectedComponents, int **Positions,
                                    int *PositionTrackers) {
  if (x < 0 || x == NrPixelsZ)
    return;
  if (y < 0 || y == NrPixelsY)
    return;
  if ((ConnectedComponents[x][y] != 0) || (BoolImage[x][y] == 0))
    return;

  ConnectedComponents[x][y] = current_label;
  Positions[current_label][PositionTrackers[current_label]] =
      (x * NrPixelsY) + y;
  PositionTrackers[current_label] += 1;
  int direction;
  for (direction = 0; direction < 8; ++direction) {
    DepthFirstSearch(x + dx[direction], y + dy[direction], current_label,
                     NrPixelsY, NrPixelsZ, BoolImage, ConnectedComponents,
                     Positions, PositionTrackers);
  }
}

static inline int FindConnectedComponents(int **BoolImage, int NrPixelsY,
                                          int NrPixelsZ,
                                          int **ConnectedComponents,
                                          int **Positions,
                                          int *PositionTrackers) {
  int i, j;
  for (i = 0; i < NrPixelsZ; i++) {
    for (j = 0; j < NrPixelsY; j++) {
      ConnectedComponents[i][j] = 0;
    }
  }
  int component = 0;
  for (i = 0; i < NrPixelsZ; ++i) {
    for (j = 0; j < NrPixelsY; ++j) {
      if ((ConnectedComponents[i][j] == 0) && (BoolImage[i][j] == 1)) {
        DepthFirstSearch(i, j, ++component, NrPixelsY, NrPixelsZ, BoolImage,
                         ConnectedComponents, Positions, PositionTrackers);
      }
    }
  }
  return component;
}

static void usage(void) {
  printf("ImageProcessing: usage: ./ImageProcessing <Dataset.hdf> <DistanceNr> "
         "<FrameNr>\n");
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    usage();
    return 1;
  }
  clock_t start, end;
  double diftotal;
  start = clock();

  herr_t status, status_n;
  htri_t avail;
  hid_t file, dataset, dcpl;
  H5Z_filter_t filter_type;
  unsigned int flags, filter_info;
  size_t nelmts;
  avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (!avail) {
    printf("GZIP filter is not available. Will not be able to read the data "
           "file.\n");
    return 1;
  }
  status = H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info);
  if (!(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
      !(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED)) {
    printf("gzip filter not available for encoding and decoding.\n");
    return 1;
  }

  char *FileName, buffer[4096];
  int FrameNr, DistanceNr, LoGMaskRadius, BlanketSubtraction,
      DoLoGFilter = 1, NrPixels = 0, MeanFiltRadius /*MedianFiltRadius*/,
      doDeblur = 0, NrFilesPerDistance, NrDistances;
  int NrPixelsY = 0, NrPixelsZ = 0;
  double sigma /*GaussFiltRadius*/;

  FileName = argv[1];
  DistanceNr = atoi(argv[2]);
  FrameNr = atoi(argv[3]);
  file = H5Fopen(FileName, H5F_ACC_RDWR, H5P_DEFAULT);
  dataset = H5Dopen(file, "/measurement/nr_files_per_distance", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &NrFilesPerDistance);
  dataset = H5Dopen(file, "/measurement/nr_distances", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &NrDistances);
  dataset = H5Dopen(file, "/parameters/LoGMaskRadius", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &LoGMaskRadius);
  dataset = H5Dopen(file, "/parameters/BlanketSubtraction", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &BlanketSubtraction);
  dataset = H5Dopen(file, "/parameters/DoLoGFilter", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &DoLoGFilter);
  dataset = H5Dopen(file, "/measurement/instrument/detector/geometry/nr_pixels",
                    H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &NrPixels);
  // Try to read NrPixelsY and NrPixelsZ from HDF5
  if (H5Lexists(file, "/measurement/instrument/detector/geometry/nr_pixels_y",
                H5P_DEFAULT) > 0) {
    dataset = H5Dopen(
        file, "/measurement/instrument/detector/geometry/nr_pixels_y",
        H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &NrPixelsY);
  }
  if (H5Lexists(file, "/measurement/instrument/detector/geometry/nr_pixels_z",
                H5P_DEFAULT) > 0) {
    dataset = H5Dopen(
        file, "/measurement/instrument/detector/geometry/nr_pixels_z",
        H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &NrPixelsZ);
  }
  // Backward-compatible defaults
  if (NrPixelsY == 0 && NrPixelsZ == 0) {
    NrPixelsY = NrPixels;
    NrPixelsZ = NrPixels;
  } else if (NrPixelsY != 0 && NrPixelsZ == 0) {
    NrPixelsZ = NrPixelsY;
  } else if (NrPixelsY == 0 && NrPixelsZ != 0) {
    NrPixelsY = NrPixelsZ;
  }
  int nTotalPixels = NrPixelsY * NrPixelsZ;
  dataset = H5Dopen(file, "/parameters/MedianFilterRadius", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &MeanFiltRadius);
  dataset = H5Dopen(file, "/parameters/NrOfDeblurIterations", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &doDeblur);
  dataset = H5Dopen(file, "/parameters/GaussFiltRadius", H5P_DEFAULT);
  status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   &sigma);

  int i, j, k;
  pixelvalue *Image, *Image2, *MedFltImg;
  Image = malloc(nTotalPixels * sizeof(*Image)); // Original image.
  Image2 =
      malloc(nTotalPixels * sizeof(*Image2)); // Median filtered image.
  MedFltImg = malloc(nTotalPixels * sizeof(*MedFltImg));
  pixelvalue **filedata;
  filedata = (pixelvalue **)malloc(NrPixelsZ * sizeof(pixelvalue *));
  filedata[0] = (pixelvalue *)malloc(nTotalPixels * sizeof(pixelvalue));
  for (i = 0; i < NrPixelsZ; i++)
    filedata[i] = filedata[0] + i * NrPixelsY;
  int FileFrameNr = FrameNr + NrFilesPerDistance * DistanceNr;
  sprintf(buffer, "/analysis/median_images/distance_%d", DistanceNr);
  dataset = H5Dopen(file, buffer, H5P_DEFAULT);
  dcpl = H5Dget_create_plist(dataset);
  filter_type =
      H5Pget_filter(dcpl, 0, &flags, &nelmts, NULL, 0, NULL, &filter_info);
  status = H5Dread(dataset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   filedata[0]);
  for (i = 0; i < NrPixelsZ; i++) {
    for (j = 0; j < NrPixelsY; j++) {
      MedFltImg[i * NrPixelsY + j] = filedata[i][j];
    }
  }
  sprintf(buffer, "/exchange/data/%d", FileFrameNr);
  hid_t dcpl2;
  H5Z_filter_t filter_type2;
  unsigned int flags2, filter_info2;
  size_t nelmts2;
  dataset = H5Dopen(file, buffer, H5P_DEFAULT);
  dcpl2 = H5Dget_create_plist(dataset);
  filter_type2 =
      H5Pget_filter(dcpl2, 0, &flags2, &nelmts2, NULL, 0, NULL, &filter_info2);
  status = H5Dread(dataset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   filedata[0]);
  int interInt;
  for (i = 0; i < NrPixelsZ; i++) {
    for (j = 0; j < NrPixelsY; j++) {
      interInt = (int)filedata[i][j] - (int)MedFltImg[i * NrPixelsY + j] -
                 (int)BlanketSubtraction;
      Image[i * NrPixelsY + j] = (pixelvalue)(interInt > 0 ? interInt : 0);
    }
  }

  printf("Applying median filter with radius = %d.\n", MeanFiltRadius);
  if (MeanFiltRadius == 1) {
    pixelvalue array[9];
    for (i = 0; i < nTotalPixels; i++) {
      if (((i + 1) % NrPixelsY <= MeanFiltRadius) ||
          ((NrPixelsY - ((i + 1) % NrPixelsY)) < MeanFiltRadius) ||
          (i < (NrPixelsY * MeanFiltRadius)) ||
          (i > (nTotalPixels - (NrPixelsY * MeanFiltRadius)))) {
        Image2[i] = Image[i];
      } else {
        int countr = 0;
        for (j = -MeanFiltRadius; j <= MeanFiltRadius; j++) {
          for (k = -MeanFiltRadius; k <= MeanFiltRadius; k++) {
            array[countr] = Image[i + (NrPixelsY * j) + k];
            countr++;
          }
        }
        Image2[i] = opt_med9(array);
      }
    }
  } else if (MeanFiltRadius == 2) {
    pixelvalue array[25];
    for (i = 0; i < nTotalPixels; i++) {
      if (((i + 1) % NrPixelsY <= MeanFiltRadius) ||
          ((NrPixelsY - ((i + 1) % NrPixelsY)) < MeanFiltRadius) ||
          (i < (NrPixelsY * MeanFiltRadius)) ||
          (i > (nTotalPixels - (NrPixelsY * MeanFiltRadius)))) {
        Image2[i] = Image[i];
      } else {
        int countr = 0;
        for (j = -MeanFiltRadius; j <= MeanFiltRadius; j++) {
          for (k = -MeanFiltRadius; k <= MeanFiltRadius; k++) {
            array[countr] = Image[i + (NrPixelsY * j) + k];
            countr++;
          }
        }
        Image2[i] = opt_med25(array);
      }
    }
  } else {
    printf("Wrong MedFiltRadius!!! Exiting.\n");
    return 0;
  }
  pixelvalue *Image3;
  Image3 =
      malloc(nTotalPixels * sizeof(*Image3)); // Median filtered image.
  for (i = 0; i < nTotalPixels; i++) {
    Image3[i] = Image2[i];
  }
  pixelvalue *FinalImage;
  FinalImage = malloc(nTotalPixels * sizeof(*FinalImage));
  int TotPixelsInt = 0;
  for (i = 0; i < nTotalPixels; i++)
    FinalImage[i] = 0;
  if (DoLoGFilter == 1) {
    int *Image4;
    Image4 = malloc(nTotalPixels * sizeof(*Image4));
    FindPeakPositions(LoGMaskRadius, sigma, Image2, NrPixelsY, NrPixelsZ, Image4);
    int LoGMaskRadius2 = 4;
    double sigma2 = 1;
    int *Image5;
    Image5 = malloc(nTotalPixels * sizeof(*Image5));
    FindPeakPositions(LoGMaskRadius2, sigma2, Image3, NrPixelsY, NrPixelsZ, Image5);
    free(Image2);
    free(Image3);
    for (i = 0; i < nTotalPixels; i++) {
      if (Image4[i] != 0) {
        FinalImage[i] = Image4[i] * 10;
        TotPixelsInt++;
      } else if (Image5[i] != 0) {
        FinalImage[i] = Image5[i] * 10;
        TotPixelsInt++;
      } else {
        FinalImage[i] = 0;
      }
    }
    free(Image4);
    free(Image5);
  } else {
    for (i = 0; i < nTotalPixels; i++) {
      FinalImage[i] = (pixelvalue)Image2[i];
      if (Image2[i] != 0)
        TotPixelsInt++;
    }
  }
  if (TotPixelsInt > 0) {
    TotPixelsInt--;
  } else {
    TotPixelsInt = 1;
    FinalImage[2045] = 1;
  }
  printf("Total number of pixels with intensity: %d\n", TotPixelsInt);

  // Write Reduced image
  hsize_t dims[2] = {NrPixelsZ, NrPixelsY},
          chunk[2] = {NrPixelsZ / 8 > 0 ? NrPixelsZ / 8 : 1, NrPixelsY / 8 > 0 ? NrPixelsY / 8 : 1};
  hid_t space = H5Screate_simple(2, dims, NULL);
  dcpl = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(dcpl, 4);
  status = H5Pset_chunk(dcpl, 2, chunk);
  sprintf(buffer, "/analysis/reduced_images/%d", FileFrameNr);
  int NrPx = 0;
  for (i = 0; i < NrPixelsZ; i++) {
    for (j = 0; j < NrPixelsY; j++) {
      filedata[i][j] = 0;
      if (FinalImage[i * NrPixelsY + j] != 0) {
        NrPx++;
        filedata[i][j] = Image[i * NrPixelsY + j];
      }
    }
  }
  printf("%d\n", NrPx);
  if (H5Lexists(file, buffer, H5P_DEFAULT) == 1) {
    printf("Dataset found, will update: %s\n", buffer);
    dataset = H5Dopen(file, buffer, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             filedata[0]);
    H5Dclose(dataset);
  } else {
    printf("Creating dataset: %s\n", buffer);
    dataset = H5Dcreate(file, buffer, H5T_NATIVE_USHORT, space, H5P_DEFAULT,
                        dcpl, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             filedata[0]);
    H5Dclose(dataset);
  }
  printf("Updating bit matrix.\n");
  int *ObsSpotsInfo;
  ObsSpotsInfo = (int *)malloc(nTotalPixels / 32 * sizeof(int));
  memset(ObsSpotsInfo, 0, nTotalPixels / 32 * sizeof(int));
  long long int binNr;
  int RowNr, ColNr, ys, zs;
  for (i = 0; i < nTotalPixels; i++) {
    if (FinalImage[i] != 0) {
      RowNr = i / NrPixelsY;
      ColNr = i % NrPixelsY;
      ys = NrPixelsY - 1 - ColNr;
      zs = NrPixelsZ - 1 - RowNr;
      binNr = ys * NrPixelsY + zs;
      SetBit(ObsSpotsInfo, binNr);
    }
  }
  dataset = H5Dopen(file, "/analysis/reduced_file", H5P_DEFAULT);
  hid_t fid, mid;
  hsize_t dim1[] = {nTotalPixels / 32};
  int FSPACE_RANK = 2;
  hsize_t fdim[] = {nTotalPixels / 32, NrFilesPerDistance * NrDistances};
  fid = H5Screate_simple(FSPACE_RANK, fdim, NULL);
  hsize_t startpos[2], stride[2], count[2], block[2];
  startpos[0] = 0;
  startpos[1] = FileFrameNr;
  stride[0] = 1;
  stride[1] = 1;
  count[0] = 1;
  count[1] = 1;
  block[0] = nTotalPixels / 32;
  block[1] = 1;
  herr_t ret;
  ret =
      H5Sselect_hyperslab(fid, H5S_SELECT_SET, startpos, stride, count, block);
  mid = H5Screate_simple(1, dim1, NULL);
  startpos[0] = 0;
  stride[0] = 1;
  count[0] = nTotalPixels / 32;
  block[0] = 1;
  ret =
      H5Sselect_hyperslab(mid, H5S_SELECT_SET, startpos, stride, count, block);
  ret = H5Dwrite(dataset, H5T_NATIVE_INT, mid, fid, H5P_DEFAULT, ObsSpotsInfo);

  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed in correcting image for layer %d, image %d: %f [s]\n",
         DistanceNr, FrameNr, diftotal);
  return 0;
}
