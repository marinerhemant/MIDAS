//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define float32_t float
#define MAX_N_OVERLAPS 155000
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
    LoGFilt[i] =
        (int)(79720 * (-1 / (M_PI * (sigma * sigma * sigma * sigma))) *
              (1 - (((FiltXYs[i][0] * FiltXYs[i][0]) +
                     (FiltXYs[i][1] * FiltXYs[i][1])) /
                    (2 * sigma * sigma))) *
              (exp(-((FiltXYs[i][0] * FiltXYs[i][0]) +
                     (FiltXYs[i][1] * FiltXYs[i][1])) /
                   (2 * sigma * sigma)))); // What is the 79720 part??????
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
  int offsets[8] = {-1,
                    1,
                    -NrPixelsY,
                    NrPixelsY,
                    -NrPixelsY - 1,
                    -NrPixelsY + 1,
                    NrPixelsY - 1,
                    NrPixelsY + 1};
  for (i = NrPixelsY + 1; i < (nTotalPixels - (NrPixelsY + 1)); i++) {
    if (Image2[i] != 0) {
      long val = Image3[i];
      for (int k = 0; k < 8; k++) {
        long neighbor = Image3[i + offsets[k]];
        if ((val >= 0 && neighbor < 0) || (val < 0 && neighbor >= 0)) {
          SetBit(ImageEdges, i);
          break;
        }
      }
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
  ImagePeakIDsCorrected = malloc(nTotalPixels * sizeof(*ImagePeakIDsCorrected));
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
          Pos = PosFill - NrPixelsY + 1; // check if the pixel below to the
                                         // right is an edge, if it is, fill it.
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
          Pos = PosFill + NrPixelsY + 1; // check if the pixel above to the
                                         // right is an edge, if it is, fill it.
          if (TestBit(ImageEdges, Pos)) {
            SetBit(FilledEdges, Pos);
            Image4[Pos] = PeakNumber;
          }
          PosF2 = PosFill + 1;
          if ((PosFill + 1) % NrPixelsY == 0 ||
              (PosFill + 1) % NrPixelsY == 1 ||
              (TestBit(ImageEdges, PosF2) &&
               ImagePeakIDsCorrected[PosFill + 1] !=
                   OrigImagePeakID)) { // If we are filling the edges of the
                                       // image, should not continue!
            if ((PosFill + 1) % NrPixelsY == 0 ||
                (PosFill + 1) % NrPixelsY == 1) {
              printf("Peak loop not closed. Reached image edge at index %d "
                     "(row %d, col %d).\n",
                     PosFill, PosFill / NrPixelsY, PosFill % NrPixelsY);
            }
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
                                    int **BoolImage, int **ConnectedComponents,
                                    int **Positions, int *PositionTrackers) {
  if (x < 0 || x == NrPixelsZ)
    return;
  if (y < 0 || y == NrPixelsY)
    return;
  if ((ConnectedComponents[x][y] != 0) || (BoolImage[x][y] == 0))
    return;

  ConnectedComponents[x][y] = current_label;
  if (Positions != NULL && PositionTrackers != NULL) {
    Positions[current_label][PositionTrackers[current_label]] =
        (x * NrPixelsY) + y;
    PositionTrackers[current_label] += 1;
  }
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
  printf("ImageProcessing: usage: ./ImageProcessing <ParameterFile> <blockNr> "
         "<nBlocks> <numProcs>\n");
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    usage();
    return 1;
  }
  double start_time = omp_get_wtime();

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  char fn2[1000], fn[1000], direct[1000], extOrig[1000], extReduced[1000],
      ReducedFileName[1024];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  int LowNr, StartNr, NrFilesPerLayer, NrPixels = 2048, BlanketSubtraction,
                                       MeanFiltRadius, WriteFinImage = 0;
  int NrPixelsY = 0, NrPixelsZ = 0;
  int LoGMaskRadius, doDeblur = 0;
  double sigma;
  int DoLoGFilter = 1, WFImages = 0, nDistances, stNr = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "StartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &stNr);
      continue;
    }
    str = "nDistances ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nDistances);
      continue;
    }
    str = "RawStartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
    str = "LoGMaskRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &LoGMaskRadius);
      continue;
    }
    str = "WFImages ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WFImages);
      continue;
    }
    str = "BlanketSubtraction ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &BlanketSubtraction);
      continue;
    }
    str = "DoLoGFilter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &DoLoGFilter);
      continue;
    }
    str = "GaussFiltRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &sigma);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixels);
      continue;
    }
    str = "WriteFinImage ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WriteFinImage);
      continue;
    }
    str = "NrFilesPerDistance ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrFilesPerLayer);
      continue;
    }
    str = "MedFiltRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MeanFiltRadius);
      continue;
    }
    str = "OrigFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
      continue;
    }
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, ReducedFileName);
      continue;
    }
    str = "extReduced ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, extReduced);
      continue;
    }
    str = "extOrig ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, extOrig);
      continue;
    }
    str = "Deblur ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &doDeblur);
      continue;
    }
    str = "NrPixelsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      continue;
    }
    str = "NrPixelsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      continue;
    }
  }
  // Apply backward-compatible defaults
  if (NrPixelsY == 0 && NrPixelsZ == 0) {
    NrPixelsY = NrPixels;
    NrPixelsZ = NrPixels;
  } else if (NrPixelsY != 0 && NrPixelsZ == 0) {
    NrPixelsZ = NrPixelsY;
  } else if (NrPixelsY == 0 && NrPixelsZ != 0) {
    NrPixelsY = NrPixelsZ;
  }
  int nTotalPixels = NrPixelsY * NrPixelsZ;
  if (doDeblur != 0)
    WriteFinImage = 1;
  sprintf(fn, "%s/%s", direct, fn2);
  fclose(fileParam);

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("    ImageProcessingLibTiffOMP: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:        %s\n", direct);
  printf("  OrigFileName:         %s\n", fn2);
  printf("  ReducedFileName:      %s\n", ReducedFileName);
  printf("  extOrig:              %s\n", extOrig);
  printf("  extReduced:           %s\n", extReduced);
  printf("  Full input path:      %s\n", fn);
  printf("\n--- Image Parameters ---\n");
  printf("  NrPixels:             %d\n", NrPixels);
  printf("  NrPixelsY (cols):     %d\n", NrPixelsY);
  printf("  NrPixelsZ (rows):     %d\n", NrPixelsZ);
  printf("  nDistances:           %d\n", nDistances);
  printf("  NrFilesPerDistance:   %d\n", NrFilesPerLayer);
  printf("  WFImages:             %d\n", WFImages);
  printf("\n--- Scan Parameters ---\n");
  printf("  RawStartNr:           %d\n", StartNr);
  printf("  StartNr:              %d\n", stNr);
  printf("\n--- Filter Parameters ---\n");
  printf("  BlanketSubtraction:   %d\n", BlanketSubtraction);
  printf("  MedFiltRadius:        %d\n", MeanFiltRadius);
  printf("  DoLoGFilter:          %d\n", DoLoGFilter);
  printf("  LoGMaskRadius:        %d\n", LoGMaskRadius);
  printf("  GaussFiltRadius:      %f\n", sigma);
  printf("\n--- Flags ---\n");
  printf("  WriteFinImage:        %d\n", WriteFinImage);
  printf("  Deblur:               %d\n", doDeblur);
  printf(
      "================================================================\n\n");

  int nBlocks = atoi(argv[3]);
  int blockNr = atoi(argv[2]);
  int numProcs = atoi(argv[4]);
  int nrFilesTotal = nDistances * NrFilesPerLayer;
  int startNr, endNr;
  startNr = (int)(ceil((double)nrFilesTotal / (double)nBlocks)) * blockNr;
  int tmp = (int)(ceil((double)nrFilesTotal / (double)nBlocks)) * (blockNr + 1);
  endNr = tmp < (nrFilesTotal) ? tmp : (nrFilesTotal);
  printf("%d %d %d %d %d %d\n", nBlocks, blockNr, numProcs, nrFilesTotal,
         startNr, endNr);
  int fnr;
  pixelvalue *MedFltImg;
  int SizeFile = sizeof(pixelvalue) * nTotalPixels;
  MedFltImg = malloc(nTotalPixels * nDistances * sizeof(*MedFltImg));
  int dNr;
  for (dNr = 0; dNr < nDistances; dNr++) {
    char MedianFileName[1024];
    sprintf(MedianFileName, "%s/%s_Median_Background_Distance_%d.%s", direct,
            ReducedFileName, dNr, extReduced);
    FILE *MFI = fopen(MedianFileName, "r");
    if (MFI == NULL) {
      printf("Median file not found. Exiting\n");
      return 1;
    }
    fread(&MedFltImg[nTotalPixels * dNr], SizeFile, 1, MFI);
    fclose(MFI);
  }
  pixelvalue *Images, *Images2, *Images3, *FinalImages;
  size_t bigArrSz = numProcs;
  bigArrSz *= NrPixelsY;
  bigArrSz *= NrPixelsZ;
  Images = malloc(bigArrSz * sizeof(*Images));   // Original image.
  Images2 = malloc(bigArrSz * sizeof(*Images2)); // Median filtered image.
  Images3 = malloc(bigArrSz * sizeof(*Images3));
  FinalImages = malloc(bigArrSz * sizeof(*FinalImages));
// OMP
#pragma omp parallel for num_threads(numProcs) private(fnr)
  for (fnr = startNr; fnr < endNr; fnr++) {
    int procNum = omp_get_thread_num();
    int ImageNr, layerNr;
    layerNr = fnr / NrFilesPerLayer + 1;
    ImageNr = fnr % NrFilesPerLayer;
    int i, j, k;
    FILE *fk;
    char OutFN[5024];
    sprintf(OutFN, "%s/%s_%06d.%s%d", direct, ReducedFileName, ImageNr,
            extReduced, layerNr - 1);
    char OutFileName[5024];
    strcpy(OutFileName, OutFN);
    fk = fopen(OutFileName, "wb");

    pixelvalue *Image, *Image2;
    char FileName[1024];
    size_t offsetPos = procNum;
    offsetPos *= NrPixelsY;
    offsetPos *= NrPixelsZ;
    Image = &Images[offsetPos];
    Image2 = &Images2[offsetPos];

    // Use LibTiff to read files
    int FileNr = ((layerNr - 1) * NrFilesPerLayer) + StartNr + ImageNr;
    FileNr += (layerNr - 1) * WFImages;
    sprintf(FileName, "%s_%06d.%s", fn, FileNr, extOrig);
    // printf("%d %d %d %d %d %d\n", fnr, startNr, endNr, ImageNr, layerNr,
    //        procNum);
    fflush(stdout);
    TIFFErrorHandler oldhandler;
    oldhandler = TIFFSetWarningHandler(NULL);
    TIFF *tif = TIFFOpen(FileName, "r");
    if (tif == NULL) {
      printf("%s not found.", FileName);
      continue;
    }
    TIFFSetWarningHandler(oldhandler);
    int interInt;
    size_t startVal;
    if (tif) {
      tdata_t buf;
      buf = _TIFFmalloc(TIFFScanlineSize(tif));
      uint16 *datar;
      int rnr;
      for (rnr = 0; rnr < NrPixelsZ; rnr++) {
        TIFFReadScanline(tif, buf, rnr, 1);
        datar = (uint16 *)buf;
        for (i = 0; i < NrPixelsY; i++) {
          startVal = layerNr - 1;
          startVal *= NrPixelsY;
          startVal *= NrPixelsZ;
          startVal += rnr * NrPixelsY + i;
          interInt = (int)datar[i] - (int)MedFltImg[startVal] -
                     (int)BlanketSubtraction;
          Image[rnr * NrPixelsY + i] =
              (pixelvalue)(interInt > 0 ? interInt : 0);
        }
      }
      _TIFFfree(buf);
    }
    TIFFClose(tif);

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
      printf("Wrong MedFiltRadius!!! Not doing median.\n");
      for (i = 0; i < nTotalPixels; i++) {
        Image2[i] = Image[i];
      }
    }
    pixelvalue *FinalImage;
    FinalImage = &FinalImages[offsetPos];
    int TotPixelsInt = 0;
    memset(FinalImage, 0, nTotalPixels * sizeof(pixelvalue));
    //~ for (i=0;i<nTotalPixels;i++) FinalImage[i] = 0;
    if (DoLoGFilter == 1) {
      pixelvalue *Image3;
      Image3 = &Images3[offsetPos];
      for (i = 0; i < nTotalPixels; i++) {
        Image3[i] = Image2[i];
      }
      int *Image4;
      Image4 = malloc(nTotalPixels * sizeof(*Image4));
      FindPeakPositions(LoGMaskRadius, sigma, Image2, NrPixelsY, NrPixelsZ,
                        Image4);
      int LoGMaskRadius2 = 4;
      double sigma2 = 1;
      int *Image5;
      Image5 = malloc(nTotalPixels * sizeof(*Image5));
      FindPeakPositions(LoGMaskRadius2, sigma2, Image3, NrPixelsY, NrPixelsZ,
                        Image5);
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
      printf("DoLoGFilter=1: Total non-zero pixels (potential peaks) = %d in "
             "image %d, layer %d.\n",
             TotPixelsInt, ImageNr, layerNr);
      free(Image4);
      free(Image5);
    } else {
      for (i = 0; i < nTotalPixels; i++) {
        FinalImage[i] = (pixelvalue)Image2[i];
        if (Image2[i] != 0)
          TotPixelsInt++;
      }
      printf("DoLoGFilter=0: Found %d non-zero pixels (potential peaks) in "
             "image %d, layer %d.\n",
             TotPixelsInt, ImageNr, layerNr);
      int **BoolImage, **ConnectedComponents;
      BoolImage = allocMatrixInt(NrPixelsZ, NrPixelsY);
      ConnectedComponents = allocMatrixInt(NrPixelsZ, NrPixelsY);

      for (i = 0; i < NrPixelsZ; i++) {
        for (j = 0; j < NrPixelsY; j++) {
          if (Image2[(i * NrPixelsY) + j] != 0) {
            BoolImage[i][j] = 1;
          } else {
            BoolImage[i][j] = 0;
          }
        }
      }
      int NrOfReg;
      NrOfReg = FindConnectedComponents(BoolImage, NrPixelsY, NrPixelsZ,
                                        ConnectedComponents, NULL, NULL);

      printf(
          "DoLoGFilter=0: Found %d connected components (peaks) in image %d, "
          "layer %d.\n",
          NrOfReg, ImageNr, layerNr);

      FreeMemMatrixInt(BoolImage, NrPixelsZ);
      FreeMemMatrixInt(ConnectedComponents, NrPixelsZ);
    }
    if (TotPixelsInt > 0) {
      TotPixelsInt--;
    } else {
      TotPixelsInt = 1;
      FinalImage[2045] = 1;
    }
    //~ printf("Total number of pixels with intensity: %d\n",TotPixelsInt);
    int SizeOutFile = sizeof(pixelvalue) * nTotalPixels;
    char OutFN2[1024];
    if (WriteFinImage == 1) {
      FILE *fw;
      sprintf(OutFN2, "%s/%s_FullImage_%06d.%s%d", direct, ReducedFileName,
              ImageNr, extReduced, layerNr - 1);
      fw = fopen(OutFN2, "wb");
      fwrite(FinalImage, SizeOutFile, 1, fw);
      fclose(fw);
    }
    if (doDeblur != 0) {
      char *homedir = getenv("HOME");
      char *midasroot = getenv("MIDAS_INSTALL_DIR");
      if (midasroot == NULL)
        midasroot = getenv("MIDAS_HOME");
      char midas_path[4096];
      if (midasroot != NULL) {
        snprintf(midas_path, sizeof(midas_path), "%s", midasroot);
      } else {
        snprintf(midas_path, sizeof(midas_path), "%s/opt/MIDAS", homedir);
      }
      char cmmd[4096];
      char cmmd2[14096];
      sprintf(cmmd,
              "%s/opt/midasconda/bin/python "
              "%s/NF_HEDM/src/RLDeconv.py %s %d",
              homedir, midas_path, OutFN2, doDeblur);
      sprintf(cmmd2, "%s/NF_HEDM/bin/ParseDeconvOutput %s.tif %s %d",
              midas_path, OutFN2, OutFileName, NrPixelsY);
      //~ printf("%s\n%s\n",cmmd,cmmd2);
      system(cmmd);
      system(cmmd2);
    }
    pixelvalue *ys, *zs, *peakID;
    float32_t *intensity;
    ys = malloc(TotPixelsInt * 2 * sizeof(*ys));
    zs = malloc(TotPixelsInt * 2 * sizeof(*zs));
    peakID = malloc(TotPixelsInt * 2 * sizeof(*peakID));
    intensity = malloc(TotPixelsInt * 2 * sizeof(*intensity));
    int PeaksFilledCounter = 0;
    int RowNr, ColNr;
    for (i = 0; i < nTotalPixels; i++) {
      if (FinalImage[i] != 0) {
        peakID[PeaksFilledCounter] = FinalImage[i];
        intensity[PeaksFilledCounter] = Image[i];
        RowNr = i / NrPixelsY;
        ColNr = i % NrPixelsY;
        ys[PeaksFilledCounter] = NrPixelsY - 1 - ColNr;
        zs[PeaksFilledCounter] = NrPixelsZ - 1 - RowNr;
        PeaksFilledCounter++;
      }
    }
    // Write the result file.
    FILE *ft;
    char OutFNt[4096];
    sprintf(OutFNt, "%s.txtOld", OutFileName);
    ft = fopen(OutFNt, "w");
    fprintf(ft, "YPos\tZPos\tpeakID\tIntensity\n");
    for (i = 0; i < TotPixelsInt; i++) {
      fprintf(ft, "%d\t%d\t%d\t%lf\n", (int)ys[i], (int)zs[i], (int)peakID[i],
              (double)intensity[i]);
    }
    fclose(ft);

    if (doDeblur != 0)
      continue;

    float32_t dummy1 = 1;
    uint32_t dummy2 = 1;
    pixelvalue dummy3 = 1;
    char dummy4 = 'x';
    uint32_t DataSize16 = TotPixelsInt * 2 + dummy3;
    uint32_t DataSize32 = TotPixelsInt * 4 + dummy3;
    fwrite(&dummy1, sizeof(float32_t), 1, fk);
    // Start writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);      // uBlockHeader
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // BlockType
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // DataFormat
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NumChildren
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NameSize
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);  // DataSize
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // ChunkNumber
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // TotalChunks
    fwrite(&dummy4, sizeof(char) * dummy3, 1, fk); // BlockName
    // Finish writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    // Start writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);      // uBlockHeader
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // BlockType
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // DataFormat
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NumChildren
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NameSize
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);  // DataSize
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // ChunkNumber
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // TotalChunks
    fwrite(&dummy4, sizeof(char) * dummy3, 1, fk); // BlockName
    // Finish writing header.
    // Now write y positions.
    fwrite(ys, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    // Start writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);      // uBlockHeader
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // BlockType
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // DataFormat
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NumChildren
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NameSize
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);  // DataSize
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // ChunkNumber
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // TotalChunks
    fwrite(&dummy4, sizeof(char) * dummy3, 1, fk); // BlockName
    // Finish writing header.
    // Now write z positions.
    fwrite(zs, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    // Start writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);      // uBlockHeader
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // BlockType
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // DataFormat
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NumChildren
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NameSize
    fwrite(&DataSize32, sizeof(uint32_t), 1, fk);  // DataSize
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // ChunkNumber
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // TotalChunks
    fwrite(&dummy4, sizeof(char) * dummy3, 1, fk); // BlockName
    // Finish writing header.
    // Now write intensities.
    fwrite(intensity, TotPixelsInt * sizeof(float32_t), 1, fk);
    // Start writing header.
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);      // uBlockHeader
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // BlockType
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // DataFormat
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NumChildren
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // NameSize
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);  // DataSize
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // ChunkNumber
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);    // TotalChunks
    fwrite(&dummy4, sizeof(char) * dummy3, 1, fk); // BlockName
    // Finish writing header.
    // Now write PeakIDs.
    fwrite(peakID, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    fclose(fk);
    free(ys);
    free(zs);
    free(peakID);
    free(intensity);
  }
  free(MedFltImg);
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return 0;
}
