//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// ProcessImagesCombined: Combines MedianImageLibTiff and
// ImageProcessingLibTiffOMP into a single pass. Reads TIFFs once,
// computes median, then processes all images from memory.
//
// Usage: ./ProcessImagesCombined <ParameterFile> <LayerNr> [nCPUs]

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define float32_t float
#define MAX_N_OVERLAPS 155000
typedef uint16_t pixelvalue;

// --- Quick-select for temporal median (from MedianImageLibTiff) ---
#define PIX_SWAP_QS(a, b)                                                      \
  {                                                                            \
    pixelvalue temp = (a);                                                     \
    (a) = (b);                                                                 \
    (b) = temp;                                                                \
  }
static pixelvalue quick_select(pixelvalue a[], int n) {
  int low = 0, high = n - 1, median = (low + high) / 2;
  int middle, ll, hh;
  for (;;) {
    if (high <= low)
      return a[median];
    if (high == low + 1) {
      if (a[low] > a[high])
        PIX_SWAP_QS(a[low], a[high]);
      return a[median];
    }
    middle = (low + high) / 2;
    if (a[middle] > a[high])
      PIX_SWAP_QS(a[middle], a[high]);
    if (a[low] > a[high])
      PIX_SWAP_QS(a[low], a[high]);
    if (a[middle] > a[low])
      PIX_SWAP_QS(a[middle], a[low]);
    PIX_SWAP_QS(a[middle], a[low + 1]);
    ll = low + 1;
    hh = high;
    for (;;) {
      do
        ll++;
      while (a[low] > a[ll]);
      do
        hh--;
      while (a[hh] > a[low]);
      if (hh < ll)
        break;
      PIX_SWAP_QS(a[ll], a[hh]);
    }
    PIX_SWAP_QS(a[low], a[hh]);
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}
#undef PIX_SWAP_QS

// --- Spatial median filters (from ImageProcessingLibTiffOMP) ---
#define PIX_SORT(a, b)                                                         \
  {                                                                            \
    if ((a) > (b))                                                             \
      PIX_SWAP_SP((a), (b));                                                   \
  }
#define PIX_SWAP_SP(a, b)                                                      \
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
#undef PIX_SWAP_SP

// --- Utility: matrix alloc/free ---
int **allocMatrixInt(int nrows, int ncols) {
  int **arr = malloc(nrows * sizeof(*arr));
  for (int i = 0; i < nrows; i++)
    arr[i] = malloc(ncols * sizeof(*arr[i]));
  return arr;
}
void FreeMemMatrixInt(int **mat, int nrows) {
  for (int r = 0; r < nrows; r++)
    free(mat[r]);
  free(mat);
}

// --- FindPeakPositions (LoG filter + edge + flood fill) ---
// This is copied verbatim from ImageProcessingLibTiffOMP.c
void FindPeakPositions(int LoGMaskRadius, double sigma, pixelvalue *Image2,
                       int NrPixelsY, int NrPixelsZ, int *Image4) {
  int nTotalPixels = NrPixelsY * NrPixelsZ;
  int i, j, k;
  int NrElsLoGMask = ((2 * LoGMaskRadius) + 1) * ((2 * LoGMaskRadius) + 1);
  long LoGFilt[NrElsLoGMask];
  double FiltXYs[NrElsLoGMask][2];
  int cr = 0;
  for (i = -LoGMaskRadius; i <= LoGMaskRadius; i++)
    for (j = -LoGMaskRadius; j <= LoGMaskRadius; j++) {
      FiltXYs[cr][0] = j;
      FiltXYs[cr][1] = i;
      cr++;
    }
  for (i = 0; i < NrElsLoGMask; i++)
    LoGFilt[i] = (int)(79720 * (-1 / (M_PI * (sigma * sigma * sigma * sigma))) *
                       (1 - (((FiltXYs[i][0] * FiltXYs[i][0]) +
                              (FiltXYs[i][1] * FiltXYs[i][1])) /
                             (2 * sigma * sigma))) *
                       (exp(-((FiltXYs[i][0] * FiltXYs[i][0]) +
                              (FiltXYs[i][1] * FiltXYs[i][1])) /
                            (2 * sigma * sigma))));
  long *Image3 = malloc(nTotalPixels * sizeof(*Image3));
  int LoGFiltSize = LoGMaskRadius;
  for (i = 0; i < nTotalPixels; i++) {
    if (((i + 1) % NrPixelsY <= LoGFiltSize) ||
        ((NrPixelsY - ((i + 1) % NrPixelsY)) < LoGFiltSize) ||
        (i < (NrPixelsY * LoGFiltSize)) ||
        (i > (nTotalPixels - (NrPixelsY * LoGFiltSize)))) {
      Image3[i] = 0;
    } else {
      long arrayLoG[NrElsLoGMask];
      int counter = 0;
      for (j = -LoGFiltSize; j <= LoGFiltSize; j++)
        for (k = -LoGFiltSize; k <= LoGFiltSize; k++)
          arrayLoG[counter++] = (long)Image2[i + (NrPixelsY * j) + k];
      Image3[i] = 0;
      for (j = 0; j < NrElsLoGMask; j++)
        Image3[i] += (LoGFilt[j] * arrayLoG[j]);
    }
  }
  // Edge detection
  int *ImageEdges = malloc((nTotalPixels * sizeof(*ImageEdges)) / 32);
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
      for (int k2 = 0; k2 < 8; k2++) {
        long neighbor = Image3[i + offsets[k2]];
        if ((val >= 0 && neighbor < 0) || (val < 0 && neighbor >= 0)) {
          SetBit(ImageEdges, i);
          break;
        }
      }
    }
  }
  // Connected components labeling
  int PeakID = 1, AlreadyFound = 0, Pos;
  int *ImagePeakIDs = malloc(nTotalPixels * sizeof(*ImagePeakIDs));
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i)) {
      Pos = i - NrPixelsY - 1;
      if (TestBit(ImageEdges, Pos)) {
        ImagePeakIDs[i] = ImagePeakIDs[Pos];
        AlreadyFound = 1;
      }
      Pos = i - NrPixelsY;
      if (TestBit(ImageEdges, Pos) && AlreadyFound == 0) {
        ImagePeakIDs[i] = ImagePeakIDs[Pos];
        AlreadyFound = 1;
      }
      Pos = i - NrPixelsY + 1;
      if (TestBit(ImageEdges, Pos) && AlreadyFound == 0) {
        ImagePeakIDs[i] = ImagePeakIDs[Pos];
        AlreadyFound = 1;
      }
      Pos = i - 1;
      if (TestBit(ImageEdges, Pos) && AlreadyFound == 0) {
        ImagePeakIDs[i] = ImagePeakIDs[Pos];
        AlreadyFound = 1;
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
  // Merge connected edge IDs
  int **PeakConnectedIDs = allocMatrixInt(PeakID, MAX_N_OVERLAPS);
  int *PeakConnectedIDsMap = malloc(PeakID * sizeof(*PeakConnectedIDsMap));
  int *PeakConnectedIDsMin = malloc(PeakID * sizeof(*PeakConnectedIDsMin));
  int *PeakConnectedIDsTracker =
      malloc(PeakID * sizeof(*PeakConnectedIDsTracker));
  for (i = 0; i < PeakID; i++) {
    PeakConnectedIDsMap[i] = 0;
    PeakConnectedIDsTracker[i] = 0;
    PeakConnectedIDsMin[i] = 0;
  }
  int Smaller, Larger, ID1, ID2, RowNr, RowNr2, Number1, Number2, Min, TotNr,
      LastRowNr = 1;
  for (i = 0; i < nTotalPixels; i++) {
    if (!TestBit(ImageEdges, i))
      continue;
    // Check 3 neighbors: above, above-right, left
    int checkPositions[3] = {i - NrPixelsY, i - NrPixelsY + 1, i - 1};
    for (int ci = 0; ci < 3; ci++) {
      Pos = checkPositions[ci];
      if (!TestBit(ImageEdges, Pos) || ImagePeakIDs[i] == ImagePeakIDs[Pos])
        continue;
      ID1 = ImagePeakIDs[i];
      ID2 = ImagePeakIDs[Pos];
      Smaller = ID1 < ID2 ? ID1 : ID2;
      Larger = ID1 > ID2 ? ID1 : ID2;
      RowNr = PeakConnectedIDsMap[ID1];
      RowNr2 = PeakConnectedIDsMap[ID2];
      if (RowNr != 0 && RowNr2 != 0 && RowNr == RowNr2)
        continue;
      if (RowNr == 0 && RowNr2 != 0) {
        PeakConnectedIDsMap[ID1] = RowNr2;
        Number2 = PeakConnectedIDsTracker[RowNr2];
        PeakConnectedIDs[RowNr2][Number2] = ID1;
        Min = ID1;
        for (j = 0; j < Number2; j++)
          if (PeakConnectedIDs[RowNr2][j] < Min)
            Min = PeakConnectedIDs[RowNr2][j];
        PeakConnectedIDsMin[RowNr2] = Min;
        PeakConnectedIDsTracker[RowNr2]++;
      } else if (RowNr != 0 && RowNr2 == 0) {
        PeakConnectedIDsMap[ID2] = RowNr;
        Number1 = PeakConnectedIDsTracker[RowNr];
        PeakConnectedIDs[RowNr][Number1] = ID2;
        Min = ID2;
        for (j = 0; j < Number1; j++)
          if (PeakConnectedIDs[RowNr][j] < Min)
            Min = PeakConnectedIDs[RowNr][j];
        PeakConnectedIDsMin[RowNr] = Min;
        PeakConnectedIDsTracker[RowNr]++;
      } else if (RowNr != 0 && RowNr2 != 0) {
        Number1 = PeakConnectedIDsTracker[RowNr];
        Number2 = PeakConnectedIDsTracker[RowNr2];
        TotNr = Number1 + Number2;
        for (j = 0; j < Number2; j++) {
          PeakConnectedIDs[RowNr][Number1 + j] = PeakConnectedIDs[RowNr2][j];
          PeakConnectedIDsMap[PeakConnectedIDs[RowNr2][j]] = RowNr;
          PeakConnectedIDs[RowNr2][j] = 0;
        }
        Min = ID1;
        for (j = 0; j < TotNr; j++)
          if (PeakConnectedIDs[RowNr][j] < Min)
            Min = PeakConnectedIDs[RowNr][j];
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
  int *ImagePeakIDsCorrected =
      malloc(nTotalPixels * sizeof(*ImagePeakIDsCorrected));
  for (i = 0; i < nTotalPixels; i++) {
    if (TestBit(ImageEdges, i)) {
      if (PeakConnectedIDsMap[ImagePeakIDs[i]] != 0)
        ImagePeakIDsCorrected[i] =
            PeakConnectedIDsMin[PeakConnectedIDsMap[ImagePeakIDs[i]]];
      else
        ImagePeakIDsCorrected[i] = ImagePeakIDs[i];
    }
  }
  // Flood fill
  memset(Image4, 0, nTotalPixels * sizeof(int));
  int *FilledEdges = malloc((nTotalPixels * sizeof(*FilledEdges)) / 32);
  memset(FilledEdges, 0, nTotalPixels * sizeof(int) / 32);
  int PeakNumber = 1;
  int FoundInsidePx, FoundEdgePx, InsidePxPos, PosI, PosInter, PosNext, PosTr;
  int LeftOutPxFound, RightEdgeMet, PosFill, PosFillI, PosF2;
  int PxPositionsToFlood[900], NrPositionsToFlood, StartInsidePxPos;
  int Flip, Flip2, DummyCntr = 1, OrigImagePeakID;
  for (i = 0; i < nTotalPixels; i++) {
    FoundInsidePx = 0;
    FoundEdgePx = 0;
    if (TestBit(ImageEdges, i) && !TestBit(FilledEdges, i)) {
      PosI = i;
      while (FoundInsidePx == 0) {
        FoundEdgePx = 0;
        Pos = PosI + NrPixelsY - 1;
        if (TestBit(ImageEdges, Pos) && !TestBit(FilledEdges, i)) {
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges, PosNext)) {
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        Pos = PosI + NrPixelsY;
        if (TestBit(ImageEdges, Pos) && !TestBit(FilledEdges, i)) {
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges, PosNext)) {
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        Pos = PosI + NrPixelsY + 1;
        if (TestBit(ImageEdges, Pos) && !TestBit(FilledEdges, i)) {
          FoundEdgePx = 1;
          PosInter = Pos;
          PosNext = Pos + 1;
          if (!TestBit(ImageEdges, PosNext)) {
            FoundInsidePx = 1;
            InsidePxPos = PosNext;
          }
        }
        if (FoundEdgePx == 0)
          break;
        if (FoundInsidePx == 0) {
          PosI = PosInter;
          LeftOutPxFound = 0;
          while (LeftOutPxFound == 0) {
            PosTr = PosI - 1;
            if (TestBit(ImageEdges, PosTr))
              PosI--;
            else
              LeftOutPxFound = 1;
          }
        }
      }
      if (FoundEdgePx == 0 || FoundInsidePx == 0)
        continue;
      for (j = 0; j < 900; j++)
        PxPositionsToFlood[j] = 0;
      OrigImagePeakID = ImagePeakIDsCorrected[i];
      PxPositionsToFlood[0] = InsidePxPos;
      NrPositionsToFlood = 1;
      while (NrPositionsToFlood > 0) {
        StartInsidePxPos = PxPositionsToFlood[0];
        NrPositionsToFlood--;
        for (j = 1; j < 900; j++) {
          if (PxPositionsToFlood[j] == 0)
            break;
          PxPositionsToFlood[j - 1] = PxPositionsToFlood[j];
        }
        if (StartInsidePxPos == 0)
          continue;
        PosFill = StartInsidePxPos;
        Flip = 0;
        while (DummyCntr == 1) {
          PosFill--;
          if (TestBit(ImageEdges, PosFill))
            Flip = 1;
          if (!TestBit(ImageEdges, PosFill) && Flip == 1) {
            PosFill++;
            break;
          }
        }
        RightEdgeMet = 0;
        PosFillI = PosFill;
        Flip2 = 0;
        while (RightEdgeMet == 0) {
          if (Image2[PosFill] < 1)
            break;
          Image4[PosFill] = PeakNumber;
          if (!TestBit(ImageEdges, PosFill))
            Flip2 = 1;
          if (TestBit(ImageEdges, PosFill))
            SetBit(FilledEdges, PosFill);
          // Fill neighbors
          int nbOffsets[6] = {-NrPixelsY - 1, -NrPixelsY, -NrPixelsY + 1,
                              NrPixelsY - 1,  NrPixelsY,  NrPixelsY + 1};
          for (int ni = 0; ni < 6; ni++) {
            Pos = PosFill + nbOffsets[ni];
            if (TestBit(ImageEdges, Pos)) {
              SetBit(FilledEdges, Pos);
              Image4[Pos] = PeakNumber;
            }
          }
          PosF2 = PosFill + 1;
          if ((PosFill + 1) % NrPixelsY == 0 ||
              (PosFill + 1) % NrPixelsY == 1 ||
              (TestBit(ImageEdges, PosF2) &&
               ImagePeakIDsCorrected[PosFill + 1] != OrigImagePeakID)) {
            while (!TestBit(ImageEdges, PosFill) &&
                   (Image4[PosFill - NrPixelsY] == 0) &&
                   (Image4[PosFill + NrPixelsY] == 0)) {
              Image4[PosFill] = 0;
              PosFill--;
            }
            break;
          }
          if (TestBit(ImageEdges, PosFill) && Flip2 == 1)
            RightEdgeMet = 1;
          if (TestBit(ImageEdges, (PosFill + NrPixelsY - 1)) &&
              !TestBit(ImageEdges, (PosFill + NrPixelsY)) &&
              Image2[PosFill + NrPixelsY - 1] < Image2[PosFill + NrPixelsY]) {
            PxPositionsToFlood[NrPositionsToFlood] = PosFill + NrPixelsY;
            NrPositionsToFlood++;
          }
          PosFill++;
        }
        while (TestBit(ImageEdges, PosFill)) {
          Image4[PosFill] = PeakNumber;
          SetBit(FilledEdges, PosFill);
          PosFill++;
        }
      }
      PeakNumber++;
    }
  }
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

// --- Connected components (for DoLoGFilter==0 path) ---
const int dx_cc[] = {+1, 0, -1, 0, +1, -1, +1, -1};
const int dy_cc[] = {0, +1, 0, -1, +1, +1, -1, -1};
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
  for (int d = 0; d < 8; ++d)
    DepthFirstSearch(x + dx_cc[d], y + dy_cc[d], current_label, NrPixelsY,
                     NrPixelsZ, BoolImage, ConnectedComponents, Positions,
                     PositionTrackers);
}
static inline int FindConnectedComponents(int **BoolImage, int NrPixelsY,
                                          int NrPixelsZ,
                                          int **ConnectedComponents,
                                          int **Positions,
                                          int *PositionTrackers) {
  for (int i = 0; i < NrPixelsZ; i++)
    for (int j = 0; j < NrPixelsY; j++)
      ConnectedComponents[i][j] = 0;
  int component = 0;
  for (int i = 0; i < NrPixelsZ; ++i)
    for (int j = 0; j < NrPixelsY; ++j)
      if ((ConnectedComponents[i][j] == 0) && (BoolImage[i][j] == 1))
        DepthFirstSearch(i, j, ++component, NrPixelsY, NrPixelsZ, BoolImage,
                         ConnectedComponents, Positions, PositionTrackers);
  return component;
}

// =====================================================================
//                              MAIN
// =====================================================================
int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    printf("ProcessImagesCombined: usage: ./ProcessImagesCombined "
           "<ParameterFile> <LayerNr> [nCPUs]\n");
    return 1;
  }
  double start = omp_get_wtime();
  int numProcs = 1;
  if (argc == 4) {
    numProcs = atoi(argv[3]);
    if (numProcs < 1)
      numProcs = 1;
  }

  // --- Parse parameter file ---
  char *ParamFN = argv[1];
  FILE *fileParam = fopen(ParamFN, "r");
  if (!fileParam) {
    printf("Cannot open %s\n", ParamFN);
    return 1;
  }
  char aline[1000], *str, dummy[1000];
  char fn2[1000], fn[1000], direct[1000], outputDir[1000];
  char extOrig[1000], extReduced[1000], ReducedFileName[1024];
  outputDir[0] = '\0';
  int LowNr, StartNr = 0, NrFilesPerLayer = 0, NrPixels = 2048;
  int NrPixelsY = 0, NrPixelsZ = 0;
  int BlanketSubtraction = 0, MeanFiltRadius = 1, WriteFinImage = 0;
  int LoGMaskRadius = 4, DoLoGFilter = 1, WFImages = 0, doDeblur = 0;
  double sigma = 1.0;
  int nLayers = atoi(argv[2]);

  while (fgets(aline, 1000, fileParam) != NULL) {
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
    str = "OutputDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, outputDir);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixels);
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
    str = "WFImages ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WFImages);
      continue;
    }
    str = "NrFilesPerDistance ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrFilesPerLayer);
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
    str = "extOrig ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, extOrig);
      continue;
    }
    str = "extReduced ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, extReduced);
      continue;
    }
    str = "BlanketSubtraction ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &BlanketSubtraction);
      continue;
    }
    str = "MedFiltRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MeanFiltRadius);
      continue;
    }
    str = "DoLoGFilter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &DoLoGFilter);
      continue;
    }
    str = "LoGMaskRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &LoGMaskRadius);
      continue;
    }
    str = "GaussFiltRadius ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &sigma);
      continue;
    }
    str = "WriteFinImage ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WriteFinImage);
      continue;
    }
    str = "Deblur ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &doDeblur);
      continue;
    }
  }
  fclose(fileParam);

  // Apply backward-compatible defaults
  if (NrPixelsY == 0 && NrPixelsZ == 0) {
    NrPixelsY = NrPixels;
    NrPixelsZ = NrPixels;
  } else if (NrPixelsY != 0 && NrPixelsZ == 0)
    NrPixelsZ = NrPixelsY;
  else if (NrPixelsY == 0 && NrPixelsZ != 0)
    NrPixelsY = NrPixelsZ;
  if (doDeblur != 0)
    WriteFinImage = 1;
  if (outputDir[0] == '\0')
    strcpy(outputDir, direct);

  StartNr = StartNr + (nLayers - 1) * WFImages;
  sprintf(fn, "%s/%s", direct, fn2);
  size_t nPixelsTotal = (size_t)NrPixelsY * NrPixelsZ;

  printf(
      "\n================================================================\n");
  printf("    ProcessImagesCombined: Parameters Summary\n");
  printf("================================================================\n");
  printf("  DataDirectory:      %s\n", direct);
  printf("  OutputDirectory:    %s\n", outputDir);
  printf("  OrigFileName:       %s\n", fn2);
  printf("  ReducedFileName:    %s\n", ReducedFileName);
  printf("  NrPixelsY:          %d\n", NrPixelsY);
  printf("  NrPixelsZ:          %d\n", NrPixelsZ);
  printf("  NrFilesPerDistance: %d\n", NrFilesPerLayer);
  printf("  LayerNr:            %d\n", nLayers);
  printf("  nCPUs:              %d\n", numProcs);
  printf("  BlanketSubtraction: %d\n", BlanketSubtraction);
  printf("  MedFiltRadius:      %d\n", MeanFiltRadius);
  printf("  DoLoGFilter:        %d\n", DoLoGFilter);
  printf("  LoGMaskRadius:      %d\n", LoGMaskRadius);
  printf("  GaussFiltRadius:    %f\n", sigma);
  printf(
      "================================================================\n\n");

  // ===================== PHASE 1: Read all TIFFs =====================
  printf("Phase 1: Reading %d TIFF files...\n", NrFilesPerLayer);
  pixelvalue *AllIntensities =
      malloc(nPixelsTotal * NrFilesPerLayer * sizeof(pixelvalue));
  if (AllIntensities == NULL) {
    printf("Could not allocate %.2f GB for intensity data.\n",
           (double)(nPixelsTotal * NrFilesPerLayer * sizeof(pixelvalue)) /
               (1024.0 * 1024.0 * 1024.0));
    return 1;
  }
  int badRead = 0;
  TIFFErrorHandler oldhandler = TIFFSetWarningHandler(NULL);
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int j = 0; j < NrFilesPerLayer; j++) {
    if (badRead)
      continue;
    char FileName[1024];
    int FileNr = ((nLayers - 1) * NrFilesPerLayer) + StartNr + j;
    sprintf(FileName, "%s_%06d.%s", fn, FileNr, extOrig);
    TIFF *tif = TIFFOpen(FileName, "r");
    if (tif == NULL) {
      printf("%s not found.\n", FileName);
      badRead = 1;
      continue;
    }
    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
    for (int roil = 0; roil < NrPixelsZ; roil++) {
      TIFFReadScanline(tif, buf, roil, 1);
      pixelvalue *datar = (pixelvalue *)buf;
      for (int i = 0; i < NrPixelsY; i++)
        AllIntensities[(size_t)(roil * NrPixelsY + i) * NrFilesPerLayer + j] =
            datar[i];
    }
    _TIFFfree(buf);
    TIFFClose(tif);
  }
  TIFFSetWarningHandler(oldhandler);
  if (badRead) {
    free(AllIntensities);
    return 1;
  }
  printf("Phase 1 complete. Time: %.2f s\n", omp_get_wtime() - start);

  // ===================== PHASE 2: Compute Median =====================
  printf("Phase 2: Computing temporal median with %d threads...\n", numProcs);
  pixelvalue *MedianArray = malloc(nPixelsTotal * sizeof(pixelvalue));
#pragma omp parallel num_threads(numProcs)
  {
    pixelvalue *SubArr = malloc(NrFilesPerLayer * sizeof(pixelvalue));
#pragma omp for schedule(dynamic, 1024)
    for (size_t i = 0; i < nPixelsTotal; i++) {
      pixelvalue *src = &AllIntensities[i * NrFilesPerLayer];
      for (int j = 0; j < NrFilesPerLayer; j++)
        SubArr[j] = src[j];
      MedianArray[i] = quick_select(SubArr, NrFilesPerLayer);
    }
    free(SubArr);
  }
  // Optionally write median to disk for backward compatibility
  char MedianFileName[1024];
  sprintf(MedianFileName, "%s/%s_Median_Background_Distance_%d.%s", outputDir,
          ReducedFileName, nLayers - 1, extReduced);
  size_t SizeOutFile = sizeof(pixelvalue) * nPixelsTotal;
  int fb =
      open(MedianFileName, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  pwrite(fb, MedianArray, SizeOutFile, 0);
  close(fb);
  printf("Phase 2 complete. Median written to %s. Time: %.2f s\n",
         MedianFileName, omp_get_wtime() - start);

  // ===================== PHASE 3: Process Images =====================
  printf("Phase 3: Processing %d images with %d threads...\n", NrFilesPerLayer,
         numProcs);
  // Allocate per-thread working buffers
  size_t bufSz = (size_t)NrPixelsY * NrPixelsZ;
  pixelvalue *ImgBuf =
      malloc(numProcs * bufSz * sizeof(pixelvalue)); // After median subtraction
  pixelvalue *FiltBuf =
      malloc(numProcs * bufSz * sizeof(pixelvalue)); // After spatial filter
  pixelvalue *FinBuf =
      malloc(numProcs * bufSz * sizeof(pixelvalue)); // Final peak image

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int imgIdx = 0; imgIdx < NrFilesPerLayer; imgIdx++) {
    int tid = omp_get_thread_num();
    size_t off = (size_t)tid * bufSz;
    pixelvalue *Image = &ImgBuf[off];
    pixelvalue *Image2 = &FiltBuf[off];
    pixelvalue *FinalImage = &FinBuf[off];
    int i, j, k;

    // 3a: Median subtraction — extract frame from AllIntensities
    for (size_t px = 0; px < nPixelsTotal; px++) {
      int interInt = (int)AllIntensities[px * NrFilesPerLayer + imgIdx] -
                     (int)MedianArray[px] - BlanketSubtraction;
      Image[px] = (pixelvalue)(interInt > 0 ? interInt : 0);
    }

    // 3b: Spatial median filter
    if (MeanFiltRadius == 1) {
      pixelvalue array[9];
      for (i = 0; i < (int)nPixelsTotal; i++) {
        if (((i + 1) % NrPixelsY <= MeanFiltRadius) ||
            ((NrPixelsY - ((i + 1) % NrPixelsY)) < MeanFiltRadius) ||
            (i < (NrPixelsY * MeanFiltRadius)) ||
            (i > ((int)nPixelsTotal - (NrPixelsY * MeanFiltRadius)))) {
          Image2[i] = Image[i];
        } else {
          int countr = 0;
          for (j = -MeanFiltRadius; j <= MeanFiltRadius; j++)
            for (k = -MeanFiltRadius; k <= MeanFiltRadius; k++)
              array[countr++] = Image[i + (NrPixelsY * j) + k];
          Image2[i] = opt_med9(array);
        }
      }
    } else if (MeanFiltRadius == 2) {
      pixelvalue array[25];
      for (i = 0; i < (int)nPixelsTotal; i++) {
        if (((i + 1) % NrPixelsY <= MeanFiltRadius) ||
            ((NrPixelsY - ((i + 1) % NrPixelsY)) < MeanFiltRadius) ||
            (i < (NrPixelsY * MeanFiltRadius)) ||
            (i > ((int)nPixelsTotal - (NrPixelsY * MeanFiltRadius)))) {
          Image2[i] = Image[i];
        } else {
          int countr = 0;
          for (j = -MeanFiltRadius; j <= MeanFiltRadius; j++)
            for (k = -MeanFiltRadius; k <= MeanFiltRadius; k++)
              array[countr++] = Image[i + (NrPixelsY * j) + k];
          Image2[i] = opt_med25(array);
        }
      }
    } else {
      for (i = 0; i < (int)nPixelsTotal; i++)
        Image2[i] = Image[i];
    }

    // 3c: LoG filtering / peak finding
    int TotPixelsInt = 0;
    memset(FinalImage, 0, nPixelsTotal * sizeof(pixelvalue));
    if (DoLoGFilter == 1) {
      int *Image4 = malloc(nPixelsTotal * sizeof(*Image4));
      FindPeakPositions(LoGMaskRadius, sigma, Image2, NrPixelsY, NrPixelsZ,
                        Image4);
      int LoGMaskRadius2 = 4;
      double sigma2 = 1;
      int *Image5 = malloc(nPixelsTotal * sizeof(*Image5));
      FindPeakPositions(LoGMaskRadius2, sigma2, Image2, NrPixelsY, NrPixelsZ,
                        Image5);
      for (i = 0; i < (int)nPixelsTotal; i++) {
        if (Image4[i] != 0) {
          FinalImage[i] = Image4[i] * 10;
          TotPixelsInt++;
        } else if (Image5[i] != 0) {
          FinalImage[i] = Image5[i] * 10;
          TotPixelsInt++;
        }
      }
      free(Image4);
      free(Image5);
    } else {
      for (i = 0; i < (int)nPixelsTotal; i++) {
        FinalImage[i] = Image2[i];
        if (Image2[i] != 0)
          TotPixelsInt++;
      }
      int **BoolImage = allocMatrixInt(NrPixelsZ, NrPixelsY);
      int **ConnComp = allocMatrixInt(NrPixelsZ, NrPixelsY);
      for (i = 0; i < NrPixelsZ; i++)
        for (j = 0; j < NrPixelsY; j++)
          BoolImage[i][j] = (Image2[(i * NrPixelsY) + j] != 0) ? 1 : 0;
      FindConnectedComponents(BoolImage, NrPixelsY, NrPixelsZ, ConnComp, NULL,
                              NULL);
      FreeMemMatrixInt(BoolImage, NrPixelsZ);
      FreeMemMatrixInt(ConnComp, NrPixelsZ);
    }
    if (TotPixelsInt > 0)
      TotPixelsInt--;
    else {
      TotPixelsInt = 1;
      FinalImage[2045] = 1;
    }

    // 3d: Write outputs (same format as ImageProcessingLibTiffOMP)
    int layerNr = nLayers;
    char OutFN[5024];
    sprintf(OutFN, "%s/%s_%06d.%s%d", outputDir, ReducedFileName, imgIdx,
            extReduced, layerNr - 1);

    if (WriteFinImage == 1) {
      char OutFN2[5024];
      sprintf(OutFN2, "%s/%s_FullImage_%06d.%s%d", outputDir, ReducedFileName,
              imgIdx, extReduced, layerNr - 1);
      FILE *fw = fopen(OutFN2, "wb");
      fwrite(FinalImage, sizeof(pixelvalue) * nPixelsTotal, 1, fw);
      fclose(fw);
    }

    // Write text output
    pixelvalue *ys = malloc(TotPixelsInt * 2 * sizeof(*ys));
    pixelvalue *zs = malloc(TotPixelsInt * 2 * sizeof(*zs));
    pixelvalue *peakID = malloc(TotPixelsInt * 2 * sizeof(*peakID));
    float32_t *intensity = malloc(TotPixelsInt * 2 * sizeof(*intensity));
    int PeaksFilledCounter = 0;
    for (i = 0; i < (int)nPixelsTotal; i++) {
      if (FinalImage[i] != 0) {
        peakID[PeaksFilledCounter] = FinalImage[i];
        intensity[PeaksFilledCounter] = Image[i];
        ys[PeaksFilledCounter] = NrPixelsY - 1 - (i % NrPixelsY);
        zs[PeaksFilledCounter] = NrPixelsZ - 1 - (i / NrPixelsY);
        PeaksFilledCounter++;
      }
    }
    char OutFNt[5024];
    sprintf(OutFNt, "%s.txtOld", OutFN);
    FILE *ft = fopen(OutFNt, "w");
    fprintf(ft, "YPos\tZPos\tpeakID\tIntensity\n");
    for (i = 0; i < TotPixelsInt; i++)
      fprintf(ft, "%d\t%d\t%d\t%lf\n", (int)ys[i], (int)zs[i], (int)peakID[i],
              (double)intensity[i]);
    fclose(ft);

    // Write binary output
    FILE *fk = fopen(OutFN, "wb");
    float32_t dummy1 = 1;
    uint32_t dummy2 = 1;
    pixelvalue dummy3 = 1;
    char dummy4 = 'x';
    uint32_t DataSize16 = TotPixelsInt * 2 + dummy3;
    uint32_t DataSize32 = TotPixelsInt * 4 + dummy3;
    // Header block 1
    fwrite(&dummy1, sizeof(float32_t), 1, fk);
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy4, sizeof(char), 1, fk);
    for (i = 0; i < 5; i++)
      fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    // Y positions block
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy4, sizeof(char), 1, fk);
    fwrite(ys, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    // Z positions block
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy4, sizeof(char), 1, fk);
    fwrite(zs, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    // Intensity block
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&DataSize32, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy4, sizeof(char), 1, fk);
    fwrite(intensity, TotPixelsInt * sizeof(float32_t), 1, fk);
    // PeakID block
    fwrite(&dummy2, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&DataSize16, sizeof(uint32_t), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy3, sizeof(pixelvalue), 1, fk);
    fwrite(&dummy4, sizeof(char), 1, fk);
    fwrite(peakID, TotPixelsInt * sizeof(pixelvalue), 1, fk);
    fclose(fk);

    free(ys);
    free(zs);
    free(peakID);
    free(intensity);
  }

  free(AllIntensities);
  free(MedianArray);
  free(ImgBuf);
  free(FiltBuf);
  free(FinBuf);

  double elapsed = omp_get_wtime() - start;
  printf("ProcessImagesCombined complete for layer %d. Total time: %.2f s\n",
         nLayers, elapsed);
  return 0;
}
