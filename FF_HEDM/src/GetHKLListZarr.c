//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Compile using:
// gcc -o GetHKLList GetHKLList.c sgclib.c sgfind.c sghkl.c sgsi.c sgio.c -ldl
// -lm -O3

#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include "sginfo.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <zip.h>

static inline double **allocMatrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }

static inline void CorrectHKLsLatC(double LatC[6], double **hklsIn, int nhkls,
                                   double **hkls) {
  double a = LatC[0], b = LatC[1], c = LatC[2], alpha = LatC[3], beta = LatC[4],
         gamma = LatC[5];
  int hklnr;
  double SinA = sind(alpha), SinB = sind(beta), SinG = sind(gamma),
         CosA = cosd(alpha), CosB = cosd(beta), CosG = cosd(gamma);
  double GammaPr = acosd((CosA * CosB - CosG) / (SinA * SinB)),
         BetaPr = acosd((CosG * CosA - CosB) / (SinG * SinA)),
         SinBetaPr = sind(BetaPr);
  double Vol = (a * (b * (c * (SinA * (SinBetaPr * (SinG)))))),
         APr = b * c * SinA / Vol, BPr = c * a * SinB / Vol,
         CPr = a * b * SinG / Vol;
  double B[3][3];
  B[0][0] = APr;
  B[0][1] = (BPr * cosd(GammaPr)), B[0][2] = (CPr * cosd(BetaPr)), B[1][0] = 0,
  B[1][1] = (BPr * sind(GammaPr)), B[1][2] = (-CPr * SinBetaPr * CosA),
  B[2][0] = 0, B[2][1] = 0, B[2][2] = (CPr * SinBetaPr * SinA);
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    double ginit[3];
    ginit[0] = hklsIn[hklnr][0];
    ginit[1] = hklsIn[hklnr][1];
    ginit[2] = hklsIn[hklnr][2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hkls[hklnr][0] = ginit[0];
    hkls[hklnr][1] = ginit[1];
    hkls[hklnr][2] = ginit[2];
    hkls[hklnr][3] = Ds;
    hkls[hklnr][4] = 0;
    hkls[hklnr][5] = GCart[0];
    hkls[hklnr][6] = GCart[1];
    hkls[hklnr][7] = GCart[2];
  }
}

struct Data {
  double h;
  double k;
  double l;
  double Ds;
  double RingNr;
  double hc;
  double kc;
  double lc;
  double SortVals;
};

static int cmpfunc(const void *a, const void *b) {
  struct Data *ia = (struct Data *)a;
  struct Data *ib = (struct Data *)b;
  return (int)(1000.f * ia->SortVals - 1000.f * ib->SortVals);
}

static inline void SortFunc(int nRows, int nCols, double **TotInfo,
                            int ColToSort, int Dir) {
  struct Data *MyData;
  MyData = malloc(nRows * sizeof(*MyData));
  int i, j, k;
  for (i = 0; i < nRows; i++) {
    MyData[i].h = TotInfo[i][0];
    MyData[i].k = TotInfo[i][1];
    MyData[i].l = TotInfo[i][2];
    MyData[i].Ds = TotInfo[i][3];
    MyData[i].RingNr = TotInfo[i][4];
    MyData[i].hc = TotInfo[i][5];
    MyData[i].kc = TotInfo[i][6];
    MyData[i].lc = TotInfo[i][7];
    MyData[i].SortVals = Dir * TotInfo[i][ColToSort];
  }
  qsort(MyData, nRows, sizeof(struct Data), cmpfunc);
  for (i = 0; i < nRows; i++) {
    TotInfo[i][0] = MyData[i].h;
    TotInfo[i][1] = MyData[i].k;
    TotInfo[i][2] = MyData[i].l;
    TotInfo[i][3] = MyData[i].Ds;
    TotInfo[i][4] = MyData[i].RingNr;
    TotInfo[i][5] = MyData[i].hc;
    TotInfo[i][6] = MyData[i].kc;
    TotInfo[i][7] = MyData[i].lc;
  }
  free(MyData);
}

static inline int CheckDirectoryCreation(char Folder[1024]) {
  int e;
  struct stat sb;
  char totOutDir[1024];
  sprintf(totOutDir, "%s/", Folder);
  e = stat(totOutDir, &sb);
  if (e != 0 && errno == ENOENT) {
    printf("Output directory did not exist, creating %s\n", totOutDir);
    e = mkdir(totOutDir, S_IRWXU);
    if (e != 0) {
      printf("Could not make the directory. Exiting\n");
      return 0;
    }
  }
  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Give a data ZarrZip file. Optionally resultFolder\n");
    exit(1);
  }
  int SpaceGrp;
  double LatC[6], wl, Lsd, MaxRingRad;
  char *DataFN = argv[1];
  blosc2_init();
  // Read zarr config
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(DataFN, 0, &errorp);
  if (errorp != NULL)
    return 1;
  struct zip_stat *finfo = NULL;
  finfo = calloc(16384, sizeof(int));
  zip_stat_init(finfo);
  zip_file_t *fd = NULL;
  int count = 0;
  char *data = NULL;
  char *s = NULL;
  char *resultFolder, *arr;
  int32_t dsize;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = 4096;
      resultFolder = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, resultFolder, dsize);
      resultFolder[dsize] = '\0';
      free(arr);
      // free(data); // Bug fix: decompresses into resultFolder, not data
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/LatticeParameter/0") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = 6 * sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      int iter;
      for (iter = 0; iter < 6; iter++)
        LatC[iter] = *(double *)&data[iter * sizeof(double)];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/Wavelength/0") != NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      wl = *(double *)&data[0];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RhoD/0") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      MaxRingRad = *(double *)&data[0];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaxRingRad/0") != NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      MaxRingRad = *(double *)&data[0];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      Lsd = *(double *)&data[0];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SpaceGroup/0") != NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      SpaceGrp = *(int *)&data[0];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    count++;
  }
  if (argc == 3)
    resultFolder = argv[2];
  int e = CheckDirectoryCreation(resultFolder);
  if (e == 0) {
    return 1;
  }

  printf("%f %f %f %d %f %f %f %f %f %f\n", wl, Lsd, MaxRingRad, SpaceGrp,
         LatC[0], LatC[1], LatC[2], LatC[3], LatC[4], LatC[5]);
  int h, k, l, iList, restriction, M, i, j;
  int Minh, Mink, Minl;
  int CCMx_PL[9], deterCCMx_LP = 0;
  double Epsilon = 0.0001;
  int Families[50000][3];
  T_SgInfo *SgInfo;
  char SgName[200];
  int F_Convention = 'A';
  const T_TabSgName *tsgn;

  printf("Generating hkl's\n");
  if ((SgInfo = (T_SgInfo *)malloc(sizeof(T_SgInfo))) == NULL) {
    printf("Unable to allocate SgInfo\n");
    printf("Aborting\n");
    exit(1);
  }
  SgInfo->GenOption = 0;
  SgInfo->MaxList = 192;
  if ((SgInfo->ListSeitzMx =
           (T_RTMx *)malloc(SgInfo->MaxList * sizeof(T_RTMx))) == NULL) {
    printf("Unable to allocate (SgInfo.ListSeitzMx\n");
    printf("Aborting\n");
    exit(1);
  }
  SgInfo->ListRotMxInfo = NULL;
  InitSgInfo(SgInfo);
  sprintf(SgName, "%d", SpaceGrp);
  tsgn = FindTabSgNameEntry(SgName, F_Convention);
  if (tsgn == NULL) {
    printf("Error: Unknown Space Group Symbol\n");
    printf("Aborting\n");
    exit(1);
  }
  sprintf(SgName, "%s", tsgn->HallSymbol);
  SgInfo->TabSgName = tsgn;
  if (tsgn)
    SgInfo->GenOption = 1;
  {
    int pos_hsym;
    pos_hsym = ParseHallSymbol(SgName, SgInfo);

    if (SgError != NULL) {
      printf("Error: Unknown Space Group Symbol\n");
      printf("Aborting\n");
      exit(1);
    }
  }
  if (CompleteSgInfo(SgInfo) != 0) {
    printf("Error in Complete\n");
    printf("Aborting\n");
    exit(1);
  }
  if (SgInfo->LatticeInfo->Code != 'P') {
    deterCCMx_LP = deterRotMx(SgInfo->CCMx_LP);
    InverseRotMx(SgInfo->CCMx_LP, CCMx_PL);
    if (deterCCMx_LP < 1) {
      printf("deterCMM failed.\n");
      return 0;
    }
  }
  int Maxh, Maxk, Maxl;
  int nrFilled = 0;
  Maxh = 10;
  Maxk = 10;
  Maxl = 10;
  SetListMin_hkl(SgInfo, Maxk, Maxl, &Minh, &Mink, &Minl);
  printf("Will go from %d to %d in h; %d to %d in k; %d to %d in l.\n", Minh,
         Maxh, Mink, Maxk, Minl, Maxl);
  for (h = Minh; h <= Maxh; h++) {
    for (k = Mink; k <= Maxk; k++) {
      for (l = Minl; l <= Maxl; l++) {
        if (h == 0 && k == 0 && l == 0) {
          continue;
        }
        iList = IsSysAbsent_hkl(SgInfo, h, k, l, &restriction);
        if (SgError != NULL) {
          printf("IsSysAbsent_hkl failed.\n");
          return 0;
        }
        if (iList == 0) {
          if ((iList = IsSuppressed_hkl(SgInfo, Minh, Mink, Minl, Maxk, Maxl, h,
                                        k, l)) !=
              0) { /* Suppressed reflections */
          } else {
            T_Eq_hkl Eq_hkl;
            M = BuildEq_hkl(SgInfo, &Eq_hkl, h, k, l);
            if (SgError != NULL) {
              return 0;
            }
            for (i = 0; i < Eq_hkl.N; i++) {
              for (j = -1; j <= 1; j += 2) {
                Families[nrFilled][0] = Eq_hkl.h[i] * j;
                Families[nrFilled][1] = Eq_hkl.k[i] * j;
                Families[nrFilled][2] = Eq_hkl.l[i] * j;
                nrFilled++;
              }
            }
          }
        }
      }
    }
  }
  int AreDuplicates[50000];
  double **UniquePlanes;
  UniquePlanes = allocMatrix(50000, 3);
  for (i = 0; i < 50000; i++)
    AreDuplicates[i] = 0;
  int nrPlanes = 0;
  for (i = 0; i < nrFilled - 1; i++) {
    if (AreDuplicates[i] == 1) {
      continue;
    }
    for (j = i + 1; j < nrFilled; j++) {
      if (Families[i][0] == Families[j][0] &&
          Families[i][1] == Families[j][1] &&
          Families[i][2] == Families[j][2] && AreDuplicates[j] == 0) {
        AreDuplicates[j] = 1;
      }
    }
    UniquePlanes[nrPlanes][0] = (double)Families[i][0];
    UniquePlanes[nrPlanes][1] = (double)Families[i][1];
    UniquePlanes[nrPlanes][2] = (double)Families[i][2];
    nrPlanes++;
  }
  double **hkls;
  hkls = allocMatrix(nrPlanes, 12);
  CorrectHKLsLatC(LatC, UniquePlanes, nrPlanes, hkls);
  SortFunc(nrPlanes, 11, hkls, 3, -1);
  double DsMin = wl / (2 * sind((atand(MaxRingRad / Lsd)) / 2));
  for (i = 0; i < nrPlanes; i++) {
    if (hkls[i][3] < DsMin) {
      nrPlanes = i;
      break;
    }
  }
  int RingNr = 1;
  double DsTemp = hkls[0][3];
  hkls[0][4] = 1;
  hkls[0][8] = asind(wl / (2 * (hkls[0][3])));
  hkls[0][9] = hkls[0][8] * 2;
  hkls[0][10] = Lsd * tand(hkls[0][9]);
  for (i = 1; i < nrPlanes; i++) {
    if (fabs(hkls[i][3] - DsTemp) < Epsilon) {
      hkls[i][4] = RingNr;
    } else {
      DsTemp = hkls[i][3];
      RingNr++;
      hkls[i][4] = RingNr;
    }
    hkls[i][8] = asind(wl / (2 * (hkls[i][3])));
    hkls[i][9] = hkls[i][8] * 2;
    hkls[i][10] = Lsd * tand(hkls[i][9]);
  }
  char fn[4096];
  sprintf(fn, "%s/hkls.csv", resultFolder);
  FILE *fp;
  fp = fopen(fn, "w");
  fprintf(fp, "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n");
  for (i = 0; i < nrPlanes; i++) {
    fprintf(fp, "%.0f %.0f %.0f %f %.0f %f %f %f %f %f %f\n", hkls[i][0],
            hkls[i][1], hkls[i][2], hkls[i][3], hkls[i][4], hkls[i][5],
            hkls[i][6], hkls[i][7], hkls[i][8], hkls[i][9], hkls[i][10]);
  }
  exit(0);
}
