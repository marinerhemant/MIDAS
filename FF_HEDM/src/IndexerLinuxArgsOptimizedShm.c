#include <stdio.h>
#include <math.h> 
#include <stdlib.h> 
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h> 
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

static void
check (int test, const char * message, ...)
{
    if (test) {
        va_list args;
        va_start (args, message);
        vfprintf (stderr, message, args);
        va_end (args);
        fprintf (stderr, "\n");
        exit (EXIT_FAILURE);
    }
}


#define RealType double       // use 1 realtype in the whole program. 
                              // Note: single single precision turned out tp give some problems
                              // most notably with acos()     

// conversions constants
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

// max array sizes    
#define MAX_N_SPOTS 6000000   // max nr of observed spots that can be stored
#define MAX_N_STEPS 1000      // Max nr of pos steps, when stepping along the diffracted ray
#define MAX_N_OR 36000        // max nr of trial orientations that can be stored (360/0.01);
#define MAX_N_MATCHES 1       // max nr of grain matches for 1 spot
#define MAX_N_RINGS 500       // max nr of rings that can be stored (applies to the arrays ringttheta, ringhkl, etc)
#define MAX_N_HKLS 5000       // max nr of hkls that can be stored
#define MAX_N_OMEGARANGES 72  // max nr of omegaranges in input file (also max no of box sizes)

#define N_COL_THEORSPOTS 14   // number of items that is stored for each calculated spot (omega, eta, etc)
#define N_COL_OBSSPOTS 9      // number of items stored for each obs spots
#define N_COL_GRAINSPOTS 17   // nr of columns for output: y, z, omega, differences for spots of grain matches
#define N_COL_GRAINMATCHES 16 // nr of columns for output: the Matches (summary) 

// Globals
RealType *ObsSpotsLab;              // spots, converted to lab coord
int n_spots = 0; // no of spots in global var ObsSpotsLab
 
// To store the orientation matrices
RealType OrMat[MAX_N_OR][3][3];                       

// hkls to use
double hkls[MAX_N_HKLS][7];  // columns: h k l ringno dspacing theta RingRad
int n_hkls = 0;
int HKLints[MAX_N_HKLS][4];
double ABCABG[6];

// 4d and 3d arrays for storing spots in bins. For fast lookup.
// - data[iRing][iEta][iOme] points to an array (a bin). It contains the rownumbers [0-based] of the spots in ObsSpotsLab matrix.
// - ndata holds for each bin how many spots are stored (ndata[iRing][iEta][iOme])
// - maxndata contains for each bin the capacity of the bin (maxndata[iRing][iEta][iOme]) 
// 
// ie: data[2][10][12] -> [4, 13] means spots ObsSpotsLab[4][*] and 
// ObsSpotsLab[13][*] fall into bin with: ring=3 (! not 2!), eta-segment = 10 and 
// omega-segment = 12.   
//
int *data;
int *ndata;           
int SGNum;

// the number of elements of the data arrays above
int n_ring_bins;  
int n_eta_bins;
int n_ome_bins;

// the binsizes used for the binning
RealType EtaBinSize = 0;
RealType OmeBinSize = 0;                                         
                                                       
// some macros for math calculations
#define crossProduct(a,b,c) \
	(a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
	(a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
	(a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

#define dot(v,q) \
   ((v)[0] * (q)[0] + \
    (v)[1] * (q)[1] + \
 	 (v)[2] * (q)[2])
 	  
#define CalcLength(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))   	  


////////////////////////////////////////////////////////////////////////////////
// get the spot row numbers of a certain bin defined by ringno, eta, omega.
// eta and omega between -180 and +180 degrees.
//
int
GetBin(
  int ringno,
  RealType eta,
  RealType omega,
  int **spotRows, 
  int *nspotRows)
{
   // now check if this spot is in the dataset (using the bins only, a bit crude but fast way )
   int iRing, iEta, iOme, iSpot;
   iRing = ringno-1;
   iEta = floor((180+eta)/EtaBinSize);
   iOme = floor((180+omega)/OmeBinSize);
   int Pos = iRing*n_eta_bins*n_ome_bins + iEta*n_ome_bins + iOme;
   int nspots = ndata[Pos*2];
   int DataPos = ndata[Pos*2+1];
    *spotRows = malloc(nspots*sizeof(**spotRows));
    if (spotRows == NULL ) {
         printf("Memory error: could not allocate memory for spotRows matrix. Memory full?\n");
         return 1;
    }
    // calc the diff. NOte: smallest diff in pos is choosen
    for ( iSpot = 0 ; iSpot < nspots ; iSpot++ ) {
        (*spotRows)[iSpot] = data[DataPos + iSpot];
    }
            
    *nspotRows = nspots;
    return 0;    
}


  
////////////////////////////////////////////////////////////////////////////////  
// Finds a value in a column of a matrix 
// returns the rowno. -1 if the value is not found.
// 
// example: FindInMatrix(&Mat[0][0], nrows, ncols, 2, 10.1, &SpotRowNo);
//    searches for 10.1 in column-index 2 (0 based, so actually column 3!)
//
void  
FindInMatrix(
  RealType *aMatrixp,
  int nrows,
  int ncols,
  int SearchColumn,  
  RealType aVal,
  int *idx)
   
{
  int r, LIndex;
  *idx = -1;
  
  for (r=0 ; r< nrows ; r++) {
    LIndex = (r*ncols) + SearchColumn;
    if (aMatrixp[LIndex] == aVal ) {
       *idx = r;
       break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// allocates 2d array
// returns NULL if failed.
RealType** 
allocMatrix(int nrows, int ncols)
{
   RealType** arr;
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


////////////////////////////////////////////////////////////////////////////////
void
FreeMemMatrix(
   RealType **mat,
   int nrows)
   
{
   int r;
   
   for ( r = 0 ; r < nrows ; r++) {
      free(mat[r]);
   }
   
   free(mat);
}


////////////////////////////////////////////////////////////////////////////////
void
InitArrayI(
  int anArray[],
  int nel)
  
{
   memset(anArray, 0, sizeof(int)*nel);   

//   int i;  
//    for (i=0 ; i < nel ; ++i)
//    {
//      anArray[i] = 0;
//    } 
}


////////////////////////////////////////////////////////////////////////////////
// NB: only works for matrices that are contiguous!
//
void
InitMatrixCI(
  int * aMatrix,
  int nrows,
  int ncols)
  
{
   memset(aMatrix, 0, sizeof(int)*nrows*ncols);

//    int r,c;
//    for (r=0 ; r < nrows ; ++r) {
//       for (c=0 ; c < ncols ; ++c) {
//          aMatrix[r * ncols + c] = 0;
//       }
//    } 
}
    
    
////////////////////////////////////////////////////////////////////////////////
// NB: only works for matrices that are contiguous!
void
InitMatrixCF(
   RealType *aMatrix,
   int nrows,
   int ncols)
  
{
   memset(aMatrix, 0, sizeof(RealType)*nrows*ncols);
   
//    int r,c;
//   
//    for (r=0 ; r < nrows ; ++r) {
//       for (c=0 ; c < ncols ; ++c) {
//          aMatrix[r * ncols + c] = 0;
//       }
//    } 
}
    
    
////////////////////////////////////////////////////////////////////////////////
void
PrintMatrix3(
   int aMatrix[][3],
   int nrows) {
    
   int r, c;
   for ( r = 0; r<nrows ; r++) {
      for ( c = 0; c<3 ; c++) {
         printf("%d ", aMatrix[r][c]);
      }
      printf("\n");
   }
}


////////////////////////////////////////////////////////////////////////////////
// writes a matrix with ints to file.
//
int
WriteMatrixIp(
   char FileName[],
   int **aMatrixp,
   int nrows,
   int ncols) 
  
{
   int r, c;
   FILE *fp;
  
   fp = fopen(FileName, "w");
    
   if (fp==NULL) {
      printf("Cannot open file, %s\n", FileName);
      return (1);
   }
    
   for(r=0; r<nrows; r++) { 
      for(c=0;c<ncols; c++) {
         fprintf(fp, "%d ", aMatrixp[r][c]);
      }
      fprintf(fp, "\n");
   }
  
   fclose(fp);
   return(0);
}



////////////////////////////////////////////////////////////////////////////////
// writes matrix with RealType values, with optional header
//
void
WriteMatrixWithHeaderFp(
   char FileName[],
   RealType **aMatrixp,
   int nrows,
   int ncols,
   char header[])  // give empty string ("\0") to skip header) 

{
   int r, c;
   FILE *fp;
  
   fp = fopen(FileName, "w");
    
   if (fp==NULL) {
      printf("Cannot open file, %s\n", FileName);
      return;
   }
  
   // write header
   if (header[0] != '\0')   {
      fprintf(fp, "%s\n", header);
   }
  
   // write data
   for(r=0; r<nrows; r++) { 
      for(c=0;c<ncols; c++) {
         fprintf(fp, "%14f ", aMatrixp[r][c]);         
      }
      fprintf(fp, "\n");
   }
  
   fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
// writes a matrix with RealTypes to file.
//
void
WriteMatrixFp(
   char FileName[],
   RealType **aMatrixp,
   int nrows,
   int ncols) 
  
{
   WriteMatrixWithHeaderFp(FileName, aMatrixp, nrows,  ncols, "");
}



////////////////////////////////////////////////////////////////////////////////
//  write output file with the spots found for a grain.
//
int
WriteMatrixGrainsSpots(
   char FileName[],
   RealType **aMatrix,
   int nrows,
   int ncols)

{
   int r, c;
   FILE *fp;
  
   fp = fopen(FileName, "w");
    
   if (fp == NULL) {
      printf("Cannot open file, %s\n", FileName);
      return 1;
   }
  
   // write header
   char header[] = "No Dummy YCalc YObs YDiff ZCalc ZObs ZDiff WCalc WObs WDiff RadRef RadObs RadDiff SpotId MatchNo IA";
   fprintf(fp, "%s\n", header);
  
   // write data
   for(r=0; r<nrows; r++) { 
      for(c=0; c<ncols; c++) {
         fprintf(fp, "%lf ", aMatrix[r][c]);
      }
      fprintf(fp, "\n");
   }
  
   fclose(fp);
   return 0;   
}


////////////////////////////////////////////////////////////////////////////////
// NB only works for 3d matrices that are contiguous in memory!
void
WriteMatrix3DFp(
  char FileName[],
  RealType *aMatrixp,
  int nlayers, 
  int nrows,
  int ncols) {

  int l, r, c;
  FILE *fp;
  
  fp = fopen(FileName, "w");
    
  if (fp==NULL) {
    printf("Cannot open file, %s\n", FileName);
    return;
  }
    
  for (l=0; l<nlayers; l++) {
    for(r=0; r<nrows; r++) { 
      for(c=0; c<ncols; c++) {
        fprintf(fp, "%lf ",aMatrixp[l*ncols*nrows + r * ncols + c]);
      }
      fprintf(fp, "\n");
    }
  }
  
  fclose(fp);
}



////////////////////////////////////////////////////////////////////////////////
void
WriteArrayF(
   char FileName[],
   RealType anArray[],
   int nel)  {
  
   int i;
   FILE *fp;
  
   fp = fopen(FileName, "w");
    
   if (fp==NULL) {
      printf("Cannot open file: %s\n", FileName);
      return;
   }   
  
   for (i = 0; i<nel ; i++) {
      fprintf(fp, "%lf\n", anArray[i]);
   }
  
   fclose(fp);  
}


////////////////////////////////////////////////////////////////////////////////
void
WriteArrayI(
   char FileName[],
   int anArray[],
   int nel)  {
  
   int i;
   FILE *fp;
  
   fp = fopen(FileName, "w");
    
   if (fp==NULL) {
      printf("Cannot open file: %s\n", FileName);
      return;
   }  
  
   for (i = 0; i<nel ; i++) {
      fprintf(fp, "%d\n", anArray[i]);
   }
  
   fclose(fp);  
}


////////////////////////////////////////////////////////////////////////////////
void
PrintMatrixFp(
   RealType **aMatrixp,
   int nrows,
   int ncols,
   int linenr) {
    
   int r, c;
   for ( r = 0; r<nrows ; r++) {
      if (linenr == 1) printf("%d ", r+1);
      for ( c = 0; c<ncols ; c++) {
         printf("%lf ", aMatrixp[r][c]);
      }
      printf("\n");
   }
}


////////////////////////////////////////////////////////////////////////////////
// for printing a 2d matrix, treating it like a long 1d matrix. 
// so the first argument is a pointer
// call with PrintMatrix1Df(&matrix[0][0]...)
// NB: only works for matrices that are contiguous in memory.
void
PrintMatrixCF(
   RealType *aMatrixp,
   int nrows,
   int ncols,
   int linenr) {
    
   int r, c;
   for ( r = 0; r<nrows ; r++) {
      if (linenr == 1) printf("%d ", r+1);
      for ( c = 0; c<ncols ; c++) {
         printf("%lf ", aMatrixp[r * ncols + c]); 
      }
      printf("\n");
   }
}



////////////////////////////////////////////////////////////////////////////////
void
PrintMatrixF(
   RealType aMatrix[][3],
   int nrows) {
    
   int r, c;
   for ( r = 0; r<nrows ; r++) {
      printf("%d ", r);
      for ( c = 0; c<3 ; c++) {
         printf("%lf ", aMatrix[r][c]);
      }
      printf("\n");
   }
}


////////////////////////////////////////////////////////////////////////////////
void
PrintArray(
   int anArray[],
   int nel)  
{
   int i;
  
   for (i = 0; i<nel ; i++) {
      printf("%d ", anArray[i]);
   }
   printf("\n");
}


////////////////////////////////////////////////////////////////////////////////
void
PrintArrayF(
   RealType anArray[],
   int nel)  
{
   int i;
  
   for (i = 0; i<nel ; i++) {
      printf("%lf ", anArray[i]);
   } 
   printf("\n");
}


////////////////////////////////////////////////////////////////////////////////
void
PrintArrayVertF(
   RealType anArray[],
   int nel)  {
  
   int i;
  
   for (i = 0; i<nel ; i++) {
      printf("%lf\n", anArray[i]);
   }
}


////////////////////////////////////////////////////////////////////////////////
void
MatrixMultF33(
   RealType m[3][3],
   RealType n[3][3],
   RealType res[3][3]) 

{
   int r;
   
   for (r=0; r<3; r++) {
      res[r][0] = m[r][0]*n[0][0] +
                  m[r][1]*n[1][0] +
                  m[r][2]*n[2][0];

      res[r][1] = m[r][0]*n[0][1] +
                  m[r][1]*n[1][1] +
                  m[r][2]*n[2][1];
                   
      res[r][2] = m[r][0]*n[0][2] +
                  m[r][1]*n[1][2] +
                  m[r][2]*n[2][2];
                    
   }    
}


////////////////////////////////////////////////////////////////////////////////
void
MatrixMultF(
   RealType m[3][3],
   RealType v[3],
   RealType r[3])  

{
   int i;
   
   for (i=0; i<3; i++) {
      r[i] = m[i][0]*v[0] +
             m[i][1]*v[1] +
             m[i][2]*v[2]; 
                    
   }    
} 


////////////////////////////////////////////////////////////////////////////////
void
MatrixMult(
   RealType m[3][3],
   int  v[3],
   RealType r[3]) 

{

   int i;
   
   for (i=0; i<3; i++) {
      r[i] = m[i][0]*v[0] +
             m[i][1]*v[1] +
             m[i][2]*v[2]; 
   }    
} 


////////////////////////////////////////////////////////////////////////////////
RealType
min(RealType a, RealType b)
{
    return (a < b ? a : b);
}


////////////////////////////////////////////////////////////////////////////////
RealType
max(RealType a, RealType b)
{
    return (a > b ? a : b);
}
      
////////////////////////////////////////////////////////////////////////////////
void
CalcInternalAngle(
   RealType x1, 
   RealType y1, 
   RealType z1, 
   RealType x2, 
   RealType y2, 
   RealType z2, 
   RealType *ia)
   
{
   RealType v1[3];
   RealType v2[3];
   
   v1[0] = x1;
   v1[1] = y1;   
   v1[2] = z1;
   
   v2[0] = x2;
   v2[1] = y2;   
   v2[2] = z2;
    
   RealType l1 = CalcLength(x1, y1 ,z1);
   RealType l2 = CalcLength(x2, y2, z2);
   RealType tmp = dot(v1, v2)/(l1*l2);
   
   if (tmp > 1 ) { tmp = 1;  }  
   if (tmp < -1 ) {tmp = -1; }   
   
   *ia = rad2deg * acos(tmp);
}


////////////////////////////////////////////////////////////////////////////////
int
ReadMatrix(
   char  FileName[],
   RealType aMatrix[][30],
   int * nrows)
{
   int i, j;
   FILE *fp;
   char buffer[1000];
   
   *nrows = 0;   
   
   fp = fopen(FileName, "r"); // 
   if (fp==NULL) {
     printf("Cannot open file: %s.\n", FileName);
     return 1;
   }
  
   // skip header
   fgets(buffer, 1001, fp);   
   int EofReached = 0;
   for(i=0; i<MAX_N_SPOTS; i++) {
      for(j=0; j<30; j++) {
         if (fscanf(fp, "%lf", &aMatrix[i][j]) == EOF)     EofReached = 1; 
        
       }
       if (EofReached == 1) break;
       *nrows = *nrows + 1;            
   }
    
   fclose(fp);
   
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
int 
ReadArrayI(
   char  FileName[],
   int aArray[],
   int * nel)
   
{
   int i;
   FILE *fp;
   
   *nel = 0;   
   
   fp = fopen(FileName, "r"); // 
   if (fp==NULL) {
      printf("Cannot open file: %s.\n", FileName);
      return (1);
   }
  
   // skip header
//   fgets(buffer, 1001, fp);   
   for(i=0; i<MAX_N_SPOTS; i++) {
      if (fscanf(fp, "%d", &aArray[i]) == EOF)  break;
      *nel = *nel + 1;       
   }
   
   fclose(fp);
   
   return (0);
}


////////////////////////////////////////////////////////////////////////////////
// 
// % Rotates a 3d vector around z over an angle alpha.
// % For right hand system (as in fable) this means +alpha = ccw.
// %
// % Input:
// %     v1:  xyz vector 
// %     alpha: angle [degrees]
// %
// % Output:
// %     rotated vector
// %
// % Date: 18-08-2010
// %
void
RotateAroundZ(
   RealType v1[3],        
   RealType alpha, 
   RealType v2[3]) 
{
   RealType cosa = cos(alpha*deg2rad);
   RealType sina = sin(alpha*deg2rad);
   
   RealType mat[3][3] = {{ cosa, -sina, 0 },
          { sina,  cosa, 0 },
          { 0, 0, 1}};
      
   MatrixMultF(mat, v1, v2);   
}




////////////////////////////////////////////////////////////////////////////////
// % Calculates eta angle of a coordinate y,z, with 0 degrees = the z axis pointing up.
// % 
// % input:
// %   y, z: coordinates with y to the right, and z up (right hand system)
// %
// % output:
// %   alpha: angle between 0 (up) and -180 ccw or 180 cw [degrees]
// %
// % Date: 18-08-2010
// % 
void
CalcEtaAngle(
   RealType y,
   RealType z, 
   RealType *alpha) {
  
   // calc angle using dot product.
   printf("%lf %lf %lf\n",rad2deg,y,z);
   fflush(stdout);
   *alpha = rad2deg * acos(z/sqrt(y*y+z*z));  // rad -> deg
    
   // alpha is now [0, 180], for positive y it should be between [0, -180]
   if (y > 0)    *alpha = -*alpha;
}



////////////////////////////////////////////////////////////////////////////////
//       
// % Calculates the position of a spot on a detector in lab coordinates. 
// % The origin is the center of the sample rotation.
// %
// % Input:
// %    RingRadius : the radius of the ring of the spot [any unit, output is in same units]
// %    eta : angle along the ring [degrees] 0 degrees = up, clockwise is
// %          positive (seen from the beam).
// %
// % Output:
// %    yl,zl: spot position in lab coordinates. [unit of distance input]
// %        (x= direction of beam, y=to the left, z=up)
// %
// % Date: 23-08-2010
// %
void
CalcSpotPosition(
   RealType RingRadius, 
   RealType eta, 
   RealType *yl,
   RealType *zl)
  
{
   RealType etaRad = deg2rad * eta;
    
   *yl = -(sin(etaRad)*RingRadius); // y inversed: to the 'door' (=left) is +!
   *zl =   cos(etaRad)*RingRadius;
}       
       

////////////////////////////////////////////////////////////////////////////////
// % This function finds the rotation angles (omegas) for which diffration occurs
// %
// % Input:
// %   x y z: the coordinates of the gvector (for omega = 0)
// %          +x forward
// %          +y to the left (!)
// %          +z up
// %   theta: diffraction angle [degrees]
// % 
// % Output: 
// %   The rotation angles w for which diffraction occurs [degrees: 0-360].
// %   Positive omega means: going to the quadrant where x and y are
// %   positive. 
// %   In fable this means ccw!
// %
// % NB: for very small values of y (small but > 1e6) the routine might give
// % 4 answers (instead of 2) which lie very close together.
// %
// % Date: 18-08-2010
// %
// %

void
CalcOmega(
   RealType x,
   RealType y,
   RealType z,
   RealType theta, 
   RealType omegas[4],
   RealType etas[4],
   int * nsol) {

// % solve the equation -x cos w + y sin w = sin(theta) * len
// % it simply comes from calculating the angle between incoming beam
// % (represented by -1 0 0, note the minus sign!) and the gvec.
// %

   *nsol = 0;
   RealType ome;
   RealType len= sqrt(x*x + y*y + z*z);
   RealType v=sin(theta*deg2rad)*len;
  
   // % in case y is 0 use short version: -x cos w = v
   // % NB use radians in this part of the code: its faster
   RealType almostzero = 1e-4;
   if ( fabs(y) < almostzero ) {
      if (x != 0) {
         RealType cosome1 = -v/x;
         if (fabs(cosome1 <= 1)) {
            ome = acos(cosome1)*rad2deg;
            omegas[*nsol] = ome;
            *nsol = *nsol + 1;
            omegas[*nsol] = -ome;  // todo: not in range[0 360]
            *nsol = *nsol + 1;
         }
      }
   }    
   else { //% y != 0
      RealType y2 = y*y;
      RealType a = 1 + ((x*x) / y2);
      RealType b = (2*v*x) / y2;
      RealType c = ((v*v) / y2) - 1;
      RealType discr = b*b - 4*a*c;    
      
      RealType ome1a;
      RealType ome1b;
      RealType ome2a;
      RealType ome2b;
      RealType cosome1;
      RealType cosome2;      
      
      RealType eqa, eqb, diffa, diffb;
  
      if (discr >= 0) {
         cosome1 = (-b + sqrt(discr))/(2*a);
         if (fabs(cosome1) <= 1) {       
            ome1a = acos(cosome1);
            ome1b = -ome1a;
       
            // not all omegas found are valid! Accept answer only if it equals v!
            eqa = -x*cos(ome1a) + y*sin(ome1a);
            diffa = fabs(eqa - v);
            eqb = -x*cos(ome1b) + y*sin(ome1b);
            diffb = fabs(eqb - v);
            
            // take the closest answer
            if (diffa < diffb ) {
               omegas[*nsol] = ome1a*rad2deg;
               *nsol = *nsol + 1;
            }               
            else {
               omegas[*nsol] = ome1b*rad2deg;
               *nsol = *nsol + 1;             
            }
         }
  
         cosome2 = (-b - sqrt(discr))/(2*a);
         if (fabs(cosome2) <= 1) {    
            ome2a = acos(cosome2);
            ome2b = -ome2a;
          
            eqa = -x*cos(ome2a) + y*sin(ome2a);
            diffa = fabs(eqa - v);
            eqb = -x*cos(ome2b) + y*sin(ome2b);
            diffb = fabs(eqb - v); 
            
            if (diffa < diffb) {
               omegas[*nsol] = ome2a*rad2deg;
               *nsol = *nsol + 1;
            }
            else {
               omegas[*nsol] = ome2b*rad2deg;
               *nsol = *nsol + 1;                           
            }
         }   
      }
   }      

   // find corresponding eta's
   RealType gw[3];
   RealType gv[3]={x,y,z};
   RealType eta;
   int indexOme;
   for (indexOme = 0; indexOme < *nsol; indexOme++) {
      // first get the rotated vector, then the y and z coordinates gives
      // directly the eta angle [-180, 180]. 
      // y and z are in the detector plane
      RotateAroundZ(gv, omegas[indexOme], gw);
      CalcEtaAngle(gw[1],gw[2], &eta);
      etas[indexOme] = eta; 
   }
}


////////////////////////////////////////////////////////////////////////////////
// % This function calculates the diffraction spots for transmission
// % diffraction, while the sample rotates 360 degrees around the z-axis
// % (vertical axis).
// %
// % Input: 
// %   orientation (rotation matrix 3x3)
// %   hkl planes
// %   wavelength of x-ray beam [Angstrom]
// %   distance sample-detector [micron]
// %
// % Output:
// %   spots: For each orientation the diffraction spots (see below)
// %   spotsnr: The number of spots
// %
// %   For each spot (in this order, columnwise):
// %     OrientID   : a number from 1..number of orientations. (obsolete) 
// %     spotid     : a number from 1..number of spots (total)
// %     indexhkl   : a number that indicates the hkl plane. To get the hkl plane
// %                  use hkls(indexhkl)
// %     xl, yl, zl : the lab coordinates of the spot on the detector [micron] (x=direction of the
// %                  beam,  y=to the left, z=up, (0,0,0) is the center of rotation)
// %     omega      : rotation angle, from top seen: ccw is positive [degrees]
// %     eta        : position of spot on the ring, seen from the beam: cw is postive [deg]
// %     theta      : diffraction angle [degrees].
//       ringnr     : the ringnr of the spot
// %  
// % Date: 03-09-2010
// %
// % version _Furnace:
// %   - added 3rd and 4th ring, but only certain etas for certain omegas (special case
// %       when the furnace is used: the top and bottom are missing on 1 side)

////////////////////////////////////////////////////////////////////////////////
void   
CalcDiffrSpots_Furnace(
   RealType OrientMatrix[3][3], 
   RealType LatticeConstant, 
   RealType Wavelength , 
   RealType distance,
   RealType RingRadii[],  // the observed ring radii
   RealType OmegaRange[][2],  // omegaranges min - max [-180 .. 180]]  
   RealType BoxSizes[][4],     // For each omegerange: size of the box on the detector: all spots outside are removed.
   int NOmegaRanges,  // no of OmegaRanges and BoxSizes    
   RealType ExcludePoleAngle,
   RealType **spots,  // TODO check spots returned correctly?
   int   *nspots) 
  
{
  // get number of planes
//  printf("number of planes: %d\n", n_hkls);
  
   // calculate dspacing and theta for the planes in hkls (store in the table)
   int i, OmegaRangeNo;
   RealType DSpacings[MAX_N_HKLS];
   RealType thetas[MAX_N_HKLS];
   RealType ds;
   RealType theta; 
   int KeepSpot;
   
   // calculate theta and dspacing
   for (i = 0 ;i < n_hkls; i++) { 
      DSpacings[i] = hkls[i][4];
      thetas[i] = hkls[i][5];
   }
   
   
   // Each spots has 1 row with information about eta, omega, etc..
   // each hkl plane gives 2 spots for 360 degrees.
   
   // use rotation matrix for given orientation
   // generate for this orientation for each hkl the spots
   double Ghkl[3]; // vector
   int indexhkl;
   RealType Gc[3]; 
   RealType omegas[4];
   RealType etas[4];
   RealType yl;
   RealType zl;
   int nspotsPlane;
   int spotnr = 0;  // spot id's for this orientation
   int spotid = 0;
   int OrientID = 0;
   int ringnr = 0;
   
       
   for (indexhkl=0; indexhkl < n_hkls ; indexhkl++)  {
      // g vector for non rotated crystal (in ref system)
      Ghkl[0] = hkls[indexhkl][0];
      Ghkl[1] = hkls[indexhkl][1];    
      Ghkl[2] = hkls[indexhkl][2];
      
      // get ringnr   
      ringnr = (int)(hkls[indexhkl][3]);
      RealType RingRadius = RingRadii[ringnr];
      
      // calculate gvector for given orientation: Gc (in lab coordinates!)
      MatrixMultF(OrientMatrix,Ghkl, Gc);
      
      // calculate omega angles for which diffraction occurs. Eta is
      // also calculated.
      ds    = DSpacings[indexhkl];
      theta = thetas[indexhkl];
      CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
      
      // calculate spot on detector in lab coordinates (xl,yl,zl)
      for (i=0 ; i<nspotsPlane ; i++) {
         RealType Omega = omegas[i];
         RealType Eta = etas[i]; 
         RealType EtaAbs =  fabs(Eta);
      
         // remove spots on the poles
         if ((EtaAbs < ExcludePoleAngle ) || ((180-EtaAbs) < ExcludePoleAngle)) continue; 
         
         CalcSpotPosition(RingRadius, etas[i], &(yl), &(zl));
         
         for (OmegaRangeNo = 0 ; OmegaRangeNo < NOmegaRanges ; OmegaRangeNo++ ) {
            KeepSpot = 0;
            // if spot inside Omegarange and inside box, keep it
            if ( (Omega > OmegaRange[OmegaRangeNo][0]) && 
                 (Omega < OmegaRange[OmegaRangeNo][1]) &&
                 (yl > BoxSizes[OmegaRangeNo][0]) && 
                 (yl < BoxSizes[OmegaRangeNo][1]) && 
                 (zl > BoxSizes[OmegaRangeNo][2]) && 
                 (zl < BoxSizes[OmegaRangeNo][3]) ) {
               KeepSpot = 1;
               break;
            }
         }
      
         if (KeepSpot) {
            spots[spotnr][0] = OrientID;
            spots[spotnr][1] = spotid;
            spots[spotnr][2] = indexhkl;
            spots[spotnr][3] = distance;  // xl
            spots[spotnr][4] = yl;
            spots[spotnr][5] = zl;
            spots[spotnr][6] = omegas[i];
            spots[spotnr][7] = etas[i];
            spots[spotnr][8] = theta;
            spots[spotnr][9] = ringnr;
            spotnr++;  // spot nr for this orientation             
            spotid++;  // overal spot id    
         }
      }
   }
   *nspots = spotnr;
}


 
////////////////////////////////////////////////////////////////////////////////
//
// returns the index of the minimum value in an array.
// 
// The array can be masked by array idxs
// 
void 
FindMinimum( RealType *array, int *idxs, int idxsSize, int *minIndex){ 
   int i;
   if (idxsSize == 0) { 
      *minIndex = -1;
      return;
   }
  
   *minIndex = idxs[0];
   for(i=0;i<idxsSize;++i)  { 
      if(array[idxs[i]] < array[*minIndex]) *minIndex = idxs[i];
   }
}

 
////////////////////////////////////////////////////////////////////////////////
//
// Compares a set of Theoretical spots with obs spots.
//
// returns the number of matches (within the given ranges)
// and for each match the difference between the theoretical and obs spot. 
// 
void    
CompareSpots(
   RealType **TheorSpots,
   int   nTheorSpots,
   RealType *ObsSpots,
   RealType RefRad,
   RealType MarginRad,
   RealType MarginRadial,
   RealType etamargins[],
   RealType omemargins[],
   int   *nMatch,
   RealType **GrainSpots)
   
{
   int nMatched = 0;      
   int nNonMatched = 0; 
   int sp;
   int RingNr;
   int iOme, iEta;
   int spotRow, spotRowBest;      
   int MatchFound ;
   RealType diffOme;  
   RealType diffOmeBest;
   int iRing;
   int iSpot;
   RealType etamargin, omemargin;

   // for each spot in TheorSpots check if there is an equivalent one in ObsSpots
   for ( sp = 0 ; sp < nTheorSpots ; sp++ )  {
      RingNr = (int) TheorSpots[sp][9];
      iRing = RingNr-1;
      iEta = floor((180+TheorSpots[sp][12])/EtaBinSize);
      iOme = floor((180+TheorSpots[sp][6])/OmeBinSize);
      etamargin = etamargins[RingNr];
      omemargin = omemargins[(int) floor(fabs(TheorSpots[sp][12]))];  // omemargin depends on eta
  
      // calc the diff. Note: smallest in Omega difference is choosen
      MatchFound = 0;
      diffOmeBest = 100000;
      long long int Pos = iRing*n_eta_bins*n_ome_bins + iEta*n_ome_bins + iOme;
      printf("%lld\n",Pos);
      fflush(stdout);
	  long long int nspots = ndata[Pos*2];
      long long int DataPos = ndata[Pos*2+1];
      for ( iSpot = 0 ; iSpot < nspots; iSpot++ ) {
         spotRow = data[DataPos + iSpot];
         if ( fabs(TheorSpots[sp][13] - ObsSpots[spotRow*9+8]) < MarginRadial )  {
            if ( fabs(RefRad - ObsSpots[spotRow*9+3]) < MarginRad ) {
               if ( fabs(TheorSpots[sp][12] - ObsSpots[spotRow*9+6]) < etamargin ) {
                  diffOme = fabs(TheorSpots[sp][6] - ObsSpots[spotRow*9+2]);                    
                  if ( diffOme < diffOmeBest ) {
                     diffOmeBest = diffOme;
                     spotRowBest = spotRow;
                     MatchFound = 1;
                  }
               }
            }
         }
      }
      
      if (MatchFound == 1) {
         // To be stored for each spot: 
         GrainSpots[nMatched][0] = nMatched;
         GrainSpots[nMatched][1] = 999.0;  // dummy
        
         GrainSpots[nMatched][2] = TheorSpots[sp][10];
         GrainSpots[nMatched][3] = ObsSpots[spotRowBest*9+0];
         GrainSpots[nMatched][4] = ObsSpots[spotRowBest*9+0] - TheorSpots[sp][10];       
         
         GrainSpots[nMatched][5] = TheorSpots[sp][11];
         GrainSpots[nMatched][6] = ObsSpots[spotRowBest*9+1];
         GrainSpots[nMatched][7] = ObsSpots[spotRowBest*9+1] - TheorSpots[sp][11];
        
         GrainSpots[nMatched][8] = TheorSpots[sp][6];
         GrainSpots[nMatched][9] = ObsSpots[spotRowBest*9+2];
         GrainSpots[nMatched][10]= ObsSpots[spotRowBest*9+2] - TheorSpots[sp][6];
        
         GrainSpots[nMatched][11] = RefRad;
         GrainSpots[nMatched][12] = ObsSpots[spotRowBest*9+3];
         GrainSpots[nMatched][13] = ObsSpots[spotRowBest*9+3] - RefRad;
         
         GrainSpots[nMatched][14] = ObsSpots[spotRowBest*9+4];
        
         nMatched++;    
      }
      else {  // the theoritcal spot is not found, store it at the end with negative id
         nNonMatched++;       
         int idx = nTheorSpots-nNonMatched;
         GrainSpots[idx][0] = -nNonMatched;
         GrainSpots[idx][1] = 999.0;  
         GrainSpots[idx][2] = TheorSpots[sp][10];
         GrainSpots[idx][3] = 0;         
         GrainSpots[idx][4] = 0;         
         GrainSpots[idx][5] = TheorSpots[sp][11];
         GrainSpots[idx][6] = 0;
         GrainSpots[idx][7] = 0;                  
         GrainSpots[idx][8] = TheorSpots[sp][6];
         GrainSpots[idx][9] = 0;                  
         GrainSpots[idx][10] = 0;
         GrainSpots[idx][11] = 0;                  
         GrainSpots[idx][12] = 0;         
         GrainSpots[idx][13] = 0;         
         GrainSpots[idx][14] = 0;         
      }
   }
  
   *nMatch = nMatched;  
}     
     
     


////////////////////////////////////////////////////////////////////////////////
// %%
// % Calculates the rotation matrix R from an axis/angle pair.
// %
// % NB: The rotation matrix R is such that: Crystal = R * Reference. 
// % Crystal is expressed in terms of the reference system!
// %
// % Input: 
// %    axis: a 3d vector (vector does not have to be normalized; this is done
// %          in the routine)
// %    angle: rotation angle [degrees]
// % 
// % Ouput: 3x3 rotation matrix.
// %
// % Date: 18-10-2010
// %
// % version 1.1 (11-1-2011): rearranged terms to make it a bit faster (~30%
// % faster)
// %
// 
// %%
void
AxisAngle2RotMatrix(
   RealType axis[3], 
   RealType angle,
   RealType R[3][3])  {
  
   // if axis is 0 0 0 then return just the Identy matrix
   if ( (axis[0] == 0) && (axis[1] == 0) && (axis[2] == 0) ) {
      R[0][0] = 1; 
      R[1][0] = 0;
      R[2][0] = 0;
      
      R[0][1] = 0;
      R[1][1] = 1;
      R[2][1] = 0;
      
      R[0][2] = 0;
      R[1][2] = 0;
      R[2][2] = 1;
      return;
   }
    
   // normalize vector first
   RealType lenInv = 1/sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
   RealType u = axis[0]*lenInv;
   RealType v = axis[1]*lenInv;    
   RealType w = axis[2]*lenInv;    
   RealType angleRad = deg2rad * angle;
   
   // source: http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
   // note: precalc of u*v and 1-rcos made it a bit slower (suprisingly)
   RealType rcos = cos(angleRad);
   RealType rsin = sin(angleRad);

   R[0][0] =      rcos + u*u*(1-rcos);
   R[1][0] =  w * rsin + v*u*(1-rcos);
   R[2][0] = -v * rsin + w*u*(1-rcos);
   
   R[0][1] = -w * rsin + u*v*(1-rcos);
   R[1][1] =      rcos + v*v*(1-rcos);
   R[2][1] =  u * rsin + w*v*(1-rcos);
   
   R[0][2] =  v * rsin + u*w*(1-rcos);
   R[1][2] = -u * rsin + v*w*(1-rcos);
   R[2][2] =      rcos + w*w*(1-rcos);
}


double CalcRotationAngle (int RingNr){
	int habs, kabs, labs;
	int i;
	for (i=0;i<MAX_N_HKLS;i++){
		if (HKLints[i][3] == RingNr){
			habs = abs(HKLints[i][0]);
			kabs = abs(HKLints[i][1]);
			labs = abs(HKLints[i][2]);
			break;
		}
	}
	int nzeros = 0;
	if (habs == 0) nzeros++;
	if (kabs == 0) nzeros++;
	if (labs == 0) nzeros++;
	if (nzeros == 3) return 0;
	if (SGNum == 1 || SGNum == 2){ // Triclinic
		return 360;
	}else if (SGNum >= 3 && SGNum <= 15){ // Monoclinic
		if (nzeros != 2) return 360;
		else if (ABCABG[3] == 90 && ABCABG[4] == 90 && labs != 0){
			return 180;
		}else if (ABCABG[3] == 90 && ABCABG[5] == 90 && habs != 0){
			return 180;
		}else if (ABCABG[3] == 90 && ABCABG[5] == 90 && kabs != 0){
			return 180;
		}else return 360;
	}else if (SGNum >= 16 && SGNum <= 74){ // Orthorhombic
		if (nzeros !=2) return 360;
		else return 180;
	}else if (SGNum >= 75 && SGNum <= 142){ // Tetragonal
		if (nzeros == 0) return 360;
		else if (nzeros == 1 && labs == 0 && habs == kabs){
			return 180;
		}else if (nzeros == 2){
			if (labs == 0){
				return 180;
			}else{
				return 90;
			}
		}else return 360;
	}else if (SGNum >= 143 && SGNum <= 167){ // Trigonal
		if (nzeros == 0) return 360;
		else if (nzeros == 2 && labs != 0) return 120;
		else return 360;
	}else if (SGNum >= 168 && SGNum <= 194){ // Hexagonal
		if (nzeros == 2 && labs != 0) return 60;
		else return 360;
	}else if (SGNum >= 195 && SGNum <= 230){ // Cubic
		if (nzeros == 2) return 90;
		else if (nzeros == 1){
			if (habs == kabs || kabs == labs || habs == labs) return 180;
		} else if (habs == kabs && kabs == labs) return 120;
		else return 360;
	}
	else return 0;
}



////////////////////////////////////////////////////////////////////////////////    
// % Calculate a list of unique orientations that have one plane in common (in this
// % case the diffraction plane).
// %
// % Input:
// %   hkl : the miller indices of the plane ie [ 1 1 0 ]
// %   hklnormal : the vector in 3d space that is the normal of the hkl plane
// %                (ie [ 1 3 2 ] )
// %   stepsize : stepsize of rotation around the plane normal. [degrees]
// %                           
// % Output:
// %   Orientations (rotation matrices) which all have the given hkl plane and
// %   hkl-normal (as input) in common. The 3x3 matrix is stored as 1x9 row
// %   (for each orientation 1 row).
// %
// % Date: oct 2010
// %
int
GenerateCandidateOrientationsF(
   double hkl[3], // miller indices of plane 
   RealType hklnormal[3],// direction of this plane in space (just the normal of the plane)
   RealType stepsize,
   RealType OrMat[][3][3],
   int * nOrient,
   int RingNr)  {
  
   RealType v[3];
   RealType MaxAngle = 0;   
  
   // calculate orientation via axis/angle pair
   crossProduct(v, hkl, hklnormal);      // axis is just the vector that is orthogonal to v1 and v2!
   RealType hkllen = sqrt(hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2]);
   RealType hklnormallen = sqrt(hklnormal[0]*hklnormal[0] + hklnormal[1]*hklnormal[1] + hklnormal[2]*hklnormal[2]);
   RealType dotpr = dot(hkl, hklnormal); 
   RealType angled = rad2deg * acos(dotpr/(hkllen*hklnormallen));  // Angle is just the rotation in the plane defined by v1 v2!

   // calc the rotation matrix for this orientation (nb, one axis is assumed
   // now: vector v)
   // maybe faster way, using hkl and uvw
   RealType RotMat[3][3];
   RealType RotMat2[3][3];
   RealType RotMat3[3][3];
   AxisAngle2RotMatrix(v, angled, RotMat);
  
   // determine the rotation angle around the plane-normal
   MaxAngle = CalcRotationAngle(RingNr);
   
#ifdef DEBUG
   printf("Rotation angle, maxangle: %lf\n", MaxAngle);
#endif    
  
   RealType nsteps = (MaxAngle/stepsize);
   int nstepsi = (int) nsteps;
   int or;
   int row, col;
   RealType angle2;
  
   // calculate all orientations 
   for ( or=0 ; or < nstepsi ; or++) {
      angle2 = or*stepsize;
      
      // now calculate the rotation matrix to rotate around the normal of the plane
      AxisAngle2RotMatrix(hklnormal, angle2, RotMat2);
      MatrixMultF33(RotMat2, RotMat, RotMat3);
      
      for (row = 0 ; row < 3 ; row++) {
         for (col = 0 ; col < 3 ; col++) {
             OrMat[or][row][col] = RotMat3[row][col];
         }
      }
   }
   *nOrient = nstepsi;
   
   return 0;      
}


////////////////////////////////////////////////////////////////////////////////
// %% 
// % Code to calculate displacement of spots needed if original position of
// % the center of mass of the grain is a,b,c when omega = 0. Vector in
// % direction of the diffracted beam is xi,yi,zi.
// %
// % All vectors (input and output) are in fable lab coordinates.
// %
// % Input: 
// %   a,b,c: position of grain [micron, mm, etc ]
// %   xi,yi,zi: vector of diffracted beam 
// %   omega: rotation of sample around the vertical axis (z), + is ccw [degrees]
// %
// % Output:
// %   Displ_y, displ_z: shift in y and z of the spot on 
// %        the detector.  [units same as input units]
// % 
// % Date: 22-10-2010
// %
void
displacement_spot_needed_COM(
   RealType a, 
   RealType b, 
   RealType c, 
   RealType xi, 
   RealType yi, 
   RealType zi, 
   RealType omega, 
   RealType *Displ_y, 
   RealType *Displ_z)
  
{
   RealType lenInv = 1/sqrt(xi*xi + yi*yi + zi*zi);
   xi = xi*lenInv;
   yi = yi*lenInv;
   zi = zi*lenInv;
  
   RealType OmegaRad = deg2rad * omega;
   RealType sinOme = sin(OmegaRad);
   RealType cosOme = cos(OmegaRad);
   RealType t = (a*cosOme - b*sinOme)/xi;
  
   *Displ_y = ((a*sinOme)+(b*cosOme)) -(t*yi);
   *Displ_z = c - t*zi;
}  


////////////////////////////////////////////////////////////////////////////////
// % Code to calculate g-vector from the spot position on the ring.
// % Important: The spot should be the idealized spot (from center of rotation) .
// %
// % INPUT:
// % - xi, yi, zi = vector in direction of the spot from the lab
// %                coordinate system. (diffracted ray)
// % - Omega [ degrees]: rotation angle of sample, when grain comes in diffraction.
////////////////////////////////////////////////////////////////////////////////
void
spot_to_gv(
   RealType xi,
   RealType yi,
   RealType zi,
   RealType Omega,
   RealType *g1,
   RealType *g2,
   RealType *g3)
{  
   // normalize
   RealType len = sqrt(xi*xi + yi*yi + zi*zi);
   
   if (len == 0) {
      *g1 = 0;
      *g2 = 0;
      *g3 = 0;
      printf("len o!\n");            
      return;
   }
   
   RealType xn = xi/len;
   RealType yn = yi/len;
   RealType zn = zi/len;

   RealType g1r = (-1 + xn);  // r means rotated (gvec at rotation omega)
   RealType g2r = yn;
  
   //% rotate back to omega = 0
   RealType CosOme = cos(-Omega*deg2rad);
   RealType SinOme = sin(-Omega*deg2rad);   

   *g1 = g1r * CosOme - g2r * SinOme;
   *g2 = g1r * SinOme + g2r * CosOme;
   *g3 = zn;   
   // g3 (z) does not change during rotation   
}


////////////////////////////////////////////////////////////////////////////////
// % Code to calculate g-vector from the spot position on the ring.
// % This version takes into account the pos of the grain in the sample.
// %
// % INPUT:
// % - xi, yi, zi = vector in direction of the spot from the lab
// %                coordinate system.
// % - Omega [ degrees]
// % - cx, cy, cz = pos of the crystal (at omega = 0)
// %
// % OUTPUT:
// % - g vector (at omega = 0 !)
// %
////////////////////////////////////////////////////////////////////////////////
void
spot_to_gv_pos(
   RealType xi,
   RealType yi,
   RealType zi,
   RealType Omega,
   RealType cx,
   RealType cy,
   RealType cz,
   RealType *g1,
   RealType *g2,
   RealType *g3)
   
{  
   RealType v[3], vr[3];
   
   // first correct the vector xi, yi, zi for the pos of grain in the sample.
   // subtract the grain pos (use pos when it is rotated! not the normal pos (ome = 0)
   v[0] = cx;
   v[1] = cy;
   v[2] = cz;
   RotateAroundZ(v, Omega, vr); // vr is the pos of the grain after rotation
   xi = xi - vr[0];
   yi = yi - vr[1];
   zi = zi - vr[2];
   
   spot_to_gv( xi, yi, zi, Omega, g1, g2, g3);
}


////////////////////////////////////////////////////////////////////////////////
void
FriedelEtaCalculation(
   RealType ys,
   RealType zs,
   RealType ttheta,
   RealType eta,
   RealType Ring_rad,
   RealType Rsample,
   RealType Hbeam, 
   RealType *EtaMinFr,
   RealType *EtaMaxFr)
   
{
   RealType quadr_coeff2 = 0;
   RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0, y0_min_z0, y0_max = 0, y0_min = 0, z0_min = 0, z0_max = 0;

   // % Calculate some parameters (These can also be calculated once for each hkl
   // % set and then called during the program).
   
   //RealType Ring_rad = Lsd*tan(ttheta*deg2rad);
   
   if (eta > 90)
     eta_Hbeam = 180 - eta;
   else if (eta < -90)
     eta_Hbeam = 180 - fabs(eta);
   else
     eta_Hbeam = 90 - fabs(eta);
   
   Hbeam = Hbeam + 2*(Rsample*tan(ttheta*deg2rad))*(sin(eta_Hbeam*deg2rad));
   
   RealType eta_pole = 1 + rad2deg*acos(1-(Hbeam/Ring_rad));
   RealType eta_equator = 1 + rad2deg*acos(1-(Rsample/Ring_rad));
   
   // Find out which quadrant is the spot in
   if ((eta >= eta_pole) && (eta <= (90-eta_equator)) ) { // % 1st quadrant
      quadr_coeff = 1;
      coeff_y0 = -1;                                      
      coeff_z0 = 1;
   }
   else if ( (eta >=(90+eta_equator)) && (eta <= (180-eta_pole)) ) {//% 4th quadrant
      quadr_coeff = 2;
      coeff_y0 = -1;
      coeff_z0 = -1;
   }
   else if ( (eta >= (-90+eta_equator) ) && (eta <= -eta_pole) )   { // % 2nd quadrant
      quadr_coeff = 2;
      coeff_y0 = 1;
      coeff_z0 = 1;
   }  
   else if ( (eta >= (-180+eta_pole) ) && (eta <= (-90-eta_equator)) )  { // % 3rd quadrant
      quadr_coeff = 1;
      coeff_y0 = 1;
      coeff_z0 = -1;
   }
   else
     quadr_coeff = 0;
   
   
   //% Calculate y0 max and min due to Rsample.
   RealType y0_max_Rsample = ys + Rsample;
   RealType y0_min_Rsample = ys - Rsample;
   
   // Calculate z0 max and min due to Hbeam
   RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
   RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
   
   // Calculate y0 max and min due to z0
   if (quadr_coeff == 1) {
      y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
      y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
   }
   else if (quadr_coeff == 2) {
      y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
      y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
   }
   
   
   // Select whether to choose the limit due to Rsample or due to z0
   if (quadr_coeff > 0)  {
      y0_max = min(y0_max_Rsample, y0_max_z0);
      y0_min = max(y0_min_Rsample, y0_min_z0);
   }
   else {
      if ((eta > -eta_pole) && (eta < eta_pole ))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = 1;
      }
      else if (eta < (-180+eta_pole))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = -1;
      }
      else if (eta > (180-eta_pole))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = -1;
      }
      else if (( eta > (90-eta_equator)) && (eta < (90+eta_equator)) ) {
          quadr_coeff2 = 1;
          z0_max = z0_max_Hbeam;
          z0_min = z0_min_Hbeam;
          coeff_y0 = -1;
      }
      else if ((eta > (-90-eta_equator)) && (eta < (-90+eta_equator)) ) {
          quadr_coeff2 = 1;
          z0_max = z0_max_Hbeam;
          z0_min = z0_min_Hbeam;
          coeff_y0 = 1;
      }
   }
   
   // Calculate y0_min (max) or z0_min (max).
   if ( quadr_coeff2 == 0 ) {
       z0_min = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_min * y0_min));
       z0_max = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_max * y0_max));
   }
   else {
       y0_min = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min * z0_min));
       y0_max = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max * z0_max));
   }
   
   RealType dYMin = ys - y0_min;
   RealType dYMax = ys - y0_max;
   RealType dZMin = zs - z0_min;
   RealType dZMax = zs - z0_max;
   
   // Calculate the Friedel pair locations on ideal ring
   RealType YMinFrIdeal =  y0_min;
   RealType YMaxFrIdeal =  y0_max;
   RealType ZMinFrIdeal = -z0_min;
   RealType ZMaxFrIdeal = -z0_max;
   
   // Calculate the Extremities on the displaced location
   RealType YMinFr = YMinFrIdeal - dYMin;
   RealType YMaxFr = YMaxFrIdeal - dYMax;
   RealType ZMinFr = ZMinFrIdeal + dZMin;
   RealType ZMaxFr = ZMaxFrIdeal + dZMax;
   
   // Calculate etas for friedel circle
   RealType Eta1, Eta2;
   CalcEtaAngle((YMinFr + ys),(ZMinFr - zs), &Eta1);
   CalcEtaAngle((YMaxFr + ys),(ZMaxFr - zs), &Eta2);
   
   *EtaMinFr = min(Eta1,Eta2);
   *EtaMaxFr = max(Eta1,Eta2);
}   


RealType
sign(RealType anumber) 
{
  if (anumber < 0) return -1.0;
  else return 1.0;

}

////////////////////////////////////////////////////////////////////////////////
// %%
// % Code to step in orientation by changing the location of the ideal spot
// % along the ring. 
// %
// % Input:
// %  - ys, zs: lab coordinates of real spot [micron]
// %  - 2theta: diffraction angle [degrees]
// %  - eta: azimuth angle + = ccw [degrees]
// %  - Ring_rad: radius of the diffraction ring [micron]
// %  - Rsample : radius of sample [micron]
// %  - Hbeam: Height of the beam [micron]
// %  - step_size: Step size in the sample [microns]
// %
// % Output: 
// %  - y0_vector, z0_vector: the coordinates of ideal spots for a grain in the 
// %    center of the sample (lab coordinates). [micron]
// %
// 

void
GenerateIdealSpots(
   RealType ys, 
   RealType zs, 
   RealType ttheta, 
   RealType eta, 
   RealType Ring_rad, 
   RealType Rsample, 
   RealType Hbeam, 
   RealType step_size,
   RealType y0_vector[], 
   RealType z0_vector[], 
   int * NoOfSteps) 
  
{
   int quadr_coeff2 = 0;
   RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0, y0_min_z0, y0_max = 0, y0_min = 0, z0_min = 0, z0_max = 0;
   RealType y01, z01, y02, z02, y_diff, z_diff, length;
   int nsteps;  
   
   // % Calculate some parameters (These can also be calculated once for each hkl
   // % set and then called during the program).
   
   //RealType Ring_rad = Lsd*tan(ttheta*deg2rad);
   
   if (eta > 90)
     eta_Hbeam = 180 - eta;
   else if (eta < -90)
     eta_Hbeam = 180 - fabs(eta);
   else
     eta_Hbeam = 90 - fabs(eta);
   
   Hbeam = Hbeam + 2*(Rsample*tan(ttheta*deg2rad))*(sin(eta_Hbeam*deg2rad));
   
   RealType eta_pole = 1 + rad2deg*acos(1-(Hbeam/Ring_rad));
   RealType eta_equator = 1 + rad2deg*acos(1-(Rsample/Ring_rad));
   
   // Find out which quadrant is the spot in
   if ((eta >= eta_pole) && (eta <= (90-eta_equator)) ) { // % 1st quadrant
      quadr_coeff = 1;
      coeff_y0 = -1;                                      
      coeff_z0 = 1;
   }
   else if ( (eta >=(90+eta_equator)) && (eta <= (180-eta_pole)) ) {//% 4th quadrant
      quadr_coeff = 2;
      coeff_y0 = -1;
      coeff_z0 = -1;
   }
   else if ( (eta >= (-90+eta_equator) ) && (eta <= -eta_pole) )   { // % 2nd quadrant
      quadr_coeff = 2;
      coeff_y0 = 1;
      coeff_z0 = 1;
   }  
   else if ( (eta >= (-180+eta_pole) ) && (eta <= (-90-eta_equator)) )  { // % 3rd quadrant
      quadr_coeff = 1;
      coeff_y0 = 1;
      coeff_z0 = -1;
   }
   else
     quadr_coeff = 0;
   
   
   //% Calculate y0 max and min due to Rsample.
   RealType y0_max_Rsample = ys + Rsample;
   RealType y0_min_Rsample = ys - Rsample;
   
   // Calculate z0 max and min due to Hbeam
   RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
   RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
   
   // Calculate y0 max and min due to z0
   if (quadr_coeff == 1) {
      y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
      y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
   }
   else if (quadr_coeff == 2) {
      y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
      y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
   }
   
   
   // Select whether to choose the limit due to Rsample or due to z0
   if (quadr_coeff > 0)  {
      y0_max = min(y0_max_Rsample, y0_max_z0);
      y0_min = max(y0_min_Rsample, y0_min_z0);
   }
   else {
      if ((eta > -eta_pole) && (eta < eta_pole ))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = 1;
      }
      else if (eta < (-180+eta_pole))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = -1;
      }
      else if (eta > (180-eta_pole))  {
          y0_max = y0_max_Rsample;
          y0_min = y0_min_Rsample;
          coeff_z0 = -1;
      }
      else if (( eta > (90-eta_equator)) && (eta < (90+eta_equator)) ) {
          quadr_coeff2 = 1;
          z0_max = z0_max_Hbeam;
          z0_min = z0_min_Hbeam;
          coeff_y0 = -1;
      }
      else if ((eta > (-90-eta_equator)) && (eta < (-90+eta_equator)) ) {
          quadr_coeff2 = 1;
          z0_max = z0_max_Hbeam;
          z0_min = z0_min_Hbeam;
          coeff_y0 = 1;
      }
   }
   
   //% calculate nsteps
   if (quadr_coeff2 == 0 ) {
      y01 = y0_min;
      z01 = coeff_z0 * sqrt((Ring_rad * Ring_rad )-(y01 * y01));
      y02 = y0_max;
      z02 = coeff_z0 * sqrt((Ring_rad * Ring_rad )-(y02 * y02));
      y_diff = y01 - y02;
      z_diff = z01 - z02;
      length = sqrt(y_diff * y_diff + z_diff * z_diff);
      nsteps = ceil(length/step_size);
   }
   else {
      z01 = z0_min;
      y01 = coeff_y0 * sqrt((Ring_rad * Ring_rad )-((z01 * z01)));
      z02 = z0_max;
      y02 = coeff_y0 * sqrt((Ring_rad * Ring_rad )-((z02 * z02)));
      y_diff = y01 - y02;
      z_diff = z01 - z02;
      length = sqrt(y_diff * y_diff + z_diff * z_diff);
      //nsteps = (int)length/step_size;
      nsteps = ceil(length/step_size);      
   }
   
   // make nsteps odd, to make sure we have a middle point
   if ((nsteps % 2) == 0 ) {
     nsteps = nsteps +1;
   }
   
   // special case: if no of steps is only 1 take the middle point
   if ( nsteps == 1 ) {
      if (quadr_coeff2 == 0) {
         y0_vector[0] = (y0_max+y0_min)/2;
         z0_vector[0] = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_vector[0] * y0_vector[0]));
      }
      else {
         z0_vector[0] = (z0_max+z0_min)/2;
         y0_vector[0] = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_vector[0] * z0_vector[0]));
      }
   }
   else {
      int i;
      RealType stepsizeY = (y0_max-y0_min)/(nsteps-1);
      RealType stepsizeZ = (z0_max-z0_min)/(nsteps-1);
      
      
      if (quadr_coeff2 == 0) {
         for (i=0 ; i < nsteps ; i++) {
            y0_vector[i] = y0_min + i*stepsizeY;
            z0_vector[i] = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_vector[i] * y0_vector[i]));
         }
      }
      else {
         for (i=0 ; i < nsteps ; i++) {  
            z0_vector[i] = z0_min + i*stepsizeZ;
            y0_vector[i] = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_vector[i] * z0_vector[i]));
           
         }
      }
   }
   *NoOfSteps = nsteps;        
}



////////////////////////////////////////////////////////////////////////////////   
// % Code to calculate n_max and n_min for use in the code
// % spot_to_unrotated_coordinates (should be run after this code). It needs
// % to be run for each combination of ys and y0 as obtained from the 
// % stepping_orientation_spot code.
// %
// % INPUT
// % - xi,yi: Unit vector in direction of the diffracted beam. [microns]
// % - ys, y0: y-position of the spot in Lab coordinates for the real spot and
// %           ideal spot along the ring. [microns]
// % - Rsample: radius of the sample [microns]
// % - step_size: step size in the sample [microns]
// %
// % OUTPUT
// % - n_max, n_min: Inputs in the spot_to_unrotated_coordinates code.
//
//

void
calc_n_max_min(
   RealType xi, 
   RealType yi, 
   RealType ys, 
   RealType y0, 
   RealType R_sample, 
   int   step_size, 
   int * n_max, 
   int * n_min)
{
   RealType dy = ys-y0;
   RealType a = xi*xi + yi*yi;
   RealType b = 2*yi*dy;
   RealType c = dy*dy - R_sample*R_sample;
   RealType D = b*b - 4*a*c;
//   if (D < 0) printf("Error calculating number of steps in sample\n");
   RealType P = sqrt(D);
   RealType lambda_max = (-b+P)/(2*a) + 20; // 20 microns is the inaccuracy in aligning the sample.

   //% Calculate the values of n_max and n_min
   *n_max = (int)((lambda_max*xi)/(step_size));
   *n_min = - *n_max;
}




//////////////////////////////////////////////////////////////////////////////// 
// %%
// % Code to find the un-rotated coordinates (a,b,c) of the center of mass of
// % a grain from a spot. Assumes an orientation along the ring (which can be 
// % stepped through using the code stepping_orientation_spot), and a plane in 
// % the sample along the beam direction (this can also be stepped through using
// % lambda equation given below). 
// %
// % Input:
// %   - xi,yi,zi: Unit vector along the diffracted beam.  
// %   - ys,zs: Spot coordinates on detector (normalized w.r.t. beam center).
// %            [micron]
// %   - y0,z0: coordinates according to orientation at the ring
// %            (normalized w.r.t. beam center). [units same as ys and zs]
// %   - step_size_in_x: the step in x [micron]
// %   - n : goes from -RadiusOfSample/stepsize to +RadiusOfSample/stepsize.
// %        0 = center of sample.
// %   - omega: rotation of sample around vertical axis + = ccw [degrees]
// %
// % Output:
// %   a, b, c: unrotated coordinates of the (center of mass) of the grain. 
// %           [units same as ys and zs] 
// %
// % Date: 22-10-2010
// %

void
spot_to_unrotated_coordinates(
   RealType xi, 
   RealType yi, 
   RealType zi, 
   RealType ys, 
   RealType zs, 
   RealType y0,  
   RealType z0, 
   RealType step_size_in_x, 
   int   n, 
   RealType omega,
   RealType *a,
   RealType *b,
   RealType *c)
{
   RealType lambda = (step_size_in_x)*(n/xi);
   //   % Where n can be from -100 to 100 in case of a 1 mm thick sample and 5
   //   % microns step size.

   //% Rotated coordinates
   RealType x1 = lambda*xi;
   RealType y1 = ys - y0 + lambda*yi;
   RealType z1 = zs - z0 + lambda*zi;
  
   //% Un-rotated coordinates
   RealType cosOme = cos(omega*deg2rad);
   RealType sinOme = sin(omega*deg2rad);
   *a = (x1*cosOme) + (y1*sinOme);
   *b = (y1*cosOme) - (x1*sinOme);
   *c = z1;
}




////////////////////////////////////////////////////////////////////////////////
// tries to find the friedel pair. And returns coordinates (y,z) of ideal spots
// on the ring (which defines the diffracted rays, which gives the planenormals). 
//
//
void
GenerateIdealSpotsFriedel(
   RealType ys, 
   RealType zs, 
   RealType ttheta, 
   RealType eta, 
   RealType omega,
   int ringno,
   RealType Ring_rad, 
   RealType Rsample, 
   RealType Hbeam,
   RealType OmeTol, // tolerance for omega difference of the friedelpair
   RealType RadiusTol,    // tolerance for difference of the friedelpair in radial direction 
   RealType y0_vector[],   // returns: y and z coordinates of the spots found
   RealType z0_vector[], 
   int * NoOfSteps)   // no of spots found
    
{
   RealType EtaF;
   RealType OmeF;
   RealType EtaMinF, EtaMaxF, etaIdealF;
   RealType IdealYPos, IdealZPos;
   *NoOfSteps = 0;
   
   if (omega < 0 )  OmeF = omega + 180;
   else             OmeF = omega - 180;
    
   if ( eta < 0 )  EtaF = -180 - eta;
   else            EtaF = 180 - eta;
   
   // now find the candidate friedel spots
   int r;
   int rno_obs;
   RealType ome_obs, eta_obs;
   //printf("omega, OmeFr: %lf %lf\n", omega, OmeF);
   for (r=0 ; r < n_spots ; r++) {
       // filter on ringno
       rno_obs = round(ObsSpotsLab[r*9+5]);
       ome_obs = ObsSpotsLab[r*9+2];
       eta_obs = ObsSpotsLab[r*9+6];
       
       // skip if outside margins
       if ( rno_obs != ringno ) continue;
       if ( fabs(ome_obs - OmeF) > OmeTol) continue;
       
       // radial margins
       RealType yf = ObsSpotsLab[r*9+0];
       RealType zf = ObsSpotsLab[r*9+1];
       RealType EtaTransf;  // NB this eta is defined differently: the orignal spot is the origin now.
       CalcEtaAngle(yf + ys, zf - zs, &EtaTransf); 
       RealType radius = sqrt((yf + ys)*(yf + ys) + (zf - zs)*(zf - zs));
       if ( fabs(radius - 2*Ring_rad) > RadiusTol)  continue;
       
       // calculate eta boundaries
       FriedelEtaCalculation(ys, zs, ttheta, eta, Ring_rad, Rsample, Hbeam, &EtaMinF, &EtaMaxF);
              
       // in eta segment?
       if (( EtaTransf < EtaMinF) || (EtaTransf > EtaMaxF) ) continue;
       
       // found a candidate friedel pair!
       // calculate ideal spot (on the ring-> gives orientation)
       RealType ZPositionAccZ = zs - (( zf + zs)/2);
       RealType YPositionAccY = ys - ((-yf + ys)/2);

       CalcEtaAngle(YPositionAccY, ZPositionAccZ, &etaIdealF);
       CalcSpotPosition(Ring_rad, etaIdealF, &IdealYPos, &IdealZPos); 

       // save spot positions                               
       y0_vector[*NoOfSteps] = IdealYPos;
       z0_vector[*NoOfSteps] = IdealZPos;       
       (*NoOfSteps)++;
   }
}


int
AddUnique(int *arr, int *n, int val)
{
   int i;
   
   for (i=0 ; i < *n ; ++i) {
      if (arr[i] == val) {
          return 0;  // dont add
      }
   }
   
   // not in array, so add
   arr[*n] = val;
   (*n)++;
   return 1;
}



void
MakeUnitLength(
   RealType x,
   RealType y,
   RealType z, 
   RealType *xu, 
   RealType *yu, 
   RealType *zu )
{
   RealType len = CalcLength(x, y, z);
   
   if (len == 0) {
      *xu = 0;
      *yu = 0;
      *zu = 0;
      return; 
   }
   
   *xu = x/len;
   *yu = y/len;
   *zu = z/len;
}


////////////////////////////////////////////////////////////////////////////////
// tries to find the mixed friedel pair of a plane hkl (-h-k-l) And returns 
// coordinates (y,z) of ideal spots on the ring (which defines the 
// diffracted rays, which gives the planenormals).
// 
//%
//%% IMPORTANT: Please note that this works for spots with asind(abs(sind(eta))) > 10 degrees only.
//%
//%  Version: 1
//
void
GenerateIdealSpotsFriedelMixed(
   RealType ys,          // y coord of starting spot
   RealType zs,          // z coord ,,      
   RealType Ttheta,      // 2-theta ,,
   RealType Eta,         // eta ,,
   RealType Omega,       // omega ,,
   int RingNr,           // ring number ,,
   RealType Ring_rad,    // ring radius of ideal ring [micron]
   RealType Lsd,         // sample detector distance [micron]
   RealType Rsample,     // radius of sample [micron]
   RealType Hbeam,       // height of beam [micron]
   RealType StepSizePos, // stepsize of plane normals (and along the beam) [micron]   
   RealType OmeTol,      // tolerance for omega difference of the friedelpair
   RealType RadialTol,   // tolerance for difference of the friedelpair in radial direction
   RealType EtaTol,      // tolerance for eta difference of the fp  
   RealType spots_y[],   // returns: y and z coordinates of the ideal spots that gave a fp 
   RealType spots_z[], 
   int * NoOfSteps)      // no of ideal spots found
   
{
   const int MinEtaReject = 10; // [degrees] dont try spots near the pole.

   RealType omegasFP[4];
   RealType etasFP[4];
   int nsol;
   int nFPCandidates;
   RealType theta = Ttheta/2;
   RealType SinMinEtaReject = sin(MinEtaReject * deg2rad);
   RealType y0_vector[2000]; // yz of ideal spots
   RealType z0_vector[2000];
   RealType G1, G2, G3;
   int SpOnRing, NoOfSpots;
   int FPCandidatesUnique[2000]; // just stores the id's of the planenormals 
   RealType FPCandidates[2000][3];
   RealType xi, yi, zi;
   RealType y0, z0;
   RealType YFP1, ZFP1;
   int nMax, nMin;

   RealType EtaTolDeg;
   
   
   EtaTolDeg = rad2deg* atan(EtaTol / Ring_rad) ;// convert from micron to degrees
    
   // some init
   *NoOfSteps = 0;
   nFPCandidates = 0;

   // check border situation
   if (fabs(sin(Eta * deg2rad)) < SinMinEtaReject) {
      printf("The spot is too close to the poles. This technique to find mixed friedel pair would not work satisfactorily. So don't use mixed friedel pair.\n");
      return;
   }

   // Try points on the 'surface':
   // - calc ideal spots on ring: these are the plane normals 
   // - for each plane normal walk along the ray
   // - calc ideal spot pos of mixed fp
   // - displace spot due to pos in sample
   // - and check if this spot is in the dataset: if yes potential fp spot.
   // in the end returns for each hit, the best planenormals (=smallest difference in pos) 

   // first generate ideal spots
   GenerateIdealSpots(ys, zs, Ttheta, Eta, Ring_rad, Rsample, Hbeam, StepSizePos, y0_vector, z0_vector, &NoOfSpots);

   // check each planenormal    
   for (SpOnRing = 0 ; SpOnRing < NoOfSpots ; ++SpOnRing ) {
//       printf("spotno of spot: %d %d", SpOnRing, NoOfSpots);
       y0 = y0_vector[SpOnRing];
       z0 = z0_vector[SpOnRing];

       // unit vector in the direction of the ideal spot.  
       MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi);

       // calc gv and when the back of it comes into diffraction (omega)
       spot_to_gv(Lsd, y0, z0, Omega, &G1, &G2, &G3);
       CalcOmega(-G1, -G2, -G3, theta, omegasFP, etasFP, &nsol);     // take the back of plane!  
        
       // if no solutions go to next planenormal (if 1 solution: then there is also no mixed fp)
       if (nsol <= 1) {
          printf("no omega solutions. skipping plane.\n");
          continue;
       }
        
       // take the FP Omega that is closest to the original omega (this is the mixed fp)
       RealType OmegaFP, EtaFP, diff0, diff1;
       diff0 = fabs(omegasFP[0] - Omega);
       if (diff0 > 180)  { diff0 = 360 - diff0;  }  // in case the omegas are close to 180 and -180 the difference is big

       diff1 = fabs(omegasFP[1] - Omega);
       if (diff1 > 180)  { diff1 = 360 - diff1;  }  // in case the omegas are close to 180 and -180 the difference is big 

       // use the smallest
       if (  diff0 < diff1)  {
          OmegaFP = omegasFP[0];
          EtaFP   = etasFP[0];  
       }     
       else {
          OmegaFP = omegasFP[1];
          EtaFP   = etasFP[1];        
       }
        
       // calc the (ideal) spot pos of this mixed fp.
       CalcSpotPosition(Ring_rad, EtaFP, &YFP1, &ZFP1);
        
       // now step along the ray
       calc_n_max_min(xi, yi, ys, y0, Rsample, StepSizePos, &nMax, &nMin);
//       printf("xi, yi, ys, y0, Rsample, StepSizePos, RadialTol, &nMax, &nMin: %f %f %f %f %f %f %f %d %d\n", xi, yi, ys, y0, Rsample, StepSizePos, RadialTol, nMax, nMin);       
       RealType a, b, c, YFP, ZFP, RadialPosFP, EtaFPCorr;
       int n;
       for (n = nMin ; n <= nMax ; ++n ){
           spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0, StepSizePos, n, Omega, &a, &b, &c); 
               
            // if outside sample go to next n
            if (fabs(c) > Hbeam/2) {   continue;    }
            
            // now displace spot due to displacement of grain in sample
            RealType Dy,Dz;
            displacement_spot_needed_COM(a, b, c, Lsd, YFP1, ZFP1, OmegaFP, &Dy, &Dz); 
            YFP = YFP1 + Dy;
            ZFP = ZFP1 + Dz;
            
            RadialPosFP = sqrt(YFP*YFP + ZFP*ZFP) - Ring_rad;
            CalcEtaAngle(YFP,ZFP, &EtaFPCorr);
            
            // now check if this spot is in the dataset (using the bins only, a bit crude but fast way )
            int *spotRows;
            int nspotRows, iSpot, spotRow;            
            GetBin(RingNr, EtaFPCorr, OmegaFP, &spotRows, &nspotRows);
           
            RealType diffPos2, dy, dz;
            
            // check for each spot in the bin, if it close enough to the calculated spot
            for ( iSpot = 0 ; iSpot < nspotRows ; iSpot++ ) {
                spotRow = spotRows[iSpot];
                if ( (fabs(RadialPosFP - ObsSpotsLab[spotRow*9+8]) < RadialTol ) &&
                     (fabs(OmegaFP - ObsSpotsLab[spotRow*9+2]) < OmeTol ) &&                
                     (fabs(EtaFPCorr - ObsSpotsLab[spotRow*9+6]) < EtaTolDeg )  )
                {
                    // found a candidate fp: add (if not in the list already);
                    // if already in the list: keep the one with smallest diff
                    dy = (YFP-ObsSpotsLab[spotRow*9+0]);
                    dz = (ZFP-ObsSpotsLab[spotRow*9+1]);
                    diffPos2 = dy*dy + dz*dz;
                                               
                    int i;
                    int idx = nFPCandidates; // defaults to add 
                    for (i=0 ; i < nFPCandidates ; ++i) {
                        // check spot id
                        if (FPCandidates[i][0] == ObsSpotsLab[spotRow*9+4] )   {
                           // already in list: check if the diff is smaller
//                           printf("spotRow diffPos2 FPCandidates: %d %f %f\n", spotRow, diffPos2 ,FPCandidates[i][2]);
                           if (diffPos2 < FPCandidates[i][2] )  {
                              idx = i;  // item i will be replaced now
                           } else {
                              idx = -1; // indicating nothing should be done
                           }
                           break;
                           
                        }
                    }

                    // add or replace the new FP candidate  
                    // table FPCandidates: spotid_of_FP sponring diffpos                    
                    if (idx >= 0) {
                       FPCandidates[idx][0] = ObsSpotsLab[spotRow*9+4]; 
                       FPCandidates[idx][1] = SpOnRing;
                       FPCandidates[idx][2] = diffPos2;                 
                       if (idx == nFPCandidates) nFPCandidates++;
                    }                                            
                }
            }
        }
    }
    
    int i;
    // finally remove any double planenormals (a pn can be found multiple times for different spots)
    int nFPCandidatesUniq = 0;
    for (i=0 ; i < nFPCandidates ; ++i) {
       AddUnique(FPCandidatesUnique, &nFPCandidatesUniq, FPCandidates[i][1]);
    }
    
    // return the spot pos      
    int iFP;    
    for (iFP = 0 ; iFP < nFPCandidatesUniq ; ++iFP) {
       spots_y[iFP] = y0_vector[FPCandidatesUnique[iFP]];
       spots_z[iFP] = z0_vector[FPCandidatesUnique[iFP]];
//       printf("%d %d %f %f \n", iFP, FPCandidatesUnique[iFP], spots_y[iFP], spots_z[iFP]);
    }
    *NoOfSteps = nFPCandidatesUniq;      
}      

////////////////////////////////////////////////////////////////////////////////
// % Calculates diffraction angle theta 
// %
// % Input:
// %  - hkl plane (at least one should be nonzero)
// %  - LatticeParameter [A] 
// %  - Wavelength [A]
// %
// % Output: 
// %  Theta [degrees]
// %
// % Date: Nov 2010
// %
void
CalcTheta(
   int h, 
   int k, 
   int l, 
   RealType LatticeParameter, 
   RealType Wavelength, 
   RealType *theta) 
{
   RealType dspacing; 

   RealType h2k2l2 = h*h + k*k + l*l;
   dspacing = sqrt(LatticeParameter * LatticeParameter/h2k2l2);
   *theta = rad2deg*asin(Wavelength/(2*dspacing));
}

////////////////////////////////////////////////////////////////////////////////
// Calculates 2-theta [deg] from spotpos (y,z) [micron] and distance [micron]
//
void
CalcTtheta(
   RealType y, 
   RealType z,
   RealType distance,
   RealType *ttheta)
{
   RealType rad = sqrt((y*y) + (z*z));
   *ttheta = rad2deg * atan(rad/distance);
}

    
////////////////////////////////////////////////////////////////////////////////
// Parameters in the parameter file
//    
struct TParams {
   int RingNumbers[MAX_N_RINGS];   // the ring numbers to use for indexing (1, 2, 4, etc) 
   int SpaceGroupNum;                 // 1=bcc 2=fcc
   RealType LatticeConstant;          // [Angstrom]
   RealType Wavelength;               // Wavelength of incoming beam [Angstrom] 
   RealType Distance;                 // Distance between sample and detector [micron]
   RealType Rsample;                  // Radius of the sample [micron] 
   RealType Hbeam;                    // Height of the beam [micron]    
//   char HKLsFileName[1024];       
   RealType StepsizePos;              // step size in position [micron]  
   RealType StepsizeOrient;           // step size in orientation (rotation around the plane normal) [degrees]
   int NrOfRings;                  // No of rings to use (not explicit input by user, but set via RingNumbers[])
   RealType RingRadii[MAX_N_RINGS];   // Radii of the rings [micron]. this is a used internally: ringrad of ring 1 is at index 1 etc. 
   RealType RingRadiiUser[MAX_N_RINGS];   // Radii of the rings [micron]. stores only radii of the used rings!! Used for user input.    
   RealType MarginOme;                // Margin in Omega [degrees], when assigning theoretical spots to experimental spots. (|omeT-omeO| < MarginOme)
   RealType MarginEta;                // Margin in eta [degrees], ,,
   RealType MarginRad;                // Margin in radius [micron], ,, 
   RealType MarginRadial;             // Margin in radial direction (ortogonal to the ring) [micron], ,,
   RealType EtaBinSize;               // Size of bin for eta [degrees]
   RealType OmeBinSize;               // Size of bin for omega [degrees]
   RealType ExcludePoleAngle;         // Spots can be excluded at the poles: the range is |Eta| < ExcludePoleAngle and 180-|Eta| < ExcludePoleAngle [degrees]
   RealType MinMatchesToAcceptFrac;   // Minimum fraction (matched_spots/exp_spots) to accept an orientation+position. 
   RealType BoxSizes[MAX_N_OMEGARANGES][4];          // for each omegarange a box (window: left  right  bottom top) that defines the spots to include during indexing [micron]  
   RealType OmegaRanges[MAX_N_OMEGARANGES][2];       // Omegaranges: min, max [degrees], multiple possible.
   char OutputFolder[1024];        // output folder    
   int NoOfOmegaRanges;            // Automaticly set from Omegaranges (not explicit input by user)
   char SpotsFileName[1024];       // filename containing observed spots (see top for definition of columns) 
   char IDsFileName [1024];        // filename containing the spot-ids that will be used for indexing
   int UseFriedelPairs;            // 0=do not use friedelpairs  1=try to use friedelpairs
}; 


////////////////////////////////////////////////////////////////////////////////
int
ReadParams(
   char FileName[],
   struct TParams * Params)
   
{
   #define MAX_LINE_LENGTH 1024
            
   FILE *fp;
   char line[MAX_LINE_LENGTH];   
   char dummy[MAX_LINE_LENGTH];
   char *str;
   int NrOfBoxSizes = 0;
   int cmpres;   
   int NoRingNumbers = 0; // should be equal to Params->NrOfRings
   Params->NrOfRings = 0;
   Params->NoOfOmegaRanges = 0;
   
   fp = fopen(FileName, "r");
   if (fp==NULL) {
      printf("Cannot open file: %s.\n", FileName);
      return(1);
   }   
   
   // now get the params: format: "string" value(s)
   while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {

      str = "RingNumbers ";
      cmpres = strncmp(line, str, strlen(str));
      if (cmpres == 0) {
         sscanf(line, "%s %d", dummy, &(Params->RingNumbers[NoRingNumbers]) );
         NoRingNumbers++;
         continue;
      }   

      str = "SpaceGroup ";
      cmpres = strncmp(line, str, strlen(str));
      if (cmpres == 0) {
         sscanf(line, "%s %d", dummy, &(Params->SpaceGroupNum) );
         SGNum = Params->SpaceGroupNum;
         continue;
      }   
      
      str = "LatticeParameter ";
      cmpres = strncmp(line, str, strlen(str));
      if (cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->LatticeConstant) );
         sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &ABCABG[0], &ABCABG[1],
				&ABCABG[2], &ABCABG[3], &ABCABG[4], &ABCABG[5]);
         continue;
      }       
                             
      str = "Wavelength ";                           
      cmpres = strncmp(line, str, strlen(str));
      if (cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->Wavelength) );
         continue;         
      }
      
      str = "Distance ";
      cmpres = strncmp(line, str, strlen(str));   
      if (cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->Distance) );
         continue;         
      }     

      str = "Rsample ";
      cmpres = strncmp(line, str, strlen(str));      
      if ( cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->Rsample) );
         continue;         
      }     
      
      str = "Hbeam ";
      cmpres = strncmp(line, str, strlen(str));             
      if ( cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->Hbeam) );
         continue;         
      }

      str = "StepsizePos ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->StepsizePos) );
         continue;         
      }     
      
      str = "StepsizeOrient ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {      
         sscanf(line, "%s %lf", dummy, &(Params->StepsizeOrient) );
         continue;         
      }     

      str = "MarginOme ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {      
         sscanf(line, "%s %lf", dummy, &(Params->MarginOme) );
         continue;         
      }     

      str = "MarginRadius ";
      cmpres = strncmp(line, str , strlen(str));   
      if (cmpres == 0) {         
         sscanf(line, "%s %lf", dummy, &(Params->MarginRad) );
         continue;         
      }
      
      str = "MarginRadial ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {         
         sscanf(line, "%s %lf", dummy, &(Params->MarginRadial) );
         continue;         
      }
      
      str = "EtaBinSize ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {         
         sscanf(line, "%s %lf", dummy, &(Params->EtaBinSize) );
         continue;         
      }

      str = "OmeBinSize ";
      cmpres = strncmp(line, str, strlen(str));         
      if (cmpres == 0) {         
         sscanf(line, "%s %lf", dummy, &(Params->OmeBinSize) );
         continue;         
      }      

      str = "MinMatchesToAcceptFrac ";      
      cmpres = strncmp(line, str, strlen(str));            
      if (cmpres == 0) {   
         sscanf(line, "%s %lf", dummy, &(Params->MinMatchesToAcceptFrac) );
         continue;         
      }  
      
      str = "ExcludePoleAngle ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {   
         sscanf(line, "%s %lf", dummy, &(Params->ExcludePoleAngle) );
         continue;         
      }  
            
      str = "RingRadii ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {
         sscanf(line, "%s %lf", dummy, &(Params->RingRadiiUser[Params->NrOfRings]));  
         Params->NrOfRings = Params->NrOfRings + 1;           
         continue;
      }
      
      str = "OmegaRange ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {       
         sscanf(line, "%s %lf %lf", dummy, &(Params->OmegaRanges[Params->NoOfOmegaRanges][0]),
                                         &(Params->OmegaRanges[Params->NoOfOmegaRanges][1]));
         (Params->NoOfOmegaRanges)++;
         continue;                                                            
      }
 
      str = "BoxSize ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {       
         sscanf(line, "%s %lf %lf %lf %lf", dummy, &(Params->BoxSizes[NrOfBoxSizes][0]),
                                               &(Params->BoxSizes[NrOfBoxSizes][1]),
                                               &(Params->BoxSizes[NrOfBoxSizes][2]),
                                               &(Params->BoxSizes[NrOfBoxSizes][3]));
         NrOfBoxSizes++;
         continue;                                                        
      }
            
      str = "SpotsFileName ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {            
         sscanf(line, "%s %s", dummy, Params->SpotsFileName );
         continue;         
      }
      
      str = "IDsFileName ";
      cmpres = strncmp(line, str, strlen(str));             
      if (cmpres == 0) {            
         sscanf(line, "%s %s", dummy, Params->IDsFileName  );
         continue;         
      }
      
      str = "MarginEta ";      
      cmpres = strncmp(line, str, strlen(str));            
      if (cmpres == 0) {   
         sscanf(line, "%s %lf", dummy, &(Params->MarginEta) );
         continue;         
      }

      str = "UseFriedelPairs ";      
      cmpres = strncmp(line, str, strlen(str));    
      if (cmpres == 0) {   
         sscanf(line, "%s %d", dummy, &(Params->UseFriedelPairs) );
         continue;         
      }                 
      
      str = "OutputFolder ";      
      cmpres = strncmp(line, str, strlen(str));    
      if (cmpres == 0) {   
         sscanf(line, "%s %s", dummy, Params->OutputFolder );
         continue;         
      }   
            
      // if string is empty 
      str = "";
      cmpres = strncmp(line, str, strlen(str));
      if (cmpres == 0) {
         continue;
      }
      
      // if string not recognized: print warning all other cases
      printf("Warning: skipping line in parameters file:\n");
      printf("%s\n", line);
   }
   
   // make a Params->RingRadii for internal use: ringno is directly the index in array (RingRadii[5] = ringradius from ring 5)
   int i;
   for (i = 0 ; i < MAX_N_RINGS ; i++ ) { Params->RingRadii[i] = 0; }
   for (i = 0 ; i < Params->NrOfRings ; i++ ) {
      Params->RingRadii[Params->RingNumbers[i]] = Params->RingRadiiUser[i]; 
   }
   
   return(0);
}


////////////////////////////////////////////////////////////////////////////////
int
WriteParams(
   char FileName[],
   struct TParams Params)
{
   
   FILE *fp;
   int i;
   fp = fopen(FileName, "w");
   if (fp==NULL) {
      printf("Cannot open file: %s.\n", FileName);
      return (1);
   }
   int col1w = -22;
   // use alphabetic order
   for (i=0; i<Params.NoOfOmegaRanges; i++) {
      fprintf(fp, "%*s %lf %lf %lf %lf\n", col1w, "BoxSize ", Params.BoxSizes[i][0], Params.BoxSizes[i][1], Params.BoxSizes[i][2], Params.BoxSizes[i][3]);
   }
   fprintf(fp, "%*s %d\n", col1w, "SpaceGroupNum ", Params.SpaceGroupNum);
   fprintf(fp, "%*s %lf\n", col1w, "Distance ", Params.Distance);
   fprintf(fp, "%*s %lf\n", col1w, "EtaBinSize ", Params.EtaBinSize);
   fprintf(fp, "%*s %lf\n", col1w, "ExcludePoleAngle ", Params.ExcludePoleAngle);
   fprintf(fp, "%*s %lf\n", col1w, "Hbeam ", Params.Hbeam);
   fprintf(fp, "%*s %s\n", col1w, "IDsFileName ", Params.IDsFileName );
   fprintf(fp, "%*s %lf\n", col1w, "LatticeConstant ", Params.LatticeConstant);
   fprintf(fp, "%*s %lf\n", col1w, "MarginEta ", Params.MarginEta);
   fprintf(fp, "%*s %lf\n", col1w, "MarginOme ", Params.MarginOme);
   fprintf(fp, "%*s %lf\n", col1w, "MarginRadius ", Params.MarginRad);
   fprintf(fp, "%*s %lf\n", col1w, "MarginRadial ", Params.MarginRadial);
   fprintf(fp, "%*s %lf\n", col1w, "MinMatchesToAcceptFrac ", Params.MinMatchesToAcceptFrac);
   fprintf(fp, "%*s %lf\n", col1w, "OmeBinSize ", Params.OmeBinSize);
   for (i=0; i<Params.NoOfOmegaRanges; i++) {
      fprintf(fp, "%*s %lf %lf\n", col1w, "OmegaRange ", Params.OmegaRanges[i][0], Params.OmegaRanges[i][1]);
   }
   fprintf(fp, "%*s %s\n", col1w, "OutputFolder ", Params.OutputFolder);
   for (i=0; i<Params.NrOfRings; i++) {
      fprintf(fp, "%*s %lf\n", col1w, "RingRadii ", Params.RingRadiiUser[i]);
   }
   for (i=0; i<Params.NrOfRings; i++) {
      fprintf(fp, "%*s %d\n", col1w, "RingNumbers ", Params.RingNumbers[i]);
   }   
   fprintf(fp, "%*s %lf\n", col1w, "Rsample ", Params.Rsample);
   fprintf(fp, "%*s %s\n", col1w, "SpotsFileName ", Params.SpotsFileName);
   fprintf(fp, "%*s %lf\n", col1w, "StepsizePos ", Params.StepsizePos);
   fprintf(fp, "%*s %lf\n", col1w, "StepsizeOrient ", Params.StepsizeOrient);
   fprintf(fp, "%*s %d\n", col1w, "UseFriedelPairs ",  Params.UseFriedelPairs);
   fprintf(fp, "%*s %lf\n", col1w, "Wavelength ",  Params.Wavelength);
   fclose(fp);
   return(0);
}


////////////////////////////////////////////////////////////////////////////////
void
CreateNumberedFilename(
   char stem[1000],
   int aNumber,
   char ext[10],
   char fn[1000+10+4]) 
{
   sprintf(fn, "%s%04d%s", stem, aNumber, ext);
}


////////////////////////////////////////////////////////////////////////////////
// create a filename with a number: ie file0003.txt
// the number of digits is given by numberOfDigits (in above example 4)
//
void
CreateNumberedFilenameW(
   char stem[1000],
   int aNumber,
   int numberOfDigits,
   char ext[10],
   char fn[1000+10+numberOfDigits+1]) 
{
   sprintf(fn, "%s%0*d%s", stem, numberOfDigits, aNumber, ext);
}

////////////////////////////////////////////////////////////////////////////////
void
PrintRingInfo(
   RealType RingTtheta[],
   int RingMult[],
   int RingHKL[][3],
   int RingNos[],
   int nRings) 
   
{
   int i, RingNo;
   
   printf("Ring info:\n");
   printf("RingNo  h k l  mult  2-theta \n");  
   for (i=0 ; i < nRings ; i++){
      RingNo = RingNos[i];
      printf("%6d  %d %d %d   %2d   %lf\n", RingNo, RingHKL[RingNo][0], RingHKL[RingNo][1], RingHKL[RingNo][2], RingMult[RingNo], RingTtheta[RingNo] );
   }
   printf("\n");
}    


////////////////////////////////////////////////////////////////////////////////
RealType
CalcAvgIA(RealType *Arr, int n)
{
   RealType total = 0;
   int nnum = 0;
   int i;
   
   for (i=0 ; i < n ; i++) {
       if (Arr[i] == 999)     continue;  // skip special number
       total = total + fabs(Arr[i]);
       nnum++;
   }
   
   if (nnum == 0)  
      return 0;
   else 
      return total / nnum;

}

////////////////////////////////////////////////////////////////////////////////
// selects the best grain (smallest avg IA) and writes it to a file, including
// all spots belonging to this grain.
//
int
WriteBestMatch(
   char *FileName,
   RealType **GrainMatches, 
   int ngrains,
   RealType **AllGrainSpots,  // N_COL_GRAINSPOTS 
   int nrows,
   char *FileName2)
   
{
   int r, g, c;
   RealType smallestIA = 99999;
   int bestGrainIdx = -1;
   
   // find the smallest ia
   for ( g = 0 ; g < ngrains ; g++ ) {
      if ( GrainMatches[g][15] < smallestIA ) {
         
         smallestIA = GrainMatches[g][15];
         bestGrainIdx = g;
      }
   }
   

   // if found, write to file
   if (bestGrainIdx != -1) {
      FILE *fp2;
      fp2 = fopen(FileName2,"w");
      if (fp2==NULL) {
           printf("Cannot open file: %s\n", FileName2);
           return(1);
      }
      RealType bestGrainID =  GrainMatches[bestGrainIdx][14];
      fprintf(fp2, "%lf\n%lf\n",bestGrainID,bestGrainID);
      fprintf(fp2,"%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
				GrainMatches[bestGrainIdx][15], GrainMatches[bestGrainIdx][0], GrainMatches[bestGrainIdx][1], 
				GrainMatches[bestGrainIdx][2], GrainMatches[bestGrainIdx][3], GrainMatches[bestGrainIdx][4], 
				GrainMatches[bestGrainIdx][5], GrainMatches[bestGrainIdx][6], GrainMatches[bestGrainIdx][7], 
				GrainMatches[bestGrainIdx][8], GrainMatches[bestGrainIdx][9], GrainMatches[bestGrainIdx][10], 
				GrainMatches[bestGrainIdx][11], GrainMatches[bestGrainIdx][12], GrainMatches[bestGrainIdx][13]);
      for (r = 0 ; r < nrows ; r++ ) {
         if (AllGrainSpots[r][15] == bestGrainID ) {
            for (c = 0; c < N_COL_GRAINSPOTS; c++) {
                if (c!=1) {
                    fprintf(fp2,"%14lf, ", AllGrainSpots[r][c]);
                }
            }
            fprintf(fp2, "\n");
         }
      }
      fclose(fp2);
   }
   else {
       FILE *fp2;
       fp2 = fopen(FileName2,"w");
       fclose(fp2);
   }
   return (0);
}



////////////////////////////////////////////////////////////////////////////////
void
CalcIA(
   RealType **GrainMatches, //N_COL_GRAINMATCHES,
   int ngrains,
   RealType **AllGrainSpots, // N_COL_GRAINSPOTS
   RealType distance)
   
   
{
   RealType *IAgrainspots;
   int r, g;
   RealType g1x, g1y, g1z;
   RealType x1, y1, z1, w1, x2, y2, z2, w2, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
   
   // Calc for AllGrainSpots Internal angle for each spot
   // nr dummy yo yt yd z... w... rad (12) spotid matchnr
   // calc ia between the 2 g-vectors 
   // yzw1 is the vector to observed spot
   // yzw2 is the vector to theoretical spot
   // substract grain origin to get a vector from the grain origin (instead of 
   // grain centre)
   
   int nspots;    
   int rt = 0;    // rt = current row in all grain spots matrix
   
  // for faster calc, alloc only once (NB should be large enough to hold all spots of 1 grain) 
  IAgrainspots = malloc(1000 * sizeof(* IAgrainspots));   
   for ( g = 0 ; g < ngrains ; g++ ) {
      nspots = GrainMatches[g][12]; // no of spots stored (also the theoretical spots not found, just skip those)
      for (r=0 ; r < nspots ; r++) {
        
         // skip negative ids
         if (AllGrainSpots[rt][0] < 0) {
            AllGrainSpots[rt][16] = 999;   // a special number (avoid 0)
            IAgrainspots[r] = AllGrainSpots[rt][16];  
            rt++; 
            continue;
         } 
         
         x1 = distance;
         x2 = distance; 
         
         y1 = AllGrainSpots[rt][2];
         y2 = AllGrainSpots[rt][3];
         
         z1 = AllGrainSpots[rt][5];
         z2 = AllGrainSpots[rt][6];
         
         w1 = AllGrainSpots[rt][8];
         w2 = AllGrainSpots[rt][9];
         
         // grain pos
         g1x = GrainMatches[g][9];
         g1y = GrainMatches[g][10];
         g1z = GrainMatches[g][11];
         
         spot_to_gv_pos(x1, y1, z1, w1, g1x, g1y, g1z, &gv1x, &gv1y, &gv1z);
         spot_to_gv_pos(x2, y2, z2, w2, g1x, g1y, g1z, &gv2x, &gv2y, &gv2z);      
         
         CalcInternalAngle(gv1x, gv1y, gv1z, gv2x, gv2y, gv2z, &AllGrainSpots[rt][16]);
         IAgrainspots[r] = AllGrainSpots[rt][16];
         rt++; 
      }
      GrainMatches[g][15] = CalcAvgIA(IAgrainspots, nspots);      
   }
   free(IAgrainspots);   
}


////////////////////////////////////////////////////////////////////////////////
void MakeFullFileName(char* fullFileName, char* aPath, char* aFileName)
{
   if (aPath[0] == '\0' )  {
       strcpy(fullFileName, aFileName); 
   }
   else {
      // else concat path and fn
      strcpy(fullFileName, aPath);
      strcat(fullFileName, "/");
      strcat(fullFileName, aFileName);
   }
}

////////////////////////////////////////////////////////////////////////////////
// tries to find grains (orientations for each spot in spotIDs) 
int 
DoIndexing(
   int SpotIDs,
   int nSpotIDs,
   struct TParams Params )

{
   clock_t start, end;
   double dif;  
      
   RealType HalfBeam = Params.Hbeam /2 ;      
   RealType MinMatchesToAccept;  
   RealType ga, gb, gc;  
   int   nTspots;
   int   bestnMatchesIsp, bestnMatchesRot, bestnMatchesPos;
   int   bestnTspotsIsp, bestnTspotsRot, bestnTspotsPos;
   int   matchNr;
   int   nOrient;
   RealType hklnormal[3]; 
   RealType Displ_y;
   RealType Displ_z;
   int   or;
   int   sp;
   int   nMatches;
   int   r,c, i;
   RealType y0_vector[MAX_N_STEPS];
   RealType z0_vector[MAX_N_STEPS];
   int   nPlaneNormals;
   double   hkl[3];
   RealType g1, g2, g3;
   int   isp;
   RealType xi, yi, zi;
   int   n_max, n_min, n;
   RealType y0, z0; 
   RealType RingTtheta[MAX_N_RINGS];
   int   RingMult[MAX_N_RINGS];
   double   RingHKL[MAX_N_RINGS][3];
   int   orDelta, ispDelta, nDelta;
   RealType fracMatches;
   int   rownr;
   int   SpotRowNo;
   int usingFriedelPair; // 0 = not, 1 = yes, using fridelpair
   RealType **BestMatches;
   
   RealType omemargins[181];
   RealType etamargins[MAX_N_RINGS];
   char fn[1000]; 
   char ffn[1000]; // full fn
    char fn2[1000];
   char ffn2[1000];
   
   // output matrix: grains found for 1 spot
//   RealType GrainMatches[MAX_N_MATCHES][N_COL_GRAINMATCHES];    // orientat (9), pos (3), nTheorectical, nMatched, machtnr
   RealType **GrainMatches; //[MAX_N_MATCHES][N_COL_GRAINMATCHES];    // orientat (9), pos (3), nTheorectical, nMatched, machtnr   
   RealType **TheorSpots;   // theoretical spots (2d matrix, see top for columns)   
   RealType **GrainSpots;   // spots found for 1 grain ( pos + orient )     
   RealType **AllGrainSpots; // for each match, the corresponding spots and their differences:
                         // nr dummy yo yt yd z... w... rad (12) spotid matchnr
   RealType **GrainMatchesT;   // spots found for 1 grain ( pos + orient )     
   RealType **AllGrainSpotsT; // for each match, the corresponding spots and their differences:
                         // nr dummy yo yt yd z... w... rad (12) spotid matchnr


   // allocate output matrices
   
   // grainspots for all grains (from 1 starting spot)
   int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
   AllGrainSpots = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
   if (AllGrainSpots == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }      

   AllGrainSpotsT = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
   if (AllGrainSpotsT == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }      
   GrainMatchesT = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES); 
   if (GrainMatchesT == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }    
      
   // grain matches (pos + orient)
   GrainMatches = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES); 
   if (GrainMatches == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }      
   
   // grainspots for 1 grain
   int nRowsPerGrain = 2 * n_hkls;
   GrainSpots = allocMatrix(nRowsPerGrain, N_COL_GRAINSPOTS ); 
   

   // theorspots
   TheorSpots = allocMatrix(nRowsPerGrain, N_COL_THEORSPOTS);
   if (TheorSpots == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }      
   
   // bestmatches
   BestMatches = allocMatrix(nSpotIDs, 5);
   if (BestMatches == NULL ) {
      printf("Memory error: could not allocate memory for output matrix. Memory full?\n");
      return 1;
   }      
   
   // pre calc omega margins
   for ( i = 1 ; i < 180 ; i++) {
      omemargins[i] = Params.MarginOme + ( 0.5 * Params.StepsizeOrient / fabs(sin(i * deg2rad))); 
   }
   omemargins[0] = omemargins[1]; // officially undefined
   omemargins[180] = omemargins[1];   
    
   // etamargins 
   for ( i = 0 ; i < MAX_N_RINGS ; i++) {
      if ( Params.RingRadii[i] == 0)  { 
         etamargins[i] = 0; 
      }
      else {
         etamargins[i] = rad2deg * atan(Params.MarginEta/Params.RingRadii[i]) + 0.5 * Params.StepsizeOrient;
      }   
   }
   
   int SpotIDIdx;
   printf("Starting indexing...\n");
   for ( SpotIDIdx=0 ; SpotIDIdx < nSpotIDs ; SpotIDIdx++) {
      start = clock();
	  RealType MinInternalAngle=1000;
      matchNr = 0;   // grain match no
      rownr = 0; // row no in the output file of all the spots of the grains found of this spot
  
      // find row for spot to check
      RealType SpotID = SpotIDs;
      FindInMatrix(&ObsSpotsLab[0*9+0], n_spots, N_COL_OBSSPOTS, 4, SpotID, &SpotRowNo);
    
      if (SpotRowNo == -1) {
         printf("WARNING: SpotId %lf not found in spots file! Ignoring this spotID.\n", SpotID);
         continue;
      }
      
      RealType ys     = ObsSpotsLab[SpotRowNo*9+0];   
      RealType zs     = ObsSpotsLab[SpotRowNo*9+1];
      RealType omega  = ObsSpotsLab[SpotRowNo*9+2];  
      RealType RefRad = ObsSpotsLab[SpotRowNo*9+3];
      RealType eta    = ObsSpotsLab[SpotRowNo*9+6];
      RealType ttheta = ObsSpotsLab[SpotRowNo*9+7];
      int   ringnr = (int) ObsSpotsLab[SpotRowNo*9+5];  
      
      // Info for each ring
    char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	char aline[1024], dummy[1000];
	fgets(aline,1000,hklf);
	int Rnr;
	double hc,kc,lc,tth;
	for (i=0;i<MAX_N_RINGS;i++) RingMult[i] = 0;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %lf %lf %lf %s %lf %s",dummy,dummy,dummy,dummy,&Rnr,&hc,&kc,&lc,dummy,&tth,dummy);
		RingMult[Rnr]++;
		//if (RingMult[Rnr] == 1){
		RingHKL[Rnr][0] = hc;
		RingHKL[Rnr][1] = kc;
		RingHKL[Rnr][2] = lc;
		RingTtheta[Rnr] = tth;
	//}	
	}
      
      hkl[0] = RingHKL[ringnr][0];
      hkl[1] = RingHKL[ringnr][1];      
      hkl[2] = RingHKL[ringnr][2];      
      

      printf("\n--------------------------------------------------------------------------\n");
      printf("Spot number being processed %d of %d\n\n",SpotIDIdx, nSpotIDs);
      printf("%8s %10s %9s %9s %9s %9s %9s %7s\n", "SpotID", "SpotRowNo", "ys", "zs", "omega", "eta", "ttheta", "ringno");
      printf("%8.0f %10d %9.2f %9.2f %9.3f %9.3f %9.3f %7d\n\n", SpotID, SpotRowNo, ys, zs, omega, eta, ttheta, ringnr);
      long long int SpotID2 = (int) SpotIDs;
      //printf("%lld",SpotID2);
      
      // Generate 'ideal spots on the ring', NB ttheta input is ttheta of ideal ring
      nPlaneNormals = 0;
      if (Params.UseFriedelPairs == 1) {
         usingFriedelPair = 1;  // 1= true 0 = false
         GenerateIdealSpotsFriedel(ys, zs, RingTtheta[ringnr], eta, omega, ringnr, 
              Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam, Params.MarginOme, Params.MarginRadial, 
              y0_vector, z0_vector, &nPlaneNormals);
         
         // check of friedelpair was found (nplanenormals > 0). If not try to find the mixed friedelpair     
         if (nPlaneNormals == 0 ) {
            GenerateIdealSpotsFriedelMixed(ys, zs, RingTtheta[ringnr], eta, omega, ringnr, 
              Params.RingRadii[ringnr], Params.Distance, Params.Rsample, Params.Hbeam, Params.StepsizePos, 
              Params.MarginOme, Params.MarginRadial, Params.MarginEta,
              y0_vector, z0_vector, &nPlaneNormals);
         }
      }

      // if friedelpair was not found (nPlaneNormals = 0): do full search of plane      
      if ( nPlaneNormals == 0 ) {
        if (usingFriedelPair == 1){
            printf("No Friedel pair found, exiting.\n");
            exit(0);
        }
         usingFriedelPair = 0; // affects skipping
         printf("Trying all plane normals.\n");
         GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta, Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam, Params.StepsizePos, y0_vector, z0_vector, &nPlaneNormals);
      }
      
      printf("No of Plane normals: %d\n\n", nPlaneNormals);
//      WriteArrayF("PlaneNY0.txt", y0_vector, nPlaneNormals);
//      WriteArrayF("PlaneNZ0.txt", z0_vector, nPlaneNormals);
      
      // try each plane normal
      bestnMatchesIsp = -1;
      bestnTspotsIsp = 0;
      isp = 0;
      int bestMatchFound = 0;
      while (isp < nPlaneNormals) {
         y0 = y0_vector[isp];
         z0 = z0_vector[isp];
         MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi );         
         spot_to_gv(xi, yi, zi, omega,  &g1, &g2, &g3);
         
         hklnormal[0] = g1;
         hklnormal[1] = g2;  
         hklnormal[2] = g3;
         //printf("%f %f %f %f %f %f\n",hkl[0],hkl[1],hkl[2],g1,g2,g3);
         
         // Generate candidate orientations for this spot (= plane)
         GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient, OrMat, &nOrient,ringnr);
        
         // try each rotation
         bestnMatchesRot = -1;
         bestnTspotsRot = 0;         
         or = 0;
         orDelta = 1; 
         while (or < nOrient) {
            // convert from row to matrix format
//            for (i = 0 ;  i < 9 ; i ++) { OrientMatrix[i/3][i%3] =  OrMat[or][i]; }
			int t;
			/*printf("%d ",or);
			for (i=0;i<3;i++){
				for (t = 0;t<3;t++){
					printf("%f ",OrMat[or][i][t]);
				}
			}
			printf("\n");*/
            CalcDiffrSpots_Furnace(OrMat[or], Params.LatticeConstant, Params.Wavelength , Params.Distance, Params.RingRadii, Params.OmegaRanges, Params.BoxSizes, Params.NoOfOmegaRanges, Params.ExcludePoleAngle, TheorSpots, &nTspots);
#ifdef DEBUG            
            printf("nTspots: %d\n", nTspots);
#endif            
            MinMatchesToAccept = nTspots * Params.MinMatchesToAcceptFrac;
            // step in sample
            bestnMatchesPos = -1;      
            bestnTspotsPos =  0;      
            calc_n_max_min(xi, yi, ys, y0, Params.Rsample, Params.StepsizePos, &n_max, &n_min);
#ifdef DEBUG            
            printf("n_min n_max: %d %d\n", n_min, n_max);   
#endif
            n = n_min;
            
            // step in the sample
            while (n <= n_max) {
               spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0, Params.StepsizePos, n, omega, &ga, &gb, &gc );
               if (fabs(gc) > HalfBeam) {
                   n++; 
                   continue; // outside sample               
               }   
 
               // displace spots due to diplacement of grain in sample
               for (sp = 0 ; sp < nTspots ; sp++) {
                  displacement_spot_needed_COM(ga, gb, gc, TheorSpots[sp][3], TheorSpots[sp][4], 
                      TheorSpots[sp][5], TheorSpots[sp][6], &Displ_y, &Displ_z );    
                                                          
                  TheorSpots[sp][10] = TheorSpots[sp][4] +  Displ_y;
                  TheorSpots[sp][11] = TheorSpots[sp][5] +  Displ_z;
                  CalcEtaAngle( TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12] ); // correct eta for displaced spot
                  TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] + TheorSpots[sp][11] * TheorSpots[sp][11]) - 
                               Params.RingRadii[(int)TheorSpots[sp][9]];  // new eta, due to displ spot                                   
               }
         
               // compare theoretical spots with experimental spots
               CompareSpots(TheorSpots, nTspots, ObsSpotsLab, RefRad, 
                        Params.MarginRad, Params.MarginRadial, etamargins, omemargins, 
                        &nMatches, GrainSpots);

               if (nMatches > bestnMatchesPos) {  
                  bestnMatchesPos = nMatches;
                  bestnTspotsPos = nTspots;   
               }

               // save output (// save the best match)
               if ( (nMatches > 0) &&
                    (matchNr < 100) &&
                    (nMatches >= MinMatchesToAccept) ) { 
				  bestMatchFound = 1;
                  for (i = 0 ;  i < 9 ; i ++) { GrainMatchesT[0][i] = OrMat[or][i/3][i%3]; }
                  GrainMatchesT[0][9]  = ga;
                  GrainMatchesT[0][10] = gb;
                  GrainMatchesT[0][11] = gc;
                  GrainMatchesT[0][12] = nTspots;
                  GrainMatchesT[0][13] = nMatches;
                  GrainMatchesT[0][14] = 1;
                   
                  // save difference for each spot and ID
                  for (r = 0 ; r < nTspots ; r++) {
                     for (c = 0 ; c < 15 ; c++) {
                        AllGrainSpotsT[r][c] = GrainSpots[r][c];
                     }
                     AllGrainSpotsT[r][15] = 1; // avoid 0 as matchnr                      
                  }
				  CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance );
				  if (GrainMatchesT[0][15] < MinInternalAngle){
					  MinInternalAngle = GrainMatchesT[0][15];
		  		      rownr = nTspots;
		  			  matchNr = 1;
					  for (i=0;i<17;i++){
						  GrainMatches[0][i] = GrainMatchesT[0][i];
					  }
	                  for (r = 0 ; r < nTspots ; r++) {
	                     for (c = 0 ; c < 17 ; c++) {
	                        AllGrainSpots[r][c] = AllGrainSpotsT[r][c];
	                     }
	                  }
					  for (r = nTspots; r < nRowsOutput; r++){
						  for (c=0;c<17;c++){
							  AllGrainSpots[r][c] = 0;
						  }
					  }
				  }
               }
               
               // optimization: if in previous run the orientation gave low no of matches skip a few rotations
               nDelta = 1;  // default
               if (nTspots != 0) { 
                  fracMatches = (RealType)nMatches/nTspots;
                  if (fracMatches < 0.5) { nDelta = 10 - round(fracMatches * (10-1) / 0.5); } 
               }
                  
               n = n + nDelta;               
            }
            
            // save the best
            if (bestnMatchesPos > bestnMatchesRot) {  
               bestnMatchesRot = bestnMatchesPos;
               bestnTspotsRot = bestnTspotsPos;   
            }
            
#ifdef DEBUG     
            printf("isp nisp or nor npos nTheor nMatches: %d %d %d %d %d %d %d\n", isp, nPlaneNormals, or, nOrient, 2*n_max+1, bestnTspotsPos,  bestnMatchesPos );
#endif                     
            or = or + orDelta;            
         }
         
         // save the best               
         if (bestnMatchesRot > bestnMatchesIsp) {  
            bestnMatchesIsp = bestnMatchesRot;
            bestnTspotsIsp = bestnTspotsRot;   
         }             
         
         // optimization: if in previous run the plane normal gave low no of matches skip a few planenormals
         // if using Friedelpair: dont skip!
         ispDelta = 1;  // default
         if ((!usingFriedelPair) && (bestnTspotsRot != 0)) { 
            fracMatches = (RealType) bestnMatchesRot/bestnTspotsRot;
            if (fracMatches < 0.5) { ispDelta = 5 - round(fracMatches * (5-1) / 0.5); } // ispdelta between 1 and 5
         }

         printf("==> planenormal #pns #or #pos #Theor #Matches: %d %d %d %d %d %d\n", isp, nPlaneNormals, nOrient, 2*n_max+1, bestnTspotsRot, bestnMatchesRot);
         isp = isp + ispDelta;
         
         if (matchNr >= 100) {
             printf("Warning: the number of grain matches exceeds maximum (%d). Not all output is saved!\n", MAX_N_MATCHES );
         }         
      }  // isp loop
      
      fracMatches = (RealType) bestnMatchesIsp/bestnTspotsIsp;      
      printf("\n==> Best Match: No_of_theoretical_spots No_of_spots_found fraction: %d %d %0.2f\n", bestnTspotsIsp, bestnMatchesIsp, fracMatches );
      if (fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 || (int)bestnMatchesIsp == -1 || bestMatchFound == 0){
		  printf("Nothing good was found. Exiting.\n");
		  return 0;
	  }
      BestMatches[SpotIDIdx][0] = SpotIDIdx+1;
      BestMatches[SpotIDIdx][1] = SpotID;      
      BestMatches[SpotIDIdx][2] = bestnTspotsIsp;      
      BestMatches[SpotIDIdx][3] = bestnMatchesIsp;      
      BestMatches[SpotIDIdx][4] = fracMatches;      
      end = clock();
      dif = ((double)(end-start))/CLOCKS_PER_SEC;
      printf("Time elapsed [s] [min]: %f %f\n", dif, dif/60);             
      CreateNumberedFilenameW("BestGrain_", (int) SpotID, 9, ".txt", fn);
      MakeFullFileName(ffn, Params.OutputFolder, fn);
      CreateNumberedFilenameW("BestPos_", (int) SpotID, 9, ".csv", fn2);
      MakeFullFileName(ffn2, Params.OutputFolder, fn2);
      WriteBestMatch(ffn, GrainMatches, matchNr, AllGrainSpots, rownr, ffn2);
   }
   FreeMemMatrix( GrainMatches, MAX_N_MATCHES);   
   FreeMemMatrix( TheorSpots, nRowsPerGrain);
   FreeMemMatrix( GrainSpots, nRowsPerGrain);   
   FreeMemMatrix( AllGrainSpots, nRowsOutput);
   FreeMemMatrix( BestMatches, nSpotIDs);
   
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
void
ConcatStr(char *str1, char *str2, char *resStr){
   strcpy(resStr, str1);
   strcat(resStr, str2);
}


int ReadBins(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "/dev/shm/Data.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    data = mmap (0, size, PROT_READ, MAP_SHARED, fd, 0);
    check (data == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));
    
    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "/dev/shm/nData.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    ndata = mmap (0, size2, PROT_READ, MAP_SHARED, fd2, 0);
    check (ndata == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));
	return 1;
}

int ReadSpots(){
	int fd;
	struct stat s;
	int status;
	size_t size;
	const char *filename = "/dev/shm/Spots.bin";
	int rc;
	fd = open(filename,O_RDONLY);
	check(fd < 0, "open %s failed: %s", filename, strerror(errno));
	status = fstat (fd , &s);
	check (status < 0, "stat %s failed: %s", filename, strerror(errno));
	size = s.st_size;
	ObsSpotsLab = mmap(0,size,PROT_READ,MAP_SHARED,fd,0);
	check (ObsSpotsLab == MAP_FAILED,"mmap %s failed: %s", filename, strerror(errno));
	return (int) size/(9*sizeof(double));
}

int UnMap(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "/dev/shm/Data.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    rc = munmap (data,size);
    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "/dev/shm/nData.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    rc = munmap (ndata,size2);
	
	int fd3;
	struct stat s3;
	int status3;
	const char *filename3 = "/dev/shm/Spots.bin";
	fd3 = open(filename3,O_RDONLY);
	check(fd3 < 0, "open %s failed: %s", filename3, strerror(errno));
	status3 = fstat (fd3 , &s3);
	check (status3 < 0, "stat %s failed: %s", filename3, strerror(errno));
	size_t size3 = s3.st_size;
	rc = munmap(ObsSpotsLab,size3);
	return 1;
}

////////////////////////////////////////////////////////////////////////////////  
int
main(int argc, char *argv[]){
   printf("\n\n\t\tIndexer v4.0\nContact hsharma@anl.gov in case of questions about the MIDAS project.\n\n");
   clock_t end, start0;
   double diftotal;
   int returncode;
   struct TParams Params;
   int SpotIDs; // [MAX_N_SPOTS];
   int nSpotIDs;
   char *ParamFN;
   char fn[1024];
   // get command line params
   if (argc != 3) {
      printf("Supply a parameter file and a spotID as argument: ie %s param.txt SpotID\n\n", argv[0]);
      exit(EXIT_FAILURE);
   }
   // read parameter file
   ParamFN = argv[1];
   SpotIDs = atoi(argv[2]);
   printf("Reading parameters from file: %s.\n", ParamFN);
   returncode = ReadParams(ParamFN, &Params);
   if ( returncode != 0 ) {
      printf("Error reading params file %s\n", ParamFN );
      exit(EXIT_FAILURE);
   }
   printf("SpaceGroup: %d\n",Params.SpaceGroupNum);
   printf("Finished reading parameters.\n");
   // Read hkls
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	char aline[1024],dummy[1024];
	fgets(aline,1000,hklf);
	int Rnr,i;
	int hi,ki,li;
	double hc,kc,lc,RRd,Ds,tht;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf",&hi,&ki,&li,&Ds,&Rnr,&hc,&kc,&lc,&tht,dummy,&RRd);
		for (i=0;i<Params.NrOfRings;i++){
			if (Rnr == Params.RingNumbers[i]){
				HKLints[n_hkls][0] = hi;
				HKLints[n_hkls][1] = ki;
				HKLints[n_hkls][2] = li;
				HKLints[n_hkls][3] = Rnr;
				hkls[n_hkls][0] = hc;
				hkls[n_hkls][1] = kc;
				hkls[n_hkls][2] = lc;
				hkls[n_hkls][3] = (double)Rnr;
				hkls[n_hkls][4] = Ds;
				hkls[n_hkls][5] = tht;
				hkls[n_hkls][6] = RRd;
				//printf("%d %d %d %d %f %f %f %f %f %f %f\n",hi,ki,li,Rnr,hc,kc,lc,(double)Rnr,Ds,tht,RRd);
				n_hkls++;
			}
		}
	}
   printf("No of hkl's: %d\n", n_hkls);
   // read spots
   int t;
   n_spots = ReadSpots();
   nSpotIDs = 1;
   printf("Binned data...\n");
   int rc = ReadBins();
   int HighestRingNo = 0;
   for (i = 0 ; i < MAX_N_RINGS ; i++ ) { 
      if ( Params.RingRadii[i] != 0) HighestRingNo = i;
   }
   // calculate no of bins (NB global vars)
   n_ring_bins = HighestRingNo;  
   n_eta_bins = ceil(360.0 / Params.EtaBinSize);
   n_ome_bins = ceil(360.0 / Params.OmeBinSize);
   EtaBinSize = Params.EtaBinSize;
   OmeBinSize = Params.OmeBinSize;
   printf("No of bins for rings : %d\n", n_ring_bins);
   printf("No of bins for eta   : %d\n", n_eta_bins);
   printf("No of bins for omega : %d\n", n_ome_bins);
   printf("Total no of bins     : %d\n\n", n_ring_bins * n_eta_bins * n_ome_bins);
   printf("Finished binning.\n\n");
   start0 = clock();
   DoIndexing(SpotIDs,  nSpotIDs, Params );
   end = clock();
   diftotal = ((double)(end-start0))/CLOCKS_PER_SEC;
   printf("\nTotal time elapsed [s] [min]: %f %f\n", diftotal, diftotal/60);
   int tc = UnMap();
   return(0);
}
