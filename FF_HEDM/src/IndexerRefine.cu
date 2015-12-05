//


#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>

#define RealType double

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
#define N_COL_THEORSPOTS 8   // number of items that is stored for each calculated spot (omega, eta, etc)
#define N_COL_OBSSPOTS 9      // number of items stored for each obs spots
#define N_COL_GRAINSPOTS 17   // nr of columns for output: y, z, omega, differences for spots of grain matches
#define N_COL_GRAINMATCHES 16 // nr of columns for output: the Matches (summary)
#define MAX_LINE_LENGTH 4096
#define MAX_N_FRIEDEL_PAIRS 50
#define N_COLS_FRIEDEL_RESULTS 16
#define N_COLS_ORIENTATION_NUMBERS 3

//BEGIN NLDRMD FUNCTION
__device__ void nelmin ( double fn ( int n_fun, double *x, void *data ), 
  int n, double *start, double *xmin, 
  double *lb, double *ub, double *scratch, double *ynewlo, 
  double reqmin, double *step, int konvge, int kcount, 
  int *icount, int *numres, int *ifault, void *data_t)
{
  double ccoeff = 0.5;
  double del;
  double dn;
  double dnn;
  double ecoeff = 2.0;
  double eps = 0.001;
  int i;
  int ihi;
  int ilo;
  int j;
  int jcount;
  int l;
  int nn;
  double *p;
  double *p2star;
  double *pbar;
  double *pstar;
  double rcoeff = 1.0;
  double rq;
  double x;
  double *y;
  double y2star;
  double ylo;
  double ystar;
  double z;
/*
  Check the input parameters.
*/
  if ( reqmin <= 0.0 )
  {
    *ifault = 1;
    return;
  }

  if ( n < 1 )
  {
    *ifault = 1;
    return;
  }

  if ( konvge < 1 )
  {
    *ifault = 1;
    return;
  }

  p = scratch;
  pstar = p + n*(n+1);
  p2star = pstar + n;
  pbar = p2star + n;
  y = pbar + n;

  *icount = 0;
  *numres = 0;

  jcount = konvge; 
  dn = ( double ) ( n );
  nn = n + 1;
  dnn = ( double ) ( nn );
  del = 1.0;
  rq = reqmin * dn;
/*
  Initial or restarted loop.
*/
  for ( ; ; )
  {
    for ( i = 0; i < n; i++ )
    { 
      p[i+n*n] = start[i];
    }
    y[n] = fn ( n, start, data_t );
    *icount = *icount + 1;

    for ( j = 0; j < n; j++ )
    {
      x = start[j];
      start[j] = start[j] + step[j] * del;
      if (start[j] < lb[j]) start[j] = lb[j]; // Constraints
      if (start[j] > ub[j]) start[j] = ub[j]; // Constraints
      for ( i = 0; i < n; i++ )
      {
        p[i+j*n] = start[i];
      }
      y[j] = fn ( n, start, data_t );
      *icount = *icount + 1;
      start[j] = x;
    }
/*                 
  The simplex construction is complete.
                    
  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
  the vertex of the simplex to be replaced.
*/                
    ylo = y[0];
    ilo = 0;

    for ( i = 1; i < nn; i++ )
    {
      if ( y[i] < ylo )
      {
        ylo = y[i];
        ilo = i;
      }
    }
/*
  Inner loop.
*/
    for ( ; ; )
    {
      if ( kcount <= *icount )
      {
        break;
      }
      *ynewlo = y[0];
      ihi = 0;

      for ( i = 1; i < nn; i++ )
      {
        if ( *ynewlo < y[i] )
        {
          *ynewlo = y[i];
          ihi = i;
        }
      }
/*
  Calculate PBAR, the centroid of the simplex vertices
  excepting the vertex with Y value YNEWLO.
*/
      for ( i = 0; i < n; i++ )
      {
        z = 0.0;
        for ( j = 0; j < nn; j++ )
        { 
          z = z + p[i+j*n];
        }
        z = z - p[i+ihi*n];  
        pbar[i] = z / dn;
      }
/*
  Reflection through the centroid.
*/
      for ( i = 0; i < n; i++ )
      {
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
        if (pstar[i] < lb[i]) pstar[i] = lb[i]; // Constraints
        if (pstar[i] > ub[i]) pstar[i] = ub[i]; // Constraints
      }
      ystar = fn ( n, pstar, data_t );
      *icount = *icount + 1;
/*
  Successful reflection, so extension.
*/
      if ( ystar < ylo )
      {
        for ( i = 0; i < n; i++ )
        {
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
          if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
          if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
        }
        y2star = fn ( n, p2star, data_t );
        *icount = *icount + 1;
/*
  Check extension.
*/
        if ( ystar < y2star )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Retain extension or contraction.
*/
        else
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = p2star[i];
          }
          y[ihi] = y2star;
        }
      }
/*
  No extension.
*/
      else
      {
        l = 0;
        for ( i = 0; i < nn; i++ )
        {
          if ( ystar < y[i] )
          {
            l = l + 1;
          }
        }

        if ( 1 < l )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Contraction on the Y(IHI) side of the centroid.
*/
        else if ( l == 0 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
            if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
            if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
          }
          y2star = fn ( n, p2star, data_t );
          *icount = *icount + 1;
/*
  Contract the whole simplex.
*/
          if ( y[ihi] < y2star )
          {
            for ( j = 0; j < nn; j++ )
            {
              for ( i = 0; i < n; i++ )
              {
                p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
                xmin[i] = p[i+j*n];
                if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
                if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
              }
              y[j] = fn ( n, xmin, data_t );
              *icount = *icount + 1;
            }
            ylo = y[0];
            ilo = 0;

            for ( i = 1; i < nn; i++ )
            {
              if ( y[i] < ylo )
              {
                ylo = y[i];
                ilo = i;
              }
            }
            continue;
          }
/*
  Retain contraction.
*/
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
/*
  Contraction on the reflection side of the centroid.
*/
        else if ( l == 1 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
            if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
            if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
          }
          y2star = fn ( n, p2star, data_t );
          *icount = *icount + 1;
/*
  Retain reflection?
*/
          if ( y2star <= ystar )
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = pstar[i];
            }
            y[ihi] = ystar;
          }
        }
      }
/*
  Check if YLO improved.
*/
      if ( y[ihi] < ylo )
      {
        ylo = y[ihi];
        ilo = ihi;
      }
      jcount = jcount - 1;

      if ( 0 < jcount )
      {
        continue;
      }
/*
  Check to see if minimum reached.
*/
      if ( *icount <= kcount )
      {
        jcount = konvge;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + y[i];
        }
        x = z / dnn;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + pow ( y[i] - x, 2 );
        }

        if ( z <= rq )
        {
          break;
        }
      }
    }
/*
  Factorial tests to check that YNEWLO is a local minimum.
*/
    for ( i = 0; i < n; i++ )
    {
      xmin[i] = p[i+ilo*n];
    }
    *ynewlo = y[ilo];

    if ( kcount < *icount )
    {
      *ifault = 2;
      break;
    }

    *ifault = 0;

    for ( i = 0; i < n; i++ )
    {
      del = step[i] * eps;
      xmin[i] = xmin[i] + del;
      if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
      if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
      z = fn ( n, xmin, data_t );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] - del - del;
      if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
      if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
      z = fn ( n, xmin, data_t );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] + del;
    }

    if ( *ifault == 0 )
    {
      break;
    }
/*
  Restart the procedure.
*/
    for ( i = 0; i < n; i++ )
    {
      start[i] = xmin[i];
    }
    del = eps;
    *numres = *numres + 1;
  }
  return;
}
//END NLDRMD FUNCTION

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

#define CHECK(call){														\
	const cudaError_t error = call;											\
	if (error != cudaSuccess){												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(-10*error);													\
	}																		\
}

RealType cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((RealType)tp.tv_sec + (RealType)tp.tv_usec*1.e-6);
}

struct ParametersStruct {
   int RingNumbers[MAX_N_RINGS];   // the ring numbers to use for indexing (1, 2, 4, etc)
   int SpaceGroupNum;                 //
   RealType LatticeConstant;          // [Angstrom]
   RealType Wavelength;               // Wavelength of incoming beam [Angstrom]
   RealType Distance;                 // Distance between sample and detector [micron]
   RealType Rsample;                  // Radius of the sample [micron]
   RealType Hbeam;                    // Height of the beam [micron]
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
   char OutputFolder[MAX_LINE_LENGTH];        // output folder
   int NoOfOmegaRanges;            // Automaticly set from Omegaranges (not explicit input by user)
   char SpotsFileName[MAX_LINE_LENGTH];       // filename containing observed spots (see top for definition of columns)
   char IDsFileName [MAX_LINE_LENGTH];        // filename containing the spot-ids that will be used for indexing
   int UseFriedelPairs;            // 0=do not use friedelpairs  1=try to use friedelpairs
   RealType ABCABG[6];				// ABC, Alpha, Beta, Gamma for the structure
   RealType MargABC;
   RealType MargABG;
   int TopLayer;
   RealType wedge;
};

int ReadParams(char FileName[], struct ParametersStruct * Params){
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
	fflush(stdout);
	// now get the params: format: "string" value(s)
	while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
		str = "RingNumbers ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &(Params->RingNumbers[NoRingNumbers]) );
			NoRingNumbers++;
			continue;
		}
		str = "TopLayer ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &(Params->TopLayer) );
			continue;
		}
		str = "SpaceGroup ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &(Params->SpaceGroupNum) );
			continue;
		}
		str = "LatticeParameter ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &(Params->LatticeConstant) );
			sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &(Params->ABCABG[0]), &(Params->ABCABG[1]),
				&(Params->ABCABG[2]), &(Params->ABCABG[3]), &(Params->ABCABG[4]), &(Params->ABCABG[5]));
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
		str = "Wedge ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &(Params->wedge) );
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
		str = "MargABC ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &(Params->MargABC) );
			continue;
		}
		str = "MargABG ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &(Params->MargABG) );
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
	for (i = 0 ; i < MAX_N_RINGS ; i++ ) {
		Params->RingRadii[i] = 0;
	}
	for (i = 0 ; i < Params->NrOfRings ; i++ ) {
		Params->RingRadii[Params->RingNumbers[i]] = Params->RingRadiiUser[i];
	}
	return(0);
}

__device__ int FindRowInMatrix(RealType *aMatrixp, int nrows, int ncols, int SearchColumn, int aVal){
	for (int r=0 ; r< nrows ; r++) {
		if (aMatrixp[(r*ncols) + SearchColumn] == aVal){
			return r;
		}
	}
	return -1;
}

__device__ RealType CalcEtaAngle(RealType y, RealType z) {
   RealType alph = rad2deg * acos(z/sqrt(y*y+z*z));
   if (y > 0) alph = -alph;
   return alph;
}

__device__ void AxisAngle2RotMatrix(RealType axis[3], RealType angle, RealType R[3][3]){
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
	RealType u = axis[0]*(1/sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]));
	RealType v = axis[1]*(1/sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]));
	RealType w = axis[2]*(1/sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]));
	RealType angleRad = deg2rad * angle;
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
	return;
}

__device__ RealType CalcRotationAngle (int RingNr, int *HKLints, int *IntParamArr,
		RealType *RTParamArr){
	int habs, kabs, labs;
	for (int i=0;i<MAX_N_HKLS;i++){
		if (HKLints[i*4+3] == RingNr){
			habs = abs(HKLints[i*4+0]);
			kabs = abs(HKLints[i*4+1]);
			labs = abs(HKLints[i*4+2]);
			break;
		}
	}
	int SGNum = IntParamArr[0];
	RealType ABCABG[6];
	for (int i=0;i<6;i++) ABCABG[i] = RTParamArr[13 + MAX_N_RINGS + i];
	int nzeros = 0;
	if (habs == 0) nzeros++;
	if (kabs == 0) nzeros++;
	if (labs == 0) nzeros++;
	if (nzeros == 3) return 0;
	if (SGNum == 1 || SGNum == 2){
		return 360;
	}else if (SGNum >= 3 && SGNum <= 15){
		if (nzeros != 2) return 360;
		else if (ABCABG[3] == 90 && ABCABG[4] == 90 && labs != 0){
			return 180;
		}else if (ABCABG[3] == 90 && ABCABG[5] == 90 && habs != 0){
			return 180;
		}else if (ABCABG[3] == 90 && ABCABG[5] == 90 && kabs != 0){
			return 180;
		}else return 360;
	}else if (SGNum >= 16 && SGNum <= 74){
		if (nzeros !=2) return 360;
		else return 180;
	}else if (SGNum >= 75 && SGNum <= 142){
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
	}else if (SGNum >= 143 && SGNum <= 167){
		if (nzeros == 0) return 360;
		else if (nzeros == 2 && labs != 0) return 120;
		else return 360;
	}else if (SGNum >= 168 && SGNum <= 194){
		if (nzeros == 2 && labs != 0) return 60;
		else return 360;
	}else if (SGNum >= 195 && SGNum <= 230){
		if (nzeros == 2) return 90;
		else if (nzeros == 1){
			if (habs == kabs || kabs == labs || habs == labs) return 180;
		} else if (habs == kabs && kabs == labs) return 120;
		else return 360;
	}
	return 0;
}

__device__ void MatrixMultF33(RealType m[3][3], RealType n[3][3], RealType res[3][3]){
	for (int r=0; r<3; r++) {
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

__device__ void MatrixMultF(RealType m[3][3], RealType v[3], RealType r[3]){
	for (int i=0; i<3; i++) {
		r[i] = 	m[i][0]*v[0] +
				m[i][1]*v[1] +
				m[i][2]*v[2];
	}
}

__device__ void RotateAroundZ(RealType v1[3], RealType alph, RealType v2[3]){
	RealType mat[3][3] = {{ cos(alph*deg2rad), -sin(alph*deg2rad), 0 },
						 { sin(alph*deg2rad),  cos(alph*deg2rad), 0 },
						 { 					0, 					 0,	1}};
	MatrixMultF(mat, v1, v2);
}

__device__ int CalcOmega(RealType x, RealType y, RealType z, RealType theta, RealType omegas[4], RealType etas[4]) {
	int nsol = 0;
	RealType v=sin(theta*deg2rad)*sqrt(x*x + y*y + z*z);
	if ( fabs(y) < 1e-4 ) {
		if (x != 0) {
			if (fabs(-v/x) <= 1) {
				omegas[nsol] = acos(-v/x)*rad2deg;
				nsol = nsol + 1;
				omegas[nsol] = -acos(-v/x)*rad2deg;
				nsol = nsol + 1;
			}
		}
	} else {
		RealType cosome1;
		RealType cosome2;
		if ((((2*v*x) / (y*y))*((2*v*x) / (y*y)) - 4*(1 + ((x*x) / (y*y)))*(((v*v) / (y*y)) - 1)) >= 0) {
			cosome1 = (-((2*v*x) / (y*y)) + sqrt((((2*v*x) / (y*y))*((2*v*x) / (y*y)) - 4*(1 + ((x*x) / (y*y)))*(((v*v) / (y*y)) - 1))))/(2*(1 + ((x*x) / (y*y))));
			if (fabs(cosome1) <= 1) {
				if (fabs(-x*cos(acos(cosome1)) + y*sin(acos(cosome1)) - v) < fabs(-x*cos(-acos(cosome1)) + y*sin(-acos(cosome1)) - v) ) {
					omegas[nsol] = acos(cosome1)*rad2deg;
					nsol = nsol + 1;
				}else {
					omegas[nsol] = -acos(cosome1)*rad2deg;
					nsol = nsol + 1;
				}
			}
			cosome2 = (-((2*v*x) / (y*y)) - sqrt((((2*v*x) / (y*y))*((2*v*x) / (y*y)) - 4*(1 + ((x*x) / (y*y)))*(((v*v) / (y*y)) - 1))))/(2*(1 + ((x*x) / (y*y))));
			if (fabs(cosome2) <= 1) {
				if (fabs(-x*cos(acos(cosome2)) + y*sin(acos(cosome2)) - v) < fabs(-x*cos(-acos(cosome2)) + y*sin(-acos(cosome2)) - v)) {
					omegas[nsol] = acos(cosome2)*rad2deg;
					nsol = nsol + 1;
				} else {
					omegas[nsol] = -acos(cosome2)*rad2deg;
					nsol = nsol + 1;
				}
			}
		}
	}
	RealType gw[3];
	RealType gv[3]={x,y,z};
	RealType eta;
	for (int indexOme = 0; indexOme < nsol; indexOme++) {
		RotateAroundZ(gv, omegas[indexOme], gw);
		eta = CalcEtaAngle(gw[1],gw[2]);
		etas[indexOme] = eta;
	}
	return nsol;
}

__device__ int CalcDiffrSpots_Furnace(RealType OrientMatrix[3][3],
	RealType *RingRadii, RealType *OmeBoxArr, int NOmegaRanges, RealType ExcludePoleAngle, RealType *spots, RealType *hkls, int *n_arr){
	int OmegaRangeNo;
	int KeepSpot;
	RealType Ghkl[3];
	RealType Gc[3];
	RealType omegas[4];
	RealType etas[4];
	RealType yl;
	RealType zl;
	int nspotsPlane;
	int spotnr = 0;
	for (int indexhkl=0; indexhkl < n_arr[1] ; indexhkl++)  {
		Ghkl[0] = hkls[indexhkl*7+0];
		Ghkl[1] = hkls[indexhkl*7+1];
		Ghkl[2] = hkls[indexhkl*7+2];
		MatrixMultF(OrientMatrix,Ghkl, Gc);
		nspotsPlane = CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl*7+5], omegas, etas);
		for (int i=0 ; i<nspotsPlane ; i++) {
			if ((fabs(etas[i]) < ExcludePoleAngle ) || ((180-fabs(etas[i])) < ExcludePoleAngle)) continue;
			yl = -(sin(deg2rad * etas[i])*RingRadii[(int)(hkls[indexhkl*7+3])]);
			zl =   cos(deg2rad * etas[i])*RingRadii[(int)(hkls[indexhkl*7+3])];
			for (OmegaRangeNo = 0 ; OmegaRangeNo < NOmegaRanges ; OmegaRangeNo++ ) {
				KeepSpot = 0;
				if ((omegas[i] > OmeBoxArr[OmegaRangeNo*6+4]) &&
					(omegas[i] < OmeBoxArr[OmegaRangeNo*6+5]) &&
					(yl > OmeBoxArr[OmegaRangeNo*6+0]) &&
					(yl < OmeBoxArr[OmegaRangeNo*6+1]) &&
					(zl > OmeBoxArr[OmegaRangeNo*6+2]) &&
					(zl < OmeBoxArr[OmegaRangeNo*6+3]) ) {
					KeepSpot = 1;
					break;
				}
			}
			if (KeepSpot) {
				spots[spotnr*N_COL_THEORSPOTS+0] = yl;
				spots[spotnr*N_COL_THEORSPOTS+1] = zl;
				spots[spotnr*N_COL_THEORSPOTS+2] = omegas[i];
				spots[spotnr*N_COL_THEORSPOTS+3] = hkls[indexhkl*7+3];
				spotnr++;
			}
		}
	}
	return spotnr;
}

__global__ void CompareDiffractionSpots(RealType *AllTheorSpots, RealType *RTParamArr,
	int maxPos, RealType *ResultArr, int PosResultArr, int *nTspotsArr,
	int *data, int *ndata, RealType *ObsSpots, RealType *etamargins, int *AllGrainSpots,
	RealType *IAs, int *n_arr, int *nMatchedArr, int n_min, int nOrients, RealType *GS){
	int nPos, orientPos, overallPos; // Position Calculate!!
	overallPos = blockIdx.x * blockDim.x + threadIdx.x;
	if (overallPos >= maxPos){
		return;
	}
	nPos = overallPos / nOrients;
	orientPos = overallPos % nOrients;
	nMatchedArr[overallPos] = 0;
	int n = n_min + nPos;
	RealType *TheorSpots;
	TheorSpots = AllTheorSpots + n_arr[1]*2*N_COL_THEORSPOTS*orientPos;
	int *GrainSpots;
	GrainSpots = AllGrainSpots + overallPos * n_arr[1] * 2;
	
	RealType y0, z0, xi, yi, zi, ys, zs,omega,RefRad;
	y0 = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 7];
	z0 = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 8];
	xi = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 9];
	yi = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 10];
	zi = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 11];
	ys = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 12];
	zs = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 13];
	omega = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 14];
	RefRad = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 15];
	RealType Displ_y, Displ_z;
	int nTspots, nMatched, MatchFound;
	RealType diffOmeBest, diffOme;
	long long int Pos;
	int nspots, DataPos, spotRow,spotRowBest;
	RealType omeo, ometh, gvo[3], gvth[3], lo, lth, tmp, go[3], gth[3],gs[3];
	RealType n_eta_bins, n_ome_bins, t;
	n_eta_bins = ceil(360.0 / RTParamArr[5 + MAX_N_RINGS + 4]);
	n_ome_bins = ceil(360.0 / RTParamArr[5 + MAX_N_RINGS + 5]);
	gs[0] = ((RTParamArr[3])*(n/xi)*xi*cos(omega*deg2rad)) +
		((ys - y0 + (RTParamArr[3])*(n/xi)*yi)*sin(omega*deg2rad));
	gs[1] = ((ys - y0 + (RTParamArr[3])*(n/xi)*yi)*cos(
		omega*deg2rad)) - ((RTParamArr[3])*(n/xi)*xi*sin(omega*deg2rad));
	gs[2] = zs - z0 + (RTParamArr[3])*(n/xi)*zi;
	GS[overallPos*3 + 0] = gs[0];
	GS[overallPos*3 + 1] = gs[1];
	GS[overallPos*3 + 2] = gs[2];
	nMatched = 0;
	nTspots = nTspotsArr[orientPos];
	IAs[overallPos] = 0;
	if (fabs(zs - z0 + (RTParamArr[3])*(n/xi)*zi) > RTParamArr[2] /2) {
		nMatchedArr[overallPos] = 0;
		return;
	}
	for (int sp = 0 ; sp < nTspots ; sp++) {
		ometh = TheorSpots[sp*N_COL_THEORSPOTS+2];
		t = (gs[0]*cos(deg2rad * ometh) - gs[1]*sin(deg2rad * ometh))/xi;
		Displ_y = ((gs[0]*sin(deg2rad * ometh))+ (gs[1]*cos(deg2rad * ometh))) - t* yi;
		Displ_z = gs[2] - t*zi;
		TheorSpots[sp*N_COL_THEORSPOTS+4] = TheorSpots[sp*N_COL_THEORSPOTS+0] +  Displ_y;
		TheorSpots[sp*N_COL_THEORSPOTS+5] = TheorSpots[sp*N_COL_THEORSPOTS+1] +  Displ_z;
		TheorSpots[sp*N_COL_THEORSPOTS+6] = CalcEtaAngle( TheorSpots[sp*N_COL_THEORSPOTS+4],
											TheorSpots[sp*N_COL_THEORSPOTS+5]);
		TheorSpots[sp*N_COL_THEORSPOTS+7] = sqrt(TheorSpots[sp*N_COL_THEORSPOTS+4] * TheorSpots[sp*N_COL_THEORSPOTS+4] +
										TheorSpots[sp*N_COL_THEORSPOTS+5] * TheorSpots[sp*N_COL_THEORSPOTS+5]) -
										RTParamArr[5 + (int)TheorSpots[sp*N_COL_THEORSPOTS+3]];
		MatchFound = 0;
		diffOmeBest = 100000;
		Pos = (((int) TheorSpots[sp*N_COL_THEORSPOTS+3])-1)*n_eta_bins*n_ome_bins
			+ ((int)(floor((180+TheorSpots[sp*N_COL_THEORSPOTS+6])/RTParamArr[5 + MAX_N_RINGS + 4])))*n_ome_bins +
			  ((int)floor((180+TheorSpots[sp*N_COL_THEORSPOTS+2])/RTParamArr[5 + MAX_N_RINGS + 5]));
		nspots = ndata[Pos*2];
		if (nspots == 0){
			continue;
		}
		DataPos = ndata[Pos*2+1];
		for (int iSpot = 0 ; iSpot < nspots; iSpot++ ) {
			spotRow = data[DataPos + iSpot];
			if ( fabs(TheorSpots[sp*N_COL_THEORSPOTS+7] - ObsSpots[spotRow*9+8]) < RTParamArr[5 + MAX_N_RINGS + 3] )  {
				if ( fabs(RefRad - ObsSpots[spotRow*9+3]) < RTParamArr[5 + MAX_N_RINGS + 2] ) {
					if ( fabs(TheorSpots[sp*N_COL_THEORSPOTS+6] - ObsSpots[spotRow*9+6]) < etamargins[(int) TheorSpots[sp*N_COL_THEORSPOTS+3]] ) {
						diffOme = fabs(TheorSpots[sp*N_COL_THEORSPOTS+2] - ObsSpots[spotRow*9+2]);
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
			GrainSpots[nMatched] = (int) ObsSpots[spotRowBest*9+4];
			omeo = ObsSpots[spotRowBest*9+2];
			ometh = TheorSpots[sp*N_COL_THEORSPOTS+2];
			RotateAroundZ(gs,omeo,go);
			RotateAroundZ(gs,ometh,gth);
			gvo[0] = (-1 + (RTParamArr[0] - go[0])/CalcLength((RTParamArr[0] - go[0]),(ObsSpots[spotRowBest*9+0] - go[1]),
				(ObsSpots[spotRowBest*9+1] - go[2]))) * cos(-omeo*deg2rad) - ((ObsSpots[spotRowBest*9+0] - go[1])/
				CalcLength((RTParamArr[0] - go[0]),(ObsSpots[spotRowBest*9+0] - go[1]),(ObsSpots[spotRowBest*9+1]
				- go[2]))) * sin(-omeo*deg2rad);
			gvo[1] = (-1 + (RTParamArr[0] - go[0])/CalcLength((RTParamArr[0] - go[0]),(ObsSpots[spotRowBest*9+0] - go[1]),
				(ObsSpots[spotRowBest*9+1] - go[2]))) * sin(-omeo*deg2rad) + ((ObsSpots[spotRowBest*9+0] - go[1])/
				CalcLength((RTParamArr[0] - go[0]),(ObsSpots[spotRowBest*9+0] - go[1]),(ObsSpots[spotRowBest*9+1]
				- go[2]))) * cos(-omeo*deg2rad);
			gvo[2] = (ObsSpots[spotRowBest*9+1] - go[2])/CalcLength((RTParamArr[0] - go[0]),(ObsSpots[spotRowBest*9+0] - go[1]),
				(ObsSpots[spotRowBest*9+1] - go[2]));
			gvth[0] = (-1 + (RTParamArr[0] - gth[0])/CalcLength((RTParamArr[0] - gth[0]),(TheorSpots[sp*N_COL_THEORSPOTS+0]
				- gth[1]),(TheorSpots[sp*N_COL_THEORSPOTS+1] - gth[2]))) * cos(-ometh*deg2rad) - ((TheorSpots[sp*N_COL_THEORSPOTS+0]
				- gth[1])/CalcLength((RTParamArr[0] - gth[0]),(TheorSpots[sp*N_COL_THEORSPOTS+0] - gth[1]),(TheorSpots[sp*N_COL_THEORSPOTS+1]
				- gth[2]))) * sin(-ometh*deg2rad);
			gvth[1] = (-1 + (RTParamArr[0] - gth[0])/CalcLength((RTParamArr[0] - gth[0]),(TheorSpots[sp*N_COL_THEORSPOTS+0]
				- gth[1]),(TheorSpots[sp*N_COL_THEORSPOTS+1] - gth[2]))) * sin(-ometh*deg2rad) + ((TheorSpots[sp*N_COL_THEORSPOTS+0]
				- gth[1])/CalcLength((RTParamArr[0] - gth[0]),(TheorSpots[sp*N_COL_THEORSPOTS+0] - gth[1]),(TheorSpots[sp*N_COL_THEORSPOTS+1]
				- gth[2]))) * cos(-ometh*deg2rad);
			gvth[2] = (TheorSpots[sp*N_COL_THEORSPOTS+1] - gth[2])/CalcLength((RTParamArr[0] - gth[0]),(TheorSpots[sp*N_COL_THEORSPOTS+0]
				- gth[1]),(TheorSpots[sp*N_COL_THEORSPOTS+1] - gth[2]));
			lo = CalcLength(gvo[0],gvo[1],gvo[2]);
			lth = CalcLength(gvth[0],gvth[1],gvth[2]);
			tmp = dot(gvo,gvth)/(lo*lth);
			if (tmp >1) tmp = 1;
			else if (tmp < -1) tmp = -1;
			IAs[overallPos] += rad2deg * acos(tmp);
			nMatched++;
		}
	}
	IAs[overallPos] /= (RealType)nMatched;
	nMatchedArr[overallPos] = nMatched;
}

__global__ void ReturnDiffractionSpots(RealType *RTParamArr, RealType *OmeBoxArr,
	int *IntParamArr, RealType *AllTheorSpots, RealType *hkls, int *n_arr, int PosResultArr,
	RealType *ResultArr, int norients, int *nSpotsArr, RealType *Orientations){
	int orient = blockIdx.x * blockDim.x + threadIdx.x;
	if (orient > norients) return;
	RealType *TheorSpots = AllTheorSpots + n_arr[1]*2*N_COL_THEORSPOTS*orient;
	RealType hkl[3], hklnormal[3];
	hkl[0] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 0];
	hkl[1] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 1];
	hkl[2] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 2];
	hklnormal[0] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 3];
	hklnormal[1] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 4];
	hklnormal[2] = ResultArr[PosResultArr * N_COLS_FRIEDEL_RESULTS + 5];
	RealType v[3];
	crossProduct(v, hkl, hklnormal);
	RealType RotMat[3][3];
	RealType RotMat2[3][3];
	RealType RotMat3[3][3];
	AxisAngle2RotMatrix(v, rad2deg * acos(dot(hkl, hklnormal)/
			(sqrt(hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2])*sqrt(
			hklnormal[0]*hklnormal[0] + hklnormal[1]*hklnormal[1] +
			hklnormal[2]*hklnormal[2]))), RotMat);
	AxisAngle2RotMatrix(hklnormal, orient*RTParamArr[4], RotMat2);
	MatrixMultF33(RotMat2, RotMat, RotMat3);
	nSpotsArr[orient] = CalcDiffrSpots_Furnace(RotMat3,
				RTParamArr + 5,  OmeBoxArr, IntParamArr[1],
				RTParamArr[5 + MAX_N_RINGS + 6], TheorSpots, hkls,n_arr);
	int PosUse = 9*orient;
	Orientations[PosUse + 0] = RotMat3[0][0];
	Orientations[PosUse + 1] = RotMat3[0][1];
	Orientations[PosUse + 2] = RotMat3[0][2];
	Orientations[PosUse + 3] = RotMat3[1][0];
	Orientations[PosUse + 4] = RotMat3[1][1];
	Orientations[PosUse + 5] = RotMat3[1][2];
	Orientations[PosUse + 6] = RotMat3[2][0];
	Orientations[PosUse + 7] = RotMat3[2][1];
	Orientations[PosUse + 8] = RotMat3[2][2];
}

__global__ void MakeOrientations(RealType *ResultArr, int *HKLints,
	int *IntParamArr, RealType *RTParamArr, int *ResultOut, int sumTotal){
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID >= sumTotal) return;
	RealType y0, xi, yi, ys;
	y0 = ResultArr[ID * N_COLS_FRIEDEL_RESULTS + 7];
	xi = ResultArr[ID * N_COLS_FRIEDEL_RESULTS + 9];
	yi = ResultArr[ID * N_COLS_FRIEDEL_RESULTS + 10];
	ys = ResultArr[ID * N_COLS_FRIEDEL_RESULTS + 12];
	RealType RotationAngles = CalcRotationAngle(((int) ResultArr[ID * N_COLS_FRIEDEL_RESULTS + 6]), HKLints, IntParamArr, RTParamArr);
	ResultOut[ID*N_COLS_ORIENTATION_NUMBERS + 0] = (int) RotationAngles/RTParamArr[4];
	ResultOut[ID*N_COLS_ORIENTATION_NUMBERS + 1] = (int)((((-(2*yi*(ys-y0))+sqrt((2*yi*(ys-y0))*(2*yi*(ys-y0))
			- 4*(xi*xi + yi*yi)*((ys-y0)*(ys-y0) - RTParamArr[1]*RTParamArr[1]
			)))/(2*(xi*xi + yi*yi)) + 20)*xi)/(RTParamArr[3]));
	ResultOut[ID*N_COLS_ORIENTATION_NUMBERS + 2] = (2*ResultOut[ID*N_COLS_ORIENTATION_NUMBERS + 1] + 1) * ResultOut[ID*N_COLS_ORIENTATION_NUMBERS + 0];
}

__device__ int TryFriedel(RealType ys, RealType zs,
	RealType ttheta, RealType eta, RealType omega, int ringno,
	RealType Ring_rad, RealType Rsample, RealType Hbeam, RealType OmeTol,
	RealType RadiusTol,	RealType *ObsSpotsLab, RealType *hkls, int *n_arr,
	RealType *RTParamArr, RealType *ResultArray, int rowID, RealType RefRad){
	int NrFriedel = 0;
	RealType OmeF;
	if (omega < 0 )  OmeF = omega + 180;
	else OmeF = omega - 180;
	int quadr_coeff2 = 0, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0;
	RealType eta_Hbeam, y0_max_z0, y0_min_z0, y0_max = 0, y0_min = 0, z0_min = 0, z0_max = 0;
	if (eta > 90) eta_Hbeam = 180 - eta;
	else if (eta < -90) eta_Hbeam = 180 - fabs(eta);
	else eta_Hbeam = 90 - fabs(eta);
	Hbeam = Hbeam + 2*(Rsample*tan(ttheta*deg2rad))*(sin(eta_Hbeam*deg2rad));
	RealType eta_pole = (1 + rad2deg*acos(1-(Hbeam/Ring_rad)));
	RealType eta_equator = (1 + rad2deg*acos(1-(Rsample/Ring_rad)));
	if ((eta >= eta_pole) && (eta <= (90-eta_equator)) ) { // % 1st quadrant
		quadr_coeff = 1;
		coeff_y0 = -1;
		coeff_z0 = 1;
	} else if ( (eta >=(90+eta_equator)) && (eta <= (180-eta_pole)) ) {//% 4th quadrant
		quadr_coeff = 2;
		coeff_y0 = -1;
		coeff_z0 = -1;
	} else if ( (eta >= (-90+eta_equator) ) && (eta <= -eta_pole) )   { // % 2nd quadrant
		quadr_coeff = 2;
		coeff_y0 = 1;
		coeff_z0 = 1;
	} else if ( (eta >= (-180+eta_pole) ) && (eta <= (-90-eta_equator)) )  { // % 3rd quadrant
		quadr_coeff = 1;
		coeff_y0 = 1;
		coeff_z0 = -1;
	} else quadr_coeff = 0;
	RealType y0_max_Rsample = ys + Rsample;
	RealType y0_min_Rsample = ys - Rsample;
	RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
	RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
	if (quadr_coeff == 1) {
		y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
		y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
	} else if (quadr_coeff == 2) {
		y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
		y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
	}
	if (quadr_coeff > 0)  {
		y0_max = min(y0_max_Rsample, y0_max_z0);
		y0_min = max(y0_min_Rsample, y0_min_z0);
	} else {
		if ((eta > -eta_pole) && (eta < eta_pole ))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = 1;
		} else if (eta < (-180+eta_pole))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = -1;
		} else if (eta > (180-eta_pole))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = -1;
		} else if (( eta > (90-eta_equator)) && (eta < (90+eta_equator)) ) {
			quadr_coeff2 = 1;
			z0_max = z0_max_Hbeam;
			z0_min = z0_min_Hbeam;
			coeff_y0 = -1;
		} else if ((eta > (-90-eta_equator)) && (eta < (-90+eta_equator)) ) {
			quadr_coeff2 = 1;
			z0_max = z0_max_Hbeam;
			z0_min = z0_min_Hbeam;
			coeff_y0 = 1;
		}
	}
	if ( quadr_coeff2 == 0 ) {
		z0_min = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_min * y0_min));
		z0_max = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_max * y0_max));
	} else {
		y0_min = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min * z0_min));
		y0_max = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max * z0_max));
	}
	RealType dYMin = ys - y0_min;
	RealType dYMax = ys - y0_max;
	RealType dZMin = zs - z0_min;
	RealType dZMax = zs - z0_max;
	RealType YMinFrIdeal =  y0_min;
	RealType YMaxFrIdeal =  y0_max;
	RealType ZMinFrIdeal = -z0_min;
	RealType ZMaxFrIdeal = -z0_max;
	RealType YMinFr = YMinFrIdeal - dYMin;
	RealType YMaxFr = YMaxFrIdeal - dYMax;
	RealType ZMinFr = ZMinFrIdeal + dZMin;
	RealType ZMaxFr = ZMaxFrIdeal + dZMax;
	RealType Eta1, Eta2;
	Eta1 = CalcEtaAngle((YMinFr + ys),(ZMinFr - zs));
	Eta2 = CalcEtaAngle((YMaxFr + ys),(ZMaxFr - zs));
	RealType EtaMinF = min(Eta1,Eta2);
	RealType EtaMaxF = max(Eta1,Eta2);
	RealType yf, zf, EtaTransf, radius, IdealY, IdealZ, xi,yi,zi, hklnormal[3], hkl[3];
	for (int r=0 ; r < n_arr[0] ; r++) {
		if ( ((int)ObsSpotsLab[r*9+5]) != ringno ) continue; // Not a Friedel pair
		if ( fabs(ObsSpotsLab[r*9+2] - OmeF) > OmeTol) continue; // Not a Friedel pair
		yf = ObsSpotsLab[r*9+0];
		zf = ObsSpotsLab[r*9+1];
		EtaTransf = CalcEtaAngle(yf + ys, zf - zs);
		radius = sqrt((yf + ys)*(yf + ys) + (zf - zs)*(zf - zs));
		if ( fabs(radius - 2*Ring_rad) > RadiusTol)  continue;
		if (( EtaTransf < EtaMinF) || (EtaTransf > EtaMaxF) ) continue;

		IdealY = Ring_rad*(ys - ((-ObsSpotsLab[r*9+0] + ys)/2))/sqrt((
			ys - ((-ObsSpotsLab[r*9+0] + ys)/2))*(ys - ((-ObsSpotsLab[r*9+0] +
			ys)/2))+(zs - (( ObsSpotsLab[r*9+1] + zs)/2))*(zs - ((
			ObsSpotsLab[r*9+1] + zs)/2)));
		IdealZ = Ring_rad*(zs - (( ObsSpotsLab[r*9+1] + zs)/2))/sqrt((
			ys - ((-ObsSpotsLab[r*9+0] + ys)/2))*(ys - ((-ObsSpotsLab[r*9+0] +
			ys)/2))+(zs - (( ObsSpotsLab[r*9+1] + zs)/2))*(zs - ((
			ObsSpotsLab[r*9+1] + zs)/2)));
		xi = RTParamArr[0]/CalcLength(RTParamArr[0],IdealY,IdealZ);
		yi = IdealY/CalcLength(RTParamArr[0],IdealY,IdealZ);
		zi = IdealZ/CalcLength(RTParamArr[0],IdealY,IdealZ);
		hklnormal[0] = (-1 + xi) * cos(-omega*deg2rad) - yi * sin(-omega*deg2rad);
		hklnormal[1] = (-1 + xi) * sin(-omega*deg2rad) + yi * cos(-omega*deg2rad);
		hklnormal[2] = zi;
		for (int i=0;i<n_arr[1];i++){
			if ((int) hkls[i*7+3] == ringno){
				hkl[0] = hkls[i*7+0];
				hkl[1] = hkls[i*7+1];
				hkl[2] = hkls[i*7+2];
				break;
			}
		}
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 0]  = hkl[0];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 1]  = hkl[1];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 2]  = hkl[2];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 3]  = hklnormal[0];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 4]  = hklnormal[1];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 5]  = hklnormal[2];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 6]  = (RealType) ringno;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 7]  = IdealY;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 8]  = IdealZ;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 9]  = xi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 10] = yi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 11] = zi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 12] = ys;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 13] = zs;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 14] = omega;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + NrFriedel * N_COLS_FRIEDEL_RESULTS + 15] = RefRad;
		NrFriedel++;
   }
   return NrFriedel;
}

__global__ void FriedelFinding (int *SpotIDs, RealType *ObsSpotsLab,
	RealType *hkls, int *n_arr, int *IntParamArr, RealType *RTParamArr, RealType *ResultArray, int *nNormals){
	int rowID = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowID >= n_arr[2]) return;
	int SpotID = SpotIDs[rowID];
	int SpotRowNo = FindRowInMatrix(ObsSpotsLab, n_arr[0], N_COL_OBSSPOTS, 4, SpotID);
	if (SpotRowNo == -1) {
		printf("WARNING: SpotId %d not found in spots file! Ignoring this spotID. n_spots = %d\n", SpotID, n_arr[0]);
		return;
	}
	RealType RefRad = ObsSpotsLab[SpotRowNo*9+3];
	int nPlaneNormals = 0;
	if (IntParamArr[2] == 1) {
		nPlaneNormals = TryFriedel(ObsSpotsLab[SpotRowNo*9+0], ObsSpotsLab[SpotRowNo*9+1],
			ObsSpotsLab[SpotRowNo*9+7], ObsSpotsLab[SpotRowNo*9+6], ObsSpotsLab[SpotRowNo*9+2], (int) ObsSpotsLab[SpotRowNo*9+5],
			RTParamArr[(int) ObsSpotsLab[SpotRowNo*9+5] + 5], RTParamArr[1], RTParamArr[2], RTParamArr[5 + MAX_N_RINGS + 0],
			RTParamArr[5 + MAX_N_RINGS + 3],ObsSpotsLab, hkls, n_arr, RTParamArr, ResultArray,rowID,RefRad);
		nNormals[rowID] = nPlaneNormals;
	}
}


int main(int argc, char *argv[]){
	printf("\n\n\t\t\tGPU Indexer v1.0\nContact hsharma@anl.gov in case of questions about the MIDAS project.\n\n");
	RealType iStart = cpuSecond();
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    size_t gpuGlobalMem = deviceProp.totalGlobalMem;
    fprintf(stderr, "GPU global memory = %zu MBytes\n", gpuGlobalMem/(1024*1024));
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	char folder[4096];
	struct ParametersStruct Parameters;
	char ParamFN[4096];
	getcwd(folder,sizeof(folder));
	sprintf(ParamFN,"%s/%s",folder,argv[1]);
	printf("Reading parameters from file: %s.\n", ParamFN);
	int returncode = ReadParams(ParamFN, &Parameters);

	int *SpotIDs_h;
	SpotIDs_h = (int *) malloc(sizeof(*SpotIDs_h)* MAX_N_SPOTS);
	char spotIDsfn[4096];
	sprintf(spotIDsfn,"%s/%s",folder,Parameters.IDsFileName);
	fflush(stdout);
	int nSpotIDs=0;
	FILE *IDsFile = fopen(spotIDsfn,"r");
	char line[MAX_LINE_LENGTH];
	while (fgets(line,MAX_LINE_LENGTH,IDsFile)!=NULL){
		SpotIDs_h[nSpotIDs] = atoi(line);
		nSpotIDs++;
	}

    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy to spotIDs Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	RealType hkls[MAX_N_HKLS*7];
	int HKLints[MAX_N_HKLS*4];
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	char aline[1024],dummy[1024];
	fgets(aline,1000,hklf);
	int Rnr,i;
	int hi,ki,li;
	RealType hc,kc,lc,RRd,Ds,tht;
	int n_hkls_h = 0;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf",&hi,&ki,&li,&Ds,&Rnr,&hc,&kc,&lc,&tht,dummy,&RRd);
		for (i=0;i<Parameters.NrOfRings;i++){
			if (Rnr == Parameters.RingNumbers[i]){
				HKLints[n_hkls_h*4+0] = hi;
				HKLints[n_hkls_h*4+1] = ki;
				HKLints[n_hkls_h*4+2] = li;
				HKLints[n_hkls_h*4+3] = Rnr;
				hkls[n_hkls_h*7+0] = hc;
				hkls[n_hkls_h*7+1] = kc;
				hkls[n_hkls_h*7+2] = lc;
				hkls[n_hkls_h*7+3] = (RealType)Rnr;
				hkls[n_hkls_h*7+4] = Ds;
				hkls[n_hkls_h*7+5] = tht;
				hkls[n_hkls_h*7+6] = RRd;
				n_hkls_h++;
			}
		}
	}

	char datafn[4096];
	sprintf(datafn,"%s/%s",folder,"Data.bin");
	char ndatafn[4096];
	sprintf(ndatafn,"%s/%s",folder,"nData.bin");
	char spotsfn[4096];
	sprintf(spotsfn,"%s/%s",folder,"Spots.bin");
	char extrafn[4096];
	sprintf(extrafn,"%s/%s",folder,"ExtraInfo.bin");

	FILE *fData = fopen(datafn,"r");
	FILE *fnData = fopen(ndatafn,"r");
	FILE *fSpots = fopen(spotsfn,"r");
	FILE *fExtraInfo = fopen(extrafn,"r");

	RealType *hkls_d, *etamargins_d;
	int  *HKLints_d;

	RealType etamargins[MAX_N_RINGS];
	for ( i = 0 ; i < MAX_N_RINGS ; i++) {
		if ( Parameters.RingRadii[i] == 0)  {
			etamargins[i] = 0;
		}else {
			etamargins[i] = rad2deg * atan(Parameters.MarginEta/Parameters.RingRadii[i]) + 0.5 * Parameters.StepsizeOrient;
		}
	}
	cudaMalloc((RealType **)&hkls_d,n_hkls_h*7*sizeof(RealType));
	cudaMalloc((int **)&HKLints_d,n_hkls_h*4*sizeof(int));
	cudaMalloc((RealType **)&etamargins_d,MAX_N_RINGS*sizeof(RealType));
	cudaMemcpy(hkls_d,hkls,n_hkls_h*7*sizeof(RealType),cudaMemcpyHostToDevice);
	cudaMemcpy(HKLints_d,HKLints,n_hkls_h*4*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(etamargins_d,etamargins,MAX_N_RINGS*sizeof(RealType),cudaMemcpyHostToDevice);

	int nspids = nSpotIDs, *sps;
	cudaMalloc((int **)&sps,nspids*sizeof(int));
	cudaMemcpy(sps,SpotIDs_h,nspids*sizeof(int),cudaMemcpyHostToDevice);

	RealType *ObsSpotsLab, *spots_h;
	fseek(fSpots,0L,SEEK_END);
	long long sizeSpots = ftell(fSpots);
	rewind(fSpots);
	spots_h = (RealType *)malloc(sizeSpots);
	fread(spots_h,sizeSpots,1,fSpots);
	cudaMalloc((RealType **)&ObsSpotsLab,(size_t)sizeSpots);
	cudaMemcpy(ObsSpotsLab,spots_h,sizeSpots,cudaMemcpyHostToDevice);
	free(spots_h);
	int n_spots_h = ((int)sizeSpots)/(9*sizeof(double));
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "End data Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "FewSpotIDs Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	int *n_arr, n_arr_h[3];
	cudaMalloc((int **)&n_arr,sizeof(int)*3);
	n_arr_h[0] = n_spots_h;
	n_arr_h[1] = n_hkls_h;
	n_arr_h[2] = nspids;
	cudaMemcpy(n_arr,n_arr_h,3*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "SpotsInfo Theor and BestGrains Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	int *IntParamArr, IntParamArr_h[3];
	IntParamArr_h[0] = Parameters.SpaceGroupNum;
	IntParamArr_h[1] = Parameters.NoOfOmegaRanges;
	IntParamArr_h[2] = Parameters.UseFriedelPairs;
	cudaMalloc((int **)&IntParamArr, sizeof(int)*3);
	cudaMemcpy(IntParamArr,IntParamArr_h,sizeof(int)*3,cudaMemcpyHostToDevice);

	RealType *RTParamArr, RTParamArr_h[5 + MAX_N_RINGS + 8 + 6];
	RTParamArr_h[0] = Parameters.Distance;
	RTParamArr_h[1] = Parameters.Rsample;
	RTParamArr_h[2] = Parameters.Hbeam;
	RTParamArr_h[3] = Parameters.StepsizePos;
	RTParamArr_h[4] = Parameters.StepsizeOrient;
	for (int cntr=0;cntr<MAX_N_RINGS;cntr++) RTParamArr_h[5+cntr] = Parameters.RingRadii[cntr];
	RTParamArr_h[5+MAX_N_RINGS+0] = Parameters.MarginOme;
	RTParamArr_h[5+MAX_N_RINGS+1] = Parameters.MarginEta;
	RTParamArr_h[5+MAX_N_RINGS+2] = Parameters.MarginRad;
	RTParamArr_h[5+MAX_N_RINGS+3] = Parameters.MarginRadial;
	RTParamArr_h[5+MAX_N_RINGS+4] = Parameters.EtaBinSize;
	RTParamArr_h[5+MAX_N_RINGS+5] = Parameters.OmeBinSize;
	RTParamArr_h[5+MAX_N_RINGS+6] = Parameters.ExcludePoleAngle;
	RTParamArr_h[5+MAX_N_RINGS+7] = Parameters.MinMatchesToAcceptFrac;
	for (int cntr=0;cntr<6;cntr++) RTParamArr_h[5+MAX_N_RINGS+8+cntr] = Parameters.ABCABG[cntr];
	cudaMalloc((RealType **)&RTParamArr,(19+MAX_N_RINGS)*sizeof(RealType));
	cudaMemcpy(RTParamArr,RTParamArr_h,(19+MAX_N_RINGS)*sizeof(RealType),cudaMemcpyHostToDevice);

	RealType *OmeBoxArr, OmeBoxArr_h[Parameters.NoOfOmegaRanges * 6];
	for (int cntr=0;cntr<Parameters.NoOfOmegaRanges;cntr++){
		OmeBoxArr_h[cntr*6 + 0] = Parameters.BoxSizes[cntr][0];
		OmeBoxArr_h[cntr*6 + 1] = Parameters.BoxSizes[cntr][1];
		OmeBoxArr_h[cntr*6 + 2] = Parameters.BoxSizes[cntr][2];
		OmeBoxArr_h[cntr*6 + 3] = Parameters.BoxSizes[cntr][3];
		OmeBoxArr_h[cntr*6 + 4] = Parameters.OmegaRanges[cntr][0];
		OmeBoxArr_h[cntr*6 + 5] = Parameters.OmegaRanges[cntr][1];
	}
	cudaMalloc((RealType **)&OmeBoxArr,Parameters.NoOfOmegaRanges * 6 * sizeof(RealType));
	cudaMemcpy(OmeBoxArr,OmeBoxArr_h,Parameters.NoOfOmegaRanges * 6 * sizeof(RealType),cudaMemcpyHostToDevice);

	int dim = nspids;
	dim3 block (256);
	dim3 grid ((dim/block.x)+1);
	printf("Time elapsed before FriedelFinding: %fs\n",cpuSecond()-iStart);

	RealType *ResultArray;
	int *nNormals;
	cudaMalloc((RealType **)&ResultArray,sizeof(RealType)*nspids*MAX_N_FRIEDEL_PAIRS*N_COLS_FRIEDEL_RESULTS);
	cudaMalloc((int **)&nNormals,sizeof(int)*nspids);

	cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Finding Friedel Pairs Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	FriedelFinding<<<grid,block>>>(sps, ObsSpotsLab, hkls_d,n_arr,IntParamArr,RTParamArr,ResultArray,nNormals);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	
	int *data, *nData, *data_h, *nData_h;

	fseek(fData,0L,SEEK_END);
	long long sizeData = ftell(fData);
	rewind(fData);
	data_h = (int *)malloc(sizeData);
	fread(data_h,sizeData,1,fData);
	cudaMalloc((int **)&data,(size_t)sizeData);
	cudaMemcpy(data,data_h,sizeData,cudaMemcpyHostToDevice);
	free(data_h);

    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy data Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	fseek(fnData,0L,SEEK_END);
	long long sizenData = ftell(fnData);
	rewind(fnData);
	nData_h = (int *)malloc(sizenData);
	fread(nData_h,sizenData,1,fnData);
	cudaMalloc((int **)&nData,(size_t)sizenData);
	cudaMemcpy(nData,nData_h,sizenData,cudaMemcpyHostToDevice);
	free(nData_h);

    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy ndata Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	int *nNormals_h;
	nNormals_h = (int *) malloc(sizeof(int) * nspids);
	cudaMemcpy(nNormals_h, nNormals, sizeof(int) * nspids, cudaMemcpyDeviceToHost);

	RealType *ResultArray_h;
	ResultArray_h = (RealType *) malloc(sizeof(RealType)*nspids*MAX_N_FRIEDEL_PAIRS*N_COLS_FRIEDEL_RESULTS);
	cudaMemcpy(ResultArray_h,ResultArray,sizeof(RealType)*nspids*MAX_N_FRIEDEL_PAIRS*N_COLS_FRIEDEL_RESULTS,cudaMemcpyDeviceToHost);
	cudaFree(ResultArray);

	int sumTotal=0, *startingIDs;
	startingIDs = (int *) malloc(sizeof(int) * nspids);
	for (int i=0;i<nspids;i++){
		startingIDs[i] = sumTotal;
		sumTotal += nNormals_h[i];
	}

	RealType *ResultArr, *ResultArr_h;
	int currentpos = 0, outerpos = 0, totalpos = 0;
	ResultArr_h = (RealType *) malloc(sizeof(RealType)*N_COLS_FRIEDEL_RESULTS*sumTotal);
	for (int i=0;i<nspids;i++){
		currentpos = 0;
		for (int j=0;j<nNormals_h[i];j++){
			memcpy(ResultArr_h + (totalpos * N_COLS_FRIEDEL_RESULTS),
				ResultArray_h + (outerpos*MAX_N_FRIEDEL_PAIRS*N_COLS_FRIEDEL_RESULTS + currentpos *N_COLS_FRIEDEL_RESULTS),
				sizeof(RealType)*N_COLS_FRIEDEL_RESULTS);
			currentpos++;
			totalpos++;
		}
		outerpos++;
	}
	if (totalpos != sumTotal){
		printf("Something wrong.\n");
		return 0;
	}

    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy data Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	printf("Time elapsed before making orientations: %fs\n",cpuSecond()-iStart);

	dim3 blocka (32);
	dim3 grida ((sumTotal/blocka.x)+1);
	cudaMalloc((RealType **)&ResultArr,sizeof(RealType)*N_COLS_FRIEDEL_RESULTS*sumTotal);
	CHECK(cudaMemcpy(ResultArr, ResultArr_h,sizeof(RealType)*N_COLS_FRIEDEL_RESULTS*sumTotal,cudaMemcpyHostToDevice));

	int *ResultMakeOrientations, *ResultMakeOrientations_h;
	cudaMalloc((int **)&ResultMakeOrientations,N_COLS_ORIENTATION_NUMBERS*sumTotal*sizeof(int));
	cudaMemset(ResultMakeOrientations,0,N_COLS_ORIENTATION_NUMBERS*sumTotal*sizeof(int));

	//// Now generate candidates and match
	MakeOrientations<<<grida,blocka>>>(ResultArr, HKLints_d, IntParamArr, RTParamArr, ResultMakeOrientations,sumTotal);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	
	ResultMakeOrientations_h = (int *) malloc(N_COLS_ORIENTATION_NUMBERS*sumTotal*sizeof(int));
	cudaMemcpy(ResultMakeOrientations_h,ResultMakeOrientations,N_COLS_ORIENTATION_NUMBERS*sumTotal*sizeof(int),cudaMemcpyDeviceToHost);

    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy before data Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	long long int totalJobs = 0;
	int maxJobs=0, maxJobsOrient=0;
	for (int i=0;i<sumTotal;i++){
		totalJobs += ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 2];
		if (ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 2] > maxJobs) maxJobs = ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 2];
		if (ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 0] > maxJobsOrient) maxJobsOrient = ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 0];
	}

	printf("Total Jobs: %lld, MaxJobs for one combination: %d\n",totalJobs,maxJobs);

	RealType *AllTheorSpots, *IAs, *IAs_h, *GS, *Orientations, *GS_h, *Orientations_h, *AllInfo;
	int *AllGrainSpots,*nSpotsArr,*nMatchedArr,*nMatchedArr_h,*nSpotsArr_h, *SpotsInfoTotal;
	cudaMalloc((RealType **)&AllTheorSpots,maxJobsOrient*n_hkls_h*N_COL_THEORSPOTS*2*sizeof(RealType));
	cudaMalloc((int **)&AllGrainSpots,maxJobs*n_hkls_h*2*sizeof(int));
	cudaMalloc((int **)&nSpotsArr,maxJobsOrient*sizeof(int));
	cudaMalloc((RealType **)&IAs,maxJobs*sizeof(RealType));
	cudaMalloc((int **)&nMatchedArr,maxJobs*sizeof(int));
	cudaMemset(nMatchedArr,0,maxJobs*sizeof(int));
	nMatchedArr_h = (int *) malloc(maxJobs*sizeof(int));
	nSpotsArr_h = (int *) malloc(maxJobsOrient*sizeof(int));
	IAs_h = (RealType *) malloc(maxJobs*sizeof(RealType));
	memset(nMatchedArr_h,0,maxJobs*sizeof(int));
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Memcpy ndata Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	RealType bestFraction, tempFraction;
	int nJobsOrient, posResultArr, nJobsTotal, n_min, BestPosition;
	RealType bestIA, tempIA;
	cudaMalloc((RealType **)&GS,3*maxJobs*sizeof(RealType));
	cudaMalloc((RealType **)&Orientations,9*maxJobsOrient*sizeof(RealType));
	GS_h = (RealType *) malloc(3*maxJobs*sizeof(RealType));
	Orientations_h = (RealType *) malloc(9*maxJobsOrient*sizeof(RealType));
	AllInfo = (RealType *) malloc(N_COL_GRAINMATCHES*sumTotal*sizeof(RealType));
	memset(AllInfo,0,N_COL_GRAINMATCHES*sumTotal*sizeof(RealType));
	SpotsInfoTotal = (int *) malloc(sumTotal*n_hkls_h*2*sizeof(int));
	memset(SpotsInfoTotal,0,sumTotal*n_hkls_h*2*sizeof(int));
	printf("Time elapsed before calculation of matches: %fs\n",cpuSecond()-iStart);
	for (int jobNr=0;jobNr<sumTotal;jobNr++){//sumTotal
		posResultArr = jobNr;
		nJobsOrient = ResultMakeOrientations_h[jobNr*N_COLS_ORIENTATION_NUMBERS + 0];
		dim3 blockb (32);
		dim3 gridb ((nJobsOrient/blockb.x)+1);
		ReturnDiffractionSpots<<<gridb,blockb>>>(RTParamArr,OmeBoxArr,IntParamArr,
				AllTheorSpots,hkls_d,n_arr,posResultArr,ResultArr,nJobsOrient,nSpotsArr,
				Orientations);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(nSpotsArr_h,nSpotsArr,nJobsOrient*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(Orientations_h,Orientations,nJobsOrient*9*sizeof(RealType),cudaMemcpyDeviceToHost);
		nJobsTotal = ResultMakeOrientations_h[jobNr*N_COLS_ORIENTATION_NUMBERS + 2];
		dim3 blockc (32);
		dim3 gridc ((nJobsTotal/blockc.x)+1);
		n_min = -ResultMakeOrientations_h[jobNr*N_COLS_ORIENTATION_NUMBERS + 1];
		CompareDiffractionSpots<<<gridc,blockc>>>(AllTheorSpots,RTParamArr,
			nJobsTotal, ResultArr, posResultArr, nSpotsArr, data, nData, ObsSpotsLab,
			etamargins_d, AllGrainSpots, IAs, n_arr, nMatchedArr, n_min, nJobsOrient,GS);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(nMatchedArr_h,nMatchedArr,nJobsTotal*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(GS_h,GS,nJobsTotal*3*sizeof(RealType),cudaMemcpyDeviceToHost);
		cudaMemcpy(IAs_h,IAs,nJobsTotal*sizeof(RealType),cudaMemcpyDeviceToHost);
		bestFraction = 0.0;
		bestIA = 1000.0;
		for (int idx=0;idx<nJobsTotal;idx++){
			tempFraction = ((RealType)nMatchedArr_h[idx])/((RealType)nSpotsArr_h[idx%(-2*n_min + 1 )]);
			tempIA = IAs_h[idx];
			if (tempFraction > bestFraction && tempFraction <= 1 && tempFraction >= 0){
				bestIA = tempIA;
				bestFraction = tempFraction;
				BestPosition = idx;
			}else if(tempFraction == bestFraction && tempIA < bestIA){
				bestIA = tempIA;
				BestPosition = idx;
			}
		}
		if (bestFraction >= Parameters.MinMatchesToAcceptFrac){
			cudaMemcpy(SpotsInfoTotal+jobNr*n_hkls_h*2, AllGrainSpots+BestPosition*n_hkls_h*2,nMatchedArr_h[BestPosition]*sizeof(int),cudaMemcpyDeviceToHost);
			AllInfo[jobNr*N_COL_GRAINMATCHES + 0] = bestIA;
			AllInfo[jobNr*N_COL_GRAINMATCHES + 1] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 0];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 2] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 1];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 3] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 2];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 4] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 3];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 5] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 4];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 6] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 5];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 7] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 6];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 8] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 7];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 9] = Orientations_h[BestPosition%(-2*n_min+1)*9 + 8];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 10] = GS_h[BestPosition*3 + 0];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 11] = GS_h[BestPosition*3 + 1];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 12] = GS_h[BestPosition*3 + 2];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 13] = (RealType)nSpotsArr_h[BestPosition%(-2*n_min+1)];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 14] = (RealType)nMatchedArr_h[BestPosition];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 15] = bestFraction;
		}
	}
	printf("Time elapsed after calculation of matches: %fs\n",cpuSecond()-iStart);

	// Now sort all the results.
	RealType *SaveAllInfo;
	int *SaveSpotsInfoAll;
	SaveAllInfo = (RealType *) malloc(nspids*(N_COL_GRAINMATCHES+1)*sizeof(RealType));
	SaveSpotsInfoAll = (int *) malloc(nspids*n_hkls_h*2*sizeof(int));
	memset(SaveAllInfo,0,nspids*(N_COL_GRAINMATCHES+1)*sizeof(RealType));
	memset(SaveSpotsInfoAll,0,nspids*n_hkls_h*2*sizeof(int));
	int StartingPosition, EndPosition, bestPos;
	for (int i=0;i<nspids;i++){
		StartingPosition = startingIDs[i];
		EndPosition = StartingPosition + nNormals_h[i];
		bestFraction = 0.0;
		bestIA = 1000.0;
		bestPos = -1;
		for (int PlanePos=StartingPosition; PlanePos<EndPosition; PlanePos++){
			tempIA = AllInfo[PlanePos*N_COL_GRAINMATCHES + 0];
			tempFraction = AllInfo[PlanePos*N_COL_GRAINMATCHES + 15];
			if (tempFraction > bestFraction){
				bestFraction = tempFraction;
				bestPos = PlanePos;
				bestIA = tempIA;
			} else if (tempFraction == bestFraction && tempIA < bestIA){
				bestIA = tempIA;
				bestPos = PlanePos;
			}
		}
		if (bestPos >-1){
			SaveAllInfo[i*(N_COL_GRAINMATCHES+1) + 0] = (RealType)SpotIDs_h[i];
			memcpy(SaveAllInfo+i*(N_COL_GRAINMATCHES+1) + 1,AllInfo + bestPos*N_COL_GRAINMATCHES, N_COL_GRAINMATCHES);
			memcpy(SaveSpotsInfoAll+i*n_hkls_h*2, SpotsInfoTotal + bestPos*n_hkls_h*2, n_hkls_h*2);
		}
	}
	
	free(nMatchedArr_h);
	free(nSpotsArr_h);
	free(IAs_h);
	free(spots_h);
	free(data_h);
	free(nData_h);
	free(nNormals_h);
	free(ResultArray_h);
	free(startingIDs);
	free(ResultArr_h);
	free(ResultMakeOrientations_h);
	free(GS_h);
	free(Orientations_h);
	free(AllInfo);
	free(SpotsInfoTotal);
	free(SaveAllInfo);
	cudaDeviceSynchronize();
	cudaFree(GS);
	cudaFree(Orientations);
	cudaFree(AllTheorSpots);
	cudaFree(AllGrainSpots);
	cudaFree(nSpotsArr);
	cudaFree(IAs);
	cudaFree(nMatchedArr);
	cudaFree(data);
	cudaFree(nData);
	cudaFree(sps);
	cudaFree(ObsSpotsLab);
	cudaFree(ResultArr);
	cudaFree(hkls_d);
	cudaFree(HKLints_d);
	cudaFree(etamargins_d);
	cudaFree(n_arr);
	cudaFree(IntParamArr);
	cudaFree(RTParamArr);
	cudaFree(OmeBoxArr);
	cudaFree(ResultArray);
	cudaFree(nNormals);
	cudaFree(ResultMakeOrientations);

	printf("Time elapsed after sorting the results: %fs\n Now refining results.\n",cpuSecond()-iStart);

	fseek(fExtraInfo,0L,SEEK_END);
	long long sizeExtra = ftell(fExtraInfo);
	rewind(fExtraInfo);
	RealType *ExtraInfo_h;
	ExtraInfo_h = (double *)malloc(sizeExtra);
	fread(ExtraInfo_h,sizeExtra,1,fExtraInfo);
	
	int sizeAllSpots = (sizeExtra/14)*8;
	int nExtraSpots = sizeAllSpots/(8*sizeof(double));
	RealType *AllSpotsYZO_h;
	AllSpotsYZO_h = (double *) malloc(sizeAllSpots);
	for (int i=0;i<nExtraSpots;i++){
		AllSpotsYZO_h[i*8+0] = ExtraInfo_h[i*14+0];
		AllSpotsYZO_h[i*8+1] = ExtraInfo_h[i*14+1];
		AllSpotsYZO_h[i*8+2] = ExtraInfo_h[i*14+2];
		AllSpotsYZO_h[i*8+3] = ExtraInfo_h[i*14+4];
		AllSpotsYZO_h[i*8+4] = ExtraInfo_h[i*14+8];
		AllSpotsYZO_h[i*8+5] = ExtraInfo_h[i*14+9];
		AllSpotsYZO_h[i*8+6] = ExtraInfo_h[i*14+10];
		AllSpotsYZO_h[i*8+7] = ExtraInfo_h[i*14+5];
	}

	/*char outfnall[MAX_LINE_LENGTH], outfnspots[MAX_LINE_LENGTH];
	sprintf(outfnall, "%s/AllInfo.bin",Parameters.OutputFolder);
	sprintf(outfnspots, "%s/SpotsInfo.bin",Parameters.OutputFolder);
	FILE *fAllInfo = fopen(outfnall,"w"), *fSpotsInfo = fopen(outfnspots,"w");
	fwrite(SaveAllInfo,nspids*(N_COL_GRAINMATCHES+1)*sizeof(RealType),1,fAllInfo);
	fwrite(SaveSpotsInfoAll,nspids*n_hkls_h*2*sizeof(int),1,fSpotsInfo);
	fclose(fAllInfo);
	fclose(fSpotsInfo);*/

	cudaDeviceReset();

	printf("Time elapsed: %fs\n",cpuSecond()-iStart);

	return 0;
}
