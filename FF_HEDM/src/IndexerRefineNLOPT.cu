// TODO: Implement FriedelMixed, other 2 are done (Friedel and noFriedel)


#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define RealType double

// conversions constants
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-12
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))

// max array sizes
#define MAX_N_SPOTS 6000000   // max nr of observed spots that can be stored
#define MAX_N_STEPS 1000      // Max nr of pos steps, when stepping along the diffracted ray
#define MAX_N_OR 36000        // max nr of trial orientations that can be stored (360/0.01);
#define MAX_N_MATCHES 1       // max nr of grain matches for 1 spot
#define MAX_N_RINGS 500       // max nr of rings that can be stored (applies to the arrays ringttheta, ringhkl, etc)
#define MAX_N_HKLS 500       // max nr of hkls that can be stored
#define MAX_N_OMEGARANGES 72  // max nr of omegaranges in input file (also max no of box sizes)
#define N_COL_THEORSPOTS 8   // number of items that is stored for each calculated spot (omega, eta, etc)
#define N_COL_OBSSPOTS 9      // number of items stored for each obs spots
#define N_COL_GRAINSPOTS 17   // nr of columns for output: y, z, omega, differences for spots of grain matches
#define N_COL_GRAINMATCHES 16 // nr of columns for output: the Matches (summary)
#define MAX_LINE_LENGTH 4096
#define MAX_N_FRIEDEL_PAIRS 1000
#define MAX_N_EVALS 1000
#define N_COLS_FRIEDEL_RESULTS 16
#define N_COLS_ORIENTATION_NUMBERS 3
#define MaxNSpotsBest 10

__device__ typedef RealType (*nlopt_func)(int n, RealType *x, void *func_data);

typedef enum {
     NLOPT_FAILURE = -1, /* generic failure code */
     NLOPT_INVALID_ARGS = -2,
     NLOPT_OUT_OF_MEMORY = -3,
     NLOPT_ROUNDOFF_LIMITED = -4,
     NLOPT_FORCED_STOP = -5,
     NLOPT_SUCCESS = 1, /* generic success code */
     NLOPT_STOPVAL_REACHED = 2,
     NLOPT_FTOL_REACHED = 3,
     NLOPT_XTOL_REACHED = 4,
     NLOPT_MAXEVAL_REACHED = 5,
     NLOPT_MAXTIME_REACHED = 6
} nlopt_result;

typedef struct {
     unsigned n;
     RealType minf_max;
     RealType ftol_rel;
     RealType xtol_rel;
     int nevals, maxeval;
} nlopt_stopping;

__device__ int relstop(RealType vold, RealType vnew, RealType reltol)
{
     if (vold != vold) return 0;
     return(fabs(vnew - vold) < reltol * (fabs(vnew) + fabs(vold)) * 0.5
	    || (reltol > 0 && vnew == vold));
}

__device__ int nlopt_stop_ftol(const nlopt_stopping *s, RealType f, RealType oldf)
{
     return (relstop(oldf, f, s->ftol_rel));
}

__device__ int nlopt_stop_f(const nlopt_stopping *s, RealType f, RealType oldf)
{
     return (f <= s->minf_max || nlopt_stop_ftol(s, f, oldf));
}

__device__ int nlopt_stop_x(const nlopt_stopping *s, const RealType *x, const RealType *oldx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(oldx[i], x[i], s->xtol_rel))
	       return 0;
     return 1;
}

__device__ int nlopt_stop_dx(const nlopt_stopping *s, const RealType *x, const RealType *dx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(x[i] - dx[i], x[i], s->xtol_rel))
	       return 0;
     return 1;
}

__device__ int nlopt_stop_evals(const nlopt_stopping *s)
{
     return (s->maxeval > 0 && s->nevals >= s->maxeval);
}

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

/* return 1 if a and b are approximately equal relative to floating-point
   precision, 0 otherwise */
__device__ int close(RealType a, RealType b)
{
     return (fabs(a - b) <= 1e-13 * (fabs(a) + fabs(b)));
}

__device__ int reflectpt(int n, RealType *xnew, 
		     const RealType *c, RealType scale, const RealType *xold,
		     const RealType *lb, const RealType *ub)
{
     int equalc = 1, equalold = 1, i;
     for (i = 0; i < n; ++i) {
	  RealType newx = c[i] + scale * (c[i] - xold[i]);
	  if (newx < lb[i]) newx = lb[i];
	  if (newx > ub[i]) newx = ub[i];
	  equalc = equalc && close(newx, c[i]);
	  equalold = equalold && close(newx, xold[i]);
	  xnew[i] = newx;
     }
     return !(equalc || equalold);
}

#define CHECK_EVAL(xc,fc) 						  \
 stop->nevals++;							  \
 if ((fc) <= *minf) {							  \
   *minf = (fc); memcpy(x, (xc), n * sizeof(RealType));			  \
   if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; } \
 }									  \
 if (nlopt_stop_evals(stop)) { ret=NLOPT_MAXEVAL_REACHED; goto done; }	  \

__device__ nlopt_result nldrmd_minimize_(int n, nlopt_func f, void *f_data,
			     const RealType *lb, const RealType *ub, /* bounds */
			     RealType *x, /* in: initial guess, out: minimizer */
			     RealType *minf,
			     const RealType *xstep, /* initial step sizes */
			     nlopt_stopping *stop,
			     RealType psi, RealType *scratch,
			     RealType *fdiff)
{
     RealType *pts; /* (n+1) x (n+1) array of n+1 points plus function val [0] */
     RealType *c; /* centroid * n */
     RealType *xcur; /* current point */
     int i, j;
     RealType ninv = 1.0 / n;
     nlopt_result ret = NLOPT_SUCCESS;
     RealType init_diam = 0;
     RealType *highi;

     pts = scratch;
     c = scratch + (n+1)*(n+1);
     xcur = c + n;

     *fdiff = HUGE_VAL;

     /* initialize the simplex based on the starting xstep */
     for (i=0;i<n;i++) pts[1+i] = x[i];
     //memcpy(pts+1, x, sizeof(RealType)*n);
     pts[0] = *minf;
     if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; }
     for (i = 0; i < n; ++i) {
	  RealType *pt = pts + (i+1)*(n+1);
	  for (j=0;j<n;j++) pt[1+j] = x[j];
	  //memcpy(pt+1, x, sizeof(RealType)*n);
	  pt[1+i] += xstep[i];
	  if (pt[1+i] > ub[i]) {
	       if (ub[i] - x[i] > fabs(xstep[i]) * 0.1)
		    pt[1+i] = ub[i];
	       else /* ub is too close to pt, go in other direction */
		    pt[1+i] = x[i] - fabs(xstep[i]);
	  }
	  if (pt[1+i] < lb[i]) {
	       if (x[i] - lb[i] > fabs(xstep[i]) * 0.1)
		    pt[1+i] = lb[i];
	       else {/* lb is too close to pt, go in other direction */
		    pt[1+i] = x[i] + fabs(xstep[i]);
		    if (pt[1+i] > ub[i]) /* go towards further of lb, ub */
			 pt[1+i] = 0.5 * ((ub[i] - x[i] > x[i] - lb[i] ?
					   ub[i] : lb[i]) + x[i]);
	       }
	  }
	  if (close(pt[1+i], x[i])) { ret=NLOPT_FAILURE; goto done; }
	  pt[0] = f(n, pt+1, f_data);
	  CHECK_EVAL(pt+1, pt[0]);
     }
 restart:
     for (i = 0; i < n + 1; ++i)
     // Create list to have f(x) and x values, it doesn't need to be a sorted list.
     // This could be avoided by using pts to calculate high and low. 

     while (1) {
	  RealType fl = pts[0], *xl = pts + 1;
	  RealType fh = pts[0], *xh = pts + 1;
	  highi = pts;
	  for (i = 1; i < n+1; ++i){
		  if (fl < pts[i*(n+1)]){
			  fl = pts[i*(n+1)];
			  xl = pts + i*(n+1) + 1;
		  }
		  if (fh > pts[i*(n+1)]){
			  fh = pts[i*(n+1)];
			  xh = pts + i*(n+1) + 1;
			  highi = pts + i*(n+1);
		  }
	  }
	  RealType fr;

	  *fdiff = fh - fl;

	  if (init_diam == 0) /* initialize diam. for psi convergence test */
	       for (i = 0; i < n; ++i) init_diam += fabs(xl[i] - xh[i]);

	  if (psi <= 0 && nlopt_stop_ftol(stop, fl, fh)) {
	       ret = NLOPT_FTOL_REACHED;
	       goto done;
	  }

	  /* compute centroid */
	  memset(c, 0, sizeof(RealType)*n);
	  for (i = 0; i < n + 1; ++i) {
	       RealType *xi = pts + i*(n+1) + 1;
	       if (xi != xh)
		    for (j = 0; j < n; ++j)
			 c[j] += xi[j];
	  }
	  for (i = 0; i < n; ++i) c[i] *= ninv;

	  /* x convergence check: find xcur = max radius from centroid */
	  memset(xcur, 0, sizeof(RealType)*n);
	  for (i = 0; i < n + 1; ++i) {
               RealType *xi = pts + i*(n+1) + 1;
	       for (j = 0; j < n; ++j) {
		    RealType dx = fabs(xi[j] - c[j]);
		    if (dx > xcur[j]) xcur[j] = dx;
	       }
	  }
	  for (i = 0; i < n; ++i) xcur[i] += c[i];
	  if (psi > 0) {
	       RealType diam = 0;
	       for (i = 0; i < n; ++i) diam += fabs(xl[i] - xh[i]);
	       if (diam < psi * init_diam) {
		    ret = NLOPT_XTOL_REACHED;
		    goto done;
	       }
	  }
	  else if (nlopt_stop_x(stop, c, xcur)) {
	       ret = NLOPT_XTOL_REACHED;
	       goto done;
	  }

	  /* reflection */
	  if (!reflectpt(n, xcur, c, 1.0, xh, lb, ub)) { 
	       ret=NLOPT_XTOL_REACHED; goto done; 
	  }
	  fr = f(n, xcur, f_data);
	  CHECK_EVAL(xcur, fr);

	  if (fr < fl) { /* new best point, expand simplex */
	       if (!reflectpt(n, xh, c, 2.0, xh, lb, ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fh = f(n, xh, f_data);
	       CHECK_EVAL(xh, fh);
	       if (fh >= fr) { /* expanding didn't improve */
		    fh = fr;
		    memcpy(xh, xcur, sizeof(RealType)*n);
	       }
	  }
	  else if (fr < fh){//rb_tree_pred(high)->k[0]) { /* accept new point */ // how is this done is unclear
	       memcpy(xh, xcur, sizeof(RealType)*n);
	       fh = fr;
	  }
	  else { /* new worst point, contract */
	       RealType fc;
	       if (!reflectpt(n,xcur,c, fh <= fr ? -0.5 : 0.5, xh, lb,ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fc = f(n, xcur, f_data);
	       CHECK_EVAL(xcur, fc);
	       if (fc < fr && fc < fh) { /* successful contraction */
		    memcpy(xh, xcur, sizeof(RealType)*n);
		    fh = fc;
	       }
	       else { /* failed contraction, shrink simplex */
		    for (i = 0; i < n+1; ++i) {
			 RealType *pt = pts + i * (n+1);
			 if (pt+1 != xl) {
			      if (!reflectpt(n,pt+1, xl,-0.5,pt+1, lb,ub)) {
				   ret = NLOPT_XTOL_REACHED;
				   goto done;
			      }
			      pt[0] = f(n, pt+1, f_data);
			      CHECK_EVAL(pt+1, pt[0]);
			 }
		    }
		    goto restart;
	       }
	  }
      *highi = fh;
     }
     
done:
     return ret;
}

__device__ nlopt_result nldrmd_minimize(int n, nlopt_func f, void *f_data,
			     const RealType *lb, const RealType *ub, /* bounds */
			     RealType *x, /* in: initial guess, out: minimizer */
			     RealType *minf,
			     const RealType *xstep, /* initial step sizes */
			     nlopt_stopping *stop, RealType *scratch)
{
     nlopt_result ret;
     RealType fdiff;

     *minf = f(n, x, f_data);
     stop->nevals++;
     if (*minf < stop->minf_max) return NLOPT_MINF_MAX_REACHED;
     if (nlopt_stop_evals(stop)) return NLOPT_MAXEVAL_REACHED;

     ret = nldrmd_minimize_(n, f, f_data, lb, ub, x, minf, xstep, stop,
			    0.0, scratch, &fdiff);
     return ret;
}
//END NLOPT NELDERMEAD

//BEGIN NLDRMD FUNCTION scratch space: 3n+(n+1)*(n+1)
__device__ void nelmin ( RealType fn ( int n_fun, RealType *x, void *data ),
  int n, RealType *start, RealType *xmin,
  RealType *lb, RealType *ub, RealType *scratch, RealType *ynewlo,
  RealType reqmin, RealType *step, int konvge, int kcount, 
  int *icount, int *numres, int *ifault, void *data_t){
  RealType ccoeff = 0.5;
  RealType del;
  RealType dn;
  RealType dnn;
  RealType ecoeff = 2.0;
  RealType eps = 0.001;
  int i;
  int ihi;
  int ilo;
  int j;
  int jcount;
  int l;
  int nn;
  RealType *p;
  RealType *p2star;
  RealType *pbar;
  RealType *pstar;
  RealType rcoeff = 1.0;
  RealType rq;
  RealType x;
  RealType *y;
  RealType y2star;
  RealType ylo;
  RealType ystar;
  RealType z;
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
  dn = ( RealType ) ( n );
  nn = n + 1;
  dnn = ( RealType ) ( nn );
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
   char ResultFolder[MAX_LINE_LENGTH];        // Results folder
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
		str = "ResultFolder ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, Params->ResultFolder );
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
	RealType *RingRadii, RealType *OmeBoxArr, int NOmegaRanges,
	RealType ExcludePoleAngle, RealType *spots, RealType *hkls, int *n_arr){
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
	RealType nrhkls;
	for (int indexhkl=0; indexhkl < n_arr[1] ; indexhkl++)  {
		Ghkl[0] = hkls[indexhkl*7+0];
		Ghkl[1] = hkls[indexhkl*7+1];
		Ghkl[2] = hkls[indexhkl*7+2];
		MatrixMultF(OrientMatrix,Ghkl, Gc);
		nspotsPlane = CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl*7+5], omegas, etas);
		nrhkls = (RealType)indexhkl*2 + 1;
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
				spots[spotnr*N_COL_THEORSPOTS+4] = nrhkls;
				nrhkls++;
				spotnr++;
			}
		}
	}
	return spotnr;
}

__device__ int CalcOmegaStrains(
          RealType x,
          RealType y,
          RealType z,
          RealType theta,
          RealType omegas[4],
          RealType etas[4])
{
    int nsol = 0;
    RealType ome;
    RealType len= sqrt(x*x + y*y + z*z);
    RealType v=sin(theta*deg2rad)*len;

    RealType almostzero = 1e-4;
    if ( fabs(y) < almostzero ) {
        if (x != 0) {
            RealType cosome1 = -v/x;
            if (fabs(cosome1) <= 1) {
                ome = acos(cosome1)*rad2deg;
                omegas[nsol] = ome;
                nsol++;
                omegas[nsol] = -ome;
                nsol++;
            }
        }
    }
    else {
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
                eqa = -x*cos(ome1a) + y*sin(ome1a);
                diffa = fabs(eqa - v);
                eqb = -x*cos(ome1b) + y*sin(ome1b);
                diffb = fabs(eqb - v);
                if (diffa < diffb ) {
                    omegas[nsol] = ome1a*rad2deg;
                    nsol ++;
                }
                else {
                    omegas[nsol] = ome1b*rad2deg;
                    nsol++;
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
                    omegas[nsol] = ome2a*rad2deg;
                    nsol++;
                }
                else {
                    omegas[nsol] = ome2b*rad2deg;
                    nsol++;
                }
            }
        }
    }
    RealType gw[3];
    RealType gv[3]={x,y,z};
    RealType eta;
    int indexOme;
    for (indexOme = 0; indexOme < nsol; indexOme++) {
        RotateAroundZ(gv, omegas[indexOme], gw);
        eta = CalcEtaAngle(gw[1],gw[2]);
        etas[indexOme] = eta;
    }
    return(nsol);
}

// Returns more stuff needed for Fitting
// N_COL_THEORSPOTS is 8, so we can store everything we need.
__device__ int CalcDiffrSpots(RealType OrientMatrix[3][3],
	RealType *RingRadii, RealType *OmeBoxArr, int NOmegaRanges,
	RealType ExcludePoleAngle, RealType *spots, RealType *hkls, int *n_arr){
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
	RealType GCr[3], NGc;
	RealType nrhkls, Ds;
	for (int indexhkl=0; indexhkl < n_arr[1] ; indexhkl++)  {
		Ghkl[0] = hkls[indexhkl*7+0];
		Ghkl[1] = hkls[indexhkl*7+1];
		Ghkl[2] = hkls[indexhkl*7+2];
		MatrixMultF(OrientMatrix,Ghkl, Gc);
		nspotsPlane = CalcOmegaStrains(Gc[0], Gc[1], Gc[2], hkls[indexhkl*7+5], omegas, etas);
		NGc=sqrt((Gc[0]*Gc[0])+(Gc[1]*Gc[1])+(Gc[2]*Gc[2]));
		Ds=hkls[indexhkl*7+4];
        GCr[0]=Ds*Gc[0]/NGc;
        GCr[1]=Ds*Gc[1]/NGc;
        GCr[2]=Ds*Gc[2]/NGc;
        nrhkls = (RealType)indexhkl*2 + 1;
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
				spots[spotnr*8+0] = yl;
				spots[spotnr*8+1] = zl;
				spots[spotnr*8+2] = omegas[i];
				spots[spotnr*8+3] = GCr[0];
				spots[spotnr*8+4] = GCr[1];
				spots[spotnr*8+5] = GCr[2];
				spots[spotnr*8+6] = hkls[indexhkl*7+3];
				spots[spotnr*8+7] = nrhkls;
				nrhkls++;
				spotnr++;
			}
		}
	}
	return spotnr;
}

__device__ int CalcDiffrSpotsStrained(RealType OrientMatrix[3][3],
	RealType *OmeBoxArr, int NOmegaRanges, RealType ExcludePoleAngle, 
	RealType *spots, RealType *hkls, int *n_arr){
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
	RealType GCr[3], NGc;
	RealType nrhkls, Ds, RingRad;
	for (int indexhkl=0; indexhkl < n_arr[1] ; indexhkl++)  {
		Ghkl[0] = hkls[indexhkl*7+0];
		Ghkl[1] = hkls[indexhkl*7+1];
		Ghkl[2] = hkls[indexhkl*7+2];
		MatrixMultF(OrientMatrix,Ghkl, Gc);
		nspotsPlane = CalcOmegaStrains(Gc[0], Gc[1], Gc[2], hkls[indexhkl*7+5], omegas, etas);
		NGc=sqrt((Gc[0]*Gc[0])+(Gc[1]*Gc[1])+(Gc[2]*Gc[2]));
		Ds=hkls[indexhkl*7+4];
        GCr[0]=Ds*Gc[0]/NGc;
        GCr[1]=Ds*Gc[1]/NGc;
        GCr[2]=Ds*Gc[2]/NGc;
        nrhkls = (RealType)indexhkl*2 + 1;
        RingRad = hkls[indexhkl*7+6];
		for (int i=0 ; i<nspotsPlane ; i++) {
			if ((fabs(etas[i]) < ExcludePoleAngle ) || ((180-fabs(etas[i])) < ExcludePoleAngle)) continue;
			yl = -(sin(deg2rad * etas[i])*RingRad);
			zl =   cos(deg2rad * etas[i])*RingRad;
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
				spots[spotnr*8+0] = yl;
				spots[spotnr*8+1] = zl;
				spots[spotnr*8+2] = omegas[i];
				spots[spotnr*8+3] = GCr[0];
				spots[spotnr*8+4] = GCr[1];
				spots[spotnr*8+5] = GCr[2];
				spots[spotnr*8+6] = hkls[indexhkl*7+3];
				spots[spotnr*8+7] = nrhkls;
				nrhkls++;
				spotnr++;
			}
		}
	}
	return spotnr;
}

#define sind(x) sin(deg2rad*x)
#define cosd(x) cos(deg2rad*x)
#define tand(x) tan(deg2rad*x)
#define asind(x) rad2deg*asin(x)
#define acosd(x) rad2deg*acos(x)
#define atand(x) rad2deg*atan(x)

__device__ void CorrectHKLsLatCInd(RealType *LatC_d, RealType *hklsIn,
	int *n_arr, RealType *RTParamArr, RealType *hklscorr, int *HKLints_d){
	RealType *hkls;
	hkls = hklscorr;
	RealType a=LatC_d[0],b=LatC_d[1],c=LatC_d[2],alph=LatC_d[3],bet=LatC_d[4],gamma=LatC_d[5];
	int hklnr;
	RealType ginit[3], SinA, SinB, SinG, CosA, CosB, CosG, GammaPr, BetaPr, SinBetaPr,
		Vol, APr, BPr, CPr, B[3][3], GCart[3], Ds, Theta, Rad;
	SinA = sind(alph);
	SinB = sind(bet);
	SinG = sind(gamma);
	CosA = cosd(alph);
	CosB = cosd(bet);
	CosG = cosd(gamma);
	GammaPr = acosd((CosA*CosB - CosG)/(SinA*SinB));
	BetaPr  = acosd((CosG*CosA - CosB)/(SinG*SinA));
	SinBetaPr = sind(BetaPr);
	Vol = (a*(b*(c*(SinA*(SinBetaPr*(SinG))))));
	APr = b*c*SinA/Vol;
	BPr = c*a*SinB/Vol;
	CPr = a*b*SinG/Vol;
	B[0][0] = APr;
	B[0][1] = (BPr*cosd(GammaPr));
	B[0][2] = (CPr*cosd(BetaPr));
	B[1][0] = 0,
	B[1][1] = (BPr*sind(GammaPr));
	B[1][2] = (-CPr*SinBetaPr*CosA);
	B[2][0] = 0;
	B[2][1] = 0;
	B[2][2] = (CPr*SinBetaPr*SinA);
	for (hklnr=0;hklnr<n_arr[1];hklnr++){
		ginit[0] = (RealType) HKLints_d[hklnr*4+0];
		ginit[1] = (RealType) HKLints_d[hklnr*4+1];
		ginit[2] = (RealType) HKLints_d[hklnr*4+2];
		MatrixMultF(B,ginit,GCart);
		Ds = 1/(sqrt((GCart[0]*GCart[0])+(GCart[1]*GCart[1])+(GCart[2]*GCart[2])));
		hkls[hklnr*7+0] = GCart[0];
		hkls[hklnr*7+1] = GCart[1];
		hkls[hklnr*7+2] = GCart[2];
		hkls[hklnr*7+3] = hklsIn[hklnr*7+3];
        hkls[hklnr*7+4] = Ds;
        Theta = (asind((RTParamArr[5+MAX_N_RINGS+8+6])/(2*Ds)));
        hkls[hklnr*7+5] = Theta;
        Rad = RTParamArr[0]*(tand(2*Theta));
        hkls[hklnr*7+6] = Rad;
	}
}

__device__ void Euler2OrientMat(RealType Euler[3], RealType m_out[3][3]){
    RealType psi, phi, theta, cps, cph, cth, sps, sph, sth;
    psi = Euler[0];
    phi = Euler[1];
    theta = Euler[2];
    cps = cosd(psi) ; cph = cosd(phi); cth = cosd(theta);
    sps = sind(psi); sph = sind(phi); sth = sind(theta);
    m_out[0][0] = cth * cps - sth * cph * sps;
    m_out[0][1] = -cth * cph * sps - sth * cps;
    m_out[0][2] = sph * sps;
    m_out[1][0] = cth * sps + sth * cph * cps;
    m_out[1][1] = cth * cph * cps - sth * sps;
    m_out[1][2] = -sph * cps;
    m_out[2][0] = sth * sph;
    m_out[2][1] = cth * sph;
    m_out[2][2] = cph;
}

void Euler2OrientMat_h(RealType Euler[3], RealType m_out[3][3]){
    RealType psi, phi, theta, cps, cph, cth, sps, sph, sth;
    psi = Euler[0];
    phi = Euler[1];
    theta = Euler[2];
    cps = cosd(psi) ; cph = cosd(phi); cth = cosd(theta);
    sps = sind(psi); sph = sind(phi); sth = sind(theta);
    m_out[0][0] = cth * cps - sth * cph * sps;
    m_out[0][1] = -cth * cph * sps - sth * cps;
    m_out[0][2] = sph * sps;
    m_out[1][0] = cth * sps + sth * cph * cps;
    m_out[1][1] = cth * cph * cps - sth * sps;
    m_out[1][2] = -sph * cps;
    m_out[2][0] = sth * sph;
    m_out[2][1] = cth * sph;
    m_out[2][2] = cph;
}


__device__ void DisplacementInTheSpot(RealType a, RealType b, RealType c,
RealType xi, RealType yi, RealType zi, RealType omega, RealType wedge,
RealType chi, RealType *Displ_y, RealType *Displ_z){
	RealType sinOme=sind(omega), cosOme=cosd(omega), AcosOme=a*cosOme, BsinOme=b*sinOme;
	RealType XNoW=AcosOme-BsinOme, YNoW=(a*sinOme)+(b*cosOme), ZNoW=c;
	RealType WedgeRad=deg2rad*wedge, CosW=cos(WedgeRad), SinW=sin(WedgeRad), XW=XNoW*CosW-ZNoW*SinW, YW=YNoW;
    RealType ZW=(XNoW*SinW)+(ZNoW*CosW), ChiRad=deg2rad*chi, CosC=cos(ChiRad), SinC=sin(ChiRad), XC=XW;
    RealType YC=(CosC*YW)-(SinC*ZW), ZC=(SinC*YW)+(CosC*ZW);
    RealType IK[3],NormIK; IK[0]=xi-XC; IK[1]=yi-YC; IK[2]=zi-ZC; NormIK=sqrt((IK[0]*IK[0])+(IK[1]*IK[1])+(IK[2]*IK[2]));
    IK[0]=IK[0]/NormIK;IK[1]=IK[1]/NormIK;IK[2]=IK[2]/NormIK;
    *Displ_y = YC - ((XC*IK[1])/(IK[0]));
    *Displ_z = ZC - ((XC*IK[2])/(IK[0]));
}

__device__
void CorrectForOme(RealType yc, RealType zc, RealType Lsd, RealType OmegaIni,
	RealType wl, RealType wedge, RealType *ysOut, RealType *zsOut, RealType *OmegaOut)
{
	RealType SinTheta = sin(deg2rad*rad2deg*atan(sqrt((yc*yc)+(zc*zc))/Lsd)/2);
	RealType CosTheta = cos(deg2rad*rad2deg*atan(sqrt((yc*yc)+(zc*zc))/Lsd)/2);
	RealType ds = 2*SinTheta/wl;
	RealType CosW = cos(deg2rad*wedge);
	RealType SinW = sin(deg2rad*wedge);
	RealType SinEta = sin(deg2rad*CalcEtaAngle(yc,zc));
	RealType CosEta = cos(deg2rad*CalcEtaAngle(yc,zc));
	RealType k1 = -ds*SinTheta;
	RealType k2 = -ds*CosTheta*SinEta;
	RealType k3 =  ds*CosTheta*CosEta;
	if (CalcEtaAngle(yc,zc) == 90){k3 = 0; k2 = -CosTheta;}
	else if (CalcEtaAngle(yc,zc) == -90) {k3 = 0; k2 = CosTheta;}
	RealType k1f = (k1*CosW) + (k3*SinW);
	RealType k2f = k2;
	RealType k3f = (k3*CosW) - (k1*SinW);
	RealType G1a = (k1f*cos(deg2rad*OmegaIni)) + (k2f*sin(deg2rad*OmegaIni));
	RealType G2a = (k2f*cos(deg2rad*OmegaIni)) - (k1f*sin(deg2rad*OmegaIni));
	RealType G3a = k3f;
	RealType LenGa = sqrt((G1a*G1a)+(G2a*G2a)+(G3a*G3a));
	RealType g1 = G1a*ds/LenGa;
	RealType g2 = G2a*ds/LenGa;
	RealType g3 = G3a*ds/LenGa;
	SinW = 0;
	CosW = 1;
	RealType LenG = sqrt((g1*g1)+(g2*g2)+(g3*g3));
	RealType k1i = -(LenG*LenG*wl)/2;
	RealType A = (k1i+(g3*SinW))/(CosW);
	RealType a_Sin = (g1*g1) + (g2*g2);
	RealType b_Sin = 2*A*g2;
	RealType c_Sin = (A*A) - (g1*g1);
	RealType a_Cos = a_Sin;
	RealType b_Cos = -2*A*g1;
	RealType c_Cos = (A*A) - (g2*g2);
	RealType Par_Sin = (b_Sin*b_Sin) - (4*a_Sin*c_Sin);
	RealType Par_Cos = (b_Cos*b_Cos) - (4*a_Cos*c_Cos);
	RealType P_check_Sin = 0;
	RealType P_check_Cos = 0;
	RealType P_Sin,P_Cos;
	if (Par_Sin >=0) P_Sin=sqrt(Par_Sin);
	else {P_Sin=0;P_check_Sin=1;}
	if (Par_Cos>=0) P_Cos=sqrt(Par_Cos);
	else {P_Cos=0;P_check_Cos=1;}
	RealType SinOmega1 = (-b_Sin-P_Sin)/(2*a_Sin);
	RealType SinOmega2 = (-b_Sin+P_Sin)/(2*a_Sin);
	RealType CosOmega1 = (-b_Cos-P_Cos)/(2*a_Cos);
	RealType CosOmega2 = (-b_Cos+P_Cos)/(2*a_Cos);
	if      (SinOmega1 < -1) SinOmega1=0;
	else if (SinOmega1 >  1) SinOmega1=0;
	else if (SinOmega2 < -1) SinOmega2=0;
	else if (SinOmega2 >  1) SinOmega2=0;
	if      (CosOmega1 < -1) CosOmega1=0;
	else if (CosOmega1 >  1) CosOmega1=0;
	else if (CosOmega2 < -1) CosOmega2=0;
	else if (CosOmega2 >  1) CosOmega2=0;
	if (P_check_Sin == 1){SinOmega1=0;SinOmega2=0;}
	if (P_check_Cos == 1){CosOmega1=0;CosOmega2=0;}
	RealType Option1 = fabs((SinOmega1*SinOmega1)+(CosOmega1*CosOmega1)-1);
	RealType Option2 = fabs((SinOmega1*SinOmega1)+(CosOmega2*CosOmega2)-1);
	RealType Omega1, Omega2;
	if (Option1 < Option2){Omega1=rad2deg*atan2(SinOmega1,CosOmega1);Omega2=rad2deg*atan2(SinOmega2,CosOmega2);}
	else {Omega1=rad2deg*atan2(SinOmega1,CosOmega2);Omega2=rad2deg*atan2(SinOmega2,CosOmega1);}
	RealType OmeDiff1 = fabs(Omega1-OmegaIni);
	RealType OmeDiff2 = fabs(Omega2-OmegaIni);
	RealType Omega;
	if (OmeDiff1 < OmeDiff2)Omega=Omega1;
	else Omega=Omega2;
	RealType SinOmega=sin(deg2rad*Omega);
	RealType CosOmega=cos(deg2rad*Omega);
	RealType Fact = (g1*CosOmega) - (g2*SinOmega);
	RealType Eta = CalcEtaAngle(k2,k3);
	RealType Sin_Eta = sin(deg2rad*Eta);
	RealType Cos_Eta = cos(deg2rad*Eta);
	*ysOut = -Lsd*tan(deg2rad*2*rad2deg*asin(wl*LenG/2))*Sin_Eta;
	*zsOut = Lsd*tan(deg2rad*2*rad2deg*asin(wl*LenG/2))*Cos_Eta;
	*OmegaOut = Omega;
}

__device__ void SpotToGv(RealType xi, RealType yi, RealType zi, RealType Omega,
	RealType theta, RealType *g1, RealType *g2, RealType *g3)
{
	RealType CosOme = cosd(Omega), SinOme = sind(Omega), eta = CalcEtaAngle(yi,zi), TanEta = tand(-eta), SinTheta = sind(theta);
    RealType CosTheta = cosd(theta), CosW = 1, SinW = 0, k3 = SinTheta*(1+xi)/((yi*TanEta)+zi), k2 = TanEta*k3, k1 = -SinTheta;
    if (eta == 90){
		k3 = 0;
		k2 = -CosTheta;
	} else if (eta == -90){
		k3 = 0;
		k2 = CosTheta;
	}
    RealType k1f = (k1*CosW) + (k3*SinW), k3f = (k3*CosW) - (k1*SinW), k2f = k2;
    *g1 = (k1f*CosOme) + (k2f*SinOme);
    *g2 = (k2f*CosOme) - (k1f*SinOme);
    *g3 = k3f;
}

struct func_data_pos_ini{
	int *IntParamArr;
	RealType *OmeBoxArr;
	RealType *spotsYZO;
	RealType *hkls;
	int *HKLInts;
	int nMatched;
	RealType *RTParamArr;
	int *n_arr;
	RealType *TheorSpots;
	RealType *hklspace;
};

__device__ RealType pf_posIni(int n, RealType *x, void *f_data_trial){
	struct func_data_pos_ini *f_data = (struct func_data_pos_ini *) f_data_trial;
	RealType *TheorSpots, *spotsYZO, *RTParamArr, *OmeBoxArr, *hkls,
		*hklscorr, *SpotsCorrected;
	OmeBoxArr = &(f_data->OmeBoxArr[0]);
	spotsYZO = &(f_data->spotsYZO[0]);
	hkls = &(f_data->hkls[0]);
	RTParamArr = &(f_data->RTParamArr[0]);
	TheorSpots = &(f_data->TheorSpots[0]);
	int nMatched = f_data->nMatched;
	int *IntParamArr,*HKLInts,*n_arr;
	IntParamArr = &(f_data->IntParamArr[0]);
	HKLInts = &(f_data->HKLInts[0]);
	n_arr = &(f_data->n_arr[0]);
	hklscorr = &(f_data->hklspace[0]);
	CorrectHKLsLatCInd(x+6,hkls,n_arr,RTParamArr,hklscorr,HKLInts);
	RealType OrientMatrix[3][3];
	Euler2OrientMat(x+3,OrientMatrix);
	RealType DisplY, DisplZ, Y, Z, Ome;
	int spnr;
	RealType Error = 0;
	int nTspots = CalcDiffrSpots(OrientMatrix,RTParamArr+5,OmeBoxArr,IntParamArr[1],
			RTParamArr[5+MAX_N_RINGS+6],TheorSpots,hklscorr,n_arr);
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		DisplacementInTheSpot(x[0],x[1],x[2],RTParamArr[0],spotsYZO[nrSp*9+5],
			spotsYZO[nrSp*9+6],spotsYZO[nrSp*9+4],RTParamArr[20+MAX_N_RINGS]
			,0,&DisplY,&DisplZ);
		if (fabs(RTParamArr[20+MAX_N_RINGS]) > 0.02){
			CorrectForOme(spotsYZO[nrSp*9+5]-DisplY,
				spotsYZO[nrSp*9+6]-DisplZ,RTParamArr[0],
				spotsYZO[nrSp*9+4],RTParamArr[19+MAX_N_RINGS],
				RTParamArr[20+MAX_N_RINGS],&Y, &Z, &Ome);
		}else{
			Y = spotsYZO[nrSp*9+5]-DisplY;
			Z = spotsYZO[nrSp*9+6]-DisplZ;
			Ome = spotsYZO[nrSp*9+4];
		}
		spnr = (int) spotsYZO[nrSp*9+8];
		for (int j=0;j<nTspots;j++){
			if ((int)TheorSpots[j*8+7] == spnr){
				Error += CalcNorm2(Y-TheorSpots[j*8+0],Z-TheorSpots[j*8+1]);
				break;
			}
		}
	}
	return Error;
}

struct func_data_orient{
	int *IntParamArr;
	RealType *OmeBoxArr;
	RealType *spotsCorrected;
	RealType *hkls;
	int *HKLInts;
	int nMatched;
	RealType *RTParamArr;
	int *n_arr;
	RealType *TheorSpots;
	RealType *hklspace;
};

__device__ RealType pf_orient(int n, RealType *x, void *f_data_trial){
	struct func_data_orient *f_data = (struct func_data_orient *) f_data_trial;
	RealType *TheorSpots, *spotsYZO, *RTParamArr, *OmeBoxArr, *hkls,
		*hklscorr, *SpotsCorrected;
	OmeBoxArr = &(f_data->OmeBoxArr[0]);
	spotsYZO = &(f_data->spotsCorrected[0]);
	hkls = &(f_data->hkls[0]);
	RTParamArr = &(f_data->RTParamArr[0]);
	TheorSpots = &(f_data->TheorSpots[0]);
	int nMatched = f_data->nMatched;
	int *IntParamArr,*HKLInts,*n_arr;
	IntParamArr = &(f_data->IntParamArr[0]);
	HKLInts = &(f_data->HKLInts[0]);
	n_arr = &(f_data->n_arr[0]);
	hklscorr = &(f_data->hklspace[0]);
	CorrectHKLsLatCInd(x+3,hkls,n_arr,RTParamArr,hklscorr,HKLInts);
	RealType OrientMatrix[3][3];
	Euler2OrientMat(x,OrientMatrix);
	RealType *gObs, *gTh;
	int spnr;
	RealType Error = 0;
	RealType tmpL;
	int nTspots = CalcDiffrSpots(OrientMatrix,RTParamArr+5,OmeBoxArr,IntParamArr[1],
			RTParamArr[5+MAX_N_RINGS+6],TheorSpots,hklscorr,n_arr);
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		gObs = spotsYZO + nrSp*6 + 2;
		spnr = (int) spotsYZO[nrSp*6+5];
		for (int j=0;j<nTspots;j++){
			if ((int)TheorSpots[j*8+7] == spnr){
				gTh = TheorSpots + j*8 + 3;
				tmpL = ((dot(gObs,gTh))/(CalcNorm3(gObs[0],gObs[1],gObs[2])*CalcNorm3(gTh[0],gTh[1],gTh[2])));
				if (tmpL > 1) tmpL = 1;
				if (tmpL < -1) tmpL = -1;
				Error += fabs(acosd(tmpL));
				break;
			}
		}
	}
	return Error;
}

struct func_data_strains{
	int *IntParamArr;
	RealType *OmeBoxArr;
	RealType *spotsCorrected;
	RealType *hkls;
	int *HKLInts;
	int nMatched;
	RealType *RTParamArr;
	int *n_arr;
	RealType *Euler;
	RealType *TheorSpots;
	RealType *hklspace;
};

__device__ RealType pf_strains(int n, RealType *x, void *f_data_trial){
	struct func_data_strains *f_data = (struct func_data_strains *) f_data_trial;
	RealType *TheorSpots, *spotsYZO, *RTParamArr, *OmeBoxArr, *hkls,
		*hklscorr, *SpotsCorrected, *Euler;
	OmeBoxArr = &(f_data->OmeBoxArr[0]);
	spotsYZO = &(f_data->spotsCorrected[0]);
	hkls = &(f_data->hkls[0]);
	RTParamArr = &(f_data->RTParamArr[0]);
	TheorSpots = &(f_data->TheorSpots[0]);
	int nMatched = f_data->nMatched;
	int *IntParamArr,*HKLInts,*n_arr;
	IntParamArr = &(f_data->IntParamArr[0]);
	HKLInts = &(f_data->HKLInts[0]);
	n_arr = &(f_data->n_arr[0]);
	hklscorr = &(f_data->hklspace[0]);
	CorrectHKLsLatCInd(x,hkls,n_arr,RTParamArr,hklscorr,HKLInts);
	RealType OrientMatrix[3][3];
	Euler = &(f_data->Euler[0]);
	Euler2OrientMat(Euler,OrientMatrix);
	RealType Y,Z;
	int spnr;
	RealType Error = 0;
	int nTspots = CalcDiffrSpotsStrained(OrientMatrix,OmeBoxArr,IntParamArr[1],
			RTParamArr[5+MAX_N_RINGS+6],TheorSpots,hklscorr,n_arr);
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		Y = spotsYZO[nrSp*6+0];
		Z = spotsYZO[nrSp*6+1];
		spnr = (int) spotsYZO[nrSp*6+5];
		for (int j=0;j<nTspots;j++){
			if ((int)TheorSpots[j*8+7] == spnr){
				Error += CalcNorm2(Y-TheorSpots[j*8+0],Z-TheorSpots[j*8+1]);
				break;
			}
		}
	}
	return Error;
}

struct func_data_pos_sec{
	int nMatched;
	RealType *TheorSpots;
	int nTspots;
	RealType *spotsYZO;
	RealType *RTParamArr;
};

__device__ RealType pf_posSec(int n, RealType *x, void *f_data_trial){
	struct func_data_pos_sec *f_data = (struct func_data_pos_sec *) f_data_trial;
	RealType *TheorSpots, *spotsYZO, *RTParamArr;
	spotsYZO = &(f_data->spotsYZO[0]);
	TheorSpots = &(f_data->TheorSpots[0]);
	RTParamArr = &(f_data->RTParamArr[0]);
	int nMatched = f_data->nMatched;
	int nTspots = f_data->nTspots;
	int spnr;
	int Error = 0;
	RealType DisplY, DisplZ, Y, Z, Ome;
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		DisplacementInTheSpot(x[0],x[1],x[2],RTParamArr[0],spotsYZO[nrSp*9+5],
			spotsYZO[nrSp*9+6],spotsYZO[nrSp*9+4],RTParamArr[20+MAX_N_RINGS]
			,0,&DisplY,&DisplZ);
		if (fabs(RTParamArr[20+MAX_N_RINGS]) > 0.02){
			CorrectForOme(spotsYZO[nrSp*9+5]-DisplY,
				spotsYZO[nrSp*9+6]-DisplZ,RTParamArr[0],
				spotsYZO[nrSp*9+4],RTParamArr[19+MAX_N_RINGS],
				RTParamArr[20+MAX_N_RINGS],&Y, &Z, &Ome);
		}else{
			Y = spotsYZO[nrSp*9+5]-DisplY;
			Z = spotsYZO[nrSp*9+6]-DisplZ;
		}
		spnr = (int) spotsYZO[nrSp*9+8];
		for (int j=0;j<nTspots;j++){
			if ((int)TheorSpots[j*8+7] == spnr){
				Error += CalcNorm2(Y-TheorSpots[j*8+0],Z-TheorSpots[j*8+1]);
				break;
			}
		}
	}
	return Error;
}

__global__ void FitGrain(RealType *RTParamArr, int *IntParamArr,
	int *n_arr, RealType *OmeBoxArr, RealType *hklsIn, int *HKLints,
	int *nMatchedArr, RealType *spotsYZO_d, RealType *FitParams_d,
	RealType *TheorSpots_d, RealType *scratch_d, RealType *hklspace_d,
	RealType *x_d, RealType *xl_d, RealType *xu_d, RealType *xout_d,
	RealType *xstep_d, RealType *CorrectSpots,RealType *Result_d){
	int spotNr = blockIdx.x * blockDim.x + threadIdx.x;
	if (spotNr >= n_arr[2]){
		return;
	}
	RealType *spotsYZO, *FitParams, *TheorSpots, *scratch, *hklspace, *x,
		*xl, *xu, *xout, *xstep, *spotsCorrected,
		*Result;
	int nMatched, nMatchedTillNowRowNr, i;
	nMatched = nMatchedArr[spotNr*3+0];
	nMatchedTillNowRowNr = nMatchedArr[spotNr*3+2];
	spotsYZO = spotsYZO_d + nMatchedTillNowRowNr * 9;
	FitParams = FitParams_d + spotNr * 12;
	Result = Result_d + spotNr *12;
	TheorSpots = TheorSpots_d + n_arr[1]*2*spotNr*8;
	scratch = scratch_d + spotNr*((12+1)*(12+1)+3*12);
	hklspace = hklspace_d + spotNr*n_arr[1]*7;
	spotsCorrected = CorrectSpots + nMatchedTillNowRowNr*6;
	x = x_d + 12*spotNr;
	xl = xl_d + 12*spotNr;
	xu = xu_d + 12*spotNr;
	xout = xout_d + 12*spotNr;
	xstep = xstep_d + 12*spotNr;
	int n = 12;
	for (i=0;i<12;i++){
		x[i] = FitParams[i];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i] - RTParamArr[1];
		xl[i+3] = x[i+3] - 0.01;
		xl[i+6] = x[i+6]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+9] = x[i+9]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i] + RTParamArr[1];
		xu[i+3] = x[i+3] + 0.01;
		xu[i+6] = x[i+6]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+9] = x[i+9]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_pos_ini f_data;
	f_data.HKLInts = HKLints;
	f_data.IntParamArr = IntParamArr;
	f_data.OmeBoxArr = OmeBoxArr;
	f_data.RTParamArr = RTParamArr;
	f_data.hkls = hklsIn;
	f_data.nMatched = nMatched;
	f_data.n_arr = n_arr;
	f_data.spotsYZO = spotsYZO;
	f_data.TheorSpots = TheorSpots;
	f_data.hklspace = hklspace;
	struct func_data_pos_ini *f_datat;
	f_datat = &f_data;
	void *trp = (struct func_data_pos_ini *)  f_datat;
	RealType minf;
	RealType reqmin = 1e-8;
	int konvge = 10;
	int kcount = MAX_N_EVALS;
	int icount, numres, ifault;
	//if (spotNr == 0) printf("Pos in: %lf %lf %lf %lf\n",pf_posIni(n,x,trp),x[0],x[1],x[2]);
	nelmin(pf_posIni, n, x, xout, xl, xu, scratch, &minf, reqmin, xstep, konvge, kcount/4, &icount, &numres, &ifault, trp);
	//if (spotNr == 0) printf("Pos out: %lf %lf %lf %lf\n",pf_posIni(n,xout,trp),xout[0],xout[1],xout[2]);
	//if (ifault !=0) printf("Not optimized completely.\n");
	RealType Pos[3] = {xout[0],xout[1],xout[2]};
	RealType DisplY, DisplZ, Y, Z, Ome, g[3], Theta, lenK;
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		DisplacementInTheSpot(xout[0],xout[1],xout[2],RTParamArr[0],spotsYZO[nrSp*9+5],
			spotsYZO[nrSp*9+6],spotsYZO[nrSp*9+4],RTParamArr[20+MAX_N_RINGS],
			0,&DisplY,&DisplZ);
		if (fabs(RTParamArr[20+MAX_N_RINGS]) > 0.02){
			CorrectForOme(spotsYZO[nrSp*9+5]-DisplY,
				spotsYZO[nrSp*9+6]-DisplZ,RTParamArr[0],
				spotsYZO[nrSp*9+4],RTParamArr[19+MAX_N_RINGS],
				RTParamArr[20+MAX_N_RINGS],&Y, &Z, &Ome);
		}else{
			Y = spotsYZO[nrSp*9+5]-DisplY;
			Z = spotsYZO[nrSp*9+6]-DisplZ;
			Ome = spotsYZO[nrSp*9+4];
		}
		Theta = atand(CalcNorm2(Y,Z)/RTParamArr[0])/2;
		lenK = CalcNorm3(RTParamArr[0],Y,Z);
		SpotToGv(RTParamArr[0]/lenK,Y/lenK,Z/lenK,Ome,Theta,&spotsCorrected[nrSp*6+2],
			&spotsCorrected[nrSp*6+3],&spotsCorrected[nrSp*6+4]);
		spotsCorrected[nrSp*6+0] = Y;
		spotsCorrected[nrSp*6+1] = Z;
		spotsCorrected[nrSp*6+5] = spotsYZO[nrSp*9+8];
	}
	n = 9;
	for (i=0;i<9;i++){
		x[i] = FitParams[i+3];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i] - 2;
		xl[i+3] = x[i+3]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+6] = x[i+6]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i] + 2;
		xu[i+3] = x[i+3]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+6] = x[i+6]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_orient f_data2;
	f_data2.HKLInts = HKLints;
	f_data2.IntParamArr = IntParamArr;
	f_data2.OmeBoxArr = OmeBoxArr;
	f_data2.RTParamArr = RTParamArr;
	f_data2.hkls = hklsIn;
	f_data2.nMatched = nMatched;
	f_data2.n_arr = n_arr;
	f_data2.spotsCorrected = spotsCorrected;
	f_data2.TheorSpots = TheorSpots;
	f_data2.hklspace = hklspace;
	struct func_data_orient *f_datat2;
	f_datat2 = &f_data2;
	void *trp2 = (struct func_data_orient *)  f_datat2;
	//if (spotNr == 0) printf("Orient in: %lf %lf %lf %lf\n",pf_orient(n,x,trp2),x[0],x[1],x[2]);
	nelmin(pf_orient, n, x, xout, xl, xu, scratch, &minf, reqmin, xstep, konvge, kcount/3, &icount, &numres, &ifault, trp2);
	//if (spotNr == 0) printf("Orient out: %lf %lf %lf %lf\n",pf_orient(n,xout,trp2),xout[0],xout[1],xout[2]);
    //if (ifault !=0) printf("Not optimized completely.\n");
    RealType Euler[3] = {xout[0],xout[1],xout[2]};
    n = 6;
    for (i=0;i<n;i++){
		x[i] = FitParams[i+6];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+3] = x[i+3]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+3] = x[i+3]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_strains f_data3;
	f_data3.Euler = Euler;
	f_data3.HKLInts = HKLints;
	f_data3.IntParamArr = IntParamArr;
	f_data3.OmeBoxArr = OmeBoxArr;
	f_data3.RTParamArr = RTParamArr;
	f_data3.hkls = hklsIn;
	f_data3.nMatched = nMatched;
	f_data3.n_arr = n_arr;
	f_data3.spotsCorrected = spotsCorrected;
	f_data3.TheorSpots = TheorSpots;
	f_data3.hklspace = hklspace;
	struct func_data_strains *f_datat3;
	f_datat3 = &f_data3;
	void *trp3 = (struct func_data_strains *)  f_datat3;
	//if (spotNr == 0) printf("Strains in: %lf %lf %lf %lf %lf %lf %lf\n",pf_strains(n,x,trp3),x[0],x[1],x[2],x[3],x[4],x[5]);
	nelmin(pf_strains, n, x, xout, xl, xu, scratch, &minf, reqmin, xstep, konvge, kcount/2, &icount, &numres, &ifault, trp3);
	//if (spotNr == 0) printf("Strains out: %lf %lf %lf %lf %lf %lf %lf\n",pf_strains(n,xout,trp3),xout[0],xout[1],xout[2],xout[3],xout[4],xout[5]);
	//if (ifault !=0) printf("Not optimized completely.\n");
    RealType LatCFit[6] = {xout[0],xout[1],xout[2],xout[3],xout[4],xout[5]};
    n = 3;
    RealType OM[3][3];
    Euler2OrientMat(Euler,OM);
    CorrectHKLsLatCInd(LatCFit,hklsIn,n_arr,RTParamArr,hklspace,HKLints);
    int nTspots = CalcDiffrSpotsStrained(OM,OmeBoxArr,IntParamArr[1],
		RTParamArr[5+MAX_N_RINGS+6],TheorSpots,hklspace,n_arr);
	for (int i=0;i<3;i++){
		x[i] = Pos[i];
		xl[i] = x[i] - RTParamArr[1];
		xu[i] = x[i] + RTParamArr[1];
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_pos_sec f_data4;
	f_data4.RTParamArr = RTParamArr;
	f_data4.nMatched = nMatched;
	f_data4.TheorSpots = TheorSpots;
	f_data4.nTspots = nTspots;
	f_data4.spotsYZO = spotsYZO;
	struct func_data_pos_sec *f_datat4;
	f_datat4 = &f_data4;
	void *trp4 = (struct func_data_pos_sec *)  f_datat4;
	//if (spotNr == 0) printf("Pos2 in: %lf %lf %lf %lf\n",pf_posSec(n,x,trp4),x[0],x[1],x[2]);
	nelmin(pf_posSec, n, x, xout, xl, xu, scratch, &minf, reqmin, xstep, konvge, kcount, &icount, &numres, &ifault, trp4);
	//if (spotNr == 0) printf("Pos2 out: %lf %lf %lf %lf\n",pf_posSec(n,xout,trp4),xout[0],xout[1],xout[2]);
    //if (ifault !=0) printf("Not optimized completely.\n");
    RealType Pos2[3] = {xout[0],xout[1],xout[2]};
    for (i=0;i<3;i++){
		Result[i] = Pos2[i];
		Result[i+3] = Euler[i];
		Result[i+6] = LatCFit[i];
		Result[i+9] = LatCFit[i+3];
	}
}

__global__ void FitGrain_NLOPT(RealType *RTParamArr, int *IntParamArr,
	int *n_arr, RealType *OmeBoxArr, RealType *hklsIn, int *HKLints,
	int *nMatchedArr, RealType *spotsYZO_d, RealType *FitParams_d,
	RealType *TheorSpots_d, RealType *scratch_d, RealType *hklspace_d,
	RealType *x_d, RealType *xl_d, RealType *xu_d, RealType *xout_d,
	RealType *xstep_d, RealType *CorrectSpots, RealType *TheorSpotsCorr,
	RealType *Result_d){
	int spotNr = blockIdx.x * blockDim.x + threadIdx.x;
	if (spotNr >= n_arr[2]){
		return;
	}
	RealType *spotsYZO, *FitParams, *TheorSpots, *scratch, *hklspace, *x,
		*xl, *xu, *xout, *xstep, *spotsCorrected, *TheorSpotsCorrected,
		*Result;
	int nMatched, nMatchedTillNowRowNr, i;
	nMatched = nMatchedArr[spotNr*3+0];
	nMatchedTillNowRowNr = nMatchedArr[spotNr*3+2];
	spotsYZO = spotsYZO_d + nMatchedTillNowRowNr * 9;
	FitParams = FitParams_d + spotNr * 12;
	Result = Result_d + spotNr *12;
	TheorSpots = TheorSpots_d + n_arr[1]*2*spotNr*8;
	TheorSpotsCorrected = TheorSpotsCorr + n_arr[1]*2*spotNr*8;
	scratch = scratch_d + spotNr*((12+1)*(12+1)+3*12);
	hklspace = hklspace_d + spotNr*n_arr[1]*7;
	spotsCorrected = CorrectSpots + nMatchedTillNowRowNr*6;
	x = x_d + 12*spotNr;
	xl = xl_d + 12*spotNr;
	xu = xu_d + 12*spotNr;
	xout = xout_d + 12*spotNr;
	xstep = xstep_d + 12*spotNr;
	int n = 12;
	for (i=0;i<12;i++){
		x[i] = FitParams[i];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i] - RTParamArr[1];
		xl[i+3] = x[i+3] - 0.01;
		xl[i+6] = x[i+6]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+9] = x[i+9]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i] + RTParamArr[1];
		xu[i+3] = x[i+3] + 0.01;
		xu[i+6] = x[i+6]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+9] = x[i+9]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_pos_ini f_data;
	f_data.HKLInts = HKLints;
	f_data.IntParamArr = IntParamArr;
	f_data.OmeBoxArr = OmeBoxArr;
	f_data.RTParamArr = RTParamArr;
	f_data.hkls = hklsIn;
	f_data.nMatched = nMatched;
	f_data.n_arr = n_arr;
	f_data.spotsYZO = spotsYZO;
	f_data.TheorSpots = TheorSpots;
	f_data.hklspace = hklspace;
	struct func_data_pos_ini *f_datat;
	f_datat = &f_data;
	void *trp = (struct func_data_pos_ini *)  f_datat;
	RealType minf;
	RealType reqmin = 1e-8;
	int konvge = 10;
	int kcount = MAX_N_EVALS;
	int icount, numres, ifault;
	nlopt_stopping stop;
	stop.n = n;
	stop.maxeval = MAX_N_EVALS;
	stop.ftol_rel = reqmin;
	stop.xtol_rel = reqmin;
	stop.minf_max = reqmin;
	nlopt_func f = &pf_posIni;
	nlopt_result res = NLOPT_SUCCESS;
	if (spotNr == 0) printf("%lf\n",pf_posIni(n,x,trp));
	res = nldrmd_minimize(n,f,trp,xl,xu,x,&minf,xstep,&stop,scratch);
	if (spotNr == 0) printf("%lf\n",pf_posIni(n,x,trp));
	for (i=0;i<n;i++) xout[i] = x[i];
	if (res !=1) printf("Not optimized completely. %d, %lf\n",res,minf);
	RealType Pos[3] = {xout[0],xout[1],xout[2]};
	RealType DisplY, DisplZ, Y, Z, Ome, g[3], Theta, lenK;
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		DisplacementInTheSpot(xout[0],xout[1],xout[2],RTParamArr[0],spotsYZO[nrSp*9+5],
			spotsYZO[nrSp*9+6],spotsYZO[nrSp*9+4],RTParamArr[20+MAX_N_RINGS],
			0,&DisplY,&DisplZ);
		if (fabs(RTParamArr[20+MAX_N_RINGS]) > 0.02){
			CorrectForOme(spotsYZO[nrSp*9+5]-DisplY,
				spotsYZO[nrSp*9+6]-DisplZ,RTParamArr[0],
				spotsYZO[nrSp*9+4],RTParamArr[19+MAX_N_RINGS],
				RTParamArr[20+MAX_N_RINGS],&Y, &Z, &Ome);
		}else{
			Y = spotsYZO[nrSp*9+5]-DisplY;
			Z = spotsYZO[nrSp*9+6]-DisplZ;
			Ome = spotsYZO[nrSp*9+4];
		}
		Theta = atand(CalcNorm2(Y,Z)/RTParamArr[0])/2;
		lenK = CalcNorm3(RTParamArr[0],Y,Z);
		SpotToGv(RTParamArr[0]/lenK,Y/lenK,Z/lenK,Ome,Theta,&spotsCorrected[nrSp*6+2],
			&spotsCorrected[nrSp*6+3],&spotsCorrected[nrSp*6+4]);
		spotsCorrected[nrSp*6+0] = Y;
		spotsCorrected[nrSp*6+1] = Z;
		spotsCorrected[nrSp*6+5] = spotsYZO[nrSp*9+8];
	}
	n = 9;
	for (i=0;i<9;i++){
		x[i] = FitParams[i+3];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i] - 2;
		xl[i+3] = x[i+3]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+6] = x[i+6]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i] + 2;
		xu[i+3] = x[i+3]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+6] = x[i+6]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_orient f_data2;
	f_data2.HKLInts = HKLints;
	f_data2.IntParamArr = IntParamArr;
	f_data2.OmeBoxArr = OmeBoxArr;
	f_data2.RTParamArr = RTParamArr;
	f_data2.hkls = hklsIn;
	f_data2.nMatched = nMatched;
	f_data2.n_arr = n_arr;
	f_data2.spotsCorrected = spotsCorrected;
	f_data2.TheorSpots = TheorSpots;
	f_data2.hklspace = hklspace;
	struct func_data_orient *f_datat2;
	f_datat2 = &f_data2;
	void *trp2 = (struct func_data_orient *)  f_datat2;
    stop.n = n;
	f = &pf_orient;
	res = nldrmd_minimize(n,f,trp2,xl,xu,x,&minf,xstep,&stop,scratch);
	for (i=0;i<n;i++) xout[i] = x[i];
	if (res !=1) printf("Not optimized completely. %d, %lf\n",res,minf);
    RealType Euler[3] = {xout[0],xout[1],xout[2]};
    n = 6;
    for (i=0;i<n;i++){
		x[i] = FitParams[i+6];
	}
	for (i=0;i<3;i++){
		xl[i] = x[i]*(1 - RTParamArr[21+MAX_N_RINGS]/100);
		xl[i+3] = x[i+3]*(1 - RTParamArr[22+MAX_N_RINGS]/100);
		xu[i] = x[i]*(1 + RTParamArr[21+MAX_N_RINGS]/100);
		xu[i+3] = x[i+3]*(1 + RTParamArr[22+MAX_N_RINGS]/100);
	}
	for (i=0;i<n;i++){
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_strains f_data3;
	f_data3.Euler = Euler;
	f_data3.HKLInts = HKLints;
	f_data3.IntParamArr = IntParamArr;
	f_data3.OmeBoxArr = OmeBoxArr;
	f_data3.RTParamArr = RTParamArr;
	f_data3.hkls = hklsIn;
	f_data3.nMatched = nMatched;
	f_data3.n_arr = n_arr;
	f_data3.spotsCorrected = spotsCorrected;
	f_data3.TheorSpots = TheorSpots;
	f_data3.hklspace = hklspace;
	struct func_data_strains *f_datat3;
	f_datat3 = &f_data3;
	void *trp3 = (struct func_data_strains *)  f_datat3;
    stop.n = n;
	f = &pf_strains;
	res = nldrmd_minimize(n,f,trp3,xl,xu,x,&minf,xstep,&stop,scratch);
	for (i=0;i<n;i++) xout[i] = x[i];
	if (res !=1) printf("Not optimized completely. %d, %lf\n",res,minf);
    RealType LatCFit[6] = {xout[0],xout[1],xout[2],xout[3],xout[4],xout[5]};
    n = 3;
    RealType OM[3][3];
    Euler2OrientMat(Euler,OM);
    CorrectHKLsLatCInd(LatCFit,hklsIn,n_arr,RTParamArr,hklspace,HKLints);
    int nTspots = CalcDiffrSpots(OM,RTParamArr+5,OmeBoxArr,IntParamArr[1],
		RTParamArr[5+MAX_N_RINGS+6],TheorSpotsCorrected,hklspace,n_arr);
	for (int i=0;i<3;i++){
		x[i] = Pos[i];
		xl[i] = x[i] - RTParamArr[1];
		xu[i] = x[i] + RTParamArr[1];
		xstep[i] = fabs(xu[i]-xl[i])*0.25;
	}
	struct func_data_pos_sec f_data4;
	f_data4.RTParamArr = RTParamArr;
	f_data4.nMatched = nMatched;
	f_data4.TheorSpots = TheorSpots;
	f_data4.nTspots = nTspots;
	f_data4.spotsYZO = spotsYZO;
	struct func_data_pos_sec *f_datat4;
	f_datat4 = &f_data4;
	void *trp4 = (struct func_data_pos_sec *)  f_datat4;
    stop.n = n;
	f = &pf_posSec;
	res = nldrmd_minimize(n,f,trp4,xl,xu,x,&minf,xstep,&stop,scratch);
	for (i=0;i<n;i++) xout[i] = x[i];
	if (res !=1) printf("Not optimized completely. %d, %lf\n",res,minf);
    RealType Pos2[3] = {xout[0],xout[1],xout[2]};
    for (i=0;i<3;i++){
		Result[i] = Pos2[i];
		Result[i+3] = Euler[i];
		Result[i+6] = LatCFit[i];
		Result[i+9] = LatCFit[i+3];
	}
}

__global__ void CalcAngleErrors(RealType *RTParamArr, int *IntParamArr,
	int *n_arr, RealType *OmeBoxArr, RealType *hkls_c, int *nMatchedArr,
	RealType *spotsYZO_d, RealType *x_d, RealType *TheorSpots_d,
	RealType *SpotsComp_d, RealType *Error_d, RealType *hklsIn, int *HKLints)
{
	int spotNr = blockIdx.x * blockDim.x + threadIdx.x;
	if (spotNr >= n_arr[2]){
		return;
	}
	RealType *hkls, *spotsYZO, *x, *TheorSpots;
	RealType *SpotsComp, *Error;
	int nMatched, nspots, nMatchedTillNowRowNr;
	hkls = hkls_c + spotNr*n_arr[1]*7;
	RealType *LatC_d;
	LatC_d = x_d + spotNr*12 + 6;
	CorrectHKLsLatCInd(LatC_d, hklsIn, n_arr, RTParamArr, hkls, HKLints);
	nMatched = nMatchedArr[spotNr*3+0];
	nMatchedTillNowRowNr = nMatchedArr[spotNr*3+2];
	spotsYZO = spotsYZO_d + nMatchedTillNowRowNr*9;
	x = x_d + spotNr*12;
	SpotsComp = SpotsComp_d + nMatchedTillNowRowNr*22;
	Error = Error_d + spotNr*3;
	Error[0] = 0; Error[1] = 0; Error[2] = 0;
	TheorSpots = TheorSpots_d + n_arr[1]*2*spotNr*8;
	RealType OrientationMatrix[3][3];
	Euler2OrientMat(x+3,OrientationMatrix);
	int nTspots = CalcDiffrSpotsStrained(OrientationMatrix,OmeBoxArr,IntParamArr[1],
			RTParamArr[5+MAX_N_RINGS+6],TheorSpots,hkls,n_arr);
	RealType DisplY, DisplZ, Y, Z, Ome, Theta, lenK, go[3], *gth, angle, distt, omediff, tmpL;
	int spnr;
	for (int nrSp=0;nrSp<nMatched;nrSp++){
		DisplacementInTheSpot(x[0],x[1],x[2],RTParamArr[0],spotsYZO[nrSp*9+5],
			spotsYZO[nrSp*9+6],spotsYZO[nrSp*9+4],RTParamArr[20+MAX_N_RINGS],
			0,&DisplY,&DisplZ);
		if (fabs(RTParamArr[20+MAX_N_RINGS]) > 0.02){
			CorrectForOme(spotsYZO[nrSp*9+5]-DisplY,
				spotsYZO[nrSp*9+6]-DisplZ,RTParamArr[0],
				spotsYZO[nrSp*9+4],RTParamArr[19+MAX_N_RINGS],
				RTParamArr[20+MAX_N_RINGS],&Y,
				&Z,&Ome);
		}else{
			Y = spotsYZO[nrSp*9+5]-DisplY;
			Z = spotsYZO[nrSp*9+6]-DisplZ;
			Ome = spotsYZO[nrSp*9+4];
		}
		Theta = 0.5*atand(CalcNorm2(Y,Z)/RTParamArr[0]);
		lenK = CalcNorm3(RTParamArr[0],Y,Z);
		SpotToGv(RTParamArr[0]/lenK,Y/lenK,Z/lenK,Ome,Theta,&go[0],&go[1],&go[2]);
		spnr = (int) spotsYZO[nrSp*9+8];
		for (int i=0;i<nTspots;i++){
			if ((int)TheorSpots[i*8+7] == spnr){
				gth = TheorSpots + i*8 + 3;
				tmpL = ((dot(go,gth))/(CalcNorm3(go[0],go[1],go[2])*CalcNorm3(gth[0],gth[1],gth[2])));
				if (tmpL > 1) tmpL = 1;
				if (tmpL < -1) tmpL = -1;
				angle = fabs(acosd(tmpL));
				distt = CalcNorm2(Y-TheorSpots[i*8+0],Z-TheorSpots[i*8+1]);
				omediff = fabs(Ome - TheorSpots[i*8+2]);
				Error[0] += fabs(angle/nMatched);
				Error[1] += fabs(distt/nMatched);
				Error[2] += fabs(omediff/nMatched);
				SpotsComp[nrSp*22+0] = spotsYZO[nrSp*9+3];
				SpotsComp[nrSp*22+1] = Y;
				SpotsComp[nrSp*22+2] = Z;
				SpotsComp[nrSp*22+3] = Ome;
				SpotsComp[nrSp*22+4] = go[0];
				SpotsComp[nrSp*22+5] = go[1];
				SpotsComp[nrSp*22+6] = go[2];
				for (int j=0;j<6;j++){
					SpotsComp[nrSp*22+j+7] = TheorSpots[i*8+j];
				}
				SpotsComp[nrSp*22+13]=spotsYZO[nrSp*9+0];
				SpotsComp[nrSp*22+14]=spotsYZO[nrSp*9+1];
				SpotsComp[nrSp*22+15]=spotsYZO[nrSp*9+2];
				SpotsComp[nrSp*22+16]=spotsYZO[nrSp*9+4];
				SpotsComp[nrSp*22+17]=spotsYZO[nrSp*9+5];
				SpotsComp[nrSp*22+18]=spotsYZO[nrSp*9+6];
				SpotsComp[nrSp*22+19]=angle;
				SpotsComp[nrSp*22+20]=distt;
				SpotsComp[nrSp*22+21]=omediff;
				break;
			}
		}
	}
}

__global__ void CompareDiffractionSpots(RealType *AllTheorSpots, RealType *RTParamArr,
	int maxPos, RealType *ResultArr, int PosResultArr, int *nTspotsArr,
	int *data, int *ndata, RealType *ObsSpots, RealType *etamargins, int *AllGrainSpots,
	RealType *IAs, int *n_arr, int *nMatchedArr, int n_min, int nOrients, RealType *GS,
	RealType *AllSpotsYZO, RealType *SpotsInfo_d, RealType *Orientations, RealType *OrientationsOut){
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
	for (int i=0;i<9;i++){
		OrientationsOut[10*overallPos + i] = Orientations[9*orientPos + i];
	}
	OrientationsOut[10*overallPos + 9] = (RealType) nTspotsArr[orientPos];
	int *GrainSpots;
	GrainSpots = AllGrainSpots + overallPos * n_arr[1] * 2;
	RealType *SpotsInfo;
	SpotsInfo = SpotsInfo_d + overallPos * n_arr[1] * 2 * 9;
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
	long long unsigned Pos, Pos1, Pos2, Pos3;
	int nspots, DataPos;
	long long unsigned spotRow,spotRowBest;
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
	RealType theta, lenK, yobs, zobs, thy, thz, thEta, thrad;
	for (int sp = 0 ; sp < nTspots ; sp++) {
		ometh = TheorSpots[sp*N_COL_THEORSPOTS+2];
		t = (gs[0]*cos(deg2rad * ometh) - gs[1]*sin(deg2rad * ometh))/xi;
		Displ_y = ((gs[0]*sin(deg2rad * ometh))+ (gs[1]*cos(deg2rad * ometh))) - t* yi;
		Displ_z = gs[2] - t*zi;
		thy = TheorSpots[sp*N_COL_THEORSPOTS+0] +  Displ_y;
		thz = TheorSpots[sp*N_COL_THEORSPOTS+1] +  Displ_z;
		thEta = CalcEtaAngle(thy,thz);
		thrad = CalcNorm2(thy,thz) - RTParamArr[5 + (int)TheorSpots[sp*N_COL_THEORSPOTS+3]];
		MatchFound = 0;
		diffOmeBest = 100000;
		Pos1 = (((int) TheorSpots[sp*N_COL_THEORSPOTS+3])-1)*n_eta_bins*n_ome_bins;
		Pos2 = ((int)(floor((180+thEta)/RTParamArr[5 + MAX_N_RINGS + 4])))*n_ome_bins;
		Pos3 = ((int)floor((180+TheorSpots[sp*N_COL_THEORSPOTS+2])/RTParamArr[5 + MAX_N_RINGS + 5]));
		Pos = Pos1 + Pos2 + Pos3;
		nspots = *(ndata+ Pos*2);
		if (nspots == 0){
			continue;
		}
		DataPos = *(ndata + Pos*2+1);
		for (int iSpot = 0 ; iSpot < nspots; iSpot++ ) {
			spotRow = *(data+DataPos + iSpot);
			if ( fabs(thrad - ObsSpots[spotRow*9+8]) < RTParamArr[5 + MAX_N_RINGS + 3] )  {
				if ( fabs(RefRad - ObsSpots[spotRow*9+3]) < RTParamArr[5 + MAX_N_RINGS + 2] ) {
					if ( fabs(thEta - ObsSpots[spotRow*9+6]) < etamargins[(int) TheorSpots[sp*N_COL_THEORSPOTS+3]] ) {
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
			if ((int)AllSpotsYZO[spotRowBest*8+3] != (int)ObsSpots[spotRowBest*9+4]) return;
			for (int i=0;i<8;i++){
				SpotsInfo[nMatched * 9 + i] = AllSpotsYZO[spotRowBest * 8 + i];
			}
			SpotsInfo[nMatched * 9 + 8] = TheorSpots[sp*N_COL_THEORSPOTS+4];
			GrainSpots[nMatched] = (int) ObsSpots[spotRowBest*9+4];
			omeo = ObsSpots[spotRowBest*9+2];
			ometh = TheorSpots[sp*N_COL_THEORSPOTS+2];
			theta = atand(CalcNorm2(TheorSpots[sp*N_COL_THEORSPOTS+0],TheorSpots[sp*N_COL_THEORSPOTS+1])/RTParamArr[0])/2;
			lenK = CalcNorm3(RTParamArr[0],TheorSpots[sp*N_COL_THEORSPOTS+0],TheorSpots[sp*N_COL_THEORSPOTS+1]);
			SpotToGv(RTParamArr[0]/lenK,TheorSpots[sp*N_COL_THEORSPOTS+0]/lenK,TheorSpots[sp*N_COL_THEORSPOTS+1]/lenK,ometh,theta,&gvth[0],&gvth[1],&gvth[2]);
			t = (gs[0]*cos(deg2rad * omeo) - gs[1]*sin(deg2rad * omeo))/xi;
			Displ_y = ((gs[0]*sin(deg2rad * omeo))+ (gs[1]*cos(deg2rad * omeo))) - t* yi;
			Displ_z = gs[2] - t*zi;
			yobs = ObsSpots[spotRowBest*9+0]-Displ_y;
			zobs = ObsSpots[spotRowBest*9+1]-Displ_z;
			theta = atand(CalcNorm2(yobs,zobs)/RTParamArr[0])/2;
			lenK = CalcNorm3(RTParamArr[0],yobs,zobs);
			SpotToGv(RTParamArr[0]/lenK,yobs/lenK,zobs/lenK,omeo,theta,&gvo[0],&gvo[1],&gvo[2]);
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
	if (orient >= norients) return;
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
	RealType hkllen = sqrt(hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2]);
	RealType hklnormallen = sqrt(hklnormal[0]*hklnormal[0] + hklnormal[1]*hklnormal[1] + hklnormal[2]*hklnormal[2]);
	RealType dotpr = dot(hkl, hklnormal);
	RealType angled = rad2deg * acos(dotpr/(hkllen*hklnormallen));
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


__device__ int CalcAllPlanes(RealType ys, RealType zs,
	RealType ttheta, RealType eta, RealType omega, int ringno,
	RealType Ring_rad, RealType Rsample, RealType Hbeam, RealType *hkls, int *n_arr,
	RealType *RTParamArr, RealType *ResultArray, int rowID, RealType RefRad){
	int nPlanes=0;
	RealType hkl[3];
	for (int i=0;i<n_arr[1];i++){
		if ((int) hkls[i*7+3] == ringno){
			hkl[0] = hkls[i*7+0];
			hkl[1] = hkls[i*7+1];
			hkl[2] = hkls[i*7+2];
			break;
		}
	}
	int quadr_coeff2 = 0;
	RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0, y0_min_z0, y0_max = 0, y0_min = 0, z0_min = 0, z0_max = 0;
	RealType y01, z01, y02, z02, y_diff, z_diff, length;
	int nsteps;
	RealType step_size = RTParamArr[3];
	if (eta > 90)
		eta_Hbeam = 180 - eta;
	else if (eta < -90)
		eta_Hbeam = 180 - fabs(eta);
	else
		eta_Hbeam = 90 - fabs(eta);
	Hbeam = Hbeam + 2*(Rsample*tan(ttheta*deg2rad))*(sin(eta_Hbeam*deg2rad));
	RealType eta_pole = 1 + rad2deg*acos(1-(Hbeam/Ring_rad));
	RealType eta_equator = 1 + rad2deg*acos(1-(Rsample/Ring_rad));
	if ((eta >= eta_pole) && (eta <= (90-eta_equator)) ) { // % 1st quadrant
		quadr_coeff = 1;
		coeff_y0 = -1;
		coeff_z0 = 1;
	}else if ( (eta >=(90+eta_equator)) && (eta <= (180-eta_pole)) ) {//% 4th quadrant
		quadr_coeff = 2;
		coeff_y0 = -1;
		coeff_z0 = -1;
	}else if ( (eta >= (-90+eta_equator) ) && (eta <= -eta_pole) )   { // % 2nd quadrant
		quadr_coeff = 2;
		coeff_y0 = 1;
		coeff_z0 = 1;
	}  else if ( (eta >= (-180+eta_pole) ) && (eta <= (-90-eta_equator)) )  { // % 3rd quadrant
		quadr_coeff = 1;
		coeff_y0 = 1;
		coeff_z0 = -1;
	}else
		quadr_coeff = 0;
	RealType y0_max_Rsample = ys + Rsample;
	RealType y0_min_Rsample = ys - Rsample;
	RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
	RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
	if (quadr_coeff == 1) {
		y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
		y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
	}else if (quadr_coeff == 2) {
		y0_max_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_min_Hbeam * z0_min_Hbeam));
		y0_min_z0 = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_max_Hbeam * z0_max_Hbeam));
	}
	if (quadr_coeff > 0)  {
		y0_max = min(y0_max_Rsample, y0_max_z0);
		y0_min = max(y0_min_Rsample, y0_min_z0);
	}else {
		if ((eta > -eta_pole) && (eta < eta_pole ))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = 1;
		}else if (eta < (-180+eta_pole))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = -1;
		}else if (eta > (180-eta_pole))  {
			y0_max = y0_max_Rsample;
			y0_min = y0_min_Rsample;
			coeff_z0 = -1;
		}else if (( eta > (90-eta_equator)) && (eta < (90+eta_equator)) ) {
			quadr_coeff2 = 1;
			z0_max = z0_max_Hbeam;
			z0_min = z0_min_Hbeam;
			coeff_y0 = -1;
		}else if ((eta > (-90-eta_equator)) && (eta < (-90+eta_equator)) ) {
			quadr_coeff2 = 1;
			z0_max = z0_max_Hbeam;
			z0_min = z0_min_Hbeam;
			coeff_y0 = 1;
		}
	}
	if (quadr_coeff2 == 0 ) {
		y01 = y0_min;
		z01 = coeff_z0 * sqrt((Ring_rad * Ring_rad )-(y01 * y01));
		y02 = y0_max;
		z02 = coeff_z0 * sqrt((Ring_rad * Ring_rad )-(y02 * y02));
		y_diff = y01 - y02;
		z_diff = z01 - z02;
		length = sqrt(y_diff * y_diff + z_diff * z_diff);
		nsteps = ceil(length/step_size);
	}else {
		z01 = z0_min;
		y01 = coeff_y0 * sqrt((Ring_rad * Ring_rad )-((z01 * z01)));
		z02 = z0_max;
		y02 = coeff_y0 * sqrt((Ring_rad * Ring_rad )-((z02 * z02)));
		y_diff = y01 - y02;
		z_diff = z01 - z02;
		length = sqrt(y_diff * y_diff + z_diff * z_diff);
		nsteps = ceil(length/step_size);
	}
	if ((nsteps % 2) == 0 ) {
		nsteps = nsteps +1;
	}
	// Now we know nsteps, we know ys, zs, y0_min, z0_min, y0_max, z0_max
	// Calculate y0_vector and z0_vector are IdealY and IdealZ and these can be used to calc hklnormal
	RealType y0_vector, z0_vector, xi, yi, zi, hklnormal[3], lenK;
	if ( nsteps == 1 ) {
		if (quadr_coeff2 == 0) {
			y0_vector = (y0_max+y0_min)/2;
			z0_vector = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_vector * y0_vector));
		}else {
			z0_vector = (z0_max+z0_min)/2;
			y0_vector = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_vector * z0_vector));
		}
		lenK = CalcNorm3(RTParamArr[0],y0_vector,z0_vector);
		xi = RTParamArr[0]/lenK;
		yi = y0_vector/lenK;
		zi = z0_vector/lenK;
		hklnormal[0] = (-1 + xi) * cos(-omega*deg2rad) - yi * sin(-omega*deg2rad);
		hklnormal[1] = (-1 + xi) * sin(-omega*deg2rad) + yi * cos(-omega*deg2rad);
		hklnormal[2] = zi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 0]  = hkl[0];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 1]  = hkl[1];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 2]  = hkl[2];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 3]  = hklnormal[0];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 4]  = hklnormal[1];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 5]  = hklnormal[2];
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 6]  = (RealType) ringno;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 7]  = y0_vector;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 8]  = z0_vector;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 9]  = xi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 10] = yi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 11] = zi;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 12] = ys;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 13] = zs;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 14] = omega;
		ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 15] = RefRad;
		nPlanes++;
	}else {
		int i;
		RealType stepsizeY = (y0_max-y0_min)/(nsteps-1);
		RealType stepsizeZ = (z0_max-z0_min)/(nsteps-1);
		if (quadr_coeff2 == 0) {
			for (i=0 ; i < nsteps ; i++) {
				y0_vector = y0_min + i*stepsizeY;
				z0_vector = coeff_z0 * sqrt((Ring_rad * Ring_rad)-(y0_vector * y0_vector));
				lenK = CalcNorm3(RTParamArr[0],y0_vector,z0_vector);
				xi = RTParamArr[0]/lenK;
				yi = y0_vector/lenK;
				zi = z0_vector/lenK;
				hklnormal[0] = (-1 + xi) * cos(-omega*deg2rad) - yi * sin(-omega*deg2rad);
				hklnormal[1] = (-1 + xi) * sin(-omega*deg2rad) + yi * cos(-omega*deg2rad);
				hklnormal[2] = zi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 0]  = hkl[0];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 1]  = hkl[1];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 2]  = hkl[2];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 3]  = hklnormal[0];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 4]  = hklnormal[1];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 5]  = hklnormal[2];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 6]  = (RealType) ringno;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 7]  = y0_vector;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 8]  = z0_vector;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 9]  = xi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 10] = yi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 11] = zi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 12] = ys;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 13] = zs;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 14] = omega;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 15] = RefRad;
				nPlanes++;
			}
		}else {
			for (i=0 ; i < nsteps ; i++) {
				z0_vector = z0_min + i*stepsizeZ;
				y0_vector = coeff_y0 * sqrt((Ring_rad * Ring_rad)-(z0_vector * z0_vector));
				lenK = CalcNorm3(RTParamArr[0],y0_vector,z0_vector);
				xi = RTParamArr[0]/lenK;
				yi = y0_vector/lenK;
				zi = z0_vector/lenK;
				hklnormal[0] = (-1 + xi) * cos(-omega*deg2rad) - yi * sin(-omega*deg2rad);
				hklnormal[1] = (-1 + xi) * sin(-omega*deg2rad) + yi * cos(-omega*deg2rad);
				hklnormal[2] = zi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 0]  = hkl[0];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 1]  = hkl[1];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 2]  = hkl[2];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 3]  = hklnormal[0];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 4]  = hklnormal[1];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 5]  = hklnormal[2];
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 6]  = (RealType) ringno;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 7]  = y0_vector;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 8]  = z0_vector;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 9]  = xi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 10] = yi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 11] = zi;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 12] = ys;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 13] = zs;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 14] = omega;
				ResultArray[rowID * MAX_N_FRIEDEL_PAIRS * N_COLS_FRIEDEL_RESULTS + nPlanes * N_COLS_FRIEDEL_RESULTS + 15] = RefRad;
				nPlanes++;
			}
		}
	}
   return nPlanes;
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
		if (nPlaneNormals == 0){
			//nPlaneNormals = TryFriedelMixed();
			//nNormals[rowID] = nPlaneNormals;
			if (nPlaneNormals != 0){
				return;
			}else{
				nPlaneNormals = CalcAllPlanes(ObsSpotsLab[SpotRowNo*9+0],
					ObsSpotsLab[SpotRowNo*9+1],	ObsSpotsLab[SpotRowNo*9+7],
					ObsSpotsLab[SpotRowNo*9+6], ObsSpotsLab[SpotRowNo*9+2],
					(int) ObsSpotsLab[SpotRowNo*9+5],
					RTParamArr[(int) ObsSpotsLab[SpotRowNo*9+5] + 5],
					RTParamArr[1], RTParamArr[2], hkls, n_arr, RTParamArr,
					ResultArray,rowID,RefRad);
				nNormals[rowID] = nPlaneNormals;
				return;
			}
		}else{
			return;
		}
	}
	nPlaneNormals = CalcAllPlanes(ObsSpotsLab[SpotRowNo*9+0],
		ObsSpotsLab[SpotRowNo*9+1],	ObsSpotsLab[SpotRowNo*9+7],
		ObsSpotsLab[SpotRowNo*9+6], ObsSpotsLab[SpotRowNo*9+2],
		(int) ObsSpotsLab[SpotRowNo*9+5],
		RTParamArr[(int) ObsSpotsLab[SpotRowNo*9+5] + 5],
		RTParamArr[1], RTParamArr[2], hkls, n_arr, RTParamArr,
		ResultArray,rowID,RefRad);
	nNormals[rowID] = nPlaneNormals;
	return;
}

static inline RealType sin_cos_to_angle (RealType s, RealType c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}
static inline void OrientMat2Euler(RealType m[3][3],RealType Euler[3])
{
    RealType psi, phi, theta, sph;
	if (fabs(m[2][2] - 1.0) < EPS){
		phi = 0;
	}else{
	    phi = acos(m[2][2]);
	}
    sph = sin(phi);
    if (fabs(sph) < EPS)
    {
        psi = 0.0;
        theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0]) : sin_cos_to_angle(-m[1][0], m[0][0]);
    } else{
        psi = (fabs(-m[1][2] / sph) <= 1.0) ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph) : sin_cos_to_angle(m[0][2] / sph,1);
        theta = (fabs(m[2][1] / sph) <= 1.0) ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph) : sin_cos_to_angle(m[2][0] / sph,1);
    }
    Euler[0] = rad2deg*psi;
    Euler[1] = rad2deg*phi;
    Euler[2] = rad2deg*theta;
}

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     default:
      printf("Unknown device type\n");
      break;
      }
    return cores;
}

int main(int argc, char *argv[]){
	printf("\n\n\t\t\tGPU Indexer v1.0\nContact hsharma@anl.gov in case of questions about the MIDAS project.\n\n");
	int cudaDeviceNum = atoi(argv[2]);
	cudaSetDevice(cudaDeviceNum);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int nCores = getSPcores(deviceProp);
	printf("Cuda Cores: %d\n",nCores);

	RealType iStart = cpuSecond();
    cudaGetDeviceProperties(&deviceProp,0);
    size_t gpuGlobalMem = deviceProp.totalGlobalMem;
    fprintf(stderr, "GPU global memory = %zu MBytes\n", gpuGlobalMem/(1024*1024));
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));

	char folder[MAX_LINE_LENGTH];
	struct ParametersStruct Parameters;
	char ParamFN[MAX_LINE_LENGTH];
	getcwd(folder,sizeof(folder));
	sprintf(ParamFN,"%s/%s",folder,argv[1]);
	printf("Reading parameters from file: %s.\n", ParamFN);
	int returncode = ReadParams(ParamFN, &Parameters);

	int *SpotIDs_h;
	SpotIDs_h = (int *) malloc(sizeof(*SpotIDs_h)* MAX_N_SPOTS);
	char spotIDsfn[MAX_LINE_LENGTH];
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
	char aline[MAX_LINE_LENGTH],dummy[MAX_LINE_LENGTH];
	fgets(aline,MAX_LINE_LENGTH,hklf);
	int Rnr,i;
	int hi,ki,li;
	RealType hc,kc,lc,RRd,Ds,tht;
	int n_hkls_h = 0;
	while (fgets(aline,MAX_LINE_LENGTH,hklf)!=NULL){
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

	char datafn[MAX_LINE_LENGTH];
	sprintf(datafn,"%s/%s",folder,"Data.bin");
	char ndatafn[MAX_LINE_LENGTH];
	sprintf(ndatafn,"%s/%s",folder,"nData.bin");
	char spotsfn[MAX_LINE_LENGTH];
	sprintf(spotsfn,"%s/%s",folder,"Spots.bin");
	char extrafn[MAX_LINE_LENGTH];
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
	int n_spots_h = ((int)sizeSpots)/(9*sizeof(RealType));
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

	RealType *RTParamArr, RTParamArr_h[5 + MAX_N_RINGS + 8 + 10];
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
	RTParamArr_h[5+MAX_N_RINGS+8+6] = Parameters.Wavelength;
	RTParamArr_h[5+MAX_N_RINGS+8+7] = Parameters.wedge;
	RTParamArr_h[5+MAX_N_RINGS+8+8] = Parameters.MargABC;
	RTParamArr_h[5+MAX_N_RINGS+8+9] = Parameters.MargABG;
	cudaMalloc((RealType **)&RTParamArr,(23+MAX_N_RINGS)*sizeof(RealType));
	cudaMemcpy(RTParamArr,RTParamArr_h,(23+MAX_N_RINGS)*sizeof(RealType),cudaMemcpyHostToDevice);

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
		if (ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 2] > maxJobs)
			maxJobs = ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 2];
		if (ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 0] > maxJobsOrient)
			maxJobsOrient = ResultMakeOrientations_h[i*N_COLS_ORIENTATION_NUMBERS + 0];
	}

	printf("Total Jobs: %lld, MaxJobs for one combination: %d\n",totalJobs,maxJobs);

	fseek(fExtraInfo,0L,SEEK_END);
	long long sizeExtra = ftell(fExtraInfo);
	rewind(fExtraInfo);
	RealType *ExtraInfo_h;
	ExtraInfo_h = (RealType *)malloc(sizeExtra);
	fread(ExtraInfo_h,sizeExtra,1,fExtraInfo);

	int sizeAllSpots = (sizeExtra/14)*8;
	int nExtraSpots = sizeAllSpots/(8*sizeof(RealType));
	RealType *AllSpotsYZO_h;
	AllSpotsYZO_h = (RealType *) malloc(sizeAllSpots);
	for (int i=0;i<nExtraSpots;i++){
		AllSpotsYZO_h[i*8+0] = ExtraInfo_h[i*14+0];
		AllSpotsYZO_h[i*8+1] = ExtraInfo_h[i*14+1];
		AllSpotsYZO_h[i*8+2] = ExtraInfo_h[i*14+2];
		AllSpotsYZO_h[i*8+3] = ExtraInfo_h[i*14+4]; // ID
		AllSpotsYZO_h[i*8+4] = ExtraInfo_h[i*14+8];
		AllSpotsYZO_h[i*8+5] = ExtraInfo_h[i*14+9];
		AllSpotsYZO_h[i*8+6] = ExtraInfo_h[i*14+10];
		AllSpotsYZO_h[i*8+7] = ExtraInfo_h[i*14+5];
	}
	RealType *AllSpotsYZO_d;
	cudaMalloc((RealType **)&AllSpotsYZO_d,sizeAllSpots);
	cudaMemcpy(AllSpotsYZO_d,AllSpotsYZO_h,sizeAllSpots,cudaMemcpyHostToDevice);

	RealType *AllTheorSpots, *IAs, *IAs_h, *GS, *Orientations, *GS_h, *Orientations_h, *AllInfo, *SpotsInfo_d,
		*SpotsInfo, *OrientationsOut, *OrientationsOut_h;
	int *AllGrainSpots,*nSpotsArr,*nMatchedArr,*nMatchedArr_h,*nSpotsArr_h, *SpotsInfoTotal;
	cudaMalloc((RealType **)&AllTheorSpots,maxJobsOrient*n_hkls_h*N_COL_THEORSPOTS*2*sizeof(RealType));
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
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
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	cudaMalloc((RealType **)&Orientations,9*maxJobsOrient*sizeof(RealType));
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	cudaMalloc((RealType **)&SpotsInfo_d,n_hkls_h*2*9*maxJobs*sizeof(RealType));
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	cudaMalloc((RealType **)&OrientationsOut,10*maxJobs*sizeof(RealType));
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	SpotsInfo = (RealType *)malloc(n_hkls_h*2*9*sumTotal*sizeof(RealType));
	OrientationsOut_h = (RealType *) malloc(10*maxJobs*sizeof(RealType));
	GS_h = (RealType *) malloc(3*maxJobs*sizeof(RealType));
	Orientations_h = (RealType *) malloc(9*maxJobsOrient*sizeof(RealType));
	AllInfo = (RealType *) malloc(N_COL_GRAINMATCHES*sumTotal*sizeof(RealType));
	memset(AllInfo,0,N_COL_GRAINMATCHES*sumTotal*sizeof(RealType));
	SpotsInfoTotal = (int *) malloc(sumTotal*n_hkls_h*2*sizeof(int));
	memset(SpotsInfoTotal,0,sumTotal*n_hkls_h*2*sizeof(int));
	printf("Time elapsed before calculation of matches: %fs\n",cpuSecond()-iStart);
	int PosOM;
	for (int jobNr=0;jobNr<sumTotal;jobNr++){
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
		memset(OrientationsOut_h,0,10*nJobsTotal*sizeof(RealType));
		CompareDiffractionSpots<<<gridc,blockc>>>(AllTheorSpots,RTParamArr,
			nJobsTotal, ResultArr, posResultArr, nSpotsArr, data, nData, ObsSpotsLab,
			etamargins_d, AllGrainSpots, IAs, n_arr, nMatchedArr, n_min, nJobsOrient,GS,
			AllSpotsYZO_d,SpotsInfo_d,Orientations,OrientationsOut);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(nMatchedArr_h,nMatchedArr,nJobsTotal*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(GS_h,GS,nJobsTotal*3*sizeof(RealType),cudaMemcpyDeviceToHost);
		cudaMemcpy(IAs_h,IAs,nJobsTotal*sizeof(RealType),cudaMemcpyDeviceToHost);
		cudaMemcpy(OrientationsOut_h,OrientationsOut,10*nJobsTotal*sizeof(RealType),cudaMemcpyDeviceToHost);
		bestFraction = 0.0;
		bestIA = 1000.0;
		for (int idx=0;idx<nJobsTotal;idx++){
			tempFraction = ((RealType)nMatchedArr_h[idx])/(OrientationsOut_h[idx*10+9]);
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
			cudaMemcpy(SpotsInfo+jobNr*n_hkls_h*2*9, SpotsInfo_d+BestPosition*n_hkls_h*2*9,nMatchedArr_h[BestPosition]*9*sizeof(RealType),cudaMemcpyDeviceToHost);
			AllInfo[jobNr*N_COL_GRAINMATCHES + 0] = bestIA;
			AllInfo[jobNr*N_COL_GRAINMATCHES + 1] = OrientationsOut_h[BestPosition*10+0];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 2] = OrientationsOut_h[BestPosition*10+1];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 3] = OrientationsOut_h[BestPosition*10+2];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 4] = OrientationsOut_h[BestPosition*10+3];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 5] = OrientationsOut_h[BestPosition*10+4];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 6] = OrientationsOut_h[BestPosition*10+5];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 7] = OrientationsOut_h[BestPosition*10+6];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 8] = OrientationsOut_h[BestPosition*10+7];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 9] = OrientationsOut_h[BestPosition*10+8];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 10] = GS_h[BestPosition*3 + 0];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 11] = GS_h[BestPosition*3 + 1];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 12] = GS_h[BestPosition*3 + 2];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 13] = OrientationsOut_h[BestPosition*10+9];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 14] = (RealType)nMatchedArr_h[BestPosition];
			AllInfo[jobNr*N_COL_GRAINMATCHES + 15] = bestFraction;
		}
	}
	printf("Time elapsed after calculation of matches: %fs\n",cpuSecond()-iStart);

	// Now sort all the results.
	RealType *SaveAllInfo, *spotsYZO, *LatCIn_h, *FitParams_h;
	int *nMatchedArrIndexing;
	spotsYZO = (RealType *) malloc(nspids*n_hkls_h*2*8*sizeof(RealType));
	LatCIn_h = (RealType *) malloc(nspids*6*sizeof(RealType));
	FitParams_h = (RealType *) malloc(nspids*12*sizeof(RealType));
	nMatchedArrIndexing = (int *) malloc(nspids*3*sizeof(int));
	memset(spotsYZO,0,nspids*n_hkls_h*2*9*sizeof(RealType));
	memset(LatCIn_h,0,nspids*6*sizeof(RealType));
	memset(FitParams_h,0,nspids*12*sizeof(RealType));
	memset(nMatchedArrIndexing,0,nspids*3*sizeof(int));
	int StartingPosition, EndPosition, bestPos;
	RealType OrientTr[3][3], EulerTr[3];
	int nSpotsIndexed = 0, nMatchedTillNow = 0, nSpotsMatched, nSpotsSim, *idsIndexed;
	idsIndexed = (int *) malloc(nspids*sizeof(int));
	memset(idsIndexed,0,nspids*sizeof(int));
	for (int i=0;i<nspids;i++){
		StartingPosition = startingIDs[i];
		EndPosition = StartingPosition + nNormals_h[i];
		bestFraction = Parameters.MinMatchesToAcceptFrac;
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
			nSpotsMatched = (int)AllInfo[bestPos*N_COL_GRAINMATCHES+14];
			nSpotsSim = (int)AllInfo[bestPos*N_COL_GRAINMATCHES+13];
			nMatchedArrIndexing[nSpotsIndexed*3+0] = nSpotsMatched;
			nMatchedArrIndexing[nSpotsIndexed*3+1] = nSpotsSim;
			nMatchedArrIndexing[nSpotsIndexed*3+2] = nMatchedTillNow;
			idsIndexed[nSpotsIndexed] = SpotIDs_h[i];
			memcpy(spotsYZO+nMatchedTillNow*9, SpotsInfo + bestPos*n_hkls_h*2*9, nSpotsMatched*9*sizeof(RealType));
			memcpy(LatCIn_h+nSpotsIndexed*6, RTParamArr_h+5+MAX_N_RINGS+8, 6*sizeof(RealType));
			memcpy(FitParams_h+nSpotsIndexed*12, AllInfo + bestPos*N_COL_GRAINMATCHES + 10, 3*sizeof(RealType)); // Pos
			for (int j=0;j<3;j++){
				memcpy(&OrientTr[j][0],AllInfo+bestPos*N_COL_GRAINMATCHES+1+3*j,3*sizeof(RealType));
			}
			OrientMat2Euler(OrientTr,EulerTr);
			memcpy(FitParams_h+nSpotsIndexed*12+3,EulerTr,3*sizeof(RealType)); // Orientation
			memcpy(FitParams_h+nSpotsIndexed*12+6,LatCIn_h+nSpotsIndexed*6,6*sizeof(RealType)); // LatticeParameter
			nSpotsIndexed++;
			nMatchedTillNow += nSpotsMatched;
		}
	}

	printf("Out of %d IDs, %d IDs were indexed.\n",nspids,nSpotsIndexed);
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
	cudaFree(etamargins_d);
	cudaFree(nNormals);
	cudaFree(ResultMakeOrientations);
	printf("Time elapsed after sorting the results: %lfs\nNow refining results.\n",cpuSecond()-iStart);
	// We have spotsYZO, FitParams_h, we just call the function to run things.
    int startRow, endRow, startRowNMatched, endRowNMatched, nrows, nrowsNMatched;
	RealType *SpotsCompReturnArr, *SpListArr, *ErrorArr;
	SpotsCompReturnArr = (RealType *)malloc(nMatchedTillNow*22*sizeof(RealType));
	SpListArr = (RealType *)malloc(nMatchedTillNow*9*sizeof(RealType));
	ErrorArr = (RealType *)malloc(nSpotsIndexed*3*sizeof(RealType));
	int maxNJobs = 2*nCores;
    int nJobGroups = nSpotsIndexed/(maxNJobs) + 1;
	int *nMatchedArr_d2;
	int sizeNMatched = maxNJobs*(int)(((RealType)nMatchedTillNow/(RealType)nSpotsIndexed)*1.5);
    int *tempNMatchedArr;
    tempNMatchedArr = (int *)malloc(maxNJobs*3*sizeof(int));
	RealType *scratchspace, *hklspace, *xspace, *xstepspace, *xlspace, *xuspace, *xoutspace,
		*TheorSpotsArr, *SpotsMatchedArr_d2, *FitParams_d2, *CorrectSpots,
		*FitResultArr, *FitResultArr_h, *LatCArr, *LatCIn_d2;
	cudaMalloc((int **)&nMatchedArr_d2,maxNJobs*3*sizeof(RealType));
	cudaMalloc((RealType **)&scratchspace,(3*maxNJobs+(maxNJobs+1)*(maxNJobs+1))*sizeof(RealType));
	cudaMalloc((RealType **)&hklspace,maxNJobs*n_hkls_h*7*sizeof(RealType));
	cudaMalloc((RealType **)&xspace,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&xstepspace,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&xlspace,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&xuspace,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&xoutspace,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&TheorSpotsArr,n_hkls_h*2*8*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&nMatchedArr_d2,3*maxNJobs*sizeof(int));
	cudaMalloc((RealType **)&SpotsMatchedArr_d2,sizeNMatched*9*sizeof(RealType));
	cudaMalloc((RealType **)&CorrectSpots,sizeNMatched*6*sizeof(RealType));
	cudaMalloc((RealType **)&FitParams_d2,12*maxNJobs*sizeof(RealType));
	cudaMalloc((RealType **)&FitResultArr,12*maxNJobs*sizeof(RealType));
	FitResultArr_h = (RealType *) malloc(maxNJobs*12*sizeof(RealType));
	cudaMalloc((RealType **)&LatCIn_d2,maxNJobs*6*sizeof(RealType));
	LatCArr = (RealType *) malloc(maxNJobs*6*sizeof(RealType));
	RealType *hkls_dcorr, *SpCmp_d2, *Error_d2;
	cudaMalloc((RealType **)&hkls_dcorr,maxNJobs*n_hkls_h*7*sizeof(RealType));
	cudaMalloc((RealType **)&SpCmp_d2, sizeNMatched*22*sizeof(RealType));
	cudaMalloc((RealType **)&Error_d2, maxNJobs*3*sizeof(RealType));
	for (int jobNr=0;jobNr<nJobGroups;jobNr++){
		printf("Optimization set: %d out of %d\n",jobNr,nJobGroups);
		startRow = jobNr*maxNJobs;
		endRow = (jobNr + 1 != nJobGroups) ? ((jobNr+1)*maxNJobs)-1 : ((nSpotsIndexed-1)%maxNJobs);
		nrows = endRow - startRow + 1;
		startRowNMatched = nMatchedArrIndexing[startRow*3+2];
		endRowNMatched = nMatchedArrIndexing[(endRow)*3+2] + nMatchedArrIndexing[(endRow)*3];
		nrowsNMatched = endRowNMatched - startRowNMatched;
		n_arr_h[2] = nrows;
		cudaMemcpy(n_arr,n_arr_h,3*sizeof(int),cudaMemcpyHostToDevice);
		nSpotsMatched = 0;
		for (int i=0;i<nrows;i++){
			tempNMatchedArr[i*3] = nMatchedArrIndexing[(i+startRow)*3];
			tempNMatchedArr[i*3+1] = nMatchedArrIndexing[(i+startRow)*3+1];
			tempNMatchedArr[i*3+2] = nSpotsMatched;
			nSpotsMatched += nMatchedArrIndexing[(i+startRow)*3];
		}
		printf("%d %d %d %d %d %d %d %d\n",nrows,nrowsNMatched,startRow,endRow, startRowNMatched, endRowNMatched, nSpotsIndexed, nMatchedTillNow);
		cudaMemcpy(nMatchedArr_d2,tempNMatchedArr,3*nrows*sizeof(int),cudaMemcpyHostToDevice);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(SpotsMatchedArr_d2,spotsYZO+startRowNMatched*9,nrowsNMatched*9*sizeof(RealType),cudaMemcpyHostToDevice);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(FitParams_d2,FitParams_h+12*startRow,12*nrows*sizeof(RealType),cudaMemcpyHostToDevice);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		dim3 blockf (32);
		dim3 gridf ((maxNJobs/blockf.x)+1);
		// Call the optimization routines.
		FitGrain<<<gridf,blockf>>>(RTParamArr,IntParamArr,n_arr,OmeBoxArr,
			hkls_d, HKLints_d,nMatchedArr_d2,SpotsMatchedArr_d2,FitParams_d2,
			TheorSpotsArr, scratchspace, hklspace, xspace, xlspace, xuspace,
			xoutspace,xstepspace, CorrectSpots, FitResultArr);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(FitResultArr_h,FitResultArr,12*nrows*sizeof(RealType),cudaMemcpyDeviceToHost);
		CalcAngleErrors<<<gridf,blockf>>>(RTParamArr,IntParamArr,n_arr,
			OmeBoxArr,hkls_dcorr,nMatchedArr_d2,SpotsMatchedArr_d2,FitResultArr,
			TheorSpotsArr,SpCmp_d2,Error_d2, hkls_d, HKLints_d);
		CHECK(cudaPeekAtLastError());
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(SpotsCompReturnArr+22*startRowNMatched,SpCmp_d2,22*nrowsNMatched*sizeof(RealType),cudaMemcpyDeviceToHost);
		cudaMemcpy(ErrorArr+3*startRow,Error_d2,3*nrows*sizeof(RealType),cudaMemcpyDeviceToHost);
		printf("Finished one set of optimizations in: %lfseconds.\n",cpuSecond()-iStart);
	}
	// We have 	idsIndexed with the successful IDs.
	//			nMatchedArrIndexing to guide where to look
	//			SpotsCompReturnArr with the info about each matched spot, 
	//			ErrorArr for errors and 
	//			FitResultArr_h with all the fit parameters.

	// First move spotIDsfn to a backup so that we don't overwrite this.
	char cmd[MAX_LINE_LENGTH];
	sprintf(cmd,"mv %s %s.orig",spotIDsfn,spotIDsfn);
	system(cmd);
	char outIDsfn[MAX_LINE_LENGTH];
	sprintf(outIDsfn,"%s/%s",folder,spotIDsfn);
	char fitbestfn[MAX_LINE_LENGTH];
	char opfitfn[MAX_LINE_LENGTH];
	sprintf(fitbestfn,"%s/FitBest.bin",Parameters.ResultFolder);
	FILE *fb;
	fb = fopen(fitbestfn,"w");
	fwrite(SpotsCompReturnArr,nMatchedTillNow*22*sizeof(RealType),1,fb);
	sprintf(opfitfn,"%s/OrientPosFit.bin",Parameters.ResultFolder);
	FILE *fo;
	fo = fopen(opfitfn,"w");
	FILE *outidsfile;
	outidsfile = fopen(outIDsfn,"w");
	RealType *OpArr;
	OpArr  = (RealType *) malloc(nSpotsIndexed*25*sizeof(RealType));
	RealType OrientMat[3][3];
	for (int i=0;i<nSpotsIndexed;i++){
		fprintf(outidsfile,"%d\n",idsIndexed[i]);
		OpArr[i*25+0] = (RealType) idsIndexed[i];
		Euler2OrientMat_h(FitResultArr_h+i*12+3,OrientMat);
		for (int j=0;j<3;j++){
			for (int k=0;k<3;k++){
				OpArr[i*25+1+j*3+k] = OrientMat[j][k];
			}
			OpArr[i*25+10+j] = FitResultArr_h[i*12+j];
			OpArr[i*25+13+j] = FitResultArr_h[i*12+6+j];
			OpArr[i*25+16+j] = FitResultArr_h[i*12+9+j];
			OpArr[i*25+19+j] = ErrorArr[i*3+j];
			OpArr[i*25+22+j] = (RealType)nMatchedArrIndexing[i*3+j];
		}
		printf("%d %lf %lf %lf\n",idsIndexed[i],ErrorArr[i*3+0],ErrorArr[i*3+1],ErrorArr[i*3+2]);
	}
	fwrite(OpArr,25*nSpotsIndexed*sizeof(RealType),1,fo);
	cudaDeviceReset();
	printf("Time elapsed: %fs\n",cpuSecond()-iStart);
	return 0;
}
