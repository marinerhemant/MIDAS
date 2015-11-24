//
//  Peaks.cu
//
//
//  Created by Hemant Sharma on 2015/07/04.
//

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "nldrmd.cuh"

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAX_LINE_LENGTH 10240
#define MAX_N_RINGS 5000
#define MAX_N_OVERLAPS 20
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
#define CHECK(call){														\
	const cudaError_t error = call;											\
	if (error != cudaSuccess){												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(-10*error);													\
	}																		\
}
typedef uint16_t pixelvalue;

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

static inline pixelvalue** allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
    int i;
    arr = (pixelvalue **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (pixelvalue *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

// Code from here is included from the NLOPT library, nldrmd functions.
__device__ void rb_tree_init(rb_tree *t, rb_compare compare) {
     t->compare = compare;
     t->root = NULL;
     t->N = 0;
}

__device__ void destroy(rb_node *n){
     if (n != NULL) {
	  destroy(n->l); destroy(n->r);
	  free(n);
     }
}

__device__ void rb_tree_destroy(rb_tree *t){
     destroy(t->root);
     t->root = NULL;
}

__device__ void rb_tree_destroy_with_keys(rb_tree *t){
     rb_node *n = rb_tree_min(t);
     while (n) {
	  free(n->k); n->k = NULL;
	  n = rb_tree_succ(n);
     }
     rb_tree_destroy(t);
}

__device__ void rotate_left(rb_node *p, rb_tree *t){
     rb_node *n = p->r; /* must be non-NULL */
     p->r = n->l;
     n->l = p;
     if (p->p != NULL) {
	  if (p == p->p->l) p->p->l = n;
	  else p->p->r = n;
     }
     else
	  t->root = n;
     n->p = p->p;
     p->p = n;
     if (p->r != NULL) p->r->p = p;
}

__device__ void rotate_right(rb_node *p, rb_tree *t){
     rb_node *n = p->l; /* must be non-NULL */
     p->l = n->r;
     n->r = p;
     if (p->p != NULL) {
	  if (p == p->p->l) p->p->l = n;
	  else p->p->r = n;
     }
     else
	  t->root = n;
     n->p = p->p;
     p->p = n;
     if (p->l != NULL) p->l->p = p;
}

__device__ void insert_node(rb_tree *t, rb_node *n){
     rb_compare compare = t->compare;
     rb_key k = n->k;
     rb_node *p = t->root;
     n->c = RED;
     n->p = n->l = n->r = NULL;
     t->N++;
     if (p == NULL) {
	  t->root = n;
	  n->c = BLACK;
	  return;
     }
     /* insert (RED) node into tree */
     while (1) {
	  if (compare(k, p->k) <= 0) { /* k <= p->k */
	       if (p->l != NULL)
		    p = p->l;
	       else {
		    p->l = n;
		    n->p = p;
		    break;
	       }
	  }
	  else {
	       if (p->r != NULL)
		    p = p->r;
	       else {
		    p->r = n;
		    n->p = p;
		    break;
	       }
	  }
     }
 fixtree:
     if (n->p->c == RED) { /* red cannot have red child */
	  rb_node *u = p == p->p->l ? p->p->r : p->p->l;
	  if (u != NULL && u->c == RED) {
	       p->c = u->c = BLACK;
	       n = p->p;
	       if ((p = n->p) != NULL) {
		    n->c = RED;
		    goto fixtree;
	       }
	  }
	  else {
	       if (n == p->r && p == p->p->l) {
		    rotate_left(p, t);
		    p = n; n = n->l;
	       }
	       else if (n == p->l && p == p->p->r) {
		    rotate_right(p, t);
		    p = n; n = n->r;
	       }
	       p->c = BLACK;
	       p->p->c = RED;
	       if (n == p->l && p == p->p->l)
		    rotate_right(p->p, t);
	       else if (n == p->r && p == p->p->r)
		    rotate_left(p->p, t);
	  }
	      
     }
}

__device__ rb_node *rb_tree_insert(rb_tree *t, rb_key k){
     rb_node *n = (rb_node *) malloc(sizeof(rb_node));
     if (!n) return NULL;
     n->k = k;
     insert_node(t, n);
     return n;
}

__device__ int check_node(rb_node *n, int *nblack, rb_tree *t){
     int nbl, nbr;
     rb_compare compare = t->compare;
     if (n == NULL) { *nblack = 0; return 1; }
     if (n->r != NULL && n->r->p != n) return 0;
     if (n->r != NULL && compare(n->r->k, n->k) < 0)
	  return 0;
     if (n->l != NULL && n->l->p != n) return 0;
     if (n->l != NULL && compare(n->l->k, n->k) > 0)
	  return 0;
     if (n->c == RED) {
	  if (n->r != NULL && n->r->c == RED) return 0;
	  if (n->l != NULL && n->l->c == RED) return 0;
     }
     if (!(check_node(n->r, &nbl, t) && check_node(n->l, &nbr, t))) 
	  return 0;
     if (nbl != nbr) return 0;
     *nblack = nbl + (n->c == BLACK);
     return 1;
}

__device__ rb_node *rb_tree_find(rb_tree *t, rb_key k){
     rb_compare compare = t->compare;
     rb_node *p = t->root;
     while (p != NULL) {
	  int comp = compare(k, p->k);
	  if (!comp) return p;
	  p = comp <= 0 ? p->l : p->r;
     }
     return NULL;
}

__device__ rb_node *find_le(rb_node *p, rb_key k, rb_tree *t){
     rb_compare compare = t->compare;
     while (p != NULL) {
	  if (compare(p->k, k) <= 0) { /* p->k <= k */
	       rb_node *r = find_le(p->r, k, t);
	       if (r) return r;
	       else return p;
	  }
	  else /* p->k > k */
	       p = p->l;
     }
     return NULL; /* k < everything in subtree */
}

__device__ rb_node *rb_tree_find_le(rb_tree *t, rb_key k){
     return find_le(t->root, k, t);
}

__device__ rb_node *find_lt(rb_node *p, rb_key k, rb_tree *t){
     rb_compare compare = t->compare;
     while (p != NULL) {
	  if (compare(p->k, k) < 0) { /* p->k < k */
	       rb_node *r = find_lt(p->r, k, t);
	       if (r) return r;
	       else return p;
	  }
	  else /* p->k >= k */
	       p = p->l;
     }
     return NULL; /* k <= everything in subtree */
}

__device__ rb_node *rb_tree_find_lt(rb_tree *t, rb_key k){
     return find_lt(t->root, k, t);
}

__device__ rb_node *find_gt(rb_node *p, rb_key k, rb_tree *t){
     rb_compare compare = t->compare;
     while (p != NULL) {
	  if (compare(p->k, k) > 0) { /* p->k > k */
	       rb_node *l = find_gt(p->l, k, t);
	       if (l) return l;
	       else return p;
	  }
	  else /* p->k <= k */
	       p = p->r;
     }
     return NULL; /* k >= everything in subtree */
}

__device__ rb_node *rb_tree_find_gt(rb_tree *t, rb_key k){
     return find_gt(t->root, k, t);
}

__device__ rb_node *rb_tree_min(rb_tree *t){
     rb_node *n = t->root;
     while (n != NULL && n->l != NULL)
	  n = n->l;
     return(n == NULL ? NULL : n);
}

__device__ rb_node *rb_tree_max(rb_tree *t){
     rb_node *n = t->root;
     while (n != NULL && n->r != NULL)
	  n = n->r;
     return(n == NULL ? NULL : n);
}

__device__ rb_node *rb_tree_succ(rb_node *n){
     if (!n) return NULL;
     if (n->r == NULL) {
	  rb_node *prev;
	  do {
	       prev = n;
	       n = n->p;
	  } while (prev == n->r && n != NULL);
	  return n == NULL ? NULL : n;
     }
     else {
	  n = n->r;
	  while (n->l != NULL)
	       n = n->l;
	  return n;
     }
}

__device__ rb_node *rb_tree_pred(rb_node *n){
     if (!n) return NULL;
     if (n->l == NULL) {
	  rb_node *prev;
	  do {
	       prev = n;
	       n = n->p;
	  } while (prev == n->l && n != NULL);
	  return n == NULL ? NULL : n;
     }
     else {
	  n = n->l;
	  while (n->r != NULL)
	       n = n->r;
	  return n;
     }
}

__device__ rb_node *rb_tree_remove(rb_tree *t, rb_node *n){
     rb_key k = n->k;
     rb_node *m, *mp;
     if (n->l != NULL && n->r != NULL) {
	  rb_node *lmax = n->l;
	  while (lmax->r != NULL) lmax = lmax->r;
	  n->k = lmax->k;
	  n = lmax;
     }
     m = n->l != NULL ? n->l : n->r;
     if (n->p != NULL) {
	  if (n->p->r == n) n->p->r = m;
	  else n->p->l = m;
     }
     else
	  t->root = m;
     mp = n->p;
     if (m != NULL) m->p = mp;
     if (n->c == BLACK) {
	  if (m->c == RED)
	       m->c = BLACK;
	  else {
	  deleteblack:
	       if (mp != NULL) {
		    rb_node *s = m == mp->l ? mp->r : mp->l;
		    if (s->c == RED) {
			 mp->c = RED;
			 s->c = BLACK;
			 if (m == mp->l) rotate_left(mp, t);
			 else rotate_right(mp, t);
			 s = m == mp->l ? mp->r : mp->l;
		    }
		    if (mp->c == BLACK && s->c == BLACK
			&& s->l->c == BLACK && s->r->c == BLACK) {
			 if (s != NULL) s->c = RED;
			 m = mp; mp = m->p;
			 goto deleteblack;
		    }
		    else if (mp->c == RED && s->c == BLACK &&
			     s->l->c == BLACK && s->r->c == BLACK) {
			 if (s != NULL) s->c = RED;
			 mp->c = BLACK;
		    }
		    else {
			 if (m == mp->l && s->c == BLACK &&
			     s->l->c == RED && s->r->c == BLACK) {
			      s->c = RED;
			      s->l->c = BLACK;
			      rotate_right(s, t);
			      s = m == mp->l ? mp->r : mp->l;
			 }
			 else if (m == mp->r && s->c == BLACK &&
				  s->r->c == RED && s->l->c == BLACK) {
			      s->c = RED;
			      s->r->c = BLACK;
			      rotate_left(s, t);
			      s = m == mp->l ? mp->r : mp->l;
			 }
			 s->c = mp->c;
			 mp->c = BLACK;
			 if (m == mp->l) {
			      s->r->c = BLACK;
			      rotate_left(mp, t);
			 }
			 else {
			      s->l->c = BLACK;
			      rotate_right(mp, t);
			 }
		    }
	       }
	  }
     }
     t->N--;
     n->k = k; /* n may have changed during remove */
     return n; /* the node that was deleted may be different from initial n */
}

__device__ rb_node *rb_tree_resort(rb_tree *t, rb_node *n){
     n = rb_tree_remove(t, n);
     insert_node(t, n);
     return n;
}

__device__  void shift_keys(rb_node *n, ptrdiff_t kshift){
     n->k += kshift;
     if (n->l != NULL) shift_keys(n->l, kshift);
     if (n->r != NULL) shift_keys(n->r, kshift);
}

__device__ void rb_tree_shift_keys(rb_tree *t, ptrdiff_t kshift){
     if (t->root != NULL) shift_keys(t->root, kshift);
}


__device__ int simplex_compare(double *k1, double *k2){
	if (*k1 < *k2) return -1;
	if (*k1 > *k2) return +1;
	return k1 - k2;
}

__device__ int close(double a, double b){
	return (fabs(a - b) <= 1e-13 * (fabs(a) + fabs(b)));
}

__device__ int reflectpt(int n, double *xnew, 
		     const double *c, double scale, const double *xold,
		     const double *lb, const double *ub){
	int equalc = 1, equalold = 1, i;
	for (i = 0; i < n; ++i) {
		double newx = c[i] + scale * (c[i] - xold[i]);
		if (newx < lb[i]) newx = lb[i];
		if (newx > ub[i]) newx = ub[i];
		equalc = equalc && close(newx, c[i]);
		equalold = equalold && close(newx, xold[i]);
		xnew[i] = newx;
	}
	return !(equalc || equalold);
}

__device__ int nlopt_stop_evals (nlopt_stopping *stop){
	if (stop->nevals >= stop->maxeval) return 1;
	return 0;
}

__device__ int nlopt_stop_ftol(nlopt_stopping *stop, double fl, double fh){
	if (fabs((fh-fl)/fl) < stop->ftol_rel) return 1;
	return 0;
}

__device__ int nlopt_stop_x(nlopt_stopping *stop, double *cen, double *xpos){
	int i;
	for (i=0;i<stop->n;i++){
		if (fabs((xpos[i]-cen[i])/cen[i]) > stop->xtol_rel){
			return 0;
		}
	}
	return 1;
}

__device__ nlopt_result nldrmd_minimize_(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf,
			     const double *xstep, /* initial step sizes */
			     nlopt_stopping *stop,
			     double psi, double *scratch,
			     double *fdiff){
     double *pts; /* (n+1) x (n+1) array of n+1 points plus function val [0] */
     double *c; /* centroid * n */
     double *xcur; /* current point */
     rb_tree t; /* red-black tree of simplex, sorted by f(x) */
     int i, j;
     double ninv = 1.0 / n;
     nlopt_result ret = NLOPT_SUCCESS;
     double init_diam = 0;

     pts = scratch;
     c = scratch + (n+1)*(n+1);
     xcur = c + n;

     rb_tree_init(&t, simplex_compare);

     *fdiff = HUGE_VAL;

     /* initialize the simplex based on the starting xstep */
     memcpy(pts+1, x, sizeof(double)*n);
     pts[0] = *minf;
     if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; }
     for (i = 0; i < n; ++i) {
	  double *pt = pts + (i+1)*(n+1);
	  memcpy(pt+1, x, sizeof(double)*n);
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
	  pt[0] = f(n, pt+1, NULL, f_data);
	  CHECK_EVAL(pt+1, pt[0]);
     }

 restart:
     for (i = 0; i < n + 1; ++i)
	  if (!rb_tree_insert(&t, pts + i*(n+1))) {
	       ret = NLOPT_OUT_OF_MEMORY;
	       goto done;
	  }

     while (1) {
	  rb_node *low = rb_tree_min(&t);
	  rb_node *high = rb_tree_max(&t);
	  double fl = low->k[0], *xl = low->k + 1;
	  double fh = high->k[0], *xh = high->k + 1;
	  double fr;

	  *fdiff = fh - fl;

	  if (init_diam == 0) /* initialize diam. for psi convergence test */
	       for (i = 0; i < n; ++i) init_diam += fabs(xl[i] - xh[i]);

	  if (psi <= 0 && nlopt_stop_ftol(stop, fl, fh)) {
	       ret = NLOPT_FTOL_REACHED;
	       goto done;
	  }

	  /* compute centroid ... if we cared about the perfomance of this,
	     we could do it iteratively by updating the centroid on
	     each step, but then we would have to be more careful about
	     accumulation of rounding errors... anyway n is unlikely to
	     be very large for Nelder-Mead in practical cases */
	  memset(c, 0, sizeof(double)*n);
	  for (i = 0; i < n + 1; ++i) {
	       double *xi = pts + i*(n+1) + 1;
	       if (xi != xh)
		    for (j = 0; j < n; ++j)
			 c[j] += xi[j];
	  }
	  for (i = 0; i < n; ++i) c[i] *= ninv;

	  /* x convergence check: find xcur = max radius from centroid */
	  memset(xcur, 0, sizeof(double)*n);
	  for (i = 0; i < n + 1; ++i) {
               double *xi = pts + i*(n+1) + 1;
	       for (j = 0; j < n; ++j) {
		    double dx = fabs(xi[j] - c[j]);
		    if (dx > xcur[j]) xcur[j] = dx;
	       }
	  }
	  for (i = 0; i < n; ++i) xcur[i] += c[i];
	  if (psi > 0) {
	       double diam = 0;
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
	  if (!reflectpt(n, xcur, c, alpha, xh, lb, ub)) { 
	       ret=NLOPT_XTOL_REACHED; goto done; 
	  }
	  fr = f(n, xcur, NULL, f_data);
	  CHECK_EVAL(xcur, fr);

	  if (fr < fl) { /* new best point, expand simplex */
	       if (!reflectpt(n, xh, c, gamm, xh, lb, ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fh = f(n, xh, NULL, f_data);
	       CHECK_EVAL(xh, fh);
	       if (fh >= fr) { /* expanding didn't improve */
		    fh = fr;
		    memcpy(xh, xcur, sizeof(double)*n);
	       }
	  }
	  else if (fr < rb_tree_pred(high)->k[0]) { /* accept new point */
	       memcpy(xh, xcur, sizeof(double)*n);
	       fh = fr;
	  }
	  else { /* new worst point, contract */
	       double fc;
	       if (!reflectpt(n,xcur,c, fh <= fr ? -beta : beta, xh, lb,ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fc = f(n, xcur, NULL, f_data);
	       CHECK_EVAL(xcur, fc);
	       if (fc < fr && fc < fh) { /* successful contraction */
		    memcpy(xh, xcur, sizeof(double)*n);
		    fh = fc;
	       }
	       else { /* failed contraction, shrink simplex */
		    rb_tree_destroy(&t);
		    rb_tree_init(&t, simplex_compare);
		    for (i = 0; i < n+1; ++i) {
			 double *pt = pts + i * (n+1);
			 if (pt+1 != xl) {
			      if (!reflectpt(n,pt+1, xl,-delta,pt+1, lb,ub)) {
				   ret = NLOPT_XTOL_REACHED;
				   goto done;
			      }
			      pt[0] = f(n, pt+1, NULL, f_data);
			      CHECK_EVAL(pt+1, pt[0]);
			 }
		    }
		    goto restart;
	       }
	  }

	  high->k[0] = fh;
	  rb_tree_resort(&t, high);
     }
     
done:
     rb_tree_destroy(&t);
     return ret;
}

__device__ nlopt_result nldrmd_minimize(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf, nlopt_stopping *stop){
     nlopt_result ret;
     double *scratch, fdiff;
     double *xstep; /* initial step sizes */
     for (int i=0;i<n;i++){
		 xstep[i] = fabs(ub[i]-lb[i])*0.25;
	 }
     *minf = f(n, x, NULL, f_data);
     stop->nevals++;
     if (*minf < stop->minf_max) return NLOPT_MINF_MAX_REACHED;
     if (nlopt_stop_evals(stop)) return NLOPT_MAXEVAL_REACHED;

     scratch = (double*) malloc(sizeof(double) * ((n+1)*(n+1) + 2*n));
     if (!scratch) return NLOPT_OUT_OF_MEMORY;

     ret = nldrmd_minimize_(n, f, f_data, lb, ub, x, minf, xstep, stop,
			    0.0, scratch, &fdiff);
     free(scratch);
     return ret;
}

// End NLOPT nldrmd functions.

static inline int** allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = (int **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (int *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}


static inline void FreeMemMatrixPx(pixelvalue **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *Image, int NrPixels)
{
	int i,j,k,l,m;
    pixelvalue **ImageTemp1, **ImageTemp2;
    ImageTemp1 = allocMatrixPX(NrPixels,NrPixels);
    ImageTemp2 = allocMatrixPX(NrPixels,NrPixels);
	for (k=0;k<NrPixels;k++) {
		for (l=0;l<NrPixels;l++) {
			ImageTemp1[k][l] = Image[(NrPixels*k)+l];
		}
	}
	for (k=0;k<NrTransOpt;k++) {
		if (TransOpt[k] == 1){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][NrPixels-m-1]; //Inverting Y.
		} else if (TransOpt[k] == 2){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[NrPixels-l-1][m]; //Inverting Z.
		} else if (TransOpt[k] == 3){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[m][l];
		} else if (TransOpt[k] == 0){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][m];
		}
		for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp1[l][m] = ImageTemp2[l][m];
	}
	for (k=0;k<NrPixels;k++) for (l=0;l<NrPixels;l++) Image[(NrPixels*k)+l] = ImageTemp2[k][l];
	FreeMemMatrixPx(ImageTemp1,NrPixels);
	FreeMemMatrixPx(ImageTemp2,NrPixels);
}

static inline void Transposer (double *x, int n, double *y)
{
	int i,j;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			y[(i*n)+j] = x[(j*n)+i];
		}
	}
}

const int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
const int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

static inline void DepthFirstSearch(int x, int y, int current_label, int NrPixels, int **BoolImage, int **ConnectedComponents,int **Positions, int *PositionTrackers)
{
	if (x < 0 || x == NrPixels) return;
	if (y < 0 || y == NrPixels) return;
	if ((ConnectedComponents[x][y]!=0)||(BoolImage[x][y]==0)) return;
	
	ConnectedComponents[x][y] = current_label;
	Positions[current_label][PositionTrackers[current_label]] = (x*NrPixels) + y;
	PositionTrackers[current_label] += 1;
	int direction;
	for (direction=0;direction<8;++direction){
		DepthFirstSearch(x + dx[direction], y + dy[direction], current_label, NrPixels, BoolImage, ConnectedComponents,Positions,PositionTrackers);
		
	}
}

static inline int FindConnectedComponents(int **BoolImage, int NrPixels, int **ConnectedComponents, int **Positions, int *PositionTrackers){
	int i,j;
	for (i=0;i<NrPixels;i++){
		for (j=0;j<NrPixels;j++){
			ConnectedComponents[i][j] = 0;
		}
	}
	int component = 0;
	for (i=0;i<NrPixels;++i) {
		for (j=0;j<NrPixels;++j) {
			if ((ConnectedComponents[i][j]==0) && (BoolImage[i][j] == 1)){
				DepthFirstSearch(i,j,++component,NrPixels,BoolImage,ConnectedComponents,Positions,PositionTrackers);
			}
		}
	}
	return component;
}

static inline unsigned FindRegionalMaxima(double *z,int **PixelPositions,
		int NrPixelsThisRegion,int **MaximaPositions,double *MaximaValues,
		int *IsSaturated, double IntSat)
{
	unsigned nPeaks = 0;
	int i,j,k,l;
	double zThis, zMatch;
	int xThis, yThis;
	int xNext, yNext;
	int isRegionalMax = 1;
	for (i=0;i<NrPixelsThisRegion;i++){
		isRegionalMax = 1;
		zThis = z[i];
		if (zThis > IntSat) {
			*IsSaturated = 1;
		} else {
			*IsSaturated = 0;
		}
		xThis = PixelPositions[i][0];
		yThis = PixelPositions[i][1];
		for (j=0;j<8;j++){
			xNext = xThis + dx[j];
			yNext = yThis + dy[j];
			for (k=0;k<NrPixelsThisRegion;k++){
				if (xNext == PixelPositions[k][0] && yNext == PixelPositions[k][1] && z[k] > (zThis)){
					isRegionalMax = 0;
				}
			}
		}
		if (isRegionalMax == 1){
			MaximaPositions[nPeaks][0] = xThis;
			MaximaPositions[nPeaks][1] = yThis;
			MaximaValues[nPeaks] = zThis;
			nPeaks++;
		}
	}
	if (nPeaks==0){
        MaximaPositions[nPeaks][0] = PixelPositions[NrPixelsThisRegion/2][0];	
        MaximaPositions[nPeaks][1] = PixelPositions[NrPixelsThisRegion/2][1];
        MaximaValues[nPeaks] = z[NrPixelsThisRegion/2];
        nPeaks=1;
	}
	return nPeaks;
}

struct func_data{
	int NrPixels;
	double *RsEtasZ;
};

__device__ void YZ4mREta(int NrElements, double *R, double *Eta, double *Y, double *Z){
	int i;
	for (i=0;i<NrElements;i++){
		Y[i] = -R[i]*sin(Eta[i]*deg2rad);
		Z[i] = R[i]*cos(Eta[i]*deg2rad);
	}
}

__device__ double CalcEtaAngle(double y, double z){
	double alph;
	alph = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alph = -alph;
	return alph;
}

__device__ double problem_function(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct func_data *f_data = (struct func_data *) f_data_trial;
	int NrPixels = f_data->NrPixels;
	double *REtaZ;
	REtaZ = &(f_data->RsEtasZ[0]);
	int nPeaks = (n-1)/8;
	double BG = x[0];
	double TotalDifferenceIntensity = 0, CalcIntensity, IntPeaks;
	double L, G;
	for (int i=0;i<NrPixels;i++){
		IntPeaks = 0;
		for (int j=0;j<nPeaks;j++){
			L = 1/(((((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/((x[(8*j)+6])*(x[(8*j)+6])))+1)*((((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/((x[(8*j)+8])*(x[(8*j)+8])))+1));
			G = exp(-(0.5*(((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/(x[(8*j)+5]*x[(8*j)+5])))-(0.5*(((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/(x[(8*j)+7]*x[(8*j)+7]))));
			IntPeaks += x[(8*j)+1]*((x[(8*j)+4]*L) + ((1-x[(8*j)+4])*G));
		}
		CalcIntensity = BG + IntPeaks;
		TotalDifferenceIntensity += (CalcIntensity - REtaZ[i*3+2])*(CalcIntensity - REtaZ[i+3+2]);
	}
	return TotalDifferenceIntensity;
}

__global__ void Fit2DPeaks (int *PkPx, double *yzInt, double *MaximaInfo, 
	double *ReturnMatrix, int *PosnPeaks, int *PosnPixels, double *ExtraInfo, 
	double *ThreshInfo, double *xDevice, double *xlDevice, double *xuDevice, double *REtaIntDevice){
	int RegNr = blockIdx.x * blockDim.x + threadIdx.x;
	if (RegNr >= (int)ExtraInfo[0]) return;
	int nPeaks = PkPx[RegNr*2];
	int NrPixelsThisRegion = PkPx[RegNr*2+1];
	if (PosnPeaks[RegNr+1] - PosnPeaks[RegNr] != nPeaks) {
		printf("Something wrong with nPeaks\n");
		return;
	}
	if (PosnPixels[RegNr+1] - PosnPixels[RegNr] != NrPixelsThisRegion) {
		printf("Something wrong with NrPixelsThisRegion\n");
		return;
	}
	double Thresh = ThreshInfo[RegNr];
	unsigned n = 1 + (8*nPeaks);
	double *yzIntThis, *MaximaInfoThis, *ReturnMatrixThis ;
	yzIntThis = &yzInt[PosnPixels[RegNr]*3]; // NrPixels, can be used for REtaDevice
	MaximaInfoThis = &MaximaInfo[PosnPeaks[RegNr]*3]; // nPeaks this can be used for ReturnMatrix, x, xl, xu
	ReturnMatrixThis = &ReturnMatrix[PosnPeaks[RegNr]*8];
	// Anything dependent on nPeaks is in PosnPeaks[RegNr] & anything dependent on nrPixels is in PosnPixels[RegNr]
	// ExtraInfo - TotalNrRegions, YCen, ZCen
	double *x,*xl,*xu, *RetaInt, *REtaZ;
	int Posxlu = PosnPeaks[RegNr] * 8 + RegNr;
	int Posreta = PosnPixels[RegNr]*3;
	x =  &xDevice[Posxlu];
	xl = &xlDevice[Posxlu];
	xu = &xuDevice[Posxlu];
	RetaInt = &REtaIntDevice[Posreta];
	REtaZ = RetaInt;

	x[0] = Thresh/2;
	xl[0] = 0;
	xu[0] = Thresh;
	int i;
	for (i=0;i<NrPixelsThisRegion;i++){
		RetaInt[i*3] = CalcNorm2(yzIntThis[i*3]-ExtraInfo[1],yzIntThis[i*3+1]-ExtraInfo[2]);
		RetaInt[i*3+1] = CalcEtaAngle(yzIntThis[i*3]-ExtraInfo[1],yzIntThis[i*3+1]-ExtraInfo[2]);
		RetaInt[i*3+2] = yzIntThis[i*3+2];
	}
	double Width = sqrt((double)NrPixelsThisRegion/(double)nPeaks);
	for (i=0;i<nPeaks;i++){
		x[(8*i)+1] = MaximaInfoThis[i*3]; // Imax
		x[(8*i)+2] = CalcNorm2(MaximaInfoThis[i*3+1]-ExtraInfo[1],MaximaInfoThis[i*3+2]-ExtraInfo[2]); //Radius
		x[(8*i)+3] = CalcEtaAngle(MaximaInfoThis[i*3+1]-ExtraInfo[1],MaximaInfoThis[i*3+2]-ExtraInfo[2]); // Eta
		x[(8*i)+4] = 0.5; // Mu
		x[(8*i)+5] = Width; //SigmaGR
		x[(8*i)+6] = Width; //SigmaLR
		x[(8*i)+7] = rad2deg*atan(Width/x[(8*i)+2]); //SigmaGEta //0.5;
		x[(8*i)+8] = rad2deg*atan(Width/x[(8*i)+2]); //SigmaLEta //0.5;

		double dEta = rad2deg*atan(1/x[(8*i)+2]);
		xl[(8*i)+1] = MaximaInfoThis[i*3]/2;
		xl[(8*i)+2] = x[(8*i)+2] - 1;
		xl[(8*i)+3] = x[(8*i)+3] - dEta;
		xl[(8*i)+4] = 0;
		xl[(8*i)+5] = 0.01;
		xl[(8*i)+6] = 0.01;
		xl[(8*i)+7] = 0.005;
		xl[(8*i)+8] = 0.005;

		xu[(8*i)+1] = MaximaInfoThis[i*3]*2;
		xu[(8*i)+2] = x[(8*i)+2] + 1;
		xu[(8*i)+3] = x[(8*i)+3] + dEta;
		xu[(8*i)+4] = 1;
		xu[(8*i)+5] = 30;
		xu[(8*i)+6] = 30;
		xu[(8*i)+7] = 2;
		xu[(8*i)+8] = 2;
	}
	struct func_data f_data;
	f_data.NrPixels = NrPixelsThisRegion;
	f_data.RsEtasZ = RetaInt;
	struct func_data *f_datat;
	f_datat = &f_data;
	void *trp = (struct func_data *)  f_datat;
	nlopt_result res;
	nlopt_func func = &problem_function;
	double *minf;
	nlopt_stopping stop;
	stop.nevals = 10000;
	stop.maxeval = 10000;
	stop.ftol_rel = 1e-4;
	stop.xtol_rel = 1e-4;
	res = nldrmd_minimize(n, func, trp, xl, xu, x, minf, &stop);
	double IntPeaks, L, G, BGToAdd;
	for (int j=0;j<nPeaks;j++){
		ReturnMatrixThis[j*8] = 0;
		for (i=0;i<NrPixelsThisRegion;i++){
			L = 1/(((((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/((x[(8*j)+6])*(x[(8*j)+6])))+1)*((((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/((x[(8*j)+8])*(x[(8*j)+8])))+1));
			G = exp(-(0.5*(((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/(x[(8*j)+5]*x[(8*j)+5])))-(0.5*(((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/(x[(8*j)+7]*x[(8*j)+7]))));
			IntPeaks = x[(8*j)+1]*((x[(8*j)+4]*L) + ((1-x[(8*j)+4])*G));
			if (IntPeaks > x[0]) BGToAdd = x[0];
			else BGToAdd = 0;
			ReturnMatrixThis[j*8] += (BGToAdd + IntPeaks);
		}
		ReturnMatrixThis[j*8+1] = -x[(8*j)+2]*sin(x[(8*j)+3]*deg2rad);
		ReturnMatrixThis[j*8+2] =  x[(8*j)+2]*cos(x[(8*j)+3]*deg2rad);
		ReturnMatrixThis[j*8+3] =  x[8*j+1];
		ReturnMatrixThis[j*8+4] =  x[8*j+2];
		ReturnMatrixThis[j*8+5] =  x[8*j+3];
		ReturnMatrixThis[j*8+6] =  (x[8*j+5]+x[8*j+6])/2;
		ReturnMatrixThis[j*8+7] =  (x[8*j+7]+x[8*j+8])/2;
	}
}

void CallFit2DPeaks(int *nPeaksNrPixels, double *yzInt, double *MaximaInfo, 
double *ReturnMatrix, int TotNrRegions, double *YZCen, double *ThreshInfo, 
int *PosMaximaInfoReturnMatrix, int *PosyzInt, int totalPixels, 
int totalPeaks)
{
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    size_t gpuGlobalMem = deviceProp.totalGlobalMem;
    fprintf(stderr, "GPU global memory = %zu MBytes\n", gpuGlobalMem/(1024*1024));
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	int *PkPxDevice,*PosMaxInfoRetMatDevice,*PosyzIntDevice;
	cudaMalloc((int **) &PkPxDevice, TotNrRegions*2*sizeof(int));
	cudaMemcpy(PkPxDevice,nPeaksNrPixels,TotNrRegions*2*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc((int **) &PosMaxInfoRetMatDevice, TotNrRegions*sizeof(int));
	cudaMemcpy(PosMaxInfoRetMatDevice,PosMaximaInfoReturnMatrix,TotNrRegions*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc((int **) &PosyzIntDevice, TotNrRegions*sizeof(int));
	cudaMemcpy(PosyzIntDevice,PosyzInt,TotNrRegions*sizeof(int),cudaMemcpyHostToDevice);
	double ExtraInfo[3] = {(double)TotNrRegions,YZCen[0],YZCen[1]};
	double *yzIntDevice, *MaximaInfoDevice, *ReturnMatrixDevice, *ExtraInfoDevice, *ThreshInfoDevice, *xDevice, *xlDevice, *xuDevice, *REtaIntDevice;
	cudaMalloc((double **)&yzIntDevice, totalPixels*3*sizeof(double));
	cudaMemcpy(yzIntDevice,yzInt,totalPixels*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&MaximaInfoDevice, totalPeaks*3*sizeof(double));
	cudaMemcpy(MaximaInfoDevice,MaximaInfo,totalPeaks*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&ReturnMatrixDevice, totalPeaks*8*sizeof(double));
	cudaMalloc((double **)&ThreshInfoDevice, TotNrRegions*sizeof(double));
	cudaMemcpy(ThreshInfoDevice,ThreshInfo,TotNrRegions*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&ExtraInfoDevice, 3*sizeof(double));
	cudaMemcpy(ExtraInfoDevice,ExtraInfo,3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&xDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	cudaMalloc((double **)&xlDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	cudaMalloc((double **)&xuDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	cudaMalloc((double **)&REtaIntDevice,totalPixels*3*sizeof(double));
	int dim = TotNrRegions;
	dim3 block (256);
	dim3 grid ((dim/block.x)+1);
	Fit2DPeaks<<<grid,block>>>(PkPxDevice,yzIntDevice, MaximaInfoDevice, 
		ReturnMatrixDevice, PosMaxInfoRetMatDevice, PosyzIntDevice, 
		ExtraInfoDevice, ThreshInfoDevice, xDevice, xlDevice, xuDevice, 
		REtaIntDevice);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	cudaMemcpy(ReturnMatrix,ReturnMatrixDevice,totalPeaks*8*sizeof(double),cudaMemcpyDeviceToHost);
}

int main(int argc, char *argv[]){ // Arguments: parameter file name
	if (argc != 2){
		printf("Not enough arguments, exiting. Use as:\n\t\t%s %s\n",argv[0],argv[1]);
		return 1;
	}
	//Read params file
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char line[MAX_LINE_LENGTH];
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Parameter file: %s could not be read. Exiting\n",argv[1]);
		return 1;
	}
	char *str;
	int cmpres, StartFileNr, NrFilesPerSweep, NumDarkBegin=0, NumDarkEnd=0,
		ColBeamCurrent, NrOfRings=0, RingNumbers[MAX_N_RINGS], TransOpt[10], 
		NrTransOpt=0, DoFullImage=0, Padding, NrPixels, LayerNr, FrameNumberToDo=-1;
	double OmegaOffset = 0, bc=0, RingSizeThreshold[MAX_N_RINGS][4], px, 
		Width, IntSat, Ycen, Zcen;
	char dummy[MAX_LINE_LENGTH], ParFilePath[MAX_LINE_LENGTH], 
		FileStem[MAX_LINE_LENGTH], RawFolder[MAX_LINE_LENGTH], 
		OutputFolder[MAX_LINE_LENGTH], darkcurrentfilename[MAX_LINE_LENGTH], 
		floodfilename[MAX_LINE_LENGTH], Ext[MAX_LINE_LENGTH];
	while (fgets(line, MAX_LINE_LENGTH, fileParam) != NULL) {
		str = "ParFilePath ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, ParFilePath);
			continue;
		}
		str = "RingThresh ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d %lf", dummy, RingNumbers[NrOfRings], 
				RingSizeThreshold[NrOfRings][1]);
			NrOfRings++;
			continue;
		}
		str = "FileStem ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, FileStem);
			continue;
		}
		str = "ParFileColBeamCurrent ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &ColBeamCurrent);
			continue;
		}
		str = "StartFileNr ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &StartFileNr);
			continue;
		}
		str = "NrFilesPerSweep ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NrFilesPerSweep);
			continue;
		}
		str = "NumDarkBegin ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkBegin);
			continue;
		}
		str = "NumDarkEnd ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkEnd);
			continue;
		}
		str = "OmegaOffset ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &OmegaOffset);
			continue;
		}
		str = "BeamCurrent ";
		cmpres = strncmp(line,str,strlen(str));
		if (cmpres==0){
			sscanf(line,"%s %lf", dummy, &bc);
			continue;
		}
        str = "Width ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &Width);
            continue;
        }
        str = "px ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &px);
            continue;
        }
        str = "ImTransOpt ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "DoFullImage ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &DoFullImage);
            continue;
        }
        str = "RawFolder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, RawFolder);
            continue;
        }
        str = "Folder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, OutputFolder);
            continue;
        }
        str = "Dark ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, darkcurrentfilename);
            continue;
        }
        str = "Flood ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, floodfilename);
            continue;
        }
        str = "BC ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf %lf", dummy, &Ycen, &Zcen);
            continue;
        }
        str = "UpperBoundThreshold ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &IntSat);
            continue;
        }
        str = "LayerNr ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "NrPixels ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "Padding ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &Padding);
            continue;
        }
        str = "SingleFrameNumber ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &FrameNumberToDo);
            continue;
        }
        str = "Ext ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, Ext);
            continue;
        }
	}
	if (DoFullImage == 1 && FrameNumberToDo == -1){
		printf("For processing the full image you need to provide a single"
			" Frame Number using the FrameNumberToDo parameter in the"
			" param file.\n Exiting\n");
		return (1);
	}
	Width = Width/px;
	FILE *ParFile;
	ParFile = fopen(ParFilePath,"r");
	if (ParFile == NULL){
		printf("ParFile could not be read");
		return 1;
	}
	int i, j, k;
	int NrFramesPerFile[NrFilesPerSweep],CurrFileNrOffset;
	for (i=0;i<NrFilesPerSweep;i++){
		NrFramesPerFile[i] = -(NumDarkBegin+NumDarkEnd);
	}
	char *token, *saveptr;
	int OmegaSign=1, goodLine, omegafound;
	double Omegas[NrFilesPerSweep][300],BeamCurrents[NrFilesPerSweep][300],
			maxBC=0;
	while (fgets(line, MAX_LINE_LENGTH, ParFile) != NULL) {
		strncpy(line,line,strlen(line));
		goodLine = 0;
		for (str = line; ; str=NULL){
			token = strtok_r(str, " ", &saveptr);
			if (token == NULL) break;
			if (!strncmp(token,FileStem,strlen(FileStem))){
				token = strtok_r(str, " ", &saveptr);
				token = strtok_r(str, " ", &saveptr);
				CurrFileNrOffset = atoi(token)-StartFileNr;
				if (CurrFileNrOffset >=0 && CurrFileNrOffset < NrFilesPerSweep){
					NrFramesPerFile[CurrFileNrOffset]++;
					goodLine = 1;
				}
			}
		}
		if (NrFramesPerFile[CurrFileNrOffset] < -NumDarkBegin + 1) continue;
		if (goodLine){
			strncpy(line,line,strlen(line));
			omegafound = 0;
			for (i=1, str = line; ; i++, str = NULL){
				token = strtok_r(str, " ", &saveptr);
				if (token == NULL) break;
				if (!strncmp(token,"ramsrot",strlen("ramsrot"))){
					omegafound = 1;
					OmegaSign = 1;
				} else if (!strncmp(token,"aero",strlen("aero"))){
					omegafound = 1;
					OmegaSign = -1;
				} else if (!strncmp(token,"preci",strlen("preci"))){
					omegafound = 1;
					OmegaSign = 1;
				}
				if (omegafound){
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					i+=3;
					Omegas[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] 
								= atof(token) * OmegaSign + OmegaOffset;
					omegafound = 0;
				}
				if (i == ColBeamCurrent){
					BeamCurrents[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] = atof(token);
					maxBC = (maxBC > atof(token)) ? maxBC : atof(token);
				}
			}
		}
	}
	int TotalNrFrames = 0;
	for (i=0;i<NrFilesPerSweep;i++){
		TotalNrFrames += NrFramesPerFile[i];
	}
	bc = (bc > maxBC) ? bc : maxBC;
	// Read hkls.csv
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(line,1000,hklf);
	int Rnr;
	double RRd;
	while (fgets(line,1000,hklf)!=NULL){
		sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", dummy, dummy, 
			dummy, dummy, &Rnr, dummy, dummy, dummy, dummy ,dummy, &RRd);
		for (i=0;i<NrOfRings;i++){
			if (Rnr == RingNumbers[i]){
				RingSizeThreshold[i][0] = RRd/px;
				RingSizeThreshold[i][2] = RRd/px - Width;
				RingSizeThreshold[i][3] = RRd/px + Width;
			}
		}
	}
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 3){
			printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
			return 1;
		}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
        else printf("Transpose.\n");
    }
    int *GoodCoords, *RingInfoImage, TotalGoodPixels=0, ythis, zthis;
    double Rmin, Rmax, Rt;
    GoodCoords = (int*) malloc(NrPixels*NrPixels*sizeof(*GoodCoords));
    RingInfoImage = (int*) malloc(NrPixels*NrPixels*sizeof(*RingInfoImage));
	for (i=1;i<NrPixels;i++){
		for (j=1;j<NrPixels;j++){
			Rt = sqrt((i-Ycen)*(i-Ycen)+(j-Zcen)*(j-Zcen));
			for (k=0;k<NrOfRings;k++){
				Rmin = RingSizeThreshold[k][2];
				Rmax = RingSizeThreshold[k][3];
				if (Rt > Rmin && Rt < Rmax){
					GoodCoords[((i-1)*NrPixels)+(j-1)] = 1;
					RingInfoImage[((i-1)*NrPixels)+(j-1)] = RingNumbers[k];
					TotalGoodPixels++;
				}else {
					GoodCoords[((i-1)*NrPixels)+(j-1)] = 0;
					RingInfoImage[((i-1)*NrPixels)+(j-1)] = 0;
				}
			}
		}
	}
	if (DoFullImage == 1){
		TotalNrFrames = 1;
		for (i=0;i<NrPixels*NrPixels;i++) {
			GoodCoords[i] = 1;
		}
		TotalGoodPixels = NrPixels*NrPixels;
	}
	double *dark,*flood, *darkTemp, *darkTemp2;
	dark = (double *) malloc(NrPixels*NrPixels*NrFilesPerSweep*sizeof(*dark));
	darkTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*darkTemp));
	darkTemp2 = (double *) malloc(NrPixels*NrPixels*sizeof(*darkTemp2));
	flood = (double *) malloc(NrPixels*NrPixels*sizeof(*flood));
	
	// If a darkfile is specified.
	FILE *darkfile=fopen(darkcurrentfilename,"rb");
	int sz, nFrames;
	int SizeFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	long int Skip;
	for (i=0;i<(NrPixels*NrPixels);i++){
		dark[i]=0;
		darkTemp[i]=0;
	}
	pixelvalue *darkcontents;
	darkcontents = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*darkcontents));
	if (darkfile==NULL){
		printf("No dark file was specified, will use %d frames at the beginning of each file for dark calculation.", NumDarkBegin);
	}else{
		fseek(darkfile,0L,SEEK_END);
		sz = ftell(darkfile);
		rewind(darkfile);
		nFrames = sz/(8*1024*1024);
		Skip = sz - (nFrames*8*1024*1024);
		fseek(darkfile,Skip,SEEK_SET);
		printf("Reading dark file: %s, nFrames: %d, skipping first %ld bytes.\n",darkcurrentfilename,nFrames,Skip);
		for (i=0;i<nFrames;i++){
			fread(darkcontents,SizeFile,1,darkfile);
			DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
			for (j=0;j<(NrPixels*NrPixels);j++){
				darkTemp[j] += (double) darkcontents[j];
			}
		}
		fclose(darkfile);
		for (i=0;i<(NrPixels*NrPixels);i++){
			darkTemp[i] /= nFrames;
		}
	}
	Transposer(darkTemp,NrPixels,darkTemp2);
	for (i=0;i<NrFilesPerSweep;i++){
		for (j=0;j<NrPixels*NrPixels;j++){
			dark[i*NrPixels*NrPixels + j] = darkTemp2[j];
		}
	}
	char FN[MAX_LINE_LENGTH];
	if (NumDarkBegin != 0){
		for (i=0;i<NrFilesPerSweep;i++){
			for (j=0;j<NrPixels*NrPixels;j++){
				darkTemp[j] = 0;
			}
			if (Padding == 2){sprintf(FN,"%s/%s_%02d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 3){sprintf(FN,"%s/%s_%03d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 4){sprintf(FN,"%s/%s_%04d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 5){sprintf(FN,"%s/%s_%05d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 6){sprintf(FN,"%s/%s_%06d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 7){sprintf(FN,"%s/%s_%07d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 8){sprintf(FN,"%s/%s_%08d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 9){sprintf(FN,"%s/%s_%09d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			FILE *FileTempDark = fopen(FN,"rb");
			fseek(FileTempDark, 0L, SEEK_END);
			sz = ftell(FileTempDark);
			rewind(FileTempDark);
			nFrames = sz/(8*1024*1024);
			Skip = sz - (nFrames*8*1024*1024);
			fseek(FileTempDark,Skip, SEEK_SET);
			for (j=0;j<NumDarkBegin;j++){
				fread(darkcontents,SizeFile,1,FileTempDark);
				DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
				for (k=0;k<NrPixels*NrPixels;j++){
					darkTemp[k] += (double) darkcontents[k];
				}
			}
			fclose(FileTempDark);
			for (j=0;j<NrPixels*NrPixels;j++){
				darkTemp[k] /= NumDarkBegin;
			}
			Transposer(darkTemp,NrPixels,darkTemp2);
			for (j=0;j<NrPixels*NrPixels;j++){
				dark[i*NrPixels*NrPixels + j] = darkTemp2[j];
			}
		}
	}
	
	free(darkcontents);
	FILE *floodfile=fopen(floodfilename,"rb");
	if (floodfile==NULL){
		printf("Could not read the flood file. Using no flood correction.\n");
		for(i=0;i<(NrPixels*NrPixels);i++){
			flood[i]=1;
		}
	}
	else{
		fread(flood,sizeof(double)*NrPixels*NrPixels, 1, floodfile);
		fclose(floodfile);
	}
	int FrameNr = 0, FramesToSkip, CurrentFileNr, CurrentRingNr;
	double beamcurr, Thresh;
	pixelvalue *Image;
	Image = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*Image));
	double *ImgCorrBCTemp, *ImgCorrBC;
	ImgCorrBC = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBC));
	ImgCorrBCTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBCTemp));
	char outfoldername[MAX_LINE_LENGTH];
	sprintf(outfoldername,"%s/Temp",OutputFolder);
	char extcmd[MAX_LINE_LENGTH];
	sprintf(extcmd,"mkdir -p %s",outfoldername);
	system(extcmd);
	int nOverlapsMaxPerImage = 10000;
	int **BoolImage, **ConnectedComponents, **Positions, *PositionTrackers, NrOfReg;
	BoolImage = allocMatrixInt(NrPixels,NrPixels);
	ConnectedComponents = allocMatrixInt(NrPixels,NrPixels);
	Positions = allocMatrixInt(nOverlapsMaxPerImage,NrPixels*4);
	PositionTrackers = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PositionTrackers));
	int RegNr, IsSaturated;
	char OutFile[MAX_LINE_LENGTH];
	int TotNrRegions, NrPixelsThisRegion;
	int *nPeaksNrPixels,*PosyzInt,*PosMaximaInfoReturnMatrix, *RingNumberMatrix;
	nPeaksNrPixels = (int *) malloc(nOverlapsMaxPerImage*2*sizeof(*nPeaksNrPixels));
	RingNumberMatrix = (int *) malloc(nOverlapsMaxPerImage * 100 * sizeof(*RingNumberMatrix));
	double *yzInt, *MaximaInfo, *ReturnMatrix, *ThreshInfo, *YZCen;
	yzInt = (double *) malloc(nOverlapsMaxPerImage*3*NrPixels*sizeof(*yzInt));
	MaximaInfo = (double *) malloc(nOverlapsMaxPerImage*3*100*sizeof(*MaximaInfo));
	ReturnMatrix = (double *) malloc(nOverlapsMaxPerImage*8*100*sizeof(*ReturnMatrix));
	ThreshInfo = (double *) malloc(nOverlapsMaxPerImage*sizeof(*ThreshInfo));
	PosyzInt = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PosyzInt));
	PosMaximaInfoReturnMatrix = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PosMaximaInfoReturnMatrix));
	YZCen = (double *) malloc(2*sizeof(*YZCen));
	YZCen[0] = Ycen;
	YZCen[1] = Zcen;
	int **MaximaPositions, **UsefulPixels;
	double *MaximaValues, *z, Omega;
	MaximaPositions = allocMatrixInt(NrPixels*10,2);
	MaximaValues = (double*) malloc(NrPixels*10*sizeof(*MaximaValues));
	UsefulPixels = allocMatrixInt(NrPixels*10,2);
	z = (double *) malloc(NrPixels*10*sizeof(*z));
	int counter, counteryzInt, counterMaximaInfoReturnMatrix;
	while (FrameNr < TotalNrFrames){
		if (TotalNrFrames == 1){
			FrameNr = FrameNumberToDo;
			for (i=0;i<NrFilesPerSweep;i++){
				if (NrFramesPerFile[i]/FrameNumberToDo > 0){
					FrameNumberToDo -= NrFramesPerFile[i];
				}else{
					CurrentFileNr = StartFileNr + i;
					FramesToSkip = FrameNumberToDo;
					break;
				}
			}
		}else{
			FramesToSkip = FrameNr;
			for (i=0;i<NrFilesPerSweep;i++){
				if (NrFramesPerFile[i]/FramesToSkip > 0){
					FramesToSkip -= NrFramesPerFile[i];
				}else{
					CurrentFileNr = StartFileNr + i;
					break;
				}
			}
		}
		if (Padding == 2){sprintf(FN,"%s/%s_%02d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 3){sprintf(FN,"%s/%s_%03d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 4){sprintf(FN,"%s/%s_%04d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 5){sprintf(FN,"%s/%s_%05d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 6){sprintf(FN,"%s/%s_%06d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 7){sprintf(FN,"%s/%s_%07d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 8){sprintf(FN,"%s/%s_%08d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 9){sprintf(FN,"%s/%s_%09d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		FILE *ImageFile = fopen(FN,"rb");
		if (ImageFile == NULL){
			printf("Could not read the input file. Exiting.\n");
			return 1;
		}
		fseek(ImageFile,0L,SEEK_END);
		sz = ftell(ImageFile);
		rewind(ImageFile);
		Skip = sz - ((NrFramesPerFile[StartFileNr-CurrentFileNr] + NumDarkEnd - FramesToSkip) * 8*1024*1024);
		printf("Now processing file: %s, Frame: %d\n",FN, FramesToSkip);
		fseek(ImageFile,Skip,SEEK_SET);
		fread(Image,SizeFile,1,ImageFile);
		fclose(ImageFile);
		DoImageTransformations(NrTransOpt,TransOpt,Image,NrPixels);
		beamcurr = BeamCurrents[StartFileNr - CurrentFileNr][FramesToSkip];
		Omega = Omegas[StartFileNr - CurrentFileNr][FramesToSkip];
		printf("Beam current this file: %f, Beam current scaling value: %f\n",beamcurr,bc);
		for (i=0;i<NrPixels*NrPixels;i++)
			ImgCorrBCTemp[i]=Image[i];
		Transposer(ImgCorrBCTemp,NrPixels,ImgCorrBC);
		for (i=0;i<NrPixels*NrPixels;i++){
			ImgCorrBC[i] = (ImgCorrBC[i] - dark[NrPixels*NrPixels*(StartFileNr-CurrentFileNr) + i])/flood[i];
			ImgCorrBC[i] = ImgCorrBC[i]*bc/beamcurr;
			CurrentRingNr = RingInfoImage[i];
			Thresh = RingSizeThreshold[CurrentRingNr][1];
			if (ImgCorrBC[i] < Thresh){
				ImgCorrBC[i] = 0;
			}
			if (GoodCoords[i] == 0){
				ImgCorrBC[i] = 0;
			}
		}
		for (i=0;i<nOverlapsMaxPerImage;i++)
			PositionTrackers[i] = 0;
		for (i=0;i<NrPixels;i++){
			for (j=0;j<NrPixels;j++){
				if (ImgCorrBC[(i*NrPixels)+j] != 0){
					BoolImage[i][j] = 1;
				}else{
					BoolImage[i][j] = 0;
				}
			}
		}
		NrOfReg = FindConnectedComponents(BoolImage,NrPixels,ConnectedComponents,Positions,PositionTrackers);
		if (Padding == 2) {sprintf(OutFile,"%s/%s_%d_%02d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 3) {sprintf(OutFile,"%s/%s_%d_%03d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 4) {sprintf(OutFile,"%s/%s_%d_%04d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 5) {sprintf(OutFile,"%s/%s_%d_%05d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 6) {sprintf(OutFile,"%s/%s_%d_%06d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 7) {sprintf(OutFile,"%s/%s_%d_%07d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 8) {sprintf(OutFile,"%s/%s_%d_%08d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 9) {sprintf(OutFile,"%s/%s_%d_%09d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		FILE *outfilewrite;
		outfilewrite = fopen(OutFile,"w");
		fprintf(outfilewrite,"SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta\n");
		TotNrRegions = NrOfReg;
		counter = 0;
		counteryzInt = 0;
		counterMaximaInfoReturnMatrix = 0;
		for (RegNr=1;RegNr<=NrOfReg;RegNr++){
			NrPixelsThisRegion = PositionTrackers[RegNr];
			if (NrPixelsThisRegion == 1){
				TotNrRegions--;
				continue;
			}
			for (i=0;i<NrPixelsThisRegion;i++){
				UsefulPixels[i][0] = (int)(Positions[RegNr][i]/NrPixels);
				UsefulPixels[i][1] = (int)(Positions[RegNr][i]%NrPixels);
				z[i] = ImgCorrBC[((UsefulPixels[i][0])*NrPixels) + (UsefulPixels[i][1])];
			}
			unsigned nPeaks;
			nPeaks = FindRegionalMaxima(z,UsefulPixels,NrPixelsThisRegion,
				MaximaPositions,MaximaValues,&IsSaturated,IntSat);
			if (IsSaturated == 1){ //Saturated peaks removed
				TotNrRegions--;
				continue;
			}
			if (nPeaks > MAX_N_OVERLAPS){
				printf("Please recompile the code by setting MAX_N_OVERLAPS higher. Right now it is %d. Exiting.\n",MAX_N_OVERLAPS);
				return (1);
			}
			nPeaksNrPixels[counter*2] = nPeaks;
			nPeaksNrPixels[counter*2+1] = NrPixelsThisRegion;
			PosMaximaInfoReturnMatrix[counter] = counterMaximaInfoReturnMatrix;
			PosyzInt[counter] = counteryzInt;
			for (i=0;i<NrPixelsThisRegion;i++){
				yzInt[(counteryzInt+i)*3 + 0] = (double)UsefulPixels[i][0];
				yzInt[(counteryzInt+i)*3 + 1] = (double)UsefulPixels[i][1];
				yzInt[(counteryzInt+i)*3 + 2] = z[i];
			}
			for (i=0;i<nPeaks;i++){
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 0] = MaximaValues[i];
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 1] = (double)MaximaPositions[i][0];
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 2] = (double)MaximaPositions[i][1];
				RingNumberMatrix[counterMaximaInfoReturnMatrix+i] = RingInfoImage[MaximaPositions[0][0]*NrPixels+MaximaPositions[0][1]];
			}
			ThreshInfo[counter] = RingSizeThreshold[RingInfoImage
					[MaximaPositions[0][0]*NrPixels+MaximaPositions[0][1]]][1];
			counteryzInt+= NrPixelsThisRegion;
			counterMaximaInfoReturnMatrix += nPeaks;
			counter++;
		}
		if (counter != TotNrRegions){
			printf("Number of regions calculated and observed do not match. Please check.\n");
			return (1);
		}
		// Now send all info to the GPU calling code
		CallFit2DPeaks(nPeaksNrPixels, yzInt, MaximaInfo, ReturnMatrix, 
			TotNrRegions, YZCen, ThreshInfo, PosMaximaInfoReturnMatrix, 
			PosyzInt, counteryzInt, counterMaximaInfoReturnMatrix);
		for (i=0;i<counterMaximaInfoReturnMatrix;i++){
			fprintf(outfilewrite,"%d %f %f %f %f %f %f %f %f %f %d\n",i+1,
				ReturnMatrix[i*9+0],Omega,ReturnMatrix[i*9+1]+Ycen,
				ReturnMatrix[i*9+2]+Zcen,ReturnMatrix[i*9+3],
				ReturnMatrix[i*9+4], ReturnMatrix[i*9+5],
				ReturnMatrix[i*9+6],ReturnMatrix[i*9+7], RingNumberMatrix[i]);
		}
		fclose(outfilewrite);
		FrameNr++;
	}
}
