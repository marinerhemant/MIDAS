//
//  FitPosOrStrains.c
//  
//
//  Created by Hemant Sharma on 2014/06/20.
//
//
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
#define MaxNSpots 6000000
#define MaxNSpotsBest 500
#define MaxNHKLS 5000
#define MaxLineLength 2048
#define MaxNRings 100
#define EPS 1E-12
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))

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

int main(int argc, char *argv[])
{
	char *ParamFN;
	FILE *fileParam;
	ParamFN = argv[1];
	char aline[MaxLineLength];
	fileParam = fopen(ParamFN,"r");
	char *str, dummy[MaxLineLength];
	int LowNr;
	double Wavelength, Lsd, LatCin[6], wedge, MinEta, OmegaRanges[20][2],
		BoxSizes[20][4], MaxRingRad, Rsample, Hbeam, RingRadii[MaxNRings],
		MargABC=0.3, MargABG=0.3;
	int RingNumbers[MaxNRings], cs=0, cs2=0, nOmeRanges=0, nBoxSizes=0,
		DiscModel=0, TopLayer=0, Twins=0, TakeGrainMax=0;
	char OutputFolder[MaxLineLength], ResultFolder[MaxLineLength];
}
