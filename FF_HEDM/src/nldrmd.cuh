// This is a condensed version from NLOPT, MIT license follows.
/* Copyright (c) 2007-2014 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
 */

#ifndef _NLDRMD_H_
#define _NLDRMD_H_

#define alpha 1.0
#define beta 0.5
#define gamm 2.0
#define delta 0.5

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

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

typedef struct {
     unsigned n;
     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     const double *xtol_abs;
     int nevals, maxeval;
     double maxtime, start;
     int *force_stop;
} nlopt_stopping;

#define CHECK_EVAL(xc,fc) 						  \
	stop->nevals++;							  \
	if ((fc) <= *minf) {							  \
		*minf = (fc); memcpy(x, (xc), n * sizeof(double));			  \
		if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; } \
	}									  \
	if (nlopt_stop_evals(stop)) { ret=NLOPT_MAXEVAL_REACHED; goto done; }	 // \


typedef double (*nlopt_func)(unsigned n, const double *x,
			     double *gradient, /* NULL if not needed */
			     void *func_data);

typedef double *rb_key;

typedef enum { RED, BLACK } rb_color;

typedef struct rb_node_s {
     struct rb_node_s *p, *r, *l; /* parent, right, left */
     rb_key k; /* key (and data) */
     rb_color c;
} rb_node;

typedef int (*rb_compare)(rb_key k1, rb_key k2);

typedef struct {
     rb_compare compare;
     rb_node *root;
     int N; /* number of nodes */
} rb_tree;

__device__ void rb_tree_init(rb_tree *t, rb_compare compare);
__device__ void rb_tree_destroy(rb_tree *t);
__device__ void rb_tree_destroy_with_keys(rb_tree *t);
__device__ rb_node *rb_tree_insert(rb_tree *t, rb_key k);
__device__ rb_node *rb_tree_find(rb_tree *t, rb_key k);
__device__ rb_node *rb_tree_find_le(rb_tree *t, rb_key k);
__device__ rb_node *rb_tree_find_lt(rb_tree *t, rb_key k);
__device__ rb_node *rb_tree_find_gt(rb_tree *t, rb_key k);
__device__ rb_node *rb_tree_resort(rb_tree *t, rb_node *n);
__device__ rb_node *rb_tree_min(rb_tree *t);
__device__ rb_node *rb_tree_max(rb_tree *t);
__device__ rb_node *rb_tree_succ(rb_node *n);
__device__ rb_node *rb_tree_pred(rb_node *n);
__device__ void rb_tree_shift_keys(rb_tree *t, ptrdiff_t kshift);
__device__ rb_node *rb_tree_remove(rb_tree *t, rb_node *n);

__device__ nlopt_result nldrmd_minimize(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf, nlopt_stopping *stop);

#endif
