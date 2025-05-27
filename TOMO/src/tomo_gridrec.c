//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdbool.h>

#include "tomo_heads.h"

float filterData (float x, gridrecParams *param){
	switch (param->filter_type) {
		case FILTER_NONE:
			break;
		case FILTER_SHEPP_LOGAN:
			x = shlo (x); break;
		case FILTER_HANN:
			x = hann (x); break;
		case FILTER_HAMMING:
			x = hamm (x); break;
		case FILTER_RAMP:
			x = ramp (x); break;
		default:
			break;
	}
	return (x);
}
float shlo (float x){
	return fabs(sin(PI*x)/PI);
}
float hann (float x){
	return fabs(x)*0.5*(1.0+cos(2*PI*x));
}
float hamm (float x){
	return fabs(x)*(0.54+0.46*cos(2*PI*x));
}
float ramp (float x){
	return fabs(x);
}

void initFFTMemoryStructures (gridrecParams *param){
	param->n_prev = 0;
	param->in_1d = NULL;
	param->out_1d = NULL;
	param->nx_prev = 0;
	param->ny_prev = 0;
	param->in_2d = NULL;
	param->out_2d = NULL;
}

void destroyFFTMemoryStructures (gridrecParams *param){
	if (param->in_1d != NULL) fftwf_free (param->in_1d);
	fftwf_destroy_plan(param->backward_plan_1d);
	if (param->in_2d != NULL) fftwf_free (param->in_2d);
	fftwf_destroy_plan(param->forward_plan_2d);
}

void fourn(float data[], unsigned long nn[], int ndim, int isign, gridrecParams *param){
	clock_t start0, end;
	double diftotal;
	start0 = clock();
	int nx = nn[2];
	int ny = nn[1];
	if (ndim != 2){
		printf("fourn only works with ndim=2\n");
		return;
	}
	if ((nx != param->nx_prev) || (ny != param->ny_prev)){
		if (param->nx_prev != 0) fftwf_free(param->in_1d);
		//~ printf("in_2d %ld\n",(long)(sizeof(fftwf_complex)*nx*ny));
		param->sizeMatrices += (long)(sizeof(fftwf_complex)*nx*ny);
		param->in_2d = fftwf_malloc(sizeof(fftwf_complex)*nx*ny);
		param->out_2d = param->in_2d;
		//~ printf("fft_test2f: creating plans, nx=%d, ny=%d, nx_prev=%d, ny_prev=%d\n", nx, ny, param->nx_prev, param->ny_prev);
		param->nx_prev = nx;
		param->ny_prev = ny;
		if (param->setPlan == 1){
			int fftw2d_size = nx;
			char plan2DFN[4096];
			sprintf(plan2DFN,"fftwf_wisdom_2d_%d.txt",(int)fftw2d_size);
			int rc = fftwf_import_wisdom_from_filename(plan2DFN);
			if (rc == 1){
				param->forward_plan_2d = fftwf_plan_dft_2d(ny, nx, param->in_2d, param->out_2d, FFTW_FORWARD, FFTW_WISDOM_ONLY);
			} else {
				printf("Creating wisdom file. %s\n",plan2DFN);
				param->forward_plan_2d = fftwf_plan_dft_2d(ny, nx, param->in_2d, param->out_2d, FFTW_FORWARD, FFTW_MEASURE);
				fftwf_export_wisdom_to_filename(plan2DFN);
			}
			param->wisdom_string = fftwf_export_wisdom_to_string();
		} else {
			#pragma omp critical
			{
				int rc = fftwf_import_wisdom_from_string(param->wisdom_string);
				param->forward_plan_2d = fftwf_plan_dft_2d(ny, nx, param->in_2d, param->out_2d, FFTW_FORWARD, FFTW_WISDOM_ONLY);
			}
		}
	}
	memcpy(param->in_2d, data+1, nx*ny*sizeof(fftwf_complex));
	fftwf_execute(param->forward_plan_2d);
	memcpy(data+1, param->out_2d, nx*ny*sizeof(fftwf_complex));
}

void reconstruct (gridrecParams *param){
	memset (param->H, 0, (param->M+1)*(param->M+1)*sizeof(complex));
	phase1 (param);
	phase2 (param);
	phase3 (param);
	return;
}

void setGridRecPSWF (gridrecParams *param){
    param->pswf_db[0].C = 4.0;
    param->pswf_db[0].lmbda = 0.99588549;
    param->pswf_db[0].nt = 16;
    param->pswf_db[0].coefs[0] = 0.5239891E+01;
    param->pswf_db[0].coefs[1] = -0.5308499E+01;
    param->pswf_db[0].coefs[2] = 0.1184591E+01;
    param->pswf_db[0].coefs[3] = -0.1230763E-00;
    param->pswf_db[0].coefs[4] = 0.7371623E-02;
    param->pswf_db[0].coefs[5] = -0.2864074E-03;
    param->pswf_db[0].coefs[6] = 0.7789983E-05;
    param->pswf_db[0].coefs[7] = -0.1564700E-06;
    param->pswf_db[0].coefs[8] = 0.2414647E-08;
    param->pswf_db[0].coefs[9] = 0.0;
    param->pswf_db[0].coefs[10] = 0.0;
    param->pswf_db[0].coefs[11] = 0.0;
    param->pswf_db[0].coefs[12] = 0.0;
    param->pswf_db[0].coefs[13] = 0.0;
    param->pswf_db[0].coefs[14] = 0.0;
    param->pswf_db[1].C = 4.2;
    param->pswf_db[1].lmbda = 0.99657887;
    param->pswf_db[1].nt = 16;
    param->pswf_db[1].coefs[0] = 0.6062942E+01;
    param->pswf_db[1].coefs[1] = -0.6450252E+01;
    param->pswf_db[1].coefs[2] = 0.1551875E+01;
    param->pswf_db[1].coefs[3] = -0.1755960E-01;
    param->pswf_db[1].coefs[4] = 0.1150712E-01;
    param->pswf_db[1].coefs[5] = -0.4903653E-03;
    param->pswf_db[1].coefs[6] = 0.1464986E-04;
    param->pswf_db[1].coefs[7] = -0.3235110E-06;
    param->pswf_db[1].coefs[8] = 0.5492141E-08;
    param->pswf_db[1].coefs[9] = 0.0;
    param->pswf_db[1].coefs[10] = 0.0;
    param->pswf_db[1].coefs[11] = 0.0;
    param->pswf_db[1].coefs[12] = 0.0;
    param->pswf_db[1].coefs[13] = 0.0;
    param->pswf_db[1].coefs[14] = 0.0;
    param->pswf_db[2].C = 5.0;
    param->pswf_db[2].lmbda = 0.99935241;
    param->pswf_db[2].nt = 18;
    param->pswf_db[2].coefs[0] = 0.1115509E+02;
    param->pswf_db[2].coefs[1] = -0.1384861E+02;
    param->pswf_db[2].coefs[2] = 0.4289811E+01;
    param->pswf_db[2].coefs[3] = -0.6514303E-00;
    param->pswf_db[2].coefs[4] = 0.5844993E-01;
    param->pswf_db[2].coefs[5] = -0.3447736E-02;
    param->pswf_db[2].coefs[6] = 0.1435066E-03;
    param->pswf_db[2].coefs[7] = -0.4433680E-05;
    param->pswf_db[2].coefs[8] = 0.1056040E-06;
    param->pswf_db[2].coefs[9] = -0.1997173E-08;
    param->pswf_db[2].coefs[10] = 0.0;
    param->pswf_db[2].coefs[11] = 0.0;
    param->pswf_db[2].coefs[12] = 0.0;
    param->pswf_db[2].coefs[13] = 0.0;
    param->pswf_db[2].coefs[14] = 0.0;
    param->pswf_db[3].C = 6.0;
    param->pswf_db[3].lmbda = 0.9990188;
    param->pswf_db[3].nt = 18;
    param->pswf_db[3].coefs[0] = 0.2495593E+02;
    param->pswf_db[3].coefs[1] = -0.3531124E+02;
    param->pswf_db[3].coefs[2] = 0.1383722E+02;
    param->pswf_db[3].coefs[3] = -0.2799028E+01;
    param->pswf_db[3].coefs[4] = 0.3437217E-00;
    param->pswf_db[3].coefs[5] = -0.2818024E-01;
    param->pswf_db[3].coefs[6] = 0.1645842E-02;
    param->pswf_db[3].coefs[7] = -0.7179160E-04;
    param->pswf_db[3].coefs[8] = 0.2424510E-05;
    param->pswf_db[3].coefs[9] = -0.6520875E-07;
    param->pswf_db[3].coefs[10] = 0.0;
    param->pswf_db[3].coefs[11] = 0.0;
    param->pswf_db[3].coefs[12] = 0.0;
    param->pswf_db[3].coefs[13] = 0.0;
    param->pswf_db[3].coefs[14] = 0.0;
    param->pswf_db[4].C = 7.0;
    param->pswf_db[4].lmbda = 0.99998546;
    param->pswf_db[4].nt = 20;
    param->pswf_db[4].coefs[0] = 0.5767616E+02;
    param->pswf_db[4].coefs[1] = -0.8931343E+02;
    param->pswf_db[4].coefs[2] = 0.4167596E+02;
    param->pswf_db[4].coefs[3] = -0.1053599E+02;
    param->pswf_db[4].coefs[4] = 0.1662374E+01;
    param->pswf_db[4].coefs[5] = -0.1780527E-00;
    param->pswf_db[4].coefs[6] = 0.1372983E-01;
    param->pswf_db[4].coefs[7] = -0.7963169E-03;
    param->pswf_db[4].coefs[8] = 0.3593372E-04;
    param->pswf_db[4].coefs[9] = -0.1295941E-05;
    param->pswf_db[4].coefs[10] = 0.3817796E-07;
    param->pswf_db[4].coefs[11] = 0.0;
    param->pswf_db[4].coefs[12] = 0.0;
    param->pswf_db[4].coefs[13] = 0.0;
    param->pswf_db[4].coefs[14] = 0.0;
    param->SINE = NULL;
    param->COSE = NULL;
    param->cproj    = NULL;
    param->filphase = NULL;
    param->wtbl     = NULL;
    #ifdef INTERP
    param->dwtbl = NULL;
    #endif
    param->winv = NULL;
    param->work = NULL;
    param->H    = NULL;
    param->G1 = NULL;
    param->G2 = NULL;
    param->S1 = NULL;
    param->S2 = NULL;
}

void setSinoAndReconBuffers ( int  number, float *sinogram_address, float *reconstruction_address, gridrecParams *param){
    int     loop;
    param->sizeMatrices += (param->theta_list_size*sizeof(float *));
    param->sizeMatrices += (param->theta_list_size*sizeof(float *));
    param->sizeMatrices += (param->imgsiz*sizeof(float *));
    param->sizeMatrices += (param->imgsiz*sizeof(float *));
    //~ printf("G1 %ld\n",(long)(param->theta_list_size*sizeof(float *)));
    //~ printf("G2 %ld\n",(long)(param->theta_list_size*sizeof(float *)));
    //~ printf("S1 %ld\n",(long)(param->imgsiz*sizeof(float *)));
    //~ printf("S2 %ld\n",(long)(param->imgsiz*sizeof(float *)));
    if (param->G1 == NULL)  param->G1 = (float **) malloc((size_t) (param->theta_list_size * sizeof(float *)));
    if (param->G2 == NULL)  param->G2 = (float **) malloc((size_t) (param->theta_list_size * sizeof(float *)));
    if (param->S1 == NULL)  param->S1 = (float **) malloc((size_t) (param->imgsiz * sizeof(float *)));
    if (param->S2 == NULL)  param->S2 = (float **) malloc((size_t) (param->imgsiz * sizeof(float *)));
    if (number == 1){
        param->sinogram1 = sinogram_address;
        param->reconstruction1 = reconstruction_address;
        for (loop=0;loop<param->theta_list_size;loop++)
            param->G1[loop] = &param->sinogram1[loop*param->sinogram_x_dim];
        for (loop=0;loop<param->imgsiz;loop++)
            param->S1[loop] = &param->reconstruction1[loop*param->sinogram_x_dim];
    }
    if (number == 2){
        param->sinogram2 = sinogram_address;
        param->reconstruction2 = reconstruction_address;
        for (loop=0;loop<param->theta_list_size;loop++)
            param->G2[loop] = &param->sinogram2[loop*param->sinogram_x_dim];
        for (loop=0;loop<param->imgsiz;loop++)
            param->S2[loop] = &param->reconstruction2[loop*param->sinogram_x_dim];
    }
}

#ifdef INTERP
inline float Cnvlvnt(float X, gridrecParams *param){
	return (param->wtbl[(int)X]+(X-(int)X)*param->dwtbl[(int)X]);
}
#else
inline float Cnvlvnt(float X, gridrecParams *param){
	return (param->wtbl[(int)(X+0.5)]);
}
#endif

void four1(float data[], unsigned long nn, int isign, gridrecParams *param){
	int n = nn;
	if (n != param->n_prev){
		if (param->n_prev != 0) fftwf_free(param->in_1d);
		//~ printf("in_1d %ld\n",(long)(sizeof(fftwf_complex)*n));
		param->sizeMatrices += (long)(sizeof(fftwf_complex)*n);
		param->in_1d = fftwf_malloc(sizeof(fftwf_complex)*n);
		param->out_1d = param->in_1d;
		//~ printf("fft_test1f: creating plans, n=%d, n_prev=%d\n", n, param->n_prev);
		param->n_prev = n;
		if (param->setPlan == 1){
			int fftw1d_size = n;
			char plan1DFN[4096];
			sprintf(plan1DFN,"fftwf_wisdom_1d_%d.txt",(int)fftw1d_size);
			int rc = fftwf_import_wisdom_from_filename(plan1DFN);
			if (rc == 1){
				param->backward_plan_1d = fftwf_plan_dft_1d(n, param->in_1d, param->out_1d, FFTW_BACKWARD, FFTW_WISDOM_ONLY);
			} else {
				printf("Creating wisdom file. %s\n",plan1DFN);
				param->backward_plan_1d = fftwf_plan_dft_1d(n, param->in_1d, param->out_1d, FFTW_BACKWARD, FFTW_MEASURE);
				fftwf_export_wisdom_to_filename(plan1DFN);
			}
			param->wisdom_string = fftwf_export_wisdom_to_string();
		} else {
			#pragma omp critical
			{
				int rc = fftwf_import_wisdom_from_string(param->wisdom_string);
				param->backward_plan_1d = fftwf_plan_dft_1d(n, param->in_1d, param->out_1d, FFTW_BACKWARD, FFTW_WISDOM_ONLY);
			}
		}
	}
	memcpy(param->in_1d, data+1, n*sizeof(fftwf_complex));
	fftwf_execute(param->backward_plan_1d);
	memcpy(data+1, param->out_1d, n*sizeof(fftwf_complex));
}

void phase1 (gridrecParams *param){
	complex Cdata1, Cdata2, Ctmp;
	float U, V, rtmp, L2 = param->L/2.0, convolv, tblspcg = 2*param->ltbl/param->L;
	long pdim2=param->pdim>>1, M2=param->M>>1, iul, iuh, iu, ivl, ivh, iv, n;
	float offset=0.0;
	complex phfac;
	for (n=0;n<param->theta_list_size;n++){
		int    j,k;
		if (param->flag)  offset=(param->X0*param->COSE[n]+param->Y0*param->SINE[n])*PI;
		j=1;
		while (j < param->sinogram_x_dim+1){
			param->cproj[j].r=param->G1[n][j-1];
			param->cproj[j].i=param->G2[n][j-1];
			j++;
		}
		while (j < param->pdim){
			param->cproj[j].r = param->cproj[j].i = 0.0;
			j++;
		}
		four1 ((float *) param->cproj+1, param->pdim, 1,param);
		for (j=1;j<pdim2;j++){
			if (!param->flag){
				Ctmp.r=param->filphase[j].r;
				Ctmp.i=param->filphase[j].i;
			}else{
				phfac.r = cos(j*offset);
				phfac.i = -sin(j*offset);
				Cmult (Ctmp,param->filphase[j],phfac);
			}
			Cmult (Cdata1,Ctmp,param->cproj[j+1])
			Ctmp.i=-Ctmp.i;
			Cmult (Cdata2,Ctmp,param->cproj[(param->pdim-j)+1])
			U=(rtmp=param->scale*j)*param->COSE[n]+M2;
			V=rtmp*param->SINE[n]+M2;
			iul = (long int) (ceil(U-L2));
			iuh = (long int) (floor(U+L2));
			ivl = (long int) (ceil(V-L2));
			ivh = (long int) (floor(V+L2));
			if (iul<1 )  iul=1;
			if (iuh>=param->M)  iuh=param->M-1;
			if (ivl<1 )  ivl=1;
			if (ivh>=param->M)  ivh=param->M-1;
			for (iv=ivl,k=0;iv<=ivh;iv++,k++) param->work[k] = Cnvlvnt (abs (V-iv) * tblspcg,param);
			for (iu=iul;iu<=iuh;iu++){
				rtmp=Cnvlvnt (abs(U-iu)*tblspcg,param);
				for (iv=ivl,k=0;iv<=ivh;iv++,k++){
					convolv = rtmp*param->work[k];
					param->H[iu*param->M+iv+1].r += convolv*Cdata1.r;
					param->H[iu*param->M+iv+1].i += convolv*Cdata1.i;
					param->H[(param->M-iu)*param->M+(param->M-iv)+1].r += convolv*Cdata2.r;
					param->H[(param->M-iu)*param->M+(param->M-iv)+1].i += convolv*Cdata2.i;
				}
			}
		}
	}
}

void phase2 (gridrecParams *param){
	unsigned long   H_size[3];
	H_size[1] = H_size[2] = param->M;
	fourn ((float*) param->H+1, H_size, 2, -1, param);
}

void phase3 (gridrecParams *param){
	long iu, iv, j, k, ustart, vstart, ufin, vfin;
	float corrn_u, corrn;
	j=0;
	ustart=(param->M-param->M02);
	ufin=param->M;
	while (j<param->M0){
		for (iu=ustart;iu<ufin;j++,iu++){
			corrn_u=param->winv[j];
			k=0;
			vstart=(param->M-param->M02);
			vfin=param->M;
			while (k<param->M0){
				for (iv=vstart;iv<vfin;k++,iv++){
					corrn=corrn_u*param->winv[k];
					param->S1[j][k]=corrn*param->H[iu*param->M+iv+1].r;
					param->S2[j][k]=corrn*param->H[iu*param->M+iv+1].i;
				}
				if (k<param->M0)  (vstart=0,vfin=param->M02+1);
			}
		}
		if (j<param->M0)  (ustart=0,ufin=param->M02+1);
	}
}

void get_pswf (float C, pswf_struct **P, gridrecParams *param){
	int i=0;
	while ( (i<NO_PSWFS) && (fabs(C-param->pswf_db[i].C)>0.01) ){
		i++;
	}
	if (i>=NO_PSWFS){
		fprintf(stderr, "Prolate parameter, C = %f not in data base\n",C);
		exit(2);
	}
	*P = &param->pswf_db[i];
	return;
}

void getGridRecFourSizes (gridrecParams *param){
	float MaxPixSiz, D0, R, D1;
	long itmp;
	param->pdim = 1;
	itmp = param->sinogram_x_dim-1;
	while (itmp){
		param->pdim<<=1;
		itmp>>=1;
	}
	param->sampl = 1.0;
	MaxPixSiz = 1.0;
	R = 1.0;
	D0 = R*param->sinogram_x_dim;
	D1 = param->sampl*D0;
	itmp = (long int) (D1/MaxPixSiz-1);
	param->M = 1;
	while (itmp){
		param->M<<=1;
		itmp>>=1;
	}
}

void initGridRec (gridrecParams *param){
	float   center, C, MaxPixSiz, R, D0, D1;
	long    itmp;
	pswf_struct *pswf;
	center = param->sinogram_x_dim / 2;
	param->sampl     = 1.0;
	MaxPixSiz = 1.0;
	R         = 1.0;
	param->X0        = 0.0;
	param->Y0        = 0.0;
	param->ltbl      = 512;
	get_pswf (6.0, &pswf,param);
	C = pswf->C;
	param->flag = (param->X0!=0.0||param->Y0!=0.0) ? 1 : 0;
	param->pdim = 1;
	itmp = param->sinogram_x_dim-1;
	while (itmp){
		param->pdim<<=1;
		itmp>>=1;
	}
	D0 = R*param->sinogram_x_dim;
	D1 = param->sampl*D0;
	param->M = 1;
	itmp = (long int) (D1/MaxPixSiz-1);
	while (itmp){
		param->M<<=1;
		itmp>>=1;
	}
	param->M02 = (long int) (floor(param->M/2/param->sampl-0.5));
	param->M0 = 2*param->M02+1;
	param->sampl = (float)param->M/param->M0;
	D1    = param->sampl*D0;
	param->L     = 2*C*param->sampl/PI;
	param->scale = D1/param->pdim;
	param->sizeMatrices += ((param->pdim+1) * sizeof(complex));
	param->sizeMatrices += ((param->pdim+1) * sizeof(complex));
	param->sizeMatrices += ((param->ltbl+1) * sizeof(float));
	param->sizeMatrices += ((param->ltbl+1) * sizeof(float));
	param->sizeMatrices += (param->M0 * sizeof(float));
	param->sizeMatrices += (((int) param->L+1) * sizeof(float));
	param->sizeMatrices += ((param->M+1)*(param->M+1)*sizeof(complex));
	param->sizeMatrices += (param->theta_list_size * sizeof (float));
	param->sizeMatrices += (param->theta_list_size * sizeof (float));
	//~ printf("cproj %ld\n",(long)((param->pdim+1) * sizeof(complex)));
	//~ printf("filphase %ld\n",(long)((param->pdim+1) * sizeof(complex)));
	//~ printf("wtbl %ld\n",(long)((param->ltbl+1) * sizeof(float)));
	//~ printf("dwtbl %ld\n",(long)((param->ltbl+1) * sizeof(float)));
	//~ printf("winv %ld\n",(long)(param->M0 * sizeof(float)));
	//~ printf("work %ld\n",(long)(((int) param->L+1) * sizeof(float)));
	//~ printf("H %ld\n",(long)((param->M+1)*(param->M+1)*sizeof(complex)));
	//~ printf("SINE %ld\n",(long)(param->theta_list_size * sizeof (float)));
	//~ printf("COSE %ld\n",(long)(param->theta_list_size * sizeof (float)));
	param->cproj    = (complex *) malloc ((param->pdim+1) * sizeof(complex));
	param->filphase = (complex *) malloc (((param->pdim/2)+1) * sizeof(complex));
	param->wtbl     = (float   *) malloc ((param->ltbl+1) * sizeof(float));
	#ifdef INTERP
	param->dwtbl = (float *) malloc ((param->ltbl+1) * sizeof(float));
	#endif
	param->winv = (float *) malloc (param->M0 * sizeof(float));
	param->work = (float *) malloc (((int) param->L+1) * sizeof(float));
	param->H = (complex *) malloc ((param->M+1)*(param->M+1)*sizeof(complex));
	param->SINE = (float *) malloc (param->theta_list_size * sizeof (float));
	param->COSE = (float *) malloc (param->theta_list_size * sizeof (float));
	trig_su (0, param->theta_list_size,param);
	filphase_su (param->pdim, center,param);
	pswf_su (pswf, param->ltbl, param->M02, param->wtbl, param->dwtbl, param->winv,param);
	param->imgsiz = param->M0;
}

void trig_su (int geom, int n_ang, gridrecParams *param){
	int j;
	switch (geom){
		case 0 : {
			float   theta, degtorad = PI/180, *angle = param->theta_list;
			for (j=0;j<n_ang;j++){
				theta = degtorad*angle[j];
				param->SINE[j] = sin(theta);
				param->COSE[j] = cos(theta);
			}
		};break;
		case 1 :
		case 2 : {
			float   dtheta = geom*PI/n_ang, dcos, dsin;
			dcos = cos (dtheta);
			dsin = sin (dtheta);
			param->SINE[0] = 0.0;
			param->COSE[0] = 1.0;
			for(j=1;j<n_ang;j++){
				param->SINE[j] = dcos*param->SINE[j-1]+dsin*param->COSE[j-1];
				param->COSE[j] = dcos*param->COSE[j-1]-dsin*param->SINE[j-1];
			}
		}; break;
		default : {
		fprintf (stderr, "Illegal value for angle geometry indicator.\n");
		exit(2);
		}; break;
	}
}

void filphase_su (long  pd, float  center,gridrecParams *param){
	long    j, pd2=pd>>1;
	float   x, rtmp1=2*PI*center/pd, rtmp2;
	float   norm=PI/pd/param->theta_list_size;
	for (j=0;j<pd2;j++){
		x = j*rtmp1;
		rtmp2 = filterData ((float)j/pd,param) * norm;
		param->filphase[j].r = rtmp2*cos(x);
		param->filphase[j].i = -rtmp2*sin(x);
	}
}

void pswf_su (pswf_struct *pswf, long    ltbl, long    linv, float*  wtbl, float*  dwtbl, float*  winv, gridrecParams *param){
	float C, *coefs, lmbda, polyz, norm, fac;
	long i;
	int nt;
	C=pswf->C;
	nt=pswf->nt;
	coefs=pswf->coefs;
	lmbda=pswf->lmbda;
	polyz=legendre(nt,coefs,0.,param);
	wtbl[0]=1.0;
	for (i=1;i<=ltbl;i++){
		wtbl[i]=legendre(nt,coefs,(float)i/ltbl,param)/polyz;

		#ifdef INTERP
		dwtbl[i]=wtbl[i]-wtbl[i-1];
		#endif
	}
	fac=(float)ltbl/(linv+0.5);
	norm=sqrt (PI/2/C/lmbda)/param->sampl;
	winv[linv]=norm/Cnvlvnt(0.,param);
	for (i=1;i<=linv;i++){
		norm=-norm;
		winv[linv+i] = winv[linv-i] = norm/Cnvlvnt(i*fac,param);
	}
}

float legendre (  int  n, float *coefs, float  x, gridrecParams *param){
	float penult, last, newer, y;
	int j, k, even;
	if (x>1||x<-1){
		fprintf(stderr, "\nInvalid argument to legendre()");
		exit(2);
	}
	y=coefs[0];
	penult=1.;
	last=x;
	even=1;
	k=1;
	for (j=2;j<=n;j++){
		newer=(x*(2*j-1)*last-(j-1)*penult)/j;
		if (even){
			y+=newer*coefs[k];
			even=0;
			k++;
		}
		else{
			even=1;
		}
		penult=last;
		last=newer;
	}
	return y;
}
