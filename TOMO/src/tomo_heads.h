//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef tomo_headsH
#define tomo_headsH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <sys/stat.h>
#include <fftw3.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef uint
	typedef unsigned int uint; // for compatibility with C++ code
#endif

//--------------------------------------------------------------------------------------------------------------------------
// Parameters for gridrec
#define max(A,B) ((A)>(B)?(A):(B))
#define min(A,B) ((A)<(B)?(A):(B))
#define abs(A) ((A)>0 ?(A):-(A))
#define Cmult(A,B,C) {(A).r=(B).r*(C).r-(B).i*(C).i; (A).i=(B).r*(C).i+(B).i*(C).r;}
#define TOLERANCE 0.1
#define LTBL_DEF 512
#define NO_PSWFS 5
#define FILTER_NONE 0
#define FILTER_SHEPP_LOGAN 1
#define FILTER_HANN 2
#define FILTER_HAMMING 3
#define FILTER_RAMP 4
#define MAX_N_THETAS 36000

typedef struct PSWF_STRUCT {
	float C,
		lmbda;
	int nt;
	float coefs[15];
} pswf_struct;

typedef struct {
	float r,
		i;
} complex;

typedef struct {
	long pdim,
		M,
		M0,
		M02,
		ltbl,
		imgsiz;
	float sampl,
		scale,
		L,
		X0,
		Y0,
		*SINE,
		*COSE,
		*wtbl,
		*dwtbl,
		*work,
		*winv,
		**G1,
		**G2,
		**S1,
		**S2,
		*sinogram1,
		*sinogram2,
		*reconstruction1,
		*reconstruction2,
		*theta_list;
	complex *cproj,
		*filphase,
		*H;
	pswf_struct pswf_db[NO_PSWFS];
	int flag,
		theta_list_size,
		filter_type,
		n_prev,
		nx_prev,
		ny_prev,
		setPlan;
	unsigned long sinogram_x_dim;
	fftwf_complex *in_1d,
		*out_1d;
	fftwf_plan backward_plan_1d,
		forward_plan_2d;
	fftwf_complex 	*in_2d,
		*out_2d;
	char *wisdom_string;
	long sizeMatrices;
} gridrecParams;

// Functions
inline float Cnvlvnt(float X, gridrecParams *param);
void phase1 (gridrecParams *param);
void phase2 (gridrecParams *param);
void phase3 (gridrecParams *param);
void trig_su (int geom, int n_ang, gridrecParams *param);
void filphase_su (long pd, float center, gridrecParams *param);
void pswf_su (pswf_struct *pswf, long ltbl, long linv, float* wtbl, float* dwtbl, float* winv, gridrecParams *param);
float legendre (int n, float *coefs, float x, gridrecParams *param);
void get_pswf (float C, pswf_struct **P, gridrecParams *param);
void setSinoAndReconBuffers (int   number, float *sinogram_address, float *reconstruction_address, gridrecParams *param);
float filterData (float x, gridrecParams *param);
float shlo (float x);
float hann (float x);
float hamm (float x);
float ramp (float x);
void reconstruct (gridrecParams *param);
void initGridRec (gridrecParams *param);
void getGridRecFourSizes (gridrecParams *param);
//--------------------------------------------------------------------------------------------------------------------------
// FFTW
void four1(float data[], unsigned long nn, int isign, gridrecParams *param);
void fourn(float data[], unsigned long nn[], int ndim, int isign, gridrecParams *param);
void initFFTMemoryStructures (gridrecParams *param);
void destroyFFTMemoryStructures (gridrecParams *param);

//--------------------------------------------------------------------------------------------------------------------------
// ConfigurationParameters
typedef struct {
	uint det_xdim,
		det_ydim,
		*slices_to_process;
	bool are_sinos,
		auto_centering,
		use_ring_removal;
	float start_angle,
		end_angle,
		angle_interval,
		*shift_values,
		ring_removal_coeff,
		*theta_list,
		start_shift,
		end_shift,
		shift_interval;
	char DataFileName[4096],
		ReconFileName[4096],
		SliceFileName[4096],
		thetaFileName[4096];
	int sinogram_xdim,
		sinogram_ydim,
		reconstruction_xdim,
		reconstruction_ydim,
		theta_list_size,
		n_shifts,
		n_slices,
		filter,
		debug;
	int sinogram_adjusted_xdim,
		reconstruction_size,
		sinogram_adjusted_size;
	char *wisdom_string;
	int saveReconSeparate;
	int powerIncrement;
	int doLogProj;
	long sizeMatrices;
}GLOBAL_CONFIG_OPTS;

typedef struct {
	int sinogram_adjusted_xdim,
		reconstruction_size,
		sinogram_adjusted_size;
	float *sino_calc_buffer,
		*recon_calc_buffer,
		*shifted_recon,
		*shifted_sinogram,
		*sinograms_boundary_padding,
		*reconstructions_boundary_padding,
		*mean_vect,
		*low_pass_sino_lines_data,
		*mean_sino_line_data,
		shift;
} LOCAL_CONFIG_OPTS;

typedef struct {
	unsigned short int *short_sinogram;
	float *norm_sino,
		*init_sinogram,
		*white_field_sino,
		*dark_field_sino_ave;
	long sizeMatrices;
} SINO_READ_OPTS;

//--------------------------------------------------------------------------------------------------------------------------
// Initiate Config Opts Structs
int setGlobalOpts(char inputFile[], GLOBAL_CONFIG_OPTS *recon_info_record);
void setSinoSize (LOCAL_CONFIG_OPTS *information, GLOBAL_CONFIG_OPTS recon_info_record);
void setReadStructSize (GLOBAL_CONFIG_OPTS *recon_info_record);
void memsets(LOCAL_CONFIG_OPTS *information, GLOBAL_CONFIG_OPTS recon_info_record);
void setGridRecPSWF (gridrecParams *param);

//--------------------------------------------------------------------------------------------------------------------------
// ReadData
int readSino(int sliceNr,GLOBAL_CONFIG_OPTS recon_info_record, SINO_READ_OPTS *readStruct);
int readRaw(int sliceNr,GLOBAL_CONFIG_OPTS recon_info_record,SINO_READ_OPTS *readStruct);

//--------------------------------------------------------------------------------------------------------------------------
// Corrections
void RingCorrectionSingle (float *data, float ring_coeff, LOCAL_CONFIG_OPTS *information, GLOBAL_CONFIG_OPTS recon_info_record);
void LogSinogram (float *data, int xdim, int ydim);
void LogProj(float *data, int xdim, int ydim);
void Normalize (SINO_READ_OPTS *readStruct, GLOBAL_CONFIG_OPTS recon_info_record);
void Pad (SINO_READ_OPTS *readStruct, GLOBAL_CONFIG_OPTS recon_info_record);

//--------------------------------------------------------------------------------------------------------------------------
// Processing code
void reconCentering(LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS recon_info_record,size_t offt,int doLog);
void getRecons(LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS recon_info_record,gridrecParams *param,size_t offsetRecons);
int writeRecon(int sliceNr,LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS recon_info_record,int shiftNr);
void createPlanFile(GLOBAL_CONFIG_OPTS *recon_info_record);

#endif
