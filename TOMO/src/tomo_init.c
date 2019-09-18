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
#include <stdbool.h>
#include <omp.h>
#include <unistd.h>
#include "tomo_heads.h"

/* 
 * The data can be one of two types: 
 * 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. 
 * 							dark, whites (2) and then raw images. Using number of angles, we know how many images are there. The scaling with white should be proportional to the distance from a white and appropriate dark value.
 * The functions assume one sinogram being processed for now with one fixed_shift.
*/

// TODO:
//	1. read Array of slices_to_process.
//	2. Make array with slices_to_process and shifts, it will be the (number of times we reconstruct)/2. Gridrec can process 2 slices at once.
//	3. set shift before calling the reconCentering function.
//	4. Set two slices when we call setSinoAndReconBuffers
//	5. Check difference between lines 1192 and 1198 in tomompi_client.cpp
//	6. Change output file to single file.
//	7. Safe malloc to check NULL ptrs.
//	8. Check memory usage and determine the number of processes we would be able to run per node.

void usage(){
	printf("MIDAS-TOMO Code to do tomo recon using Gridrec. Based on tomompi implementation from Brian Tiemann, APS. Maintained by Hemant Sharma, APS (hsharma@anl.gov).\nUsage is: \n"
		"tomo ParamsFile.txt numberOfParallelJobs\n"
		"Params file must have the following parameters:\n"
		"Input file is a text file name with a data link: sino data is a !!!single!!! binary file with darks, whites and tomo data in that order.\n"
		"* The rest of the file consists of the parameters required.\n"
		"* Parameters to be supplied:\n"
		"	* dataFileName: [char*] name of the file with the raw data or sino data\n"
		"	* reconFileName: [char*] Name of the file for saving the reconstruction\n"
		"	* areSinos: If the input is a sinogram instead of raw (cleaned) images [0 or 1]\n"
		"	* The data can be one of two types: \n"
		"	* 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. \n"
		"	* 							dark[float], whites (2,floats) and then raw images[shorts]. Using number of angles, we know how many images are there.\n"
		"	*							The scaling with white should be proportional to the distance from a white and appropriate dark value.\n"
		"	* detXdim - [uint]\n"
		"	* detYdim - [uint]\n"
		"	* Thetas can either be given as a range:\n"
		"	* 	thetaRange: startAngle endAngle angleInterval - [floats]\n"
		"	* or a File:\n"
		"	* 	thetaFileName [char*] with each line having an angle value [float].\n"
		"	* filter - [int] set to * 0: default\n"
		"							* 1: Shepp / Logan\n"
		"							* 2: Hann\n"
		"							* 3: Hamming\n"
		"							* 4: Ramp\n"
		"	* shiftValues: start_shift end_shift shift_interval [floats] In case of 1 shift, give start_shift=end_shift, shift_interval doesn't matter.\n"
		" 	*					ENSURE TO GIVE A RANGE WITH EVEN NUMBER OF SHIFTS\n"
		"	* ringRemovalCoefficient - If given, will do ringRemoval, otherwise comment or remove line [float] default 1.0\n"
		"	* slicesToProcess - -1 for all or FileName. ENSURE TO GIVE EVEN NUMBER OF SLICES\n"
		"Output file: float with reconstruction_xdim*reconstruction_xdim size\n"
		"OutputFileName: {recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_reconstruction_xdim_float_4byte.bin\n"
		"The code will generate two text files: fftwf_wisdom_{1,2}d.txt. "
		"These files are ways to speed up the fft calculation.\n"
		"First run on a dataset generates these files which can be used to speed up subsequent runs.\n");
}

int main(int argc, char *argv[])
{
	if (argc!=3){
		usage();
		return 1;
	}
	GLOBAL_CONFIG_OPTS recon_info_record;
	char *fileName;
	fileName = argv[1];
	int RC;
	RC = setGlobalOpts(fileName, &recon_info_record);
	setReadStructSize(&recon_info_record); // Also sets a couple of 
	if (RC!=0){
		printf("Parameter file could not be read. Exiting.\n");
		return 1;
	}
	recon_info_record.num_Jobs_Read = recon_info_record.n_slices;
	if (access ("fftwf_wisdom_2d.txt", F_OK) == -1){
		printf("FFT plan file did not exist, creating one.\n");
		createPlanFile(recon_info_record);
	} else if(access ("fftwf_wisdom_1d.txt", F_OK) == -1) {
		printf("FFT plan file did not exist, creating one.\n");
		createPlanFile(recon_info_record);
	}
	if (recon_info_record.n_shifts > 1 && recon_info_record.n_shifts %2 !=0){
		printf("Number of shifts must be even. Exiting\n");
		return 1;
	}
	if (recon_info_record.n_shifts == 1 && recon_info_record.n_slices %2 !=0){
		printf("Number of slices must be even. Exiting\n");
		return 1;
	}
	int numProcs = atoi(argv[2]);
	if (recon_info_record.n_shifts==1){
		int procNr;
		int nrSlicesThread = (int)ceil((double)recon_info_record.num_Jobs_Read / (2.0*(double)numProcs));
		printf("Number of FFT jobs per thread %d\n",nrSlicesThread);
		# pragma omp parallel num_threads(numProcs)
		{
			procNr = omp_get_thread_num();
			int startSliceNr = procNr*nrSlicesThread*2;
			int endSliceNr = startSliceNr + nrSlicesThread*2;
			if (endSliceNr > recon_info_record.num_Jobs_Read) endSliceNr = recon_info_record.num_Jobs_Read;
			printf("%d %d %d %d\n",procNr,startSliceNr,endSliceNr,-startSliceNr+endSliceNr);
			// Allocate all the structs and arrays now
			SINO_READ_OPTS readStruct;
			readStruct.norm_sino = (float *) malloc(sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
			LOCAL_CONFIG_OPTS information;
			information.shift = recon_info_record.shift_values[0];
			setSinoSize(&information,&recon_info_record);
			gridrecParams param;
			param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
			param.theta_list = recon_info_record.theta_list;
			param.filter_type = recon_info_record.filter;
			param.theta_list_size = recon_info_record.theta_list_size;
			size_t offt, offsetRecons;
			setGridRecPSWF(&param);
			initFFTMemoryStructures(&param);
			initGridRec(&param);
			int numSlice, sliceRowNr, oldSliceNr;
			for (numSlice = 0; numSlice<(endSliceNr-startSliceNr)/2; numSlice++){
				memset(readStruct.norm_sino,0,sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				memsets(&information,recon_info_record);
				int sliceNr;
				sliceRowNr = startSliceNr + numSlice*2;
				sliceNr = recon_info_record.slices_to_process[sliceRowNr];
				oldSliceNr = sliceNr;
				printf("Processing slices: %d %d\n",sliceNr,recon_info_record.slices_to_process[sliceRowNr+1]);
				if (recon_info_record.are_sinos){
					readSino(sliceNr,recon_info_record,&readStruct);
				} else {
					readRaw(sliceNr,recon_info_record,&readStruct);
				}
				memcpy(information.sino_calc_buffer,readStruct.norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = 0;
				offsetRecons = 0;
				reconCentering(&information,&recon_info_record,offt);
				setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				sliceRowNr ++;
				sliceNr = recon_info_record.slices_to_process[sliceRowNr];
				if (recon_info_record.are_sinos){
					readSino(sliceNr,recon_info_record,&readStruct);
				} else {
					readRaw(sliceNr,recon_info_record,&readStruct);
				}
				memcpy(information.sino_calc_buffer,readStruct.norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = information.sinogram_adjusted_size*2;
				offsetRecons = information.reconstruction_size*4;
				reconCentering(&information,&recon_info_record,offt);
				setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				reconstruct(&param);
				getRecons(&information,&recon_info_record,&param,0);
				writeRecon(oldSliceNr,&information,&recon_info_record);
				getRecons(&information,&recon_info_record,&param,offsetRecons);
				writeRecon(sliceNr,&information,&recon_info_record);
			}
			destroyFFTMemoryStructures(&param);
		}
	} else { // We have multiple shits, (possibly multiple slices_to_process)
		SINO_READ_OPTS readStruct[recon_info_record.n_slices];
		int i;
		for (i = 0; i < recon_info_record.n_slices; i ++)
			readStruct[i].norm_sino = (float *) malloc(sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
		// ReadStruct is now ready.
		int nJobs = (numProcs < recon_info_record.n_slices) ? numProcs : recon_info_record.n_slices;
		int procNr;
		# pragma omp parallel num_threads(nJobs)
		{
			procNr = omp_get_thread_num();
			int sliceNr;
			sliceNr = recon_info_record.slices_to_process[procNr];
			printf("Reading SliceNr: %d.\n",sliceNr);
			if (recon_info_record.are_sinos){
				readSino(sliceNr,recon_info_record,&readStruct[procNr]);
			} else {
				readRaw(sliceNr,recon_info_record,&readStruct[procNr]);
			}
		}
		nJobs = recon_info_record.n_slices * recon_info_record.n_shifts;
		int nrSlicesThread = (int)ceil((double)nJobs / (2.0*(double)numProcs));
		printf("Number of FFT jobs per thread %d\n",nrSlicesThread);
		# pragma omp parallel num_threads(numProcs)
		{
			procNr = omp_get_thread_num();
			int startJobNr, endJobNr;
			startJobNr = procNr*nrSlicesThread*2;
			endJobNr = (startJobNr + nrSlicesThread*2 > nJobs) ? startJobNr + nrSlicesThread*2 : nJobs;
			LOCAL_CONFIG_OPTS information;
			information.shift = recon_info_record.shift_values[0];
			setSinoSize(&information,&recon_info_record);
			gridrecParams param;
			param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
			param.theta_list = recon_info_record.theta_list;
			param.filter_type = recon_info_record.filter;
			param.theta_list_size = recon_info_record.theta_list_size;
			size_t offt, offsetRecons;
			setGridRecPSWF(&param);
			initFFTMemoryStructures(&param);
			initGridRec(&param);
			int jobNr, sliceNr, shiftNr, localSliceNr;
			for (jobNr = 0; jobNr < (endJobNr-startJobNr)/2; jobNr ++){
				memsets(&information,recon_info_record);
				sliceNr = (startJobNr + jobNr*2) / nJobs;
				shiftNr = (startJobNr + jobNr*2) % nJobs;
				localSliceNr = recon_info_record.slices_to_process[sliceNr];
				information.shift = recon_info_record.shift_values[shiftNr];
				printf("Processing slice: %d, shifts: %d %d for thread: %d\n",localSliceNr,recon_info_record.shift_values[shiftNr],recon_info_record.shift_values[shiftNr+1],procNr);
				memcpy(information.sino_calc_buffer,readStruct[localSliceNr].norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.theta_list_size);
				offt = 0;
				offsetRecons = 0;
				reconCentering(&information,&recon_info_record,offt);
				setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				information.shift = recon_info_record.shift_values[shiftNr+1];
				offt = information.sinogram_adjusted_size*2;
				offsetRecons = information.reconstruction_size*4;
				reconCentering(&information,&recon_info_record,offt);
				setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[offt], &information.reconstructions_boundary_padding[offsetRecons],&param);
				reconstruct(&param);
				information.shift = recon_info_record.shift_values[shiftNr];
				getRecons(&information,&recon_info_record,&param,0);
				writeRecon(localSliceNr,&information,&recon_info_record);
				information.shift = recon_info_record.shift_values[shiftNr+1];
				getRecons(&information,&recon_info_record,&param,offsetRecons);
				writeRecon(localSliceNr,&information,&recon_info_record);
			}
			destroyFFTMemoryStructures(&param);
		}
		
	}
	return 0;
}
