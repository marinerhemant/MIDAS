import numpy as np
from math import pow
import subprocess

def run_tomo(data,workingdir,thetas,filterNr,shifts,doLog,extraPad,autoCentering,numCPUs):
	# data (one dark, 2 whites and data floats, tilt corrected projections)
	# workingdir
	# thetas (array)
	# filterNr: [2-default] 0 (nothing),1(shepp/logan),2(hann),3(hamming),4(ramp)
	# shiftValue (pixels) # single or start end interval array
	# doLog (1,0)
	# extraPad (1,0)
	# autocentering (1,0)
	infn = workingdir+'/input.bin'
	data = data.astype(np.float32)
	data.tofile(infn)
	# We have tilt corrected projections, one dark and two whites in the beginning.
	xDim,nrSlices,nrThetas = data.shape
	nrThetas -= 2
	outfnstr = workingdir+'/output'
	still_smaller = True
	power = 0
	while (still_smaller):
		if (xDim > pow (2, power)):
			power+= 1
			still_smaller = True
		else:
			still_smaller = False
	if (xDim == pow (2, power)):
		xDimNew = int(xDim)
	else:
		xDimNew = int(pow(2,power))
	if (extraPad==1):
		power+=1
		xDimNew = int(pow(2,power))
	thetasFile = open(workingdir+'/midastomo_thetas.txt','w')
	for theta in thetas: thetasFile.write(str(theta)+'\n')
	# Write the config to a config file
	configFile = open(workingdir+'/midastomo.par','w')
	configFile.write('saveReconSeparate 0\n')
	configFile.write('dataFileName '+infn+'\n')
	configFile.write('reconFileName '+outfnstr+'\n')
	configFile.write('areSinos 1\n')
	configFile.write('detXdim '+str(xDim)+'\n')
	configFile.write('detYdim '+str(nrSlices)+'\n')
	configFile.write('thetaFileName '+workingdir+'/midastomo_thetas.txt\n')
	if len(shifts) == 1:
		configFile.write('shiftValues '+str(shifts)+'\n')
		nrShifts = 1
	else:
		nrShifts = round(abs((shifts[1]-shifts[0]))/shifts[2])+1
		configFile.write('shiftValues '+str(shifts[0])+' '+str(shifts[1])+' '+str(shifts[2])+'\n')
	configFile.write('ringRemovalCoefficient 0\n')
	configFile.write('doLog '+str(doLog)+'\n')
	configFile.write('slicesToProcess -1\n')
	configFile.write('ExtraPad '+str(extraPad)+'\n')
	configFile.write('AutoCentering '+str(autoCentering)+'\n')
	configFile.close()
	# Run tomo
	subprocess.call(os.path.expanduser("~/opt/MIDAS/TOMO/bin/MIDASTOMO")+" "+workingdir+'/midastomo.par '+str(numCPUs),shell=True)
	# Read result
	outfn = outfnstr+'_NrSlices_'+str(nrSlices).zfill(5)+'_NrShifts_'+str(nrShifts).zfill(3)+
			'_XDim_'+str(xDimNew).zfill(6)+'_YDim_'+str(xDimNew).zfill(6)+'_float32.bin'
	return np.fromfile(outfn,dtype=np.float32,count=(nrSlices*nrShifts*xDimNew*xDimNew)).reshape((nrSlices,nrShifts,xDimNew,xDimNew))

