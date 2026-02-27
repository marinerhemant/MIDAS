import os, subprocess
import glob
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import midas_config
midas_config.run_startup_checks()
# ~ from calcMiso import *

# ~ nScans = 21
nScans = 105
# ~ sgnum = 225
# ~ nCPUs = 64
# ~ folder = '/data/tomo1/sharma_internal_hedm/hpldrd_dec21/L2/'
folder = '/data/tomo1/kenesei_nov20_midas/buscek/all_layers/'
# ~ maxAng = 0.5


positions = np.genfromtxt(folder+'positions.csv')
positions = positions[:nScans]
xpos, ypos = np.meshgrid(positions,positions)
xpositions = np.transpose(np.transpose(xpos).reshape((nScans*nScans)))
ypositions = np.transpose(np.transpose(ypos).reshape((nScans*nScans)))

# Files should be arranged as follows: InputAllExtraInfoFittingAll0.csv, Inp....1.csv, Inp....2.csv and so on
# positions.csv file should be present in the same folder with 1 row per position of the beam.

# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinDataScanning")+' '+str(nScans),shell=True)
# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerScanningOMP")+' 0 1 '+ str(nScans)+' '+str(nCPUs),shell=True)

# Now read all the files in the results folder, then find all files for a certain voxel, then find all the unique orientations, find the best out of those and save in a single file.
nVoxels = nScans*nScans
bestCoordsList = []
files = glob.glob(folder+'Output/*.csv')
uniqueArr = np.zeros((nVoxels,3))
uniqueOrientArr = []
for voxNr in range(nVoxels):
	# find all files with that blurb
	blurb = '_'+str.zfill(str(voxNr),6)+'_'
	fns = [fn for fn in files if blurb in fn]
	PropList = []
	highestConf = -1
	for fn in fns:
		f = open(fn)
		str1= f.readline()
		str1= f.readline()
		line = f.readline().split()
		IAthis = float(line[0][:-1])
		OMthis = [float(a[:-1]) for a in line[1:10]]
		nExp = float(line[-2][:-1])
		nObs = float(line[-1][:-1])
		ConfThis = nObs/nExp
		idnr = int((fn.split('.')[-2]).split('_')[-1])
		# ~ print(idnr)
		if ConfThis > highestConf:
			highestConf = ConfThis
		PropList.append([ConfThis,IAthis,OMthis,idnr])
	sortedPropList = sorted(PropList,key=lambda x: x[0],reverse=True)
	# ~ # sortedPropList now has all the orientations, we can try to find the unique ones
	# ~ # starting with first orientation, compute miso with all next (unmarked) orientations, if angle is smaller than maxAng, mark the orientation.
	# ~ marked = np.zeros(len(sortedPropList))
	# ~ uniqueOrients = []
	# ~ for idx in range(len(sortedPropList)):
		# ~ if marked[idx] == 1:
			# ~ continue
		# ~ else:
			# ~ val1 = sortedPropList[idx]
			# ~ uniqueOrients.append(val1)
			# ~ orient1 = val1[2]
			# ~ for idx2 in range(idx+1,len(sortedPropList)):
				# ~ if marked[idx2] == 1:
					# ~ continue
				# ~ orient2 = sortedPropList[idx2][2]
				# ~ ang = GetMisOrientationAngleOM(orient1,orient2,sgnum)
				# ~ if ang*rad2deg < maxAng:
					# ~ marked[idx2] = 1
	# ~ print(['VoxelNr:',voxNr,'nSols:',len(fns),'nUniqueSols:',len(uniqueOrients)])
	# ~ uniqueArr[voxNr][0] = xpositions[voxNr]
	# ~ uniqueArr[voxNr][1] = ypositions[voxNr]
	# ~ uniqueArr[voxNr][2] = len(uniqueOrients)
	# ~ uniqueOrientArr.append(uniqueOrients)
	minIA = 1000
	bestVal = []
	for val in sortedPropList:
		if val[0] < highestConf:
			break
		if val[1] < minIA:
			minIA = val[1]
			bestVal = val
	if len(bestVal) > 0:
		bestCoordsList.append([xpositions[voxNr],ypositions[voxNr],bestVal])
	# ~ if len(bestVal) > 0:
		# ~ print(BringDownToFundamentalRegionSym(OrientMat2Quat(bestVal[2]),12,HexSym),highestConf)

finXpos = [ i[0] for i in bestCoordsList]
finYpos = [ i[1] for i in bestCoordsList]
confs = [ i[2][0] for i in bestCoordsList]
finOr = np.array([ BringDownToFundamentalRegionSym(OrientMat2Quat(i[2][2]),12,HexSym) for i in bestCoordsList ])
outarr = np.array(list(zip(finXpos,finYpos,confs,finOr[:,0],finOr[:,1],finOr[:,2],finOr[:,3])))
np.savetxt('microstrTr.csv',outarr,fmt='%.6f',delimiter=',',header='xPos,yPos,confidence,Quat0,Quat1,Quat2,Quat3')
# ~ uniqueArr = uniqueArr[uniqueArr[:,2] > 0,:]
# ~ totalGrains = np.sum(uniqueArr[:,2])
# ~ plt.scatter(uniqueArr[:,0],uniqueArr[:,1],s=300,c=uniqueArr[:,2],cmap=plt.get_cmap('jet')); plt.gca().set_aspect('equal'); plt.colorbar(); plt.show()
plt.scatter(outarr[:,0],outarr[:,1],s=300,c=outarr[:,2],cmap=plt.get_cmap('jet')); plt.gca().set_aspect('equal'); plt.colorbar(); plt.show()
