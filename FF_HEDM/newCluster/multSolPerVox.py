import glob
import sys,os
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import iradon
from PIL import Image
import subprocess
import warnings
warnings.filterwarnings('ignore')
import argparse

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def runReconMulti(folder,nScans,positions,sgnum,numProcs,maxang=1,nNodes=1,machineName='local'):
	nVoxels = nScans*nScans
	files = glob.glob(folder+'/Output/*.csv')
	xpos, ypos = np.meshgrid(positions,positions)
	xpositions = np.transpose(np.transpose(xpos).reshape((nScans*nScans)))
	ypositions = np.transpose(np.transpose(ypos).reshape((nScans*nScans)))
	uniqueArr = np.zeros((nVoxels,3))
	uniqueOrientArr = []
	for voxNr in range(nVoxels):
		# find all files with that blurb
		blurb = '_'+str.zfill(str(voxNr),6)+'_'
		fns = [fn for fn in files if blurb in fn]
		print(blurb,len(fns))
		if len(fns) == 0:
			continue
		PropList = []
		highestConf = -1
		for fn in fns:
			f = open(fn)
			str1= f.readline()
			str1= f.readline()
			line = f.readline().split()
			f.close()
			IAthis = float(line[0][:-1])
			OMthis = [float(a[:-1]) for a in line[1:10]]
			nExp = float(line[-2][:-1])
			nObs = float(line[-1][:-1])
			ConfThis = nObs/nExp
			idnr = int((fn.split('.')[-2]).split('_')[-1])
			if ConfThis > highestConf:
				highestConf = ConfThis
			PropList.append([ConfThis,IAthis,OMthis,idnr,voxNr])
		sortedPropList = sorted(PropList,key=lambda x: (x[0],x[1]),reverse=True)
		uniqueArr[voxNr][0] = xpositions[voxNr]
		uniqueArr[voxNr][1] = ypositions[voxNr]
		marked = np.zeros(len(sortedPropList))
		uniqueOrients = []
		for idx in range(len(sortedPropList)):
			if marked[idx] == 1:
				continue
			else:
				val1 = sortedPropList[idx]
				uniqueOrients.append(val1)
				orient1 = val1[2]
				for idx2 in range(idx+1,len(sortedPropList)):
					if marked[idx2] == 1:
						continue
					orient2 = sortedPropList[idx2][2]
					ang = rad2deg*GetMisOrientationAngleOM(orient1,orient2,sgnum)[0]
					if ang < maxang:
						marked[idx2] = 1
		print(['VoxelNr:',voxNr,'nSols:',len(fns),'nUniqueSols:',len(uniqueOrients)])
		uniqueArr[voxNr][2] = len(uniqueOrients)
		uniqueOrientArr.append(uniqueOrients)

	# Generate SpotsToIndex.csv file with all jobs to do.
	IDsToDo = [orient2[3] for orient in uniqueOrientArr for orient2 in orient]
	IDsToDo2 = [orient2[4] for orient in uniqueOrientArr for orient2 in orient]
	nIDs = len(IDsToDo)
	with open(folder+'/SpotsToIndex.csv','w') as SpotsF:
		for nr in range(nIDs):
			SpotsF.write(str(IDsToDo[nr])+' '+str(IDsToDo2[nr])+'\n')

	# Run FitOrStrainsScanning
	os.makedirs('Results',exist_ok=True)
	swiftcmdIdx = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') + ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runRefiningScanning.swift') + ' -folder=' + folder + ' -nrNodes=' + str(nNodes) + ' -nScans=' + str(nScans) + ' -numProcs='+ str(numProcs)
	print(swiftcmdIdx)
	subprocess.call(swiftcmdIdx,shell=True)
	# subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitOrStrainsScanningOMP")+' paramstest.txt 0 1 '+ str(nIDs)+' '+str(numProcs),shell=True)

	NrSym,Sym = MakeSymmetries(sgnum)
	# go through the output
	files2 = glob.glob(folder+'/Results/*.csv')
	filesdata = np.zeros((len(files2),43))
	i=0
	for fileN in files2:
		f = open(fileN)
		str1 = f.readline()
		data = f.readline().split()
		for j in range(len(data)):
			filesdata[i][j] = float(data[j])
		OM = filesdata[i][1:10]
		quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM),NrSym,Sym)
		filesdata[i][39:43] = quat
		i+=1
		f.close()
	np.savetxt('microstrFull.csv',filesdata,fmt='%.6f',delimiter=',',header='SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4')

if __name__ == "__main__":
	parser = MyParser(description='''Find multiple solutions per voxel, then refine''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-resultFolder', type=str, required=True, help='Working folder')
	parser.add_argument('-nScans', type=int, required=True, help='Number of scans')
	parser.add_argument('-positionsFile', type=str, required=False, default='positions.csv', help='Positions filename')
	parser.add_argument('-sgNum', type=int, required=True, help='SpaceGroup number')
	parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPU cores to use')
	parser.add_argument('-maxAng', type=float, required=False, default=1, help='Maximum angle in degrees to qualify as a different orientation.')
	parser.add_argument('-nNodes', type=int, required=False, default=1, help='Number of nodes to use, if not using local.')
	parser.add_argument('-machineName', type=str, required=False, default='local', help='Machine to run analysis on.')
	args, unparsed = parser.parse_known_args()
	resultDir = args.resultFolder
	nScans = args.nScans
	pos = args.positionsFile
	positions = np.genfromtxt(pos)
	sgnum = args.sgNum
	numProcs = args.nCPUs
	maxang = args.maxAng
	nNodes = args.nNodes
	machineName = args.machineName
	runReconMulti(resultDir,nScans,positions,sgnum,numProcs,maxang=maxang,nNodes=nNodes,machineName=machineName)


