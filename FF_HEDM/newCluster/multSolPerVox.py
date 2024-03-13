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

def runReconMulti(folder,nScans,positions,sgnum,numProcs,numProcsLocal,maxang=1,nNodes=1,machineName='local'):

	subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/findMultipleSolutionsPF')+f' {folder} {sgnum} {maxang} {nScans} {numProcsLocal}',shell=True,cwd=folder)
	# Run FitOrStrainsScanning
	IDs = np.genfromtxt(f"{folder}/SpotsToIndex.csv")
	nIDs = IDs.shape[0]
	os.makedirs('Results',exist_ok=True)
	swiftcmdIdx = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') 
	swiftcmdIdx += ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runRefiningScanning.swift') + ' -folder=' 
	swiftcmdIdx += folder + ' -nrNodes=' + str(nNodes) + ' -nIDs=' + str(nIDs) + ' -numProcs='+ str(numProcs)
	print(swiftcmdIdx)
	subprocess.call(swiftcmdIdx,shell=True)

	NrSym,Sym = MakeSymmetries(sgnum)
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
	runReconMulti(resultDir,nScans,positions,sgnum,numProcs,numProcsLocal=3,maxang=maxang,nNodes=nNodes,machineName=machineName)


