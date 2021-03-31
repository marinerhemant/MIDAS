import ctypes
import argparse
import warnings
import time
import os
import subprocess

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_PeaksFitting''', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-paramFile',     type=str, required=True, help='Parameter File')
# ~ parser.add_argument('-startLayer',     type=str, required=True, help='Start layer number to process')
# ~ parser.add_argument('-endLayer', type=str, required=True, help='End layer number to process')
parser.add_argument('-nNodes',    type=int, required=True, help='Number of CPUs to use')
# ~ parser.add_argument('-machineName', type=str, required=True, help='End layer number to process')
args, unparsed = parser.parse_known_args()

paramFN = args.paramFile

def getParameters(ParamFileName):
	paramContents = open(ParamFileName).read()
	params = {}
	params["RingNrs"] = []
	params["Thresholds"] = []
	params["nRings"] = 0
	for line in paramContents.split("\n"):
		if line.startswith("EndNr"):
			params["EndNr"] = int(line.split()[1])
		if line.startswith("StartNr"):
			params["StartNr"] = int(line.split()[1])
		if line.startswith("RingThresh"):
			params["RingNrs"].append(int(line.split()[1]))
			params["Thresholds"].append(float(line.split()[2]))
			params["nRings"] += 1
	params["nFrames"] = params["EndNr"] - params["StartNr"] + 1
	print(params)
	return params

params = getParameters(paramFN)
blockNr = 0
numProcs = 64
nBlocks = int(args.nNodes)
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMP ")+paramFN+' '+str(blockNr)+' '+str(nBlocks)+' '+str(params['nFrames'])+' '+str(numProcs),shell=True)
