import subprocess
import sys, os
from pprint import pprint as print
import time
import argparse
import signal

default_handler = None

def handler(num, frame):    
	subprocess.call("rm -rf /dev/shm/*.bin",shell=True)
	print("Ctrl-C was pressed, cleaning up.")
	return default_handler(num, frame) 

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

default_handler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, handler)
parser = MyParser(description='''Far-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save results')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name')
parser.add_argument('-dataFN', type=str, required=False, default='', help='DataFileName')
parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPU cores to use')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
psFN = args.paramFN
dataFN = args.dataFN
numProcs = int(args.nCPUs)

os.makedirs(resultDir,exist_ok=True)

t0 = time.time()

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'

if len(dataFN)>0:
	strRun = f' -paramFN={psFN} -dataFN={dataFN} -resultFolder={resultDir}'
	print("Generating combined MIDAS file from HDF and ps files.")
else:
	strRun = f' -paramFN={psFN} -resultFolder={resultDir}'
	print("Generating combined MIDAS file from GE and ps files.")

f_ge2h5 = open(f'{resultDir}/ff2midas_out.csv','w')
f_ge2h5_err = open(f'{resultDir}/ff2midas_err.csv','w')
subprocess.call('python '+os.path.expanduser("~/opt/MIDAS/utils/ff2midas.py")+strRun,shell=True,stdout=f_ge2h5,stderr=f_ge2h5_err)
f_ge2h5.close()
f_ge2h5_err.close()
f_ge2h5 = open(f'{resultDir}/ff2midas_out.csv','r')
lines = f_ge2h5.readlines()
f_ge2h5.close()
for line in lines:
	if line.startswith('Out: '):
		outFStem = line.split()[1]
	if line.startswith('ResultDir: '):
		resultDir = line.split()[1]
print(f"Generating ZarrZip file. Time till now: {time.time()-t0}")
f_h52zarr = open(f'{resultDir}/midas2zip_out.csv','w')
f_h52zarr_err = open(f'{resultDir}/midas2zip_err.csv','w')
subprocess.call('python '+os.path.expanduser("~/opt/MIDAS/utils/midas2zip.py")+' '+outFStem,shell=True,stdout=f_h52zarr,stderr=f_h52zarr_err)
f_h52zarr.close()
f_h52zarr_err.close()
os.remove(f'{outFStem}')
print(f"Generating HKLs. Time till now: {time.time()-t0}")
f_hkls = open(f'{resultDir}/hkls_out.csv','w')
f_hkls_err = open(f'{resultDir}/hkls_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLListZarr")+' '+outFStem+'.zip',shell=True,env=env,stdout=f_hkls,stderr=f_hkls_err)
f_hkls.close()
f_hkls_err.close()
print(f"Doing PeakSearch. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/peaksearch_out.csv','w')
f_err = open(f'{resultDir}/peaksearch_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMPZarr")+' '+outFStem+'.zip 0 1 '+str(numProcs),shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Merging peaks. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/merge_overlaps_out.csv','w')
f_err = open(f'{resultDir}/merge_overlaps_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAllZarr")+' '+outFStem+'.zip',shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Calculating Radii. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/calc_radius_out.csv','w')
f_err = open(f'{resultDir}/calc_radius_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAllZarr")+' '+outFStem+'.zip',shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Transforming data. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/fit_setup_out.csv','w')
f_err = open(f'{resultDir}/fit_setup_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetupZarr")+' '+outFStem+'.zip',shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
os.chdir(resultDir)
print(f"Binning data. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/binning_out.csv','w')
f_err = open(f'{resultDir}/binning_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinData")+' paramstest.txt',shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
with open("SpotsToIndex.csv", "r") as f:
	num_lines = len(f.readlines())

print(f"Indexing. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/indexing_out.csv','w')
f_err = open(f'{resultDir}/indexing_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerOMP")+' paramstest.txt 0 1 '+str(num_lines)+' '+str(numProcs),shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Refining. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/refinement_out.csv','w')
f_err = open(f'{resultDir}/refinement_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP")+' paramstest.txt 0 1 '+str(num_lines)+' '+str(numProcs),shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Making grains list. Time till now: {time.time()-t0}")
f = open(f'{resultDir}/process_grains_out.csv','w')
f_err = open(f'{resultDir}/process_grains_err.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/ProcessGrainsZarr")+' '+outFStem+'.zip',shell=True,env=env,stdout=f,stdout=f_err)
f.close()
f_err.close()
print(f"Making plots, condensing output. Time till now: {time.time()-t0}")
subprocess.call('python '+os.path.expanduser('~/opt/MIDAS/utils/plotFFSpots3d.py')+' -resultFolder '+resultDir,shell=True)
subprocess.call('python '+os.path.expanduser('~/opt/MIDAS/utils/plotFFSpots3dGrains.py')+' -resultFolder '+resultDir,shell=True)
subprocess.call('python '+os.path.expanduser('~/opt/MIDAS/utils/plotGrains3d.py')+' -resultFolder '+resultDir,shell=True)
print(f"Done. Total time elapsed: {time.time()-t0}")