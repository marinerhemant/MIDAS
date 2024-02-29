import subprocess
import sys, os

psFN = sys.argv[1]
numProcs = sys.argv[2]
f_ge2h5 = open('ge2h5_out.csv','w')
subprocess.call('python '+os.path.expanduser("~/opt/MIDAS/utils/ge2h5.py")+' '+psFN,shell=True,stdout=f_ge2h5)
f_ge2h5.close()
f_ge2h5 = open('ge2h5_out.csv','r')
lines = f_ge2h5.readlines()
f_ge2h5.close()
for line in lines:
    if line.startswith('Out: '):
        outFStem = line.split()[1]
    if line.startswith('ResultDir: '):
        resultDir = line.split()[1]

f_h52zarr = open('h5_to_zarr_zip.csv','w')
subprocess.call('python '+os.path.expanduser("~/opt/MIDAS/utils/h5_to_zarr_zip.py")+' '+outFStem+'.h5',shell=True,stdout=f_h52zarr)
f_h52zarr.close()

f_hkls = open('hkls_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLListZarr")+' '+outFStem+'.zip',shell=True,stdout=f_hkls)
f_hkls.close()
f = open('peaksearch_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMPZarr")+' '+outFStem+'.zip 0 1 '+str(numProcs),shell=True,stdout=f)
f.close()
f = open('merge_overlaps_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAllZarr")+' '+outFStem+'.zip',shell=True,stdout=f)
f.close()
f = open('calc_radius_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAllZarr")+' '+outFStem+'.zip',shell=True,stdout=f)
f.close()
f = open('fit_setup_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetupZarr")+' '+outFStem+'.zip',shell=True,stdout=f)
f.close()
os.chdir(resultDir)
f = open('binning_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinData")+' paramstest.txt',shell=True,stdout=f)
f.close()
with open("SpotsToIndex.csv", "r") as f:
    num_lines = len(f.readlines())
f = open('indexing_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerOMP")+' paramstest.txt 0 1 '+str(num_lines)+' '+str(numProcs),shell=True,stdout=f)
f.close()
f = open('refinement_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP")+' paramstest.txt 0 1 '+str(num_lines)+' '+str(numProcs),shell=True,stdout=f)
f.close()
f = open('process_grains_out.csv','w')
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/ProcessGrainsZarr")+' '+outFStem+'.zip',shell=True,stdout=f)
f.close()