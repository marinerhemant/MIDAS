from parsl.app.app import python_app
import subprocess
import os, sys
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)

def generateZip(resFol,pfn,layerNr,dfn='',dloc='',nchunks=-1,preproc=-1,outf='ZipOut.txt',errf='ZipErr.txt'):
    cmd = 'python '+os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')+' -resultFolder '+ resFol +' -paramFN ' + pfn + ' -LayerNr ' + str(layerNr)
    if dfn!='':
        cmd+= ' -dataFN ' + dfn
    if dloc!='':
        cmd+= ' -dataLoc ' + dloc
    if nchunks!=-1:
        cmd+= ' -numFrameChunks '+str(nchunks)
    if preproc!=-1:
        cmd+= ' -preProcThresh '+str(preproc)
    outf = resFol+'/'+outf
    errf = resFol+'/'+errf
    subprocess.call(cmd,shell=True,stdout=open(outf,'w'),stderr=open(errf,'w'))
    lines = open(outf,'r').readlines()
    if lines[-1].startswith('OutputZipName'):
        return lines[-1].split()[1]

@python_app
def peaks(resultDir,zipFN,numProcs,hkls_err,blockNr=0,numBlocks=1):
    import subprocess
    import os
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    f = open(f'{resultDir}/peaksearch_out.csv','w')
    f_err = open(f'{resultDir}/peaksearch_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMPZarr")+f' {zipFN} {blockNr} {numBlocks} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

@python_app
def index(resultDir,numProcs,bin_err,blockNr=0,numBlocks=1):
    import subprocess
    import os
    os.chdir(resultDir)
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    with open("SpotsToIndex.csv", "r") as f:
        num_lines = len(f.readlines())
    f = open(f'{resultDir}/indexing_out.csv','w')
    f_err = open(f'{resultDir}/indexing_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerOMP")+f' paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

@python_app
def refine(resultDir,numProcs,bin_err,blockNr=0,numBlocks=1):
    import subprocess
    import os
    os.chdir(resultDir)
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'
    with open("SpotsToIndex.csv", "r") as f:
        num_lines = len(f.readlines())
    f = open(f'{resultDir}/refining_out.csv','w')
    f_err = open(f'{resultDir}/refining_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP")+f' paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
