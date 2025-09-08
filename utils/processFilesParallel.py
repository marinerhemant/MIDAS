from multiprocessing import Pool
import sys, argparse, os
import subprocess

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def runSingle(workData):
    rf = workData[0]
    pf = workData[1]
    lN = workData[2]
    nc = workData[3]
    fNr = overAllFNr + (lN-1)*numFilesPerScan
    rf += f'/{fNr}'
    cwd = os.getcwd()
    os.makedirs(rf,exist_ok=True)
    binloc = os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZipRefactor.py')
    cmd = f'{sys.executable} {binloc} -resultFolder {rf} -paramFN {pf} -LayerNr {lN} -numFrameChunks {nc}'
    if (numFilesPerScan!=1):
        cmd += f' -numFilesPerScan {numFilesPerScan}'
    if (preProc!=-1):
        cmd += f' -preProcThresh {preProc}'
    print(cmd)
    subprocess.call(cmd,cwd=cwd,shell=True)

parser = MyParser(description='''processFilesParallel.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save results')
parser.add_argument('-LastScanNr', type=int, required=True, help='Last scanNr')
parser.add_argument('-StartScanNr', type=int, required=False, default=1, help='Start scanNr')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter filename')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-numFilesPerScan', type=int, required=False, default=1, help='Number of files that constitute a single scan. This will combine multiple ge files into one dataset. 1 will disable.')
parser.add_argument('-nCPUs', type=int, required=False, default=-1, help='Number of parallel jobs to start. If -1, will use the default.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
args, unparsed = parser.parse_known_args()
resultFolder = args.resultFolder
LastScanNr = args.LastScanNr
StartScanNr = args.StartScanNr
paramFN = args.paramFN
numFrameChunks = args.numFrameChunks
numFilesPerScan = args.numFilesPerScan
nCPUs = args.nCPUs
preProc = args.preProcThresh

psContents = open(paramFN).readlines()
for line in psContents:
    if line.startswith('StartFileNrFirstLayer '):
        overAllFNr = int(line.split()[1])

if nCPUs == -1:
    p = Pool()
else:
    p = Pool(nCPUs)

work_data = []
for i in range(StartScanNr,LastScanNr+1):
    work_data.append([resultFolder,paramFN,i,numFrameChunks])

p.map(runSingle,work_data)
