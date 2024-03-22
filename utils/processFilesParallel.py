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
    cwd = os.getcwd()
    binloc = os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')
    cmd = f'/nfs/turbo/meche-abucsek/Wenxi/ESRF_Ti_v7/.venv/bin/python {binloc} -resultFolder {rf} -paramFN {pf} -LayerNr {lN} -numFrameChunks {nc}'
    print(cmd)
    subprocess.call(cmd,cwd=cwd,shell=True)

parser = MyParser(description='''esrf2hf.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save results')
parser.add_argument('-LastScanNr', type=int, required=True, help='Last scanNr')
parser.add_argument('-StartScanNr', type=int, required=False, default=1, help='Start scanNr')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter filename')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-nCPUs', type=int, required=False, default=-1, help='Number of parallel jobs to start. If -1, will use the default.')
args, unparsed = parser.parse_known_args()
resultFolder = args.resultFolder
LastScanNr = args.LastScanNr
StartScanNr = args.StartScanNr
paramFN = args.paramFN
numFrameChunks = args.numFrameChunks
nCPUs = args.nCPUs

if nCPUs == -1:
    p = Pool()
else:
    p = Pool(nCPUs)

work_data = []
for i in range(StartScanNr,LastScanNr+1):
    work_data.append([resultFolder,paramFN,i,numFrameChunks])

p.map(runSingle,work_data)