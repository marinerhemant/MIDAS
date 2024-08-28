import h5py, sys, argparse, numpy

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''esrf2hf.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-folder', type=str, required=True, help='Folder where data exists')
parser.add_argument('-LastScanNr', type=int, required=True, help='Last scanNr, it will always start from 1')
parser.add_argument('-OutputFStem', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
parser.add_argument('-OmegaFile', type=str, required=False, default='', help='If want to provide custom omega values, provide a CSV file with one value of omega for each scan on each line.')
args, unparsed = parser.parse_known_args()
folder = args.folder
LastScanNr = args.LastScanNr
OutputFStem = args.OutputFStem
OmegaFile = args.OmegaFile
Omegas = []
if len(OmegaFile) >0:
    Omegas = numpy.genfromtxt(OmegaFile)

for fileNr in range(1,LastScanNr+1):
    fn = f'{folder}/scan{str(fileNr).zfill(4)}/eiger_0000.h5'
    hf = h5py.File(fn,'r')
    print(f"Processing dset: {fileNr}")
    dsetpath = f'/entry_0000/ESRF-ID11/eiger/data'
    dset_shape = hf[dsetpath].shape
    f = h5py.File(f'{folder}/{OutputFStem}_{str(fileNr).zfill(6)}.h5', 'w')
    grp = f.create_group('exchange')
    link = h5py.ExternalLink(fn, dsetpath)
    f['exchange/data'] = link
    if len(Omegas) > 0:
        gp1 = f.create_dataset('startOmeOverride',data=Omegas[fileNr-1],shape=(1,))
    f.close()
    hf.close()
