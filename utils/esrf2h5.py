import h5py, sys, argparse

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

parser = MyParser(description='''esrf2hf.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-folder', type=str, required=True, help='Folder where data exists')
parser.add_argument('-InputFN', type=str, required=True, help='Input filename')
parser.add_argument('-OutputFStem', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
args, unparsed = parser.parse_known_args()
folder = args.folder
InputFN = args.InputFN
OutputFStem = args.OutputFStem

fn = f'{folder}/{InputFN}'
hf = h5py.File(fn,'r')
print(hf.keys())
for key in hf.keys():
    dsetpath = f'{key}/instrument/eiger/image'
    strKey = key.split('.')[0]
    dset_shape = hf[dsetpath].shape
    f = h5py.File(f'{folder}/{OutputFStem}_{strKey.zfill(6)}.h5', 'w')
    grp = f.create_group('exchange')
    # dset = f.create_dataset('data', hf[dsetpath].shape, dtype=hf[dsetpath].dtype)
    link = h5py.ExternalLink(fn, dsetpath)
    f['exchange/data'] = link
    f.close()
hf.close()
