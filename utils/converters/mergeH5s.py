import h5py
import argparse
import sys

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

parser = MyParser(description='''Merge two separate files containing data and dark into one, writing a single file.''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dataFN', type=str, required=True, help='Data filename.')
parser.add_argument('-darkFN', type=str, required=True, help='Dark filename.')
parser.add_argument('-dataLoc', type=str, required=True, help='dataset location with information.')
parser.add_argument('-darkLoc', type=str, required=False, default='', help='dataset location in dark with information in case different from data file.')
args, unparsed = parser.parse_known_args()

dataFN = args.dataFN
darkFN = args.darkFN
dataLoc = args.dataLoc
darkLoc = args.darkLoc

if len(darkLoc) < 2:
      darkLoc = dataLoc

with h5py.File(darkFN,'r') as hf:
    dark = hf[darkLoc][()]
with h5py.File(dataFN,'r') as hf:
    data = hf[dataLoc][()]

dataFNOut = f'{dataFN}.withdark.h5'
with h5py.File(dataFNOut,'w') as hf:
    grp = hf.create_group('exchange')
    grp.create_dataset('data',shape=data.shape,dtype=data.dtype,data=data)
    grp.create_dataset('dark',shape=dark.shape,dtype=dark.dtype,data=dark)

