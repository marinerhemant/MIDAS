from numcodecs import Blosc
import zarr
import subprocess
import shutil
import os, sys
import numpy as np
import argparse

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def checkType(s):
    try:
        float(s)
        try:
            int(s)
            return 1
        except:
            return 2
    except:
        return 0

parser = MyParser(description='''Far-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-fn', type=str, required=True, help='FileName to update.')
parser.add_argument('-keyToUpdate', type=str, required=True, help='Key inside filename to update.')
parser.add_argument('-folder', type=str, required=False, default='', help='Folder with the file. If nothing is provided, it will default to the current folder.')
parser.add_argument('-UpdatedValue', type=str, required=True, nargs='*', help="Give as many values of things to update. OmegaRanges, give n-pairs, BoxSizes, give n-quadruplets, RingThresh, give n-pairs, ImTransOpt, give n-values.")
args, unparsed = parser.parse_known_args()
folder = args.folder
key = args.keyToUpdate
fnIn = args.fn
upd = args.UpdatedValue

if len(folder) == 0 or folder == '.':
    folder = os.getcwd()
os.chdir(folder)
keyTop = key.split('/')[0]
zf = zarr.open(fnIn,'r')
rf = zf[key][:]
print(f'Initial value: {rf}')
fnTemp = fnIn+'.tmp'
bnFNTemp = fnTemp

# check if one value was given, find what we have:
if len(upd) == 1:
    tp = checkType(upd[0])
    if tp == 1:
        newVal = np.array([int(upd[0])]).astype(np.int32)
    elif tp == 2:
        newVal = np.array([float(upd[0])]).astype(np.double)
    else:
        newVal = np.string_(upd[0],'utf-8')
    keyPos = '0'
else:
    keyPos = '0.0'
    if 'OmegaRanges' in key:
        newArr = []
        for i in range(len(upd)//2):
            newArr.append([upd[i*2+0],upd[i*2+1]])
        newVal = np.array(newArr).astype(np.double)
    elif 'RingThresh' in key:
        newArr = []
        for i in range(len(upd)//2):
            newArr.append([upd[i*2+0],upd[i*2+1]])
        newVal = np.array(newArr).astype(np.double)
    elif 'BoxSizes' in key:
        newArr = []
        for i in range(len(upd)//4):
            newArr.append([upd[i*4+0],upd[i*4+1],upd[i*4+2],upd[i*4+3]])
        newVal = np.array(newArr).astype(np.double)
    elif 'ImTransOpt' in key:
        newArr = []
        for i in range(len(upd)):
            newArr.append([upd[i]])
        newVal = np.array(newArr).astype(np.double)

zf2 = zarr.open(fnTemp,'w')
ds = zf2.create_dataset(key,shape=(newVal.shape),dtype=newVal.dtype,chunks=(1,),
                        compressor=Blosc(cname='zstd', clevel=3,
                                         shuffle=Blosc.BITSHUFFLE))
ds[:] = newVal
shutil.move(f'{bnFNTemp}/{keyTop}',f'{keyTop}')
subprocess.call(f'zip -u {fnIn} {key}/.zarray',shell=True)
subprocess.call(f'zip -u {fnIn} {key}/{keyPos}',shell=True)
shutil.rmtree(keyTop)
shutil.rmtree(fnTemp)
zf = zarr.open(fnIn,'r')
rf = zf[key][:]
print(f'Updated value: {rf}')
