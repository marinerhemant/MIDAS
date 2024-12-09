#!/usr/bin/env python

from numcodecs import Blosc
import zarr
import subprocess
import shutil
import os, sys
import numpy as np
import argparse
from PIL import Image

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
        return -1

parser = MyParser(description='''Code to update ZarrZip Dataset, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-fn', type=str, required=True, help='FileName to update.')
parser.add_argument('-keyToUpdate', type=str, required=True, help='Key inside filename to update.')
parser.add_argument('-folder', type=str, required=False, default='', help='Folder with the file. If nothing is provided, it will default to the current folder.')
parser.add_argument('-updatedValue', type=str, required=True, nargs='*', help="Give as many values of things to update. OmegaRanges, give n-pairs, BoxSizes, give n-quadruplets, RingThresh, give n-pairs, ImTransOpt, give n-values.")
args, unparsed = parser.parse_known_args()
folder = args.folder
key = args.keyToUpdate
fnIn = args.fn
upd = args.updatedValue

if key[0] == '/':
    key = key[1:]

if len(folder) == 0 or folder == '.':
    folder = os.getcwd()
os.chdir(folder)
keyTop = key.split('/')[0]
zf = zarr.open(fnIn,'r')
try:
    rf = zf[key][:]
    print(f'Initial value: {rf}')
except:
    print("Key did not exist, will add to the file.")
fnTemp = fnIn+'.tmp'
bnFNTemp = fnTemp

if 'data' in key:
    # We are going to update the data array.
    infoStr = upd[0].split('_')[-1].split('.')[0]
    nFrames = int(infoStr.split('x')[0])
    nPxZ = int(infoStr.split('x')[1])
    nPxY = int(infoStr.split('x')[2])
    im_arr = np.fromfile(upd[0],dtype=np.uint16).reshape((nFrames,nPxZ,nPxY))

# check if one value was given, find what we have:
tp = 0
if len(upd) == 1:
    tp = checkType(upd[0])
    if tp == 1:
        newVal = np.array([int(upd[0])]).astype(np.int32)
        keyPos = '0'
    elif tp == 2:
        newVal = np.array([float(upd[0])]).astype(np.double)
        keyPos = '0'
    else:
        if 'mask' in key:
            newVal = np.array(Image.open(upd[0])).astype(np.uint16)[np.newaxis]
            keyPos = '0.0.0'
        elif 'data' in key:
            newVal = np.copy(im_arr).astype(np.uint16)
            keyPos = '0.0.0'
        else:
            newVal = np.bytes_(upd[0],'utf-8')
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

# print(newVal)
zf2 = zarr.open(fnTemp,'w')
if tp!=-1:
    ds = zf2.create_dataset(key,shape=(newVal.shape),dtype=newVal.dtype,chunks=(newVal.shape),
                        compressor=Blosc(cname='zstd', clevel=3,
                        shuffle=Blosc.BITSHUFFLE))
else: # This means we are writing a string
    if len(keyPos)==1:
        ds = zf2.create_dataset(key,shape=(1,),dtype=newVal.dtype,chunks=(1,),
                            compressor=Blosc(cname='zstd', clevel=3,
                                            shuffle=Blosc.BITSHUFFLE))
    elif 'mask' in key:
        # WE NEED TO ADD .zarray file too!!!!!
        ds = zf2.create_dataset(key,shape=newVal.shape,dtype=newVal.dtype,chunks=newVal.shape,
                            compressor=Blosc(cname='zstd', clevel=3,
                            shuffle=Blosc.BITSHUFFLE))
    elif 'data' in key:
        # WE NEED TO ADD .zarray file too!!!!!
        ds = zf2.create_dataset(key,shape=newVal.shape,dtype=newVal.dtype,chunks=(1,nPxZ,nPxY),
                            compressor=Blosc(cname='zstd', clevel=3,
                            shuffle=Blosc.BITSHUFFLE))
ds[:] = newVal
shutil.move(f'{bnFNTemp}/{keyTop}',f'{keyTop}')
subprocess.call(f'zip -u {fnIn} {key}/.zarray',shell=True)
if 'data' not in key:
    subprocess.call(f'zip -u {fnIn} {key}/{keyPos}',shell=True)
else:
    subprocess.call(f'zip -d {fnIn} "{key}/*"',shell=True,stdout=open('/dev/null'))
    subprocess.call(f'zip -u {fnIn} {key}/.zarray',shell=True)
    for frameNr in range(nFrames):
        keyPos = f'{frameNr}.0.0'
        subprocess.call(f'zip -u {fnIn} {key}/{keyPos}',shell=True)
shutil.rmtree(keyTop)
shutil.rmtree(fnTemp)
zf = zarr.open(fnIn,'r')
rf = zf[key][:]
print(f'Updated value: {key}:{rf}')
print(rf.shape)
