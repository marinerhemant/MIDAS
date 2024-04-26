import numpy as np
import blosc2
import zipfile
archive = zipfile.ZipFile('/Users/hsharma/opt/MIDAS/FF_HEDM/Example/Au_FF_000001.ge3.zip','r')
for frameNr in range(1440):
    fn = f'{frameNr}.0.0'
    data = archive.read(fn)
    frame = np.frombuffer(memoryview(blosc2.decompress(bytearray(data))),dtype=np.uint16)
    nS = np.sum(frame>0)
    if (nS>0):
        print(nS)