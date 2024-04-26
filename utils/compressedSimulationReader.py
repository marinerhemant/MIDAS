import numpy as np
import blosc2
folder = '/Users/hsharma/opt/MIDAS/FF_HEDM/Example/Au_FF_000001.ge3'
for frameNr in range(1440):
    fn = f'{folder}/{frameNr}.0.0'
    frame = np.frombuffer(memoryview(blosc2.decompress(bytearray(open(fn,'rb').read()))),dtype=np.uint16)
    print(np.sum(frame>0))