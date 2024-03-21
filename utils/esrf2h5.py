import h5py

fn = '/Users/hsharma/Desktop/analysis/bucsek_jul22_midas/ESRF_Debug/ESRF DeBug Data/NS_Ti_deformed_Z600_3.h5'
hf = h5py.File(fn,'r')
print(hf.keys())
for key in hf.keys():
    dsetpath = f'{key}/instrument/eiger/image'
    strKey = key.split('.')[0]
    dset_shape = hf[dsetpath].shape
    f = h5py.File(f'/Users/hsharma/Desktop/analysis/bucsek_jul22_midas/ESRF_Debug/ESRF DeBug Data/data_{strKey.zfill(6)}.h5', 'w')
    grp = f.create_group('exchange')
    # dset = f.create_dataset('data', hf[dsetpath].shape, dtype=hf[dsetpath].dtype)
    link = h5py.ExternalLink(fn, dsetpath)
    f['exchange/data'] = link
    f.close()
hf.close()
