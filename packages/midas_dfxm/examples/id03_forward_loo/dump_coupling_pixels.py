import numpy as np, sys
sys.path.insert(0, '.')
import coupling_measure as cm
import midas_dfxm.forward_maps as fm
H5, OUT = sys.argv[1], sys.argv[2]        # raw 8-plane HDF5 file, output .npz
d=fm.load_planes_cube(H5, log=lambda *a: None)
cube,mu_g,ob_g,rc=d['cube'],d['mu_g'],d['ob_g'],d['rc']
rng=np.random.default_rng(0)
num,den,used,bid,S,mu,ok=cm.estimate(cube,mu_g,ob_g,rc,rng)
np.savez(OUT,rc=rc,S=S.astype(np.float32),mu=mu.astype(np.float32),ok=ok,ob=ob_g,mu_g=mu_g,num=num,den=den,used=used)
print('saved',int(used.sum()),'slope',num.sum()/den.sum())
