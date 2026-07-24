#!/usr/bin/env python
"""
midas_ff_report.py — generate a self-contained HTML reconstruction report from a
MIDAS FF-HEDM result directory (a LayerNr_N folder containing Grains.csv and,
optionally, processgrains_diagnostics.h5).

Runs where the data lives (diagnostics h5 can be large). Emits ONE report.html with
all figures base64-embedded. An agent then publishes report.html as an Artifact.

Usage:
    python midas_ff_report.py RUN_DIR [--material NAME] [--title T] [--out report.html]
                              [--c11 GPa --c12 GPa]   # optional, for d0 stress bias

Design + framing: see MIDAS_REPORT_GENERATOR.md.
"""
import argparse, base64, json, os, re, math, glob, string
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PAPER="#f7f6f3"; INK="#131619"; GRID="#d9d6cf"; TEAL="#0e7c86"; COPPER="#c07a3e"; MUT="#6b6862"
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"savefig.facecolor":PAPER,
    "font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":"#b9b6ae","axes.labelcolor":INK,
    "text.color":INK,"xtick.color":MUT,"ytick.color":MUT,"axes.titlecolor":INK,"axes.grid":True,
    "grid.color":GRID,"grid.linewidth":0.6,"axes.linewidth":0.8,"figure.dpi":135})

GCOLS=("ID O11 O12 O13 O21 O22 O23 O31 O32 O33 X Y Z a b c alpha beta gamma DiffPos DiffOme "
       "DiffAngle GrainRadius Confidence eFab11 eFab12 eFab13 eFab21 eFab22 eFab23 eFab31 eFab32 "
       "eFab33 eKen11 eKen12 eKen13 eKen21 eKen22 eKen23 eKen31 eKen32 eKen33 RMSErrorStrain "
       "PhaseNr Eul0 Eul1 Eul2").split()

# ------------------------------------------------------------------ parsing
def parse_grains_header(path):
    meta={"nominal_lat":None,"spacegroup":None,"ngrains":None}
    for ln in open(path):
        if not ln.startswith("%"): break
        if "NumGrains" in ln: meta["ngrains"]=int(ln.split()[1])
        if "SpaceGroup" in ln:
            m=re.search(r"SpaceGroup:?\s*(\d+)",ln);  meta["spacegroup"]=int(m.group(1)) if m else None
        if "Lattice" in ln and ":" in ln:
            nums=re.findall(r"[-\d.]+",ln.split(":")[-1]);
            if len(nums)>=6: meta["nominal_lat"]=[float(x) for x in nums[:6]]
    return meta

def parse_param(run_dir):
    """Pull provenance from paramstest.txt / *ParamFile*.txt if present."""
    p={}
    cands=glob.glob(os.path.join(run_dir,"paramstest.txt"))+glob.glob(os.path.join(run_dir,"*ParamFile*.txt"))\
          +glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")),"*ParamFile*.txt"))
    for f in cands:
        for ln in open(f):
            t=ln.split()
            if len(t)<2: continue
            k=t[0]
            if k in ("Wavelength","Lsd","px","NrPixels","OmegaStep","OmegaStart","Completeness") and k not in p:
                p[k]=t[1].rstrip(";")
            if k=="BC" and "BC" not in p: p["BC"]=f"{t[1].rstrip(chr(59))} {t[2].rstrip(chr(59))}"
            if k=="RingThresh": p.setdefault("rings",[]).append(t[1])
        if p: break
    return p

def is_cubic(sg):  return sg is not None and 195<=sg<=230

def ipf_cubic(OM):
    v=np.abs(np.einsum('nij,j->ni',OM,np.array([0,0,1.0])))
    v=np.sort(v,axis=1); v/=np.linalg.norm(v,axis=1,keepdims=True)
    rgb=np.stack([v[:,2]-v[:,1],v[:,1]-v[:,0],v[:,0]],1)
    rgb/=rgb.max(axis=1,keepdims=True); return np.clip(rgb,0,1)**0.6

def b64(path): return "data:image/png;base64,"+base64.b64encode(open(path,"rb").read()).decode()

def fit_sin(eta,val):
    A=np.c_[np.cos(eta),np.sin(eta),np.ones_like(eta)]
    c,*_=np.linalg.lstsq(A,val,rcond=None); return float(np.hypot(c[0],c[1])), float(c[2])

# ------------------------------------------------------------------ figures
def add_ipf_legend(fig,rect):
    axl=fig.add_axes(rect); n=140
    gx,gy=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n)); mask=gy<=gx
    d=np.stack([gy,gx-gy,np.ones_like(gx)],-1); d/=np.linalg.norm(d,axis=-1,keepdims=True)
    sv=np.sort(np.abs(d),axis=-1); tr=np.stack([sv[...,2]-sv[...,1],sv[...,1]-sv[...,0],sv[...,0]],-1)
    tr/=tr.max(axis=-1,keepdims=True); tr=np.clip(tr,0,1)**0.6
    axl.imshow(np.where(mask[...,None],tr,1.0),origin="lower",extent=[0,1,0,1]); axl.axis("off")
    axl.set_title("IPF-Z",fontsize=8,color=MUT,pad=2)
    for xy,t in [((-.02,-.03),"001"),((1.02,-.03),"101"),((1.02,1.0),"111")]:
        axl.text(*xy,t,fontsize=6,color=INK)

def fig_grain_maps(C,OM,cubic,outp):
    X,Y,Z,rad=C["X"],C["Y"],C["Z"],C["GrainRadius"]
    col=ipf_cubic(OM) if cubic else np.tile([[0.05,0.49,0.53]],(len(X),1))
    s=18*(rad-rad.min())/(np.ptp(rad)+1e-9)+5; s=np.clip(s,4,26)
    fig,ax=plt.subplots(1,3,figsize=(13,4.4))
    for a,(u,w,ul,wl) in zip(ax,[(X,Y,"X (µm)","Y (µm)"),(X,Z,"X (µm)","Z (µm)"),(Y,Z,"Y (µm)","Z (µm)")]):
        a.scatter(u,w,s=s,c=col,edgecolors="none",alpha=0.85 if len(X)<8000 else 0.6)
        a.set_xlabel(ul);a.set_ylabel(wl);a.set_aspect("equal","datalim")
    for a,t in zip(ax,["beam-plane view","along beam (X–Z)","along beam (Y–Z)"]):
        a.set_title(t,loc="left",fontsize=10,color=MUT)
    if cubic: add_ipf_legend(fig,[0.905,0.60,0.085,0.30])
    fig.suptitle("Grain centroid maps — "+("IPF-Z orientation, " if cubic else "")+"sized by grain radius",
                 x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,0.9 if cubic else 1,0.95]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

def fig_error_maps(C,D,outp):
    X,Y,conf=C["X"],C["Y"],C["Confidence"]
    ia=D.get("grain_med_internal_angle_deg"); drad=D.get("grain_med_drad_um"); nsp=D.get("grain_n_spots")
    nsp=nsp.astype(float) if nsp is not None else np.full_like(conf,np.nan)
    ia=ia if ia is not None else np.full_like(conf,np.nan); drad=drad if drad is not None else np.full_like(conf,np.nan)
    P=[("Completeness",conf,"cividis",None,"→ 1 is best"),
       ("Spots per grain",nsp,"viridis",None,"more = better constrained"),
       ("Median internal angle (°)",ia,"magma_r",None,"angular misfit"),
       ("Median radial residual (µm)",drad,"RdBu",TwoSlopeNorm(0,vmin=-60,vmax=60),"ring-by-ring bias"),
       ("DiffPos (µm)",C["DiffPos"],"inferno_r",None,"position spread"),
       ("RMS strain error (µε)",C["RMSErrorStrain"],"inferno_r",None,"strain-fit residual")]
    ms=14 if len(X)<8000 else 5
    fig,ax=plt.subplots(2,3,figsize=(13,7.6))
    for a,(ttl,val,cmp,nrm,sub) in zip(ax.ravel(),P):
        sc=a.scatter(X,Y,c=val,s=ms,cmap=cmp,norm=nrm,edgecolors="none")
        a.set_aspect("equal","datalim");a.set_xlabel("X (µm)");a.set_ylabel("Y (µm)")
        a.set_title(ttl,loc="left",fontsize=10.5,weight="bold")
        a.text(0.0,1.008,sub,transform=a.transAxes,fontsize=8,color=MUT,va="bottom")
        cb=fig.colorbar(sc,ax=a,shrink=0.82,pad=0.02);cb.ax.tick_params(labelsize=8)
    fig.suptitle("Per-grain error / quality maps (beam-plane X–Y)",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

def fig_residuals(D,S,outp):
    fig,ax=plt.subplots(2,3,figsize=(13,7.4))
    rn=D.get("ring_nr"); ppm=D.get("ring_drad_ppm"); rns=D.get("ring_n_spots")
    if rn is not None:
        ax[0,0].bar([str(int(x)) for x in rn],ppm,color=TEAL,width=0.6)
        for x,p,nn in zip(range(len(rn)),ppm,rns): ax[0,0].text(x,p,f"n={int(nn)}",ha="center",fontsize=7,color=MUT,
            va="bottom" if p>=0 else "top")
    ax[0,0].axhline(0,color=MUT,lw=0.8);ax[0,0].set_title("Radial bias per ring (ppm)",loc="left",weight="bold",fontsize=10.5)
    ax[0,0].set_xlabel("ring #");ax[0,0].set_ylabel("median Δr/r (ppm)")
    elo=D.get("eta_bin_lo_deg")
    if elo is not None:
        ax[0,1].plot(elo,D["eta_med_drad_um"],"-o",color=TEAL,label="Δrad (µm)",ms=4)
        ax[0,1].plot(elo,D["eta_med_dtan_um"],"-s",color=COPPER,label="Δtan (µm)",ms=4)
        ax[0,1].legend(fontsize=8,frameon=False)
    ax[0,1].axhline(0,color=MUT,lw=0.8);ax[0,1].set_title("Residual vs azimuth η",loc="left",weight="bold",fontsize=10.5)
    ax[0,1].set_xlabel("η bin (°)");ax[0,1].set_ylabel("median residual (µm)")
    if S is not None:
        hb=ax[0,2].hexbin(S["drad_um"],S["dtan_um"],gridsize=45,cmap="cividis",bins="log",extent=[-1500,1500,-1500,1500])
        cb=fig.colorbar(hb,ax=ax[0,2],shrink=0.82,pad=0.02);cb.set_label("log N spots",fontsize=8);cb.ax.tick_params(labelsize=8)
        ax[1,0].hist(S["internal_angle_deg"],bins=80,range=(0,1.0),color=TEAL,alpha=0.85)
        ax[1,0].axvline(np.median(S["internal_angle_deg"]),color=COPPER,lw=1.4,ls="--",
            label=f"med {np.median(S['internal_angle_deg']):.3f}°");ax[1,0].legend(fontsize=8,frameon=False)
        ax[1,1].hist(S["dome_deg"],bins=80,range=(-0.6,0.6),color=COPPER,alpha=0.85)
        uq,cnt=np.unique(S["ring_nr"].astype(int),return_counts=True)
        ax[1,2].bar([str(x) for x in uq],cnt,color=MUT,width=0.6)
    ax[0,2].set_title("Spot residuals: radial vs tangential",loc="left",weight="bold",fontsize=10.5)
    ax[0,2].set_xlabel("Δradial (µm)");ax[0,2].set_ylabel("Δtangential (µm)")
    ax[1,0].set_title("Spot internal angle (°)",loc="left",weight="bold",fontsize=10.5);ax[1,0].set_xlabel("internal angle (°)");ax[1,0].set_ylabel("spots")
    ax[1,1].set_title("Spot Δω (dome, °)",loc="left",weight="bold",fontsize=10.5);ax[1,1].set_xlabel("Δω (°)");ax[1,1].set_ylabel("spots")
    ax[1,2].set_title("Assigned spots per ring",loc="left",weight="bold",fontsize=10.5);ax[1,2].set_xlabel("ring #");ax[1,2].set_ylabel("spots")
    ns=S['drad_um'].shape[0] if S is not None else 0
    fig.suptitle(f"Residual diagnostics — geometry & spot fit quality ({ns:,} spots)",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

def fig_strain(C,nom_a,D,outp):
    X,Y,conf=C["X"],C["Y"],C["Confidence"]
    hydro=(C["eFab11"]+C["eFab22"]+C["eFab33"])/3.0
    nsp=D.get("grain_n_spots"); nsp=nsp.astype(float) if nsp is not None else C["GrainRadius"]
    ms=14 if len(X)<8000 else 5
    fig,ax=plt.subplots(2,3,figsize=(13,7.4))
    sc=ax[0,0].scatter(X,Y,c=hydro,s=ms,cmap="RdBu_r",norm=TwoSlopeNorm(0,vmin=-800,vmax=800),edgecolors="none")
    ax[0,0].set_aspect("equal","datalim");ax[0,0].set_title("Hydrostatic strain (µε)",loc="left",weight="bold",fontsize=10.5)
    ax[0,0].set_xlabel("X (µm)");ax[0,0].set_ylabel("Y (µm)");fig.colorbar(sc,ax=ax[0,0],shrink=0.82,pad=0.02).ax.tick_params(labelsize=8)
    comps=["eFab11","eFab22","eFab33","eFab12","eFab13","eFab23"]
    bp=ax[0,1].boxplot([np.clip(C[c],-6000,6000) for c in comps],tick_labels=[c.replace("eFab","ε") for c in comps],
        showfliers=False,patch_artist=True,medianprops=dict(color=INK))
    for p in bp['boxes']:p.set_facecolor(TEAL);p.set_alpha(0.55)
    ax[0,1].axhline(0,color=MUT,lw=0.8);ax[0,1].set_title("Strain components (sample frame, µε)",loc="left",weight="bold",fontsize=10.5);ax[0,1].set_ylabel("microstrain")
    ax[0,2].hist(C["a"],bins=60,color=COPPER,alpha=0.85)
    ax[0,2].axvline(nom_a,color=INK,lw=1.4,ls="--",label=f"nominal {nom_a:.4f} Å")
    ax[0,2].axvline(np.median(C["a"]),color=TEAL,lw=1.4,label=f"median {np.median(C['a']):.4f} Å")
    ax[0,2].set_title("Refined lattice a (Å)",loc="left",weight="bold",fontsize=10.5);ax[0,2].set_xlabel("a (Å)");ax[0,2].set_ylabel("grains");ax[0,2].legend(fontsize=8,frameon=False)
    ax[1,0].hist(conf,bins=40,color=TEAL,alpha=0.85);ax[1,0].axvline(0.5,color=COPPER,lw=1.4,ls="--")
    ax[1,0].set_title("Completeness",loc="left",weight="bold",fontsize=10.5);ax[1,0].set_xlabel("completeness");ax[1,0].set_ylabel("grains")
    ax[1,1].hist(nsp,bins=40,color=MUT,alpha=0.85);ax[1,1].axvline(np.median(nsp),color=COPPER,lw=1.4,ls="--",label=f"med {np.median(nsp):.0f}")
    ax[1,1].set_title("Spots per grain",loc="left",weight="bold",fontsize=10.5);ax[1,1].set_xlabel("n spots");ax[1,1].set_ylabel("grains");ax[1,1].legend(fontsize=8,frameon=False)
    ax[1,2].hist(C["DiffPos"],bins=50,color=COPPER,alpha=0.85);ax[1,2].axvline(np.median(C["DiffPos"]),color=TEAL,lw=1.4,ls="--",label=f"med {np.median(C['DiffPos']):.0f} µm")
    ax[1,2].set_title("DiffPos — position spread (µm)",loc="left",weight="bold",fontsize=10.5);ax[1,2].set_xlabel("DiffPos (µm)");ax[1,2].set_ylabel("grains");ax[1,2].legend(fontsize=8,frameon=False)
    fig.suptitle("Strain, lattice & grain-quality distributions",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)


def fig_grain_error_hists(C,D,outp):
    """Per-grain error distributions — one histogram per metric."""
    def g(k): 
        v=D.get(k); return v.astype(float) if v is not None else None
    panels=[("Completeness",C["Confidence"],None,"grains"),
            ("Spots per grain",g("grain_n_spots"),None,"grains"),
            ("Median internal angle (°)",g("grain_med_internal_angle_deg"),(0,0.6),"grains"),
            ("Median radial residual (µm)",g("grain_med_drad_um"),(-150,150),"grains"),
            ("MAD tangential residual (µm)",g("grain_mad_dtan_um"),None,"grains"),
            ("Median vertical residual dz (µm)",g("grain_med_dz_um"),(-300,300),"grains"),
            ("DiffPos (µm)",C["DiffPos"],None,"grains"),
            ("DiffAngle (°)",C["DiffAngle"],None,"grains"),
            ("RMS strain error (µε)",C["RMSErrorStrain"],None,"grains")]
    fig,ax=plt.subplots(3,3,figsize=(13,10))
    for a,(ttl,v,rng,yl) in zip(ax.ravel(),panels):
        if v is None or not np.isfinite(v).any():
            a.axis("off"); a.set_title(ttl+" — n/a",loc="left",fontsize=10,color=MUT); continue
        vv=v[np.isfinite(v)]
        a.hist(vv,bins=60,range=rng,color=TEAL,alpha=0.85)
        med=np.nanmedian(vv)
        a.axvline(med,color=COPPER,lw=1.4,ls="--",label=f"med {med:.3g}")
        a.legend(fontsize=8,frameon=False)
        a.set_title(ttl,loc="left",fontsize=10.5,weight="bold"); a.set_ylabel(yl)
    fig.suptitle("Per-grain error distributions",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

def fig_position_diag(C,D,outp,beam_um=None):
    """Position sanity: is the fitted Z supported by the vertical residual?"""
    Z=C["Z"]; dz=D.get("grain_med_dz_um"); conf=C["Confidence"]; ns=D.get("grain_n_spots")
    fig,ax=plt.subplots(1,3,figsize=(13,4.5))
    ax[0].hist(Z,bins=80,color=TEAL,alpha=0.85)
    if beam_um:
        ax[0].axvline(-beam_um/2,color=COPPER,lw=1.6,ls="--")
        ax[0].axvline( beam_um/2,color=COPPER,lw=1.6,ls="--",label=f"beam ±{beam_um/2:.0f} µm")
        inb=100*np.mean(np.abs(Z)<=beam_um/2); ax[0].legend(fontsize=8,frameon=False)
        ax[0].set_xlabel(f"Z (µm)   —   {inb:.0f}% inside beam")
    else: ax[0].set_xlabel("Z (µm)")
    ax[0].set_title("Fitted grain Z",loc="left",weight="bold",fontsize=11); ax[0].set_ylabel("grains")
    if dz is not None:
        ok=np.isfinite(dz)
        ax[1].hexbin(Z[ok],dz[ok],gridsize=60,cmap="cividis",bins="log")
        edges=np.linspace(np.percentile(Z,0.5),np.percentile(Z,99.5),14)
        cx=0.5*(edges[:-1]+edges[1:]); cy=[np.nanmedian(dz[(Z>=a_)&(Z<b_)]) for a_,b_ in zip(edges[:-1],edges[1:])]
        ax[1].plot(cx,cy,"-o",color=COPPER,ms=4,lw=1.6,label="binned median")
        ax[1].axhline(0,color="w",lw=0.8,alpha=0.6)
        r=np.corrcoef(Z[ok],dz[ok])[0,1]
        ax[1].set_title(f"Vertical residual vs fitted Z   (r = {r:+.2f})",loc="left",weight="bold",fontsize=11)
        ax[1].set_xlabel("fitted Z (µm)"); ax[1].set_ylabel("median dz residual (µm)"); ax[1].legend(fontsize=8,frameon=False)
    absZ=np.abs(Z)
    edges=[0,50,100,150,200,250,300,10000]; cxs=[];cc=[];nn=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(absZ>=lo)&(absZ<hi)
        if m.sum()>20:
            cxs.append(f"{lo}-{hi if hi<10000 else '+'}"); cc.append(np.nanmedian(conf[m]))
            nn.append(np.nanmedian(ns[m]) if ns is not None else np.nan)
    a2=ax[2]; a2.plot(range(len(cc)),cc,"-o",color=TEAL,label="completeness")
    a2.set_xticks(range(len(cxs))); a2.set_xticklabels(cxs,rotation=45,fontsize=8)
    a2.set_ylabel("median completeness",color=TEAL); a2.set_xlabel("|Z| bin (µm)")
    a3=a2.twinx(); a3.plot(range(len(nn)),nn,"-s",color=COPPER,label="spots"); a3.set_ylabel("median spots",color=COPPER)
    a3.grid(False)
    a2.set_title("Quality vs distance from beam centre",loc="left",weight="bold",fontsize=11)
    fig.suptitle("Position diagnostic — is the fitted Z supported by the data?",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--material",default=None); ap.add_argument("--title",default=None)
    ap.add_argument("--out",default="report.html"); ap.add_argument("--c11",type=float); ap.add_argument("--c12",type=float)
    ap.add_argument("--figdir",default=None); ap.add_argument("--beam-height",type=float,default=None,dest="beam")
    a=ap.parse_args()
    rd=a.run_dir; figdir=a.figdir or os.path.join(rd,"_report_figs"); os.makedirs(figdir,exist_ok=True)
    gpath=os.path.join(rd,"Grains.csv"); meta=parse_grains_header(gpath)
    nom_a=meta["nominal_lat"][0] if meta["nominal_lat"] else float("nan"); sg=meta["spacegroup"]; cubic=is_cubic(sg)
    d=np.loadtxt(gpath,comments="%");
    if d.ndim==1: d=d[None,:]
    C={n:d[:,i] for i,n in enumerate(GCOLS)}; OM=d[:,1:10].reshape(-1,3,3)
    material=a.material or f"SG{sg}"; ngr=len(d)
    prov=parse_param(rd)

    # diagnostics
    D={}; S=None
    dpath=os.path.join(rd,"processgrains_diagnostics.h5")
    if os.path.exists(dpath):
        import h5py
        with h5py.File(dpath,"r") as f:
            for grp in ("residuals","diagnostics"):
                if grp in f:
                    for k in f[grp]:
                        if k=="spot_table": continue
                        try: D[k]=f[f"{grp}/{k}"][()]
                        except Exception: pass
            if "residuals/spot_table" in f:
                st=f["residuals/spot_table"][()]
                sc="grain_idx spot_id ring_nr eta_deg dy_um dz_um drad_um dtan_um dome_deg internal_angle_deg r_exp_um".split()
                S={n:st[:,i] for i,n in enumerate(sc)}

    # d0 (cubic, midas_stress)
    d0=None
    if cubic:
        try:
            import midas_stress as ms
            latc=d[:,13:19]; ref=np.array([nom_a]*3+[90.]*3); vol=C["GrainRadius"]**3
            r=ms.recover_d0_cubic_free_standing(latc,ref,volumes=vol,confidences=C["Confidence"],min_confidence=0.5)
            a0=float(np.asarray(r["reference_recovered"]).ravel()[0]); eps=float(r["eps_iso"])*1e6
            def hydro(rr):
                e=np.array([ms.lattice_params_to_strain(latc[i],rr) for i in range(len(latc))]).reshape(len(latc),-1)
                return ((e[:,0]+e[:,1]+e[:,2])/3.0)*1e6
            m=C["Confidence"]>=0.5
            hb=hydro(ref); ha=hydro(np.array([a0]*3+[90.]*3))
            mpa=None
            if a.c11 and a.c12: mpa=ms.d0_sensitivity(C11=a.c11,C12=a.c12)["sensitivity_MPa_per_ppm"]
            d0=dict(a0=a0,eps=eps,hb=hb,ha=ha,m=m,mpa=mpa,n=int(m.sum()))
        except Exception as e:
            print("d0 skipped:",e)

    # figures
    figs={}
    figs["grain"]=os.path.join(figdir,"grain_maps.png"); fig_grain_maps(C,OM,cubic,figs["grain"])
    figs["err"]=os.path.join(figdir,"error_maps.png"); fig_error_maps(C,D,figs["err"])
    figs["res"]=os.path.join(figdir,"residuals.png"); fig_residuals(D,S,figs["res"])
    figs["strn"]=os.path.join(figdir,"strain_quality.png"); fig_strain(C,nom_a,D,figs["strn"])
    figs["ghist"]=os.path.join(figdir,"grain_error_hists.png"); fig_grain_error_hists(C,D,figs["ghist"])
    figs["pos"]=os.path.join(figdir,"position_diag.png"); fig_position_diag(C,D,figs["pos"],beam_um=a.beam)
    if d0:
        p=os.path.join(figdir,"d0_calib.png"); _fig_d0(d0,p); figs["d0"]=p

    # computed findings
    F=compute_findings(C,D,S,d0,prov,ngr,beam_um=a.beam)
    html=build_html(a.title or f"{material} — Far-Field HEDM reconstruction", material, meta, prov, C, D, S, d0, F, figs, ngr, os.path.abspath(gpath))
    open(a.out,"w").write(html); print("wrote",a.out,len(html),"bytes")

def _fig_d0(d0,outp):
    hb,ha,m=d0["hb"],d0["ha"],d0["m"]
    fig,ax=plt.subplots(1,3,figsize=(13,4.5)); bins=np.linspace(-900,900,80)
    ax[0].hist(hb[m],bins=bins,color=COPPER,alpha=0.65,label=f"before (mean {hb[m].mean():+.0f} µε)")
    ax[0].hist(ha[m],bins=bins,color=TEAL,alpha=0.7,label=f"after (mean {ha[m].mean():+.0f} µε)")
    ax[0].axvline(0,color=INK,lw=1);ax[0].set_title("Hydrostatic strain — d0 correction",loc="left",weight="bold",fontsize=11)
    ax[0].set_xlabel("hydrostatic strain (µε)");ax[0].set_ylabel("grains");ax[0].legend(fontsize=8,frameon=False,loc="upper left")
    txt=(f"d0 self-calibration (cubic, free-standing)\n  ⟨ε_hydro⟩_V = 0\n\n"
         f"  a0 recovered   {d0['a0']:.5f} Å\n  ε_iso error    {d0['eps']:+.0f} µε\n  grains used    {d0['n']} (conf≥0.5)\n")
    if d0["mpa"]: txt+=f"\n  stress bias removed ≈ {abs(d0['eps']*d0['mpa']):.0f} MPa\n"
    txt+="\n  deviatoric strain unchanged\n  scatter unchanged → geometry/ring-limited"
    ax[2].axis("off"); ax[2].text(0,0.98,txt,va="top",fontsize=10.5,family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.6",fc="#fff",ec=TEAL,lw=1.2))
    ax[1].hist(ha[m],bins=bins,color=TEAL,alpha=0.8);ax[1].axvline(0,color=INK,lw=1)
    ax[1].set_title("Hydrostatic strain after d0",loc="left",weight="bold",fontsize=11);ax[1].set_xlabel("µε");ax[1].set_ylabel("grains")
    fig.suptitle("d0 (strain-free reference) calibration with midas-stress",x=0.02,ha="left",fontsize=12.5,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(outp,bbox_inches="tight"); plt.close(fig)

def compute_findings(C,D,S,d0,prov,ngr,beam_um=None):
    F={"solid":[],"warn":[],"roadmap":[]}
    conf=C["Confidence"]; F["solid"].append(("Physical reconstruction",
        f"{ngr:,} grains; lattice a median {np.median(C['a']):.4f} Å; completeness median {np.median(conf):.2f} ({int((conf>=0.5).sum()):,} ≥ 0.5)."))
    if D.get("overall_med_internal_angle_deg") is not None:
        ia=float(D["overall_med_internal_angle_deg"]); dm=float(D.get("overall_mad_dome_deg",0))
        F["solid"].append(("Angular fit",f"Median internal angle {ia:.2f}°, dome MAD {dm:.2f}° — orientation is the most trustworthy product."))
    # eta-sinusoid → BC
    if S is not None:
        eta=np.deg2rad(S["eta_deg"]); amp,off=fit_sin(eta,S["drad_um"]); px=float(re.sub(r"[^0-9.eE+-]","",str(prov.get("px") or "150")) or "150")
        if amp>100:
            F["warn"].append(("Residual centering systematic",
                f"Δradial η-sinusoid amplitude {amp:.0f} µm (~{amp/px:.1f} px) — a beam-center / position offset."))
            F["roadmap"].append(("Detector geometry (beam center)",
                f"±{amp:.0f} µm η-sinusoid → recalibrate BC/Lsd/tilts against a matching powder calibrant, then re-index. Biggest scatter fix."))
    # ring-ppm trend
    ppm=D.get("ring_drad_ppm")
    if ppm is not None and len(ppm)>=2:
        rng=float(np.max(ppm)-np.min(ppm))
        if abs(ppm[-1])>50 or rng>200:
            F["warn"].append(("Ring-scale radial trend",f"Radial bias runs {np.min(ppm):.0f}→{np.max(ppm):.0f} ppm across rings — a 2θ-dependent detector-distance / reference-lattice scale term."))
        if rng>200:
            F["roadmap"].append(("Detector distance / reference scale",f"Radial residual spans {rng:.0f} ppm over the ring set — refine Lsd (and/or the reference lattice) against a powder calibrant, or via a grain-based geometry pass, to flatten it."))
    # rings
    nring=len(D.get("ring_nr",[])) if D.get("ring_nr") is not None else (len(np.unique(S["ring_nr"])) if S is not None else 0)
    if nring and nring<6:
        F["roadmap"].append(("Ring coverage",f"Only {nring} rings indexed — add higher-angle unsaturated rings for a better-conditioned strain tensor."))
    # d0
    if d0:
        F["solid"].append(("d0 reference calibrated",
            f"a0 = {d0['a0']:.5f} Å ({d0['eps']:+.0f} µε); hydrostatic strain re-centered"+
            (f", removes ≈{abs(d0['eps']*d0['mpa']):.0f} MPa bias" if d0['mpa'] else "")+". Deviatoric unchanged."))
    dz=D.get("grain_med_dz_um")
    if dz is not None and np.isfinite(dz).any():
        ok=np.isfinite(dz); r=float(np.corrcoef(C["Z"][ok],dz[ok])[0,1])
        if r<-0.3:
            msg=(f"Fitted Z anti-correlates with its own vertical residual (r={r:+.2f}): grains far from the "
                 f"beam centre carry residuals pointing back toward it, so their Z is not supported by their spots.")
            if beam_um: msg+=f" Only {100*np.mean(np.abs(C['Z'])<=beam_um/2):.0f}% of grains lie inside the ±{beam_um/2:.0f} µm beam."
            F["warn"].append(("Position refinement runaway",msg))
            F["roadmap"].append(("Constrain Z to the illuminated beam height",
                "Set Hbeam/BeamThickness to the true per-layer beam (not the full sample height) so the position fit cannot "
                "place grains outside the illuminated slab; then verify the dz residual stays flat vs Z."))
    F["warn"].append(("Strain scatter",
        f"Per-grain RMS strain residual median {np.median(C['RMSErrorStrain']):.0f} µε — treat individual tensors as indicative; the bias (d0) is fixable, the scatter is geometry/ring-limited."))
    return F

def build_html(title,material,meta,prov,C,D,S,d0,F,figs,ngr,gpath):
    IMG={k:b64(v) for k,v in figs.items()}
    nsp_med=np.median(D["grain_n_spots"]) if D.get("grain_n_spots") is not None else float("nan")
    nspots=(S["drad_um"].shape[0] if S is not None else 0)
    tiles=[("Grains indexed",f"{ngr:,}",f"SG {meta['spacegroup']}"),
           ("Indexed spots",f"{nspots:,}",f"median {nsp_med:.0f} / grain" if nspots else "—"),
           ("Median completeness",f"{np.median(C['Confidence']):.2f}",f"{int((C['Confidence']>=0.5).sum()):,} ≥ 0.5"),
           ("Lattice a",f"{np.median(C['a']):.4f} Å",f"nominal {meta['nominal_lat'][0]:.4f}" if meta['nominal_lat'] else "")]
    if d0: tiles.append(("Calibrated a₀",f"{d0['a0']:.4f} Å",f"{d0['eps']:+.0f} µε vs nominal"))
    tiles+=[("Hydrostatic strain",f"{((C['eFab11']+C['eFab22']+C['eFab33'])/3.0)[C['Confidence']>=0.5].mean():+.0f} µε","conf ≥ 0.5"),
            ("DiffPos median",f"{np.median(C['DiffPos']):.0f} µm","position spread")]
    tiles_html="".join(f'<div class="tile"><div class="tk">{k}</div><div class="tv">{v}</div><div class="ts">{s}</div></div>' for k,v,s in tiles)
    def finds(items,cls,tag): return "".join(
        f'<div class="find {cls}"><span class="tag {cls}">{tag}</span><h3>{t}</h3><p>{b}</p></div>' for t,b in items)
    order=[("01","Grain centroid maps","orientation + radius","grain"),
           ("02","Per-grain error & quality maps","spatial fit metrics","err"),
           ("03","Residual diagnostics","geometry & spot fit","res"),
           ("04","Strain, lattice & quality","refinement outputs","strn")]
    order.append(("05","Per-grain error distributions","histograms per metric","ghist"))
    order.append(("06","Position diagnostic","is fitted Z supported by the data?","pos"))
    if "d0" in IMG: order.append(("07","d0 strain-free reference","midas-stress","d0"))
    figs_html="".join(f'''<figure class="plate"><figcaption class="pcap"><span class="pnum">{n}</span>
      <span class="ptitle">{t}</span><span class="psub">{sub}</span></figcaption>
      <div class="pimg"><img src="{IMG[k]}" alt="{t}" loading="lazy"></div></figure>''' for n,t,sub,k in order)
    prov_rows=[("Material",material),("Phase",f"SG {meta['spacegroup']} · a {meta['nominal_lat'][0]:.4f} Å" if meta['nominal_lat'] else "—"),
               ("Wavelength",prov.get("Wavelength","—")),("Lsd (µm)",prov.get("Lsd","—")),
               ("Detector",f"{prov.get('NrPixels','?')}² · {prov.get('px','?')} µm px"),("Completeness ≥",prov.get("Completeness","—")),
               ("Backend","c-omp index + refine"),("Source",gpath)]
    prov_html="".join(f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k,v in prov_rows)
    road="".join(f'<div class="find"><span class="tag warn">next</span><h3>{t}</h3><p>{b}</p></div>' for t,b in F["roadmap"]) or '<p style="color:var(--mut)">No systematics flagged above threshold.</p>'
    return string.Template(TEMPLATE).substitute(title=title,tiles=tiles_html,prov=prov_html,
        solid=finds(F["solid"],"","solid"),warn=finds(F["warn"],"warn","watch"),
        road=road,figs=figs_html,nfig=len(order),gpath=gpath)

TEMPLATE=r'''<style>
:root{--paper:#f4f2ee;--panel:#fff;--ink:#16191c;--ink2:#3d4045;--mut:#6b6862;--line:#e4e0d8;--hair:#efece5;
--teal:#0e7c86;--teal-ink:#0b5f67;--copper:#b06f33;--plate:#f7f6f3;--plate-line:#d7d3ca;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root{--paper:#101315;--panel:#181c20;--ink:#e9e7e2;--ink2:#c3c1ba;--mut:#8f8d85;--line:#282d33;--hair:#20252a;--teal:#33aab9;--teal-ink:#7fd0da;--copper:#d69157;}}
:root[data-theme="dark"]{--paper:#101315;--panel:#181c20;--ink:#e9e7e2;--ink2:#c3c1ba;--mut:#8f8d85;--line:#282d33;--hair:#20252a;--teal:#33aab9;--teal-ink:#7fd0da;--copper:#d69157;}
:root[data-theme="light"]{--paper:#f4f2ee;--panel:#fff;--ink:#16191c;--ink2:#3d4045;--mut:#6b6862;--line:#e4e0d8;--hair:#efece5;--teal:#0e7c86;--teal-ink:#0b5f67;--copper:#b06f33;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:1200px;margin:0 auto;padding:clamp(20px,4vw,52px) clamp(16px,4vw,40px) 80px}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:650}
h1{font-size:clamp(27px,4.2vw,44px);line-height:1.05;margin:.28em 0 .12em;letter-spacing:-.02em;text-wrap:balance;font-weight:750}
.lede{font-size:clamp(15px,1.7vw,18px);color:var(--ink2);max-width:70ch}
.prov{margin-top:24px;border:1px solid var(--line);border-radius:12px;background:var(--panel);display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;overflow:hidden}
.prov div{padding:12px 16px;background:var(--panel)}.prov dt{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 3px}
.prov dd{margin:0;font-family:var(--mono);font-size:12.5px;color:var(--ink);word-break:break-word}
.sec{margin-top:50px}.sec-h{display:flex;align-items:baseline;gap:12px;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid var(--line)}
.sec-h h2{font-size:20px;margin:0;font-weight:700}.sec-h .k{font-family:var(--mono);font-size:12px;color:var(--teal)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}
.tile{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:16px;position:relative;overflow:hidden}
.tile::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--teal);opacity:.85}
.tk{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}.tv{font-size:25px;font-weight:720;margin-top:6px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.ts{font-size:12px;color:var(--ink2);margin-top:4px;font-variant-numeric:tabular-nums}
.finds{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.find{border:1px solid var(--line);border-left:3px solid var(--teal);border-radius:0 12px 12px 0;background:var(--panel);padding:15px 18px}
.find.warn{border-left-color:var(--copper)}.find h3{margin:0 0 5px;font-size:14.5px;font-weight:680}.find p{margin:0;font-size:13.5px;color:var(--ink2)}
.tag{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:2px 7px;border-radius:5px;background:color-mix(in srgb,var(--teal) 16%,transparent);color:var(--teal-ink);margin-bottom:8px}
.tag.warn{background:color-mix(in srgb,var(--copper) 20%,transparent);color:var(--copper)}
.plate{margin:0 0 30px;border:1px solid var(--line);border-radius:14px;background:var(--panel);overflow:hidden}
.pcap{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;padding:16px 20px 14px;border-bottom:1px solid var(--hair)}
.pnum{font-family:var(--mono);font-size:12px;color:var(--teal);font-weight:600}.ptitle{font-size:16px;font-weight:700}.psub{font-size:12.5px;color:var(--mut)}
.pimg{background:var(--plate);padding:14px;border-top:1px solid var(--plate-line);overflow-x:auto}
.pimg img{display:block;width:100%;max-width:100%;height:auto;min-width:640px;margin:0 auto}
footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--mut);font-size:12px}
footer .fp{font-family:var(--mono);font-size:11.5px;word-break:break-all}
</style>
<div class="wrap">
<header><div class="eyebrow">MIDAS · Far-Field HEDM · Reconstruction Report</div>
<h1>${title}</h1>
<p class="lede">Far-field HEDM reconstruction — peak search → indexing → per-grain lattice refinement on the MIDAS c-omp backend. Numbers read directly from Grains.csv and the process-grains diagnostics; findings are auto-derived from the residuals.</p>
<dl class="prov">${prov}</dl></header>
<section class="sec"><div class="sec-h"><h2>Summary</h2><span class="k">Grains.csv</span></div><div class="tiles">${tiles}</div></section>
<section class="sec"><div class="sec-h"><h2>What the data says</h2><span class="k">interpretation</span></div>
<div class="finds">${solid}${warn}</div></section>
<section class="sec"><div class="sec-h"><h2>Improvement roadmap</h2><span class="k">diagnosis → lever</span></div><div class="finds">${road}</div></section>
<section class="sec"><div class="sec-h"><h2>Figures</h2><span class="k">${nfig} plates</span></div>${figs}</section>
<footer><div class="fp">${gpath}</div><div style="margin-top:6px">MIDAS c-omp FF-HEDM · generated by midas_ff_report.py</div></footer>
</div>'''

if __name__=="__main__": main()
