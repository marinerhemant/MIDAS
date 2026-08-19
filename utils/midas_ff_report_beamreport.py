#!/usr/bin/env python
"""midas_ff_report.py — self-contained HTML report from a MIDAS FF-HEDM result dir.

A beamreport adapter. This file supplies what is specific to FF-HEDM — how to read
Grains.csv and the process-grains diagnostics, how to draw the plates, and the handful
of findings that depend on HEDM quantities. Everything technique-independent (residual
diagnostics, diagnosis-reference matching, page assembly, refusals) comes from
beamreport.

Usage:
    python midas_ff_report.py RUN_DIR [--material NAME] [--title T] [--out report.html]
                              [--c11 GPa --c12 GPa] [--beam-height UM]
"""
import argparse, glob, os, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from beamreport import Finding, Plate, Provenance, Quality, Results, Sidecar, build

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
    """Pull provenance from paramstest.txt / *ParamFile*.txt if present.

    Candidates are MERGED, earliest-wins, rather than stopping at the first file
    that yields anything. The per-layer ``paramstest.txt`` the C stages consume is
    STRIPPED: it carries ``Distance`` (not ``Lsd``) and drops ``BC``,
    ``Completeness`` and ``RingThresh`` entirely. Since it is also the first
    candidate, an early break meant every real run rendered "Lsd ?" in the
    provenance strip while the master parameter file sitting next to it had the
    value. ``k not in p`` already gives the per-layer file precedence, so merging
    only fills gaps.

    ``Distance`` is accepted as an alias for ``Lsd`` for the same reason.
    """
    ALIAS={"Distance":"Lsd"}
    WANT=("Wavelength","Lsd","px","NrPixels","OmegaStep","OmegaStart","Completeness")
    p={}
    cands=glob.glob(os.path.join(run_dir,"paramstest.txt"))+glob.glob(os.path.join(run_dir,"*ParamFile*.txt"))\
          +glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")),"*ParamFile*.txt"))
    for f in cands:
        for ln in open(f):
            ln=ln.split("#",1)[0]
            t=ln.split()
            if len(t)<2: continue
            k=ALIAS.get(t[0],t[0])
            if k in WANT and k not in p:
                p[k]=t[1].rstrip(";")
            if k=="BC" and "BC" not in p: p["BC"]=f"{t[1].rstrip(chr(59))} {t[2].rstrip(chr(59))}"
            if k=="RingThresh": p.setdefault("rings",[]).append(t[1])
    return p

def is_cubic(sg):  return sg is not None and 195<=sg<=230

def ipf_cubic(OM):
    v=np.abs(np.einsum('nij,j->ni',OM,np.array([0,0,1.0])))
    v=np.sort(v,axis=1); v/=np.linalg.norm(v,axis=1,keepdims=True)
    rgb=np.stack([v[:,2]-v[:,1],v[:,1]-v[:,0],v[:,0]],1)
    rgb/=rgb.max(axis=1,keepdims=True); return np.clip(rgb,0,1)**0.6

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

def fig_per_grain_residuals(C, S, outp):
    """Per-SPOT DiffPos and DiffOme, resolved PER GRAIN.

    The grain-level DiffPos in Grains.csv is a MEAN over that grain's matched
    spots, so a small tail of badly-matched spots dominates it (the same trap
    hard rule 6 names for calibrant strain). Measured on 20-ID Au: the core sits
    at ~161 um (about one spot width) while a ~8% tail reaches >1 mm, and 91% of
    those tail spots had a CLOSER spot available -- i.e. they are
    mis-assignments, not misfit. Plotting the per-spot distribution per grain is
    the only way to see that; the mean hides it.
    """
    if S is None:
        fig, ax = plt.subplots(figsize=(7, 2)); ax.axis("off")
        ax.text(.5, .5, "no residual sidecar", ha="center", color=MUT)
        fig.savefig(outp, bbox_inches="tight"); plt.close(fig); return
    gi = S["grain_idx"].astype(int)
    dlen = np.hypot(S["dy_um"], S["dz_um"])
    dome = S["dome_deg"]
    gids = np.unique(gi)
    order = np.argsort([np.median(dlen[gi == g]) for g in gids])
    gids = gids[order]
    show = gids if len(gids) <= 24 else gids[np.linspace(0, len(gids)-1, 24).astype(int)]

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.6))
    # (0,0) per-grain violin/strip of per-spot DiffPos
    data = [dlen[gi == g] for g in show]
    pos = np.arange(len(show))
    parts = ax[0,0].violinplot(data, positions=pos, widths=0.85, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(TEAL); b.set_alpha(0.35); b.set_edgecolor("none")
    for i, v in enumerate(data):
        ax[0,0].plot(np.full(len(v), i) + (np.random.default_rng(i).random(len(v))-.5)*0.28,
                     v, ".", ms=2.0, color=INK, alpha=0.45)
        ax[0,0].plot([i-0.42, i+0.42], [np.median(v)]*2, "-", color=COPPER, lw=2)
    ax[0,0].set_xticks(pos); ax[0,0].set_xticklabels([str(int(g)) for g in show],
                                                     rotation=90, fontsize=7)
    ax[0,0].set_ylabel("per-spot |Δpos| (µm)"); ax[0,0].set_xlabel("grain")
    ax[0,0].set_title("DiffPos per spot, by grain — copper = median", fontsize=10)

    # (0,1) the pooled distribution, log-x, with core/tail split
    allv = dlen[np.isfinite(dlen) & (dlen > 0)]
    ax[0,1].hist(allv, bins=np.logspace(np.log10(max(allv.min(),1)),
                                        np.log10(allv.max()), 70),
                 color=TEAL, alpha=.85)
    med = np.median(allv)
    ax[0,1].axvline(med, color=COPPER, lw=1.5, ls="--", label=f"median {med:.0f} µm")
    ax[0,1].axvline(allv.mean(), color=INK, lw=1.5, ls=":",
                    label=f"MEAN {allv.mean():.0f} µm  ← DiffPos")
    ax[0,1].set_xscale("log"); ax[0,1].set_xlabel("per-spot |Δpos| (µm)")
    ax[0,1].set_ylabel("spots"); ax[0,1].legend(fontsize=8, frameon=False)
    ax[0,1].set_title("pooled — the mean sits far above the median", fontsize=10)

    # (1,0) per-grain DiffOme
    dd = [dome[gi == g] for g in show]
    parts = ax[1,0].violinplot(dd, positions=pos, widths=0.85, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(COPPER); b.set_alpha(0.35); b.set_edgecolor("none")
    for i, v in enumerate(dd):
        ax[1,0].plot(np.full(len(v), i) + (np.random.default_rng(i).random(len(v))-.5)*0.28,
                     v, ".", ms=2.0, color=INK, alpha=0.45)
    ax[1,0].axhline(0, color=MUT, lw=.8)
    ax[1,0].set_xticks(pos); ax[1,0].set_xticklabels([str(int(g)) for g in show],
                                                     rotation=90, fontsize=7)
    ax[1,0].set_ylabel("per-spot Δω (deg)"); ax[1,0].set_xlabel("grain")
    ax[1,0].set_title("DiffOme per spot, by grain", fontsize=10)

    # (1,1) cumulative share of the MEAN carried by the worst spots
    sv = np.sort(allv)[::-1]
    frac = np.arange(1, len(sv)+1) / len(sv)
    ax[1,1].plot(100*frac, 100*np.cumsum(sv)/sv.sum(), color=TEAL, lw=1.8)
    for f in (0.05, 0.10):
        k = max(1, int(f*len(sv)))
        ax[1,1].plot([100*f], [100*sv[:k].sum()/sv.sum()], "o", color=COPPER, ms=5)
        ax[1,1].annotate(f"worst {100*f:.0f}% carry {100*sv[:k].sum()/sv.sum():.0f}%",
                         (100*f, 100*sv[:k].sum()/sv.sum()), textcoords="offset points",
                         xytext=(8, -10), fontsize=8, color=INK)
    ax[1,1].plot([0,100],[0,100], "--", color=MUT, lw=.8)
    ax[1,1].set_xlabel("worst N% of spots"); ax[1,1].set_ylabel("% of the summed |Δpos|")
    ax[1,1].set_title("how concentrated is the error?", fontsize=10)
    fig.tight_layout(); fig.savefig(outp, bbox_inches="tight"); plt.close(fig)


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



# ------------------------------------------------------------------ adapter
SPOT_COLS = ("grain_idx spot_id ring_nr eta_deg dy_um dz_um drad_um dtan_um dome_deg "
             "internal_angle_deg r_exp_um").split()
SPOT_UNITS = ["", "", "1", "deg", "um", "um", "um", "um", "deg", "deg", "um"]
SPOT_ROLES = ["id", "aux", "coord", "coord", "residual", "residual", "residual",
              "residual", "residual", "aux", "coord"]

# Result columns worth exposing to the generic diagnostics: each one gets tested for
# correlation against its own residual, which is how a runaway position fit is caught.
RESULT_COLS = {"X": "um", "Y": "um", "Z": "um", "a": "angstrom",
               "GrainRadius": "um", "DiffPos": "um", "RMSErrorStrain": "ue"}


def midas_findings(C, D, S, d0, nring):
    """The findings beamreport cannot derive: they depend on HEDM-specific quantities."""
    out = []
    if D.get("overall_med_internal_angle_deg") is not None:
        ia = float(D["overall_med_internal_angle_deg"])
        dm = float(D.get("overall_mad_dome_deg", 0))
        out.append(Finding(symptom="", level="solid", title="Angular fit",
            statement=f"Median internal angle {ia:.2f}°, dome MAD {dm:.2f}° — "
                      f"orientation is the most trustworthy product."))
    if nring and nring < 6:
        out.append(Finding(symptom="", level="caution", title="Ring coverage",
            statement=f"Only {nring} rings indexed. The strain tensor is poorly "
                      f"conditioned on this few rings; add higher-angle unsaturated rings."))
    if d0:
        bias = f", removes ≈{abs(d0['eps'] * d0['mpa']):.0f} MPa bias" if d0["mpa"] else ""
        out.append(Finding(symptom="", level="solid", title="d0 reference calibrated",
            statement=f"a0 = {d0['a0']:.5f} Å ({d0['eps']:+.0f} µε); hydrostatic strain "
                      f"re-centred{bias}. Deviatoric strain unchanged."))
    out.append(Finding(symptom="", level="caution", title="Strain scatter",
        statement=f"Per-grain RMS strain residual median "
                  f"{np.median(C['RMSErrorStrain']):.0f} µε — treat individual tensors as "
                  f"indicative. The bias (d0) is fixable; the scatter is geometry-limited."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--material", default=None)
    ap.add_argument("--title", default=None); ap.add_argument("--out", default="report.html")
    ap.add_argument("--c11", type=float); ap.add_argument("--c12", type=float)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--beam-height", type=float, default=None, dest="beam")
    # The diagnosis reference lives with the FF doc set, not beside this script:
    # beamreport's ADAPTER.md §5 keeps it in the technique's own repository, and
    # the doc set is where it gets maintained alongside the traps it encodes.
    ap.add_argument("--reference",
                    default=str(Path(__file__).resolve().parent.parent
                                / "manuals" / "ff-hedm" / "DIAGNOSIS.md"))
    # The envelope outranks the reference on levers. Without it the report will happily
    # recommend recalibrating something the doc set already records as a fixed property
    # of the station -- which is exactly what it did on Au3 before this was wired.
    ap.add_argument("--envelope",
                    default=str(Path(__file__).resolve().parent.parent
                                / "manuals" / "ff-hedm" / "ENVELOPE.md"))
    a = ap.parse_args()

    rd = a.run_dir
    figdir = a.figdir or os.path.join(rd, "_report_figs"); os.makedirs(figdir, exist_ok=True)
    gpath = os.path.join(rd, "Grains.csv"); meta = parse_grains_header(gpath)
    nom_a = meta["nominal_lat"][0] if meta["nominal_lat"] else float("nan")
    sg = meta["spacegroup"]; cubic = is_cubic(sg)
    d = np.loadtxt(gpath, comments="%")
    if d.ndim == 1: d = d[None, :]
    C = {n: d[:, i] for i, n in enumerate(GCOLS)}; OM = d[:, 1:10].reshape(-1, 3, 3)
    material = a.material or f"SG{sg}"; ngr = len(d)
    prov = parse_param(rd)

    D = {}; S = None
    dpath = os.path.join(rd, "processgrains_diagnostics.h5")
    if os.path.exists(dpath):
        import h5py
        with h5py.File(dpath, "r") as f:
            for grp in ("residuals", "diagnostics"):
                if grp in f:
                    for k in f[grp]:
                        if k == "spot_table": continue
                        try: D[k] = f[f"{grp}/{k}"][()]
                        except Exception: pass
            if "residuals/spot_table" in f:
                st = f["residuals/spot_table"][()]
                S = {n: st[:, i] for i, n in enumerate(SPOT_COLS)}

    d0 = None
    if cubic:
        try:
            import midas_stress as ms
            latc = d[:, 13:19]; ref = np.array([nom_a] * 3 + [90.] * 3); vol = C["GrainRadius"] ** 3
            r = ms.recover_d0_cubic_free_standing(latc, ref, volumes=vol,
                                                 confidences=C["Confidence"], min_confidence=0.5)
            a0 = float(np.asarray(r["reference_recovered"]).ravel()[0]); eps = float(r["eps_iso"]) * 1e6
            def hydro(rr):
                e = np.array([ms.lattice_params_to_strain(latc[i], rr)
                              for i in range(len(latc))]).reshape(len(latc), -1)
                return ((e[:, 0] + e[:, 1] + e[:, 2]) / 3.0) * 1e6
            m = C["Confidence"] >= 0.5
            mpa = ms.d0_sensitivity(C11=a.c11, C12=a.c12)["sensitivity_MPa_per_ppm"] \
                if (a.c11 and a.c12) else None
            d0 = dict(a0=a0, eps=eps, hb=hydro(ref), ha=hydro(np.array([a0] * 3 + [90.] * 3)),
                      m=m, mpa=mpa, n=int(m.sum()))
        except Exception as e:
            print("d0 skipped:", e)

    plates = []
    def plate(key, fn, title, caption, spatial=False):
        p = os.path.join(figdir, f"{key}.png"); fn(p)
        plates.append(Plate(p, title, caption, spatial=spatial, aspect_equal=True if spatial else None))
    plate("grain_maps",  lambda p: fig_grain_maps(C, OM, cubic, p),
          "Grain centroid maps", "X–Y / X–Z / Y–Z, IPF-Z coloured, sized by radius.", spatial=True)
    plate("error_maps",  lambda p: fig_error_maps(C, D, p),
          "Per-grain error and quality maps", "Completeness, spots, internal angle, residual, DiffPos, strain error.")
    plate("residuals",   lambda p: fig_residuals(D, S, p),
          "Residual diagnostics", "Per-ring Δr, residual vs η, Δrad–Δtan density, angle histograms.")
    plate("per_grain_resid", lambda p: fig_per_grain_residuals(C, S, p),
          "DiffPos and DiffOme per spot, by grain",
          "Grains.csv reports the MEAN over each grain's spots, so a small tail "
          "dominates it. These are the underlying per-spot distributions.")
    plate("strain",      lambda p: fig_strain(C, nom_a, D, p),
          "Strain, lattice and quality", "Hydrostatic map, component boxplots, lattice-a, completeness.")
    plate("grain_hists", lambda p: fig_grain_error_hists(C, D, p),
          "Per-grain error distributions", "Histograms per metric.")
    plate("position",    lambda p: fig_position_diag(C, D, p, beam_um=a.beam),
          "Position diagnostic", "Is the fitted Z supported by the data?")
    if d0:
        plate("d0_calib", lambda p: _fig_d0(d0, p),
              "d0 strain-free reference", "Before / after hydrostatic strain; recovered a₀ via midas-stress.")

    # The sidecar's `grain_idx` is a POSITIONAL index into the grain array (0..N-1),
    # not the indexer's grain ID -- on Au3 those are [0,1] against IDs [71,156].
    # Joining on ID silently orphans every observation, so the object_id is the row
    # position (verified: residuals/grain_n_spots is in grain_idx order and matches
    # the per-idx spot counts) and the real ID rides along as a result column.
    results = Results(
        object_id=np.arange(ngr),
        columns={"ID": (C["ID"], "1"), **{k: (C[k], u) for k, u in RESULT_COLS.items()}},
    )
    quality = Quality(values=C["Confidence"], name="completeness", threshold=0.5)
    provenance = Provenance(
        inputs=[gpath] + ([dpath] if os.path.exists(dpath) else []),
        command=" ".join(sys.argv), parameters=prov.get("_file"),
        code_version=f"MIDAS c-omp · SG {sg} · λ {prov.get('Wavelength', '?')} Å · "
                     f"Lsd {prov.get('Lsd', '?')} µm")
    sidecar = None
    if S is not None:
        sidecar = Sidecar(table=np.column_stack([S[c] for c in SPOT_COLS]),
                          columns=SPOT_COLS, units=SPOT_UNITS, roles=SPOT_ROLES)

    nring = (len(D.get("ring_nr", [])) if D.get("ring_nr") is not None
             else (len(np.unique(S["ring_nr"])) if S is not None else 0))
    bounds = {"Z": (-a.beam / 2, a.beam / 2)} if a.beam else None

    out = build(
        results=results, quality=quality, provenance=provenance, sidecar=sidecar,
        figures=plates, bounds=bounds,
        diagnosis_reference=a.reference if os.path.exists(a.reference) else None,
        envelope=a.envelope if os.path.exists(a.envelope) else None,
        title=a.title or f"{material} — Far-Field HEDM reconstruction",
        subtitle="Peak search → indexing → per-grain lattice refinement on the MIDAS c-omp "
                 "backend. Numbers read directly from Grains.csv and the process-grains "
                 "diagnostics; findings are derived from the residuals.",
        extra_findings=midas_findings(C, D, S, d0, nring),
        out=a.out,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
