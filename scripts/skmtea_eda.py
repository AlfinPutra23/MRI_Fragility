"""EDA on the SKM-TEA raw-k-space cases (real multicoil qDESS). Lazy h5 reads (never loads the full 10GB kspace).
Produces 4 figures + a stats dump. -> outputs/plots/skmtea_eda_*.png , outputs/results/skmtea_eda.json"""
import os, sys, json, glob, argparse, numpy as np, h5py, nibabel as nib
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Nimbus Sans","Liberation Sans","DejaVu Sans"]})
PLOTS="outputs/plots"; RES="outputs/results"; os.makedirs(PLOTS,exist_ok=True); os.makedirs(RES,exist_ok=True)

# SKM-TEA 6-class knee segmentation (standard order)
LAB={1:"patellar cart.",2:"femoral cart.",3:"tibial cart. (med)",4:"tibial cart. (lat)",5:"meniscus (med)",6:"meniscus (lat)"}
LCOL=["#00000000","#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]  # 0=transparent + 6 colors

ap=argparse.ArgumentParser(); ap.add_argument("--case",default="MTR_001"); a=ap.parse_args()
h5p=glob.glob(f"data/skmtea/kspace/**/{a.case}.h5",recursive=True)[0]
segp=f"data/skmtea/seg/{a.case}_raw-data-track.nii.gz"
print(f"case {a.case}\n  h5  {h5p}\n  seg {segp}")

f=h5py.File(h5p,"r")
KS=f["kspace"]; MAPS=f["maps"]; TGT=f["target"]                       # lazy datasets
X,Y,Z,E,C=KS.shape
seg=np.asanyarray(nib.load(segp).dataobj).astype(np.int16)
# choose the slice (along Z) with the most segmented voxels
zc=int(np.argmax([(seg[:,:,z]>0).sum() for z in range(Z)]))
print(f"  volume {X}x{Y}x{Z}, {E} echoes, {C} coils; richest seg slice z={zc}")

def norm(im,p=99.5):
    im=np.abs(im).astype(np.float32); hi=np.percentile(im,p)+1e-8; return np.clip(im/hi,0,1)
tgt=norm(TGT[:,:,zc,0,0])                                             # coil-combined reference, echo 0
segz=seg[:,:,zc]
cmap=ListedColormap(LCOL)

# ---- stats ----
vox=nib.load(segp).header.get_zooms()[:3]; vvol=float(np.prod(vox))
stats={"case":a.case,"kspace_shape":list(KS.shape),"maps_shape":list(MAPS.shape),"slice_z":zc,
       "voxel_mm":[float(v) for v in vox],
       "structures":{LAB[i]:{"voxels":int((seg==i).sum()),"volume_mm3":round(float((seg==i).sum()*vvol),1)} for i in LAB if (seg==i).any()},
       "official_masks":[k for k in f["masks"]]}
json.dump(stats,open(f"{RES}/skmtea_eda.json","w"),indent=2)
print("  structures:",{k:v["voxels"] for k,v in stats["structures"].items()})

# ============ FIG 1: anatomy overview + segmentation overlay + zoom ============
fig,ax=plt.subplots(1,3,figsize=(15,5.4))
ax[0].imshow(tgt.T,cmap="gray",origin="lower"); ax[0].set_title(f"{a.case} — qDESS reference (echo 0)",fontweight="bold")
ax[1].imshow(tgt.T,cmap="gray",origin="lower")
ax[1].imshow(np.ma.masked_where(segz.T==0,segz.T),cmap=cmap,vmin=0,vmax=6,alpha=.6,origin="lower")
ax[1].set_title("+ 6-structure segmentation",fontweight="bold")
ys,xs=np.where(segz>0); y0,y1,x0,x1=xs.min()-15,xs.max()+15,ys.min()-15,ys.max()+15   # zoom box (note transpose)
ax[2].imshow(tgt.T[y0:y1,x0:x1],cmap="gray",origin="lower")
ax[2].imshow(np.ma.masked_where(segz.T[y0:y1,x0:x1]==0,segz.T[y0:y1,x0:x1]),cmap=cmap,vmin=0,vmax=6,alpha=.65,origin="lower")
ax[2].set_title("zoom: thin cartilage & meniscus",fontweight="bold")
for x in ax: x.axis("off")
handles=[plt.Line2D([0],[0],marker="s",ls="",mfc=LCOL[i],mec="none",ms=11,label=LAB[i]) for i in LAB if (seg==i).any()]
fig.legend(handles=handles,loc="lower center",ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(.5,-.02))
fig.suptitle(f"SKM-TEA real k-space case {a.case}: anatomy + downstream structures",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(f"{PLOTS}/skmtea_eda_overview.png",dpi=140,bbox_inches="tight"); plt.close(fig)

# ============ FIG 2: k-space energy + official Poisson masks ============
ksl=np.fft.fftshift(np.fft.fft2(TGT[:,:,zc,0,0]))                     # true k-space of this plane
fig,axs=plt.subplots(2,4,figsize=(16,8))
axs[0,0].imshow(np.log(np.abs(ksl).T+1),cmap="magma",origin="lower"); axs[0,0].set_title("k-space log-magnitude\n(energy piles at the center)",fontweight="bold")
# radial energy profile
cy,cx=np.array(ksl.shape)//2; yy,xx=np.indices(ksl.shape); r=np.sqrt((yy-cy)**2+(xx-cx)**2).astype(int)
rad=np.bincount(r.ravel(),weights=(np.abs(ksl)**2).ravel())/ (np.bincount(r.ravel())+1e-9)
axs[0,1].semilogy(rad[:min(cx,cy)]); axs[0,1].set_title("radial energy vs frequency",fontweight="bold")
axs[0,1].set_xlabel("radius |k| (px)"); axs[0,1].set_ylabel("mean energy"); axs[0,1].grid(alpha=.3)
axs[0,2].axis("off"); axs[0,3].axis("off")
masks=sorted(f["masks"].keys(), key=lambda s:float(s.split("_")[1].replace("x","")))
for i,mk in enumerate(masks[:4]):
    m=f["masks"][mk][:]; axs[1,i].imshow(m,cmap="gray",aspect="auto"); frac=100*m.mean()
    axs[1,i].set_title(f"official {mk}\nkept {frac:.0f}%",fontweight="bold"); axs[1,i].axis("off")
fig.suptitle(f"SKM-TEA {a.case}: k-space structure + the official Poisson undersampling masks",fontsize=14,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_eda_kspace_masks.png",dpi=140,bbox_inches="tight"); plt.close(fig)

# ============ FIG 3: coil sensitivity maps (16 coils) ============
mp=np.abs(MAPS[:,:,zc,:,0])                                           # (X,Y,16)
fig,axs=plt.subplots(4,4,figsize=(12,12))
for c in range(C):
    axs[c//4,c%4].imshow(norm(mp[:,:,c]).T,cmap="gray",origin="lower"); axs[c//4,c%4].set_title(f"coil {c+1}",fontsize=9); axs[c//4,c%4].axis("off")
fig.suptitle(f"SKM-TEA {a.case}: 16 ESPIRiT coil sensitivity maps (slice z={zc})",fontsize=14,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_eda_coils.png",dpi=130,bbox_inches="tight"); plt.close(fig)

# ============ FIG 4: retrospective variable-density undersampling demo (our operator) ============
def vd_cols(W,R,acs=0.08):
    m=np.zeros(W,bool); c=W//2; na=max(1,int(acs*W)); m[c-na//2:c+na//2+1]=True
    freq=np.arange(W)-c; p=1/(np.abs(freq)+1); p[m]=0; p/=p.sum()
    need=max(0,W//R-m.sum())
    if need>0: m[np.random.choice(W,size=need,replace=False,p=p)]=True
    return m
def undersample(img,R):
    if R==1: return img
    F=np.fft.fftshift(np.fft.fft2(img)); m=vd_cols(img.shape[1],R)
    F[:,~m]=0; return np.abs(np.fft.ifft2(np.fft.ifftshift(F)))
np.random.seed(0)
img=TGT[:,:,zc,0,0]
fig,axs=plt.subplots(1,3,figsize=(15,5.6))
for i,R in enumerate([1,4,8]):
    u=undersample(img,R); u=norm(u)
    axs[i].imshow(u.T[y0:y1,x0:x1],cmap="gray",origin="lower")
    cont=np.ma.masked_where(segz.T[y0:y1,x0:x1]==0,segz.T[y0:y1,x0:x1])
    axs[i].contour(cont,levels=[0.5,1.5,2.5,3.5,4.5,5.5],colors="#00e0ff",linewidths=.8)
    axs[i].set_title(("full scan (R=1)" if R==1 else f"accelerated R={R}"),fontweight="bold"); axs[i].axis("off")
fig.suptitle(f"SKM-TEA {a.case}: retrospective variable-density undersampling — watch the thin cartilage blur",fontsize=14,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_eda_undersample.png",dpi=140,bbox_inches="tight"); plt.close(fig)

f.close()
print("wrote: skmtea_eda_overview / _kspace_masks / _coils / _undersample .png  + skmtea_eda.json")
