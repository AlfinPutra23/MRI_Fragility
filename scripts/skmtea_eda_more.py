"""Richer SKM-TEA EDA: the two qDESS echoes, and a multi-slice montage. Lazy h5 reads.
-> outputs/plots/skmtea_eda_echoes.png , skmtea_eda_slices.png"""
import os, glob, argparse, numpy as np, h5py, nibabel as nib
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Nimbus Sans","Liberation Sans","DejaVu Sans"]})
PLOTS="outputs/plots"; os.makedirs(PLOTS,exist_ok=True)
LAB={1:"patellar cart.",2:"femoral cart.",3:"tibial cart. (med)",4:"tibial cart. (lat)",5:"meniscus (med)",6:"meniscus (lat)"}
LCOL=["#00000000","#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]; CM=ListedColormap(LCOL)

ap=argparse.ArgumentParser(); ap.add_argument("--case",default="MTR_001"); a=ap.parse_args()
h5p=glob.glob(f"data/skmtea/kspace/**/{a.case}.h5",recursive=True)[0]
seg=np.asanyarray(nib.load(f"data/skmtea/seg/{a.case}_raw-data-track.nii.gz").dataobj).astype(np.int16)
f=h5py.File(h5p,"r"); TGT=f["target"]; X,Y,Z,E,C=f["kspace"].shape
def norm(im,p=99.5): im=np.abs(im).astype(np.float32); return np.clip(im/(np.percentile(im,p)+1e-8),0,1)
def ov(ax,img,sg):
    ax.imshow(img.T,cmap="gray",origin="lower")
    ax.imshow(np.ma.masked_where(sg.T==0,sg.T),cmap=CM,vmin=0,vmax=6,alpha=.6,origin="lower"); ax.axis("off")

counts=[(seg[:,:,z]>0).sum() for z in range(Z)]; zc=int(np.argmax(counts))

# ===== FIG A: the two qDESS echoes (same slice) =====
fig,axs=plt.subplots(1,3,figsize=(15,5.6))
e0,e1=norm(TGT[:,:,zc,0,0]),norm(TGT[:,:,zc,1,0])
axs[0].imshow(e0.T,cmap="gray",origin="lower"); axs[0].set_title("echo 1  (bright fluid / less T2-weight)",fontweight="bold"); axs[0].axis("off")
axs[1].imshow(e1.T,cmap="gray",origin="lower"); axs[1].set_title("echo 2  (more T2-weight)",fontweight="bold"); axs[1].axis("off")
ov(axs[2],e1,seg[:,:,zc]); axs[2].set_title("echo 2 + segmentation",fontweight="bold")
fig.suptitle(f"SKM-TEA {a.case}: qDESS acquires TWO echoes per scan (different tissue contrast)",fontsize=14,fontweight="bold")
h=[plt.Line2D([0],[0],marker="s",ls="",mfc=LCOL[i],mec="none",ms=11,label=LAB[i]) for i in LAB if (seg==i).any()]
fig.legend(handles=h,loc="lower center",ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(f"{PLOTS}/skmtea_eda_echoes.png",dpi=140,bbox_inches="tight"); plt.close(fig)

# ===== FIG B: multi-slice montage across the volume (only slices that contain structures) =====
zs=[z for z in range(Z) if counts[z]>200]
pick=np.linspace(zs[0],zs[-1],6).astype(int)
fig,axs=plt.subplots(2,3,figsize=(15,10))
for ax,z in zip(axs.ravel(),pick):
    ov(ax,norm(TGT[:,:,z,1,0]),seg[:,:,z]); ax.set_title(f"slice z={z}  ({(seg[:,:,z]>0).sum()} vox)",fontweight="bold",fontsize=11)
fig.suptitle(f"SKM-TEA {a.case}: the cartilage/meniscus across the knee (echo 2, 6 slices)",fontsize=14,fontweight="bold")
fig.legend(handles=h,loc="lower center",ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(.5,-.01))
fig.tight_layout(rect=[0,.03,1,1]); fig.savefig(f"{PLOTS}/skmtea_eda_slices.png",dpi=130,bbox_inches="tight"); plt.close(fig)
f.close()
print(f"slice range with structures: z={zs[0]}..{zs[-1]}  (n={len(zs)}); montage slices {list(pick)}")
print("wrote: skmtea_eda_echoes.png , skmtea_eda_slices.png")
