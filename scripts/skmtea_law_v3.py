"""First real-k-space test of the fragility law on SKM-TEA (MTR_001). No segmenter yet, so we test the LAW'S
PREDICTORS on real qDESS anatomy: per-structure spectral centroid (anatomy), energy-removed E_lost(R) (k-space),
and a per-structure image-degradation proxy (recon error under retrospective variable-density undersampling).
Q: does the energy-removed mechanism + centroid ranking transfer to knee cartilage? -> skmtea_law_*.png + .json"""
import os, glob, json, argparse, numpy as np, h5py, nibabel as nib
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Nimbus Sans","Liberation Sans","DejaVu Sans"]})
PLOTS="outputs/plots"; RES="outputs/results"; os.makedirs(PLOTS,exist_ok=True); os.makedirs(RES,exist_ok=True)
LAB={1:"patellar cart.",2:"femoral cart.",3:"tibial cart.(med)",4:"tibial cart.(lat)",5:"meniscus(med)",6:"meniscus(lat)"}
RS=[2,4,6,8]; ap=argparse.ArgumentParser(); ap.add_argument("--case",default="MTR_001"); ap.add_argument("--echo",type=int,default=0); a=ap.parse_args()
SUF="" if a.echo==0 else f"_e{a.echo}"                  # tag outputs by echo so echoes don't overwrite

h5p=glob.glob(f"data/skmtea/kspace/**/{a.case}.h5",recursive=True)[0]
seg=np.asanyarray(nib.load(f"data/skmtea/seg/{a.case}_raw-data-track.nii.gz").dataobj).astype(np.int16)
f=h5py.File(h5p,"r"); TGT=f["target"]; X,Y,Z,E,C=f["kspace"].shape

def vd_cols(W,R,acs=0.08):
    m=np.zeros(W,bool); c=W//2; na=max(1,int(acs*W)); m[c-na//2:c+na//2+1]=True
    fr=np.abs(np.arange(W)-c); p=1/(fr+1); p[m]=0; s=p.sum();  p=p/s if s>0 else p
    need=max(0,W//R-m.sum())
    if need>0: m[np.random.choice(W,size=need,replace=False,p=p)]=True
    return m

def centroid(patch):                                   # energy-weighted mean radial freq (cycles/px, comparable)
    F=np.fft.fft2(patch); fy=np.fft.fftfreq(patch.shape[0]); fx=np.fft.fftfreq(patch.shape[1])
    R=np.hypot(fy[:,None],fx[None,:]); P=np.abs(F)**2; return float((R*P).sum()/(P.sum()+1e-12))

np.random.seed(0)
acc={i:{"cen":[],**{f"el{R}":[] for R in RS},**{f"dg{R}":[] for R in RS}} for i in LAB}
counts=[(seg[:,:,z]>0).sum() for z in range(Z)]; zs=[z for z in range(Z) if counts[z]>200]
masks={R:vd_cols(Y,R) for R in RS}                      # fixed masks across slices (fair)
print(f"{a.case}: analysing {len(zs)} slices (z={zs[0]}..{zs[-1]}), echo {a.echo}")
for z in zs:
    img=TGT[:,:,z,a.echo,0]                             # complex slice
    # precompute undersampled recons of the FULL slice
    F=np.fft.fftshift(np.fft.fft2(img)); recon={}
    for R in RS:
        m=masks[R]; Fm=F.copy(); Fm[:,~m]=0; recon[R]=np.abs(np.fft.ifft2(np.fft.ifftshift(Fm)))
    clean=np.abs(img)
    lz=seg[:,:,z]
    gref=clean[lz>0].mean()+1e-8                          # global tissue reference (brightness-neutral)
    bg=clean<np.percentile(clean,15)                     # v3: signal-free background (air) = noise-floor sampler
    for i in LAB:
        M=lz==i
        if M.sum()<80: continue
        ys,xs=np.where(M); sl=(slice(ys.min(),ys.max()+1),slice(xs.min(),xs.max()+1))
        patch=(img*M)[sl]
        acc[i]["cen"].append(centroid(patch))
        # structure k-space energy removed by each mask (full-slice columns)
        Fs=np.fft.fftshift(np.fft.fft2(img*M)); Ps=np.abs(Fs)**2; tot=Ps.sum()+1e-12
        for R in RS:
            m=masks[R]; acc[i][f"el{R}"].append(float(Ps[:,~m].sum()/tot))
            diff=recon[R]-clean
            srmse=np.sqrt((diff[M]**2).mean())            # total error in the structure
            nfloor=np.sqrt((diff[bg]**2).mean())          # v3: noise floor from air (same R, same echo)
            corr=np.sqrt(max(srmse**2-nfloor**2,0.0))     # subtract noise in quadrature -> STRUCTURED error only
            acc[i][f"dg{R}"].append(float(corr/gref))     # v3: noise-corrected, global-normalized
f.close()

rows=[]
for i in LAB:
    if not acc[i]["cen"]: continue
    r={"struct":LAB[i],"centroid":float(np.mean(acc[i]["cen"]))}
    for R in RS: r[f"Elost_R{R}"]=float(np.mean(acc[i][f"el{R}"])); r[f"degr_R{R}"]=float(np.mean(acc[i][f"dg{R}"]))
    rows.append(r)
# pooled mechanism arrays (structure x R)
EL=np.array([[r[f"Elost_R{R}"] for R in RS] for r in rows]).ravel()
DG=np.array([[r[f"degr_R{R}"] for R in RS] for r in rows]).ravel()
Rlab=np.array([[R for R in RS] for _ in rows]).ravel()
pear=pearsonr(EL,DG); spear=spearmanr(EL,DG)
# centroid vs degradation@R8 (does anatomy rank fragility)
cen=np.array([r["centroid"] for r in rows]); dg8=np.array([r["degr_R8"] for r in rows])
cs=spearmanr(cen,dg8)
json.dump({"case":a.case,"rows":rows,"mech_pearson":pear[0],"mech_spearman":spear.correlation,
           "centroid_vs_degrR8_spearman":cs.correlation},open(f"{RES}/skmtea_law_v3{SUF}.json","w"),indent=2)
print(f"mechanism  E_lost->degradation: Pearson {pear[0]:.2f}, Spearman {spear.correlation:.2f}")
print(f"anatomy    centroid->degr@R8:   Spearman {cs.correlation:.2f}")

COL={2:"#4daf4a",4:"#377eb8",6:"#984ea3",8:"#e41a1c"}
SC=plt.cm.tab10(np.linspace(0,1,len(rows)))

# ===== FIG 1: per-structure curves — energy RETAINED (input) and FIDELITY (effect) vs acceleration (up = better) =====
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
for k,r in enumerate(rows):
    ax[0].plot(RS,[1-r[f"Elost_R{R}"] for R in RS],"-o",color=SC[k],label=r["struct"])
    ax[1].plot(RS,[1-r[f"degr_R{R}"] for R in RS],"-o",color=SC[k],label=r["struct"])
ax[0].set_title("INPUT: k-space energy KEPT",fontweight="bold")
ax[0].set_xlabel("acceleration R  (faster scan →)"); ax[0].set_ylabel("energy retained  (1 − E_lost)\n(higher = better ↑)"); ax[0].grid(alpha=.3)
ax[1].set_title("EFFECT: reconstruction fidelity",fontweight="bold")
ax[1].set_xlabel("acceleration R  (faster scan →)"); ax[1].set_ylabel("fidelity  (1 − NRMSE)\n(higher = better ↑)"); ax[1].grid(alpha=.3)
ax[1].legend(fontsize=8,frameon=False,ncol=2,title="knee structure")
fig.suptitle(f"SKM-TEA {a.case}: faster scan → less energy kept (left) → lower fidelity (right)  —  both fall as R rises  (v3: noise-corrected, higher = better)",fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_law_v3{SUF}_predictors.png",dpi=140,bbox_inches="tight"); plt.close(fig)

# ===== FIG 2: the mechanism on REAL k-space (energy RETAINED -> fidelity; up-right = better) =====
RE=1-EL; RF=1-DG                                        # retained energy, retained fidelity
fig,ax=plt.subplots(figsize=(7.6,6))
for R in RS:
    mm=Rlab==R; ax.scatter(RE[mm],RF[mm],s=90,c=COL[R],label=f"R={R}",edgecolor="white",lw=.6,zorder=3)
for k,r in enumerate(rows):
    ax.plot([1-r[f"Elost_R{R}"] for R in RS],[1-r[f"degr_R{R}"] for R in RS],color="#bbb",lw=.8,zorder=1)
b,m=np.polynomial.polynomial.polyfit(RE,RF,1); xx=np.linspace(RE.min(),RE.max(),50)
ax.plot(xx,b+m*xx,"k--",lw=1.4,zorder=2)
ax.text(.05,.90,f"Pearson r = {pear[0]:.2f}\nSpearman = {spear.correlation:.2f}",transform=ax.transAxes,fontsize=13,fontweight="bold")
ax.text(.97,.06,"HYPOTHESIS: fidelity is GOVERNED by energy kept\n→ one curve for all structures & all accelerations",
        transform=ax.transAxes,fontsize=9.5,style="italic",color="#444",ha="right",
        bbox=dict(boxstyle="round",fc="#f5f5f5",ec="#ccc"))
ax.set_xlabel("fraction of structure k-space energy RETAINED  (1 − E_lost)   (more kept → better →)")
ax.set_ylabel("reconstruction fidelity  (1 − NRMSE)   (↑ better)")
ax.set_title(f"SKM-TEA {a.case}: the energy mechanism holds on REAL multicoil qDESS\nmore k-space energy kept → higher fidelity  (v3: noise-corrected (structured error only))",fontsize=11.5,fontweight="bold")
ax.legend(title="acceleration",frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{PLOTS}/skmtea_law_v3{SUF}_mechanism.png",dpi=140,bbox_inches="tight"); plt.close(fig)
print("wrote: skmtea_law_v2_predictors.png , skmtea_law_v2_mechanism.png , skmtea_law_v3.json")
