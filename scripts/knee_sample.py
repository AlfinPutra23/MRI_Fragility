"""Qualitative knee-segmentation SAMPLES: retrain the 2D UNet (GPU-preloaded = fast), save it, then predict on unseen
TEST cases at clean / R4 / R8 and dump {image, GT, pred} to an npz for the figure. Run with magicnet (PYTHONNOUSERSITE=1);
plot with base-anaconda via knee_sample_plot.py (magicnet has no matplotlib). Same 32-train/12-test split as knee_seg."""
import os, sys, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "scripts")
from knee_seg import UNet2D, vd_mask, undersample, norm, load_case_slices, LAB, RS

NCLS = 7; dev = "cuda"; SZ = 256                       # downsample 512->256: 4x less GPU mem (~3GB, safe) + 4x faster
MASKS = {R: vd_mask(SZ, R) for R in RS}
def ds_img(a): return 0.25 * (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2])   # 2x2 mean (complex ok)
def ds_lab(a): return a[::2, ::2]                                                          # nearest for labels
h5s = sorted(glob.glob("data/skmtea/kspace/**/*.h5", recursive=True))
cases = [(h, f"data/skmtea/seg/{os.path.basename(h)[:-3]}_raw-data-track.nii.gz") for h in h5s]
cases = [(h, s) for h, s in cases if os.path.exists(s)]
test_i = set(np.random.RandomState(0).permutation(len(cases))[:12].tolist())

# ---- train (GPU-preloaded float16 img / int8 lab -> fast, ~2.4GB) ----
Xtr, Ytr = [], []
for j, (h, s) in enumerate(cases):
    if j in test_i:
        continue
    cimg, lab = load_case_slices(h, s)
    for k in range(len(cimg)):
        Xtr.append(norm(np.abs(ds_img(cimg[k]))).astype(np.float16)); Ytr.append(ds_lab(lab[k]).astype(np.int8))
Xtr = torch.from_numpy(np.stack(Xtr))[:, None].to(dev)
Ytr = torch.from_numpy(np.stack(Ytr)).to(dev)
print(f"train slices {len(Xtr)}", flush=True)
net = UNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
cw = torch.ones(NCLS, device=dev); cw[1:] = 3.0; bs = 8
for ep in range(45):
    net.train(); perm = torch.randperm(len(Xtr), device=dev); tot = 0.0
    for i in range(0, len(Xtr), bs):
        b = perm[i:i + bs]; opt.zero_grad()
        xb = Xtr[b].float(); yb = Ytr[b].long()
        lo = net(xb); ce = F.cross_entropy(lo, yb, weight=cw)
        pr = torch.softmax(lo, 1); dl = 0.0
        for k in range(1, NCLS):
            pk = pr[:, k]; gk = (yb == k).float(); dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
        loss = ce + dl / (NCLS - 1); loss.backward(); opt.step(); tot += loss.item()
    if ep % 10 == 0 or ep == 44:
        print(f"  ep {ep} loss {tot/(len(Xtr)//bs+1):.3f}", flush=True)
os.makedirs("models", exist_ok=True); torch.save(net.state_dict(), "models/knee_unet.pth")

# ---- predict sample TEST cases at clean/R4/R8, pick the slice richest in structures ----
net.eval(); samples = []
for j in sorted(test_i)[:3]:
    h, s = cases[j]; cid = os.path.basename(h)[:-3]
    cimg, lab = load_case_slices(h, s)
    best = max(range(len(lab)), key=lambda z: len(np.unique(lab[z])))   # slice with most structures
    c = ds_img(cimg[best]); gt = ds_lab(lab[best]).astype(np.int8)
    row = {"cid": cid, "gt": gt}
    with torch.no_grad():
        for cond, mg in [("clean", np.abs(c)), ("R4", undersample(c, MASKS[4])), ("R8", undersample(c, MASKS[8]))]:
            mgn = norm(mg).astype(np.float32)
            pred = net(torch.from_numpy(mgn)[None, None].to(dev)).argmax(1)[0].cpu().numpy().astype(np.int8)
            row[f"img_{cond}"] = mgn.astype(np.float16); row[f"pred_{cond}"] = pred
    samples.append(row)
os.makedirs("outputs/results", exist_ok=True)
np.savez_compressed("outputs/results/knee_samples.npz", samples=np.array(samples, dtype=object))
print("saved knee_samples.npz:", [r["cid"] for r in samples])
