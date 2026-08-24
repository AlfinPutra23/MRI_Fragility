"""SECOND-ARCHITECTURE fragility check (is per-organ fragility a DATA property or an nnU-Net artifact?). Trains a
hand-coded ResU-Net (residual blocks + BatchNorm + 5 levels -> a different architecture family than nnU-Net's plain-conv
3D self-configured U-Net) on the abdominal data, then measures per-organ Dice at clean + R{2,4,6,8} on the SAME test set.
If the fragility ORDERING correlates with nnU-Net's -> fragility is architecture-independent. Run with magicnet
(PYTHONNOUSERSITE=1: torch+numpy+nibabel consistent). -> outputs/results/arch2_fragility.json"""
import os, glob, json, argparse, numpy as np, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0, "scripts"); import labels as L

D = "nnUNet_raw/Dataset501_MRIfrag"; SZ = 256; dev = "cuda"
ORG = list(L.ABDO); NCLS = max(ORG) + 1; ORGSET = np.array(ORG)
RS = [1, 2, 4, 6, 8]; TAG = {1: "clean", 2: "R2", 4: "R4", 6: "R6", 8: "R8"}


class ResBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o))
        s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity(); s.act = nn.ReLU(inplace=True)
    def forward(s, x): return s.act(s.conv(x) + s.skip(x))


class ResUNet2D(nn.Module):
    def __init__(s, ncls, ch=(32, 64, 128, 256, 320)):
        super().__init__()
        s.enc = nn.ModuleList([ResBlock(1 if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        s.pool = nn.MaxPool2d(2)
        s.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, 2) for i in range(len(ch) - 1, 0, -1)])
        s.dec = nn.ModuleList([ResBlock(ch[i - 1] * 2, ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        s.out = nn.Conv2d(ch[0], ncls, 1)
    def forward(s, x):
        feats = []
        for i, e in enumerate(s.enc):
            x = e(x); feats.append(x)
            if i < len(s.enc) - 1: x = s.pool(x)
        for j, (u, d) in enumerate(zip(s.up, s.dec)):
            x = u(x); x = d(torch.cat([x, feats[len(feats) - 2 - j]], 1))
        return s.out(x)


def keep_abdo(lb): lb[~np.isin(lb, ORGSET)] = 0; return lb
def norm(x): p = np.percentile(x, 99.5); return (x / p).clip(0, 2.0) if p > 0 else x


def resize_stack(a, order):    # (n,H,W) -> (n,SZ,SZ), batched on GPU (fast)
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[:, None].to(dev)
    r = F.interpolate(t, size=(SZ, SZ), mode="nearest" if order == 0 else "bilinear",
                      align_corners=None if order == 0 else False)
    return r.cpu().numpy()[:, 0]


def load_case(img_path, lab_path):
    im = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
    lb = keep_abdo(np.asanyarray(nib.load(lab_path).dataobj).astype(np.int16))
    zs = [z for z in range(im.shape[2]) if (lb[:, :, z] > 0).sum() >= 50]
    if not zs: return None, None
    X = resize_stack(np.stack([im[:, :, z] for z in zs]), 1)
    Y = resize_stack(np.stack([lb[:, :, z] for z in zs]), 0).astype(np.int64)
    X = np.stack([norm(x) for x in X]).astype(np.float32)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_slices", type=int, default=9000)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    tr = sorted(glob.glob(f"{D}/imagesTr/*_0000.nii.gz"))[:args.n_train]
    Xtr, Ytr = [], []
    for p in tr:
        lb = f"{D}/labelsTr/" + os.path.basename(p).replace("_0000", "")
        if not os.path.exists(lb): continue
        X, Y = load_case(p, lb)
        if X is not None: Xtr.append(X); Ytr.append(Y)
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    if len(Xtr) > args.max_slices:
        idx = np.random.choice(len(Xtr), args.max_slices, replace=False); Xtr = Xtr[idx]; Ytr = Ytr[idx]
    Xtr = torch.from_numpy(Xtr)[:, None]; Ytr = torch.from_numpy(Ytr)      # CPU (batch->GPU, low mem)
    print(f"2nd-arch ResU-Net: {len(Xtr)} train slices, {NCLS} classes (dev={dev})", flush=True)

    net = ResUNet2D(NCLS).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); bs = 16
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            xb = Xtr[b].to(dev); yb = Ytr[b].to(dev)
            lo = net(xb); ce = F.cross_entropy(lo, yb)
            pr = torch.softmax(lo, 1); present = [k for k in ORG if (yb == k).any()]; dl = 0.0
            for k in present:
                pk = pr[:, k]; gk = (yb == k).float()
                dl = dl + (1 - (2 * (pk * gk).sum() + 1) / (pk.sum() + gk.sum() + 1))
            loss = ce + dl / max(len(present), 1); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:2d} loss {tot/(len(Xtr)//bs+1):.3f}", flush=True)
    os.makedirs("models", exist_ok=True); torch.save(net.state_dict(), "models/arch2_resunet.pth")

    # ----- eval per-organ Dice at each R (batched per case) -----
    net.eval(); gts = sorted(glob.glob(f"{D}/labelsTs/*.nii.gz"))
    inter = {R: {k: 0 for k in ORG} for R in RS}; union = {R: {k: 0 for k in ORG} for R in RS}
    with torch.no_grad():
        for gp in gts:
            case = os.path.basename(gp)[:-7]; lb = keep_abdo(np.asanyarray(nib.load(gp).dataobj).astype(np.int16))
            zs = [z for z in range(lb.shape[2]) if (lb[:, :, z] > 0).sum() >= 50]
            if not zs: continue
            gz = resize_stack(np.stack([lb[:, :, z] for z in zs]), 0).astype(np.int64)
            for R in RS:
                ip = f"{D}/imagesTs_{TAG[R]}/{case}_0000.nii.gz"
                if not os.path.exists(ip): continue
                im = np.asanyarray(nib.load(ip).dataobj).astype(np.float32)
                X = resize_stack(np.stack([im[:, :, z] for z in zs]), 1)
                X = np.stack([norm(x) for x in X]).astype(np.float32)
                pr = np.concatenate([net(torch.from_numpy(X[i:i + 16])[:, None].to(dev)).argmax(1).cpu().numpy()
                                     for i in range(0, len(X), 16)])
                for k in ORG:
                    inter[R][k] += int(np.logical_and(pr == k, gz == k).sum())
                    union[R][k] += int((pr == k).sum() + (gz == k).sum())

    def Dk(R, k): return (2 * inter[R][k] / union[R][k]) if union[R][k] else np.nan
    per = {L.ABDO[k]: {TAG[R]: Dk(R, k) for R in RS} for k in ORG}
    drop = {L.ABDO[k]: float(Dk(1, k) - Dk(8, k)) for k in ORG}
    json.dump({"n_train_slices": int(len(Xtr)), "per_organ_dice": per, "dice_drop_clean_to_R8": drop},
              open("outputs/results/arch2_fragility.json", "w"), indent=2)
    print("\n=== ResU-Net per-organ Dice (clean -> R8) ===")
    for k in ORG: print(f"  {L.ABDO[k]:14} {Dk(1,k):.3f} -> {Dk(8,k):.3f}  (drop {drop[L.ABDO[k]]:+.3f})")
    print("wrote arch2_fragility.json")


if __name__ == "__main__":
    main()
