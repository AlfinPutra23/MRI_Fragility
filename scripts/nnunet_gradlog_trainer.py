"""M1(b)-proper: nnU-Net trainer that logs per-organ seg-loss GRADIENT MASS across training.

Drop-in trainer subclass. Every PROBE_EVERY epochs it runs a fixed cached batch through the
*current* network and measures per-organ |dL/dlogits| mass (CE-term and Dice-term, as in
m1_gradient_probe.py), then appends {epoch, liver/adrenal ratios, per-organ masses} to
<output_folder>/gradient_trajectory.json. This turns the convergence-only probe (a lower bound,
liver/adrenal ~14x) into the full during-training curve (expected ~200x early -> ~14x converged).

Self-contained (no cross-imports) so it is safe to copy into the nnU-Net package trainer folder.

Install (so nnUNetv2_train -tr finds it):
  cp scripts/nnunet_gradlog_trainer.py \
     $CONDA/envs/mrifrag/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/
Run:
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_GradLog
"""
import os, json
import torch
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs

# 13 abdominal organs (label id -> name); tail = small/hard organs (authoritative map)
ABDO = {1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder", 5: "liver", 6: "esophagus",
        7: "stomach", 11: "pancreas", 12: "adrenal_R", 13: "adrenal_L", 16: "small_bowel",
        17: "duodenum", 18: "colon"}
TAIL = {4, 6, 11, 12, 13, 17}


def _per_organ_grad_mass(logits, gt, eps=1e-5):
    """logits: (1,C,...) leaf requiring grad; gt: (...) int. -> {organ: (n,ce_mass,dc_mass)}."""
    out = {}
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    for o in ABDO:
        g = (gt == o)
        n = int(g.sum())
        if n == 0:
            continue
        if logits.grad is not None:
            logits.grad = None
        (-log_p[0, o][g].sum()).backward(retain_graph=True)
        ce = float(logits.grad.abs().sum())
        logits.grad = None
        p_o, g_f = p[0, o], g.float()
        inter = (p_o * g_f).sum()
        dc = 1.0 - (2 * inter + eps) / (p_o.sum() + g_f.sum() + eps)
        dc.backward(retain_graph=True)
        out[o] = (n, ce, float(logits.grad.abs().sum()))
    return out


class nnUNetTrainer_GradLog(nnUNetTrainer_250epochs):
    PROBE_EVERY = 5

    def on_train_start(self):
        super().on_train_start()
        self._probe_batch = next(self.dataloader_val)          # fixed batch for consistent probing
        self._traj_path = os.path.join(self.output_folder, "gradient_trajectory.json")
        self._traj = []
        self.print_to_log_file(f"[gradlog] will log per-organ gradient every {self.PROBE_EVERY} epochs "
                               f"-> {self._traj_path}")

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if (self.current_epoch % self.PROBE_EVERY == 0) or (self.current_epoch == self.num_epochs - 1):
            try:
                self._probe()
            except Exception as e:
                self.print_to_log_file(f"[gradlog] probe failed @ ep{self.current_epoch}: {e}")

    def _probe(self):
        data = self._probe_batch['data'].to(self.device, non_blocking=True)
        tgt = self._probe_batch['target']
        tgt = (tgt[0] if isinstance(tgt, list) else tgt).to(self.device)   # (B,1,Z,Y,X) full-res labels
        self.network.eval()
        agg = {o: [0, 0.0, 0.0] for o in ABDO}                              # n, ce, dc summed over batch
        with torch.no_grad():
            raw = self.network(data)
        raw = (raw[0] if isinstance(raw, (list, tuple)) else raw).float()
        for b in range(raw.shape[0]):
            logits = raw[b:b+1].clone().detach().requires_grad_(True)
            for o, (n, ce, dc) in _per_organ_grad_mass(logits, tgt[b, 0]).items():
                agg[o][0] += n; agg[o][1] += ce; agg[o][2] += dc
        self.network.train()

        rows = {ABDO[o]: dict(n_vox=agg[o][0], ce_mass=agg[o][1], dc_mass=agg[o][2],
                              total_mass=agg[o][1] + agg[o][2], tail=o in TAIL)
                for o in ABDO if agg[o][0] > 0}
        liv = rows.get("liver", {})
        adr = [rows[k] for k in ("adrenal_R", "adrenal_L") if k in rows]

        def ratio(field):
            if not liv or not adr:
                return None
            sm = sum(a[field] for a in adr) / len(adr)
            return liv[field] / sm if sm else None
        entry = dict(epoch=int(self.current_epoch),
                     vol_ratio=ratio("n_vox"), ce_ratio=ratio("ce_mass"),
                     dc_ratio=ratio("dc_mass"), total_ratio=ratio("total_mass"), organs=rows)
        self._traj.append(entry)
        json.dump(self._traj, open(self._traj_path, "w"), indent=2, default=float)
        tr = entry["total_ratio"]
        self.print_to_log_file(f"[gradlog] ep{self.current_epoch}: liver/adrenal total grad ratio = "
                               f"{tr:.1f}x  (CE {entry['ce_ratio']:.1f}x / Dice {entry['dc_ratio']:.2f}x)"
                               if tr else f"[gradlog] ep{self.current_epoch}: adrenal absent in probe batch")
