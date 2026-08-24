"""nnU-Net Focal-Tversky trainer (the proxy-winning recall loss, taken to the REAL 3D pipeline).

Focal-Tversky index TI = TP/(TP + a*FP + b*FN) with b>a penalises FALSE NEGATIVES (recall); the focal power
(1-TI)^gamma concentrates on hard/vanishing classes -> fights the tiny-organ "predict nothing -> 0 Dice" collapse.
Combined with Dice (DC_and_FocalTversky) + fragility CE-style class weights (from the M0 R1->R8 drop curve).
Seed-matched (s42, cudnn-deterministic) to nnUNetTrainer_Uniform_s42 so the difference isolates the LOSS.

Install:  cp scripts/nnunet_focaltversky_trainer.py $ENV/.../nnUNetTrainer/variants/
Run:      nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_FocalTversky_s42
"""
import random, numpy as np, torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1

NORM = {1: 0.156, 2: 0.112, 3: 0.117, 4: 0.844, 5: 0.0, 6: 0.609, 7: 0.263,
        11: 0.302, 12: 0.933, 13: 1.0, 16: 0.525, 17: 0.682, 18: 0.804}


def frag_weights(ncls, device, scale=3.0):
    w = torch.ones(ncls, device=device)
    for lab, nd in NORM.items():
        if lab < ncls:
            w[lab] = 1.0 + scale * nd
    return w


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, weight=None, do_bg=False, smooth=1e-5):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.do_bg, self.smooth = alpha, beta, gamma, do_bg, smooth
        self.weight = weight

    def forward(self, net_output, target):
        # net_output (B,C,spatial); target (B,1,spatial) or (B,spatial)
        p = torch.softmax(net_output, 1)
        if target.dim() == net_output.dim():
            target = target[:, 0]
        C = net_output.shape[1]
        y = torch.zeros_like(p); y.scatter_(1, target.long().clamp(0, C - 1).unsqueeze(1), 1)
        axes = list(range(2, net_output.dim()))
        tp = (p * y).sum(axes); fp = (p * (1 - y)).sum(axes); fn = ((1 - p) * y).sum(axes)   # (B,C)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        # clamp base to a POSITIVE floor: grad of x**gamma is +inf at x=0 for gamma<1 -> NaN (learned in the 2-D proxy)
        ft = (1 - ti).clamp(min=1e-6) ** self.gamma                                          # (B,C)
        c0 = 0 if self.do_bg else 1
        ft = ft[:, c0:]
        if self.weight is not None:
            ft = ft * self.weight[c0:].to(ft.device).view(1, -1)
        return ft.mean()


class DC_and_FocalTversky(nn.Module):
    def __init__(self, soft_dice_kwargs, ft_kwargs, weight_dice=1.0, weight_ft=1.0, ignore_label=None):
        super().__init__()
        self.weight_dice, self.weight_ft, self.ignore_label = weight_dice, weight_ft, ignore_label
        self.dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ft = FocalTverskyLoss(**ft_kwargs)

    def forward(self, net_output, target):
        mask = None; target_dice = target
        if self.ignore_label is not None:
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        dc = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ft = self.ft(net_output, target_dice) if self.weight_ft != 0 else 0
        return self.weight_dice * dc + self.weight_ft * ft


class _Seeded:
    SEED = 42
    def initialize(self):
        random.seed(self.SEED); np.random.seed(self.SEED)
        torch.manual_seed(self.SEED); torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        super().initialize()


class nnUNetTrainer_FocalTversky_s42(_Seeded, nnUNetTrainer_250epochs):
    def _build_loss(self):
        w = frag_weights(self.label_manager.num_segmentation_heads, self.device, 3.0)   # fragility-weighted (max 4x)
        loss = DC_and_FocalTversky(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'alpha': 0.3, 'beta': 0.7, 'gamma': 0.75, 'weight': w, 'do_bg': False},
            weight_dice=1.0, weight_ft=1.0, ignore_label=self.label_manager.ignore_label)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            dss = self._get_deep_supervision_scales()
            ws = np.array([1 / (2 ** i) for i in range(len(dss))]); ws[-1] = 0; ws = ws / ws.sum()
            loss = DeepSupervisionWrapper(loss, ws)
        self.print_to_log_file("[FocalTversky] DC + Focal-Tversky (alpha0.3 beta0.7 gamma0.75, FN-weighted) "
                               "+ fragility class weights (max 4x); seed 42")
        return loss
