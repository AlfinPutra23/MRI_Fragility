"""Seed-controlled loss-weighting sweep trainers (M2 audit follow-up).

All variants share SEED=42 + cudnn-deterministic so differences isolate the LOSS, not the seed.
  Uniform_s42  : plain DiceCE                       (seed-control baseline)
  FragW4_s42   : fragility-weighted CE, max 4x      (= current weighting, seeded)
  FragW2_s42   : fragility-weighted CE, max 2x      (gentler -> λ-sweep)
  FragTopK_s42 : fragility-weighted top-10% CE       (hard-example: focus hardest voxels)

Weights w_o = 1 + SCALE * normalized_fragility (from the M0 R1->R8 drop curve); SCALE 3 -> max 4x.
Install into nnunetv2 .../nnUNetTrainer/variants/ and run with -tr <ClassName>.
"""
import random
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# normalized M0 fragility (label id -> (drop-min)/(max-min)); liver=0, adrenal_L=1
NORM = {1: 0.156, 2: 0.112, 3: 0.117, 4: 0.844, 5: 0.0, 6: 0.609, 7: 0.263,
        11: 0.302, 12: 0.933, 13: 1.0, 16: 0.525, 17: 0.682, 18: 0.804}


def frag_weights(ncls, device, scale):
    w = torch.ones(ncls, device=device)
    for lab, nd in NORM.items():
        if lab < ncls:
            w[lab] = 1.0 + scale * nd
    return w


class _Seeded:
    SEED = 42

    def initialize(self):
        random.seed(self.SEED); np.random.seed(self.SEED)
        torch.manual_seed(self.SEED); torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        super().initialize()


class nnUNetTrainer_Uniform_s42(_Seeded, nnUNetTrainer_250epochs):
    pass


class _FragBase(_Seeded, nnUNetTrainer_250epochs):
    SCALE = 3.0     # -> max weight 4x

    def _build_loss(self):
        loss = super()._build_loss()
        base = loss.loss if isinstance(loss, DeepSupervisionWrapper) else loss
        base.ce.weight = frag_weights(self.label_manager.num_segmentation_heads, self.device, self.SCALE)
        self.print_to_log_file(f"[sweep] frag CE weights, scale={self.SCALE}, "
                               f"max={1 + self.SCALE * max(NORM.values()):.1f}x")
        return loss


class nnUNetTrainer_FragW4_s42(_FragBase):
    SCALE = 3.0


class nnUNetTrainer_FragW2_s42(_FragBase):
    SCALE = 1.0


class nnUNetTrainer_FragW6_s42(_FragBase):
    SCALE = 5.0     # -> max weight 6x


class nnUNetTrainer_FragW8_s42(_FragBase):
    SCALE = 7.0     # -> max weight 8x


class nnUNetTrainer_FragTopK_s42(_FragBase):
    SCALE = 3.0

    def _build_loss(self):
        # use nnU-Net's dedicated DC_and_topk_loss (TopKLoss can't be slotted into DC_and_CE's .ce:
        # it squeezes target[:,0] unconditionally -> double-squeeze). Pass fragility weight + k via ce_kwargs.
        from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
        w = frag_weights(self.label_manager.num_segmentation_heads, self.device, self.SCALE)
        loss = DC_and_topk_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': w, 'k': 10}, weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            dss = self._get_deep_supervision_scales()
            ws = np.array([1 / (2 ** i) for i in range(len(dss))]); ws[-1] = 0; ws = ws / ws.sum()
            loss = DeepSupervisionWrapper(loss, ws)
        self.print_to_log_file(f"[sweep] DC_and_topk(10%) + fragility CE weights, "
                               f"max={1 + self.SCALE * max(NORM.values()):.1f}x")
        return loss
