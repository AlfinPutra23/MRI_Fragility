"""M2-entry de-risk: nnU-Net trainer with a FRAGILITY-WEIGHTED CE term.

Tests the method's loss idea in isolation (seg-only, no LOUPE/VarNet): does up-weighting the per-organ CE
by M0 fragility recover tail-organ Dice under acceleration vs the uniform-loss baseline (= the M0 model)?

Mechanism-aligned (M1): the imbalance lives in the CE gradient (Dice already over-corrects), so we reweight
*CE* per organ. Weights from the M0 drop curve:  w = 1 + 3*(drop - min_drop)/(max_drop - min_drop), for the
13 abdominal organs; 1.0 for background and all non-abdominal structures (never down-weighted below 1).

Install + run:
  cp scripts/nnunet_fragweighted_trainer.py $CONDA/envs/mrifrag/.../nnunetv2/training/nnUNetTrainer/variants/
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_FragWeighted
"""
import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# fragility CE weights (label id -> weight), derived from M0 R1->R8 drops (see docstring)
FRAG_W = {1: 1.47, 2: 1.34, 3: 1.35, 4: 3.53, 5: 1.00, 6: 2.83, 7: 1.79,
          11: 1.91, 12: 3.80, 13: 4.00, 16: 2.58, 17: 3.05, 18: 3.41}


class nnUNetTrainer_FragWeighted(nnUNetTrainer_250epochs):
    def _build_loss(self):
        loss = super()._build_loss()                       # standard DiceCE (+ deep supervision)
        ncls = self.label_manager.num_segmentation_heads
        w = torch.ones(ncls, device=self.device)
        for k, v in FRAG_W.items():
            if k < ncls:
                w[k] = v
        base = loss.loss if isinstance(loss, DeepSupervisionWrapper) else loss
        base.ce.weight = w                                 # per-class CE weighting (buffer reassign)
        self.print_to_log_file(
            f"[fragweighted] per-class CE weights applied (ncls={ncls}): "
            f"adrenals 3.8-4.0x, gallbladder 3.5x, colon 3.4x, duodenum 3.0x ... liver 1.0x, non-abdo 1.0x")
        return loss
