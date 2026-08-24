#!/usr/bin/env bash
# GPU-gated SIMULATION-FIDELITY LADDER (weakness #1). After fidelity_ladder.py --stage gen writes the 4 rungs
# (imagesTs_R8_L{mag,phase,coils,noise}, 960 images), segment each with the EXISTING Uniform_s42 nnU-Net (inference
# only) and run --stage eval -> is the per-organ fragility ranking + centroid law invariant across acquisition realism?
# NEVER fights the user's GPUs: predicts only on a genuinely idle card (<3000 MiB AND no train/predict proc). Detached.
# GATED: do NOT launch until the user says GO and a GPU is free.
set -uo pipefail
ENVBIN=/home/user/anaconda3/envs/mrifrag/bin
P=/media/user/B4864CD4864C98AE/mri_fragility; cd "$P"
export nnUNet_raw=$P/nnUNet_raw nnUNet_preprocessed=$P/nnUNet_preprocessed nnUNet_results=$P/nnUNet_results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$nnUNet_raw/Dataset501_MRIfrag; TR=nnUNetTrainer_Uniform_s42
log=$P/outputs/logs/fidelity_ladder.log; say(){ echo "[$(date '+%F %T')] $*" | tee -a "$log"; }

wait_free_gpu(){ for t in $(seq 1 360); do for c in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c 2>/dev/null | tr -d ' ')
    busy=$(ps -eo args 2>/dev/null | grep -cE "[n]nUNetv2_train|[n]nUNetv2_predict")
    [ "${m:-99999}" -lt 3000 ] && [ "$busy" -eq 0 ] && { echo $c; return; }
  done; sleep 120; done; echo -1; }

say "=== fidelity-ladder armed ==="
# 1) generate the 4 rungs if not present (CPU, safe)
n=$(ls $D/imagesTs_R8_L{mag,phase,coils,noise}/*.nii.gz 2>/dev/null | wc -l)
if [ "$n" -lt 960 ]; then say "generating rungs ($n/960 present)"; "$ENVBIN/python" scripts/fidelity_ladder.py --stage gen >> "$log" 2>&1; fi

# 2) GPU-gated inference per rung (clean pred is forward-model-independent -> reuse predsTs_clean)
for rg in mag phase coils noise; do
  o=$D/predsLAD_$rg
  [ "$(ls $o/*.nii.gz 2>/dev/null | wc -l)" -ge 240 ] && { say "$rg already done"; continue; }
  g=$(wait_free_gpu); [ "$g" = "-1" ] && { say "no free GPU after wait; abort"; exit 1; }
  say "predict rung=$rg on GPU$g"
  CUDA_VISIBLE_DEVICES=$g "$ENVBIN/nnUNetv2_predict" -i "$D/imagesTs_R8_L${rg}" -o "$o" -d 501 -c 3d_fullres -f 0 -tr "$TR" --disable_tta >> "$log" 2>&1
done

# 3) CPU eval: ranking invariance + per-rung law
say "eval"
"$ENVBIN/python" scripts/fidelity_ladder.py --stage eval >> "$log" 2>&1
say "=== FIDELITY LADDER DONE (see outputs/results/fidelity_ladder.json) ==="
