#!/bin/bash
# BD3-LM vs AR Baseline Comparison
BLOCK_SIZE=${1:-4}
NUM_SAMPLES=${2:-50}

echo "============================================"
echo "BD3-LM vs AR Baseline Comparison"
echo "Block Size: ${BLOCK_SIZE}"
echo "Samples: ${NUM_SAMPLES}"
echo "============================================"

echo -e "\n[1/1] Running AR baseline comparison..."
python -u main.py \
  loader.eval_batch_size=8 \
  model=small \
  algo=bd3lm \
  sampling.nucleus_p=0.9\
  algo.T=5000 \
  algo.backbone=hf_dit \
  data=openwebtext-split \
  model.length=1024 \
  block_size=${BLOCK_SIZE} \
  eval.num_samples=${NUM_SAMPLES} \
  eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
  eval.ar_checkpoint_path=kuleshov-group/ar-noeos-owt \
  wandb=null \
  mode=ar_comparison_eval \
  model.attn_backend=sdpa

echo -e "\n============================================"
echo "AR comparison complete!"
echo "Results saved in: ./results/ar_comparison_bs${BLOCK_SIZE}.json"
echo "============================================"