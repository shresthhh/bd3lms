#!/bin/bash

# BD3-LM Comprehensive Evaluation with Correct Settings
# Critical fixes: nucleus_p=0.9, first_hitting=True

BLOCK_SIZE=${1:-4}
NUM_SAMPLES=${2:-50}

echo "============================================"
echo "BD3-LM Comprehensive Evaluation"
echo "Block Size: ${BLOCK_SIZE}"
echo "Samples: ${NUM_SAMPLES}"
echo "Key Settings: nucleus_p=0.9, T=5000"
echo "============================================"

# 1. Comprehensive metrics evaluation
echo -e "\n[1/3] Running comprehensive block metrics evaluation..."
python -u main.py \
    loader.eval_batch_size=8 \
    model=small \
    algo=bd3lm \
    algo.T=5000 \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    sampling.nucleus_p=0.9 \
    sampling.first_hitting=True \
    eval.num_samples=${NUM_SAMPLES} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb=null \
    mode=block_metrics_eval \
    model.attn_backend=sdpa

# 2. Sampling efficiency evaluation
echo -e "\n[2/3] Running sampling efficiency evaluation (testing different T values)..."
python -u main.py \
    loader.eval_batch_size=4 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    sampling.nucleus_p=0.9 \
    sampling.first_hitting=True \
    eval.num_samples=20 \
    eval.T_values=[500,1000,2500,5000] \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb=null \
    mode=sampling_efficiency_eval \
    model.attn_backend=sdpa

# 3. Length robustness evaluation
echo -e "\n[3/3] Running length robustness evaluation..."
python -u main.py \
    loader.eval_batch_size=4 \
    model=small \
    algo=bd3lm \
    algo.T=5000 \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    block_size=${BLOCK_SIZE} \
    sampling.nucleus_p=0.9 \
    sampling.first_hitting=True \
    eval.num_samples=20 \
    eval.test_lengths=[512,1024,2048] \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb=null \
    mode=length_robustness_eval \
    model.attn_backend=sdpa

echo -e "\n============================================"
echo "Evaluation complete!"
echo "Results saved in: ./results/"
echo "============================================"

