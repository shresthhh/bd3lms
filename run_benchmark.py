#!/usr/bin/env python3
"""
Benchmark runner for BD3-LM inference experiments.
Runs baseline and modified inference, saves metrics, and generates comparison plots.

Usage:
    # Run baseline only
    python run_benchmark.py --tag baseline
    
    # Run modified (after making changes)
    python run_benchmark.py --tag modified
    
    # Compare results
    python run_benchmark.py --compare
"""

import argparse
import subprocess
import sys
import os


def run_inference(tag: str = "baseline", **kwargs):
    """Run inference with benchmarking enabled."""
    
    block_size = kwargs.get('block_size', 4)
    length = kwargs.get('length', 256)
    diffusion_steps = kwargs.get('diffusion_steps', 100)
    checkpoint = kwargs.get('checkpoint', 'kuleshov-group/bd3lm-owt-block_size4')
    
    cmd = [
        "python", "-u", "main.py",
        "loader.eval_batch_size=1",
        "model=small",
        "algo=bd3lm",
        f"algo.T={diffusion_steps}",
        "algo.backbone=hf_dit",
        "data=openwebtext-split",
        f"model.length={length}",
        f"block_size={block_size}",
        "wandb=null",
        "mode=sample_eval",
        f"eval.checkpoint_path={checkpoint}",
        "model.attn_backend=sdpa",
        "sampling.nucleus_p=0.9",
        "sampling.kv_cache=true",
        f"sampling.logdir=./sample_logs/{tag}_samples",
        # Enable benchmarking via environment variable
        f"+benchmark_tag={tag}",
    ]
    
    # Set environment variable to enable benchmarking
    env = os.environ.copy()
    env['BD3LM_BENCHMARK_TAG'] = tag
    
    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} inference benchmark")
    print(f"{'='*60}")
    print(f"Block size: {block_size}")
    print(f"Sequence length: {length}")
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def run_visualization():
    """Run visualization script to compare results."""
    cmd = ["python", "visualize_results.py", "--results_dir", "./results/logs"]
    print(f"\n{'='*60}")
    print("Generating comparison visualizations")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='BD3-LM Benchmark Runner')
    parser.add_argument('--tag', type=str, default='baseline',
                       choices=['baseline', 'modified'],
                       help='Tag for this run (baseline or modified)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison visualizations')
    parser.add_argument('--block_size', type=int, default=4,
                       help='Block size for generation')
    parser.add_argument('--length', type=int, default=256,
                       help='Sequence length to generate')
    parser.add_argument('--diffusion_steps', type=int, default=100,
                       help='Number of diffusion steps')
    parser.add_argument('--checkpoint', type=str, 
                       default='kuleshov-group/bd3lm-owt-block_size4',
                       help='Model checkpoint path')
    args = parser.parse_args()
    
    if args.compare:
        run_visualization()
    else:
        success = run_inference(
            tag=args.tag,
            block_size=args.block_size,
            length=args.length,
            diffusion_steps=args.diffusion_steps,
            checkpoint=args.checkpoint,
        )
        if success:
            print(f"\n[Benchmark] {args.tag} run completed successfully!")
            print(f"[Benchmark] Results saved to: ./results/logs/{args.tag}_*.json")
        else:
            print(f"\n[Benchmark] {args.tag} run failed!")
            sys.exit(1)


if __name__ == '__main__':
    main()

