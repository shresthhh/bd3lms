#!/usr/bin/env python3
"""
Visualization script for BD3-LM benchmark results.
Compares baseline vs modified inference metrics.

Usage:
    python visualize_results.py --baseline results/logs/baseline_*.json --modified results/logs/modified_*.json
    python visualize_results.py --results_dir results/logs/
"""

import argparse
import json
import glob
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_metrics(pattern: str) -> List[Dict[str, Any]]:
    """Load all metrics matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    return [load_metrics(f) for f in files]


def extract_tag(filename: str) -> str:
    """Extract tag (baseline/modified) from filename."""
    basename = os.path.basename(filename)
    return basename.split('_')[0]


def plot_timing_comparison(
    baseline_metrics: List[Dict],
    modified_metrics: List[Dict],
    output_dir: str = "./results/plots"
):
    """Create timing comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract timing data
    baseline_times = {
        'total': [m['total_time_ms'] for m in baseline_metrics],
        'ttfb': [m['ttfb_ms'] for m in baseline_metrics],
        'ttcb': [m['ttcb_ms'] for m in baseline_metrics],
        'avg_block': [m['avg_block_time_ms'] for m in baseline_metrics],
    }
    
    modified_times = {
        'total': [m['total_time_ms'] for m in modified_metrics],
        'ttfb': [m['ttfb_ms'] for m in modified_metrics],
        'ttcb': [m['ttcb_ms'] for m in modified_metrics],
        'avg_block': [m['avg_block_time_ms'] for m in modified_metrics],
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('BD3-LM Inference Timing Comparison', fontsize=14, fontweight='bold')
    
    metrics_labels = [
        ('total', 'Total Inference Time (ms)'),
        ('ttfb', 'Time to First Block (ms)'),
        ('ttcb', 'Time to First Coherent Block - TTCB (ms)'),
        ('avg_block', 'Average Block Generation Time (ms)'),
    ]
    
    colors = ['#2ecc71', '#e74c3c']  # Green for baseline, Red for modified
    
    for idx, (key, label) in enumerate(metrics_labels):
        ax = axes[idx // 2, idx % 2]
        
        baseline_data = baseline_times[key]
        modified_data = modified_times[key]
        
        # Bar positions
        x = np.arange(2)
        width = 0.6
        
        baseline_mean = np.mean(baseline_data) if baseline_data else 0
        modified_mean = np.mean(modified_data) if modified_data else 0
        baseline_std = np.std(baseline_data) if len(baseline_data) > 1 else 0
        modified_std = np.std(modified_data) if len(modified_data) > 1 else 0
        
        bars = ax.bar(x, [baseline_mean, modified_mean], width, 
                     yerr=[baseline_std, modified_std],
                     color=colors, capsize=5, alpha=0.8)
        
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'Modified'])
        ax.set_title(label)
        
        # Add value labels on bars
        for bar, val in zip(bars, [baseline_mean, modified_mean]):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        # Calculate improvement
        if baseline_mean > 0:
            improvement = ((baseline_mean - modified_mean) / baseline_mean) * 100
            color = 'green' if improvement > 0 else 'red'
            ax.text(0.5, 0.95, f'Change: {improvement:+.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'timing_comparison_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved timing comparison: {filepath}")
    plt.close()
    
    return filepath


def plot_block_timeline(
    metrics: Dict[str, Any],
    output_dir: str = "./results/plots",
    tag: str = "baseline"
):
    """Plot timeline of block generation."""
    os.makedirs(output_dir, exist_ok=True)
    
    block_metrics = metrics.get('block_metrics', [])
    if not block_metrics:
        print("[Visualize] No block metrics to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Block Generation Timeline ({tag})', fontsize=14, fontweight='bold')
    
    # Extract data
    block_indices = [b['block_idx'] for b in block_metrics]
    durations = [b['duration_ms'] for b in block_metrics]
    entropies = [b['entropy'] for b in block_metrics]
    coherent = [b['is_coherent'] for b in block_metrics]
    
    # Plot 1: Block duration timeline
    colors = ['#2ecc71' if c else '#e74c3c' for c in coherent]
    ax1.bar(block_indices, durations, color=colors, alpha=0.8)
    ax1.set_ylabel('Duration (ms)')
    ax1.set_title('Block Generation Duration (Green = Coherent)')
    ax1.axhline(y=np.mean(durations), color='blue', linestyle='--', 
                label=f'Avg: {np.mean(durations):.1f}ms')
    ax1.legend()
    
    # Plot 2: Entropy over blocks
    ax2.plot(block_indices, entropies, 'b-o', markersize=4)
    ax2.axhline(y=4.0, color='red', linestyle='--', label='Coherence Threshold')
    ax2.fill_between(block_indices, entropies, 4.0, 
                     where=[e >= 4.0 for e in entropies],
                     alpha=0.3, color='green', label='Coherent')
    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Block Entropy Over Time')
    ax2.legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'block_timeline_{tag}_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved block timeline: {filepath}")
    plt.close()
    
    return filepath


def plot_quality_comparison(
    baseline_metrics: List[Dict],
    modified_metrics: List[Dict],
    output_dir: str = "./results/plots"
):
    """Create quality metrics comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Quality Metrics Comparison', fontsize=14, fontweight='bold')
    
    # Generative PPL
    baseline_ppl = [m['generative_ppl'] for m in baseline_metrics if m['generative_ppl'] > 0]
    modified_ppl = [m['generative_ppl'] for m in modified_metrics if m['generative_ppl'] > 0]
    
    ax = axes[0]
    x = np.arange(2)
    baseline_mean = np.mean(baseline_ppl) if baseline_ppl else 0
    modified_mean = np.mean(modified_ppl) if modified_ppl else 0
    bars = ax.bar(x, [baseline_mean, modified_mean], color=['#3498db', '#9b59b6'], alpha=0.8)
    ax.set_ylabel('Generative Perplexity')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Modified'])
    ax.set_title('Generative Perplexity (lower is better)')
    for bar, val in zip(bars, [baseline_mean, modified_mean]):
        if val > 0:
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Coherent blocks ratio
    baseline_coherent = [m['num_coherent_blocks'] / max(m['num_blocks_generated'], 1) 
                        for m in baseline_metrics]
    modified_coherent = [m['num_coherent_blocks'] / max(m['num_blocks_generated'], 1) 
                        for m in modified_metrics]
    
    ax = axes[1]
    baseline_mean = np.mean(baseline_coherent) * 100 if baseline_coherent else 0
    modified_mean = np.mean(modified_coherent) * 100 if modified_coherent else 0
    bars = ax.bar(x, [baseline_mean, modified_mean], color=['#3498db', '#9b59b6'], alpha=0.8)
    ax.set_ylabel('Coherent Blocks (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Modified'])
    ax.set_title('Percentage of Coherent Blocks')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [baseline_mean, modified_mean]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'quality_comparison_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved quality comparison: {filepath}")
    plt.close()
    
    return filepath


def generate_summary_report(
    baseline_metrics: List[Dict],
    modified_metrics: List[Dict],
    output_dir: str = "./results"
):
    """Generate a text summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'summary_report_{timestamp}.txt')
    
    def avg(lst, key):
        values = [m[key] for m in lst if m.get(key, 0) > 0]
        return np.mean(values) if values else 0
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BD3-LM INFERENCE BENCHMARK SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        if baseline_metrics:
            m = baseline_metrics[0]
            f.write(f"Model:            {m.get('model_checkpoint', 'N/A')}\n")
            f.write(f"Block Size:       {m.get('block_size', 'N/A')}\n")
            f.write(f"Sequence Length:  {m.get('sequence_length', 'N/A')}\n")
            f.write(f"Diffusion Steps:  {m.get('num_diffusion_steps', 'N/A')}\n")
        f.write("\n")
        
        f.write("TIMING COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<35} {'Baseline':>12} {'Modified':>12} {'Change':>12}\n")
        f.write("-"*70 + "\n")
        
        timing_keys = [
            ('total_time_ms', 'Total Time (ms)'),
            ('ttfb_ms', 'Time to First Block (ms)'),
            ('ttcb_ms', 'TTCB (ms)'),
            ('avg_block_time_ms', 'Avg Block Time (ms)'),
        ]
        
        for key, label in timing_keys:
            b_val = avg(baseline_metrics, key)
            m_val = avg(modified_metrics, key)
            if b_val > 0:
                change = ((b_val - m_val) / b_val) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"
            f.write(f"{label:<35} {b_val:>12.2f} {m_val:>12.2f} {change_str:>12}\n")
        
        f.write("\n")
        f.write("QUALITY COMPARISON\n")
        f.write("-"*70 + "\n")
        
        b_ppl = avg(baseline_metrics, 'generative_ppl')
        m_ppl = avg(modified_metrics, 'generative_ppl')
        f.write(f"{'Generative PPL':<35} {b_ppl:>12.2f} {m_ppl:>12.2f}\n")
        
        b_blocks = avg(baseline_metrics, 'num_blocks_generated')
        m_blocks = avg(modified_metrics, 'num_blocks_generated')
        f.write(f"{'Blocks Generated':<35} {b_blocks:>12.0f} {m_blocks:>12.0f}\n")
        
        b_coherent = avg(baseline_metrics, 'num_coherent_blocks')
        m_coherent = avg(modified_metrics, 'num_coherent_blocks')
        f.write(f"{'Coherent Blocks':<35} {b_coherent:>12.0f} {m_coherent:>12.0f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"[Visualize] Saved summary report: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Visualize BD3-LM benchmark results')
    parser.add_argument('--baseline', type=str, help='Glob pattern for baseline metrics')
    parser.add_argument('--modified', type=str, help='Glob pattern for modified metrics')
    parser.add_argument('--results_dir', type=str, default='./results/logs',
                       help='Directory containing result logs')
    parser.add_argument('--output_dir', type=str, default='./results/plots',
                       help='Directory for output plots')
    args = parser.parse_args()
    
    # Load metrics
    if args.baseline and args.modified:
        baseline_metrics = load_all_metrics(args.baseline)
        modified_metrics = load_all_metrics(args.modified)
    else:
        # Auto-discover from results_dir
        baseline_pattern = os.path.join(args.results_dir, 'baseline_*.json')
        modified_pattern = os.path.join(args.results_dir, 'modified_*.json')
        baseline_metrics = load_all_metrics(baseline_pattern)
        modified_metrics = load_all_metrics(modified_pattern)
    
    print(f"[Visualize] Loaded {len(baseline_metrics)} baseline, {len(modified_metrics)} modified metrics")
    
    if not baseline_metrics and not modified_metrics:
        print("[Visualize] No metrics found. Run inference first.")
        return
    
    # Generate plots
    if baseline_metrics and modified_metrics:
        plot_timing_comparison(baseline_metrics, modified_metrics, args.output_dir)
        plot_quality_comparison(baseline_metrics, modified_metrics, args.output_dir)
        generate_summary_report(baseline_metrics, modified_metrics, 
                               os.path.dirname(args.output_dir))
    
    # Plot individual timelines
    if baseline_metrics:
        plot_block_timeline(baseline_metrics[0], args.output_dir, "baseline")
    if modified_metrics:
        plot_block_timeline(modified_metrics[0], args.output_dir, "modified")
    
    print("[Visualize] Done!")


if __name__ == '__main__':
    main()

