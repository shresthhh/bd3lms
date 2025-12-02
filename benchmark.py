"""
Benchmarking module for BD3-LM inference timing and metrics.
Tracks block generation times, TTCB (Time to First Coherent Block), and other metrics.
"""

import time
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import torch


@dataclass
class BlockMetrics:
    """Metrics for a single block generation."""
    block_idx: int
    start_time: float
    end_time: float
    duration_ms: float
    num_diffusion_steps: int
    is_coherent: bool = False
    entropy: float = 0.0
    tokens_unmasked: int = 0
    prediction_entropy: float = 0.0  # Model's prediction entropy at step 0
    early_exit_step: int = 0  # Step at which early exit was triggered (0 = no early exit)


@dataclass 
class InferenceMetrics:
    """Complete metrics for an inference run."""
    run_id: str
    timestamp: str
    model_checkpoint: str
    block_size: int
    sequence_length: int
    num_diffusion_steps: int
    
    # Timing metrics
    total_time_ms: float = 0.0
    ttfb_ms: float = 0.0  # Time to First Block
    ttcb_ms: float = 0.0  # Time to First Coherent Block
    avg_block_time_ms: float = 0.0
    
    # Block-level metrics
    block_metrics: List[Dict] = field(default_factory=list)
    num_blocks_generated: int = 0
    num_coherent_blocks: int = 0
    
    # Quality metrics
    final_entropy: float = 0.0
    generative_ppl: float = 0.0
    
    # Generated text sample
    generated_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InferenceBenchmark:
    """Benchmark tracker for BD3-LM inference."""
    
    def __init__(
        self,
        model_checkpoint: str,
        block_size: int,
        sequence_length: int,
        num_diffusion_steps: int,
        results_dir: str = "./results/logs",
        coherence_entropy_threshold: float = 4.0,
    ):
        self.model_checkpoint = model_checkpoint
        self.block_size = block_size
        self.sequence_length = sequence_length
        self.num_diffusion_steps = num_diffusion_steps
        self.results_dir = results_dir
        self.coherence_entropy_threshold = coherence_entropy_threshold
        
        # Initialize metrics
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = datetime.now().isoformat()
        
        # Timing state
        self._inference_start: Optional[float] = None
        self._block_start: Optional[float] = None
        self._first_block_time: Optional[float] = None
        self._first_coherent_block_time: Optional[float] = None
        
        # Block tracking
        self._block_metrics: List[BlockMetrics] = []
        self._current_block_idx = 0
        self._diffusion_steps_in_block = 0
        
        os.makedirs(results_dir, exist_ok=True)
    
    def start_inference(self):
        """Call at the start of inference."""
        self._inference_start = time.perf_counter()
        self._current_block_idx = 0
        self._block_metrics = []
        self._first_block_time = None
        self._first_coherent_block_time = None
    
    def start_block(self):
        """Call at the start of each block generation."""
        self._block_start = time.perf_counter()
        self._diffusion_steps_in_block = 0
    
    def record_diffusion_step(self):
        """Call after each diffusion step within a block."""
        self._diffusion_steps_in_block += 1
    
    def end_block(self, block_tokens: torch.Tensor, is_coherent: bool = False,
                  prediction_entropy: float = 0.0, early_exit_step: int = 0):
        """Call at the end of each block generation.
        
        Args:
            block_tokens: Generated tokens for this block
            is_coherent: Whether the block is coherent
            prediction_entropy: Model's prediction entropy at step 0 (for early exit decisions)
            early_exit_step: Step at which early exit was triggered (0 = no early exit)
        """
        end_time = time.perf_counter()
        start_time = self._block_start or end_time
        duration_ms = (end_time - start_time) * 1000
        
        # Calculate entropy for this block (from final tokens)
        if block_tokens is not None and len(block_tokens) > 0:
            _, counts = torch.unique(block_tokens, return_counts=True, sorted=False)
            entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
            tokens_unmasked = len(block_tokens)
        else:
            entropy = 0.0
            tokens_unmasked = 0
        
        # Check coherence based on entropy threshold
        is_coherent = entropy >= self.coherence_entropy_threshold
        
        block_metric = BlockMetrics(
            block_idx=self._current_block_idx,
            start_time=start_time - self._inference_start,
            end_time=end_time - self._inference_start,
            duration_ms=duration_ms,
            num_diffusion_steps=self._diffusion_steps_in_block,
            is_coherent=is_coherent,
            entropy=entropy,
            tokens_unmasked=tokens_unmasked,
            prediction_entropy=prediction_entropy,
            early_exit_step=early_exit_step,
        )
        self._block_metrics.append(block_metric)
        
        # Track first block time
        if self._first_block_time is None:
            self._first_block_time = end_time - self._inference_start
        
        # Track first coherent block time
        if is_coherent and self._first_coherent_block_time is None:
            self._first_coherent_block_time = end_time - self._inference_start
        
        self._current_block_idx += 1
        return block_metric
    
    def end_inference(
        self, 
        generated_text: str = "",
        generative_ppl: float = 0.0,
        final_entropy: float = 0.0,
    ) -> InferenceMetrics:
        """Call at the end of inference. Returns complete metrics."""
        end_time = time.perf_counter()
        total_time_ms = (end_time - self._inference_start) * 1000
        
        # Calculate aggregate metrics
        num_blocks = len(self._block_metrics)
        num_coherent = sum(1 for b in self._block_metrics if b.is_coherent)
        avg_block_time = sum(b.duration_ms for b in self._block_metrics) / max(num_blocks, 1)
        
        metrics = InferenceMetrics(
            run_id=self.run_id,
            timestamp=self.timestamp,
            model_checkpoint=self.model_checkpoint,
            block_size=self.block_size,
            sequence_length=self.sequence_length,
            num_diffusion_steps=self.num_diffusion_steps,
            total_time_ms=total_time_ms,
            ttfb_ms=(self._first_block_time or 0) * 1000,
            ttcb_ms=(self._first_coherent_block_time or 0) * 1000,
            avg_block_time_ms=avg_block_time,
            block_metrics=[asdict(b) for b in self._block_metrics],
            num_blocks_generated=num_blocks,
            num_coherent_blocks=num_coherent,
            final_entropy=final_entropy,
            generative_ppl=generative_ppl,
            generated_text=generated_text[:500],  # Truncate for log
        )
        
        return metrics
    
    def save_metrics(self, metrics: InferenceMetrics, tag: str = "baseline"):
        """Save metrics to a JSON log file."""
        filename = f"{tag}_{self.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"[Benchmark] Saved metrics to: {filepath}")
        return filepath
    
    def print_summary(self, metrics: InferenceMetrics):
        """Print a summary of the metrics."""
        print("\n" + "="*60)
        print(f"INFERENCE BENCHMARK RESULTS - {metrics.run_id}")
        print("="*60)
        print(f"Timestamp:              {metrics.timestamp}")
        print(f"Model:                  {metrics.model_checkpoint}")
        print(f"Block Size:             {metrics.block_size}")
        print(f"Sequence Length:        {metrics.sequence_length}")
        print(f"Diffusion Steps:        {metrics.num_diffusion_steps}")
        print("-"*60)
        print("TIMING METRICS:")
        print(f"  Total Time:           {metrics.total_time_ms:.2f} ms")
        print(f"  Time to First Block:  {metrics.ttfb_ms:.2f} ms")
        print(f"  Time to First Coherent Block (TTCB): {metrics.ttcb_ms:.2f} ms")
        print(f"  Avg Block Time:       {metrics.avg_block_time_ms:.2f} ms")
        print("-"*60)
        print("BLOCK METRICS:")
        print(f"  Blocks Generated:     {metrics.num_blocks_generated}")
        print(f"  Coherent Blocks:      {metrics.num_coherent_blocks}")
        print("-"*60)
        print("QUALITY METRICS:")
        print(f"  Generative PPL:       {metrics.generative_ppl:.4f}")
        print(f"  Final Entropy:        {metrics.final_entropy:.4f}")
        print("="*60 + "\n")


def create_benchmark(config) -> InferenceBenchmark:
    """Factory function to create a benchmark from a config."""
    return InferenceBenchmark(
        model_checkpoint=config.eval.checkpoint_path,
        block_size=config.block_size,
        sequence_length=config.model.length,
        num_diffusion_steps=config.algo.T,
        results_dir="./results/logs",
    )

