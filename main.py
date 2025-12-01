import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils
from tqdm import tqdm
import time
import json
from pathlib import Path

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  # Enable benchmarking if requested via env var or config
  benchmark_tag = os.environ.get('BD3LM_BENCHMARK_TAG', None)
  if benchmark_tag is None and hasattr(config, 'benchmark_tag'):
    benchmark_tag = config.benchmark_tag
  if benchmark_tag:
    logger.info(f'Enabling benchmark with tag: {benchmark_tag}')
    model.enable_benchmark(tag=benchmark_tag)
  
  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _block_metrics_eval(config, logger, tokenizer):
  """Evaluate block-specific metrics."""
  logger.info('Starting Block Metrics Evaluation.')
  
  # Load model
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  model.eval()
  
  results = {}
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  # Get number of samples to generate
  num_samples = config.eval.get('num_samples', 10)
  batch_size = config.loader.eval_batch_size
  num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
  
  logger.info(f'Generating {num_samples} samples in {num_batches} batches...')
  
  text_samples = []
  generation_times = []
  batch_indices = list(range(num_batches))

  for batch_idx in tqdm(batch_indices,desc='Generating batches'):
    start_time = time.time()
    
    # Generate a batch of samples
    # restore_model_and_sample generates batch_size_per_gpu samples
    batch_samples = model.restore_model_and_sample(
      num_steps=config.algo.T,
      seqlen=config.model.length
    )
    
    elapsed = time.time() - start_time
    
    # Track timing per sample in batch
    samples_in_batch = len(batch_samples) if isinstance(batch_samples, list) else 1
    time_per_sample = elapsed / samples_in_batch
    
    for _ in range(samples_in_batch):
      generation_times.append(time_per_sample)
    
    # Collect samples
    if isinstance(batch_samples, list):
      text_samples.extend(batch_samples)
    else:
      text_samples.append(batch_samples)
    
    # Stop if we have enough samples
    if len(text_samples) >= num_samples:
      break
  
  # Trim to exact number requested
  text_samples = text_samples[:num_samples]
  generation_times = generation_times[:num_samples]
  
  logger.info(f'Generated {len(text_samples)} samples.')
  
  # Compute Block Efficiency Ratio
  logger.info('Computing Block Efficiency Ratio...')
  
  num_blocks = config.model.length // config.block_size
  
  # Reset metrics before computing
  model.metrics.gen_ppl.reset()
  model.metrics.gen_entropy.reset()
  
  ber_result = model.metrics.record_block_efficiency_ratio(
      text_samples=text_samples,
      nfes_per_block=config.algo.T,
      num_blocks=num_blocks,
      max_length=config.model.length,
      device=device
  )
  results['block_efficiency_ratio'] = ber_result
  
  logger.info(f'Block Efficiency Ratio: {ber_result["block_efficiency_ratio"]:.8f}')
  logger.info(f'Generative Perplexity: {ber_result["generative_perplexity"]:.2f}')
  
  # Compute Time to First Block
  logger.info('Computing Time to First Block...')
  
  ttfb_result = model.metrics.record_time_to_first_block(
      generation_times=generation_times,
      block_size=config.block_size,
      model_length=config.model.length
  )
  results['time_to_first_block'] = ttfb_result
  
  logger.info(f'Time to First Block: {ttfb_result["time_to_first_block_ms"]:.2f} ms')
  logger.info(f'Throughput: {ttfb_result["throughput_tokens_per_sec"]:.1f} tokens/sec')
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  
  output_file = output_dir / f'block_metrics_bs{config.block_size}.json'
  
  # Add config info to results
  results['config'] = {
      'block_size': config.block_size,
      'model_length': config.model.length,
      'num_samples': len(text_samples),
      'diffusion_steps': config.algo.T,
      'checkpoint': config.eval.checkpoint_path
  }
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  logger.info(f'Results saved to: {output_file}')
  
  # Print summary
  print('\n' + '='*80)
  print('BLOCK METRICS SUMMARY')
  print('='*80)
  print(f'Checkpoint: {config.eval.checkpoint_path}')
  print(f'Block Size: {config.block_size}')
  print(f'Samples Generated: {len(text_samples)}')
  print(f'Model Length: {config.model.length}')
  print(f'Diffusion Steps (T): {config.algo.T}')
  print(f'\nGENERATIVE QUALITY:')
  print(f'  Perplexity: {ber_result["generative_perplexity"]:.2f}')
  print(f'  Quality Score: {ber_result["quality_score"]:.6f}')
  print(f'\nEFFICIENCY METRICS:')
  print(f'  Block Efficiency Ratio: {ber_result["block_efficiency_ratio"]:.8f}')
  print(f'  Total NFEs: {ber_result["total_nfes"]}')
  print(f'  NFEs per Block: {ber_result["nfes_per_block"]}')
  print(f'\nSPEED METRICS:')
  print(f'  Time to First Block: {ttfb_result["time_to_first_block_ms"]:.2f} ms')
  print(f'  Avg Generation Time: {ttfb_result["avg_generation_time_s"]:.2f} s')
  print(f'  Throughput: {ttfb_result["throughput_tokens_per_sec"]:.1f} tokens/sec')
  print('='*80 + '\n')
  
  return results

def _comprehensive_block_metrics_eval(config, logger, tokenizer):
  """Evaluate comprehensive block-specific metrics with all new additions."""
  logger.info('Starting Comprehensive Block Metrics Evaluation.')
  
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  model.eval()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  results = {}
  num_samples = config.eval.get('num_samples', 50)
  batch_size = config.loader.eval_batch_size
  num_batches = (num_samples + batch_size - 1) // batch_size
  
  logger.info(f'Generating {num_samples} samples in {num_batches} batches...')
  
  text_samples = []
  generation_times = []

  for batch_idx in tqdm(range(num_batches), desc='Generating batches'):
    start_time = time.time()
    
    batch_samples = model.restore_model_and_sample(
      num_steps=config.algo.T,
      seqlen=config.model.length
    )
    
    elapsed = time.time() - start_time
    
    samples_in_batch = len(batch_samples) if isinstance(batch_samples, list) else 1
    time_per_sample = elapsed / samples_in_batch
    
    for _ in range(samples_in_batch):
      generation_times.append(time_per_sample)
    
    if isinstance(batch_samples, list):
      text_samples.extend(batch_samples)
    else:
      text_samples.append(batch_samples)
    
    if len(text_samples) >= num_samples:
      break
  
  text_samples = text_samples[:num_samples]
  generation_times = generation_times[:num_samples]
  
  logger.info(f'Generated {len(text_samples)} samples.')
  
  num_blocks = config.model.length // config.block_size
  
  # === CORE METRICS ===
  
  # 1. Block Efficiency Ratio
  logger.info('Computing Block Efficiency Ratio...')
  model.metrics.gen_ppl.reset()
  model.metrics.gen_entropy.reset()
  
  results['block_efficiency'] = model.metrics.record_block_efficiency_ratio(
      text_samples=text_samples,
      nfes_per_block=config.algo.T,
      num_blocks=num_blocks,
      max_length=config.model.length,
      device=device
  )
  
  # 2. Time to First Block
  logger.info('Computing Time to First Block...')
  results['time_metrics'] = model.metrics.record_time_to_first_block(
      generation_times=generation_times,
      block_size=config.block_size,
      model_length=config.model.length
  )
  
  # === NEW METRICS ===
  
  # 3. Block Variance Analysis
  logger.info('Computing Block Variance Analysis...')
  results['block_variance'] = model.metrics.record_block_variance_analysis(
      text_samples=text_samples,
      block_size=config.block_size,
      model_length=config.model.length,
      device=device
  )
  
  # 4. Token-Level Confidence
  logger.info('Computing Token-Level Confidence...')
  results['token_confidence'] = model.metrics.record_token_level_confidence(
      text_samples=text_samples,
      model_length=config.model.length,
      block_size=config.block_size,
      device=device
  )
  
  # 5. Repetition Metrics
  logger.info('Computing Repetition Metrics...')
  results['repetition'] = model.metrics.record_repetition_metrics(
      text_samples=text_samples,
      n_gram_sizes=[2, 3, 4]
  )
  
  # 6. Computational Breakdown
  logger.info('Computing Computational Breakdown...')
  results['computational'] = model.metrics.record_computational_breakdown(
      generation_times=generation_times,
      block_size=config.block_size,
      model_length=config.model.length,
      diffusion_steps=config.algo.T
  )
  
  # 7. Quality Consistency
  logger.info('Computing Quality Consistency...')
  results['consistency'] = model.metrics.record_quality_consistency(
      text_samples=text_samples,
      model_length=config.model.length,
      device=device,
      num_runs=min(3, len(text_samples) // 10)
  )
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  
  output_file = output_dir / f'comprehensive_metrics_bs{config.block_size}.json'
  
  results['config'] = {
      'block_size': config.block_size,
      'model_length': config.model.length,
      'num_samples': len(text_samples),
      'diffusion_steps': config.algo.T,
      'checkpoint': config.eval.checkpoint_path
  }
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  logger.info(f'Results saved to: {output_file}')
  
  # === COMPREHENSIVE SUMMARY ===
  print('\n' + '='*80)
  print('COMPREHENSIVE BD3-LM METRICS REPORT')
  print('='*80)
  print(f'Checkpoint: {config.eval.checkpoint_path}')
  print(f'Configuration: Block Size={config.block_size}, Length={config.model.length}, T={config.algo.T}')
  print(f'Samples Analyzed: {len(text_samples)}')
  
  print(f'\n📊 QUALITY METRICS:')
  print(f'  Generative Perplexity: {results["block_efficiency"]["generative_perplexity"]:.2f}')
  print(f'  Quality Score: {results["block_efficiency"]["quality_score"]:.6f}')
  print(f'  Quality Consistency (std): {results["consistency"]["ppl_std"]:.2f}')
  print(f'  Is Consistent: {"✓" if results["consistency"]["is_consistent"] else "✗"}')
  
  print(f'\n⚡ EFFICIENCY METRICS:')
  print(f'  Block Efficiency Ratio: {results["block_efficiency"]["block_efficiency_ratio"]:.8f}')
  print(f'  Time to First Block: {results["time_metrics"]["time_to_first_block_ms"]:.2f} ms')
  print(f'  Throughput: {results["time_metrics"]["throughput_tokens_per_sec"]:.1f} tokens/sec')
  print(f'  Speedup vs AR: {results["computational"]["speedup_vs_ar"]:.2f}x')
  print(f'  Parallel Efficiency: {results["computational"]["parallel_efficiency_pct"]:.1f}%')
  
  print(f'\n BLOCK-LEVEL ANALYSIS:')
  print(f'  Block PPL Mean: {results["block_variance"]["block_ppl_mean"]:.2f}')
  print(f'  Block PPL Std: {results["block_variance"]["block_ppl_std"]:.2f}')
  print(f'  Block PPL Range: {results["block_variance"]["block_ppl_range"]:.2f}')
  print(f'  Quality CV: {results["block_variance"]["block_ppl_cv"]:.4f}')
  print(f'  Degradation Trend: {results["block_variance"]["degradation_trend"]:.4f}')
  
  print(f'\n CONFIDENCE METRICS:')
  print(f'  Overall Avg Confidence: {results["token_confidence"]["overall_avg_confidence"]:.4f}')
  print(f'  Overall Avg Entropy: {results["token_confidence"]["overall_avg_entropy"]:.4f}')
  print(f'  Confidence Std: {results["token_confidence"]["confidence_std"]:.4f}')
  print(f'  First Block Confidence: {results["token_confidence"]["first_block_confidence"]:.4f}')
  print(f'  Last Block Confidence: {results["token_confidence"]["last_block_confidence"]:.4f}')
  
  print(f'\n REPETITION ANALYSIS:')
  print(f'  Overall Repetition Score: {results["repetition"]["summary"]["overall_repetition_score"]:.4f}')
  print(f'  Avg Unique Ratio: {results["repetition"]["summary"]["avg_unique_ratio"]:.4f}')
  print(f'  Has Excessive Repetition: {"⚠️ YES" if results["repetition"]["summary"]["has_excessive_repetition"] else "✓ NO"}')
  for n in [2, 3, 4]:
    print(f'  {n}-gram Unique Ratio: {results["repetition"][f"{n}gram"]["unique_ratio_mean"]:.4f}')
  
  print(f'\n  COMPUTATIONAL BREAKDOWN:')
  print(f'  Time per Block: {results["computational"]["time_per_block_ms"]:.2f} ms')
  print(f'  Time per Diffusion Step: {results["computational"]["time_per_diffusion_step_ms"]:.3f} ms')
  print(f'  Blocks per Second: {results["computational"]["blocks_per_second"]:.2f}')
  print(f'  Overhead per Block: {results["computational"]["overhead_per_block_ms"]:.2f} ms')
  
  print('='*80 + '\n')
  
  # Additional diagnostic warnings
  if results["block_variance"]["degradation_trend"] > 0.5:
    print("  WARNING: Significant quality degradation detected across blocks (trend > 0.5)")
  
  if results["repetition"]["summary"]["has_excessive_repetition"]:
    print(" WARNING: Excessive repetition detected. Check for mode collapse.")
  
  if results["computational"]["parallel_efficiency_pct"] < 50:
    print("  WARNING: Low parallel efficiency. Consider checking implementation bottlenecks.")
  
  if not results["consistency"]["is_consistent"]:
    print(" WARNING: High variance in quality across runs. Results may be unstable.")
  
  print()
  
  return results


def _sampling_efficiency_eval(config, logger, tokenizer):
  """
  Evaluate quality-compute tradeoff across different T values.
  Usage: python main.py mode=sampling_efficiency_eval eval.T_values=[100,500,1000,2500,5000]
  """
  logger.info('Starting Sampling Efficiency Evaluation.')
  
  T_values = config.eval.get('T_values', [100, 500, 1000, 2500, 5000])
  num_samples_per_T = config.eval.get('num_samples', 20)
  
  samples_by_steps = {}
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  for T in T_values:
    logger.info(f'Generating samples with T={T}')
    
    # Update config
    config.algo.T = T
    
    # Load model
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
      model.ema = None
    model.eval()
    
    # Generate samples
    text_samples = []
    batch_size = config.loader.eval_batch_size
    num_batches = (num_samples_per_T + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches), desc=f'T={T}'):
      batch_samples = model.restore_model_and_sample(
          num_steps=T,
          seqlen=config.model.length
      )
      if isinstance(batch_samples, list):
        text_samples.extend(batch_samples)
      else:
        text_samples.append(batch_samples)
      
      if len(text_samples) >= num_samples_per_T:
        break
    
    samples_by_steps[T] = text_samples[:num_samples_per_T]
  
  # Analyze efficiency curve
  logger.info('Analyzing sampling efficiency curve...')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  
  results = model.metrics.record_sampling_efficiency_curve(
      samples_by_steps=samples_by_steps,
      steps_list=T_values,
      model_length=config.model.length,
      device=device
  )
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / f'sampling_efficiency_bs{config.block_size}.json'
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  logger.info(f'Results saved to: {output_file}')
  
  # Print summary
  print('\n' + '='*80)
  print('SAMPLING EFFICIENCY ANALYSIS')
  print('='*80)
  print(f'Block Size: {config.block_size}')
  print(f'Optimal T (knee point): {results["optimal_T"]}')
  print(f'Best Quality T: {results["min_ppl_T"]}')
  print(f'Diminishing Returns Threshold: {results["diminishing_returns_threshold"]:.6f}')
  
  print('\nEfficiency Curve:')
  print(f'{"T":<8} {"PPL":<10} {"Δ PPL/ΔT":<15}')
  print('-' * 80)
  for point in results['efficiency_curve']:
    improvement = point['quality_improvement_per_step']
    print(f'{point["diffusion_steps"]:<8} {point["perplexity"]:<10.2f} {improvement:<15.6f}')
  print('='*80 + '\n')
  
  return results


def _length_robustness_eval(config, logger, tokenizer):
  """
  Evaluate model performance across different sequence lengths.
  Usage: python main.py mode=length_robustness_eval eval.test_lengths=[512,1024,2048,4096]
  """
  logger.info('Starting Length Robustness Evaluation.')
  
  test_lengths = config.eval.get('test_lengths', [512, 1024, 2048])
  num_samples_per_length = config.eval.get('num_samples', 20)
  
  samples_by_length = {}
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  for length in test_lengths:
    logger.info(f'Generating samples of length {length}')
    
    # Update config
    config.model.length = length
    
    # Load model
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
      model.ema = None
    model.eval()
    
    # Generate samples
    text_samples = []
    batch_size = config.loader.eval_batch_size
    num_batches = (num_samples_per_length + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches), desc=f'Length={length}'):
      batch_samples = model.restore_model_and_sample(
          num_steps=config.algo.T,
          seqlen=length
      )
      if isinstance(batch_samples, list):
        text_samples.extend(batch_samples)
      else:
        text_samples.append(batch_samples)
      
      if len(text_samples) >= num_samples_per_length:
        break
    
    samples_by_length[length] = text_samples[:num_samples_per_length]
  
  # Analyze length robustness
  logger.info('Analyzing length robustness...')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  
  results = model.metrics.record_length_robustness(
      samples_by_length=samples_by_length,
      lengths=test_lengths,
      device=device
  )
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / f'length_robustness_bs{config.block_size}.json'
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  logger.info(f'Results saved to: {output_file}')
  
  # Print summary
  print('\n' + '='*80)
  print('LENGTH ROBUSTNESS ANALYSIS')
  print('='*80)
  print(f'Block Size: {config.block_size}')
  print(f'Length-PPL Correlation: {results["length_ppl_correlation"]:.4f}')
  print(f'PPL Std Across Lengths: {results["ppl_std_across_lengths"]:.2f}')
  print(f'Maintains Quality: {"✓ YES" if results["maintains_quality"] else "✗ NO"}')
  
  print('\nLength Robustness Curve:')
  print(f'{"Length":<10} {"PPL":<10} {"PPL (normalized)":<18}')
  print('-' * 80)
  for point in results['length_robustness_curve']:
    print(f'{point["sequence_length"]:<10} {point["perplexity"]:<10.2f} {point["ppl_normalized"]:<18.2f}')
  print('='*80 + '\n')
  
  if not results["maintains_quality"]:
    print("WARNING: Model shows quality degradation with increased length")
    print("   Consider: reducing max length, adjusting block size, or retraining\n")
  
  return results

def _ar_comparison_eval(config, logger, tokenizer):
  """
  Comprehensive comparison between BD3-LM and AR baseline.
  CRITICAL for validating that BD3-LM provides actual benefits.
  
  Usage: python main.py mode=ar_comparison_eval \
              eval.ar_checkpoint_path=<ar_model_path> \
              block_size=4 eval.num_samples=50
  """
  logger.info('Starting AR Baseline Comparison.')
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  num_samples = config.eval.get('num_samples', 50)
  batch_size = config.loader.eval_batch_size
  num_batches = (num_samples + batch_size - 1) // batch_size
  
  results = {}
  
  # === 1. Generate BD3-LM Samples ===
  logger.info('Generating BD3-LM samples...')
  
  bd3lm_model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    bd3lm_model.ema = None
  bd3lm_model.eval()
  
  bd3lm_samples = []
  bd3lm_times = []
  
  for _ in tqdm(range(num_batches), desc='BD3-LM'):
    start_time = time.time()
    batch_samples = bd3lm_model.restore_model_and_sample(
        num_steps=config.algo.T,
        seqlen=config.model.length
    )
    elapsed = time.time() - start_time
    
    if isinstance(batch_samples, list):
      bd3lm_samples.extend(batch_samples)
      samples_in_batch = len(batch_samples)
    else:
      bd3lm_samples.append(batch_samples)
      samples_in_batch = 1
    
    for _ in range(samples_in_batch):
      bd3lm_times.append(elapsed / samples_in_batch)
    
    if len(bd3lm_samples) >= num_samples:
      break
  
  bd3lm_samples = bd3lm_samples[:num_samples]
  bd3lm_times = bd3lm_times[:num_samples]
  
  # === 2. Generate AR Samples ===
  logger.info('Generating AR baseline samples...')
  
  # Load AR model
  ar_checkpoint = config.eval.get('ar_checkpoint_path', None)
  if ar_checkpoint is None:
    logger.error('AR checkpoint path not specified. Set eval.ar_checkpoint_path')
    return None
  
  config_ar = config.copy()
  config_ar.eval.checkpoint_path = ar_checkpoint
  config_ar.block_size = config.model.length  # AR is equivalent to block_size = length
  
  ar_model = _load_from_checkpoint(config=config_ar, tokenizer=tokenizer)
  if config.eval.disable_ema:
    ar_model.ema = None
  ar_model.eval()
  
  ar_samples = []
  ar_times = []
  
  for _ in tqdm(range(num_batches), desc='AR Baseline'):
    start_time = time.time()
    batch_samples = ar_model.restore_model_and_sample(
        num_steps=config.algo.T,
        seqlen=config.model.length
    )
    elapsed = time.time() - start_time
    
    if isinstance(batch_samples, list):
      ar_samples.extend(batch_samples)
      samples_in_batch = len(batch_samples)
    else:
      ar_samples.append(batch_samples)
      samples_in_batch = 1
    
    for _ in range(samples_in_batch):
      ar_times.append(elapsed / samples_in_batch)
    
    if len(ar_samples) >= num_samples:
      break
  
  ar_samples = ar_samples[:num_samples]
  ar_times = ar_times[:num_samples]
  
  # === 3. Run Comparisons ===
  
  logger.info('Running comprehensive comparison...')
  
  # Main comparison
  results['baseline_comparison'] = bd3lm_model.metrics.record_ar_baseline_comparison(
      bd3lm_samples=bd3lm_samples,
      ar_samples=ar_samples,
      bd3lm_times=bd3lm_times,
      ar_times=ar_times,
      block_size=config.block_size,
      model_length=config.model.length,
      device=device
  )
  
  # Failure mode comparison
  logger.info('Analyzing failure modes...')
  results['failure_modes'] = bd3lm_model.metrics.record_failure_mode_comparison(
      bd3lm_samples=bd3lm_samples,
      ar_samples=ar_samples,
      model_length=config.model.length,
      device=device
  )
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / f'ar_comparison_bs{config.block_size}.json'
  
  results['config'] = {
      'bd3lm_checkpoint': config.eval.checkpoint_path,
      'ar_checkpoint': ar_checkpoint,
      'block_size': config.block_size,
      'model_length': config.model.length,
      'num_samples': len(bd3lm_samples),
      'diffusion_steps': config.algo.T
  }
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  logger.info(f'Results saved to: {output_file}')
  
  # === COMPREHENSIVE REPORT ===
  
  bc = results['baseline_comparison']
  fm = results['failure_modes']
  
  print('\n' + '='*80)
  print('BD3-LM vs AUTOREGRESSIVE BASELINE COMPARISON')
  print('='*80)
  print(f'BD3-LM Block Size: {config.block_size}')
  print(f'Sequence Length: {config.model.length}')
  print(f'Samples: {len(bd3lm_samples)}')
  
  print(f'\n QUALITY COMPARISON:')
  print(f'  BD3-LM PPL: {bc["quality"]["bd3lm_ppl"]:.2f}')
  print(f'  AR PPL:     {bc["quality"]["ar_ppl"]:.2f}')
  print(f'  Ratio:      {bc["quality"]["ppl_ratio"]:.3f}x')
  print(f'  Gap:        {bc["quality"]["quality_gap_pct"]:+.1f}%')
  if bc["quality"]["quality_competitive"]:
    print(f'  Status:     ✓ Competitive (within 15%)')
  else:
    print(f'  Status:       Quality gap exceeds 15%')
  
  print(f'\n SPEED COMPARISON:')
  print(f'  BD3-LM:     {bc["speed"]["bd3lm_time_s"]:.2f}s')
  print(f'  AR:         {bc["speed"]["ar_time_s"]:.2f}s')
  print(f'  Speedup:    {bc["speed"]["speedup"]:.2f}x')
  print(f'  Time Saved: {bc["speed"]["time_saved_pct"]:.1f}%')
  if bc["speed"]["achieves_speedup"]:
    print(f'  Status:     ✓ Faster than AR')
  else:
    print(f'  Status:       Slower than AR')
  
  print(f'\n LATENCY COMPARISON (Time to First Output):')
  print(f'  BD3-LM TTFB: {bc["latency"]["bd3lm_ttfb_ms"]:.1f}ms')
  print(f'  AR TTFB:     {bc["latency"]["ar_ttfb_ms"]:.1f}ms')
  print(f'  Improvement: {bc["latency"]["latency_improvement"]:.2f}x')
  print(f'  Advantage:   {bc["latency"]["latency_advantage_ms"]:.1f}ms faster')
  if bc["latency"]["interactive_suitable"]:
    print(f'  Status:     ✓ Suitable for interactive apps')
  else:
    print(f'  Status:      May be too slow for real-time use')
  
  print(f'\n EFFICIENCY (Quality per Second):')
  print(f'  BD3-LM:     {bc["efficiency"]["bd3lm_quality_per_sec"]:.6f}')
  print(f'  AR:         {bc["efficiency"]["ar_quality_per_sec"]:.6f}')
  print(f'  Ratio:      {bc["efficiency"]["efficiency_ratio"]:.3f}x')
  if bc["efficiency"]["better_efficiency"]:
    print(f'  Status:     ✓ More efficient than AR')
  else:
    print(f'  Status:       Less efficient than AR')
  
  print(f'\n DIVERSITY COMPARISON:')
  print(f'  BD3-LM Repetition Score: {bc["diversity"]["bd3lm_repetition_score"]:.4f}')
  print(f'  AR Repetition Score:     {bc["diversity"]["ar_repetition_score"]:.4f}')
  if bc["diversity"]["bd3lm_more_diverse"]:
    print(f'  Status:     ✓ BD3-LM more diverse')
  else:
    print(f'  Status:     AR more diverse')
  
  print(f'\n  FAILURE MODE ANALYSIS:')
  print(f'  Repetition Issues:')
  print(f'    BD3-LM: {"✗ Excessive" if fm["repetition_comparison"]["bd3lm_has_excessive_repetition"] else "✓ OK"}')
  print(f'    AR:     {"✗ Excessive" if fm["repetition_comparison"]["ar_has_excessive_repetition"] else "✓ OK"}')
  
  print(f'  Coherence Issues:')
  print(f'    BD3-LM: {fm["coherence_breakdown"]["bd3lm_issue_rate"]*100:.1f}%')
  print(f'    AR:     {fm["coherence_breakdown"]["ar_issue_rate"]*100:.1f}%')
  
  print(f'  Quality Variance:')
  print(f'    BD3-LM CV: {fm["consistency_comparison"]["bd3lm_cv"]:.4f}')
  print(f'    AR CV:     {fm["consistency_comparison"]["ar_cv"]:.4f}')
  
  if fm["summary"]["introduces_new_failure_modes"]:
    print(f'\n   WARNING: BD3-LM introduces new failure modes')
  elif fm["summary"]["failure_modes_comparable"]:
    print(f'\n  ✓ Failure modes comparable to AR')
  
  print(f'\n OVERALL VERDICT:')
  print(f'  {bc["verdict"]["tradeoff_summary"]}')
  print(f'  Achieves Claimed Benefits: {"✓ YES" if bc["verdict"]["achieves_claimed_benefits"] else "✗ NO"}')
  print(f'  Recommended for Interactive: {"✓ YES" if bc["verdict"]["recommended_for_interactive"] else "✗ NO"}')
  print(f'  Pareto Efficient: {"✓ YES" if bc["verdict"]["pareto_efficient"] else "✗ NO"}')
  
  print('='*80 + '\n')
  
  # === INSIGHTS ===
  print(' INSIGHTS:')
  
  if bc["quality"]["quality_competitive"] and bc["speed"]["speedup"] > 1.5:
    print('  ✓ BD3-LM successfully achieves the core goal: similar quality with speedup')
  elif bc["quality"]["quality_gap_pct"] < 0 and bc["speed"]["speedup"] > 1:
    print('  ✓✓ BD3-LM is BETTER on both quality AND speed (Pareto improvement!)')
  elif bc["latency"]["latency_improvement"] > 2 and bc["quality"]["quality_competitive"]:
    print('  ✓ BD3-LM excels at latency—ideal for interactive applications')
  elif bc["speed"]["speedup"] > 2 and bc["quality"]["quality_gap_pct"] < 20:
    print('  BD3-LM trades some quality for substantial speed—suitable for use cases where speed matters')
  else:
    print(' Tradeoffs unclear—may need hyperparameter tuning or larger block size')
  
  if fm["summary"]["introduces_new_failure_modes"]:
    print(' Parallelization introduces failure modes—needs further investigation')
  
  print()
  
  return results


def _multi_block_size_comparison(config, logger, tokenizer):
  """
  Compare AR with multiple BD3-LM block sizes to show interpolation.
  This validates the core BD3-LM hypothesis.
  
  Usage: python main.py mode=multi_block_size_comparison \
              eval.ar_checkpoint_path=<ar_path> \
              eval.block_sizes=[1,4,8,16,32]
  """
  logger.info('Starting Multi-Block-Size Comparison with AR.')
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  num_samples = config.eval.get('num_samples', 30)
  block_sizes = config.eval.get('block_sizes', [1, 4, 8, 16])
  
  results = {'block_size_comparisons': []}
  
  # Generate AR samples once
  logger.info('Generating AR baseline samples...')
  ar_checkpoint = config.eval.get('ar_checkpoint_path')
  config_ar = config.copy()
  config_ar.eval.checkpoint_path = ar_checkpoint
  config_ar.block_size = config.model.length
  
  ar_model = _load_from_checkpoint(config=config_ar, tokenizer=tokenizer)
  ar_model.eval()
  
  ar_samples = []
  ar_times = []
  batch_size = config.loader.eval_batch_size
  num_batches = (num_samples + batch_size - 1) // batch_size
  
  for _ in tqdm(range(num_batches), desc='AR'):
    start = time.time()
    batch = ar_model.restore_model_and_sample(num_steps=config.algo.T, seqlen=config.model.length)
    elapsed = time.time() - start
    
    if isinstance(batch, list):
      ar_samples.extend(batch)
      ar_times.extend([elapsed/len(batch)] * len(batch))
    else:
      ar_samples.append(batch)
      ar_times.append(elapsed)
    
    if len(ar_samples) >= num_samples:
      break
  
  ar_samples = ar_samples[:num_samples]
  ar_times = ar_times[:num_samples]
  
  # Compare each block size
  for bs in block_sizes:
    logger.info(f'\n{"="*60}')
    logger.info(f'Evaluating Block Size = {bs}')
    logger.info(f'{"="*60}')
    
    config.block_size = bs
    checkpoint_pattern = config.eval.get('checkpoint_pattern', 
                                         'kuleshov-group/bd3lm-owt-block_size{bs}')
    config.eval.checkpoint_path = checkpoint_pattern.format(bs=bs)
    
    # Generate BD3-LM samples
    bd3lm_model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    bd3lm_model.eval()
    
    bd3lm_samples = []
    bd3lm_times = []
    
    for _ in tqdm(range(num_batches), desc=f'BS={bs}'):
      start = time.time()
      batch = bd3lm_model.restore_model_and_sample(num_steps=config.algo.T, seqlen=config.model.length)
      elapsed = time.time() - start
      
      if isinstance(batch, list):
        bd3lm_samples.extend(batch)
        bd3lm_times.extend([elapsed/len(batch)] * len(batch))
      else:
        bd3lm_samples.append(batch)
        bd3lm_times.append(elapsed)
      
      if len(bd3lm_samples) >= num_samples:
        break
    
    bd3lm_samples = bd3lm_samples[:num_samples]
    bd3lm_times = bd3lm_times[:num_samples]
    
    # Compare
    comparison = bd3lm_model.metrics.record_ar_baseline_comparison(
        bd3lm_samples=bd3lm_samples,
        ar_samples=ar_samples,
        bd3lm_times=bd3lm_times,
        ar_times=ar_times,
        block_size=bs,
        model_length=config.model.length,
        device=device
    )
    
    comparison['block_size'] = bs
    results['block_size_comparisons'].append(comparison)
  
  # Save results
  output_dir = Path('results')
  output_dir.mkdir(exist_ok=True)
  output_file = output_dir / 'multi_blocksize_ar_comparison.json'
  
  with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  # Visualization summary
  print('\n' + '='*80)
  print('INTERPOLATION ANALYSIS: BD3-LM Block Sizes vs AR')
  print('='*80)
  print(f'\n{"Block Size":<12} {"PPL":<8} {"Speedup":<10} {"TTFB (ms)":<12} {"Quality Gap %":<15}')
  print('-'*80)
  
  for comp in results['block_size_comparisons']:
    bs = comp['block_size']
    ppl = comp['quality']['bd3lm_ppl']
    speedup = comp['speed']['speedup']
    ttfb = comp['latency']['bd3lm_ttfb_ms']
    gap = comp['quality']['quality_gap_pct']
    
    print(f'{bs:<12} {ppl:<8.2f} {speedup:<10.2f} {ttfb:<12.1f} {gap:<15.1f}')
  
  # Add AR row for reference
  ar_comp = results['block_size_comparisons'][0]  # Get AR metrics from first comparison
  print('-'*80)
  print(f'{"AR (ref)":<12} {ar_comp["quality"]["ar_ppl"]:<8.2f} {"1.00":<10} '
        f'{ar_comp["latency"]["ar_ttfb_ms"]:<12.1f} {"0.0":<15}')
  
  print('='*80)
  
  # Analysis
  block_sizes_arr = np.array([c['block_size'] for c in results['block_size_comparisons']])
  speedups = np.array([c['speed']['speedup'] for c in results['block_size_comparisons']])
  ppls = np.array([c['quality']['bd3lm_ppl'] for c in results['block_size_comparisons']])
  
  print('\n💡 INTERPOLATION INSIGHTS:')
  print(f'  Quality trend: {"↓ Improves" if np.corrcoef(block_sizes_arr, ppls)[0,1] < 0 else "↑ Degrades"} with larger blocks')
  print(f'  Speed trend: {"↓ Decreases" if np.corrcoef(block_sizes_arr, speedups)[0,1] < 0 else "↑ Increases"} with larger blocks')
  print(f'  Validates interpolation hypothesis: '
        f'{" YES" if np.corrcoef(block_sizes_arr, ppls)[0,1] < 0 and np.corrcoef(block_sizes_arr, speedups)[0,1] < 0 else "  PARTIAL"}')
  print()
  
  return results

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict)
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'block_metrics_eval':
    config.wandb = None
    results = _block_metrics_eval(config, logger, tokenizer)
  elif config.mode == 'sampling_efficiency_eval':
    config.wandb = None
    results = _sampling_efficiency_eval(config, logger, tokenizer)
  elif config.mode == 'length_robustness_eval':
    config.wandb = None
    results = _length_robustness_eval(config, logger, tokenizer)
  elif config.mode == 'ar_comparison_eval':
    config.wandb = None
    results = _ar_comparison_eval(config, logger, tokenizer)
  elif config.mode == 'multi_block_size_comparison':
    config.wandb = None
    results = _multi_block_size_comparison(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()