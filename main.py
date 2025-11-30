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
import tqdm
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
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()