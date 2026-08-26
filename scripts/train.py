import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import argparse

import numpy as np
import torch
from config.model_config import LamoLLMConfig, get_default_config, get_small_config, get_tiny_config
from model.llm import LamoLLM
from tokenizer.tokenizer import LamoTokenizer
from training.trainer import Trainer, TextDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train LamoLLM")
    parser.add_argument("--config", type=str, default="default", choices=["default", "small", "tiny"])
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="Dataset config name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, mps, or auto")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where checkpoints are written")
    parser.add_argument("--auto_resume", action="store_true",
                        help="If --checkpoint is not given and <checkpoint_dir>/lamollm_latest.pt exists, resume from it automatically (crash recovery)")

    # Percentage-based checkpointing / logging (default behavior)
    parser.add_argument("--checkpoint_every_pct", type=float, default=5.0,
                        help="Save a checkpoint every N%% of training progress (default 5). 0 disables.")
    parser.add_argument("--log_every_pct", type=float, default=5.0,
                        help="Print one training log line every N%% of progress (default 5). 0 disables.")
    parser.add_argument("--keep_all_checkpoints", action="store_true",
                        help="Keep every milestone checkpoint file (otherwise only lamollm_latest.pt is refreshed each time; lamollm_final.pt is always saved at the end)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Legacy step-based checkpointing (ignored unless --checkpoint_every_pct 0)")

    # Evaluation
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable the periodic validation-loss check (validation split of the dataset by default)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    config_map = {
        "default": get_default_config(),
        "small": get_small_config(),
        "tiny": get_tiny_config(),
    }
    config = config_map[args.config]

    # Imported lazily so `--help` works without the datasets package installed.
    from datasets import load_dataset

    print(f"Initializing LamoLLM ({args.config})...")
    model = LamoLLM(config)
    tokenizer = LamoTokenizer(config.vocab_size)

    print(f"Loading dataset: {args.dataset} ({args.dataset_config})...")
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)

    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    print(f"Loaded {len(texts)} text samples")

    train_dataset = TextDataset(texts, config.max_seq_len, tokenizer)
    print(f"Created {len(train_dataset)} training examples")

    eval_dataset = None
    if not args.no_eval:
        try:
            eval_split = load_dataset(args.dataset, name=args.dataset_config, split="validation")
            eval_texts = [item['text'] for item in eval_split if len(item['text']) > 100]
            eval_dataset = TextDataset(eval_texts, config.max_seq_len, tokenizer)
            print(f"Created {len(eval_dataset)} validation examples")
        except Exception as e:
            print(f"No validation split available ({type(e).__name__}); skipping eval loss.")

    trainer = Trainer(model, config, tokenizer, device=args.device)

    if args.checkpoint:
        info = trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {info['epoch'] + 1}, step {info['global_step']}")
    elif args.auto_resume:
        latest = os.path.join(args.checkpoint_dir, "lamollm_latest.pt")
        if os.path.exists(latest):
            print(f"Auto-resuming from last checkpoint: {latest}")
            info = trainer.load_checkpoint(latest)
            if trainer.step_count >= config.max_steps:
                print("Last checkpoint already reached max_steps - nothing to train.")
                return
        else:
            print("No previous checkpoint found - starting fresh.")

    print("Starting training...")
    results = trainer.train(
        train_dataset,
        num_epochs=args.epochs,
        checkpoint_every_pct=args.checkpoint_every_pct,
        log_every_pct=args.log_every_pct,
        checkpoint_dir=args.checkpoint_dir,
        keep_all_checkpoints=args.keep_all_checkpoints,
        eval_dataset=eval_dataset,
        save_every=args.save_every,
    )
    print(f"Training complete! Avg loss: {results['avg_loss']:.4f} | steps: {results['steps']}")


if __name__ == "__main__":
    main()
