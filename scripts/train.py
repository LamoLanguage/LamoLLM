import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datasets import load_dataset
from config.model_config import LamoLLMConfig, get_default_config, get_small_config, get_tiny_config
from model.llm import LamoLLM
from tokenizer.tokenizer import LamoTokenizer
from training.trainer import Trainer, TextDataset


def main():
    parser = argparse.ArgumentParser(description="Train LamoLLM")
    parser.add_argument("--config", type=str, default="default", choices=["default", "small", "tiny"])
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="Dataset config name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, mps, or auto")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps (0 to disable)")
    args = parser.parse_args()

    config_map = {
        "default": get_default_config(),
        "small": get_small_config(),
        "tiny": get_tiny_config(),
    }
    config = config_map[args.config]

    print(f"Initializing LamoLLM ({args.config})...")
    model = LamoLLM(config)
    tokenizer = LamoTokenizer(config.vocab_size)

    print(f"Loading dataset: {args.dataset} ({args.dataset_config})...")
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)

    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    print(f"Loaded {len(texts)} text samples")

    train_dataset = TextDataset(texts, config.max_seq_len, tokenizer)
    print(f"Created {len(train_dataset)} training examples")

    trainer = Trainer(model, config, tokenizer, device=args.device)

    if args.checkpoint:
        info = trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {info['epoch'] + 1}, step {info['global_step']}")

    print("Starting training...")
    results = trainer.train(train_dataset, num_epochs=args.epochs, save_every=args.save_every)
    print(f"Training complete! Total loss: {results['total_loss']:.4f}")

    trainer.save_checkpoint("checkpoints/lamollm_final.pt", epoch=args.epochs - 1, global_step=results['steps'])


if __name__ == "__main__":
    main()
