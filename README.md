# LamoLLM

A 1B parameter decoder-only transformer language model built from scratch with PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LamoDev/LamoLLM/blob/main/LamoLLM_Colab.ipynb)

## Architecture

- **Model Type**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~1.1B
- **Hidden Size**: 2048
- **Layers**: 24
- **Attention Heads**: 32
- **Max Sequence Length**: 2048
- **Vocabulary Size**: 50,257

### Features

- Pre-norm Transformer architecture
- Rotary Position Embeddings (RoPE)
- Flash Attention support
- Mixed precision training (BF16)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- SwiGLU activation in FFN

## Installation

```bash
pip install -r requirements.txt
```

### Google Colab

Open the notebook directly in Colab using the badge above, or manually:

1. Upload `LamoLLM_Colab.ipynb` to Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells

## Usage

### Training

```bash
# Train with default config (1B params)
python scripts/train.py --config default --epochs 1

# Train with small config (125M params)
python scripts/train.py --config small --epochs 1

# Train with tiny config (for testing)
python scripts/train.py --config tiny --epochs 1
```

#### Progress logging & checkpoints (percentage-based)

Training output is quiet by design: instead of one line per step, you get a
**single log line every 5% of progress**, and a **checkpoint is saved every 5%**
of progress. Example of what you will see:

```
Training plan: 12,200 steps (step 0 -> 12,200) | batch size 8 | checkpoints every 5% (20 saves) | logging every 5%
[ 15.0%|###-----------------] step 1830/12200 | loss 6.4212 | ppl 616.28 | lr 3.00e-04 | 31,240 tok/s | ETA 00:04:52 | ckpt -> lamollm_latest.pt
[ 20.0%|####---------------] step 2440/12200 | loss 6.0138 | ppl 409.42 | lr 3.00e-04 | 31,105 tok/s | ETA 00:04:34 | val_loss 5.9872 | ckpt -> lamollm_latest.pt
...
Final checkpoint -> checkpoints/lamollm_final.pt
```

Options:

```bash
python scripts/train.py --config tiny \
    --checkpoint_every_pct 5 \      # checkpoint cadence in % of run (default 5)
    --log_every_pct 5 \             # console log cadence in % (default 5)
    --keep_all_checkpoints          # keep every milestone file, not just the latest
```

- Every milestone refreshes `checkpoints/lamollm_latest.pt` (safe to resume from).
- With `--keep_all_checkpoints`, each milestone is also kept as
  `lamollm_<pct>_step_<N>.pt` (e.g. `lamollm_25pct_step_3050.pt`).
- The end of training always writes `checkpoints/lamollm_final.pt`.
- Metrics for every milestone are appended to `checkpoints/train_log.jsonl`
  (loss, ppl, lr, tokens/sec, eval loss) - handy for plotting afterwards.
- Validation loss is computed on the dataset's `validation` split at each
  milestone; pass `--no_eval` to disable.
- Classic step-based saving still works if you need it:
  `--checkpoint_every_pct 0 --save_every 1000`.

#### Crash recovery (Google Colab)

Checkpoint writes are atomic (saved to a `.tmp` file first, then renamed), so a
crash mid-save never corrupts the previous checkpoint. To survive Colab
disconnects entirely:

```python
# Cell 1 - mount Drive so checkpoints outlive the runtime
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell 2 - train into Drive, resuming automatically if a checkpoint exists
!python scripts/train.py --config tiny --epochs 1 \
    --checkpoint_dir /content/drive/MyDrive/LamoLLM/checkpoints \
    --auto_resume
```

If the runtime crashes, reconnect and re-run the same cells:
`--auto_resume` detects `lamollm_latest.pt` in the checkpoint dir and continues
from that step (model + optimizer + LR schedule all restored). Because
checkpoints fire every 5%, a crash costs you at most ~5% of progress.

### Generation

```bash
# Single prompt
python scripts/generate.py --checkpoint checkpoints/lamollm_final.pt --prompt "Hello, I am"

# Interactive chat
python scripts/generate.py --checkpoint checkpoints/lamollm_final.pt --interactive
```

### Python API

```python
from inference.generator import LamoGenerator

generator = LamoGenerator.from_checkpoint("checkpoints/lamollm_final.pt")

# Generate text
output = generator.generate("The future of AI is")

# Chat
response = generator.chat("What is machine learning?")
```

## Project Structure

```
LamoLLM/
├── config/
│   └── model_config.py      # Model configuration
├── model/
│   ├── transformer.py       # Transformer components
│   └── llm.py               # Main LamoLLM class
├── tokenizer/
│   └── tokenizer.py         # BPE tokenizer
├── training/
│   └── trainer.py           # Training loop
├── inference/
│   └── generator.py         # Text generation
├── scripts/
│   ├── train.py             # Training script
│   └── generate.py          # Generation script
└── tests/
    └── test_model.py        # Unit tests
```

## Model Variants

| Config | Params | Layers | Hidden | Heads | FFN |
|--------|--------|--------|--------|-------|-----|
| Default | 1.1B | 24 | 2048 | 32 | 8192 |
| Small | 125M | 12 | 768 | 12 | 3072 |
| Tiny | 25M | 6 | 384 | 6 | 1536 |

## License

MIT
