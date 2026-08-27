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
