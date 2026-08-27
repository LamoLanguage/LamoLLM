from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LamoLLMConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    d_model: int = 2048
    d_ff: int = 8192
    dropout: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    norm_eps: float = 1e-5
    flash_attention: bool = True
    dtype: str = "bfloat16"
    # Tying input/output embeddings saves ~vocab_size * d_model params (e.g. ~103M
    # for the default config) and the matching AdamW optimizer state for them.
    tie_word_embeddings: bool = True

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 600000
    lr_scheduler: str = "cosine"
    min_lr: float = 3e-5
    # Use bitsandbytes 8-bit AdamW to roughly halve optimizer state memory.
    # Falls back to regular AdamW automatically if bitsandbytes isn't installed.
    use_8bit_optimizer: bool = False

    # Tokenizer
    tokenizer_type: str = "gpt2"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads

    def __post_init__(self):
        self.d_ff = self.d_ff or 4 * self.d_model


def get_default_config() -> LamoLLMConfig:
    return LamoLLMConfig()


def get_small_config() -> LamoLLMConfig:
    return LamoLLMConfig(
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        max_seq_len=1024,
    )


def get_tiny_config() -> LamoLLMConfig:
    return LamoLLMConfig(
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        max_seq_len=512,
        batch_size=16,
    )
