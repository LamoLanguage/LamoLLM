from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from .transformer import TransformerBlock, precompute_freqs_cis, RMSNorm
from config.model_config import LamoLLMConfig


class LamoLLM(nn.Module):
    def __init__(self, config: LamoLLMConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output.weight = self.token_embedding.weight

        self.freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len * 2,
            theta=config.rope_theta
        )

        self.apply(self._init_weights)
        self._count_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Model size: {total * 4 / (1024**3):.2f} GB (FP32)")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        bsz, seq_len = input_ids.shape

        h = self.token_embedding(input_ids)
        h = self.dropout(h)

        freqs_cis = self.freqs_cis[:seq_len].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token: Optional[int] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        generated = 0
        stop_reason = "stop_token"
        bar = tqdm(total=max_new_tokens, desc="Generating", disable=not show_progress)
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            result = self(idx_cond)
            logits = result["logits"][:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if stop_token is not None and idx_next.item() == stop_token:
                stop_reason = "stop_token"
                break

            input_ids = torch.cat([input_ids, idx_next], dim=1)
            generated += 1
            bar.update(1)

        bar.close()
        if show_progress and generated < max_new_tokens:
            print(f"\nGenerated {generated} tokens ({stop_reason})")
        return input_ids
