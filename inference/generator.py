from typing import Optional, List
import torch

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from model.llm import LamoLLM
from tokenizer.tokenizer import LamoTokenizer
from config.model_config import LamoLLMConfig


def _resolve_device(device: str) -> torch.device:
    if device != 'auto':
        if device == 'tpu' and HAS_XLA:
            return xm.xla_device()
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if HAS_XLA:
        return xm.xla_device()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class LamoGenerator:
    def __init__(self, model: LamoLLM, tokenizer: LamoTokenizer, device: str = 'auto'):
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'auto'):
        resolved = _resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=resolved)
        config = checkpoint['config']
        model = LamoLLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = LamoTokenizer(config.vocab_size)
        return cls(model, tokenizer, str(resolved))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token: Optional[str] = None,
        echo: bool = False
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stop_id = None
        if stop_token:
            stop_id = self.tokenizer.encode(stop_token)[0]

        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token=stop_id
        )

        generated_ids = output_ids[0, len(input_ids):].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        if echo:
            return prompt + generated_text

        return generated_text

    def chat(self, message: str, history: List[dict] = None, **kwargs) -> str:
        if history is None:
            history = []

        prompt = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += f"User: {message}\nAssistant:"

        response = self.generate(prompt, **kwargs)
        return response.strip()
