from typing import Optional, List
import tiktoken


class LamoTokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self._encoder = tiktoken.get_encoding("gpt2")
        self.eot_token = self._encoder.eot_token

    @property
    def eot(self) -> int:
        return self.eot_token

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._encoder.encode(text)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self._encoder.decode(tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
