import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from config.model_config import get_tiny_config
from model.llm import LamoLLM
from tokenizer.tokenizer import LamoTokenizer


@pytest.fixture
def tiny_config():
    return get_tiny_config()


@pytest.fixture
def tiny_model(tiny_config):
    return LamoLLM(tiny_config)


@pytest.fixture
def tokenizer():
    return LamoTokenizer()


class TestLamoLLM:
    def test_model_creation(self, tiny_model):
        assert tiny_model is not None
        assert isinstance(tiny_model, LamoLLM)

    def test_forward_pass(self, tiny_model, tiny_config):
        input_ids = torch.randint(0, 100, (2, 32))
        output = tiny_model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 32, tiny_config.vocab_size)

    def test_forward_with_labels(self, tiny_model):
        input_ids = torch.randint(0, 100, (2, 32))
        labels = torch.randint(0, 100, (2, 32))
        output = tiny_model(input_ids, labels=labels)
        assert output["loss"] is not None

    def test_generation(self, tiny_model):
        input_ids = torch.randint(0, 100, (1, 10))
        output = tiny_model.generate(input_ids, max_new_tokens=5)
        assert output.shape[1] == 15

    def test_parameter_count(self, tiny_model, tiny_config):
        total = sum(p.numel() for p in tiny_model.parameters())
        assert total > 0


class TestTokenizer:
    def test_encode_decode(self, tokenizer):
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_vocab_size(self, tokenizer):
        assert len(tokenizer) == 50257
