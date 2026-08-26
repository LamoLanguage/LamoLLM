import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob

import pytest
import torch
from torch.utils.data import Dataset

from config.model_config import LamoLLMConfig
from model.llm import LamoLLM
from training.trainer import Trainer


class DummyTokenizer:
    """Minimal tokenizer stand-in so tests do not need tiktoken."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.eot_token = 0

    def encode(self, text):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens):
        return "".join(chr(int(t)) for t in tokens)

    def __len__(self):
        return self.vocab_size


class RandomTextDataset(Dataset):
    def __init__(self, num_examples, seq_len, vocab_size):
        g = torch.Generator().manual_seed(7)
        data = torch.randint(0, vocab_size, (num_examples, seq_len + 1), generator=g)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {"input_ids": ex[:-1], "labels": ex[1:]}


def tiny_run_config(**overrides):
    cfg_kwargs = dict(
        vocab_size=256,
        max_seq_len=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=128,
        batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=10_000,
        learning_rate=1e-3,
    )
    cfg_kwargs.update(overrides)
    return LamoLLMConfig(**cfg_kwargs)


def make_trainer(tmp_path, **config_overrides):
    config = tiny_run_config(**config_overrides)
    model = LamoLLM(config)
    tokenizer = DummyTokenizer(config.vocab_size)
    ckpt_dir = str(tmp_path / "checkpoints")
    trainer = Trainer(model, config, tokenizer, device="cpu")
    return trainer, ckpt_dir


class TestPercentageMilestones:
    def test_log_and_ckpt_every_25pct(self, tmp_path, capsys):
        trainer, ckpt_dir = make_trainer(tmp_path)
        dataset = RandomTextDataset(160, 16, 256)  # 160 examples / batch_size 4 -> exactly 40 steps

        trainer.train(
            dataset,
            num_epochs=1,
            checkpoint_every_pct=25.0,
            log_every_pct=25.0,
            checkpoint_dir=ckpt_dir,
            keep_all_checkpoints=True,
            eval_dataset=RandomTextDataset(4, 16, 256),
        )

        out = capsys.readouterr().out
        milestone_lines = [ln for ln in out.splitlines() if "tok/s" in ln]
        # one consolidated line per milestone => 4 lines (25/50/75/100)
        assert len(milestone_lines) == 4
        assert any("[100.0%" in ln for ln in milestone_lines)

        # rolling latest + final always exist
        assert os.path.exists(os.path.join(ckpt_dir, "lamollm_latest.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "lamollm_final.pt"))

        # all four milestone files kept
        milestones = sorted(glob.glob(os.path.join(ckpt_dir, "lamollm_*pct_step_*.pt")))
        assert len(milestones) == 4

        # jsonl log written: 4 milestone records with val loss recorded
        with open(os.path.join(ckpt_dir, "train_log.jsonl"), encoding="utf-8") as f:
            records = [json.loads(ln) for ln in f if ln.strip()]
        assert len(records) == 4
        assert [r["percent"] for r in records] == [25.0, 50.0, 75.0, 100.0]
        assert all(r["val_loss"] is not None for r in records)

    def test_only_every_5pct_prints_not_every_step(self, tmp_path, capsys):
        trainer, ckpt_dir = make_trainer(tmp_path)
        dataset = RandomTextDataset(200, 16, 256)  # 200 examples / batch_size 4 -> 50 steps

        trainer.train(
            dataset,
            num_epochs=1,
            checkpoint_every_pct=5.0,
            log_every_pct=5.0,
            checkpoint_dir=ckpt_dir,
        )

        out = capsys.readouterr().out
        progress_lines = [ln for ln in out.splitlines() if "tok/s" in ln]
        # 50-step run at 5% cadence -> exactly 20 milestone lines (not 50)
        assert len(progress_lines) == 20

    def test_quiet_when_big_run(self, tmp_path, capsys):
        trainer, ckpt_dir = make_trainer(tmp_path, batch_size=1)
        dataset = RandomTextDataset(100, 16, 256)  # 100 steps at batch 1

        trainer.train(
            dataset,
            num_epochs=1,
            checkpoint_every_pct=5.0,
            log_every_pct=5.0,
            checkpoint_dir=ckpt_dir,
        )

        out = capsys.readouterr().out
        progress_lines = [ln for ln in out.splitlines() if "tok/s" in ln]
        assert len(progress_lines) == 20  # 100/5 -> every 5% -> 20 lines exactly
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        # latest refreshed each time + final; without keep_all there are exactly 2 files left
        assert {os.path.basename(p) for p in ckpts} == {"lamollm_latest.pt", "lamollm_final.pt"}

    def test_resume_preserves_schedule_and_counts(self, tmp_path):
        trainer, ckpt_dir = make_trainer(tmp_path)
        dataset = RandomTextDataset(48, 16, 256)  # 48 examples / batch_size 4 -> 12 steps

        res1 = trainer.train(dataset, num_epochs=1, checkpoint_every_pct=50.0,
                             log_every_pct=50.0, checkpoint_dir=ckpt_dir)
        assert res1["steps"] == 12

        # resume into a NEW trainer from latest checkpoint
        trainer2, _ = make_trainer(tmp_path)
        info = trainer2.load_checkpoint(os.path.join(ckpt_dir, "lamollm_latest.pt"))
        assert info["global_step"] == 12
        # scheduler state restored so warmup/cosine continues where it stopped
        assert trainer2.scheduler.step_count > 0

        res2 = trainer2.train(dataset, num_epochs=1, checkpoint_every_pct=50.0,
                              log_every_pct=50.0, checkpoint_dir=ckpt_dir)
        assert res2["steps"] == 24
        assert trainer2.scheduler.step_count > 12

    def test_legacy_save_every_mode(self, tmp_path):
        trainer, ckpt_dir = make_trainer(tmp_path)
        dataset = RandomTextDataset(32, 16, 256)  # 32 examples / batch_size 4 -> 8 steps

        trainer.train(dataset, num_epochs=1, checkpoint_every_pct=0.0,
                      log_every_pct=5.0, checkpoint_dir=ckpt_dir, save_every=4)

        step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "lamollm_step_*.pt")))
        assert [os.path.basename(p) for p in step_ckpts] == [
            "lamollm_step_4.pt", "lamollm_step_8.pt"
        ]
        assert os.path.exists(os.path.join(ckpt_dir, "lamollm_final.pt"))

    def test_max_steps_cutoff(self, tmp_path):
        trainer, ckpt_dir = make_trainer(tmp_path, max_steps=6)
        dataset = RandomTextDataset(50, 16, 256)  # more than allowed

        res = trainer.train(dataset, num_epochs=1, checkpoint_every_pct=50.0,
                            log_every_pct=50.0, checkpoint_dir=ckpt_dir)
        assert res["steps"] == 6
