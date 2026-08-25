import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_device(device: str = 'auto') -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TextDataset(Dataset):
    def __init__(self, data: list, seq_len: int, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.examples = []

        all_tokens = []
        for text in data:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        for i in range(0, len(all_tokens) - seq_len - 1, seq_len):
            chunk = all_tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        return {
            "input_ids": example[:-1],
            "labels": example[1:]
        }


class CosineScheduleWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr_scale = self._get_lr_scale()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = max(base_lr * lr_scale, self.min_lr)

    def _get_lr_scale(self) -> float:
        if self.step_count < self.warmup_steps:
            return self.step_count / max(1, self.warmup_steps)
        progress = (self.step_count - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model, config, tokenizer, device='auto'):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.config = config
        self.tokenizer = tokenizer
        self.use_amp = self.device.type == 'cuda'

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scheduler = None
        self.total_loss = 0.0
        self.step_count = 0

        print(f"Using device: {self.device}")

    def train(self, dataset: TextDataset, num_epochs: int = 1, save_every: int = 1000):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            self.config.warmup_steps,
            self.config.max_steps,
            self.config.min_lr
        )

        self.model.train()
        global_step = self.step_count
        epoch_bar = tqdm(range(num_epochs), desc="Epochs", position=0)
        batch_bar = tqdm(dataloader, desc="Training", position=1, leave=False)

        for epoch in epoch_bar:
            epoch_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for batch in batch_bar:
                if global_step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.use_amp else torch.cpu.amp.autocast(enabled=False)
                with ctx:
                    result = self.model(input_ids, labels=labels)
                    loss = result["loss"] / self.config.gradient_accumulation_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.total_loss += loss.item() * self.config.gradient_accumulation_steps
                self.step_count += 1

                avg_loss = self.total_loss / self.step_count
                lr = self.optimizer.param_groups[0]['lr']
                batch_bar.set_postfix(step=global_step, loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if save_every > 0 and global_step % save_every == 0 and global_step > 0:
                    self.save_checkpoint(f"checkpoints/lamollm_step_{global_step}.pt", epoch=epoch, global_step=global_step)

                global_step += 1

            batch_bar.reset()

        epoch_bar.close()
        batch_bar.close()

        print(f"\nTraining complete! Final avg loss: {self.total_loss / self.step_count:.4f}")
        return {"total_loss": self.total_loss, "steps": self.step_count}

    def save_checkpoint(self, path: str, epoch: int = 0, global_step: int = 0):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'global_step': global_step,
            'total_loss': self.total_loss,
            'step_count': self.step_count,
        }
        if self.scheduler is not None:
            state['scheduler_step_count'] = self.scheduler.step_count
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_loss = checkpoint.get('total_loss', 0.0)
        self.step_count = checkpoint.get('step_count', 0)
        if self.scheduler is not None and 'scheduler_step_count' in checkpoint:
            self.scheduler.step_count = checkpoint['scheduler_step_count']
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        print(f"Checkpoint loaded from {path} (epoch {epoch}, step {global_step}, loss {self.total_loss:.4f})")
        return {'epoch': epoch, 'global_step': global_step}
