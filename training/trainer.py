import json
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def get_device(device: str = 'auto') -> torch.device:
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


def _fmt_pct(value: float) -> str:
    """Format a percentage compactly: 5 -> '5%', 2.5 -> '2.5%'."""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}pct"
    return f"{value:g}pct"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _render_bar(progress: float, width: int = 20) -> str:
    filled = int(round(progress * width))
    filled = min(width, max(0, filled))
    return "#" * filled + "-" * (width - filled)


class Trainer:
    def __init__(self, model, config, tokenizer, device='auto'):
        self.device = get_device(device)
        self.is_tpu = self.device.type == 'xla'
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

        # Scheduler is created here (not in train()) so that a checkpoint can be
        # loaded BEFORE training starts and the LR schedule resumes correctly.
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            config.warmup_steps,
            config.max_steps,
            config.min_lr
        )

        self.total_loss = 0.0
        self.step_count = 0

        print(f"Using device: {self.device}")

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 1,
        checkpoint_every_pct: float = 5.0,
        log_every_pct: float = 5.0,
        checkpoint_dir: str = "checkpoints",
        keep_all_checkpoints: bool = False,
        eval_dataset: Optional[Dataset] = None,
        save_every: int = 0,
        log_file: str = None,
    ):
        """Train the model.

        Progress reporting and checkpointing are percentage-based by default:

        - A single log line is printed every ``log_every_pct`` % of the run
          (instead of one line per step).
        - A checkpoint is saved every ``checkpoint_every_pct`` % of the run.
          The rolling ``lamollm_latest.pt`` is always refreshed; pass
          ``keep_all_checkpoints=True`` to also keep each milestone file.
        - Set ``checkpoint_every_pct=0`` and ``save_every=N`` to fall back to
          classic step-based checkpointing.

        Returns a dict with loss totals and step counts.
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty - cannot start training.")

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        eval_loader = None
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)

        # ---- Plan the run -------------------------------------------------
        steps_per_epoch = len(dataloader)
        remaining_cap = max(self.config.max_steps - self.step_count, 0)
        total_planned = min(steps_per_epoch * num_epochs, remaining_cap)
        if total_planned <= 0:
            print("Nothing to train: max_steps already reached.")
            return {"total_loss": self.total_loss, "steps": self.step_count}

        pct_interval = float(checkpoint_every_pct) if checkpoint_every_pct else 0.0
        log_interval = float(log_every_pct) if log_every_pct else 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = log_file or os.path.join(checkpoint_dir, "train_log.jsonl")

        start_step = self.step_count
        end_step = start_step + total_planned

        n_ckpts = math.ceil(100.0 / pct_interval) if pct_interval > 0 else 0
        plan_line = (
            f"\nTraining plan: {total_planned:,} steps "
            f"(step {start_step:,} -> {end_step:,}) | batch size {self.config.batch_size}"
        )
        if pct_interval > 0:
            plan_line += f" | checkpoints every {pct_interval:g}% ({n_ckpts} saves)"
        else:
            plan_line += f" | checkpoints every {save_every} steps"
        if log_interval > 0:
            plan_line += f" | logging every {log_interval:g}%"
        print(plan_line)

        self.model.train()
        run_start = time.time()
        window_start = run_start
        window_tokens = 0
        window_steps = 0
        next_ckpt_pct = pct_interval
        next_log_pct = log_interval
        EPS = 1e-9

        stop_reason = "completed"

        for epoch in range(num_epochs):
            for batch in dataloader:
                if self.step_count >= end_step:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.use_amp:
                    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                elif self.is_tpu:
                    ctx = torch.amp.autocast('cpu', dtype=torch.bfloat16)
                else:
                    ctx = torch.amp.autocast('cpu', enabled=False)

                with ctx:
                    result = self.model(input_ids, labels=labels)
                    loss = result["loss"] / self.config.gradient_accumulation_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                elif self.is_tpu:
                    loss.backward()
                    xm.mark_step()
                else:
                    loss.backward()

                if (self.step_count + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.is_tpu:
                        xm.optimizer_step(self.optimizer)
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.total_loss += loss.item() * self.config.gradient_accumulation_steps
                self.step_count += 1
                window_tokens += input_ids.numel()
                window_steps += 1

                done = self.step_count - start_step
                pct = 100.0 * done / total_planned

                # ---- Classic step-based checkpoints (legacy mode) --------
                if pct_interval == 0 and save_every > 0 and self.step_count % save_every == 0:
                    path = os.path.join(checkpoint_dir, f"lamollm_step_{self.step_count}.pt")
                    self.save_checkpoint(path, global_step=self.step_count, verbose=False)
                    print(f"[{pct:5.1f}%] checkpoint saved -> {path}")

                # ---- Fire milestones (percentage-based) ------------------
                # A single console line is emitted per ~5% milestone instead
                # of spamming progress output after every step.
                fire_ckpt = False
                ckpt_marker = None
                while pct_interval > 0 and pct + EPS >= next_ckpt_pct and next_ckpt_pct <= 100 + EPS:
                    fire_ckpt = True
                    ckpt_marker = min(next_ckpt_pct, 100.0)
                    next_ckpt_pct += pct_interval

                fire_log = False
                log_marker = None
                while log_interval > 0 and pct + EPS >= next_log_pct and next_log_pct <= 100 + EPS:
                    fire_log = True
                    log_marker = min(next_log_pct, 100.0)
                    next_log_pct += log_interval

                if fire_log or fire_ckpt:
                    marker = max(m for m in (log_marker, ckpt_marker) if m is not None)
                    avg_loss = self.total_loss / self.step_count
                    ppl = math.exp(min(avg_loss, 20.0))
                    lr = self.optimizer.param_groups[0]['lr']

                    now = time.time()
                    window_dt = max(now - window_start, 1e-6)
                    tok_s = window_tokens / window_dt
                    steps_s = window_steps / window_dt
                    eta = (total_planned - done) / max(steps_s, 1e-6)

                    # Evaluate once per milestone (when an eval set is given).
                    val_loss = None
                    if eval_loader is not None:
                        val_loss = self.evaluate(eval_loader)

                    segs = []
                    if fire_log:
                        segs.append(f"step {self.step_count}/{end_step}")
                        segs.append(f"loss {avg_loss:.4f}")
                        segs.append(f"ppl {ppl:.2f}")
                        segs.append(f"lr {lr:.2e}")
                        segs.append(f"{tok_s:,.0f} tok/s")
                        segs.append(f"ETA {_fmt_duration(eta)}")

                    saved_to = None
                    if fire_ckpt:
                        latest_path = os.path.join(checkpoint_dir, "lamollm_latest.pt")
                        self.save_checkpoint(latest_path, global_step=self.step_count, verbose=False)
                        if keep_all_checkpoints:
                            name = f"lamollm_{_fmt_pct(ckpt_marker)}_step_{self.step_count}.pt"
                            milestone_path = os.path.join(checkpoint_dir, name)
                            self.save_checkpoint(milestone_path, global_step=self.step_count, verbose=False)
                            saved_to = milestone_path
                        else:
                            saved_to = latest_path
                        segs.append(f"ckpt -> {os.path.basename(saved_to)}")

                    if val_loss is not None:
                        segs.append(f"val_loss {val_loss:.4f}")

                    bar = _render_bar(marker / 100.0)
                    print(f"[{marker:5.1f}%|{bar}] " + " | ".join(segs))

                    record = {
                        "percent": round(marker, 2),
                        "actual_percent": round(min(pct, 100.0), 2),
                        "step": self.step_count,
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 6),
                        "val_loss": round(val_loss, 6) if val_loss is not None else None,
                        "lr": lr,
                        "tokens_per_sec": round(tok_s, 1),
                        "elapsed_sec": round(now - run_start, 1),
                        "checkpoint": os.path.basename(saved_to) if saved_to else None,
                    }
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")
                    except OSError:
                        pass

                    window_start = time.time()
                    window_tokens = 0
                    window_steps = 0

                if self.step_count >= end_step:
                    stop_reason = "max_steps" if self.config.max_steps <= end_step else "completed"
                    break

            if self.step_count >= end_step:
                break

        # ---- Final wrap-up -------------------------------------------------
        elapsed = time.time() - run_start
        final_avg_loss = self.total_loss / self.step_count
        final_ppl = math.exp(min(final_avg_loss, 20.0))

        final_val = None
        if eval_loader is not None:
            final_val = self.evaluate(eval_loader)

        final_path = os.path.join(checkpoint_dir, "lamollm_final.pt")
        self.save_checkpoint(final_path, global_step=self.step_count, verbose=False)
        val_txt = f" | val_loss {final_val:.4f}" if final_val is not None else ""
        print(
            f"\nTraining finished ({stop_reason}) in {_fmt_duration(elapsed)} | "
            f"avg loss {final_avg_loss:.4f} | ppl {final_ppl:.2f}{val_txt}\n"
            f"Final checkpoint -> {final_path}"
        )

        return {
            "total_loss": self.total_loss,
            "avg_loss": final_avg_loss,
            "val_loss": final_val,
            "steps": self.step_count,
        }

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, max_batches: Optional[int] = None) -> float:
        """Average loss over the evaluation set."""
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(eval_loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
            elif self.is_tpu:
                ctx = torch.amp.autocast('cpu', dtype=torch.bfloat16)
            else:
                ctx = torch.amp.autocast('cpu', enabled=False)
            with ctx:
                result = self.model(input_ids, labels=labels)
            total += result["loss"].item()
            count += 1
        if was_training:
            self.model.train()
        return total / max(count, 1)

    def save_checkpoint(self, path: str, epoch: int = 0, global_step: int = 0, verbose: bool = True):
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
        if self.is_tpu:
            state['model_state_dict'] = {k: v.cpu() for k, v in state['model_state_dict'].items()}
            state['optimizer_state_dict'] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                               for k, v in state['optimizer_state_dict'].items()}
        if self.scheduler is not None:
            state['scheduler_step_count'] = self.scheduler.step_count
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(state, path)
        if verbose:
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        if self.is_tpu:
            optimizer_state = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in optimizer_state.items()}
        self.optimizer.load_state_dict(optimizer_state)
        self.total_loss = checkpoint.get('total_loss', 0.0)
        self.step_count = checkpoint.get('step_count', checkpoint.get('global_step', 0))
        if self.scheduler is not None and 'scheduler_step_count' in checkpoint:
            self.scheduler.step_count = checkpoint['scheduler_step_count']
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        print(f"Checkpoint loaded from {path} (epoch {epoch}, step {global_step}, avg loss {self.total_loss / max(self.step_count, 1):.4f})")
        return {'epoch': epoch, 'global_step': global_step}
