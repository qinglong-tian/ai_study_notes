# training_module.py (workshop version: simple + readable)
from dataclasses import dataclass
import torch


# ----------------------------
# Curriculum (Section A.2)
# ----------------------------
class CurriculumScheduler:
    """
    Linear / sparse-linear curriculum (Section A.2):

        start: d_cur = 5,  k_cur = 11
        every 2000 steps:  d_cur += 1,  k_cur += 2
        end:   d_cur = d,  k_cur = 2d + 1

    We return (d_cur, k_cur) for the current training step.
    """

    def __init__(self, d, step_every=2000):
        self.d = d
        self.step_every = step_every

    def get(self, step):
        inc = step // self.step_every
        d_cur = min(self.d, 5 + inc)
        k_cur = min(2 * self.d + 1, 11 + 2 * inc)
        return d_cur, k_cur


# ----------------------------
# Training config
# ----------------------------
@dataclass
class TrainConfig:
    task: str = "linear"          # "linear" or "sparse_linear"
    total_steps: int = 500_000
    batch_size: int = 64
    lr: float = 3e-4
    grad_clip: float = 1.0
    log_every: int = 200
    device: str = "cuda"


# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    """
    Simple trainer for curriculum learning.

    Assumptions:
      - data_gen.sample_batch(task, batch_size, k, effective_dim, sparse_s) returns:
            batch.tokens:  (B, 2k+1, d)
            batch.targets: (B, k+1)
      - model(tokens) returns:
            preds: (B, k+1)
    """

    def __init__(self, model, data_gen, cfg: TrainConfig, sparse_s=3):
        self.model = model.to(cfg.device)
        self.data_gen = data_gen
        self.cfg = cfg
        self.sparse_s = sparse_s

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.curriculum = CurriculumScheduler(d=data_gen.d, step_every=2000)

    def train(self):
        self.model.train()

        for step in range(self.cfg.total_steps):
            d_cur, k_cur = self.curriculum.get(step)

            # 1) generate a fresh batch
            batch = self.data_gen.sample_batch(
                task=self.cfg.task,
                batch_size=self.cfg.batch_size,
                k=k_cur,
                effective_dim=d_cur,
                sparse_s=self.sparse_s,
            )
            tokens = batch.tokens.to(self.cfg.device)
            targets = batch.targets.to(self.cfg.device)

            # 2) forward + loss
            preds = self.model(tokens)                       # (B, k+1)
            loss = (preds - targets).pow(2).mean()           # MSE over batch and positions

            # 3) backward + update
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

            # 4) print progress
            if (step + 1) % self.cfg.log_every == 0:
                print(
                    f"step {step+1:>7} | "
                    f"d_cur={d_cur:>2} k_cur={k_cur:>3} | "
                    f"loss={loss.item():.4e}"
                )
