# train.py
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from prior_data import PriorGeneratorConfig, generate_batch
from model import CustomNanoTabPFNModel
from eval_pu_osls import evaluate_pu_osls, EvalConfig, print_results


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model: CustomNanoTabPFNModel,
    cfg: PriorGeneratorConfig,
    *,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_steps: int,
    unseen_label: int,
    eval_cfg: EvalConfig | None = None,
    eval_interval: int = 100,
) -> Tuple[CustomNanoTabPFNModel, List[float]]:
    """
    True on-the-fly training:
      Each step:
        - generate one batch of synthetic datasets (B tasks)
        - run model + loss + update
        - discard batch
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ignore_index handles padded test rows only (unseen label is a valid class)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    rng = np.random.default_rng(cfg.seed)
    losses: List[float] = []

    for step in range(num_steps):
        batch = generate_batch(cfg, batch_size=batch_size, device=device, rng=rng)
        x = batch["x"]                    # (B,R,F)
        y_full = batch["y"]               # (B,R)
        split = batch["train_test_split_index"]  # (B,)
        removed_count = batch["removed_class_count"]  # (B,)
        num_classes = batch["num_classes"]  # (B,)
        row_mask = batch["row_mask"]  # (B,R)

        # Build label input for model: only provide y_train; encoder will pad internally.
        # Since splits vary per task, slice per task.
        y_train_list = []
        for b in range(x.shape[0]):
            split_b = int(split[b].item())
            y_train_list.append(y_full[b:b+1, :split_b])
        # Pad y_train_list to max split so we can batch; model handles per-task split anyway.
        max_split = int(split.max().item())
        y_train_padded = torch.full(
            (x.shape[0], max_split),
            fill_value=unseen_label,
            device=x.device,
            dtype=y_full.dtype,
        )
        for b, ytb in enumerate(y_train_list):
            y_train_padded[b, : ytb.shape[1]] = ytb

        logits = model((x, y_train_padded), split)  # (B, max_test, num_outputs)

        # Build targets for test rows; map removed classes to unseen_label,
        # then remap unseen_label to the last index in each task's sliced logits.
        B, R = y_full.shape
        max_test = logits.shape[1]
        targets = torch.full((B, max_test), fill_value=-100, device=x.device, dtype=torch.long)
        seen_counts = (num_classes - removed_count).to(torch.long)
        for b in range(B):
            split_b = int(split[b].item())
            seen_count = int(seen_counts[b].item())
            # Actual test rows for this task (exclude padding rows)
            row_count = int(row_mask[b].sum().item())
            y_test = y_full[b, split_b:row_count].clone()
            # Map labels >= seen_count to unseen_label
            y_test = torch.where(y_test >= seen_count, torch.tensor(unseen_label, device=x.device), y_test)
            # Remap unseen_label to the last index in the sliced logits for this task
            y_test = torch.where(y_test == unseen_label, torch.tensor(seen_count, device=x.device), y_test)
            targets[b, : y_test.shape[0]] = y_test.to(torch.long)

        # Slice logits to seen classes + unseen for each task, pad to common class dim
        max_seen = int(seen_counts.max().item())
        logits_sliced = torch.full(
            (B, max_test, max_seen + 1),
            fill_value=torch.finfo(logits.dtype).min,
            device=logits.device,
            dtype=logits.dtype,
        )
        for b in range(B):
            seen_count = int(seen_counts[b].item())
            if seen_count > 0:
                logits_sliced[b, :, :seen_count] = logits[b, :, :seen_count]
            # unseen logit is always the last index of full head
            logits_sliced[b, :, seen_count] = logits[b, :, -1]

        targets_flat = targets.reshape(-1)                 # (B*max_test,)
        logits_flat = logits_sliced.reshape(-1, logits_sliced.shape[-1])  # (B*max_test, C')

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(float(loss.detach().cpu()))

        # Progress logging
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = sum(losses[-10:]) / min(len(losses), 10)
            print(
                f"step {step+1:5d}/{num_steps} | "
                f"loss {losses[-1]:.4f} | avg10 {avg_loss:.4f} | "
                f"splits {tuple(split.tolist())} | "
                f"seen_counts {tuple(seen_counts.tolist())} | "
                f"x {tuple(x.shape)}"
            )

        if eval_interval > 0 and (step + 1) % eval_interval == 0:
            print(f"\nEval at step {step+1}...")
            res = evaluate_pu_osls(
                model,
                cfg_prior=cfg,
                unseen_label=unseen_label,
                device=device,
                eval_cfg=eval_cfg or EvalConfig(),
            )
            print_results(res)

    return model, losses

def main():
    cfg = PriorGeneratorConfig(
        max_classes=10,
        min_features=3,
        max_features=8,
        min_rows=500,
        max_rows=1000,
        min_train_fraction=0.4,
        max_train_fraction=0.8,
        remove_poisson_lambda=1.0,
        seed=0,
        label_noise=0.1,
    )

    device = get_device()
    print(f"Training on device: {device}")

    unseen_label = cfg.max_classes
    num_outputs = cfg.max_classes + 1  # include unseen_label as a reserved id

    model = CustomNanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=4,
        mlp_hidden_size=64,
        num_layers=2,
        num_outputs=num_outputs,
        unseen_label=unseen_label,
    )

    start = time.time()
    eval_cfg = EvalConfig(
        n_tasks=200,
        batch_size=8,
        seed=999,
        outlier_score="msp",
    )

    model, losses = train(
        model,
        cfg,
        batch_size=8,         # <-- dataset-level minibatch (8 tasks per step)
        lr=5e-4,
        device=device,
        num_steps=5000,
        unseen_label=unseen_label,
        eval_cfg=eval_cfg,
        eval_interval=100,
    )
    print(f"Done. time={time.time()-start:.2f}s | final loss={losses[-1]:.4f}")

    print("\nFinal eval...")
    res = evaluate_pu_osls(
        model,
        cfg_prior=cfg,
        unseen_label=unseen_label,
        device=device,
        eval_cfg=eval_cfg,
    )
    print_results(res)

if __name__ == "__main__":
    main()
