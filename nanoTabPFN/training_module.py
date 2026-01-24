# training_module.py
"""
Training utilities mirroring the official nanoTabPFN `train.py`.

Key points matching the repo:
- criterion: CrossEntropyLoss
- gradient clipping: clip_grad_norm_(..., 1.0)
- optimizer: AdamW Schedule-Free in the repo; here we try to import `schedulefree`
  and fall back to torch.optim.AdamW if unavailable.
- periodic evaluation: controlled by `steps_per_eval` + `eval_func`
- return: (trained_model, eval_history)

Default hyperparameters (repo):
  lr=4e-3, steps_per_eval=25, num_steps=2500, batch_size=32
"""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from model_module import NanoTabPFNClassifier, NanoTabPFNModel


def set_randomness_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_optimizer(model: torch.nn.Module, lr: float):
    """Use schedulefree if available; otherwise fall back to AdamW."""
    try:
        import schedulefree  # type: ignore

        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
        opt_is_schedulefree = True
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        opt_is_schedulefree = False
    return opt, opt_is_schedulefree


def train(
    model: NanoTabPFNModel,
    prior: Iterable[Dict[str, torch.Tensor]],
    lr: float = 1e-4,
    device: Optional[torch.device | str] = None,
    steps_per_eval: int = 10,
    eval_func: Optional[Callable[[NanoTabPFNClassifier], Dict[str, float]]] = None,
) -> Tuple[NanoTabPFNModel, List[Tuple[float, Dict[str, float]]]]:
    """
    Train nanoTabPFN on a prior iterator.

    Args:
      model: NanoTabPFNModel
      prior: iterator yielding dicts with keys {"x","y","train_test_split_index"}
      lr: learning rate
      device: torch device (defaults to model device)
      steps_per_eval: run eval every this many steps (if eval_func is provided)
      eval_func: callable that takes NanoTabPFNClassifier -> dict of metrics

    Returns:
      trained model, and eval_history list of (train_time_seconds, metrics_dict)
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    optimizer, opt_is_schedulefree = _make_optimizer(model, lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    if opt_is_schedulefree and hasattr(optimizer, "train"):
        optimizer.train()

    train_time = 0.0
    eval_history: List[Tuple[float, Dict[str, float]]] = []

    try:
        for step, full_data in enumerate(prior):
            step_start = time.time()
            split = int(full_data["train_test_split_index"])

            # input: x is full table; y_train are context labels
            x = full_data["x"].to(device)
            y_train = full_data["y"][:, :split].to(device)

            # targets: test labels
            targets = full_data["y"].to(device)[:, split:]

            # forward
            output = model((x, y_train), train_test_split_index=split)  # (B, R_test, K)

            # CE expects (N, K) and (N,)
            targets_flat = targets.reshape(-1).to(torch.long)
            output_flat = output.reshape(-1, output.shape[-1])

            loss = criterion(output_flat, targets_flat)
            loss.backward()
            total_loss = float(loss.detach().cpu().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_time += (time.time() - step_start)

            # periodic eval
            if (step % steps_per_eval == steps_per_eval - 1):
                if eval_func is not None:
                    model.eval()
                    if opt_is_schedulefree and hasattr(optimizer, "eval"):
                        optimizer.eval()

                    classifier = NanoTabPFNClassifier(model, device)
                    scores = eval_func(classifier)
                    eval_history.append((train_time, scores))

                    score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                    print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                    model.train()
                    if opt_is_schedulefree and hasattr(optimizer, "train"):
                        optimizer.train()
                else:
                    print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")

    except KeyboardInterrupt:
        pass

    return model, eval_history


def eval_model(
    classifier: NanoTabPFNClassifier,
    datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """
    Evaluate on a list of datasets: [(X_train, X_test, y_train, y_test), ...]

    Metrics:
      - acc
      - balanced_acc
      - roc_auc:
          * binary: roc_auc_score(y, prob[:, 1])
          * multiclass: macro-average OVR AUC

    If AUC cannot be computed for a dataset, it is skipped for AUC averaging.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    acc_sum = 0.0
    bacc_sum = 0.0
    auc_sum = 0.0
    auc_count = 0

    for X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)

        acc_sum += float(accuracy_score(y_test, pred))
        bacc_sum += float(balanced_accuracy_score(y_test, pred))

        # AUC (skip if undefined)
        try:
            if len(np.unique(y_test)) < 2:
                raise ValueError("AUC undefined: y_test has <2 classes.")

            if prob.shape[1] == 2:
                auc = float(roc_auc_score(y_test, prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_test, prob, multi_class="ovr", average="macro"))

            auc_sum += auc
            auc_count += 1
        except Exception:
            pass

    n = len(datasets)
    return {
        "roc_auc": (auc_sum / auc_count) if auc_count > 0 else float("nan"),
        "acc": acc_sum / n if n > 0 else float("nan"),
        "balanced_acc": bacc_sum / n if n > 0 else float("nan"),
    }
