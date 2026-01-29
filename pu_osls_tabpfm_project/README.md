# PU/OSLS TabPFN (synthetic prior)

This repo trains a lightweight TabPFN-style model on **synthetic tabular tasks**
to study **positive–unlabeled (PU)** learning and **open-set label shift (OSLS)**.
Each task is generated on the fly and some classes are removed from the *training*
portion only; those classes can still appear at test time as **unseen/outliers**.

## What’s in here

- **`prior_data.py`**  
  Synthetic task generator. For each task:
  - sample rows/features/classes
  - generate labels from a simple linear model
  - split into train/test (on original rows)
  - remove some classes from *train only*
  - pad variable-length rows across a batch

- **`model.py`**  
  TabPFN-style transformer with a **custom target encoder** for PU/OSLS:
  - training labels are embedded normally
  - test labels are replaced by a learnable **unknown** embedding (not label mean)
  - transformer attends over features and datapoints, then predicts test logits

- **`train.py`**  
  True on-the-fly training loop:
  - sample a fresh batch of synthetic tasks every step
  - pass only training labels to the model
  - evaluate on test rows
  - map removed classes to a single unseen class in the loss
  - optional periodic evaluation using `eval_pu_osls.py`

- **`eval_pu_osls.py`**  
  Evaluation on the same synthetic prior:
  - outlier detection (AUROC/AP/TPR@FPR) using configurable scores
  - seen-class accuracy (conditional on inliers)
  - overall accuracy with unseen mapped to one class

- **`train_smoke_test.ipynb`**  
  Quick smoke test notebook: short training run, final eval, and checkpoint save.

## Quick start

Install dependencies:

```bash
pip install torch numpy scikit-learn
```

Run a short training run:

```bash
python train.py
```

Or open the notebook:

```bash
jupyter notebook train_smoke_test.ipynb
```

## How evaluation works (high level)

For each synthetic task:
1. Generate full dataset.
2. Remove some classes from the **training** rows only.
3. Train-time loss treats **removed classes** as one extra **unseen** class.
4. Evaluation reports:
   - outlier detection metrics (if you want them),
   - seen-class accuracy on inliers,
   - overall accuracy with unseen mapped to one class.

## Notes

- The synthetic generator does **not** force all classes to appear in test rows,
  so some classes can be absent in the test split of a task.
- The model’s output head is `max_classes + 1` with the last index reserved for
  the unseen class.

## References

- Grinsztajn et al., *TabPFN* (ICML 2023)
- Aigul et al., *TabICL* (NeurIPS 2023)
