# PU/OSLS TabPFN Project

This repository contains an example implementation for experimenting with
positive–unlabeled (PU) data and open‑set label shift (OSLS) using a
lightweight TabPFN model.

The goal of this project is to:

1. **Generate synthetic tabular datasets** that follow the TabICL prior
   distribution but with the additional functionality to randomly
   remove some classes from the training portion.  Removed classes are
   mapped to an unseen/outlier label.  The synthetic generator
   supports configurable dataset sizes, numbers of features and
   classes, and probabilistic control over how many classes are
   removed.
2. **Impute missing labels** in the test portion of each dataset
   without using the mean of the training labels.  In the PU setting
   the mean of the training labels is not informative because the
   training labels may all correspond to a single positive class.
   Instead we impute the unknown labels with half of the maximum seen
   label index.  This places the unknown labels roughly in the middle
   of the valid label range.
3. **Train a custom TabPFN model** that incorporates a modified
   target encoder respecting the PU/OSLS imputation strategy.  The
   architecture is based on the official `nanoTabPFN` but includes
   changes to the target embedding step.  A training script is
   provided that iterates over the synthetic prior and optimises the
   model using cross‑entropy loss while ignoring the unseen class.

## File overview

* `prior_data.py`: Implements the synthetic prior data generator
  (`SyntheticPriorDataset`) and a helper function (`impute_test_labels`)
  for imputing missing labels in the test portion.  The generator
  yields dictionaries containing feature tensors, label tensors and
  train/test split indices.
* `model.py`: Contains the custom TabPFN implementation.  The
  ``CustomTargetEncoder`` imputes unknown labels with a robust value
  suitable for PU/OSLS learning before embedding them.  The rest of
  the architecture mirrors the lightweight `nanoTabPFN` implementation.
  A convenience wrapper (`CustomNanoTabPFNClassifier`) exposes a
  scikit‑learn–like API.
* `train.py`: Demonstrates how to use the synthetic prior and the
  custom model to perform training.  It defines a training loop that
  iterates over a stream of synthetic datasets, performs imputation,
  computes the loss ignoring the unseen class and updates the model.
  A simple evaluation routine is provided to assess the model on
  random binary classification tasks.

## Usage

The scripts in this project are designed for research and may require
modifications depending on the experimental setup.  A basic usage
example (assuming PyTorch is available in your environment) is:

```bash
# Install dependencies (PyTorch is required but not bundled)
pip install torch numpy scikit-learn

# Run the training script
python train.py

# The script will train the model on synthetic PU/OSLS data and then
# report a simple evaluation accuracy on randomly generated binary
# classification tasks.
```

Note that PyTorch is not installed in the default environment used to
generate this repository.  To actually execute the training script you
will need to install PyTorch or run the code in an environment where
PyTorch is available.

## References

* **TabPFN** – Grinsztajn et al. *TabPFN: A Transformer That Solves
  Small Tabular Classification Datasets in a Single Forward Pass.*
  International Conference on Machine Learning (ICML) 2023.
* **TabICL** – Aigul *et al.* *TabICL: A Tabular Foundation Model for
  In‑Context Learning on Large Data.* NeurIPS 2023.

These works inspired the synthetic prior generation and the
transformer‑based architecture employed in this project.