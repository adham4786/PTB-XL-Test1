# 1D CNN for Atrial Fibrillation Detection (PTB-XL subset)

This repository contains a training script `train_cnn_ecg.py` that builds a 1D CNN to detect atrial fibrillation (AF) using the PTB-XL dataset files provided under `data/records500/...`.

# Results of the model
Accuracy: 0.928

AUC: 0.959

Loss: 0.179

## Accuracy
What it is: fraction of examples that were classified correctly using a decision threshold (usually 0.5 for binary).
Range: [0, 1], higher is better.
Interpretation: easy-to-understand overall correctness, but insensitive to confidence and may be misleading with class imbalance.
## AUC (Area Under the ROC Curve)
What it is: measures how well the model ranks positive examples above negatives; computed from model scores over many thresholds (ROC curve).
Range: [0, 1]. 0.5 = random chance, 1.0 = perfect ranking.
Interpretation: robust to threshold choice and to some kinds of class imbalance. Higher is better; good discriminative measure.
## Loss (binary cross-entropy)
What it is: the training objective the model minimizes. For a single example with true label y∈{0,1} and predicted probability p, binary cross-entropy = −[y·log(p) + (1−y)·log(1−p)].
Range: [0, +∞), lower is better. 0 means perfect confident predictions.
Interpretation: measures how wrong and how confident predictions are. Unlike accuracy it penalizes being confidently wrong.

# Quick start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run training (defaults expect `data/ptbxl_database.csv` and `data/records500/...` present):

```bash
python train_cnn_ecg.py --csv data/ptbxl_database.csv --data-dir data --epochs 20 --batch-size 32
```

Notes
- The script reads `scp_codes` from `ptbxl_database.csv` and marks records as AF when scp keys contain `AFIB`, `AFLT`, or `AFL`.
- The loader reads HR files from `filename_hr` (e.g. `records500/00000/00001_hr`) and expects `.dat/.hea` alongside.
- Signals are bandpass filtered (0.5-40Hz), resampled/padded to `--target-samples` (default 5000), and normalized.
- The model is a small convolutional network; tune hyperparameters and augmentations as needed.

If you want the code inserted into an existing notebook, tell me which cell, and I will add a compact training cell version.

