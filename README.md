1D CNN for Atrial Fibrillation Detection (PTB-XL subset)

This repository contains a training script `train_cnn_ecg.py` that builds a 1D CNN to detect atrial fibrillation (AF) using the PTB-XL dataset files provided under `data/records500/...`.

Quick start

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