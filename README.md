# Predictive Anomaly Detection on NASA Turbofan Engine Data
### Comparative Study of LSTM Autoencoder vs Self-Supervised Masked Autoencoder

## Overview

This project investigates predictive anomaly detection for aircraft engine health monitoring using the NASA C-MAPSS turbofan engine dataset (FD001 subset).

The objective was to compare a traditional sequence-aware deep learning baseline (**LSTM Autoencoder**) against a self-supervised learning alternative (**Masked Autoencoder**) for anomaly detection in multivariate time-series sensor data.

The study evaluates both architectures under identical preprocessing, training, and evaluation conditions using precision-recall focused metrics suitable for imbalanced anomaly detection tasks.

---

## Problem Statement

Unexpected aircraft engine failures can lead to major operational disruptions, safety risks, and high maintenance costs.

Predictive maintenance aims to detect abnormal degradation patterns before catastrophic failure occurs.

This project explores whether a self-supervised masked reconstruction approach can outperform a traditional recurrent autoencoder for predictive anomaly detection in aerospace sensor data.

---

## Dataset

### NASA C-MAPSS FD001 Dataset

The dataset contains simulated turbofan engine degradation trajectories.

Each row represents one operational cycle of an engine and includes:

- Engine ID
- Cycle number
- 3 operational settings
- 21 sensor measurements

Dataset characteristics:

- Multivariate time-series data
- Run-to-failure engine trajectories
- Highly imbalanced anomaly detection setting
- Single operating condition (FD001 subset)

**Use Case:** Predictive Maintenance / Anomaly Detection

---

## Data Preprocessing

The preprocessing pipeline was designed to ensure reproducibility and prevent data leakage.

### Key preprocessing steps:

- Parsed raw NASA text files into structured tabular format
- Removed low-information / near-constant sensors
- Applied **Z-score standardization** using training-set statistics only
- Generated fixed-length sliding windows:
  - Sequence length = **80 cycles**
  - Stride = **1**
- Assigned anomaly labels using horizon-based labeling
- Split data into:
  - Training
  - Validation
  - Independent test set

Additional reproducibility artifacts include:

- preprocessing metadata
- dataset split definitions
- configuration tracking

---

## Methodology

Two deep learning architectures were evaluated.

### 1. LSTM Autoencoder (Baseline)

A sequence-aware recurrent autoencoder trained to reconstruct normal engine behavior.

Architecture:

- Encoder:
  - LSTM (128 units)
  - LSTM (64 units)
  - Latent layer (32)
- Decoder:
  - mirrored LSTM reconstruction pipeline
- Dropout regularization
- Reconstruction loss: Mean Squared Error (MSE)

Anomaly detection:

- reconstruction error scoring
- variance normalization
- threshold calibration on validation data
- warm-up normalization during testing

---

### 2. Self-Supervised Masked Autoencoder (MAE)

A masked reconstruction model adapted for time-series anomaly detection.

Approach:

- random masking of time steps and sensor channels
- masked-value reconstruction objective
- hybrid reconstruction loss
- Monte Carlo inference for stable anomaly scoring

This approach was intended to test whether self-supervised masked reconstruction improves anomaly detection performance over traditional sequence reconstruction.

---

## Experimental Design

To ensure fair comparison:

- identical preprocessing pipeline
- same train / validation / test split
- consistent evaluation metrics
- early stopping
- validation-based threshold tuning
- warm-up score calibration
- post-processing via anomaly run filtering

Evaluation focused on imbalanced anomaly detection metrics rather than accuracy.

---

## Results

### Comparative Performance

| Model | Dataset | PR-AUC | F1-score | Precision | Recall |
|------|---------|--------|----------|-----------|--------|
| LSTM-AE | Validation | 0.838 | 0.743 | 0.828 | 0.674 |
| MAE | Validation | 0.612 | 0.517 | 0.543 | 0.494 |
| LSTM-AE | Test | 0.537 | 0.436 | 0.361 | 0.550 |
| MAE | Test | 0.187 | 0.183 | 0.147 | 0.242 |

---

## Key Findings

- LSTM Autoencoder consistently outperformed the Masked Autoencoder
- Stronger PR-AUC, F1-score, precision, and recall across validation and test sets
- Sequence-aware recurrent modeling proved more effective in this experimental setting
- Self-supervised masked reconstruction showed promise but underperformed the LSTM baseline here

---

## Visual Results

### Precision-Recall Comparison

#### Validation
![Validation PR Curve](artifacts/figures/pr_VAL_LSTM_vs_MAE.png)

#### Test
![Test PR Curve](artifacts/figures/pr_TEST_warm_LSTM_vs_MAE.png)

### Class Distribution
![Label Distribution](artifacts/figures/val_test_label_hist.png)

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Repository Structure

```text
predictive-anomaly-detection-nasa-cmapss/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── lstm_ae.ipynb
│   ├── mae.ipynb
│   └── visualization.ipynb
│
├── artifacts/
│   ├── figures/
│   ├── metrics_lstm_test.json
│   ├── metrics_lstm_val.json
│   ├── metrics_mae_test.json
│   ├── metrics_mae_val.json
│   ├── preprocessing_meta.json
│   └── splits.json
│
├── config.yaml
├── requirements.txt
└── README.md
