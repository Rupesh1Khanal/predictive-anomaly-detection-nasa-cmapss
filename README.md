# Predictive Anomaly Detection on NASA Turbofan Engine Data
### Master's Thesis | Comparative Study of LSTM Autoencoder vs Self-Supervised Masked Autoencoder

A comparative deep learning study on predictive anomaly detection using the NASA C-MAPSS FD001 turbofan engine dataset.

This master's thesis project evaluates whether a self-supervised masked reconstruction approach can outperform a traditional sequence-aware LSTM Autoencoder for anomaly detection in multivariate time-series sensor data.

---

## Results at a Glance

The **LSTM Autoencoder consistently outperformed the self-supervised Masked Autoencoder baseline** across validation and test sets.

| Model | Dataset | PR-AUC | F1-score | Precision | Recall |
|------|---------|--------|----------|-----------|--------|
| LSTM-AE | Validation | 0.838 | 0.743 | 0.828 | 0.674 |
| MAE | Validation | 0.612 | 0.517 | 0.543 | 0.494 |
| LSTM-AE | Test | 0.537 | 0.436 | 0.361 | 0.550 |
| MAE | Test | 0.187 | 0.183 | 0.147 | 0.242 |

**Key takeaway:** In this experimental setting, the sequence-aware recurrent baseline demonstrated substantially stronger anomaly detection performance than the tested self-supervised masked reconstruction approach.

---

## Visual Results

### Precision–Recall Comparison (Validation)
![Validation PR Curve](artifacts/figures/pr_VAL_LSTM_vs_MAE.png)

### Precision–Recall Comparison (Test)
![Test PR Curve](artifacts/figures/pr_TEST_warm_LSTM_vs_MAE.png)

### Validation / Test Label Distribution
![Label Distribution](artifacts/figures/val_test_label_hist.png)

---

## Problem Statement

Unexpected aircraft engine failures can lead to operational disruption, safety risks, and significant maintenance costs.

Predictive maintenance aims to identify abnormal degradation patterns before catastrophic failure occurs.

This project explores whether self-supervised masked reconstruction can improve anomaly detection performance compared to a traditional LSTM Autoencoder in aerospace sensor monitoring.

---

## Dataset

### NASA C-MAPSS FD001

The project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001 subset.

Dataset characteristics:

- Multivariate time-series sensor data
- Simulated turbofan engine degradation trajectories
- Run-to-failure operational sequences
- Single operating condition
- Highly imbalanced anomaly detection setting

Each observation contains:

- Engine ID
- Operational cycle
- 3 operational settings
- 21 sensor measurements

---

## Data Preprocessing

The preprocessing pipeline was designed for reproducibility and leakage prevention.

Key steps:

- Raw NASA text files parsed into structured tabular format
- Low-information / near-constant sensors removed
- Z-score standardization using **training-set statistics only**
- Fixed-length sliding window generation:
  - Sequence length: **80 cycles**
  - Stride: **1**
- Horizon-based anomaly labeling
- Train / validation / independent test split

Reproducibility metadata includes:

- preprocessing metadata
- split definitions
- configuration tracking

---

## Methodology

Two anomaly detection architectures were compared under identical evaluation conditions.

### 1. LSTM Autoencoder (Baseline)

A sequence-aware recurrent autoencoder trained to reconstruct normal engine behavior.

Approach:

- sequence reconstruction learning
- reconstruction-error anomaly scoring
- variance-normalized error aggregation
- validation-based threshold calibration
- warm-up normalization at inference

---

### 2. Self-Supervised Masked Autoencoder (MAE)

A masked reconstruction approach adapted for multivariate time-series anomaly detection.

Approach:

- random masking of temporal positions and sensor features
- masked reconstruction objective
- self-supervised representation learning
- stochastic inference-based anomaly scoring
- validation-based threshold calibration

---

## Evaluation Metrics

Because anomaly detection is highly imbalanced, accuracy was not prioritized.

Evaluation focused on:

- **PR-AUC (primary metric)**
- F1-score
- Precision
- Recall

PR-AUC was selected as the primary comparison metric due to its suitability for imbalanced anomaly detection tasks.

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
│   ├── metrics_*.json
│   ├── preprocessing_meta.json
│   └── splits.json
│
├── .ignore
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Limitations

- Evaluation limited to FD001 subset only
- Single anomaly detection framing
- Performance may vary under different operating conditions
- Further tuning of masked architectures may improve performance

---

## Future Improvements

Potential extensions:

- Evaluate FD002 / FD003 / FD004 datasets
- Explore transformer-based time-series architectures
- Hyperparameter optimization
- Alternative masking strategies
- Probabilistic anomaly scoring
- Deployment-oriented inference optimization

---

## Author

**Rupesh Khanal**

Master’s Thesis Project  
MSc Data Science, AI & Digital Business
