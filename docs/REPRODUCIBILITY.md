# CAEG-Net: Full Reproducibility & Verification Guide

This document provides complete, unambiguous instructions to reproduce all aspects of **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting), including environment configuration, regression testing, benchmark evaluation, research dashboard execution, and faculty notebook review.

---

## 1. Environment Setup

### 1.1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.13, 3.14)
- Git
- PyTorch 2.0+ (CUDA optional; all evaluations run on CPU or GPU)

### 1.2. Installation
Clone the repository and install project dependencies:
```bash
git clone https://github.com/vinay-0208/CAEGNET.git
cd CAEGNET

# Optional: Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install locked dependencies
pip install -r requirements.txt
```

---

## 2. Automated Test Suite

CAEG-Net includes two comprehensive test suites verifying model parameter counts, causality constraints, data windowing, and regression integrity:

### 2.1. Top-Level Integration Tests
Verifies causality, zero-leakage scaling, and data sliding window integrity:
```bash
python -m unittest discover -s tests
```
*Expected: 4 tests ran in ~1.2s -> OK*

### 2.2. Comprehensive Research Unit Tests
Verifies phase architectures, expert submodules, causal OOF gating, and locked parameter budgets:
```bash
python -m unittest discover -s research/tests
```
*Expected: 172 tests ran in ~41s -> OK*

### 2.3. Full Codebase Compilation Check
Ensures all Python modules compile cleanly without syntax errors:
```bash
python -m py_compile dashboard/app.py
python -m compileall -q dashboard/ tests/ research/ src/
```

---

## 3. Verifying Authoritative Model Parameters

To programmatically verify that CAEG-Net contains exactly **121,724 trainable parameters**, run:
```bash
python -c "
from src.models import ConfidenceFallbackCAEGNet
from src.utils import count_parameters

model = ConfidenceFallbackCAEGNet()
counts = count_parameters(model)
print('Total Trainable Parameters:', counts['total_trainable'])
assert counts['total_trainable'] == 121724
print('Param count verified successfully!')
"
```
*Expected Output:*
```text
Total Trainable Parameters: 121724
Param count verified successfully!
```

---

## 4. Launching the Interactive Research Dashboard

The interactive Streamlit research dashboard allows comprehensive exploration of all 12 academic evaluation panels, including 192-hour trajectory plots and step-by-step forecast residuals:

```bash
# Direct Streamlit launch
streamlit run dashboard/app.py

# Or via convenience script
python scripts/run_dashboard.py
```
Open your web browser at `http://localhost:8501`.

---

## 5. Executing the Faculty Review Notebook

The authoritative faculty review notebook (`notebooks/CAEG_Net_Faculty_Review.ipynb`) is genuinely executable from a fresh kernel and generates all 11 tables and figures dynamically from verified repository artifacts:

```bash
# Execute the notebook end-to-end via nbconvert
python -m jupyter nbconvert --to notebook --execute "notebooks/CAEG_Net_Faculty_Review.ipynb" --output "CAEG_Net_Faculty_Review.ipynb"
```
*Expected: 39 cells executed with 0 errors.*

---

## 6. Authoritative Benchmark Reference Values

All benchmark metrics in CAEG-Net are certified from 5-seed chronological evaluations (`[42, 123, 999, 2024, 3407]`). Raw artifacts reside in `research/results/final_results.csv`:

| Benchmark Grid | Geography & Metric | CAEG-Net (Locked F2) | Best Standalone Expert | Static Equal Ensemble | Conventional Input MoE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PJM** | MAE (MW) | **$250.97 \pm 10.69$** | $259.33$ (TCN) | $279.83$ | $276.30 \pm 13.64$ |
| | RMSE (MW) | **$335.38$** | $349.50$ | $371.12$ | $368.45$ |
| | $R^2$ Score | **$0.8714$** | $0.8601$ | $0.8415$ | $0.8441$ |
| **GEFCom2014** | MAE (kW) | **$12.41 \pm 0.15$** | $12.57$ (TCN) | $12.62$ | — |
| | RMSE (kW) | **$18.04$** | $18.45$ | $18.33$ | — |
| | $R^2$ Score | **$0.8610$** | $0.8542$ | $0.8561$ | — |
| **UCI Electricity**| MAE (MW) | **$7.74 \pm 0.30$** | $7.55$ (LSTM) | $8.17$ | — |
| | RMSE (MW) | **$10.96$** | $10.62$ | $11.51$ | — |
| | $R^2$ Score | **$0.9831$** | $0.9841$ | $0.9813$ | — |

*Protocol Note: Unmeasured baselines without 5-seed protocol-comparable runs are designated by '—'; zero numbers are fabricated.*