# CAEG-Net: Full Reproducibility & Verification Guide

This document provides complete, unambiguous instructions to reproduce all aspects of **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting), including fresh environment configuration, automated testing, live inference simulation, benchmark evaluation, and faculty notebook review.

---

## 1. Quickstart Environment Setup

### 1.1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Git
- PyTorch 2.0+ (CUDA supported; CPU execution supported out-of-the-box)

### 1.2. Fresh Clone & Installation
```bash
git clone https://github.com/vinay-0208/CAEGNET.git
cd CAEGNET

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install locked dependencies
pip install -r requirements.txt
```

---

## 2. Bundled Datasets & Checkpoints (Self-Contained Repository)

To ensure **100% out-of-the-box reproducibility** without requiring gigabytes of external data downloads or hours of GPU retraining:

1. **Modern PJM Interconnection (`data/Modern_PJM/pjm_load.csv`, ~315 KB):**
   - Bundled directly in git. Full hourly load series (2018–2024, 54,768 records in MW).
   - Used for training, causal context feature extraction, and live forecast simulation.

2. **GEFCom2014 (`data/Load/`, ~14.9 MB):**
   - Bundled directly in git across all 15 rolling competition tasks (`L*-train.csv`, `L*-benchmark.csv`, and solutions).

3. **Pre-Trained Frozen Model Checkpoints (`checkpoints/`, ~34.5 MB):**
   - Complete 5-seed checkpoints (`seed_42`, `seed_123`, `seed_999`, `seed_2024`, `seed_3407`).
   - Includes Canonical V1 models (`caeg_full.pt`), standalone experts (`lstm.pt`, `tcn.pt`, `cnn.pt`), and ablation models (`caeg_no_rec_error.pt`, `input_moe.pt`).

4. **Multi-Seed Benchmark Cache (`results/` & `dashboard/assets/data/`, ~31 MB):**
   - Pre-computed predictions, routing weights, and metrics for instant evaluation without re-training.

5. **Optional: UCI Electricity Load Diagrams (678 MB):**
   - Due to GitHub's 100 MB per-file limit, the 678 MB raw text file (`LD2011_2014.txt`) is excluded from git.
   - However, its pre-processed evaluation cache and 5-seed results are already bundled in `dashboard/assets/data/`.
   - To re-download the raw file: [UCI ML Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) $\to$ place in `data/ElectricityLoadDiagrams20112014/LD2011_2014.txt`.

---

## 3. Automated Verification & Test Suite

The repository includes a comprehensive 35-assertion test suite verifying causal zero-leakage constraints, parameter budgets, and live simulator inference parity:

```bash
# Run full test suite via pytest
pytest tests/
```
*Expected: 35 passed in ~15-18s (OK)*

### Specific Test Coverage:
- `tests/test_causality.py` (18 tests):
  - Counterfactual future target perturbation test (ISS-01).
  - Causal recent forecast error availability test (ISS-02).
  - Train-only `StandardScaler` parameter isolation test.
  - Partition boundary guard & chronological order validation.
- `tests/test_dashboard_cache.py` (7 tests):
  - Verified presence and valid simplex property of 5-seed routing weights.
  - Ground truth array alignment and preset window bounds.
- `tests/test_forecast_simulator.py` (10 tests):
  - Strict loading of all 5 Canonical V1 checkpoints (121,531 parameters).
  - Live PyTorch inference parity against pre-computed cache ($< 10^{-4}$ MW difference).
  - Dynamic input validation (CSV schema, monotonic timestamps, physical load bounds).

---

## 4. Model Architecture & Parameter Verification

To verify that the model architecture adheres to the exact parameter budget:

```bash
python scripts/verify_reproducibility.py
```
*Expected Output:*
```text
======================================================================
  CAEG-Net Architecture & Parameter Verification
======================================================================
  [PASS] LSTM Expert                        :   56152
  [PASS] TCN Expert                         :   36952
  [PASS] CNN Expert                         :   27400
  [PASS] Context Router (Encoder + Gating)  :    1075
  [PASS] Confidence Fallback Head           :     145
  [PASS] CAEG-Net Total Parameters          :  121724
  [SUCCESS] All architecture and unit test verifications passed.
```

---

## 5. Console Benchmark Evaluator

To inspect the 5-seed certified performance table across PJM, GEFCom2014, and UCI Electricity without launching a browser:

```bash
python scripts/evaluate_benchmarks.py
```

---

## 6. Launching the Interactive Research Dashboard

Launch the Streamlit research dashboard for live inference simulation and interactive evaluation:

```bash
# Convenience launcher
python scripts/run_dashboard.py

# Or directly via streamlit
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Executing the Faculty Review Notebook

The authoritative review notebook is self-contained and pre-rendered, and can be re-executed from a clean kernel:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/CAEG_Net_Faculty_Review.ipynb
```
*Expected: All 25 cells execute cleanly from scratch using local repository artifacts.*

---

## 8. Academic Project Report

The complete, anonymous Deep Learning project report is available in Microsoft Word format:
- [`docs/CAEG_Net_Deep_Learning_Project_Report.docx`](docs/CAEG_Net_Deep_Learning_Project_Report.docx) *(Contains all 14 high-resolution figures, 12 data tables, and 20+ literature citations).*
