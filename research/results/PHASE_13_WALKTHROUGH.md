# Phase 13 Execution Walkthrough: Genuine OOF Routing Validation

**Date:** September 10, 2026  
**Environment:** `caeg-gpu` (Python 3.10+, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 4050 Laptop GPU)  
**Execution Runtime:** 9,451.99 seconds (~2.63 hours)  
**Status:** Complete & Fully Validated  

---

## 1. Execution Overview
Phase 13 resolved the methodological limitation of Phase 12 by implementing genuine expanding-window chronological out-of-fold (OOF) expert performance generation, evaluating candidates A0, A1, A2, and A3 across Modern PJM, GEFCom2014, and UCI Cohort 320.

---

## 2. Key Milestones Completed

1. **Pre-Experiment Unit Tests:**
   - Implemented `research/tests/test_phase13_oof_performance_aware_gating.py`.
   - Passed 6/6 tests in 0.12s verifying parameter counts, OOF causality, relative error normalization, and fallback bounds.
2. **Expanding-Window Chronological OOF Generation:**
   - Partitioned training data into 4 chronological blocks.
   - Fold 1 trained on Block 1 $\to$ predicted Block 2 OOF.
   - Fold 2 trained on Blocks 1+2 $\to$ predicted Block 3 OOF.
   - Fold 3 trained on Blocks 1+2+3 $\to$ predicted Block 4 OOF.
   - Full train model predicted Validation and Test partitions out-of-sample.
3. **Stage B Validation Screening (Seeds 42, 123):**
   - Evaluated A0, A1, A2, and A3 on validation partitions.
   - All three candidates qualified under the pre-registered rule ($\ge 2/3$ datasets improved, degradation $\le 2.0\%$).
4. **Stage C Five-Seed Finalist Benchmark:**
   - Evaluated all finalists on locked test partitions across seeds `[42, 123, 999, 2024, 3407]`.
   - Results:
     - A1_P2_OOF achieved best PJM test MAE ($250.63 \pm 5.55\text{ MW}$).
     - A2_C1_OOF achieved best UCI test MAE ($7.59 \pm 0.15\text{ MW}$) and record GEFCom MAE ($12.46 \pm 0.12\text{ kW}$).
     - A3_Confidence_Only achieved record GEFCom test MAE ($12.44 \pm 0.21\text{ kW}$) and $7.71\text{ MW}$ on UCI.
5. **Stage D Daily-Block Inferential Statistics:**
   - Evaluated paired daily-block differences on non-overlapping 24h blocks ($K=53, 456, 163$).
   - A2 statistically beat Canonical V1 on GEFCom ($p_{\text{adj}} = 6.10 \times 10^{-23}$) and UCI ($p_{\text{adj}} = 1.30 \times 10^{-4}$).
6. **Publication Figures:**
   - Generated 8 publication figures in `research/results/phase13_plots/`.

---

## 3. Verified Artifacts
- `research/results/phase13_complexity.csv`
- `research/results/phase13_validation_results.csv`
- `research/results/phase13_candidate_comparison.csv`
- `research/results/phase13_five_seed_results.csv`
- `research/results/phase13_statistical_tests.csv`
- `research/results/phase13_routing_analysis.csv`
- `research/results/phase13_lambda_distribution.csv`
- `research/results/phase13_dataset_summary.csv`
- `research/results/phase13_plots/phase13_01_validation_candidate_comparison.png` through `phase13_08_complexity_vs_performance.png`
- `research/results/PHASE_13_REPORT.md`
- `research/results/PHASE_13_POST_EXECUTION_AUDIT.md`
- `research/results/PHASE_13_WALKTHROUGH.md`

---
**End of Phase 13 Walkthrough**
