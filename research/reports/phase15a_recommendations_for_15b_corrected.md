# Phase 15B — Architectural Recommendations & Mandatory Controls
**CAEG-Net: Controlled Refinements Grounded in Reconciled Phase 15A Diagnostics**

**Date:** 2026-09-11  
**Authoritative Reference Model:** Candidate F2 / A2-OOF (`ConfidenceFallbackCAEGNet`, 121,724 parameters)  
**Evaluation Seeds:** {42, 123, 999, 2024, 3407}  
**Datasets:** Modern PJM (MW), GEFCom2014 (kW), UCI Electricity Cohort 320 Aggregate (MW)  
**Status:** AUDITED, RECONCILED & LOCKED

---

## 1. Executive Summary

Phase 15A forensic diagnostic reconciliations have clearly established four key empirical realities:
1. **Horizon Specialization is Substantial:** TCN dominates short horizons ($h \le 8$) while recurrent LSTM dominates longer horizons ($h \ge 17$). A single global 3-simplex routing vector $\mathbf{w} \in \Delta^2$ forces the network to compromise across lead times, ignoring lead-time crossover.
2. **Confidence Head Operates as Fixed Shrinkage:** The dynamic parameter $\lambda_t$ displays narrow dispersion (mean $\approx 0.51$, CV $< 1.5\%$) and lacks statistical calibration to instance-specific adaptive advantage ($p > 0.20$). A fixed scalar $\lambda = 0.50$ performs within $\le 0.01$ units of the dynamic MLP head across all benchmarks.
3. **Adaptive Routing Requires Regularization:** On GEFCom and UCI, pure adaptive routing alone moves performance in an adverse direction relative to equal weighting; the shrinkage toward equal centroid offsets this error.
4. **Disagreement Carries Regime Signal:** Expert forecast spread correlates with adaptive win rate during high-volatility regimes ($53\% - 58\%$ win rate).

Phase 15B must NOT engage in open-ended architecture search or introduce unrelated model families (Transformers, attention). It must execute controlled, hypothesis-driven refinements designed to isolate each mechanism.

---

## 2. Mandatory Experimental Controls for Phase 15B

To rigorously disentangle routing alpha, horizon specialization, dynamic confidence, and fixed centroid shrinkage, Phase 15B MUST evaluate the following four experimental controls alongside any proposed candidate:

### Control A (Canonical Baseline)
- **Architecture:** Current Candidate F2 / A2-OOF (`ConfidenceFallbackCAEGNet`, 121,724 parameters).
- **Formulation:** Global routing vector $\mathbf{w} \in \Delta^2$ + 145-parameter dynamic confidence head $\lambda(c)$.
- **Purpose:** Authoritative benchmark anchor establishing whether any new candidate provides statistically significant improvement under non-overlapping daily-block paired tests.

### Control B (Constant Shrinkage Control)
- **Architecture:** Current global router with a fixed scalar shrinkage factor $\lambda^* = \bar{\lambda}_{\mathrm{F2}} \approx 0.51$ (zero parameter confidence head).
- **Formulation:** $\hat{y}_{\mathrm{final}} = \lambda^* \hat{y}_{\mathrm{adaptive}} + (1 - \lambda^*) \hat{y}_{\mathrm{equal}}$.
- **Purpose:** Isolates whether a dynamic confidence head provides genuine instance-varying value over a stationary centroid shrinkage regularizer.

### Control C (Pure Horizon-Aware Routing Control)
- **Architecture:** Horizon-partitioned router WITHOUT confidence fallback or shrinkage ($\\lambda = 1.0$ permanently).
- **Formulation:** $\hat{y}_{\mathrm{final}, h} = \sum_{i} w_{i, h} \hat{y}_{i, h}$.
- **Purpose:** Isolates the pure performance contribution of horizon-specific specialization from the regularization effect of equal-centroid shrinkage.

### Control D (Pure Dynamic Shrinkage Control)
- **Architecture:** Global unpartitioned router with dynamic confidence head (equivalent to Control A but evaluated strictly against Control B and Control C).
- **Purpose:** Isolates whether dynamic reliability gating alone without horizon specialization provides measurable advantage.

---

## 3. Recommended Phase 15B Refinement Candidates

Supported by the reconciled diagnostic evidence, the following hypothesis-driven candidates are recommended for evaluation in Phase 15B:

### Candidate 1: Horizon-Grouped Routing (HGR)
- **Hypothesis:** Partitioning the 24-hour horizon into 3 distinct temporal heads—Short ($h \in \{1, \dots, 8\}$), Medium ($h \in \{9, \dots, 16\}$), and Long ($h \in \{17, \dots, 24\}$)—will allow the network to exploit empirical expert crossover (TCN for short, LSTM for long) without exploding parameter counts.
- **Budget Constraint:** Router head increases from $9 \times 3 = 27$ parameters to $9 \times 9 = 81$ parameters (+54 parameters total, well within the 120k–125k parameter budget).

### Candidate 2: Disagreement-Gated Shrinkage (DGS)
- **Hypothesis:** Replacing abstract context features in the confidence head with explicit expert forecast disagreement spread ($D_t = \max |\hat{y}_i - \hat{y}_j|$) will establish genuine calibration, increasing shrinkage toward equal ensemble when experts diverge erratically.
- **Formulation:** $\lambda_t = \sigma(W_D D_t + b_D)$.

### Candidate 3: Horizon-Grouped Routing + Disagreement Fallback (HGR-DGS)
- **Hypothesis:** Combining horizon-grouped routing (to exploit crossover) with disagreement-gated shrinkage (to prevent catastrophic divergence in volatile regimes) will capture complementary gains.

---

## 4. Methodological Safeguards for Phase 15B

1. **Strict 5-Seed Evaluation:** Every candidate and control must be trained and evaluated across seeds {42, 123, 999, 2024, 3407}.
2. **Deterministic Reproducibility:** Deterministic data loaders, CUDNN seeding, and identical chronological train/val/test splits.
3. **Non-Overlapping Daily-Block Statistical Testing:** Paired $t$-tests, Wilcoxon signed-rank tests, and Holm-Bonferroni correction over non-overlapping 24h blocks ($K=53, 456, 163$).
4. **No Test-Driven Model Tuning:** Screening must occur strictly on validation data using cached causal OOF features.
