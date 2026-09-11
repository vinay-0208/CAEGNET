# Phase 15A — Evidence-Backed Recommendations for Phase 15B
**CAEG-Net: Controlled Architecture Refinement & Targeted Opportunity Capture**

**Author:** Phase 15A Diagnostic Audit Panel  
**Date:** 2026-09-11  
**Authoritative Reference:** Phase 15A Diagnostic Suite (`research/analysis/phase15a_*.csv`, `research/plots/phase15a_*.png`)  
**Objective:** Formulate, evaluate, and prioritize candidate mechanisms for Phase 15B based strictly on empirical diagnostic evidence.

---

## 1. Executive Summary & Diagnostic Context

Phase 15A conducted an exhaustive empirical dissection of CAEG-Net Candidate F2 (`F2_A2_OOF`) across Modern PJM, GEFCom2014, and UCI Cohort 320. The findings establish three definitive empirical realities:

1. **Routing Dynamicity Reality:** The current router does not perform high-frequency dynamic expert switching (top expert change frequency is 0.0, effective number of experts $N_{\mathrm{eff}} \approx 2.97$). Instead, it learns a stationary per-dataset inductive bias around the equal centroid $[0.33, 0.33, 0.33]$.
2. **Confidence Lambda Reality:** The learned confidence head parameter $\lambda_t$ settles at a near-constant value (0.509 on PJM, 0.513 on GEFCom, 0.514 on UCI, $CV < 1.5\%$) with zero extreme values. Crucially, $\lambda_t$ exhibits no statistically significant correlation with realized adaptive advantage ($r = -0.05, -0.06, +0.10$, all $p > 0.15$). It functions purely as a learned global shrinkage regularizer rather than dynamic decision confidence.
3. **Horizon Specialization Opportunity:** While the router applies a single global weight vector to all 24 forecast steps, expert performance diverges sharply across horizons: on Modern PJM, CNN achieves the lowest error on steps 1–8 ($100\%$ of short-horizon steps), while LSTM achieves the lowest error on steps 17–24 ($100\%$ of long-horizon steps).
4. **Oracle Headroom:** F2 captures only $2.3\%$ (PJM), $9.6\%$ (GEFCom), and $22.1\%$ (UCI) of the available theoretical oracle headroom over equal ensembling, leaving substantial untapped potential.

Based strictly on these findings, candidate mechanisms for Phase 15B are ranked and evaluated below.

---

## 2. Evaluation of Candidate Mechanisms

### Candidate 1: Horizon-Grouped Routing (HGR)
- **Diagnostic Motivation:** Diagnostic J demonstrated clear, systematic horizon specialization. On PJM, CNN dominates short steps (1–8h, MAE 141.5 MW vs LSTM 165.2 MW), while LSTM dominates long steps (17–24h, MAE 326.8 MW vs CNN 391.2 MW). A single global routing vector forcibly ignores this structural divergence.
- **Expected Mechanism:** Divide the 24-hour horizon into 3 operational regimes: Short ($h=1\dots 8$), Medium ($h=9\dots 16$), and Long ($h=17\dots 24$). The router outputs 3 distinct convex weight vectors $\mathbf{w}_{\mathrm{short}}, \mathbf{w}_{\mathrm{med}}, \mathbf{w}_{\mathrm{long}} \in \Delta^2$.
- **Evidence For:** Direct empirical proof of horizon divergence on PJM and GEFCom.
- **Evidence Against / Risks:** Modest increase in router parameters (+64 parameters). Potential risk of step-boundary discontinuity, which can be mitigated with smooth interpolation or independent block projection.
- **Implementation Complexity:** Low-to-Moderate.
- **Recommendation:** **PRIORITY 1 — PROCEED TO 15B**.

---

### Candidate 2: Disagreement-Gated Shrinkage Fallback (DGS)
- **Diagnostic Motivation:** Diagnostic H proved that when expert disagreement is high, adaptive routing error degrades relative to equal ensembling (PJM: $-0.57$ MW advantage; GEFCom: $-0.78$ kW advantage; UCI: $-0.95$ MW advantage). Conversely, in low disagreement regimes, adaptive routing consistently outperforms equal ensembling (+0.95 MW, +0.18 kW, +0.94 MW).
- **Expected Mechanism:** Formulate the shrinkage parameter as an inverse function of normalized expert spread or disagreement:
  $$\lambda_t = \sigma\left(\gamma_0 - \gamma_1 \frac{\bar{D}_t}{\bar{y}_t}\right)$$
  When experts agree, $\lambda_t \to 1$ (router autonomy). When experts disagree sharply, $\lambda_t \to 0$ (safe fallback to unweighted equal centroid).
- **Evidence For:** Statistically significant negative correlation between disagreement and adaptive advantage ($p < 10^{-50}$ on GEFCom and UCI). Provides a principled power systems engineering rationale (risk-averse dispatch).
- **Evidence Against / Risks:** Requires robust scaling of disagreement across load tiers.
- **Implementation Complexity:** Low (+2 learned parameters or parameter-free calibration).
- **Recommendation:** **PRIORITY 2 — PROCEED TO 15B**.

---

### Candidate 3: Simplified Static Scalar Shrinkage Control (F5 Simplification)
- **Diagnostic Motivation:** Diagnostic G revealed that a fixed scalar shrinkage $\lambda = 0.50$ achieves performance virtually identical to the 145-parameter MLP confidence head across all three benchmarks (PJM: 249.91 vs 249.90 MW; GEFCom: 12.57 vs 12.58 kW; UCI: 8.21 vs 8.21 MW).
- **Expected Mechanism:** Replace the uncalibrated MLP confidence head with a single global scalar parameter $\lambda^* \in [0.4, 0.6]$ tuned on validation data.
- **Evidence For:** Eliminates 145 unnecessary parameters, simplifies theoretical presentation, avoids misleading reviewers with pseudo-confidence claims.
- **Evidence Against / Risks:** Forfeits temporal variation in shrinkage (which was already $<1.5\%$ in practice).
- **Implementation Complexity:** Very Low.
- **Recommendation:** **PRIORITY 3 — PROCEED TO 15B (Essential baseline / structural control)**.

---

### Candidate 4: Multi-Scale Temporal Performance Feedback (Short + Medium OOF)
- **Diagnostic Motivation:** Diagnostic K revealed that historical 24h error persistence is moderate ($0.64-0.79$), while multi-day smoothed error windows (48h and 72h) maintain high stability on distribution and consumer loads.
- **Expected Mechanism:** Provide both 24h immediate relative error and 72h moving average relative error (or exponentially smoothed $\alpha=0.50$) to the context vector (total context dim 10).
- **Evidence For:** Better signal-to-noise ratio in noisy regimes.
- **Evidence Against / Risks:** Marginal gain if router weights remain strongly regularized toward the centroid.
- **Implementation Complexity:** Low.
- **Recommendation:** **PRIORITY 4 — PROCEED TO 15B (Secondary ablation)**.

---

## 3. Mechanisms Strictly Rejected for Phase 15B

| Rejected Mechanism | Diagnostic Reason for Rejection |
| :--- | :--- |
| **Open-Ended Architecture Search** | Directly violates research governance rules; empirical core is locked to LSTM+TCN+CNN. |
| **Adding Transformers / Attention** | Violates canonical expert family; parameter overhead and high-frequency self-attention are unmotivated by diagnostics. |
| **Unconstrained Dynamic Confidence** | Diagnostics prove unconstrained dynamic switching destabilizes routing; loss landscape strongly favors conservative shrinkage. |
| **Single-Window Instantaneous Error** | 1-hour error snapshots have near-zero autocorrelation ($<0.12$), introducing pure high-frequency noise. |
| **Per-Dataset Hyperparameter Tuning** | Would violate strict zero-test-tuning protocol and undermine cross-dataset generalization claims. |

---

## 4. Synthesis of Phase 15B Experimental Plan

Based on the prioritized candidates, Phase 15B should implement a disciplined, factorial 4-candidate comparison:
1. **Control F2 (Baseline):** Current F2 architecture (7D context, global router, MLP confidence head).
2. **Candidate M1 (Horizon-Grouped Routing):** 7D context, 3-group router (Short, Medium, Long), fixed scalar shrinkage $\lambda=0.50$.
3. **Candidate M2 (Disagreement-Gated Fallback):** 7D context, global router, variance-adaptive disagreement fallback $\lambda(\bar{D}_t)$.
4. **Candidate M3 (Combined Horizon-Grouped + Disagreement Fallback):** 3-group router + variance-adaptive fallback.

This targeted roadmap directly attacks the two largest proven weaknesses identified in Phase 15A—horizon rigidity and failure under high disagreement—while preserving full experimental provenance and computational tractability.
