# CAEG-Net: Literature & Scientific Novelty Positioning

## 1. What Existing Work Already Does
The time-series forecasting and machine learning literature has thoroughly investigated ensemble learning and mixture-of-experts (MoE) architectures:
1. **Classical & Dynamic Stacking**: Linearly combining multiple model forecasts via time-varying regression, Kalman filter blending, or exponentially weighted moving averages of historical errors has a multi-decade history in econometric and energy forecasting (Bates & Granger, 1969; Timmermann, 2006; Hong et al., 2016).
2. **Standard Input-Dependent MoE**: Neural mixture-of-experts architectures (Jacobs et al., 1991; Jordan & Jacobs, 1994; Shazeer et al., 2017) commonly pass the raw input sequence $X_t$ directly into a gating network to compute sample-level routing coefficients $\mathbf{w}_t = \text{Softmax}(g(X_t))$.
3. **Temporal Ensembles for Load Forecasting**: Hybrid combinations of recurrent models (LSTM/GRU), convolutional nets (CNN/TCN), and gradient-boosted trees (XGBoost/LightGBM) are standard practice in modern energy system operations.

---

## 2. What CAEG-Net Shares with Existing Work
- **Parallel Multi-Expert Backbone**: Uses established deep learning architectures (stacked LSTM, causal dilated TCN, and multi-stage 1D CNN) as specialized inductive biases for long-range dependency, multi-scale receptive field, and localized motif extraction.
- **Convex Linear Blending**: Gating weights are constrained to the probability simplex ($\sum_i w_i = 1, w_i > 0$), ensuring that the ensemble forecast is a convex combination of candidate expert outputs.
- **End-to-End Gradient Training**: The gating network and experts are jointly optimizable via standard backpropagation on mean squared error (or robust Huber loss).

---

## 3. What is Different in CAEG-Net
1. **Explicit Operational Context vs. Raw Input Gating**:
   Instead of forcing a gating network to discover high-level system states from high-dimensional noisy raw load inputs $X \in \mathbb{R}^{168 \times 1}$ (which often suffers from high variance and over-parameterization, as seen in Standard Input-MoE), CAEG-Net routes using a low-dimensional, physically interpretable operational context vector $C_t \in \mathbb{R}^k$:
   - Real-time window trend ($\beta$)
   - Normalized local volatility ($\sigma$)
   - Diurnal lag-24 autocorrelation ($r_{24}$)
   - Causal out-of-sample recent forecast error ($e_{\text{recent}}$)
2. **Causal Closed-Loop Forecast Error Feedback**:
   Recent forecasting error is generated through an expanding walk-forward out-of-sample baseline. It is provided to the gate as a diagnostic sensor reflecting recent model accuracy, rather than being added as an arbitrary post-hoc residual subtraction.
3. **Forecast-Aware Gating (Inter-Expert Disagreement)**:
   In our research track variant, the gating network additionally observes the *disagreement* (pairwise dispersion, standard deviation, and range) among the candidate expert forecasts generated from the input window $X_t$ prior to finalizing routing decisions. This acts as an endogenous uncertainty signal.

---

## 4. What Claims We Can Safely Make (Paper-Ready)
- **Defensible Claim 1**: Explicit operational context routing statistically outperforms static equal weighting ($p = 0.0153 < 0.05$, paired t-test across 5 seeds; $p = 0.0332$ on 53 non-overlapping 24h episodes).
- **Defensible Claim 2**: Explicit operational context routing statistically outperforms raw input-dependent gating (Standard Input-MoE) on identical expert backbones ($p = 0.0415 < 0.05$).
- **Defensible Claim 3**: The benefit of context-adaptive routing is strongly regime-dependent: CAEG-Net provides modest improvements during calm operational periods (+7.54 MW in low disagreement) but delivers outsized gains during high-stress operational regimes (+27.78 MW in high expert disagreement, an 8.79% error reduction over the strongest standalone expert).
- **Defensible Claim 4**: Individual context features exhibit heterogeneous importance: ablation experiments prove that diurnal periodicity ($+3.74\text{ MW}$ degradation when removed) and recent forecast error ($+1.05\text{ MW}$ degradation) are the primary drivers of routing accuracy.

---

## 5. What Claims We Must NOT Make (Unsubstantiated Overclaims)
- **Do NOT claim** to be the "first dynamic MoE for load forecasting" or the "first adaptive ensemble". Dynamic weighting has been studied for decades.
- **Do NOT claim** "state-of-the-art" across all international load benchmarks. We have rigorously validated performance on modern PJM operational data across 5 seeds; cross-dataset universality remains to be proven.
- **Do NOT claim** that gating correlations imply causality. When routing weights correlate with context variables (e.g. $r = +0.40$ between trend and CNN weight), describe these as empirical associations or predictive routing preferences, not direct causal control.
- **Do NOT claim** that calendar features universally improve load routing. Our empirical results clearly show that feeding raw cyclic calendar features degraded performance ($258.02 \to 270.78\text{ MW}$ MAE), providing an honest negative result that the 168h lookback already captures cyclical patterns more effectively.
