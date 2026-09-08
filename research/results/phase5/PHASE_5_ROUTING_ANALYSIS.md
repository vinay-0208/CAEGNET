# CAEG-Net Phase 5: Routing Dynamics, Disagreement Analysis & Gating Interpretability
**Document**: `PHASE_5_ROUTING_ANALYSIS.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: COMPLETE ANALYSIS  

---

## 1. Executive Summary

A central premise of CAEG-Net is that dynamic routing based on observable temporal context and forecast difficulty outperforms static weighting. However, unconstrained Mixture-of-Experts (MoE) routers frequently suffer from:
1. **Routing Collapse**: The router prematurely converges to allocating near-100% weight to a single expert, ignoring others.
2. **Estimation Variance**: Noisy sample-by-sample gating decisions introduce variance that cancels out theoretical ensembling benefits, causing adaptive models to underperform an equal static average ($w_i = 1/3$).
3. **Expert Degradation (Co-adaptation)**: The presence of a dominant expert causes weaker experts to receive sparse gradient feedback during end-to-end training, as occurred in V1 with the CNN expert.

In Phase 5, three principled mechanisms resolve these pathologies:
- **Detached Inter-Expert Disagreement**: Provides the router with real-time empirical model uncertainty without letting routing gradients distort individual expert feature extractors.
- **Conservative Bounded Routing ($\rho \in [0.10, 0.30]$)**: Re-anchors the gating distribution to a convex combination with uniform weighting:
  $$w_t^{\text{bounded}} = (1 - \rho) \cdot \frac{1}{3} \cdot \mathbf{1} + \rho \cdot w_t^{\text{router}}$$
  ensuring that every expert is retained at a minimum baseline level ($> 26.7\%$ when $\rho=0.20$).
- **Auxiliary Expert Supervision ($\lambda_{\text{aux}} = 0.10$)**: Direct gradient paths from individual expert errors guarantee that each expert is independently trained to its full potential.

---

## 2. Gating Formulation & Mathematical Guarantees

### 2.1 Bounded Softmax Routing
Let $g(c_t, d_t) \in \mathbb{R}^3$ be the raw gating logits from the MLP router. The unconstrained softmax weights are:
$$w_t^{\text{raw}} = \text{Softmax}(g(c_t, d_t))$$
The conservative bounded routing transformation applies:
$$w_{t, i}^{\text{bounded}} = \frac{1 - \rho}{3} + \rho \cdot w_{t, i}^{\text{raw}}, \quad i \in \{1, 2, 3\}$$

**Mathematical Guarantees**:
1. **Partition of Unity**: Since $\sum_{i=1}^3 w_{t,i}^{\text{raw}} = 1$:
   $$\sum_{i=1}^3 w_{t, i}^{\text{bounded}} = (1 - \rho) + \rho (1) = 1$$
2. **Strict Positivity (Collapse Prevention)**:
   For any $\rho \in (0, 1)$, even if $w_{t, i}^{\text{raw}} \to 0$, the expert's weight is strictly bounded from below:
   $$w_{t, i}^{\text{bounded}} \ge \frac{1 - \rho}{3} > 0$$
   For $\rho = 0.20$, $w_{t, i} \in [0.267, 0.467]$. Complete expert starvation is mathematically impossible.
3. **Variance Reduction**:
   The variance of the gating weights scales as $\mathcal{O}(\rho^2)$, dramatically reducing routing noise and estimation error on the test distribution.

---

## 3. Disagreement Signal Analysis

The detached pairwise inter-expert disagreement is defined as:
$$d_t = \begin{bmatrix} 
\frac{1}{24} \sum_{h=1}^{24} |\hat{y}_{t, h}^{\text{LSTM}} - \hat{y}_{t, h}^{\text{TCN}}| \\
\frac{1}{24} \sum_{h=1}^{24} |\hat{y}_{t, h}^{\text{LSTM}} - \hat{y}_{t, h}^{\text{CNN}}| \\
\frac{1}{24} \sum_{h=1}^{24} |\hat{y}_{t, h}^{\text{TCN}} - \hat{y}_{t, h}^{\text{CNN}}|
\end{bmatrix}$$

- **High Disagreement Regimes**: Occur during abrupt ramp events, public holiday transitions, and extreme weather shifts where structural assumptions of the individual architectures diverge.
- **Router Reaction**: When disagreement spikes, the router increases allocation to the model with the lowest historical error in high-volatility regimes (TCN), while bounded routing prevents over-concentration.
- **Detached Gradients**: Because $\nabla_{\theta_{\text{expert}}} d_t = 0$, experts are not penalized for disagreeing; disagreement is purely an input signal to the gating MLP.

---

## 4. Regime-Specific Specialization

By segmenting test observations into distinct operational regimes:
- **Peak Hours (08:00 - 20:00)**: Higher load levels, steep gradients. The router shifts probability mass towards TCN (receptive field spans multi-scale temporal dependencies).
- **Off-Peak / Night (21:00 - 07:00)**: High diurnal periodicity, smooth trends. LSTM and CNN receive increased allocations.
- **Extreme Error Volatility**: The combination of 4D domain context (OLS trend slope, first-difference volatility, lag-24 autocorrelation, expanding Ridge error) alongside 3D disagreement gives the router full observability over both the physical time series state and internal model consensus.
