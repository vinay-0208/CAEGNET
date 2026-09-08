# CAEG-Net Phase 5: Bounded / Conservative Routing Design Specification
**Model**: `CAEG-Net BR` (Bounded Routing Mixture of Experts)  
**Track**: `research-track` | **Phase**: Phase 5 Optimization & Validation  
**Date**: September 2026  
**Status**: APPROVED DESIGN SPECIFICATION  

---

## 1. Executive Summary & Scientific Motivation

### 1.1 The Benchmark Challenge
Across rigorous multi-seed testing on 1,272 test hours ($K=53$ non-overlapping daily blocks), the **Static Standalone Equal Ensemble** ($w_0 = [1/3, 1/3, 1/3]$) established an empirical accuracy benchmark:
$$\text{MAE}_{\text{Equal}} = 232.51 \pm 3.48\text{ MW}$$

Previous unconstrained and decoupled mixture-of-experts models suffered from router estimation variance:
- **CAEG-Net V2 Full (Joint MoE)**: $\text{MAE} = 239.67 \pm 5.26\text{ MW}$ ($+7.16\text{ MW}$ vs Equal)
- **Decoupled CAEG (Unconstrained)**: $\text{MAE} = 239.53 \pm 10.91\text{ MW}$ ($+7.02\text{ MW}$ vs Equal)
- **CAEG-Net SR ($\lambda_{\text{dev}}=0.025$)**: $\text{MAE} = 233.75 \pm 6.95\text{ MW}$ ($+1.24\text{ MW}$ vs Equal)

While **CAEG-Net SR** demonstrated that shrinking router deviations toward equal weighting closed over $82\%$ of the gap and provided a statistically significant $+3.49\text{ MW}$ advantage during high-difficulty regimes, its soft exponential parameterization ($\text{softmax}(\log w_0 + (\alpha/\tau)\delta)$) still allowed the router to shift weights non-trivially.

### 1.2 The Bounded Routing (BR) Hypothesis
**Hypothesis BR**:
> *By explicitly bounding the router's convex mixture convexly between the uninformative equal ensemble $w_0$ and the adaptive router distribution $q_t$, we can directly control maximum deviation, guarantee exact convergence to the equal ensemble at $\rho=0$, and discover whether a small, bounded adaptive correction can outperform the $232.51\text{ MW}$ static benchmark.*

---

## 2. Mathematical Formulation

### 2.1 The Bounded Convex Router
Let $w_0 = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]^T$ be the equal-weight prior.

Let $u_t = [C_t \parallel D_t] \in \mathbb{R}^9$ be the causal router input combining 6D observable context and 3D detached inter-expert disagreement metrics.

The router network maps $u_t$ to raw routing logits $\delta_t \in \mathbb{R}^3$:
$$e_t = \text{ReLU}(\text{Linear}(\text{ReLU}(\text{LayerNorm}(\text{Linear}(u_t)))))$$
$$\delta_t = \text{Linear}(\text{Dropout}(\text{ReLU}(\text{Linear}(e_t))))$$

The unconstrained adaptive probability distribution is:
$$q_t = \text{softmax}\left(\frac{\delta_t}{\tau}\right) \in \Delta^2, \quad \tau = 1.0$$

The **Bounded Routing weights** $w_t \in \Delta^2$ are formed via explicit convex interpolation:
$$w_t = (1 - \rho) \cdot w_0 + \rho \cdot q_t, \quad \rho \in [0, 1]$$

### 2.2 Mathematical Properties of Bounded Routing
1. **Convexity & Simplex Preservation**:
   Since $w_0 \in \Delta^2$ and $q_t \in \Delta^2$, and $\rho \in [0, 1]$, any convex combination $w_t = (1-\rho)w_0 + \rho q_t$ strictly satisfies:
   $$\sum_{i=1}^3 w_{t,i} = (1 - \rho)\sum_{i=1}^3 \frac{1}{3} + \rho \sum_{i=1}^3 q_{t,i} = (1 - \rho)(1) + \rho(1) = 1.0$$
   $$w_{t,i} \ge (1 - \rho)\frac{1}{3} + \rho(0) = \frac{1 - \rho}{3} \ge 0 \quad \forall i$$
2. **Exact Identity at $\rho = 0$**:
   $$\lim_{\rho \to 0} w_t = w_0 = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]^T$$
   The model output $\hat{y}_{\text{fused}, t} = \sum_{i=1}^3 w_{t,i} \hat{y}_{i,t} \equiv \frac{1}{3}(\hat{y}_1 + \hat{y}_2 + \hat{y}_3)$ identically.
3. **Strict Deviation Bound**:
   The maximum possible $L_1$ deviation from equal weighting is bounded by $\rho$:
   $$\|w_t - w_0\|_1 = \sum_{i=1}^3 \left| (1 - \rho)\frac{1}{3} + \rho q_{t,i} - \frac{1}{3} \right| = \rho \sum_{i=1}^3 \left| q_{t,i} - \frac{1}{3} \right| = \rho \cdot \|q_t - w_0\|_1$$
   Since $\|q_t - w_0\|_1 \le \frac{4}{3}$, we have $\|w_t - w_0\|_1 \le \frac{4}{3}\rho$.
   - For $\rho = 0.10$, $\|w_t - w_0\|_1 \le 0.133$, and individual weights are strictly bounded in $[0.300, 0.400]$.
   - For $\rho = 0.20$, $\|w_t - w_0\|_1 \le 0.267$, and individual weights are strictly bounded in $[0.267, 0.467]$.
4. **Smooth Gradient Propagation**:
   $$\frac{\partial \hat{y}_{\text{fused}, t}}{\partial \delta_t} = \rho \cdot \frac{\partial q_t}{\partial \delta_t} \mathbf{\hat{y}}_t$$
   Gradients to the router scale linearly with $\rho$. At $\rho=0$, gradients are identically zero; as $\rho$ increases, the router experiences smooth, well-conditioned learning dynamics.

---

## 3. Loss Function & Regularization

The tripartite loss function for training the bounded router is:
$$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{y}_{\text{fused}}, y_{\text{true}}) + \lambda_{\text{dev}} \cdot D_{\mathrm{KL}}(w_t \parallel w_0)$$

where the analytical Kullback-Leibler divergence from the uniform distribution $w_0$ is:
$$D_{\mathrm{KL}}(w_t \parallel w_0) = \sum_{i=1}^3 w_{t,i} \ln\left(\frac{w_{t,i}}{1/3}\right) = \sum_{i=1}^3 w_{t,i} \ln(3 w_{t,i})$$

### Regularization Regimes:
- **Variant A (Bounded Only)**: $\lambda_{\text{dev}} = 0.0$ (pure bounded interpolation via $\rho$).
- **Variant B (Bounded + KL)**: $\lambda_{\text{dev}} = 0.025$ (bounded interpolation with explicit deviation entropy penalty).

---

## 4. Rigorous Temporal Causality & Leakage Audit

A comprehensive audit was performed across the data pipeline and model definitions:

| Pipeline Component | Temporal Lookback Window | Implementation Guarantee | Leakage Status |
| :--- | :--- | :--- | :--- |
| **Chronological Split** | Train (70%), Val (15%), Test (15%) | Strict ordering $t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$. No random shuffling across partitions. | **ZERO LEAKAGE** |
| **StandardScaler** | Train-only fit | Scaler mean and standard deviation computed strictly on `train_df['load']`. Applied identically to Val and Test. | **ZERO LEAKAGE** |
| **Model History Input ($X_t$)** | $[t-167 : t]$ (168h) | Contains strictly observed load values up to origin $t$. | **ZERO LEAKAGE** |
| **Context Buffer ($X_{\text{ctx},t}$)** | $[t-191 : t]$ (192h) | Context features (trend, volatility, range, diurnal, weekly) computed strictly on $X_{\text{ctx},t}$. Boundary prepending ensures exact history without consulting targets. | **ZERO LEAKAGE** |
| **Expanding Ridge Error** | $[t-23 : t]$ (24h) | Forecast made at origin $s=t-24$ predicting $[t-23:t]$ evaluated at time $t$ when actuals are observed. Out-of-sample walk-forward for training, frozen baseline for Val/Test. | **ZERO LEAKAGE** |
| **Expert Disagreement** | $D_t = \text{diff}(\hat{y}_1, \hat{y}_2, \hat{y}_3)$ | Forecasts $\hat{y}_i$ generated from $X_t$ (past $\le t$). Autograd detached before concatenation with $C_t$. | **ZERO LEAKAGE** |
| **Expert Backbones** | Pretrained & Frozen | Pretrained independently on training partition. Evaluated in `eval()` mode with `requires_grad=False`. Router learns on frozen predictions. | **ZERO LEAKAGE** |

---

## 5. Validation Selection Protocol

To prevent test-set overfitting and optimistic bias:
1. **Candidate Values**:
   $$\rho \in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]$$
   Evaluated with $\lambda_{\text{dev}} \in \{0.0, 0.025\}$ across 5 seeds.
2. **Primary Selection Metric**: Validation MAE (MW).
3. **Secondary Validation Metrics**: Validation RMSE, $R^2$, routing entropy, mean $L_1$ deviation from equal, max weight.
4. **Tie-Breaking Rule**:
   If two or more configurations yield validation MAE within $0.25\text{ MW}$ of each other, **the configuration with the smaller $\rho$ is strictly selected** because it is simpler, more conservative, and closer to the equal-weight prior.
5. **Locking Protocol**:
   The optimal $(\rho^*, \lambda_{\text{dev}}^*)$ is locked strictly before touching the test set.

---

## 6. Verification and Testing Suite

### Unit & Architecture Tests (`research/tests/test_phase5_bounded_routing.py`):
1. `test_rho_zero_exact_identity`: Verify $\hat{y}_{\text{BR}}(\rho=0) \equiv \hat{y}_{\text{Equal}}$ to $10^{-6}$ precision.
2. `test_rho_one_matches_unconstrained`: Verify $\hat{y}_{\text{BR}}(\rho=1) \equiv \hat{y}_{\text{D-CAEG}}$.
3. `test_simplex_sum_and_bounds`: Verify $\sum w_i = 1.0$ and $w_i \ge (1-\rho)/3$ for all $\rho$.
4. `test_gradient_isolation`: Verify backbones have zero gradients; only router parameters receive gradients.
5. `test_kl_divergence_properties`: Verify $D_{\mathrm{KL}} \ge 0$ and $D_{\mathrm{KL}}(\rho=0) = 0.0$.
6. `test_monotonic_deviation`: Verify mean $\|w_t - w_0\|_1$ strictly increases monotonically with $\rho$.
7. `test_parameter_accounting`: Verify 139,838 frozen and 2,595 trainable parameters.
8. `test_causal_determinism`: Verify reproducible outputs given identical seed.

---

## 7. Statistical Evaluation Protocol

Evaluations on untouched test data ($1,272$ hours) use the canonical $K=53$ non-overlapping daily blocks:
- **Primary Hypothesis**: CAEG-Net BR beats Static Equal Ensemble.
- **Statistical Tests**:
  - Two-sided paired $t$-test ($t$, $p$).
  - Wilcoxon signed-rank test ($W$, $p$).
  - Diebold-Mariano test with lag-1 autocorrelation correction.
  - Cohen's $d$ effect size and 95% bootstrap confidence intervals.
  - Holm-Bonferroni correction across multi-model comparisons (BR vs Equal, BR vs SR, BR vs D-CAEG, BR vs TCN, BR vs V2).
