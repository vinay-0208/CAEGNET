# PHASE 9 — UCI ELECTRICITY LOAD DIAGRAMS: DATASET INSPECTION & GENERALIZATION PROTOCOL

**Document Version:** 2.0.0 (Phase 9A Corrected)  
**Date:** September 9, 2026  
**Repository Branch:** `research-track`  
**Status:** Protocol Corrected & Established — Awaiting Final Approval Before Neural Benchmark  

---

## Executive Summary & Phase 9A Corrections

This document establishes the formal dataset inspection, client onboarding audit, fixed-cohort selection, and experimental protocol for **Phase 9: Cross-Dataset Generalization on the UCI ElectricityLoadDiagrams20112014 Benchmark**.

In strict accordance with the canonical CAEG-Net research roadmap, this phase performs **dataset inspection, onboarding audit, and protocol design only**. No neural model training, architecture exploration, hyperparameter tuning, or test-set optimization is performed.

### Critical Phase 9A Methodological Corrections Applied:
1. **Resolution of Non-Stationary Onboarding Flaw:** The initial protocol assumed all 370 clients were stationary across 2012–2014. Detailed audit revealed that 49 clients were onboarded gradually throughout 2012–2014, causing an artificial $+5.70\%$ drift in the all-370 aggregate load due to meter installation rather than grid demand.
2. **Establishment of an Invariant Fixed Cohort (Cohort 320):** We define and implement an objective inclusion rule ($t_{\text{first\_active}} \le \text{2012-01-01 00:15:00}$) establishing a strictly constant cohort of $N = 320$ clients whose observations are continuously available throughout 2012–2014 ($26,304\text{ hours}$), with zero onboarding composition changes.
3. **Verification of Hour-Ending Timestamp Convention:** Verified that four 15-minute readings (`00:15`, `00:30`, `00:45`, `01:00`) represent the hour ending at `01:00:00`.
4. **Empirical DST Transition Audit:** Verified March 1-hour drops (artificial zero-filling) and October 1-hour double-counting, confirming that daily interval counts remain strictly invariant at 96 intervals/day.

---

## 1. Dataset Provenance

* **Dataset Name:** ElectricityLoadDiagrams20112014
* **Source Archive:** UCI Machine Learning Repository (Dataset ID: 321)
* **DOI:** [10.24432/C58C86](https://doi.org/10.24432/C58C86)
* **Donor:** Artur Trindade (2015), University of Porto / INESC TEC, Portugal
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Canonical Literature References:** 
  - Trindade, A. (2015). *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository.
  - Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2018). *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (LSTNet)*. SIGIR 2018.
  - Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*. International Journal of Forecasting.

---

## 2. Raw File Description

* **Repository Location:** `data/ElectricityLoadDiagrams20112014/LD2011_2014.txt`
* **File Size:** $710,998,915$ bytes ($678.06\text{ MB}$)
* **SHA-256 Checksum:** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
* **File Encoding:** UTF-8
* **Delimiter:** Semicolon (`;`)
* **Header Structure:** 371 columns:
  - First column: Timestamp header (empty quoted string `""`)
  - Columns 2 through 371: Client identifiers `MT_001` through `MT_370`

> [!IMPORTANT]
> The raw dataset file remains completely unaltered. In accordance with Step 16 git discipline, the `data/ElectricityLoadDiagrams20112014/` path is strictly excluded via `.gitignore`. No metadata from `__MACOSX` is utilized.

---

## 3. Dataset Structure & Dimensions

* **Total Raw Rows:** $140,256$ observation rows (plus 1 header row)
* **Total Columns:** $371$ ($1$ timestamp + $370$ client series)
* **Total Value Cells:** $140,256 \times 370 = 51,894,720$ individual measurements
* **Calendar Duration:** $1,461$ full calendar days ($4$ complete Gregorian years: 2011, 2012 [leap year], 2013, 2014)
* **Daily Interval Structure:** Exactly $96$ fifteen-minute intervals per calendar day ($96 \times 1,461 = 140,256$)
* **Total Missing (NaN) Values:** Exactly $0$ ($0.0\%$)
* **Total Negative Values:** Exactly $0$ ($0.0\%$)

---

## 4. Client Onboarding Audit & Zero Structure

Our audit tracked the exact activation timestamp (first non-zero measurement) and zero distributions for all 370 clients:

### Cumulative Active Clients Over Time:
* **`2011-01-01 00:15:00` (Dataset Start):** $158$ clients active
* **`2011-12-31 23:45:00` (End of 2011):** $160$ clients active ($+2$ clients added in 2011: `MT_164` on Feb 22, `MT_280` on Apr 7)
* **`2012-01-01 00:15:00` (Start of 2012):** $\mathbf{320\text{ clients active}}$ ($+160$ clients activated simultaneously at row $35,040$)
* **`2012-01-08 00:00:00` (Early Jan 2012):** $\mathbf{321\text{ clients active}}$ ($+1$ client, `MT_146`, activated on `2012-01-07 16:30:00`)
* **`2012-12-31 23:45:00` (End of 2012):** $333$ clients active ($+12$ clients added between March and Nov 2012)
* **`2013-12-31 23:45:00` (End of 2013):** $351$ clients active ($+18$ clients added in 2013)
* **`2014-12-31 23:45:00` (End of 2014):** $370$ clients active ($+19$ clients added in 2014)

### Zero Structure & Client Creation:
* $212$ clients exhibit solid leading zero blocks before their first non-zero observation.
* As documented by Artur Trindade (2015): *"Some clients were created after 2011. In these cases consumption were considered zero."*
* Once a client activates, their zero frequency drops from $100\%$ to a median of **$0.01\%$** (mean $1.19\%$, $95\text{th}$ percentile $0.13\%$).
* **Conclusion:** Leading zeros represent non-existent/uncreated meters prior to installation, NOT periods of zero consumption by an established customer.

---

## 5. Objective Fixed-Cohort Formulation

To avoid confounding structural load patterns with meter onboarding expansion, we evaluated four candidate cohort rules:

| Candidate Cohort | Size ($N$) | Inclusion Criterion | Stationarity of Membership | Methodological Assessment |
| :--- | :---: | :--- | :---: | :--- |
| **Option A: All 370 Clients** | $370$ | None (all columns) | **Non-Stationary** | Flawed: 49 clients onboarded dynamically across 2012–2014, inducing artificial $+5.70\%$ upward drift. |
| **Option B: Fixed Cohort 320 (RECOMMENDED)** | $\mathbf{320}$ | $t_{\text{first\_active}} \le \text{2012-01-01 00:15:00}$ | **Strictly Stationary** | **Optimal:** 100% of clients are active at or before day 1 hour 1 of 2012. Invariant membership across all 26,304 hours of 2012–2014. |
| **Option C: Canonical Literature Cohort 321** | $321$ | $t_{\text{first\_active}} \le \text{2012-01-08 00:00:00}$ | Near-Stationary | Established in literature (LSTNet, DeepAR). Differs from Cohort 320 by exactly 1 client (`MT_146`), which was inactive Jan 1–7. |
| **Option D: Full 4-Year Cohort 160** | $160$ | $t_{\text{first\_active}} \le \text{2011-12-31 23:45:00}$ | Stationary (2011–2014) | Discards $210$ clients ($56.8\%$ of dataset); load pool is substantially smaller. |

### Formal Recommendation: Option B — Fixed Cohort 320
* **Inclusion Rule:** $\text{client\_id} \in \{ c \mid t_{\text{first\_active}}(c) \le \text{2012-01-01 00:15:00} \}$
* **Cohort Size:** Exactly **$320$ clients**.
* **Cohort Audit Record:** Full client-by-client provenance is documented in `research/results/phase9_cohort_audit.csv`.
* **Zero Contamination:** Invariant membership across all $26,304$ hours of 2012–2014. Zero clients are added after the study period commences.
* **Coverage:** In 2012, Cohort 320 represents **$99.76\%$** of the total load of all 370 clients ($211.81\text{ MW}$ vs $212.32\text{ MW}$).

---

## 6. Study Period Selection (2012–2014)

* **Selected Study Period:** `2012-01-01 00:15:00` to `2014-12-31 24:00:00` (timestamped as `2015-01-01 00:00:00`).
* **Duration:** Exactly $3$ full Gregorian years ($1,096\text{ days}$, including leap year 2012), producing exactly **$26,304\text{ hourly intervals}$**.
* **Scientific Justification:**
  1. Year 2011 has only 160 active clients. On Jan 1, 2012, 160 new clients were created simultaneously, producing a massive non-stationary step jump in total load from ~122 MW to ~212 MW.
  2. Commencing the benchmark on Jan 1, 2012 with Fixed Cohort 320 eliminates the 2011 creation step while preserving a massive continuous 3-year forecasting horizon.
  3. $26,304$ hours provides ample data for chronological partitioning ($18,413$ train, $3,945$ val, $3,946$ test hours).

---

## 7. Descriptive Comparison: All 370 vs Fixed Cohort 320 (2012–2014)

| Statistic | All 370 Clients Aggregate | Fixed Cohort 320 Aggregate | Structural Interpretation |
| :--- | :---: | :---: | :--- |
| **Number of Clients** | $370$ (dynamic $320 \to 370$) | **$320$ (strictly invariant)** | Fixed cohort avoids composition changes |
| **Hourly Observations** | $26,304\text{ hours}$ | $26,304\text{ hours}$ | Identical temporal grid |
| **Overall Mean Load** | $219.82\text{ MW}$ | $203.50\text{ MW}$ | Difference is the load of the 50 late-onboarded clients |
| **Overall Std Dev** | $81.63\text{ MW}$ | $79.73\text{ MW}$ | Stable variability |
| **Minimum Load** | $33.10\text{ MW}$ | $28.07\text{ MW}$ | Preserves realistic non-zero base load |
| **Median Load** | $224.98\text{ MW}$ | $210.37\text{ MW}$ | Consistent distribution |
| **Maximum Load** | $447.58\text{ MW}$ | $446.84\text{ MW}$ | Peak demand preserved |
| **2012 Annual Mean** | $212.32\text{ MW}$ | $211.81\text{ MW}$ | **$99.76\%$ agreement** at baseline year |
| **2013 Annual Mean** | $222.74\text{ MW}$ | $200.49\text{ MW}$ | All-370 rises due to +18 clients; Cohort 320 reflects true grid |
| **2014 Annual Mean** | $224.41\text{ MW}$ | $198.17\text{ MW}$ | All-370 rises due to +19 clients; Cohort 320 reflects true grid |
| **2-Year Net Change** | **$+5.70\%$ (artificial meter growth)** | **$-6.44\%$ (true macroeconomic stabilization)** | **Critical distinction resolved** |

---

## 8. 15-Minute $\to$ Hourly Transformation Decision & Formula

* **Semantic Meaning:** Raw values represent average active power demand in kW over a 15-minute interval ($\Delta t = 0.25\text{ h}$).
* **Energy Equivalence:** Energy in 15 minutes is $E_i = P_i \times 0.25\text{ kWh}$.
* **Hourly Aggregation Formula:**
  For any hour $h$ comprising four consecutive 15-minute intervals $i \in \{1, 2, 3, 4\}$:
  $$P_{\text{hour}} = \frac{\sum_{i=1}^4 (P_i \times 0.25\text{ h})}{1.0\text{ h}} = \frac{P_1 + P_2 + P_3 + P_4}{4}\quad (\text{kW})$$
* **System Aggregate Load:**
  For Fixed Cohort 320 ($C_{320}$):
  $$L_t^{\text{MW}} = \frac{1}{1000.0} \sum_{c \in C_{320}} P_{t, c}^{\text{hour}}$$

---

## 9. Hourly Timestamp Labeling Convention (Hour-Ending)

* **Raw Sequence:** Quarters are recorded at `00:15`, `00:30`, `00:45`, `01:00`.
* **Convention:** **Hour-Ending (HE).**
  - Quarters `00:15`, `00:30`, `00:45`, `01:00` are averaged and assigned the timestamp `01:00:00` (representing the completed hour ending at 01:00).
  - Quarters `23:15`, `23:30`, `23:45`, `00:00` are averaged and assigned the timestamp `00:00:00` next day (representing Hour 24).
* **First Hourly Timestamp (2012–2014 Study Period):** `2012-01-01 01:00:00`.
* **Last Hourly Timestamp:** `2015-01-01 00:00:00` (Hour 24 of `2014-12-31`).
* **Total Hourly Count:** Exactly $26,304\text{ hours}$ ($1,096\text{ days} \times 24\text{ h/day}$).

---

## 10. Daylight Saving Time (DST) Audit Findings

Our audit investigated the exact raw intervals around all March and October DST transition dates across 2011–2014:

1. **March Spring-Forward (23-Hour Civil Day):**
   - In civil time, the hour 01:00 to 02:00 does not exist.
   - *Empirical Finding:* The data provider inserted 4 zero (or near-zero) readings between `01:00:00` and `01:45:00` (e.g. on 2012-03-25, 2013-03-31, 2014-03-30, aggregate power drops to $0.4\text{--}4.4\text{ MW}$).
   - *Purpose:* Preserves exactly 96 intervals per day so every day has invariant 24-hour length.
2. **October Fall-Back (25-Hour Civil Day):**
   - In civil time, the hour 01:00 to 02:00 repeats twice.
   - *Empirical Finding:* The data provider aggregated both hours together into the 4 intervals between `01:00:00` and `01:45:00` (e.g. aggregate power roughly doubles to $240\text{--}275\text{ MW}$ vs the normal $120\text{--}135\text{ MW}$).
3. **Forecasting Policy:** Retain these rows unaltered. Because each day has exactly 96 rows, calendar continuity is strictly preserved without ad-hoc manual row deletion.

---

## 11. Final Decision on Client-Level vs Aggregate Forecasting

### Why are we forecasting a single system aggregate series rather than 370 independent client series?
1. **Core Scientific Question:** CAEG-Net is formulated for short-term *system-level* electricity load forecasting (combining temporal experts dynamically based on macroscopic regime indicators). Forecasting aggregate microgrid load evaluates the exact same scientific question tested on Modern PJM and GEFCom2014.
2. **Cross-Dataset Comparability:** Modeling the macro-aggregate maintains 1-to-1 parity in task definition ($168\text{h} \to 24\text{h}$ multi-step point forecasting), evaluation metrics, and baselines across all three research phases.
3. **Microscopic vs Macroscopic Dynamics:** Individual meters exhibit high intermittency, discrete switching, and substantial vacancy periods, which constitute an entirely different research domain (hierarchical load aggregation / disaggregation).

---

## 12. Chronological Partitioning & 168 $\to$ 24 Window Protocol

* **Total Horizon:** $26,304\text{ hours}$
* **Chronological 70/15/15 Partitions:**
  - **Train ($70\%$):** $18,413\text{ hours}$ (`2012-01-01 01:00:00` to `2014-02-07 05:00:00`)
  - **Validation ($15\%$):** $3,945\text{ hours}$ (`2014-02-07 06:00:00` to `2014-07-21 14:00:00`)
  - **Test ($15\%$):** $3,946\text{ hours}$ (`2014-07-21 15:00:00` to `2015-01-01 00:00:00`)
* **Sliding Window Dimensions:**
  - Input $X_t$: previous $168$ hourly observations ($[t-167 : t]$), shape `[N, 168, 1]`.
  - Target $Y_t$: next $24$ hourly observations ($[t+1 : t+24]$), shape `[N, 24]`.
  - Step size: $1\text{ hour}$ rolling origin.
  - Causal lookback prepending: Validation and test partitions prepend the trailing $167$ observations from the prior partition to preserve 100% of forecast targets without leakage.

---

## 13. Scaling Protocol

* **Scaler:** `StandardScaler` (zero mean, unit variance).
* **Fitting Rule:** Fitted **strictly on the training partition**:
  $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{t \in \text{train}} L_t, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{t \in \text{train}} (L_t - \mu_{\text{train}})^2}$$
* **Transformation:** Validation and test partitions are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
* **Evaluation:** All evaluation metrics (MAE, RMSE, MAPE) are calculated on the original unscaled MW scale.

---

## 14. Canonical CAEG Context Features

The canonical 4D dynamic context vector $c_t \in \mathbb{R}^4$ is extracted strictly causally:
1. **Trend ($c_{t, 1}$):** Normalized linear regression slope over the 168h lookback.
2. **Volatility ($c_{t, 2}$):** Standard deviation over the 168h lookback.
3. **Lag-24 Periodicity ($c_{t, 3}$):** Autocorrelation at lag 24 within the 168h lookback.
4. **Recent Forecast Error ($c_{t, 4}$):** Causal out-of-sample forecast error from a completed 24-step forecast concluding at or before origin $t$ (derived via walk-forward multi-step Ridge regression).

---

## 15. Planned Benchmark Models & Metrics

### Benchmark Suite (5 Models, 5 Seeds):
1. **Standalone LSTM:** Canonical 2-layer LSTM ($64$ hidden units).
2. **Standalone TCN:** Canonical Dilated TCN ($64$ channels, kernel $3$, dilations $[1, 2, 4, 8]$).
3. **Standalone CNN:** Canonical Multi-layer 1D CNN ($64$ filters, kernel $3$).
4. **Static Equal Ensemble:** Uniform combination ($\frac{1}{3} \hat{y}_{\text{LSTM}} + \frac{1}{3} \hat{y}_{\text{TCN}} + \frac{1}{3} \hat{y}_{\text{CNN}}$).
5. **Original CAEG-Net V1:** Canonical architecture with LSTM + TCN + CNN experts, Context Encoder, Adaptive Soft Gate, and Weighted Fusion.

### Research Metrics:
* **Primary Metric:** **Mean Absolute Error (MAE)** in MW.
* **Training Objective:** Mean Squared Error (MSE) loss.
* **Checkpoint Criterion:** Validation MSE (consistent with Phase 6A audit).
* **Secondary Metrics:** RMSE, MSE, $R^2$, MAPE.

---

## 16. Leakage Safeguards & Integrity Audit

1. **Strictly Chronological Splits:** No random sampling or cross-validation shuffling.
2. **Train-Only Scaling:** Zero test/validation target information enters scaling.
3. **Causal Context Boundaries:** Origin $t$ features consult strictly observations $\le t$.
4. **Out-of-Sample Recent Error:** Completed walk-forward historical forecasts only.
5. **Read-Only Dataset:** Raw file is never modified; `.gitignore` protects data.

---

## 17. Reproducibility & Test Suite Verification

All protocol rules are enforced by unit tests in `research/tests/test_phase9_uci_protocol.py`:
```text
research/tests/test_phase9_uci_protocol.py:
  test_01_fixed_cohort_membership_deterministic ..... OK
  test_02_selected_clients_satisfy_inclusion_rule .... OK
  test_03_selected_cohort_remains_constant ........... OK
  test_04_no_client_outside_cohort_contributes ....... OK
  test_05_each_hourly_aggregate_contains_four_intervals OK
  test_06_hourly_timestamp_labeling_deterministic .... OK
  test_07_no_duplicate_hourly_timestamps ............. OK
  test_08_chronological_split ........................ OK
  test_09_no_train_test_leakage ...................... OK
  test_10_train_only_scaler .......................... OK
  test_11_causal_context ............................. OK

Ran 11 tests in 2.917s
OK
```
