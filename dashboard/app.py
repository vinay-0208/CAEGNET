"""
CAEG-Net Interactive Results Dashboard (Streamlit)
==================================================
Visualizes authoritative Phase 15 final results, baseline comparisons,
learned routing allocations, confidence parameters, and horizon curves.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
cur = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(cur, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

st.set_page_config(
    page_title="CAEG-Net Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ CAEG-Net: Context-Adaptive Expert Gating Network")
st.subheader("Short-Term Electricity Load Forecasting — Final Locked Model Dashboard")

st.markdown("""
> **Model Status:** LOCKED_FINAL_MODEL  
> **Architecture Class:** `ConfidenceFallbackCAEGNet` (`F2_A2_OOF`)  
> **Trainable Parameters:** 121,724 (<0.5 MB)  
> **Input / Output:** 168h Lookback -> 24h Day-Ahead Forecast  
""")

# Load authoritative results
results_csv = os.path.join(repo_root, "research", "results", "final_results.csv")
horizon_csv = os.path.join(repo_root, "research", "analysis", "phase15b_horizon_results.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary",
    "🧠 Architecture & Parameters",
    "📈 Multi-Grid Benchmarks",
    "⏱️ 24-Hour Horizon Analysis",
    "⚖️ Statistical Significance"
])

with tab1:
    st.header("Executive Summary: 5-Seed Benchmark Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PJM Regional Grid", "250.97 MW MAE", "RMSE: 335.38 MW | R2: 0.8714")
    with col2:
        st.metric("GEFCom2014 Zonal Grid", "12.41 kW MAE", "RMSE: 18.04 kW | R2: 0.8610")
    with col3:
        st.metric("UCI Electricity Cohort", "7.74 MW MAE", "RMSE: 10.96 MW | R2: 0.9831")

    st.markdown("---")
    st.markdown("### Routing & Confidence Summary")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("""
        **Learned Expert Allocations (Mean Weights):**
        - **PJM:** LSTM: 0.3511 | TCN: 0.3283 | CNN: 0.3207 ($N_{\\text{eff}} = 2.9931$)
        - **GEFCom:** LSTM: 0.3528 | TCN: 0.2982 | CNN: 0.3490 ($N_{\\text{eff}} = 2.9765$)
        - **UCI:** LSTM: 0.3428 | TCN: 0.2994 | CNN: 0.3578 ($N_{\\text{eff}} = 2.9842$)
        """)
    with r_col2:
        st.markdown("""
        **Learned Confidence Parameter $\\lambda$:**
        - **PJM:** $0.5066 \\pm 0.0038$ ($CV = 0.75\\%$)
        - **GEFCom:** $0.5170 \\pm 0.0055$ ($CV = 1.06\\%$)
        - **UCI:** $0.5064 \\pm 0.0030$ ($CV = 0.59\\%$)
        - *Interpretation:* Stable regularizing anchor toward equal-expert centroid.
        """)

with tab2:
    st.header("Parameter Distribution Breakdown")
    params = pd.DataFrame([
        {"Component": "LSTM Expert", "Type": "Recurrent", "Parameters": 56152, "Share": "46.1%"},
        {"Component": "TCN Expert", "Type": "Dilated Causal", "Parameters": 36952, "Share": "30.4%"},
        {"Component": "CNN Expert", "Type": "Multi-scale Ramp", "Parameters": 27400, "Share": "22.5%"},
        {"Component": "Context Router", "Type": "Gating MLP", "Parameters": 1075, "Share": "0.9%"},
        {"Component": "Confidence Head", "Type": "Shrinkage MLP", "Parameters": 145, "Share": "0.1%"},
        {"Component": "Total CAEG-Net", "Type": "End-to-End Champion", "Parameters": 121724, "Share": "100.0%"}
    ])
    st.table(params.set_index("Component"))

with tab3:
    st.header("Authoritative Multi-Grid Benchmark Comparison")
    if os.path.exists(results_csv):
        df_res = pd.read_csv(results_csv)
        st.dataframe(df_res, use_container_width=True)
    else:
        st.info("final_results.csv located at research/results/final_results.csv")

with tab4:
    st.header("Step-by-Step Horizon Degradation (h=1 to 24)")
    if os.path.exists(horizon_csv):
        df_h = pd.read_csv(horizon_csv)
        sub_f2 = df_h[df_h["candidate_id"] == "Control_A_F2"]
        dataset_choice = st.selectbox("Select Grid Dataset", ["PJM", "GEFCom", "UCI"])
        grid_data = sub_f2[sub_f2["dataset"] == dataset_choice]
        st.line_chart(grid_data.set_index("horizon_step")["mae"])
    else:
        st.info("Horizon results CSV located at research/analysis/phase15b_horizon_results.csv")

with tab5:
    st.header("Statistical Significance (Daily Blocks)")
    stats = pd.DataFrame([
        {"Dataset": "PJM", "Blocks": 53, "Mean Diff": "-9.66 MW", "t-stat": -2.56, "p-value (t)": 0.0135, "Wilcoxon p": 0.0298, "Holm-Adj p": 0.0406, "Significance": "Significant"},
        {"Dataset": "GEFCom", "Blocks": 456, "Mean Diff": "-0.688 kW", "t-stat": -11.85, "p-value (t)": "2.04e-28", "Wilcoxon p": "2.53e-29", "Holm-Adj p": "1.02e-27", "Significance": "Significant"},
        {"Dataset": "UCI", "Blocks": 163, "Mean Diff": "-0.202 MW", "t-stat": -3.66, "p-value (t)": 0.00034, "Wilcoxon p": 0.00005, "Holm-Adj p": 0.0017, "Significance": "Significant"}
    ])
    st.table(stats.set_index("Dataset"))
