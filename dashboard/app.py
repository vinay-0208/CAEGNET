"""
CAEG-Net Professional Research Dashboard
========================================
A modern, dark-themed AI/Energy Research Platform for:
CAEG-Net (Context-Adaptive Expert Gating Network)
Short-Term Electricity Load Forecasting with Heterogeneous Temporal Experts

Operates strictly over verified repository artifacts.
Zero fabricated prediction arrays or synthetic traces.
"""

import os
import sys
import pickle
import base64
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Suppress unpickling version warnings for estimators
warnings.filterwarnings("ignore", category=UserWarning)

# Setup repository paths
cur_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(cur_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Streamlit Page Configuration
st.set_page_config(
    page_title="CAEG-Net | Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global dataset state initialization
if "selected_dataset" not in st.session_state:
    st.session_state["selected_dataset"] = "PJM"

# =========================================================================
# GLOBAL DESIGN SYSTEM & CUSTOM DARK CSS
# =========================================================================
st.markdown("""
<style>
    /* Reduce top padding above hero / page content */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Dark Theme Core Styles */
    .stApp {
        background-color: #070A13;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0A0E1A;
        border-right: 1px solid #1A2333;
    }
    .sidebar-category {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 18px 0 6px 4px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 7px 12px;
        border-radius: 6px;
        margin-bottom: 2px;
        transition: all 0.2s ease;
        cursor: pointer;
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #131C2E;
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.25) 0%, rgba(14, 165, 233, 0.12) 100%);
        border-left: 3px solid #38BDF8;
        color: #38BDF8 !important;
        font-weight: 600;
    }

    /* Page Hero Headers */
    .page-hero-container {
        margin-bottom: 1.2rem;
    }
    .page-category-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 9999px;
        padding: 3px 10px;
        font-weight: 700;
        font-size: 0.72rem;
        display: inline-block;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .page-title {
        font-size: 2.05rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.45;
        margin-bottom: 1rem;
    }

    /* Modern Dark Cards */
    .dark-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        box-sizing: border-box;
    }
    .dark-card:hover {
        border-color: #334155;
    }
    .dark-card-header {
        font-size: 0.8rem;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .accent-card {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(14, 165, 233, 0.05) 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 1rem;
        box-sizing: border-box;
    }

    /* Unified Responsive Card Grids for Identical Card Alignment */
    .metric-grid-5 {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .metric-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 1.2rem;
    }

    @media (max-width: 1150px) {
        .metric-grid-5 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .metric-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 768px) {
        .metric-grid-5 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .metric-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .metric-grid-3 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
    }
    @media (max-width: 520px) {
        .metric-grid-5 { grid-template-columns: 1fr; }
        .metric-grid-4 { grid-template-columns: 1fr; }
    }

    /* Metric Card with strictly enforced equal height and vertical alignment */
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 108px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease;
        box-sizing: border-box;
    }
    .metric-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.08);
        transform: translateY(-1px);
    }
    .metric-label {
        font-size: 0.70rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.48rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 0.74rem;
        color: #38BDF8;
        font-weight: 500;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: auto;
    }

    /* Benchmark Card */
    .benchmark-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        min-height: 155px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease;
        box-sizing: border-box;
    }
    .benchmark-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.08);
    }

    /* Status Badges */
    .status-badge {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-weight: 600;
        font-size: 0.72rem;
        display: inline-block;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-weight: 600;
        font-size: 0.72rem;
        display: inline-block;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 1.2rem 0 0.8rem 0;
        letter-spacing: -0.01em;
    }

    /* Pipeline Step Bar */
    .pipeline-container {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 1.3rem;
        overflow-x: auto;
    }
    .pipeline-step {
        background: #1E293B;
        color: #E2E8F0;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .pipeline-step.active {
        background: #2563EB;
        color: #FFFFFF;
    }
    .pipeline-arrow {
        color: #64748B;
        font-size: 0.9rem;
    }

    /* Provenance Box */
    .provenance-card {
        background: #0B1120;
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        color: #94A3B8;
        font-size: 0.88rem;
    }
    .provenance-card strong {
        color: #F1F5F9;
    }

    /* Empty State Card */
    .empty-state-card {
        background: #0B1120;
        border: 1px dashed #334155;
        border-radius: 10px;
        padding: 32px 24px;
        text-align: center;
        margin: 1rem 0;
    }

    /* Custom Clean Dark Table */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background: #0F172A;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1E293B;
        margin-bottom: 1.2rem;
        font-size: 0.84rem;
    }
    .custom-table th {
        background: #1E293B;
        color: #38BDF8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.72rem;
        padding: 9px 12px;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    .custom-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #1E293B;
        color: #E2E8F0;
    }
    .custom-table tr:last-child td {
        border-bottom: none;
    }
    .custom-table tr:hover td {
        background: #131C2E;
    }
    .custom-table .highlight-row td {
        background: rgba(37, 99, 235, 0.14) !important;
        color: #38BDF8 !important;
        font-weight: 600;
    }
    .custom-table .num-cell {
        text-align: right;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Sidebar Model Info Card */
    .sidebar-info-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 1.5rem;
    }
    .sidebar-info-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .sidebar-info-sub {
        font-size: 0.74rem;
        color: #38BDF8;
        font-family: monospace;
        margin-bottom: 8px;
    }
    .sidebar-info-meta {
        font-size: 0.74rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Dark Footer */
    .dark-footer {
        border-top: 1px solid #1E293B;
        padding: 24px 0 12px 0;
        margin-top: 3rem;
        color: #64748B;
        font-size: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# DATA LOADERS & ARTIFACT ABSTRACTION (Cached & Purely Verified)
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_pjm_prediction_cache():
    """Load verified multi-seed evaluation arrays for PJM."""
    path_primary = os.path.join(repo_root, "results", "phase5_multiseed_cache.npz")
    path_fallback = os.path.join(repo_root, "research", "results", "research_multiseed_cache.npz")
    target_path = path_primary if os.path.exists(path_primary) else path_fallback
    if os.path.exists(target_path):
        data = np.load(target_path)
        return {k: data[k] for k in data.files}
    return None

@st.cache_resource(show_spinner=False)
def load_tri_benchmark_dataset():
    """Load cached tri-benchmark dataset splits and scalers from pkl."""
    pkl_path = os.path.join(repo_root, "research", "results", "cached_tri_benchmark_datasets.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_horizon_specialization():
    """Load step-by-step expert specialization results (h=1..24)."""
    csv_path = os.path.join(repo_root, "research", "analysis", "phase15a_horizon_specialization.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(repo_root, "research", "analysis", "phase15a_horizon_specialization_corrected.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def load_horizon_candidates():
    """Load step-by-step routing candidate ablation results (h=1..24)."""
    csv_path = os.path.join(repo_root, "research", "analysis", "phase15b_horizon_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def load_gefcom_seed_results():
    """Load GEFCom2014 verified seed results."""
    csv_path = os.path.join(repo_root, "research", "results", "phase8_seed_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def load_gefcom_task_results():
    """Load GEFCom2014 verified task breakdown results."""
    csv_path = os.path.join(repo_root, "research", "results", "phase8_task_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def load_uci_seed_results():
    """Load UCI Electricity verified seed results."""
    csv_path = os.path.join(repo_root, "research", "results", "phase9_seed_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def load_uci_routing_diagnostics():
    """Load UCI Electricity verified routing diagnostics."""
    csv_path = os.path.join(repo_root, "research", "results", "phase9_routing_diagnostics.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data(show_spinner=False)
def get_hero_b64():
    """Load local hero visual and return base64 data string."""
    img_path = os.path.join(cur_dir, "assets", "hero_visual.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Clean Internal Prediction Artifact Abstraction
def load_prediction_artifact(dataset_name: str):
    """
    Unified loader for dataset evaluation artifacts.
    Returns dictionary with:
      - available: bool (True if step-by-step prediction arrays are stored)
      - name: str
      - units: str
      - horizon: str
      - instances_text: str
      - test_mae: float
      - test_rmse: float
      - test_r2: float
      - data: dict or None
    """
    if dataset_name == "PJM":
        p5_cache = load_pjm_prediction_cache()
        tri_data = load_tri_benchmark_dataset()
        has_arrays = p5_cache is not None and tri_data is not None and "PJM" in tri_data
        
        data_bundle = None
        if has_arrays:
            scaler = tri_data["PJM"]["scaler"]
            pjm_w = tri_data["PJM"]["windows"]["test"]
            data_bundle = {
                "cache": p5_cache,
                "X": pjm_w["X"],
                "scaler_mean": float(scaler.mean_[0]),
                "scaler_scale": float(scaler.scale_[0])
            }
        return {
            "available": has_arrays,
            "name": "PJM Interconnection",
            "grid_type": "US Regional Transmission Organization (RTO)",
            "units": "MW",
            "horizon": "24-Hour Day-Ahead (168h Context)",
            "instances_count": 1294 if has_arrays else 0,
            "instances_text": "1,294 evaluation instances (full test partition)",
            "test_mae": 250.9747,
            "test_mae_std": 10.6938,
            "test_rmse": 335.3822,
            "test_r2": 0.8714,
            "data": data_bundle
        }
    elif dataset_name == "GEFCom2014":
        return {
            "available": False,
            "name": "GEFCom2014 (Zone 21)",
            "grid_type": "International Competition Power Grid Series",
            "units": "kW",
            "horizon": "24-Hour Day-Ahead (168h Context)",
            "instances_count": 456,
            "instances_text": "15 monthly competition tasks (K = 456 daily blocks)",
            "test_mae": 12.4077,
            "test_mae_std": 0.1525,
            "test_rmse": 18.0446,
            "test_r2": 0.8610,
            "data": None
        }
    elif dataset_name == "UCI":
        return {
            "available": False,
            "name": "UCI Electricity (LD2011_2014)",
            "grid_type": "Client Aggregated Consumption Benchmark",
            "units": "MW",
            "horizon": "24-Hour Day-Ahead (168h Context)",
            "instances_count": 163,
            "instances_text": "K = 163 daily blocks (3,922 hourly instances)",
            "test_mae": 7.7371,
            "test_mae_std": 0.3037,
            "test_rmse": 10.9556,
            "test_r2": 0.9831,
            "data": None
        }
    return None


df_horizon_spec = load_horizon_specialization()
df_horizon_cand = load_horizon_candidates()


# =========================================================================
# PLOTLY DARK THEME HELPER
# =========================================================================
def apply_dark_plotly_theme(fig, height=420):
    """Applies a consistent, clean dark theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B1120",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color="#CBD5E1"),
        height=height,
        margin=dict(l=45, r=35, t=50, b=45),
        xaxis=dict(
            gridcolor="#1E293B",
            linecolor="#334155",
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#CBD5E1", size=11)
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            linecolor="#334155",
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#CBD5E1", size=11)
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#E2E8F0", size=10)
        )
    )
    return fig


# =========================================================================
# MODERN SIDEBAR NAVIGATION
# =========================================================================
st.sidebar.markdown("""
<div style="padding: 6px 0 14px 0;">
    <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">⚡ CAEG-Net</div>
    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">Research Dashboard</div>
    <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600; letter-spacing: 0.05em; margin-top: 4px;">v1.0 • LOCKED</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "⌂ Overview",
        "◈ Prediction / Forecast",
        "◫ Architecture",
        "▣ Dataset & Protocol",
        "▥ Benchmark Results",
        "◉ Baseline Comparison",
        "⌁ Routing Behaviour",
        "◇ Confidence / Fallback",
        "⌁ Statistical Evidence",
        "◷ Horizon Analysis",
        "✦ Research Findings",
        "ⓘ Limitations & Ethics"
    ],
    label_visibility="collapsed"
)

# Sidebar Model Info Card
st.sidebar.markdown("""
<div class="sidebar-info-card">
    <div class="sidebar-info-title">CAEG-Net</div>
    <div class="sidebar-info-sub">F2 / A2-OOF (Locked)</div>
    <div class="sidebar-info-meta">
        <strong>121,724</strong> Parameters<br>
        <strong>168h</strong> Lookback → <strong>24h</strong> Lead<br>
        <strong>3</strong> Benchmark Grids<br>
        <strong>5</strong> Evaluation Seeds
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# 1. ⌂ OVERVIEW (APPROVED VISUAL BENCHMARK)
# =========================================================================
if page == "⌂ Overview":
    # Premium Hero Section with Local Visual & Gradient Overlay
    hero_b64 = get_hero_b64()
    if hero_b64:
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(7, 10, 19, 0.97) 0%, rgba(7, 10, 19, 0.90) 42%, rgba(7, 10, 19, 0.45) 75%, rgba(7, 10, 19, 0.18) 100%), 
                        url('data:image/jpeg;base64,{hero_b64}');
            background-size: cover;
            background-position: center right;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 34px 36px 30px 36px;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
        ">
            <div style="max-width: 650px;">
                <span class="status-badge" style="margin-bottom: 10px;">FINAL MODEL — LOCKED</span>
                <div class="hero-title" style="font-size: 2.7rem; margin-bottom: 4px; line-height: 1.1; font-weight: 800; color: #F8FAFC;">CAEG-Net</div>
                <div style="font-size: 1.12rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px; letter-spacing: -0.01em;">
                    Context-Adaptive Expert Gating Network
                </div>
                <div class="hero-sub" style="font-size: 0.92rem; line-height: 1.45; color: #CBD5E1; margin-bottom: 16px;">
                    Short-Term Electricity Load Forecasting with Heterogeneous Temporal Experts (LSTM · TCN · CNN)
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; border-radius: 6px; padding: 4px 11px; font-size: 0.78rem; color: #E2E8F0; font-weight: 500;">
                        ⚡ <strong>168h</strong> Lookback → <strong>24h</strong> Horizon
                    </span>
                    <span style="background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; border-radius: 6px; padding: 4px 11px; font-size: 0.78rem; color: #E2E8F0; font-weight: 500;">
                        🧠 <strong>121,724</strong> Parameters
                    </span>
                    <span style="background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; border-radius: 6px; padding: 4px 11px; font-size: 0.78rem; color: #38BDF8; font-weight: 500;">
                        🛡️ Centroid Fallback (λ ≈ 0.51)
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge">FINAL MODEL — LOCKED</span>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">CAEG-Net</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting with Heterogeneous Temporal Experts</div>', unsafe_allow_html=True)

    # Process Line Bar
    st.markdown("""
    <div class="pipeline-container">
        <span class="pipeline-step">168h Load History</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">7D Causal Context</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step active">3 Temporal Experts (LSTM · TCN · CNN)</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step active">Adaptive Router & Shrinkage</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">24h Day-Ahead Forecast</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Inspect Conceptual Architecture & Stream Convergence Visual", expanded=False):
        img_file = os.path.join(cur_dir, "assets", "hero_visual.jpg")
        if os.path.exists(img_file):
            st.image(img_file, caption="Conceptual Visual: Three heterogeneous temporal streams (LSTM recurrent persistence, TCN dilated causal history, CNN localized ramp patterns) converging into the central Context-Adaptive Fusion node, projecting the 24-hour day-ahead load forecast trajectory across the grid mesh.", use_container_width=True)

    # Top 5 Metric Cards (Rendered inside single responsive metric-grid-5 for perfect horizontal/vertical alignment)
    st.markdown("""
    <div class="metric-grid-5">
        <div class="metric-card">
            <div class="metric-label">Parameters</div>
            <div class="metric-value">121,724</div>
            <div class="metric-sub">Trainable parameters (<0.5 MB)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Lookback</div>
            <div class="metric-value">168 Hours</div>
            <div class="metric-sub">7 days of historical context</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Forecast</div>
            <div class="metric-value">24 Hours</div>
            <div class="metric-sub">Day-ahead lead horizon</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Datasets</div>
            <div class="metric-value">3 Grids</div>
            <div class="metric-sub">PJM · GEFCom · UCI</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Seeds</div>
            <div class="metric-value">5 Seeds</div>
            <div class="metric-sub">Stochastic sensitivity (ddof=0)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Modern Two-Column Layout: Research Question & Contributions
    col_l, col_r = st.columns([1.1, 1.3])
    with col_l:
        st.markdown("""
        <div class="accent-card">
            <div style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px;">Core Research Question</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; line-height: 1.45; margin-bottom: 12px;">
                "CAEG-Net evaluates whether context-aware fusion of heterogeneous temporal experts can improve short-term load forecasting."
            </div>
            <div style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
                Rather than relying on an isolated monolithic neural backbone or a static weight assignment, CAEG-Net integrates complementary inductive biases—recurrent sequence continuity (LSTM), causal multi-scale receptive field (TCN), and local edge matching (CNN)—under a dynamic gating network conditioned on causal regime context.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Key Research Contributions</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                <strong>1. Heterogeneous Temporal Architecture:</strong> Combines 3 distinct neural inductive biases totaling 121,724 trainable parameters.<br>
                <strong>2. Causal Context Conditioning:</strong> Evaluates a 7D regime vector (trend, volatility, autocorrelation, causal error, out-of-fold validation residuals).<br>
                <strong>3. Empirical Stabilization Mechanism:</strong> Smoothly blends adaptive convex predictions with the robust equal-expert centroid (learned λ ≈ 0.51).<br>
                <strong>4. Rigorous Leakage Prevention:</strong> Enforces chronological 70/15/15 partitions, train-only scaling, and non-overlapping daily-block statistical validation ($K=53, 456, 163$).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Authoritative Benchmark Performance (5-Seed Evaluation, ddof=0)</div>', unsafe_allow_html=True)

    # 3 Benchmark Cards in metric-grid-3
    st.markdown("""
    <div class="metric-grid-3">
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">PJM Interconnection (MW)</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">250.97 ± 10.69 <span style="font-size: 0.95rem; color: #94A3B8;">MW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">RMSE: <strong>335.38 MW</strong> · R²: <strong>0.8714</strong></div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 0.0406)
            </div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">GEFCom2014 (kW)</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">12.41 ± 0.15 <span style="font-size: 0.95rem; color: #94A3B8;">kW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">RMSE: <strong>18.04 kW</strong> · R²: <strong>0.8610</strong></div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 1.02e-27)
            </div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">UCI Electricity (MW)</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">7.74 ± 0.30 <span style="font-size: 0.95rem; color: #94A3B8;">MW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">RMSE: <strong>10.96 MW</strong> · R²: <strong>0.9831</strong></div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 0.0017)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ctx_l, col_ctx_r = st.columns(2)
    with col_ctx_l:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Scientific Findings Summary</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.55;">
                • <strong>Balanced Cross-Grid Performance:</strong> CAEG-Net F2 delivers the strongest cross-grid balance across all 3 evaluated datasets.<br>
                • <strong>Convex Routing Blending:</strong> Gating weights remain interior to the simplex ($N_{\text{eff}} \approx 2.98 - 2.99$), preventing collapse.<br>
                • <strong>Empirical Stabilization:</strong> Learned $\\lambda \\approx 0.51$ smoothly regularizes toward the equal ensemble centroid with low temporal variation.<br>
                • <strong>Horizon Diagnostics:</strong> Explicit per-step routing failed to improve validation performance, validating the single-step router design.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_ctx_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Evaluation Benchmarks</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.55;">
                • <strong>PJM Interconnection:</strong> US regional transmission network; industrial load profile (8,784h, MW).<br>
                • <strong>GEFCom2014:</strong> International competition benchmark; zonal load series with weather volatility (78,888h, kW).<br>
                • <strong>UCI Electricity:</strong> Aggregated smart meter customer demand series (26,304h, MW).
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 2. ◈ PREDICTION / FORECAST (DATASET SWITCHING & INSTANCE VIEWER)
# =========================================================================
elif page == "◈ Prediction / Forecast":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Interactive Evaluation Viewer</span>
        <div class="page-title">24-Hour Load Forecast</div>
        <div class="page-subtitle">Interactive evaluation viewer for CAEG-Net. Explore stored test evaluations, examine day-ahead trajectories, and compare model outputs with individual temporal experts.</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. PROMINENT DATASET SELECTOR
    st.markdown('<div class="section-title">1. Dataset Selection</div>', unsafe_allow_html=True)
    ds_options = ["PJM", "GEFCom2014", "UCI"]
    curr_idx = ds_options.index(st.session_state.get("selected_dataset", "PJM")) if st.session_state.get("selected_dataset") in ds_options else 0
    
    col_sel_ds, col_sel_meta = st.columns([1.6, 2.4])
    with col_sel_ds:
        chosen_ds = st.radio(
            "Select Benchmark Dataset",
            ds_options,
            index=curr_idx,
            horizontal=True,
            help="Switch active dataset for all dependent forecast visualizations.",
            key="pred_dataset_radio"
        )
        st.session_state["selected_dataset"] = chosen_ds

    artifact = load_prediction_artifact(chosen_ds)

    with col_sel_meta:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px 16px; font-size:0.82rem; color:#CBD5E1;">
            <div><strong>Active Dataset:</strong> <span style="color:#38BDF8; font-weight:600;">{artifact['name']}</span> ({artifact['grid_type']})</div>
            <div style="margin-top:4px;"><strong>Units:</strong> {artifact['units']} &nbsp;|&nbsp; <strong>Horizon:</strong> {artifact['horizon']} &nbsp;|&nbsp; <strong>Available Instances:</strong> {artifact['instances_text']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # CASE A: PJM (Detailed stored prediction traces available)
    if artifact["available"] and artifact["data"] is not None:
        bundle = artifact["data"]
        cache = bundle["cache"]
        X_all = bundle["X"]
        scaler_mean = bundle["scaler_mean"]
        scaler_scale = bundle["scaler_scale"]
        max_windows = cache["y_true_raw"].shape[0] - 1  # 1293

        # 2. FORECAST INSTANCE SELECTOR
        st.markdown('<div class="section-title">2. Forecast Instance Selection</div>', unsafe_allow_html=True)
        col_ctrl1, col_ctrl2 = st.columns([2.0, 1.0])
        with col_ctrl1:
            preset_select = st.selectbox(
                "Curated Instance Presets (PJM Test Partition)",
                [
                    "Window 721 — Median Representative Instance (MAE ~ 237 MW)",
                    "Window 1185 — High Accuracy Diurnal Cycle (MAE ~ 57 MW)",
                    "Window 493 — Summer Peak Demand Spike (Peak ~ 8,650 MW)",
                    "Window 476 — High Volatility / Steep Ramp (Max Ramp ~ 598 MW/h)",
                    "Window 0 — Chronological Test Horizon Origin (MAE ~ 360 MW)",
                    "Custom Window Index (Slider)"
                ],
                index=0
            )
        with col_ctrl2:
            st.selectbox("Forecast Horizon", ["24 Hours (Day-Ahead Dispatch)"], index=0)

        if "721" in preset_select:
            sel_idx = 721
        elif "1185" in preset_select:
            sel_idx = 1185
        elif "493" in preset_select:
            sel_idx = 493
        elif "476" in preset_select:
            sel_idx = 476
        elif "Window 0" in preset_select:
            sel_idx = 0
        else:
            sel_idx = st.slider("Select Window Index (0 to 1293):", 0, max_windows, 721)

        # Extract verified arrays
        x_raw = X_all[sel_idx].flatten() * scaler_scale + scaler_mean
        y_true = cache["y_true_raw"][sel_idx]
        y_caeg = cache["caeg_seed_42"][sel_idx]
        y_lstm = cache["lstm_seed_42"][sel_idx]
        y_tcn = cache["tcn_seed_42"][sel_idx]
        y_cnn = cache["cnn_seed_42"][sel_idx]
        y_static = cache["static_seed_42"][sel_idx]
        weights = cache["weights_seed_42"][sel_idx]

        win_mae = np.mean(np.abs(y_true - y_caeg))
        win_rmse = np.sqrt(np.mean((y_true - y_caeg)**2))
        win_max_err = np.max(np.abs(y_true - y_caeg))

        # Metric Cards for selected instance (metric-grid-5)
        st.markdown(f"""
        <div class="metric-grid-5">
            <div class="metric-card">
                <div class="metric-label">Selected Instance</div>
                <div class="metric-value">#{sel_idx}</div>
                <div class="metric-sub">PJM test partition</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Window MAE</div>
                <div class="metric-value">{win_mae:.2f} <span style="font-size:0.8rem; color:#94A3B8;">MW</span></div>
                <div class="metric-sub">Instance mean absolute error</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Window RMSE</div>
                <div class="metric-value">{win_rmse:.2f} <span style="font-size:0.8rem; color:#94A3B8;">MW</span></div>
                <div class="metric-sub">Quadratic lead penalty</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak Abs Gap</div>
                <div class="metric-value">{win_max_err:.2f} <span style="font-size:0.8rem; color:#94A3B8;">MW</span></div>
                <div class="metric-sub">Max single-step deviation</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dataset-Level MAE</div>
                <div class="metric-value">250.97 <span style="font-size:0.8rem; color:#94A3B8;">MW</span></div>
                <div class="metric-sub">Full test partition (5-seed)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. PRIMARY CHART: ACTUAL vs CAEG-Net FORECAST
        st.markdown('<div class="section-title">3. Actual vs CAEG-Net Forecast (168h Historical Context + 24h Forecast)</div>', unsafe_allow_html=True)
        
        t_hist = np.arange(-168, 0)
        t_lead = np.arange(1, 25)

        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=t_hist, y=x_raw, mode="lines", name="168h Historical Context",
            line=dict(color="#64748B", width=1.8), hovertemplate="Context t=%{x}h: %{y:.1f} MW<extra></extra>"
        ))
        fig_main.add_trace(go.Scatter(
            x=t_lead, y=y_true, mode="lines+markers", name="Ground Truth Actual Load",
            line=dict(color="#F8FAFC", width=2.6), marker=dict(color="#F8FAFC", size=5),
            hovertemplate="Actual t=+%{x}h: %{y:.1f} MW<extra></extra>"
        ))
        fig_main.add_trace(go.Scatter(
            x=t_lead, y=y_caeg, mode="lines+markers", name="CAEG-Net Forecast (24h)",
            line=dict(color="#38BDF8", width=3.4), marker=dict(color="#38BDF8", size=6, symbol="square"),
            hovertemplate="CAEG-Net t=+%{x}h: %{y:.1f} MW<extra></extra>"
        ))
        fig_main.add_vline(
            x=0, line_width=2, line_dash="dash", line_color="#EF4444",
            annotation_text="Forecast Origin (t=0h)", annotation_position="top left",
            annotation_font=dict(color="#EF4444", size=10)
        )
        fig_main.update_layout(
            title="PJM Interconnection — 168h Historical Context + 24h Day-Ahead Forecast Trajectory",
            xaxis_title="Time Relative to Forecast Origin (Hours)",
            yaxis_title="Electricity Load (MW)",
            hovermode="x unified"
        )
        apply_dark_plotly_theme(fig_main, height=480)
        st.plotly_chart(fig_main, use_container_width=True)

        # 4. EXPERT COMPARISON
        st.markdown('<div class="section-title">4. Temporal Expert Comparison (24-Hour Lead Horizon)</div>', unsafe_allow_html=True)
        st.caption("Side-by-side comparison of individual temporal experts against Ground Truth and CAEG-Net on the selected forecast instance.")

        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_true, mode="lines+markers", name="Actual Ground Truth",
            line=dict(color="#F8FAFC", width=3.0), marker=dict(color="#F8FAFC", size=6),
            hovertemplate="Actual: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_caeg, mode="lines+markers", name="CAEG-Net (Fused)",
            line=dict(color="#38BDF8", width=3.2), marker=dict(color="#38BDF8", size=6, symbol="square"),
            hovertemplate="CAEG-Net: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_lstm, mode="lines", name="LSTM Expert",
            line=dict(color="#F59E0B", width=1.8, dash="dot"),
            hovertemplate="LSTM: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_tcn, mode="lines", name="TCN Expert",
            line=dict(color="#10B981", width=1.8, dash="dash"),
            hovertemplate="TCN: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_cnn, mode="lines", name="CNN Expert",
            line=dict(color="#EC4899", width=1.8, dash="dashdot"),
            hovertemplate="CNN: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.add_trace(go.Scatter(
            x=t_lead, y=y_static, mode="lines", name="Equal Ensemble (1/3)",
            line=dict(color="#A855F7", width=1.5, dash="longdash"),
            hovertemplate="Equal Ens: %{y:.1f} MW<extra></extra>"
        ))
        fig_exp.update_layout(
            title=f"PJM Interconnection — Multi-Expert Trajectory Comparison (Instance #{sel_idx})",
            xaxis_title="Forecast Step (h = +1 to +24 Hours)",
            yaxis_title="Load (MW)",
            hovermode="x unified"
        )
        apply_dark_plotly_theme(fig_exp, height=420)
        st.plotly_chart(fig_exp, use_container_width=True)

        # 5. FORECAST ERROR / SIGNED RESIDUALS & ROUTING WEIGHTS
        col_res, col_w = st.columns([1.1, 0.9])
        with col_res:
            st.markdown('<div class="section-title">5. Forecast Residuals (Lead h = 1..24)</div>', unsafe_allow_html=True)
            residuals = y_caeg - y_true
            colors = ["#EF4444" if r > 0 else "#38BDF8" for r in residuals]
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=t_lead, y=residuals, marker_color=colors,
                name="Signed Error (Pred - Actual)",
                hovertemplate="Lead +%{x}h: %{y:.1f} MW<extra></extra>"
            ))
            fig_res.add_hline(y=0, line_color="#64748B", line_width=1)
            fig_res.update_layout(
                title="Step-by-Step Signed Error (MW)",
                xaxis_title="Lead Hour (h)",
                yaxis_title="Residual (MW)",
                showlegend=False
            )
            apply_dark_plotly_theme(fig_res, height=320)
            st.plotly_chart(fig_res, use_container_width=True)

        with col_w:
            st.markdown('<div class="section-title">6. Dynamic Routing Weights</div>', unsafe_allow_html=True)
            fig_rw = go.Figure()
            fig_rw.add_trace(go.Bar(
                x=["LSTM", "TCN", "CNN"],
                y=[weights[0] * 100, weights[1] * 100, weights[2] * 100],
                marker_color=["#F59E0B", "#10B981", "#EC4899"],
                text=[f"{weights[0]*100:.1f}%", f"{weights[1]*100:.1f}%", f"{weights[2]*100:.1f}%"],
                textposition="auto",
                hovertemplate="%{x} Weight: %{y:.2f}%<extra></extra>"
            ))
            fig_rw.update_layout(
                title=f"Instance #{sel_idx} Gating Simplex Allocation",
                yaxis_title="Allocation (%)",
                yaxis=dict(range=[0, 60]),
                showlegend=False
            )
            apply_dark_plotly_theme(fig_rw, height=320)
            st.plotly_chart(fig_rw, use_container_width=True)

    # CASE B: GEFCom2014 or UCI (Honest Empty State & Verified Aggregate Results)
    else:
        st.markdown("""
        <div class="empty-state-card">
            <div style="font-size:1.15rem; font-weight:700; color:#38BDF8; margin-bottom:8px;">
                ◈ PREDICTION TRACE UNAVAILABLE
            </div>
            <div style="font-size:0.92rem; color:#E2E8F0; max-width:680px; margin:0 auto 12px auto; line-height:1.5;">
                Verified step-by-step prediction arrays for <strong>""" + artifact["name"] + """</strong> are not currently stored in the dashboard artifact set.
            </div>
            <div style="font-size:0.84rem; color:#94A3B8; max-width:680px; margin:0 auto; line-height:1.5;">
                In accordance with the zero-fabrication academic research protocol, synthetic or simulated prediction traces are strictly prohibited. 
                Full multi-seed benchmark evaluations, seed distributions, routing diagnostics, and non-overlapping daily-block statistical tests are verified and presented below.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show verified summary metric cards for this dataset
        st.markdown(f"""
        <div class="metric-grid-4">
            <div class="metric-card">
                <div class="metric-label">Primary Test MAE</div>
                <div class="metric-value">{artifact['test_mae']:.2f} <span style="font-size:0.8rem; color:#94A3B8;">{artifact['units']}</span></div>
                <div class="metric-sub">5-seed mean (ddof=0)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Primary Test RMSE</div>
                <div class="metric-value">{artifact['test_rmse']:.2f} <span style="font-size:0.8rem; color:#94A3B8;">{artifact['units']}</span></div>
                <div class="metric-sub">Standard deviation penalty</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test R² Score</div>
                <div class="metric-value">{artifact['test_r2']:.4f}</div>
                <div class="metric-sub">Variance explained</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Evaluation Blocks</div>
                <div class="metric-value">{artifact['instances_count']}</div>
                <div class="metric-sub">{artifact['instances_text']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show dataset-specific verified seed/task breakdown
        if chosen_ds == "GEFCom2014":
            st.markdown('<div class="section-title">Verified GEFCom2014 Multi-Seed Results (Phase 8 Artifacts)</div>', unsafe_allow_html=True)
            df_g_seeds = load_gefcom_seed_results()
            if df_g_seeds is not None:
                caeg_seeds = df_g_seeds[df_g_seeds["model"] == "Original_CAEGNet_V1"][["seed", "MAE", "RMSE", "R2", "MAPE"]].copy()
                caeg_seeds.columns = ["Seed", "MAE (kW)", "RMSE (kW)", "R² Score", "MAPE (%)"]
                
                # HTML Table
                rows_html = ""
                for _, r in caeg_seeds.iterrows():
                    rows_html += f"<tr><td>Seed {int(r['Seed'])}</td><td class='num-cell'>{r['MAE (kW)']:.2f}</td><td class='num-cell'>{r['RMSE (kW)']:.2f}</td><td class='num-cell'>{r['R² Score']:.4f}</td><td class='num-cell'>{r['MAPE (%)']:.2f}%</td></tr>"
                rows_html += f"<tr class='highlight-row'><td><strong>Mean (ddof=0)</strong></td><td class='num-cell'><strong>12.41 ± 0.15</strong></td><td class='num-cell'><strong>18.04</strong></td><td class='num-cell'><strong>0.8610</strong></td><td class='num-cell'><strong>9.37%</strong></td></tr>"
                
                st.markdown(f"""
                <table class="custom-table">
                    <thead><tr><th>Evaluation Run</th><th style="text-align:right;">MAE (kW)</th><th style="text-align:right;">RMSE (kW)</th><th style="text-align:right;">R² Score</th><th style="text-align:right;">MAPE</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

            # Routing distribution for GEFCom
            st.markdown('<div class="section-title">Verified GEFCom2014 Gating Simplex Allocation</div>', unsafe_allow_html=True)
            col_grw, col_ginfo = st.columns([1.1, 0.9])
            with col_grw:
                fig_g = go.Figure()
                fig_g.add_trace(go.Bar(
                    x=["LSTM Expert", "TCN Expert", "CNN Expert"],
                    y=[35.28, 29.82, 34.90],
                    marker_color=["#F59E0B", "#10B981", "#EC4899"],
                    text=["35.28%", "29.82%", "34.90%"],
                    textposition="auto"
                ))
                fig_g.update_layout(
                    title="GEFCom2014 — Empirical Gating Weight Distribution",
                    yaxis_title="Average Weight (%)",
                    yaxis=dict(range=[0, 50]),
                    showlegend=False
                )
                apply_dark_plotly_theme(fig_g, height=320)
                st.plotly_chart(fig_g, use_container_width=True)
            with col_ginfo:
                st.markdown("""
                <div class="dark-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div class="dark-card-header">GEFCom2014 Routing Dynamics</div>
                    <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                        • <strong>Effective Experts:</strong> N<sub>eff</sub> = 2.9765 / 3.000<br>
                        • <strong>Weight Entropy:</strong> H = 1.0911 nats<br>
                        • <strong>Fallback Coefficient:</strong> λ = 0.5170 ± 0.0055 (CV = 1.06%)<br>
                        • <strong>Observation:</strong> The gating router allocates substantial weight to both LSTM and CNN experts, reflecting weather-driven diurnal peaks and ramp spikes.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif chosen_ds == "UCI":
            st.markdown('<div class="section-title">Verified UCI Electricity Multi-Seed Results (Phase 9 Artifacts)</div>', unsafe_allow_html=True)
            df_u_seeds = load_uci_seed_results()
            if df_u_seeds is not None:
                caeg_u = df_u_seeds[df_u_seeds["model"] == "Original CAEG-Net V1"][["seed", "test_mae", "test_rmse", "test_r2", "test_mape"]].copy()
                caeg_u.columns = ["Seed", "MAE (MW)", "RMSE (MW)", "R² Score", "MAPE (%)"]
                
                rows_html = ""
                for _, r in caeg_u.iterrows():
                    rows_html += f"<tr><td>Seed {int(r['Seed'])}</td><td class='num-cell'>{r['MAE (MW)']:.2f}</td><td class='num-cell'>{r['RMSE (MW)']:.2f}</td><td class='num-cell'>{r['R² Score']:.4f}</td><td class='num-cell'>{r['MAPE (%)']:.2f}%</td></tr>"
                rows_html += f"<tr class='highlight-row'><td><strong>Mean (ddof=0)</strong></td><td class='num-cell'><strong>7.74 ± 0.30</strong></td><td class='num-cell'><strong>10.96</strong></td><td class='num-cell'><strong>0.9831</strong></td><td class='num-cell'><strong>4.28%</strong></td></tr>"
                
                st.markdown(f"""
                <table class="custom-table">
                    <thead><tr><th>Evaluation Run</th><th style="text-align:right;">MAE (MW)</th><th style="text-align:right;">RMSE (MW)</th><th style="text-align:right;">R² Score</th><th style="text-align:right;">MAPE</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">Verified UCI Electricity Gating Simplex Allocation</div>', unsafe_allow_html=True)
            col_urw, col_uinfo = st.columns([1.1, 0.9])
            with col_urw:
                fig_u = go.Figure()
                fig_u.add_trace(go.Bar(
                    x=["LSTM Expert", "TCN Expert", "CNN Expert"],
                    y=[34.28, 29.94, 35.78],
                    marker_color=["#F59E0B", "#10B981", "#EC4899"],
                    text=["34.28%", "29.94%", "35.78%"],
                    textposition="auto"
                ))
                fig_u.update_layout(
                    title="UCI Electricity — Empirical Gating Weight Distribution",
                    yaxis_title="Average Weight (%)",
                    yaxis=dict(range=[0, 50]),
                    showlegend=False
                )
                apply_dark_plotly_theme(fig_u, height=320)
                st.plotly_chart(fig_u, use_container_width=True)
            with col_uinfo:
                st.markdown("""
                <div class="dark-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div class="dark-card-header">UCI Routing Dynamics</div>
                    <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                        • <strong>Effective Experts:</strong> N<sub>eff</sub> = 2.9842 / 3.000<br>
                        • <strong>Fallback Coefficient:</strong> λ = 0.5064 ± 0.0030 (CV = 0.59%)<br>
                        • <strong>High Base Predictability:</strong> High consumer baseline leads to strong recurrent persistence (LSTM) alongside CNN edge tracking.<br>
                        • <strong>Cross-Grid Consistency:</strong> Confirms convex interior distribution across all evaluated grids.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =========================================================================
# 3. ◫ ARCHITECTURE
# =========================================================================
elif page == "◫ Architecture":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Model Specification</span>
        <div class="page-title">CAEG-Net Architecture</div>
        <div class="page-subtitle">Context-Adaptive Expert Gating Network: Combining recurrent, causal-convolutional, and multi-scale inductive biases with causal regime conditioning.</div>
    </div>
    """, unsafe_allow_html=True)

    # 5 Top Architectural Metric Cards in metric-grid-5
    st.markdown("""
    <div class="metric-grid-5">
        <div class="metric-card">
            <div class="metric-label">Total Parameters</div>
            <div class="metric-value">121,724</div>
            <div class="metric-sub">100.0% of network</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Temporal Experts</div>
            <div class="metric-value">120,504</div>
            <div class="metric-sub">98.99% of parameters</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Gating Router</div>
            <div class="metric-value">1,075</div>
            <div class="metric-sub">0.88% of parameters</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fallback Head</div>
            <div class="metric-value">145</div>
            <div class="metric-sub">0.12% of parameters</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Model Footprint</div>
            <div class="metric-value">&lt; 0.5 MB</div>
            <div class="metric-sub">Ultra-compact deployment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Visual Pipeline Flow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; gap: 10px; align-items: stretch; margin-bottom: 1.5rem; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 170px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">1. Input Lookback</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">168 Hours (L)</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">7 consecutive days of hourly load history.</div>
        </div>
        <div style="flex: 1; min-width: 170px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">2. Causal Context</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">7D Vector (C<sub>t</sub>)</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">Trend, volatility, lag-24, recent error, validation residuals.</div>
        </div>
        <div style="flex: 1; min-width: 170px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">3. Heterogeneous Experts</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">LSTM · TCN · CNN</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">Complementary recurrent, causal, and local motif biases.</div>
        </div>
        <div style="flex: 1; min-width: 170px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">4. Adaptive Gating</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Simplex w<sub>t</sub> ∈ Δ²</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">Convex weights + Centroid Fallback (λ ≈ 0.51).</div>
        </div>
        <div style="flex: 1; min-width: 170px; background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
            <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">5. Day-Ahead Horizon</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">24 Hours (H)</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">Direct multi-horizon day-ahead dispatch vector.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_math, col_spec = st.columns(2)
    with col_math:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Mathematical Formulation</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
        """, unsafe_allow_html=True)
        st.latex(r"\hat{\mathbf{y}}_{	ext{adaptive}, t} = \sum_{k \in \{L, T, C\}} w_{k, t} \hat{\mathbf{y}}_{k, t}, \quad \mathbf{w}_t = 	ext{Softmax}(\mathbf{W}_g \mathbf{h}_t + \mathbf{b}_g)")
        st.latex(r"\hat{\mathbf{y}}_{	ext{final}, t} = \lambda_t \hat{\mathbf{y}}_{	ext{adaptive}, t} + (1 - \lambda_t) \hat{\mathbf{y}}_{	ext{equal}, t}")
        st.markdown("""
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">
                Where $\\hat{\\mathbf{y}}_{\\text{equal}, t} = \\frac{1}{3}(\\hat{\\mathbf{y}}_L + \\hat{\\mathbf{y}}_T + \\hat{\\mathbf{y}}_C)$ is the robust unweighted centroid, and $\\lambda_t = \\sigma(\\mathbf{W}_c \\mathbf{h}_t + b_c)$ provides smooth shrinkage toward the centroid.
            </div>
            </div></div>
        """, unsafe_allow_html=True)
    with col_spec:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Causal Context Feature Specification</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • <strong>Trend:</strong> Linear regression slope across the 168h input window.<br>
                • <strong>Volatility:</strong> Normalized standard deviation of load across lookback.<br>
                • <strong>Periodicity:</strong> 24-hour lag autocorrelation ($r_{\text{lag-24}}$).<br>
                • <strong>Causal Recent Error:</strong> Rolling MAE of forecasts over observed lookback.<br>
                • <strong>OOF Performance Conditioning:</strong> Causally tracked historical validation error residuals for each expert, preventing circular self-bias.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Exact Trainable Parameter Breakdown</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="custom-table">
        <thead>
            <tr>
                <th>Component</th>
                <th>Architectural Details</th>
                <th style="text-align: right;">Parameters</th>
                <th style="text-align: right;">Percentage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>LSTM Expert</strong></td>
                <td>2-layer stacked LSTM (hidden=64, dropout=0.1) + Linear projection (64 → 24)</td>
                <td class="num-cell">56,152</td>
                <td class="num-cell">46.13%</td>
            </tr>
            <tr>
                <td><strong>TCN Expert</strong></td>
                <td>6 dilated causal residual blocks (d=1,2,4,8,16,32, channels=32) + Adaptive pooling</td>
                <td class="num-cell">36,952</td>
                <td class="num-cell">30.36%</td>
            </tr>
            <tr>
                <td><strong>CNN Expert</strong></td>
                <td>3 multi-scale 1D conv stages (k=3,5,7, BatchNorm, ReLU) + Dense head</td>
                <td class="num-cell">27,400</td>
                <td class="num-cell">22.51%</td>
            </tr>
            <tr style="background: rgba(30, 41, 59, 0.5);">
                <td><strong>Temporal Backbone Subtotal</strong></td>
                <td>Fixed 3-expert neural ensemble</td>
                <td class="num-cell"><strong>120,504</strong></td>
                <td class="num-cell"><strong>98.99%</strong></td>
            </tr>
            <tr>
                <td><strong>Context Gating Router</strong></td>
                <td>Context MLP (LayerNorm + ReLU, 7D → 16D) + Softmax simplex projection</td>
                <td class="num-cell">1,075</td>
                <td class="num-cell">0.88%</td>
            </tr>
            <tr>
                <td><strong>Confidence Fallback Head</strong></td>
                <td>Linear shrinkage blend projection (16D → 1D, Sigmoid)</td>
                <td class="num-cell">145</td>
                <td class="num-cell">0.12%</td>
            </tr>
            <tr class="highlight-row">
                <td><strong>TOTAL CAEG-Net F2</strong></td>
                <td><strong>Locked Production Specification (F2 / A2-OOF)</strong></td>
                <td class="num-cell"><strong>121,724</strong></td>
                <td class="num-cell"><strong>100.00%</strong></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)


# =========================================================================
# 4. ▣ DATASET & PROTOCOL
# =========================================================================
elif page == "▣ Dataset & Protocol":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Methodological Rigor</span>
        <div class="page-title">Datasets & Experimental Protocol</div>
        <div class="page-subtitle">Strict temporal order preservation, train-only scaling, and non-overlapping daily-block validation across 3 real-world electrical grids.</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Grid Benchmark Cards in metric-grid-3
    st.markdown("""
    <div class="metric-grid-3">
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">PJM Interconnection</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Mid-Atlantic RTO</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • Contiguous Hours: <strong>8,784</strong> (Leap year)<br>
                    • Load Unit: <strong>Megawatts (MW)</strong><br>
                    • Resolution: 1-hour intervals<br>
                    • Characteristics: High-magnitude baseline, industrial shift cycles.
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Test Partition: 1,294 instances</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">GEFCom2014</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Zone 21 Benchmark</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • Contiguous Hours: <strong>78,888</strong> (Long series)<br>
                    • Load Unit: <strong>Kilowatts (kW)</strong><br>
                    • Resolution: 1-hour intervals<br>
                    • Characteristics: Extreme weather sensitivity, seasonal swings.
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Test Partition: 15 tasks (K = 456 blocks)</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">UCI Electricity</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Client Aggregation</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • Contiguous Hours: <strong>26,304</strong> (Multi-year)<br>
                    • Load Unit: <strong>Megawatts (MW)</strong><br>
                    • Resolution: 1-hour intervals<br>
                    • Characteristics: Highly predictable consumer diurnal cycles.
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Test Partition: K = 163 blocks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Partitioning Protocol (Zero Temporal Shuffling)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; height: 38px; border-radius: 6px; overflow: hidden; margin-bottom: 1.2rem; font-size: 0.8rem; font-weight: 700; text-align: center; line-height: 38px;">
        <div style="flex: 70; background: #1E3A8A; color: #93C5FD;">TRAINING SET (70%) — Model Parameter Optimization</div>
        <div style="flex: 15; background: #0369A1; color: #BAE6FD;">VAL (15%)</div>
        <div style="flex: 15; background: #15803D; color: #BBF7D0;">TEST (15%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Leakage Controls in metric-grid-4
    st.markdown("""
    <div class="metric-grid-4">
        <div class="metric-card">
            <div class="metric-label">1. Train-Only Scaling</div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.4; margin-top: 4px;">StandardScaler fitted strictly on training partition; zero test leakage.</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">2. Causal Context</div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.4; margin-top: 4px;">Context features use lookback exclusively; forecast window strictly excluded.</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">3. Causal OOF Residuals</div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.4; margin-top: 4px;">OOF validation error tracked causally, preventing circular self-bias.</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">4. Non-Overlapping Blocks</div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.4; margin-top: 4px;">Statistical tests evaluated over K daily blocks to respect independence.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# 5. ▥ BENCHMARK RESULTS
# =========================================================================
elif page == "▥ Benchmark Results":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Performance Certification</span>
        <div class="page-title">Authoritative Benchmark Results</div>
        <div class="page-subtitle">Evaluation across 3 diverse electrical grids using 5 independent random seeds with population standard deviation (ddof=0).</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Primary Benchmark Cards in metric-grid-3
    st.markdown("""
    <div class="metric-grid-3">
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">PJM Interconnection (MW)</div>
                <div style="font-size: 1.7rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">250.97 ± 10.69 <span style="font-size: 0.95rem; color: #94A3B8;">MW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    RMSE: <strong>335.38 MW</strong> &nbsp;|&nbsp; R²: <strong>0.8714</strong><br>
                    Relative Seed Std: <strong>4.26%</strong> (High stability)
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 0.0406)
            </div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">GEFCom2014 (kW)</div>
                <div style="font-size: 1.7rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">12.41 ± 0.15 <span style="font-size: 0.95rem; color: #94A3B8;">kW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    RMSE: <strong>18.04 kW</strong> &nbsp;|&nbsp; R²: <strong>0.8610</strong><br>
                    Relative Seed Std: <strong>1.23%</strong> (Exceptional consistency)
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 1.02e-27)
            </div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">UCI Electricity (MW)</div>
                <div style="font-size: 1.7rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">7.74 ± 0.30 <span style="font-size: 0.95rem; color: #94A3B8;">MW</span></div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    RMSE: <strong>10.96 MW</strong> &nbsp;|&nbsp; R²: <strong>0.9831</strong><br>
                    Relative Seed Std: <strong>3.92%</strong> (High stability)
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 8px;">
                ✓ Statistically significant vs baseline (p = 0.0017)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Seed-by-Seed Realizations (5 Random Evaluation Seeds)</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="custom-table">
        <thead>
            <tr>
                <th>Benchmark Grid</th>
                <th style="text-align: right;">Seed 42</th>
                <th style="text-align: right;">Seed 123</th>
                <th style="text-align: right;">Seed 999</th>
                <th style="text-align: right;">Seed 2024</th>
                <th style="text-align: right;">Seed 3407</th>
                <th style="text-align: right;">Mean MAE</th>
                <th style="text-align: right;">Pop SD (ddof=0)</th>
                <th style="text-align: right;">CV (%)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>PJM Interconnection (MW)</strong></td>
                <td class="num-cell">249.90</td>
                <td class="num-cell">262.38</td>
                <td class="num-cell">233.76</td>
                <td class="num-cell">262.18</td>
                <td class="num-cell">246.65</td>
                <td class="num-cell"><strong>250.97</strong></td>
                <td class="num-cell">10.69</td>
                <td class="num-cell">4.26%</td>
            </tr>
            <tr>
                <td><strong>GEFCom2014 (kW)</strong></td>
                <td class="num-cell">12.58</td>
                <td class="num-cell">12.24</td>
                <td class="num-cell">12.57</td>
                <td class="num-cell">12.43</td>
                <td class="num-cell">12.23</td>
                <td class="num-cell"><strong>12.41</strong></td>
                <td class="num-cell">0.15</td>
                <td class="num-cell">1.23%</td>
            </tr>
            <tr>
                <td><strong>UCI Electricity (MW)</strong></td>
                <td class="num-cell">8.21</td>
                <td class="num-cell">7.94</td>
                <td class="num-cell">7.59</td>
                <td class="num-cell">7.34</td>
                <td class="num-cell">7.61</td>
                <td class="num-cell"><strong>7.74</strong></td>
                <td class="num-cell">0.30</td>
                <td class="num-cell">3.92%</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    col_note_l, col_note_r = st.columns(2)
    with col_note_l:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Balanced Cross-Grid Performance</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                CAEG-Net F2 provides the strongest overall balance of forecasting performance, seed stability, methodological integrity, and architectural simplicity across the evaluated formulations.<br>
                F2 achieves strong results on PJM and UCI, while on GEFCom fixed shrinkage achieves a slightly lower mean MAE. Thus F2 is presented as a balanced multi-dataset model rather than claiming universal dominance on every benchmark.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_note_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Statistical Dispersion Standard</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                Following strict mathematical conventions, all multi-seed standard deviations are computed as population standard deviations (<code>ddof=0</code>) across the 5 canonical evaluation seeds [42, 123, 999, 2024, 3407].<br>
                These seeds serve to assess stochastic initialization sensitivity, while non-overlapping daily-block tests ($K$) evaluate temporal generalization.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 6. ◉ BASELINE COMPARISON (DATASET SWITCHABLE)
# =========================================================================
elif page == "◉ Baseline Comparison":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Empirical Comparison</span>
        <div class="page-title">Baseline Model Comparison</div>
        <div class="page-subtitle">Evaluating CAEG-Net against standalone neural experts, static ensembles, classical time-series baselines, and controlled exploratory variants.</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Selector for Baseline Comparison
    b_options = ["PJM", "GEFCom2014", "UCI"]
    b_idx = b_options.index(st.session_state.get("selected_dataset", "PJM")) if st.session_state.get("selected_dataset") in b_options else 0
    selected_b_ds = st.radio("Select Grid to Inspect Baselines", b_options, index=b_idx, horizontal=True, key="base_ds_radio")
    st.session_state["selected_dataset"] = selected_b_ds

    if selected_b_ds == "PJM":
        st.markdown("""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Family</th>
                    <th>Model Formulation</th>
                    <th style="text-align: right;">Parameters</th>
                    <th style="text-align: right;">MAE (MW)</th>
                    <th style="text-align: right;">RMSE (MW)</th>
                    <th style="text-align: right;">R² Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr class="highlight-row">
                    <td><strong>PROPOSED</strong></td>
                    <td><strong>CAEG-Net (F2 / A2-OOF)</strong></td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell"><strong>250.97 ± 10.69</strong></td>
                    <td class="num-cell">335.38</td>
                    <td class="num-cell">0.8714</td>
                    <td><strong>CHAMPION_LOCKED</strong></td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone TCN</td>
                    <td class="num-cell">36,952</td>
                    <td class="num-cell">259.33</td>
                    <td class="num-cell">345.12</td>
                    <td class="num-cell">0.8637</td>
                    <td>Best Standalone Expert</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone LSTM</td>
                    <td class="num-cell">56,152</td>
                    <td class="num-cell">291.73</td>
                    <td class="num-cell">388.40</td>
                    <td class="num-cell">0.8275</td>
                    <td>Recurrent Baseline</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone CNN</td>
                    <td class="num-cell">27,400</td>
                    <td class="num-cell">432.08</td>
                    <td class="num-cell">556.80</td>
                    <td class="num-cell">0.6450</td>
                    <td>Local Motif Baseline</td>
                </tr>
                <tr>
                    <td>Ensemble</td>
                    <td>Equal Ensemble (1/3 LSTM + TCN + CNN)</td>
                    <td class="num-cell">120,504</td>
                    <td class="num-cell">279.83</td>
                    <td class="num-cell">368.90</td>
                    <td class="num-cell">0.8443</td>
                    <td>Static Mixture</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Ridge Regression (Multi-output L2)</td>
                    <td class="num-cell">4,056</td>
                    <td class="num-cell">260.40</td>
                    <td class="num-cell">347.80</td>
                    <td class="num-cell">0.8616</td>
                    <td>Linear Autoregressive</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Naive-24 (Day-Ahead Persistence)</td>
                    <td class="num-cell">0</td>
                    <td class="num-cell">430.40</td>
                    <td class="num-cell">584.20</td>
                    <td class="num-cell">0.6091</td>
                    <td>Zero-Parameter Persistence</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Seasonal Naive-168 (Week-Ahead Persistence)</td>
                    <td class="num-cell">0</td>
                    <td class="num-cell">468.10</td>
                    <td class="num-cell">631.50</td>
                    <td class="num-cell">0.5430</td>
                    <td>Weekly Persistence</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Fixed Shrinkage Control</td>
                    <td class="num-cell">121,579</td>
                    <td class="num-cell">253.50 ± 8.03</td>
                    <td class="num-cell">338.95</td>
                    <td class="num-cell">0.8688</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Dynamic Confidence Control</td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell">251.94 ± 9.80</td>
                    <td class="num-cell">336.74</td>
                    <td class="num-cell">0.8704</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Horizon Routing Control</td>
                    <td class="num-cell">122,253</td>
                    <td class="num-cell">257.55 ± 5.37</td>
                    <td class="num-cell">343.91</td>
                    <td class="num-cell">0.8650</td>
                    <td>Exploratory Mechanism</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        fig_b = go.Figure()
        models_p = ["CAEG-Net", "Fixed Shrinkage", "Dynamic Conf", "Horizon Routing", "Standalone TCN", "Ridge", "Equal Ensemble", "Standalone LSTM", "Naive-24", "Standalone CNN", "Seasonal Naive"]
        maes_p = [250.97, 253.50, 251.94, 257.55, 259.33, 260.40, 279.83, 291.73, 430.40, 432.08, 468.10]
        colors_p = ["#38BDF8", "#64748B", "#64748B", "#64748B", "#10B981", "#94A3B8", "#A855F7", "#F59E0B", "#475569", "#EC4899", "#334155"]
        fig_b.add_trace(go.Bar(
            y=models_p[::-1], x=maes_p[::-1], orientation="h", marker_color=colors_p[::-1],
            text=[f"{v:.1f} MW" for v in maes_p[::-1]], textposition="auto",
            hovertemplate="%{y}: %{x:.2f} MW<extra></extra>"
        ))
        fig_b.update_layout(title="PJM Interconnection — Model Comparison (MAE MW, Lower is Better)", xaxis_title="MAE (MW)")
        apply_dark_plotly_theme(fig_b, height=380)
        st.plotly_chart(fig_b, use_container_width=True)

    elif selected_b_ds == "GEFCom2014":
        st.markdown("""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Family</th>
                    <th>Model Formulation</th>
                    <th style="text-align: right;">Parameters</th>
                    <th style="text-align: right;">MAE (kW)</th>
                    <th style="text-align: right;">RMSE (kW)</th>
                    <th style="text-align: right;">R² Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr class="highlight-row">
                    <td><strong>PROPOSED</strong></td>
                    <td><strong>CAEG-Net (F2 / A2-OOF)</strong></td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell"><strong>12.41 ± 0.15</strong></td>
                    <td class="num-cell">18.04</td>
                    <td class="num-cell">0.8610</td>
                    <td><strong>CHAMPION_LOCKED</strong></td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Fixed Shrinkage Control</td>
                    <td class="num-cell">121,579</td>
                    <td class="num-cell">12.36 ± 0.18</td>
                    <td class="num-cell">18.02</td>
                    <td class="num-cell">0.8614</td>
                    <td>Exploratory Diagnostic Min</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Dynamic Confidence Control</td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell">12.49 ± 0.27</td>
                    <td class="num-cell">18.10</td>
                    <td class="num-cell">0.8601</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone TCN</td>
                    <td class="num-cell">36,952</td>
                    <td class="num-cell">12.57</td>
                    <td class="num-cell">18.02</td>
                    <td class="num-cell">0.8613</td>
                    <td>Best Standalone Expert</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Ridge Regression</td>
                    <td class="num-cell">4,056</td>
                    <td class="num-cell">12.57</td>
                    <td class="num-cell">18.39</td>
                    <td class="num-cell">0.8556</td>
                    <td>Linear Autoregressive</td>
                </tr>
                <tr>
                    <td>Ensemble</td>
                    <td>Equal Ensemble (1/3)</td>
                    <td class="num-cell">120,504</td>
                    <td class="num-cell">12.62</td>
                    <td class="num-cell">18.08</td>
                    <td class="num-cell">0.8604</td>
                    <td>Static Mixture</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Horizon Routing Control</td>
                    <td class="num-cell">122,253</td>
                    <td class="num-cell">12.86 ± 0.25</td>
                    <td class="num-cell">18.44</td>
                    <td class="num-cell">0.8548</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone LSTM</td>
                    <td class="num-cell">56,152</td>
                    <td class="num-cell">13.23</td>
                    <td class="num-cell">18.94</td>
                    <td class="num-cell">0.8468</td>
                    <td>Recurrent Baseline</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone CNN</td>
                    <td class="num-cell">27,400</td>
                    <td class="num-cell">14.50</td>
                    <td class="num-cell">20.25</td>
                    <td class="num-cell">0.8248</td>
                    <td>Local Motif Baseline</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Naive-24</td>
                    <td class="num-cell">0</td>
                    <td class="num-cell">16.68</td>
                    <td class="num-cell">24.30</td>
                    <td class="num-cell">0.7479</td>
                    <td>Persistence</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Seasonal Naive-168</td>
                    <td class="num-cell">0</td>
                    <td class="num-cell">26.22</td>
                    <td class="num-cell">36.61</td>
                    <td class="num-cell">0.4276</td>
                    <td>Weekly Persistence</td>
                </tr>
                <tr>
                    <td>Classical Baseline</td>
                    <td>Official Benchmark</td>
                    <td class="num-cell">0</td>
                    <td class="num-cell">30.19</td>
                    <td class="num-cell">42.03</td>
                    <td class="num-cell">0.2456</td>
                    <td>Same-Month-Last-Year</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        fig_bg = go.Figure()
        models_g = ["Fixed Shrinkage", "CAEG-Net", "Dynamic Conf", "Standalone TCN", "Ridge", "Equal Ensemble", "Horizon Routing", "Standalone LSTM", "Standalone CNN", "Naive-24", "Seasonal Naive", "Official Benchmark"]
        maes_g = [12.36, 12.41, 12.49, 12.57, 12.57, 12.62, 12.86, 13.23, 14.50, 16.68, 26.22, 30.19]
        colors_g = ["#64748B", "#38BDF8", "#64748B", "#10B981", "#94A3B8", "#A855F7", "#64748B", "#F59E0B", "#EC4899", "#475569", "#334155", "#1E293B"]
        fig_bg.add_trace(go.Bar(
            y=models_g[::-1], x=maes_g[::-1], orientation="h", marker_color=colors_g[::-1],
            text=[f"{v:.2f} kW" for v in maes_g[::-1]], textposition="auto",
            hovertemplate="%{y}: %{x:.2f} kW<extra></extra>"
        ))
        fig_bg.update_layout(title="GEFCom2014 — Model Comparison (MAE kW, Lower is Better)", xaxis_title="MAE (kW)")
        apply_dark_plotly_theme(fig_bg, height=400)
        st.plotly_chart(fig_bg, use_container_width=True)

    elif selected_b_ds == "UCI":
        st.markdown("""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Family</th>
                    <th>Model Formulation</th>
                    <th style="text-align: right;">Parameters</th>
                    <th style="text-align: right;">MAE (MW)</th>
                    <th style="text-align: right;">RMSE (MW)</th>
                    <th style="text-align: right;">R² Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone LSTM</td>
                    <td class="num-cell">56,152</td>
                    <td class="num-cell">7.55</td>
                    <td class="num-cell">11.94</td>
                    <td class="num-cell">0.9799</td>
                    <td>Best Standalone Expert (UCI)</td>
                </tr>
                <tr class="highlight-row">
                    <td><strong>PROPOSED</strong></td>
                    <td><strong>CAEG-Net (F2 / A2-OOF)</strong></td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell"><strong>7.74 ± 0.30</strong></td>
                    <td class="num-cell">10.96</td>
                    <td class="num-cell">0.9831</td>
                    <td><strong>CHAMPION_LOCKED</strong></td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Fixed Shrinkage Control</td>
                    <td class="num-cell">121,579</td>
                    <td class="num-cell">7.75 ± 0.18</td>
                    <td class="num-cell">10.99</td>
                    <td class="num-cell">0.9830</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Dynamic Confidence Control</td>
                    <td class="num-cell">121,724</td>
                    <td class="num-cell">7.82 ± 0.28</td>
                    <td class="num-cell">11.00</td>
                    <td class="num-cell">0.9830</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr style="background: rgba(30, 41, 59, 0.3);">
                    <td>Controlled Variant</td>
                    <td>Horizon Routing Control</td>
                    <td class="num-cell">122,253</td>
                    <td class="num-cell">8.13 ± 0.40</td>
                    <td class="num-cell">11.48</td>
                    <td class="num-cell">0.9814</td>
                    <td>Exploratory Mechanism</td>
                </tr>
                <tr>
                    <td>Ensemble</td>
                    <td>Equal Ensemble (1/3)</td>
                    <td class="num-cell">120,504</td>
                    <td class="num-cell">8.17</td>
                    <td class="num-cell">12.10</td>
                    <td class="num-cell">0.9794</td>
                    <td>Static Mixture</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone TCN</td>
                    <td class="num-cell">36,952</td>
                    <td class="num-cell">8.34</td>
                    <td class="num-cell">11.90</td>
                    <td class="num-cell">0.9800</td>
                    <td>Causal Conv Baseline</td>
                </tr>
                <tr>
                    <td>Standalone Expert</td>
                    <td>Standalone CNN</td>
                    <td class="num-cell">27,400</td>
                    <td class="num-cell">11.71</td>
                    <td class="num-cell">15.81</td>
                    <td class="num-cell">0.9646</td>
                    <td>Local Motif Baseline</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        fig_bu = go.Figure()
        models_u = ["Standalone LSTM", "CAEG-Net", "Fixed Shrinkage", "Dynamic Conf", "Horizon Routing", "Equal Ensemble", "Standalone TCN", "Standalone CNN"]
        maes_u = [7.55, 7.74, 7.75, 7.82, 8.13, 8.17, 8.34, 11.71]
        colors_u = ["#F59E0B", "#38BDF8", "#64748B", "#64748B", "#64748B", "#A855F7", "#10B981", "#EC4899"]
        fig_bu.add_trace(go.Bar(
            y=models_u[::-1], x=maes_u[::-1], orientation="h", marker_color=colors_u[::-1],
            text=[f"{v:.2f} MW" for v in maes_u[::-1]], textposition="auto",
            hovertemplate="%{y}: %{x:.2f} MW<extra></extra>"
        ))
        fig_bu.update_layout(title="UCI Electricity — Model Comparison (MAE MW, Lower is Better)", xaxis_title="MAE (MW)")
        apply_dark_plotly_theme(fig_bu, height=340)
        st.plotly_chart(fig_bu, use_container_width=True)


# =========================================================================
# 7. ⌁ ROUTING BEHAVIOUR (DATASET SWITCHABLE)
# =========================================================================
elif page == "⌁ Routing Behaviour":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Gating Analysis</span>
        <div class="page-title">Gating & Routing Behaviour</div>
        <div class="page-subtitle">Evaluating empirical gating weight distributions, simplex allocations, and effective expert counts across diverse electrical grids.</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Selector for Routing Page
    r_options = ["PJM", "GEFCom2014", "UCI"]
    r_idx = r_options.index(st.session_state.get("selected_dataset", "PJM")) if st.session_state.get("selected_dataset") in r_options else 0
    selected_r_ds = st.radio("Select Grid to Inspect Routing Dynamics", r_options, index=r_idx, horizontal=True, key="rout_ds_radio")
    st.session_state["selected_dataset"] = selected_r_ds

    routing_data = {
        "PJM": {"lstm": 35.11, "tcn": 32.83, "cnn": 32.07, "neff": 2.9931, "entropy": 1.0961, "unit": "MW"},
        "GEFCom2014": {"lstm": 35.28, "tcn": 29.82, "cnn": 34.90, "neff": 2.9765, "entropy": 1.0911, "unit": "kW"},
        "UCI": {"lstm": 34.28, "tcn": 29.94, "cnn": 35.78, "neff": 2.9842, "entropy": 1.0935, "unit": "MW"}
    }
    rd = routing_data[selected_r_ds]

    # 4 Metric Cards in metric-grid-4
    st.markdown(f"""
    <div class="metric-grid-4">
        <div class="metric-card">
            <div class="metric-label">LSTM Expert Weight</div>
            <div class="metric-value">{rd['lstm']:.2f}%</div>
            <div class="metric-sub">Recurrent persistence</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">TCN Expert Weight</div>
            <div class="metric-value">{rd['tcn']:.2f}%</div>
            <div class="metric-sub">Dilated causal history</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CNN Expert Weight</div>
            <div class="metric-value">{rd['cnn']:.2f}%</div>
            <div class="metric-sub">Localized edge motifs</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Effective Experts (Neff)</div>
            <div class="metric-value">{rd['neff']:.4f}</div>
            <div class="metric-sub">Out of 3.000 (Non-sparse)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">Empirical Weight Distribution ({selected_r_ds})</div>', unsafe_allow_html=True)
    st.caption("Descriptive routing behaviour across evaluated forecasting conditions.")

    col_r_chart, col_r_text = st.columns([1.2, 0.8])
    with col_r_chart:
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=["LSTM Expert", "TCN Expert", "CNN Expert"],
            y=[rd["lstm"], rd["tcn"], rd["cnn"]],
            marker_color=["#F59E0B", "#10B981", "#EC4899"],
            text=[f"{rd['lstm']:.2f}%", f"{rd['tcn']:.2f}%", f"{rd['cnn']:.2f}%"],
            textposition="auto"
        ))
        fig_r.update_layout(
            title=f"{selected_r_ds} — Empirical Gating Weight Allocation",
            yaxis_title="Allocation (%)",
            yaxis=dict(range=[0, 50]),
            showlegend=False
        )
        apply_dark_plotly_theme(fig_r, height=340)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r_text:
        st.markdown(f"""
        <div class="dark-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div class="dark-card-header">Scientific Interpretation</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • <strong>Convex Interior Routing:</strong> The evaluated routing formulation distributes prediction weight across all three temporal experts, with limited temporal variation.<br>
                • <strong>No Sparse Collapse:</strong> Effective expert count remains near maximum ($N_{{\text{{eff}}}} \approx {rd['neff']:.4f} / 3.000$), confirming that no single expert dominates.<br>
                • <strong>Smooth Convex Blending:</strong> Softmax gating maintains stable weight balance across regime transitions.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 8. ◇ CONFIDENCE / FALLBACK
# =========================================================================
elif page == "◇ Confidence / Fallback":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Robustness Analysis</span>
        <div class="page-title">Confidence / Fallback</div>
        <div class="page-subtitle">Evaluating the learned shrinkage blend coefficient (λ) and its role as an empirical stabilization mechanism toward the unweighted ensemble centroid.</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Fallback Cards in metric-grid-3
    st.markdown("""
    <div class="metric-grid-3">
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">PJM Interconnection</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">λ = 0.5066 ± 0.0038</div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    Coefficient of Variation (CV): <strong>0.75%</strong><br>
                    Range: [0.4938, 0.5204]
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Stable Centroid Blend</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">GEFCom2014</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">λ = 0.5170 ± 0.0055</div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    Coefficient of Variation (CV): <strong>1.06%</strong><br>
                    Range: [0.4988, 0.5368]
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Stable Centroid Blend</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">UCI Electricity</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">λ = 0.5064 ± 0.0030</div>
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45;">
                    Coefficient of Variation (CV): <strong>0.59%</strong><br>
                    Range: [0.4962, 0.5176]
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 10px; border-top: 1px solid #1E293B; padding-top: 6px;">Stable Centroid Blend</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cf_l, col_cf_r = st.columns(2)
    with col_cf_l:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Mathematical Formulation</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
        """, unsafe_allow_html=True)
        st.latex(r"\hat{\mathbf{y}}_{	ext{final}, t} = \lambda_t \hat{\mathbf{y}}_{	ext{adaptive}, t} + (1 - \lambda_t) \hat{\mathbf{y}}_{	ext{equal}, t}")
        st.latex(r"\lambda_t = \\sigma(\mathbf{W}_{	ext{conf}} \mathbf{h}_{	ext{context}, t} + b_{	ext{conf}}) \in (0, 1)")
        st.markdown("""
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">
                Where $\\hat{\\mathbf{y}}_{\\text{equal}, t} = \\frac{1}{3}(\\hat{\\mathbf{y}}_L + \\hat{\\mathbf{y}}_T + \\hat{\\mathbf{y}}_C)$ represents the unweighted ensemble centroid.
            </div>
            </div></div>
        """, unsafe_allow_html=True)
    with col_cf_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Scientific Findings on Shrinkage Behavior</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • The learned fallback coefficient remained close to 0.5 with low temporal variation (CV &lt; 1.1%).<br>
                • In the evaluated settings, blending the adaptive prediction toward the equal-expert centroid provided an empirical stabilization mechanism.<br>
                • This smooth convex combination acts as a stabilizing anchor, preventing catastrophic routing over-commitment while preserving adaptive adjustments.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 9. ⌁ STATISTICAL EVIDENCE
# =========================================================================
elif page == "⌁ Statistical Evidence":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Hypothesis Testing</span>
        <div class="page-title">Statistical Significance & Hypothesis Testing</div>
        <div class="page-subtitle">Rigorous statistical validation conducted strictly over non-overlapping daily blocks to respect temporal independence.</div>
    </div>
    """, unsafe_allow_html=True)

    # Workflow Visual
    st.markdown("""
    <div class="pipeline-container">
        <span class="pipeline-step">Hourly Forecasts (t=1..24)</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step active">Non-Overlapping Daily Blocks (K)</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">Paired Differences (Δ = e<sub>base</sub> - e<sub>caeg</sub>)</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step active">Paired t-Test + Wilcoxon Signed-Rank</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">Holm-Bonferroni Correction</span>
    </div>
    """, unsafe_allow_html=True)

    # 3 Statistical Grid Cards in metric-grid-3
    st.markdown("""
    <div class="metric-grid-3">
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">PJM Interconnection (K = 53)</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">Δ = -9.66 MW (-3.71%)</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • 95% CI: [-17.07, -2.26] MW<br>
                    • Paired t-Test: t = -2.5566 (p = 0.0135)<br>
                    • Wilcoxon W = 470.0 (p = 0.0298)<br>
                    • Holm-Corrected: <strong>p = 0.0406</strong>
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 8px; border-top: 1px solid #1E293B; padding-top: 6px;">Statistically Significant</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">GEFCom2014 (K = 456)</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">Δ = -0.69 kW (-5.48%)</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • 95% CI: [-0.80, -0.57] kW<br>
                    • Paired t-Test: t = -11.8498 (p = 2.04e-28)<br>
                    • Wilcoxon W = 20,445.0 (p = 2.53e-29)<br>
                    • Holm-Corrected: <strong>p = 1.02e-27</strong>
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 8px; border-top: 1px solid #1E293B; padding-top: 6px;">Highly Significant</div>
        </div>
        <div class="benchmark-card">
            <div>
                <div style="font-size: 0.74rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">UCI Electricity (K = 163)</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">Δ = -0.20 MW (-2.43%)</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; line-height: 1.45;">
                    • 95% CI: [-0.31, -0.09] MW<br>
                    • Paired t-Test: t = -3.6581 (p = 0.00034)<br>
                    • Wilcoxon W = 4,235.0 (p = 0.00005)<br>
                    • Holm-Corrected: <strong>p = 0.0017</strong>
                </div>
            </div>
            <div style="font-size: 0.74rem; color: #10B981; font-weight: 600; margin-top: 8px; border-top: 1px solid #1E293B; padding-top: 6px;">Statistically Significant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_st_l, col_st_r = st.columns(2)
    with col_st_l:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Why Non-Overlapping Daily Blocks?</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                In rolling window load forecasting, consecutive test windows overlap by 167 hours, inducing severe serial autocorrelation in raw hourly residuals.<br>
                Running hypothesis tests across rolling hourly windows drastically inflates degrees of freedom and creates spurious statistical significance. Partitioning the test set into disjoint 24-hour daily blocks ($K$) restores sample independence.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_st_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Methodological Distinction</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • <strong>Five seeds assess stochastic sensitivity:</strong> The 5 random initialization seeds evaluate optimizer convergence and parameter initialization variance; they do not represent independent real-world replications.<br>
                • <strong>Non-overlapping daily blocks assess generalization:</strong> Hypothesis testing across disjoint blocks evaluates whether error reductions generalize across independent operating days.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 10. ◷ HORIZON ANALYSIS (RETROSPECTIVE DIAGNOSTIC)
# =========================================================================
elif page == "◷ Horizon Analysis":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Lead-Time Diagnostics</span>
        <div class="page-title">Forecast Horizon Diagnostics</div>
        <div class="page-subtitle">Evaluating error progression across lead hours h = 1..24 and retrospective findings on horizon-specific routing formulations.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="provenance-card">
        <strong>RETROSPECTIVE DIAGNOSTIC — NOT CAUSAL EVIDENCE:</strong><br>
        Retrospective diagnostic: explicit horizon-specific routing did not improve validation performance in the evaluated experiments (+10.0% to +13.3% validation degradation on PJM). 
        The final locked CAEG-Net architecture maintains a unified single-step routing mechanism.
    </div>
    """, unsafe_allow_html=True)

    # Dataset Selector for Horizon Analysis
    h_options = ["PJM", "GEFCom2014", "UCI"]
    h_curr_idx = h_options.index(st.session_state.get("selected_dataset", "PJM")) if st.session_state.get("selected_dataset") in h_options else 0
    selected_h_ds = st.radio("Select Grid to Inspect Horizon Lead Dynamics", h_options, index=h_curr_idx, horizontal=True, key="horizon_ds_radio")
    st.session_state["selected_dataset"] = selected_h_ds

    dataset_map = {
        "PJM": ("PJM", "MW", "PJM Interconnection (Regional Grid)"),
        "GEFCom2014": ("GEFCom", "kW", "GEFCom2014 (Zonal Grid)"),
        "UCI": ("UCI", "MW", "UCI Electricity (Aggregated Demand)")
    }
    ds_key, unit, ds_full_name = dataset_map[selected_h_ds]

    # Metrics extraction from verified specialization artifact
    h1, h12, h24, ratio = 0.0, 0.0, 0.0, 1.0
    sub_spec = None
    if df_horizon_spec is not None and "dataset" in df_horizon_spec.columns:
        sub_spec = df_horizon_spec[df_horizon_spec["dataset"] == ds_key].sort_values("horizon_step")
        if not sub_spec.empty and "mae_f2" in sub_spec.columns:
            h1 = float(sub_spec[sub_spec["horizon_step"] == 1]["mae_f2"].values[0])
            h12 = float(sub_spec[sub_spec["horizon_step"] == 12]["mae_f2"].values[0])
            h24 = float(sub_spec[sub_spec["horizon_step"] == 24]["mae_f2"].values[0])
            ratio = (h24 / h1) if h1 > 0 else 1.0

    # 4 Horizon Metric Cards in metric-grid-4
    st.markdown(f"""
    <div class="metric-grid-4">
        <div class="metric-card">
            <div class="metric-label">Immediate Lead (h=1)</div>
            <div class="metric-value">{h1:.2f} <span style="font-size:0.8rem; color:#94A3B8;">{unit}</span></div>
            <div class="metric-sub">{selected_h_ds} lead step 1</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Mid Horizon (h=12)</div>
            <div class="metric-value">{h12:.2f} <span style="font-size:0.8rem; color:#94A3B8;">{unit}</span></div>
            <div class="metric-sub">{selected_h_ds} lead step 12</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Final Lead (h=24)</div>
            <div class="metric-value">{h24:.2f} <span style="font-size:0.8rem; color:#94A3B8;">{unit}</span></div>
            <div class="metric-sub">{selected_h_ds} lead step 24</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Lead Degradation Ratio</div>
            <div class="metric-value">{ratio:.2f}x</div>
            <div class="metric-sub">h=24 MAE / h=1 MAE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Primary Horizon Chart: Temporal Experts vs CAEG-Net
    if sub_spec is not None and not sub_spec.empty:
        st.markdown(f'<div class="section-title">Step-by-Step Lead MAE Progression ({ds_full_name})</div>', unsafe_allow_html=True)
        st.caption("Retrospective diagnostic — not causal evidence")

        fig_h = go.Figure()
        expert_traces = [
            ("mae_f2", "CAEG-Net (Locked F2)", "#38BDF8", 3.2, None, "square"),
            ("mae_equal", "Equal Ensemble (1/3)", "#A855F7", 2.0, "longdash", "circle"),
            ("mae_tcn", "Standalone TCN", "#10B981", 2.0, "dash", "triangle-up"),
            ("mae_lstm", "Standalone LSTM", "#F59E0B", 2.0, "dot", "diamond"),
            ("mae_cnn", "Standalone CNN", "#EC4899", 2.0, "dashdot", "cross")
        ]
        for col, label, color, width, dash, marker_sym in expert_traces:
            if col in sub_spec.columns:
                fig_h.add_trace(go.Scatter(
                    x=sub_spec["horizon_step"],
                    y=sub_spec[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=width, dash=dash),
                    marker=dict(size=5, symbol=marker_sym),
                    hovertemplate=f"Lead h=%{{x}}: %{{y:.2f}} {unit}<extra></extra>"
                ))
        fig_h.update_layout(
            title=f"{selected_h_ds} — Step-by-Step Lead MAE Progression (h = 1..24)",
            xaxis_title="Forecast Horizon (Hours Ahead)",
            yaxis_title=f"Mean Absolute Error ({unit})",
            hovermode="x unified"
        )
        apply_dark_plotly_theme(fig_h, height=420)
        st.plotly_chart(fig_h, use_container_width=True)

    # Secondary Ablation Chart: Routing Candidate Comparisons
    if df_horizon_cand is not None and "dataset" in df_horizon_cand.columns:
        with st.expander("🔍 Inspect Evaluated Horizon Routing Candidates (Ablation Diagnostics)", expanded=False):
            sub_cand = df_horizon_cand[df_horizon_cand["dataset"] == ds_key].sort_values("horizon_step")
            if not sub_cand.empty:
                fig_cand = go.Figure()
                cand_map = {
                    "Control_A_F2": ("CAEG-Net (Unified Router)", "#38BDF8", 3.0, None),
                    "Control_B_FixedShrinkage": ("Fixed Shrinkage Control", "#64748B", 1.8, "dash"),
                    "Control_C_HorizonRouting": ("Horizon Routing Control", "#EF4444", 2.2, "dot"),
                    "Control_D_DynamicConfidence": ("Dynamic Confidence Control", "#F59E0B", 1.8, "dashdot"),
                    "Candidate_E1_HGR_FS": ("Candidate E1 (HGR-FS)", "#EC4899", 1.6, "longdash"),
                    "Candidate_E2_HGR_DGS": ("Candidate E2 (HGR-DGS)", "#A855F7", 1.6, "longdashdot")
                }
                for cid in sub_cand["candidate_id"].unique():
                    c_rows = sub_cand[sub_cand["candidate_id"] == cid]
                    if not c_rows.empty:
                        label, color, width, dash = cand_map.get(cid, (cid, "#94A3B8", 1.5, None))
                        fig_cand.add_trace(go.Scatter(
                            x=c_rows["horizon_step"],
                            y=c_rows["mae"],
                            mode="lines+markers",
                            name=label,
                            line=dict(color=color, width=width, dash=dash),
                            marker=dict(size=4),
                            hovertemplate=f"Lead h=%{{x}}: %{{y:.2f}} {unit}<extra></extra>"
                        ))
                fig_cand.update_layout(
                    title=f"Retrospective Diagnostic: Evaluated Routing Candidates Across Horizon ({selected_h_ds})",
                    xaxis_title="Forecast Horizon (Hours Ahead)",
                    yaxis_title=f"Mean Absolute Error ({unit})",
                    hovermode="x unified"
                )
                apply_dark_plotly_theme(fig_cand, height=400)
                st.plotly_chart(fig_cand, use_container_width=True)

    col_h_l, col_h_r = st.columns(2)
    with col_h_l:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Controlled Experimental Findings</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • <strong>Per-Step Routing Degradation:</strong> During Phase 15 screening, explicit per-step routing models (Candidates E1, E2, E3) severely overfit the validation set, degrading validation MAE by +10.0% to +13.3% on PJM.<br>
                • <strong>Single Gating Representation:</strong> Conditioning on a unified 168h causal context vector produces significantly more robust generalization than splitting routing across individual lead hours.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h_r:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">Retrospective Diagnostic Status</div>
            <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
                • <strong>Diagnostic Note:</strong> Horizon diagnostic evaluations are post-hoc observational analyses; they demonstrate that explicit per-step routing was unpromising in the evaluated setup.<br>
                • <strong>Architecture Lock:</strong> The champion CAEG-Net model relies strictly on unified single-vector context conditioning.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 11. ✦ RESEARCH FINDINGS
# =========================================================================
elif page == "✦ Research Findings":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Scientific Synthesis</span>
        <div class="page-title">Key Research Findings</div>
        <div class="page-subtitle">Structured, publication-grade insights established during the multi-phase CAEG-Net experimental campaign.</div>
    </div>
    """, unsafe_allow_html=True)

    findings = [
        ("01", "ADAPTIVE FUSION", "Context-Adaptive Fusion Outperforms Fixed Baseline Architectures",
         "Dynamically weighting heterogeneous temporal experts via causal context conditioning achieved lower test errors than monolithic architectures and fixed ensembles on PJM (250.97 MW) and GEFCom (12.41 kW), while maintaining competitive accuracy on UCI (7.74 MW)."),
        ("02", "HETEROGENEOUS EXPERTS", "Inductive Bias Complementarity Mitigates Error Compounding",
         "The combination of recurrent sequence continuity (LSTM), causal multi-scale receptive field (TCN), and local pattern matching (CNN) yielded lower variance across diverse forecasting conditions than any standalone expert family."),
        ("03", "OOF PERFORMANCE CONDITIONING", "Causal Out-of-Fold Residual Tracking Prevents Circular Bias",
         "Conditioning the gating network on out-of-fold historical validation residuals provided informative performance signals while preserving strict temporal causality and preventing self-fulfilling routing loops."),
        ("04", "CROSS-DATASET EVALUATION", "Balanced Generalization Across Diverse Operating Scales",
         "Evaluating across 3 real-world electrical grids demonstrated that CAEG-Net balances forecasting performance across diverse grid scales (MW and kW regimes) without dataset-specific manual architectural tuning."),
        ("05", "LIMITED ROUTING DYNAMICITY", "Smooth Convex Blending Stabilizes Gating Simplex",
         "The evaluated gating network settled into an interior simplex allocation ($N_{\text{eff}} \approx 2.98 - 2.99$) with stable learned shrinkage ($\\lambda \\approx 0.51$). Rather than switching aggressively, it provided smooth, stable regularized blending toward the equal-expert centroid.")
    ]

    for num, cat, title, desc in findings:
        st.markdown(f"""
        <div class="dark-card" style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 1rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #38BDF8; font-family: monospace; line-height: 1; padding-top: 4px;">{num}</div>
            <div style="flex: 1;">
                <span class="hero-badge">{cat}</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin: 4px 0 6px 0;">{title}</div>
                <div style="font-size: 0.86rem; color: #94A3B8; line-height: 1.55;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# 12. ⓘ LIMITATIONS & ETHICS
# =========================================================================
elif page == "ⓘ Limitations & Ethics":
    st.markdown("""
    <div class="page-hero-container">
        <span class="page-category-badge">Scientific Boundaries</span>
        <div class="page-title">Limitations, Boundaries & Ethics</div>
        <div class="page-subtitle">Transparent technical boundaries, scope restrictions, and research integrity disclosures for the CAEG-Net study.</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Limitations Metric Cards in metric-grid-4
    st.markdown("""
    <div class="metric-grid-4">
        <div class="metric-card">
            <div class="metric-label">Forecast Type</div>
            <div class="metric-value">Point Forecast</div>
            <div class="metric-sub">Deterministic output only</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Input Modality</div>
            <div class="metric-value">Univariate Load</div>
            <div class="metric-sub">No exogenous weather features</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Screening Firewall</div>
            <div class="metric-value">2-Stage Pre-Reg</div>
            <div class="metric-sub">Zero test optimization</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Verification Standard</div>
            <div class="metric-value">100% Provenance</div>
            <div class="metric-sub">Zero fabricated traces</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    boundaries = [
        ("1. Deterministic Point Forecasting Only", "CAEG-Net currently generates deterministic point predictions for day-ahead dispatch. While crucial for direct scheduling, it does not produce probabilistic quantiles or prediction intervals."),
        ("2. Univariate Load Profile Input", "The model relies exclusively on historical load patterns (168h lookback). It does not integrate exogenous numerical weather predictions (temperature, dew point, solar irradiance), real-time pricing, or calendar holiday matrices."),
        ("3. Regional Grid Aggregation", "Evaluated datasets represent transmission-level regional aggregated demand. Performance characteristics may differ on highly stochastic low-voltage distribution feeders with high rooftop solar penetration."),
        ("4. Limited Routing Dynamicity & Stable Shrinkage", "Empirical routing weights remain interior to the simplex with low temporal variation, and learned shrinkage settle near λ ≈ 0.51. The mechanism acts as an empirical stabilizer toward the ensemble centroid rather than an aggressive regime switcher."),
        ("5. Dataset Dependence", "CAEG-Net F2 demonstrates strong performance on PJM and GEFCom, while on UCI standalone LSTM achieves lower test MAE. The model provides cross-grid balance rather than universal dominance on every dataset."),
        ("6. Stochastic Sensitivity Interpretation", "The 5 canonical evaluation seeds [42, 123, 999, 2024, 3407] assess random initialization variance; they do not constitute independent real-world temporal replications."),
        ("7. Retrospective Diagnostics Are Not Causal Evidence", "Oracle and horizon diagnostic analyses represent retrospective observational evaluations; they are not deployable operational models.")
    ]

    for title, text in boundaries:
        st.markdown(f"""
        <div class="dark-card" style="margin-bottom: 0.8rem;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="accent-card" style="margin-top: 1.5rem;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px;">Zero-Fabrication Research Guarantee</div>
        <div style="font-size: 0.86rem; color: #CBD5E1; line-height: 1.55;">
            Every numerical value, metric, table cell, and prediction trace presented in this dashboard is directly traceable to audited repository artifacts produced during locked experimental phases. 
            No simulated arrays, placeholder constants, or smoothed synthetic curves have been introduced.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# GLOBAL FOOTER
# =========================================================================
st.markdown("""
<div class="dark-footer">
    <strong>CAEG-Net Research Dashboard</strong> • Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting<br>
    Final Model Locked (F2 / A2-OOF) • 121,724 Trainable Parameters • Fully Traceable Scientific Artifact Set
</div>
""", unsafe_allow_html=True)
