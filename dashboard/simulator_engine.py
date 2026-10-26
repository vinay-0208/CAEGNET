"""
Canonical CAEG-Net V1 Interactive Forecast Simulator Engine
===========================================================
Provides verified checkpoint loading, strict input validation (text/table and CSV),
train-only standardization, causal 4D context extraction (including exact walk-forward
Recent Error when >= 192 contiguous hours are supplied and disclosed training-prior
fallback when 168..191 hours are supplied), live PyTorch inference in evaluation mode,
CSV export generation, and historical test-window live-vs-cache parity verification.

Non-Negotiable Scientific Safeguards Enforced:
- Uses strictly the frozen Canonical CAEG-Net V1 architecture from root `caeg_net.py`
  (121,531 parameters; 168h lookback -> 24h horizon; 4D context [Trend, Volatility,
  Periodicity, Recent_Error]).
- Loads `checkpoints/seed_{seed}/caeg_full.pt` with `strict=True`.
- Never fits a new scaler on user input; uses exclusively the PJM training scaler.
- Never fabricates recent errors or touches future observations.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

import caeg_net
import data_utils


# =====================================================================
# 1. Constants & Authoritative Canonical V1 Specification
# =====================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PJM_CSV_PATH = os.path.join(REPO_ROOT, "data", "Modern_PJM", "pjm_load.csv")
PJM_CONTEXT_NPZ_PATH = os.path.join(REPO_ROOT, "dashboard", "assets", "data", "pjm_test_context.npz")
PHASE5_CACHE_NPZ_PATH = os.path.join(REPO_ROOT, "results", "phase5_multiseed_cache.npz")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")

SUPPORTED_SEEDS: Tuple[int, ...] = (42, 123, 999, 2024, 3407)
LOOKBACK_HOURS: int = 168
HORIZON_HOURS: int = 24
EXACT_RECENT_ERROR_MIN_HOURS: int = LOOKBACK_HOURS + HORIZON_HOURS  # 192 hours
CONTEXT_FEATURE_NAMES: Tuple[str, str, str, str] = (
    "Trend",
    "Volatility",
    "Periodicity",
    "Recent_Error",
)

EXPECTED_TOTAL_PARAMS: int = 121531
EXPECTED_SUBMODULE_PARAMS: Dict[str, int] = {
    "lstm_expert": 56152,
    "tcn_expert": 36952,
    "cnn_expert": 27400,
    "context_encoder": 384,
    "gating_network": 643,
}

# Physical validity bounds for PJM regional transmission load (MW)
MIN_PHYSICAL_LOAD_MW: float = 1.0
MAX_PHYSICAL_LOAD_MW: float = 100000.0

# Recognised load column names for strict CSV validation (case-insensitive)
RECOGNIZED_LOAD_COLUMNS: Tuple[str, ...] = (
    "load",
    "load_mw",
    "mw",
    "demand",
    "demand_mw",
    "pjm_load",
    "electricity_load",
)

# Recognised timestamp column names for strict CSV validation (case-insensitive)
RECOGNIZED_TIMESTAMP_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "datetime",
    "date",
    "time",
    "datetime_beginning_utc",
    "datetime_beginning_ept",
)


# =====================================================================
# 2. Data Structures
# =====================================================================

class InputValidationError(ValueError):
    """Raised when user-supplied load inputs fail scientific or physical validation."""


@dataclass
class ValidatedSimulatorInput:
    """Validated user input ready for train-only scaling and causal context extraction."""
    full_series_mw: np.ndarray          # All valid contiguous observations [M] (M >= 168)
    lookback_168_mw: np.ndarray         # Latest 168 observations [168] used as model input X
    total_observations: int             # M
    input_source: str                   # e.g., "custom_text", "custom_csv", "historical_test_window"
    is_hypothetical_scenario: bool      # True for custom user inputs; False for historical PJM window
    timestamps_all: Optional[List[str]] = None
    timestamps_lookback_168: Optional[List[str]] = None
    timestamps_horizon_24: Optional[List[str]] = None
    distribution_warnings: List[str] = field(default_factory=list)
    load_column_used: Optional[str] = None
    timestamp_column_used: Optional[str] = None


@dataclass
class PJMTrainingReference:
    """Authoritative PJM training scaler, walk-forward Ridge forecaster, and test partition."""
    scaler_mean: float
    scaler_scale: float
    warmup_prior_mae: float
    forecaster: data_utils.ChronologicalWalkForwardForecaster
    train_min_mw: float
    train_max_mw: float
    train_p01_mw: float
    train_p99_mw: float
    train_mean_mw: float
    train_std_mw: float
    X_test_std: np.ndarray              # [1294, 168, 1] standardized
    Y_test_raw: np.ndarray              # [1294, 24] raw MW
    C_test: np.ndarray                  # [1294, 4] causal context
    rec_err_test: np.ndarray            # [1294] causal recent error
    test_origin_timestamps: List[str]   # [1294] ISO strings
    test_series_extended_mw: np.ndarray # [167 + 1318] raw MW
    test_timestamps_extended: List[str] # [167 + 1318] ISO strings


@dataclass
class LiveForecastResult:
    """Complete output of a live Canonical CAEG-Net V1 inference execution."""
    seed_label: str                     # e.g. "42" or "5-Seed Ensemble (42, 123, 999, 2024, 3407)"
    checkpoint_paths: List[str]
    model_parameter_count: int
    input_source: str
    is_hypothetical_scenario: bool
    total_input_observations: int
    units: str
    lookback_168_mw: np.ndarray         # [168]
    y_caeg_mw: np.ndarray               # [24]
    y_lstm_mw: np.ndarray               # [24]
    y_tcn_mw: np.ndarray                # [24]
    y_cnn_mw: np.ndarray                # [24]
    y_equal_mw: np.ndarray              # [24]
    routing_weights: np.ndarray         # [3] -> [w_LSTM, w_TCN, w_CNN]
    routing_weights_sum: float
    routing_weights_valid: bool
    context_vector: np.ndarray          # [4] -> [Trend, Volatility, Periodicity, Recent_Error]
    context_dict: Dict[str, float]
    recent_error_mode: str              # "exact_causal_walk_forward" | "training_warmup_prior_fallback" | "historical_test_partition_exact"
    recent_error_fallback_used: bool
    recent_error_explanation: str
    scaler_mean_mw: float
    scaler_scale_mw: float
    distribution_warnings: List[str]
    timestamps_lookback_168: Optional[List[str]] = None
    timestamps_horizon_24: Optional[List[str]] = None
    # Populated only for 5-seed ensemble mode
    per_seed_caeg_mw: Optional[Dict[int, np.ndarray]] = None
    y_caeg_std_mw: Optional[np.ndarray] = None
    routing_weights_std: Optional[np.ndarray] = None
    # Populated only for optional historical test-window validation mode
    historical_window_idx: Optional[int] = None
    historical_y_true_mw: Optional[np.ndarray] = None
    historical_cached_pred_mw: Optional[np.ndarray] = None
    historical_cached_weights: Optional[np.ndarray] = None
    historical_max_abs_pred_diff_mw: Optional[float] = None
    historical_max_abs_weight_diff: Optional[float] = None
    historical_parity_passed: Optional[bool] = None
    historical_retrospective_metrics: Optional[Dict[str, float]] = None


# =====================================================================
# 3. Authoritative Model Verification & Loading
# =====================================================================

_MODEL_CACHE: Dict[int, caeg_net.CAEGNet] = {}
_PJM_REF_CACHE: Optional[PJMTrainingReference] = None


def get_checkpoint_path(seed: int) -> str:
    """Return absolute path to the Canonical V1 checkpoint for a given seed."""
    if seed not in SUPPORTED_SEEDS:
        raise ValueError(f"Unsupported seed {seed}. Supported seeds are {SUPPORTED_SEEDS}.")
    return os.path.join(CHECKPOINTS_DIR, f"seed_{seed}", "caeg_full.pt")


def verify_and_load_canonical_v1_model(
    seed: int,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> Tuple[caeg_net.CAEGNet, str, Dict[str, Any]]:
    """
    Instantiate the authoritative Canonical V1 `caeg_net.CAEGNet` model, verify its
    exact parameter count (121,531) and submodule breakdown, and load the trained
    checkpoint with `strict=True`.
    """
    if seed not in SUPPORTED_SEEDS:
        raise ValueError(f"Seed {seed} is not in supported seeds {SUPPORTED_SEEDS}.")

    if device is None:
        device = torch.device("cpu")

    ckpt_path = get_checkpoint_path(seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Canonical V1 checkpoint not found at: {ckpt_path}")

    if use_cache and seed in _MODEL_CACHE and device.type == "cpu":
        cached_model = _MODEL_CACHE[seed]
        cached_model.eval()
        counts = caeg_net.count_parameters(cached_model)
        return cached_model, ckpt_path, counts

    model = caeg_net.CAEGNet(
        input_dim=1,
        horizon=HORIZON_HOURS,
        context_dim=4,
        latent_context_dim=16,
        lstm_hidden=64,
        lstm_layers=2,
        tcn_channels=32,
        dropout=0.1,
        horizon_dependent=False,
    )

    counts = caeg_net.count_parameters(model)
    if counts["total_params"] != EXPECTED_TOTAL_PARAMS or counts["total_trainable"] != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"Canonical V1 parameter count mismatch: expected {EXPECTED_TOTAL_PARAMS}, "
            f"got total={counts['total_params']}, trainable={counts['total_trainable']}."
        )

    for sub_name, exp_cnt in EXPECTED_SUBMODULE_PARAMS.items():
        actual_cnt = counts["submodules"].get(sub_name)
        if actual_cnt != exp_cnt:
            raise RuntimeError(
                f"Canonical V1 submodule '{sub_name}' parameter mismatch: "
                f"expected {exp_cnt}, got {actual_cnt}."
            )

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed for {ckpt_path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("Model failed to switch to evaluation mode (model.eval()).")

    if use_cache and device.type == "cpu":
        _MODEL_CACHE[seed] = model

    return model, ckpt_path, counts


# =====================================================================
# 4. Authoritative PJM Training Pipeline & Reference Loader
# =====================================================================

def load_pjm_training_reference(use_cache: bool = True) -> PJMTrainingReference:
    """
    Load the PJM dataset using the authoritative `data_utils.py` pipeline,
    verify exact agreement with the saved `pjm_test_context.npz` scaler, and
    fit the causal `ChronologicalWalkForwardForecaster` on the training split.
    """
    global _PJM_REF_CACHE
    if use_cache and _PJM_REF_CACHE is not None:
        return _PJM_REF_CACHE

    if not os.path.isfile(PJM_CSV_PATH):
        raise FileNotFoundError(
            f"Authoritative PJM dataset file not found at {PJM_CSV_PATH}. "
            "Cannot verify training scaler and causal walk-forward forecaster."
        )

    df, _ = data_utils.load_and_clean_data(PJM_CSV_PATH)
    train_df, val_df, test_df, _ = data_utils.chronological_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    scaler, train_sc, val_sc, test_sc = data_utils.fit_and_transform_scaler(
        train_df, val_df, test_df, load_col="load"
    )

    scaler_mean = float(scaler.mean_[0])
    scaler_scale = float(scaler.scale_[0])

    # Cross-verify against bundled pjm_test_context.npz if present
    if os.path.isfile(PJM_CONTEXT_NPZ_PATH):
        ctx_npz = np.load(PJM_CONTEXT_NPZ_PATH)
        npz_mean = float(ctx_npz["scaler_mean"])
        npz_scale = float(ctx_npz["scaler_scale"])
        if not (np.isclose(scaler_mean, npz_mean, rtol=1e-6) and np.isclose(scaler_scale, npz_scale, rtol=1e-6)):
            raise RuntimeError(
                f"Training scaler mismatch vs pjm_test_context.npz: "
                f"computed ({scaler_mean}, {scaler_scale}) vs npz ({npz_mean}, {npz_scale})."
            )

    windows = data_utils.create_partition_windows_with_context(
        train_sc, val_sc, test_sc, lookback=LOOKBACK_HOURS, horizon=HORIZON_HOURS, load_col="load"
    )
    _, _, rec_err_test, forecaster = data_utils.compute_causal_recent_forecast_errors(windows)
    C_test = data_utils.extract_context_features(windows["test"]["X"], rec_err_test)

    X_test_std = windows["test"]["X"].astype(np.float32)
    Y_test_raw = (windows["test"]["Y"].astype(np.float64) * scaler_scale + scaler_mean).astype(np.float32)

    # Extended test series & timestamps for historical window inspection
    val_tail_mw = val_df["load"].values[-(LOOKBACK_HOURS - 1):].astype(np.float64)
    test_mw = test_df["load"].values.astype(np.float64)
    test_series_extended_mw = np.concatenate([val_tail_mw, test_mw])

    val_tail_ts = [str(ts) for ts in val_df["timestamp"].iloc[-(LOOKBACK_HOURS - 1):]]
    test_ts = [str(ts) for ts in test_df["timestamp"]]
    test_timestamps_extended = val_tail_ts + test_ts

    num_test_windows = X_test_std.shape[0]
    test_origin_timestamps = [str(test_df["timestamp"].iloc[k]) for k in range(num_test_windows)]

    train_loads = train_df["load"].values.astype(np.float64)
    ref = PJMTrainingReference(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        warmup_prior_mae=float(forecaster.warmup_prior_mae),
        forecaster=forecaster,
        train_min_mw=float(np.min(train_loads)),
        train_max_mw=float(np.max(train_loads)),
        train_p01_mw=float(np.percentile(train_loads, 1)),
        train_p99_mw=float(np.percentile(train_loads, 99)),
        train_mean_mw=float(np.mean(train_loads)),
        train_std_mw=float(np.std(train_loads)),
        X_test_std=X_test_std,
        Y_test_raw=Y_test_raw,
        C_test=C_test,
        rec_err_test=rec_err_test,
        test_origin_timestamps=test_origin_timestamps,
        test_series_extended_mw=test_series_extended_mw,
        test_timestamps_extended=test_timestamps_extended,
    )

    if use_cache:
        _PJM_REF_CACHE = ref
    return ref


# =====================================================================
# 5. Input Validation (Text, Editable Table, and CSV Upload)
# =====================================================================

def check_pjm_distribution_warnings(
    lookback_168_mw: np.ndarray,
    ref: PJMTrainingReference,
) -> List[str]:
    """
    Check whether the 168-hour input window falls substantially outside the
    PJM training distribution. Returns non-blocking informational warnings.
    """
    warnings_list: List[str] = []
    w_min = float(np.min(lookback_168_mw))
    w_max = float(np.max(lookback_168_mw))
    w_mean = float(np.mean(lookback_168_mw))

    low_3sigma = ref.train_mean_mw - 3.0 * ref.train_std_mw
    high_3sigma = ref.train_mean_mw + 3.0 * ref.train_std_mw

    if w_min < ref.train_min_mw or w_max > ref.train_max_mw:
        warnings_list.append(
            f"Input load profile contains observations ([{w_min:,.2f}, {w_max:,.2f}] MW) outside the "
            f"observed PJM training range ([{ref.train_min_mw:,.2f}, {ref.train_max_mw:,.2f}] MW). "
            "This indicates an extrapolated hypothetical scenario rather than proof that the model is invalid."
        )
    elif w_min < low_3sigma or w_max > high_3sigma:
        warnings_list.append(
            f"Input load profile reaches extreme tail values ([{w_min:,.2f}, {w_max:,.2f}] MW) beyond "
            f"±3σ of the PJM training distribution ([{low_3sigma:,.2f}, {high_3sigma:,.2f}] MW). "
            "This distribution notice reflects operating regime shift and does not imply model invalidity."
        )

    if abs(w_mean - ref.train_mean_mw) > 2.5 * ref.train_std_mw:
        warnings_list.append(
            f"168-hour mean load ({w_mean:,.2f} MW) deviates by more than 2.5σ from the PJM training "
            f"mean ({ref.train_mean_mw:,.2f} ± {ref.train_std_mw:,.2f} MW)."
        )

    return warnings_list


def _validate_numeric_load_array(
    values: np.ndarray,
    ref: Optional[PJMTrainingReference] = None,
    input_source: str = "custom_text",
    is_hypothetical_scenario: bool = True,
    timestamps_all: Optional[List[str]] = None,
    load_column_used: Optional[str] = None,
    timestamp_column_used: Optional[str] = None,
) -> ValidatedSimulatorInput:
    """Validate a 1D array of hourly load values against all physical and statistical rules."""
    if values.ndim != 1:
        values = values.flatten()

    n = len(values)
    if n < LOOKBACK_HOURS:
        raise InputValidationError(
            f"Insufficient historical observations: received {n} hourly values, "
            f"but Canonical CAEG-Net V1 requires at least {LOOKBACK_HOURS} contiguous hourly observations."
        )

    if not np.all(np.isfinite(values)):
        bad_indices = np.where(~np.isfinite(values))[0]
        first_bad = int(bad_indices[0]) + 1
        raise InputValidationError(
            f"Input contains NaN or infinite values (first occurrence at position {first_bad})."
        )

    if np.any(values < MIN_PHYSICAL_LOAD_MW):
        bad_idx = int(np.where(values < MIN_PHYSICAL_LOAD_MW)[0][0])
        bad_val = float(values[bad_idx])
        raise InputValidationError(
            f"Physically invalid electricity load value ({bad_val} MW at position {bad_idx + 1}). "
            f"Under the PJM regional grid validation policy, hourly load must be >= {MIN_PHYSICAL_LOAD_MW:.1f} MW."
        )

    if np.any(values > MAX_PHYSICAL_LOAD_MW):
        bad_idx = int(np.where(values > MAX_PHYSICAL_LOAD_MW)[0][0])
        bad_val = float(values[bad_idx])
        raise InputValidationError(
            f"Physically invalid electricity load value ({bad_val:,.2f} MW at position {bad_idx + 1}). "
            f"Under the PJM regional grid validation policy, hourly load must be <= {MAX_PHYSICAL_LOAD_MW:,.1f} MW."
        )

    lookback_168 = values[-LOOKBACK_HOURS:].astype(np.float64)
    if float(np.std(lookback_168, ddof=1)) < 1e-6:
        raise InputValidationError(
            "The 168-hour lookback window is constant (zero variance). "
            "Normalized OLS trend and lag-24 autocorrelation require non-constant load observations."
        )

    ts_lookback_168: Optional[List[str]] = None
    ts_horizon_24: Optional[List[str]] = None
    if timestamps_all is not None and len(timestamps_all) == n:
        ts_lookback_168 = timestamps_all[-LOOKBACK_HOURS:]
        last_ts = pd.to_datetime(ts_lookback_168[-1])
        future_range = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=HORIZON_HOURS,
            freq="h",
        )
        ts_horizon_24 = [str(ts) for ts in future_range]

    dist_warnings: List[str] = []
    if ref is not None:
        dist_warnings = check_pjm_distribution_warnings(lookback_168, ref)

    return ValidatedSimulatorInput(
        full_series_mw=values.astype(np.float64),
        lookback_168_mw=lookback_168,
        total_observations=n,
        input_source=input_source,
        is_hypothetical_scenario=is_hypothetical_scenario,
        timestamps_all=timestamps_all,
        timestamps_lookback_168=ts_lookback_168,
        timestamps_horizon_24=ts_horizon_24,
        distribution_warnings=dist_warnings,
        load_column_used=load_column_used,
        timestamp_column_used=timestamp_column_used,
    )


def parse_and_validate_text_input(
    raw_text: str,
    ref: Optional[PJMTrainingReference] = None,
) -> ValidatedSimulatorInput:
    """
    Parse newline-, comma-, or semicolon-separated hourly load values from text
    and validate against all physical and numerical rules.
    """
    if raw_text is None or not str(raw_text).strip():
        raise InputValidationError(
            f"No load values provided. Please enter at least {LOOKBACK_HOURS} hourly load values in MW."
        )

    normalized = str(raw_text).replace(";", ",").replace("\r", "\n")
    tokens: List[str] = []
    for line in normalized.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        for part in line_clean.split(","):
            p = part.strip()
            if p:
                tokens.append(p)

    if not tokens:
        raise InputValidationError(
            f"No numeric load values found. Please enter at least {LOOKBACK_HOURS} hourly load values in MW."
        )

    parsed_floats: List[float] = []
    for idx, tok in enumerate(tokens):
        try:
            val = float(tok)
        except ValueError as exc:
            raise InputValidationError(
                f"Non-numeric value '{tok}' encountered at item #{idx + 1}. "
                "All hourly observations must be valid numbers."
            ) from exc
        parsed_floats.append(val)

    values = np.asarray(parsed_floats, dtype=np.float64)
    return _validate_numeric_load_array(
        values=values,
        ref=ref,
        input_source="custom_text",
        is_hypothetical_scenario=True,
    )


def parse_and_validate_Sequence_input(
    values_seq: Sequence[Any],
    ref: Optional[PJMTrainingReference] = None,
    input_source: str = "custom_table",
) -> ValidatedSimulatorInput:
    """Validate a sequence or editable table column of hourly load values."""
    if values_seq is None or len(values_seq) == 0:
        raise InputValidationError(
            f"Empty input table. At least {LOOKBACK_HOURS} hourly load values are required."
        )

    parsed_floats: List[float] = []
    for idx, item in enumerate(values_seq):
        if item is None or (isinstance(item, str) and not item.strip()):
            raise InputValidationError(
                f"Missing/empty value at row #{idx + 1}. All hourly observations must be present."
            )
        try:
            val = float(item)
        except (ValueError, TypeError) as exc:
            raise InputValidationError(
                f"Non-numeric value '{item}' at row #{idx + 1}. All hourly observations must be numeric."
            ) from exc
        parsed_floats.append(val)

    values = np.asarray(parsed_floats, dtype=np.float64)
    return _validate_numeric_load_array(
        values=values,
        ref=ref,
        input_source=input_source,
        is_hypothetical_scenario=True,
    )


def parse_and_validate_csv_input(
    csv_source: Union[str, bytes, io.IOBase, pd.DataFrame],
    ref: Optional[PJMTrainingReference] = None,
    explicit_load_col: Optional[str] = None,
    explicit_timestamp_col: Optional[str] = None,
) -> ValidatedSimulatorInput:
    """
    Parse and strictly validate a user-uploaded CSV (or DataFrame) for custom forecasting.
    - Requires a clearly identified load column (either matching RECOGNIZED_LOAD_COLUMNS
      or explicitly specified by the user; rejects ambiguous schemas).
    - If a timestamp column is present, validates parseability, strict ascending order,
      absence of duplicates, and exact 1-hour continuity without gaps.
    """
    if isinstance(csv_source, pd.DataFrame):
        df = csv_source.copy()
    elif isinstance(csv_source, bytes):
        try:
            df = pd.read_csv(io.BytesIO(csv_source))
        except Exception as exc:
            raise InputValidationError(f"Failed to parse CSV file: {exc}") from exc
    elif isinstance(csv_source, str) and not os.path.exists(csv_source) and ("\n" in csv_source or "," in csv_source):
        try:
            df = pd.read_csv(io.StringIO(csv_source))
        except Exception as exc:
            raise InputValidationError(f"Failed to parse CSV content: {exc}") from exc
    else:
        try:
            df = pd.read_csv(csv_source)
        except Exception as exc:
            raise InputValidationError(f"Failed to read CSV file: {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise InputValidationError("Uploaded CSV contains no rows or columns.")

    col_map_lower = {str(c).strip().lower(): c for c in df.columns}

    # 1. Identify Load Column strictly (never guess silently among arbitrary columns)
    load_col: Optional[str] = None
    if explicit_load_col is not None and explicit_load_col.strip():
        if explicit_load_col in df.columns:
            load_col = explicit_load_col
        elif explicit_load_col.strip().lower() in col_map_lower:
            load_col = col_map_lower[explicit_load_col.strip().lower()]
        else:
            raise InputValidationError(
                f"Specified load column '{explicit_load_col}' not found in CSV columns: {list(df.columns)}."
            )
    else:
        matched_load_cols = [
            orig_col
            for low_name, orig_col in col_map_lower.items()
            if low_name in RECOGNIZED_LOAD_COLUMNS
        ]
        if len(matched_load_cols) == 1:
            load_col = matched_load_cols[0]
        elif len(matched_load_cols) > 1:
            raise InputValidationError(
                f"Ambiguous CSV schema: multiple candidate load columns found ({matched_load_cols}). "
                "Please provide a CSV with a single clearly identified load column (e.g., 'load' or 'load_mw')."
            )
        else:
            raise InputValidationError(
                f"Could not identify a recognized load column in CSV columns {list(df.columns)}. "
                f"Expected one of: {list(RECOGNIZED_LOAD_COLUMNS)}."
            )

    # 2. Identify & Validate Timestamp Column if present
    ts_col: Optional[str] = None
    if explicit_timestamp_col is not None and explicit_timestamp_col.strip():
        if explicit_timestamp_col in df.columns:
            ts_col = explicit_timestamp_col
        elif explicit_timestamp_col.strip().lower() in col_map_lower:
            ts_col = col_map_lower[explicit_timestamp_col.strip().lower()]
        else:
            raise InputValidationError(
                f"Specified timestamp column '{explicit_timestamp_col}' not found in CSV columns: {list(df.columns)}."
            )
    else:
        matched_ts_cols = [
            orig_col
            for low_name, orig_col in col_map_lower.items()
            if low_name in RECOGNIZED_TIMESTAMP_COLUMNS and orig_col != load_col
        ]
        if len(matched_ts_cols) == 1:
            ts_col = matched_ts_cols[0]
        elif len(matched_ts_cols) > 1:
            raise InputValidationError(
                f"Ambiguous CSV schema: multiple candidate timestamp columns found ({matched_ts_cols})."
            )

    timestamps_list: Optional[List[str]] = None
    if ts_col is not None:
        raw_ts = df[ts_col]
        if raw_ts.isna().any():
            bad_row = int(raw_ts.isna().idxmax()) + 1
            raise InputValidationError(f"Missing timestamp value at CSV row #{bad_row}.")

        parsed_ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)
        if parsed_ts.isna().any():
            bad_idx = int(parsed_ts.isna().idxmax())
            bad_val = df[ts_col].iloc[bad_idx]
            raise InputValidationError(
                f"Invalid/unparseable timestamp '{bad_val}' at CSV row #{bad_idx + 1}."
            )

        if parsed_ts.duplicated().any():
            dup_idx = int(parsed_ts[parsed_ts.duplicated()].index[0])
            dup_val = df[ts_col].iloc[dup_idx]
            raise InputValidationError(
                f"Duplicate timestamp '{dup_val}' detected at CSV row #{dup_idx + 1}."
            )

        diffs = parsed_ts.diff().iloc[1:]
        if (diffs <= pd.Timedelta(0)).any():
            bad_order_idx = int(diffs[diffs <= pd.Timedelta(0)].index[0])
            raise InputValidationError(
                f"Out-of-order timestamps detected at CSV row #{bad_order_idx + 1} "
                f"('{df[ts_col].iloc[bad_order_idx - 1]}' -> '{df[ts_col].iloc[bad_order_idx]}'). "
                "Timestamps must be in strictly ascending chronological order."
            )

        expected_step = pd.Timedelta(hours=1)
        non_hourly_mask = diffs != expected_step
        if non_hourly_mask.any():
            gap_idx = int(diffs[non_hourly_mask].index[0])
            actual_delta = diffs.loc[gap_idx]
            raise InputValidationError(
                f"Missing or non-hourly timestamp interval ({actual_delta}) detected between "
                f"row #{gap_idx} ('{df[ts_col].iloc[gap_idx - 1]}') and "
                f"row #{gap_idx + 1} ('{df[ts_col].iloc[gap_idx]}'). "
                "Input sequence must have contiguous 1-hour resolution."
            )

        timestamps_list = [str(ts) for ts in parsed_ts]

    # 3. Validate Load Column Values
    raw_load_series = df[load_col]
    parsed_loads: List[float] = []
    for idx, val in enumerate(raw_load_series.tolist()):
        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
            raise InputValidationError(
                f"Missing/NaN load value at CSV row #{idx + 1} in column '{load_col}'."
            )
        try:
            fval = float(val)
        except (ValueError, TypeError) as exc:
            raise InputValidationError(
                f"Non-numeric load value '{val}' at CSV row #{idx + 1} in column '{load_col}'."
            ) from exc
        parsed_loads.append(fval)

    values = np.asarray(parsed_loads, dtype=np.float64)
    return _validate_numeric_load_array(
        values=values,
        ref=ref,
        input_source="custom_csv",
        is_hypothetical_scenario=True,
        timestamps_all=timestamps_list,
        load_column_used=load_col,
        timestamp_column_used=ts_col,
    )


# =====================================================================
# 6. Causal Context Feature Extraction & Preprocessing
# =====================================================================

def prepare_standardized_input_and_causal_context(
    validated_input: ValidatedSimulatorInput,
    ref: PJMTrainingReference,
) -> Tuple[np.ndarray, np.ndarray, str, bool, str]:
    """
    Apply the exact training-only StandardScaler and construct the 4D causal context vector:
    C_t = [Trend, Volatility, Periodicity, Recent_Error].

    Recent Error Causal Policy:
    - If `validated_input.total_observations >= 192` (168h lookback + 24h preceding origin):
      Computes exact causal walk-forward error at origin s = t - 24 using the training-fitted
      `ChronologicalWalkForwardForecaster` (`Ridge(alpha=100.0)`), evaluating its 24-hour
      forecast from `z[-192:-24]` against the observed standardized actuals `z[-24:]`.
    - If `168 <= validated_input.total_observations < 192`:
      Uses the verified training-time neutral prior `ref.warmup_prior_mae` (0.387859 in
      standardized units, computed strictly on training walk-forward cohort [500:1000] in
      `data_utils.ChronologicalWalkForwardForecaster`) and sets `fallback_used = True`.
    """
    full_std = (validated_input.full_series_mw - ref.scaler_mean) / ref.scaler_scale
    lookback_std = full_std[-LOOKBACK_HOURS:].astype(np.float32)
    X_window = lookback_std.reshape(1, LOOKBACK_HOURS, 1)

    if validated_input.total_observations >= EXACT_RECENT_ERROR_MIN_HOURS:
        prev_lookback_std = full_std[-EXACT_RECENT_ERROR_MIN_HOURS:-HORIZON_HOURS].astype(np.float32).reshape(1, LOOKBACK_HOURS, 1)
        prev_target_std = full_std[-HORIZON_HOURS:].astype(np.float32)
        prev_pred_std = ref.forecaster.predict_deployment(prev_lookback_std)[0]
        rec_err_val = float(np.mean(np.abs(prev_pred_std - prev_target_std)))
        rec_err_mode = "exact_causal_walk_forward"
        fallback_used = False
        explanation = (
            f"Exact causal calculation: {validated_input.total_observations} contiguous hours supplied "
            f"(>= {EXACT_RECENT_ERROR_MIN_HOURS}h). Computed as the 24-hour MAE of the training-fitted "
            "ChronologicalWalkForwardForecaster (Ridge α=100.0) launched at origin t-24h and evaluated "
            "strictly on observed hours [t-23h .. t]."
        )
    else:
        rec_err_val = float(ref.warmup_prior_mae)
        rec_err_mode = "training_warmup_prior_fallback"
        fallback_used = True
        explanation = (
            f"Disclosed fallback prior used: input contains {validated_input.total_observations} hours "
            f"(< {EXACT_RECENT_ERROR_MIN_HOURS}h required to reconstruct the 168h lookback at origin t-24h). "
            f"Recent_Error is set to the verified PJM training walk-forward neutral prior "
            f"(warmup_prior_mae = {rec_err_val:.6f} standardized units, from data_utils.ChronologicalWalkForwardForecaster)."
        )

    C_window = data_utils.extract_context_features(
        X_window,
        recent_errors=np.array([rec_err_val], dtype=np.float32),
    )

    if C_window.shape != (1, 4) or not np.all(np.isfinite(C_window)):
        raise RuntimeError(f"Context feature extraction produced invalid array of shape {C_window.shape}.")

    return X_window, C_window, rec_err_mode, fallback_used, explanation


# =====================================================================
# 7. Genuine Live Inference Execution (Single Seed & 5-Seed Ensemble)
# =====================================================================

def _run_single_seed_forward(
    seed: int,
    X_window: np.ndarray,
    C_window: np.ndarray,
    scaler_mean: float,
    scaler_scale: float,
) -> Dict[str, Any]:
    """Run deterministic evaluation-mode forward pass for a single Canonical V1 seed."""
    model, ckpt_path, counts = verify_and_load_canonical_v1_model(seed)
    model.eval()

    x_tensor = torch.from_numpy(X_window).float()
    c_tensor = torch.from_numpy(C_window).float()

    if x_tensor.shape != (1, LOOKBACK_HOURS, 1):
        raise ValueError(f"Expected input tensor shape (1, 168, 1), got {tuple(x_tensor.shape)}.")
    if c_tensor.shape != (1, 4):
        raise ValueError(f"Expected context tensor shape (1, 4), got {tuple(c_tensor.shape)}.")

    with torch.no_grad():
        y_pred_std, weights_t, expert_dict = model(x_tensor, c_tensor, return_diagnostics=True)

    y_caeg_std = y_pred_std[0].detach().cpu().numpy().astype(np.float64)
    y_lstm_std = expert_dict["lstm"][0].detach().cpu().numpy().astype(np.float64)
    y_tcn_std = expert_dict["tcn"][0].detach().cpu().numpy().astype(np.float64)
    y_cnn_std = expert_dict["cnn"][0].detach().cpu().numpy().astype(np.float64)
    w_arr = weights_t[0].detach().cpu().numpy().astype(np.float64)

    # Inverse-transform to physical MW scale using training StandardScaler
    y_caeg_mw = y_caeg_std * scaler_scale + scaler_mean
    y_lstm_mw = y_lstm_std * scaler_scale + scaler_mean
    y_tcn_mw = y_tcn_std * scaler_scale + scaler_mean
    y_cnn_mw = y_cnn_std * scaler_scale + scaler_mean
    y_equal_mw = (y_lstm_mw + y_tcn_mw + y_cnn_mw) / 3.0

    # Validate output shapes, finiteness, and convex simplex routing normalization
    for name, arr in [
        ("y_caeg_mw", y_caeg_mw),
        ("y_lstm_mw", y_lstm_mw),
        ("y_tcn_mw", y_tcn_mw),
        ("y_cnn_mw", y_cnn_mw),
        ("y_equal_mw", y_equal_mw),
    ]:
        if arr.shape != (HORIZON_HOURS,):
            raise RuntimeError(f"Output '{name}' has invalid shape {arr.shape}, expected ({HORIZON_HOURS},).")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Output '{name}' contains non-finite values.")

    if w_arr.shape != (3,) or not np.all(np.isfinite(w_arr)):
        raise RuntimeError(f"Routing weights invalid shape or non-finite: {w_arr}")

    w_sum = float(np.sum(w_arr))
    w_valid = bool(np.all(w_arr >= 0.0) and abs(w_sum - 1.0) <= 1e-5)
    if not w_valid:
        raise RuntimeError(f"Routing weights failed simplex verification: weights={w_arr}, sum={w_sum}")

    # Verify fused forecast equals weighted sum of internal expert forecasts
    y_reconstructed_mw = w_arr[0] * y_lstm_mw + w_arr[1] * y_tcn_mw + w_arr[2] * y_cnn_mw
    if float(np.max(np.abs(y_caeg_mw - y_reconstructed_mw))) > 1e-2:
        raise RuntimeError("Fused CAEG-Net output does not match convex combination of expert outputs.")

    return {
        "seed": seed,
        "ckpt_path": ckpt_path,
        "param_count": counts["total_params"],
        "y_caeg_mw": y_caeg_mw,
        "y_lstm_mw": y_lstm_mw,
        "y_tcn_mw": y_tcn_mw,
        "y_cnn_mw": y_cnn_mw,
        "y_equal_mw": y_equal_mw,
        "routing_weights": w_arr,
        "routing_weights_sum": w_sum,
        "routing_weights_valid": w_valid,
    }


def run_live_custom_forecast(
    validated_input: ValidatedSimulatorInput,
    seed: Union[int, str] = 42,
    ref: Optional[PJMTrainingReference] = None,
) -> LiveForecastResult:
    """
    Execute genuine live inference using Canonical CAEG-Net V1 on a validated user input.
    `seed` can be an integer in `(42, 123, 999, 2024, 3407)` or `"ensemble_5seed"` for
    an independently executed 5-seed ensemble aggregation.
    """
    if ref is None:
        ref = load_pjm_training_reference()

    X_window, C_window, rec_mode, fallback_used, rec_expl = prepare_standardized_input_and_causal_context(
        validated_input, ref
    )

    c_vec = C_window[0].astype(np.float64)
    c_dict = {name: float(c_vec[i]) for i, name in enumerate(CONTEXT_FEATURE_NAMES)}

    if isinstance(seed, str) and seed == "ensemble_5seed":
        per_seed_runs = [
            _run_single_seed_forward(s, X_window, C_window, ref.scaler_mean, ref.scaler_scale)
            for s in SUPPORTED_SEEDS
        ]
        y_caeg_stack = np.stack([r["y_caeg_mw"] for r in per_seed_runs], axis=0)
        y_lstm_stack = np.stack([r["y_lstm_mw"] for r in per_seed_runs], axis=0)
        y_tcn_stack = np.stack([r["tcn_mw"] if "tcn_mw" in r else r["y_tcn_mw"] for r in per_seed_runs], axis=0)
        y_cnn_stack = np.stack([r["y_cnn_mw"] for r in per_seed_runs], axis=0)
        y_equal_stack = np.stack([r["y_equal_mw"] for r in per_seed_runs], axis=0)
        w_stack = np.stack([r["routing_weights"] for r in per_seed_runs], axis=0)

        y_caeg_mean = np.mean(y_caeg_stack, axis=0)
        y_caeg_std = np.std(y_caeg_stack, axis=0, ddof=0)
        w_mean = np.mean(w_stack, axis=0)
        w_std = np.std(w_stack, axis=0, ddof=0)
        w_sum = float(np.sum(w_mean))
        w_valid = bool(np.all(w_mean >= 0.0) and abs(w_sum - 1.0) <= 1e-5)

        return LiveForecastResult(
            seed_label="5-Seed Ensemble (42, 123, 999, 2024, 3407)",
            checkpoint_paths=[r["ckpt_path"] for r in per_seed_runs],
            model_parameter_count=per_seed_runs[0]["param_count"],
            input_source=validated_input.input_source,
            is_hypothetical_scenario=validated_input.is_hypothetical_scenario,
            total_input_observations=validated_input.total_observations,
            units="MW",
            lookback_168_mw=validated_input.lookback_168_mw.copy(),
            y_caeg_mw=y_caeg_mean,
            y_lstm_mw=np.mean(y_lstm_stack, axis=0),
            y_tcn_mw=np.mean(y_tcn_stack, axis=0),
            y_cnn_mw=np.mean(y_cnn_stack, axis=0),
            y_equal_mw=np.mean(y_equal_stack, axis=0),
            routing_weights=w_mean,
            routing_weights_sum=w_sum,
            routing_weights_valid=w_valid,
            context_vector=c_vec,
            context_dict=c_dict,
            recent_error_mode=rec_mode,
            recent_error_fallback_used=fallback_used,
            recent_error_explanation=rec_expl,
            scaler_mean_mw=ref.scaler_mean,
            scaler_scale_mw=ref.scaler_scale,
            distribution_warnings=list(validated_input.distribution_warnings),
            timestamps_lookback_168=validated_input.timestamps_lookback_168,
            timestamps_horizon_24=validated_input.timestamps_horizon_24,
            per_seed_caeg_mw={r["seed"]: r["y_caeg_mw"] for r in per_seed_runs},
            y_caeg_std_mw=y_caeg_std,
            routing_weights_std=w_std,
        )

    seed_int = int(seed)
    single_res = _run_single_seed_forward(
        seed_int, X_window, C_window, ref.scaler_mean, ref.scaler_scale
    )

    return LiveForecastResult(
        seed_label=str(seed_int),
        checkpoint_paths=[single_res["ckpt_path"]],
        model_parameter_count=single_res["param_count"],
        input_source=validated_input.input_source,
        is_hypothetical_scenario=validated_input.is_hypothetical_scenario,
        total_input_observations=validated_input.total_observations,
        units="MW",
        lookback_168_mw=validated_input.lookback_168_mw.copy(),
        y_caeg_mw=single_res["y_caeg_mw"],
        y_lstm_mw=single_res["y_lstm_mw"],
        y_tcn_mw=single_res["y_tcn_mw"],
        y_cnn_mw=single_res["y_cnn_mw"],
        y_equal_mw=single_res["y_equal_mw"],
        routing_weights=single_res["routing_weights"],
        routing_weights_sum=single_res["routing_weights_sum"],
        routing_weights_valid=single_res["routing_weights_valid"],
        context_vector=c_vec,
        context_dict=c_dict,
        recent_error_mode=rec_mode,
        recent_error_fallback_used=fallback_used,
        recent_error_explanation=rec_expl,
        scaler_mean_mw=ref.scaler_mean,
        scaler_scale_mw=ref.scaler_scale,
        distribution_warnings=list(validated_input.distribution_warnings),
        timestamps_lookback_168=validated_input.timestamps_lookback_168,
        timestamps_horizon_24=validated_input.timestamps_horizon_24,
    )


# =====================================================================
# 8. Optional Historical-Window Live Validation Mode
# =====================================================================

def run_historical_window_validation(
    window_idx: int,
    seed: int = 42,
    ref: Optional[PJMTrainingReference] = None,
    pred_atol_mw: float = 1e-3,
    weight_atol: float = 1e-6,
) -> LiveForecastResult:
    """
    Run live Canonical CAEG-Net V1 inference on an existing PJM test window `window_idx`
    (0..1293) using the exact test partition lookback and causal context vector, and
    verify numerical parity against `results/phase5_multiseed_cache.npz`.
    """
    if ref is None:
        ref = load_pjm_training_reference()

    num_windows = ref.X_test_std.shape[0]
    if not (0 <= window_idx < num_windows):
        raise ValueError(f"window_idx must be in [0, {num_windows - 1}], got {window_idx}.")
    if seed not in SUPPORTED_SEEDS:
        raise ValueError(f"seed must be in {SUPPORTED_SEEDS}, got {seed}.")

    X_window = ref.X_test_std[window_idx : window_idx + 1]
    C_window = ref.C_test[window_idx : window_idx + 1]
    lookback_168_mw = (X_window[0, :, 0].astype(np.float64) * ref.scaler_scale + ref.scaler_mean)
    y_true_mw = ref.Y_test_raw[window_idx].astype(np.float64)

    ts_lookback = ref.test_timestamps_extended[window_idx : window_idx + LOOKBACK_HOURS]
    ts_horizon = ref.test_timestamps_extended[
        window_idx + LOOKBACK_HOURS : window_idx + LOOKBACK_HOURS + HORIZON_HOURS
    ]

    single_res = _run_single_seed_forward(
        seed, X_window, C_window, ref.scaler_mean, ref.scaler_scale
    )

    # Compare against saved Phase 5 cache (`results/phase5_multiseed_cache.npz`)
    if not os.path.isfile(PHASE5_CACHE_NPZ_PATH):
        raise FileNotFoundError(f"Phase 5 cache not found at {PHASE5_CACHE_NPZ_PATH}")

    p5_cache = np.load(PHASE5_CACHE_NPZ_PATH)
    cached_pred_mw = p5_cache[f"caeg_seed_{seed}"][window_idx].astype(np.float64)
    cached_weights = p5_cache[f"weights_seed_{seed}"][window_idx].astype(np.float64)

    max_pred_diff = float(np.max(np.abs(single_res["y_caeg_mw"] - cached_pred_mw)))
    max_weight_diff = float(np.max(np.abs(single_res["routing_weights"] - cached_weights)))
    parity_passed = bool((max_pred_diff <= pred_atol_mw) and (max_weight_diff <= weight_atol))

    if not parity_passed:
        raise RuntimeError(
            f"Historical live-vs-cache parity failed for window #{window_idx}, seed {seed}: "
            f"max_pred_diff={max_pred_diff:.6f} MW (tol={pred_atol_mw}), "
            f"max_weight_diff={max_weight_diff:.8f} (tol={weight_atol})."
        )

    retro_mae = float(np.mean(np.abs(y_true_mw - single_res["y_caeg_mw"])))
    retro_rmse = float(np.sqrt(np.mean((y_true_mw - single_res["y_caeg_mw"]) ** 2)))
    retro_max_err = float(np.max(np.abs(y_true_mw - single_res["y_caeg_mw"])))
    retro_mape = float(np.mean(np.abs((y_true_mw - single_res["y_caeg_mw"]) / y_true_mw)) * 100.0)

    c_vec = C_window[0].astype(np.float64)
    c_dict = {name: float(c_vec[i]) for i, name in enumerate(CONTEXT_FEATURE_NAMES)}

    return LiveForecastResult(
        seed_label=str(seed),
        checkpoint_paths=[single_res["ckpt_path"]],
        model_parameter_count=single_res["param_count"],
        input_source=f"historical_pjm_test_window_{window_idx}",
        is_hypothetical_scenario=False,
        total_input_observations=LOOKBACK_HOURS,
        units="MW",
        lookback_168_mw=lookback_168_mw,
        y_caeg_mw=single_res["y_caeg_mw"],
        y_lstm_mw=single_res["y_lstm_mw"],
        y_tcn_mw=single_res["y_tcn_mw"],
        y_cnn_mw=single_res["y_cnn_mw"],
        y_equal_mw=single_res["y_equal_mw"],
        routing_weights=single_res["routing_weights"],
        routing_weights_sum=single_res["routing_weights_sum"],
        routing_weights_valid=single_res["routing_weights_valid"],
        context_vector=c_vec,
        context_dict=c_dict,
        recent_error_mode="historical_test_partition_exact",
        recent_error_fallback_used=False,
        recent_error_explanation=(
            f"Exact causal test-partition Recent_Error at PJM test window #{window_idx} "
            f"(origin {ts_lookback[-1]}) computed via data_utils.compute_causal_recent_forecast_errors."
        ),
        scaler_mean_mw=ref.scaler_mean,
        scaler_scale_mw=ref.scaler_scale,
        distribution_warnings=[],
        timestamps_lookback_168=ts_lookback,
        timestamps_horizon_24=ts_horizon,
        historical_window_idx=window_idx,
        historical_y_true_mw=y_true_mw,
        historical_cached_pred_mw=cached_pred_mw,
        historical_cached_weights=cached_weights,
        historical_max_abs_pred_diff_mw=max_pred_diff,
        historical_max_abs_weight_diff=max_weight_diff,
        historical_parity_passed=parity_passed,
        historical_retrospective_metrics={
            "MAE_MW": retro_mae,
            "RMSE_MW": retro_rmse,
            "MAX_ERR_MW": retro_max_err,
            "MAPE_PCT": retro_mape,
        },
    )


# =====================================================================
# 9. Forecast Table & CSV Export Utilities
# =====================================================================

def build_forecast_dataframe(result: LiveForecastResult) -> pd.DataFrame:
    """
    Build a 24-row DataFrame containing the newly generated 24-hour forecast,
    individual expert predictions, equal-weight reference, routing weights,
    and provenance metadata. For hypothetical scenarios, ground-truth columns
    are strictly omitted.
    """
    hours = np.arange(1, HORIZON_HOURS + 1, dtype=int)
    data: Dict[str, Any] = {
        "Horizon_Step_h": hours,
        "Lead_Label": [f"t+{h}h" for h in hours],
    }

    if result.timestamps_horizon_24 is not None and len(result.timestamps_horizon_24) == HORIZON_HOURS:
        data["Forecast_Timestamp"] = result.timestamps_horizon_24

    data["CAEG_Net_V1_Forecast_MW"] = np.round(result.y_caeg_mw, 4)
    if result.y_caeg_std_mw is not None:
        data["CAEG_Net_V1_Seed_Std_MW"] = np.round(result.y_caeg_std_mw, 4)

    data["LSTM_Expert_MW"] = np.round(result.y_lstm_mw, 4)
    data["TCN_Expert_MW"] = np.round(result.y_tcn_mw, 4)
    data["CNN_Expert_MW"] = np.round(result.y_cnn_mw, 4)
    data["Equal_Weight_Avg_MW"] = np.round(result.y_equal_mw, 4)

    data["Weight_LSTM"] = np.round(np.full(HORIZON_HOURS, result.routing_weights[0]), 6)
    data["Weight_TCN"] = np.round(np.full(HORIZON_HOURS, result.routing_weights[1]), 6)
    data["Weight_CNN"] = np.round(np.full(HORIZON_HOURS, result.routing_weights[2]), 6)
    data["Forecast_Mode"] = (
        "Hypothetical_Scenario_Forecast"
        if result.is_hypothetical_scenario
        else f"Historical_Test_Window_{result.historical_window_idx}_Retrospective"
    )
    data["Seed"] = result.seed_label
    data["Recent_Error_Mode"] = result.recent_error_mode

    if not result.is_hypothetical_scenario and result.historical_y_true_mw is not None:
        data["Observed_Actual_MW"] = np.round(result.historical_y_true_mw, 4)
        data["Signed_Residual_MW"] = np.round(result.y_caeg_mw - result.historical_y_true_mw, 4)
        data["Absolute_Error_MW"] = np.round(np.abs(result.y_caeg_mw - result.historical_y_true_mw), 4)

    return pd.DataFrame(data)


def export_forecast_to_csv_bytes(result: LiveForecastResult) -> bytes:
    """Serialize the 24-hour forecast DataFrame to UTF-8 CSV bytes for download."""
    df = build_forecast_dataframe(result)
    return df.to_csv(index=False).encode("utf-8")


def generate_hypothetical_scenario_dataframe(
    num_hours: int = 192,
    scenario_type: str = "diurnal_baseline",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a clearly labelled synthetic hypothetical scenario DataFrame
    (with columns `timestamp` and `load_mw`) for UI initialization or CSV template download.
    """
    base_df = data_utils.generate_synthetic_load_data(
        num_hours=max(num_hours, LOOKBACK_HOURS),
        start_date="2025-01-01 00:00:00",
        seed=seed,
    ).iloc[:num_hours].copy()

    t = np.arange(len(base_df), dtype=np.float64)
    load = base_df["load"].values.astype(np.float64)

    if scenario_type == "high_peak_ramp":
        load = load * 1.12 + 450.0 * np.sin(2.0 * np.pi * t / 24.0)
    elif scenario_type == "low_weekend_demand":
        load = load * 0.86 - 200.0

    return pd.DataFrame({
        "timestamp": [str(ts) for ts in base_df["timestamp"]],
        "load_mw": np.round(load, 2),
    })

