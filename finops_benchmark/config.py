"""Shared configuration for the FinOps anomaly benchmark."""

import os

RANDOM_SEED = 42

N_DAYS = 730
YEAR2_START = 365
BASE_LEVEL = 1000.0
MONTHLY_GROWTH = 0.015
NOISE_PCT = 0.05
START_DATE = "2024-01-01"

SPIKE_LEVELS = {"low": 0.30, "mid": 1.00, "high": 3.00}
CONTEXTUAL_LEVELS = {"low": 0.50, "mid": 1.00, "high": 2.00}
GRADUAL_LEVELS = {"low": 0.03, "mid": 0.05, "high": 0.10}

N_EVENTS_LOW = 3
N_EVENTS_HIGH = 6

DF_COLUMNS = [
    "date", "day", "y", "y_expected_baseline",
    "is_anomaly", "anomaly_type", "intensity_level", "event_id",
    "excess_cost", "cost_impact", "score", "alert",
]

OUTPUT_DIR = "./outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGS_DIR_NEW = os.path.join(OUTPUT_DIR, "figures")

SEEDS = list(range(10))
BUDGETS = [
    (0.005, 99.5),
    (0.010, 99.0),
    (0.020, 98.0),
]

METRIC_COLS = [
    "threshold", "total_alerts", "false_alarm_rate",
    "precision", "recall", "f1", "auprc",
    "event_recall", "mean_detection_delay",
    "cost_weighted_recall", "mean_mctd", "alert_cost_efficiency",
]

EVENT_KEEP_COLS = [
    "seed", "year1_fpr_target", "model_name", "event_id",
    "anomaly_type", "intensity_level",
    "detected", "detection_delay", "total_excess_cost", "mctd",
]

# FOCUS real-data integration
FOCUS_DATA_URL = (
    "https://raw.githubusercontent.com/FinOps-Open-Cost-and-Usage-Spec/"
    "FOCUS-Sample-Data/main/FOCUS-1.0/focus_sample_10000.csv"
)
FOCUS_CACHE_DIR = ".focus_cache"
# Cost column is auto-selected in focus_loader (EffectiveCost preferred,
# BilledCost fallback); no single fixed column is configured here.
FOCUS_GROUP_BY = ["ProviderName", "ServiceCategory"]

# Real-FOCUS benchmark settings
FOCUS_REAL_SPLIT_RATIO = 0.5      # Year 1 (train) proportion of available days
FOCUS_REAL_N_EVENTS_LOW = 1       # min events per (type, intensity) in Year 2
FOCUS_REAL_N_EVENTS_HIGH = 3      # exclusive upper bound: rng.integers(low, high) -> [1, 2]
FOCUS_REAL_MIN_YEAR2_DAYS = 10    # minimum Year 2 evaluation days

PAPER_DPI = 200
PAPER_COLOR_MAP = {
    "EWMA": "#1f77b4",
    "IsolationForest": "#d62728",
    "LSTM_AE": "#2ca02c",
    "Prophet": "#9467bd",
}


def ensure_output_dirs(output_dir=OUTPUT_DIR, fig_dir=FIG_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return output_dir, fig_dir


def ensure_results_dirs(results_dir=RESULTS_DIR, figures_dir=FIGS_DIR_NEW):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return results_dir, figures_dir
