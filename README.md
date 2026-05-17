# FinOps Cost Anomaly Benchmark

Benchmark code for comparing cloud-cost anomaly detectors under both fully
synthetic data and FOCUS-calibrated data.

The project started as a notebook experiment and is now organized as importable
Python modules plus command-line runners. It generates daily cloud-cost series,
injects labeled anomalies, scores them with several models, and evaluates the
results with point-wise, event-wise, and cost-aware metrics.

## What It Runs

### 1. Synthetic Benchmark

`scripts/run_benchmark.py` runs the main benchmark on generated 730-day daily
cost series.

The synthetic generator includes:

- linear spend trend
- weekly seasonality
- month-end cost effect
- Gaussian noise
- injected anomaly events in year 2 only
- anomaly types: `spike`, `contextual`, `gradual`
- intensity levels: `low`, `mid`, `high`

### 2. FOCUS-Calibrated Benchmark

`scripts/run_focus_benchmark.py` downloads a FOCUS sample CSV, aggregates it to
daily service-level cost series, extracts real-data calibration statistics, then
generates labeled synthetic benchmark series using those statistics.

This keeps ground-truth labels available while making the baseline spend pattern
more realistic.

FOCUS handling currently:

- downloads to `.focus_cache/`
- prefers `EffectiveCost`, falling back to `BilledCost`
- normalizes cost into an internal `_eff_cost` column
- aggregates by `ProviderName,ServiceCategory` by default
- filters very sparse or tiny service groups
- clips calibration parameters to avoid impossible negative synthetic baselines

### 3. Real FOCUS Sanity Check

`scripts/run_focus_unsupervised.py` runs a simple look-ahead-free rolling
z-score detector directly on the real aggregated FOCUS daily series. This is not
a labeled benchmark; it is a sanity check for suspicious real cost spikes.

## Models

The benchmark compares four scoring methods:

- `EWMA`: exponentially weighted moving average residual z-score
- `IsolationForest`: calendar and lag-ratio feature based anomaly score
- `Prophet`: forecast residual z-score
- `LSTM_AE`: LSTM autoencoder reconstruction error

All models output a long-format score table:

```text
model_name, date, day, score
```

## Metrics

Thresholds are learned from year 1 scores only. Evaluation is performed on year
2, where anomalies are injected.

Primary metrics include:

- precision, recall, F1
- AUPRC from raw anomaly scores
- false alarm rate
- event recall
- mean detection delay
- cost-weighted recall
- mean cost-to-detect (`mean_mctd`)
- alert cost efficiency

Default alert budgets are:

- `0.5%` of days alerted
- `1.0%` of days alerted
- `2.0%` of days alerted

## Repository Structure

```text
finops_benchmark/
  config.py              shared constants, paths, benchmark settings
  data.py                synthetic data generation and anomaly injection
  models.py              EWMA, IsolationForest, Prophet, LSTM-AE scoring
  evaluation.py          thresholding and metric computation
  experiment.py          seed loops, summaries, rank tables
  focus_loader.py        FOCUS download, parsing, daily aggregation
  focus_calibration.py   calibration stats from FOCUS daily series
  visualization.py       exploratory and paper-ready figures
  sanity.py              validation helpers

scripts/
  run_benchmark.py           full synthetic benchmark
  run_focus_benchmark.py     FOCUS-calibrated synthetic benchmark
  run_focus_unsupervised.py  direct unsupervised FOCUS sanity check

FinOps.ipynb             notebook driver
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Dependencies:

- numpy
- pandas
- matplotlib
- scikit-learn
- prophet
- torch
- tqdm

On Windows, the code routes CmdStan/Prophet temporary files through `.tmp/` to
avoid failures caused by non-ASCII user-profile paths.

## Run Commands

### Synthetic Benchmark

```bash
python scripts/run_benchmark.py
```

Outputs:

```text
outputs/results/all_model_metrics.csv
outputs/results/summary_metrics.csv
outputs/results/all_event_results.csv
outputs/results/paper_core_results_budget1pct.csv
outputs/results/paper_rank_comparison_budget1pct.csv
outputs/figures/paper_f1_bar_budget1pct.png
outputs/figures/paper_dollar_recall_bar_budget1pct.png
outputs/figures/paper_ace_bar_budget1pct.png
outputs/figures/paper_f1_by_budget.png
outputs/figures/paper_mctd_by_budget.png
outputs/figures/paper_far_by_budget.png
```

### FOCUS-Calibrated Benchmark

```bash
python scripts/run_focus_benchmark.py
```

Useful options:

```bash
python scripts/run_focus_benchmark.py --n-seeds 3
python scripts/run_focus_benchmark.py --group-by ProviderName,ServiceName
python scripts/run_focus_benchmark.py --min-days 14 --min-nonzero-days 7 --min-mean-cost 0.1
```

Outputs:

```text
outputs/results/focus_calibration_stats.csv
outputs/results/focus_metrics_<service>.csv
outputs/results/focus_events_<service>.csv
outputs/results/focus_service_summary.csv
outputs/results/focus_core_metrics_by_service.csv
outputs/results/focus_rank_reversal_by_service.csv
outputs/results/focus_overall_model_ranking.csv
```

### Direct FOCUS Unsupervised Check

```bash
python scripts/run_focus_unsupervised.py
```

Useful options:

```bash
python scripts/run_focus_unsupervised.py --sigma 2.0 --window 5
python scripts/run_focus_unsupervised.py --group-by ProviderName,ServiceName
```

Outputs:

```text
outputs/results/focus_unsupervised_alerts.csv
outputs/results/focus_unsupervised_summary.csv
```

## Minimal Python Usage

```python
from finops_benchmark.data import build_dataset
from finops_benchmark.models import run_all_models
from finops_benchmark.evaluation import run_evaluation

df, events_df = build_dataset(seed=0)
scores_long, aux = run_all_models(df, events_df, seed=0)
predictions, model_metrics, event_results = run_evaluation(
    df,
    events_df,
    scores_long,
    percentile=99.0,
)
```

FOCUS-calibrated example:

```python
from finops_benchmark.focus_loader import download_focus_data, load_focus_data, aggregate_daily
from finops_benchmark.focus_calibration import compute_global_stats, calibrate_all_services
from finops_benchmark.data import build_focus_calibrated_dataset

path = download_focus_data()
raw = load_focus_data(str(path))
daily = aggregate_daily(raw)
stats_all = calibrate_all_services(daily, compute_global_stats(daily))

service, stats = next(iter(stats_all.items()))
df, events_df = build_focus_calibrated_dataset(stats, seed=0)
```

## Generated Files and Git

Downloaded FOCUS data and generated outputs are intentionally not committed.
The following paths are ignored:

```text
.focus_cache/
.tmp/
outputs/
data/external/
data/processed/
.claude/settings.local.json
```

Use `outputs/results/` and `outputs/figures/` for local experiment artifacts.

## Notes

- FOCUS sample data is short, so direct real-data anomaly detection is only a
  sanity check.
- The FOCUS-calibrated path is the recommended way to combine real spend
  patterns with labeled anomaly evaluation.
- Runtime depends mostly on Prophet and the LSTM autoencoder. The default full
  synthetic run uses 10 seeds; the default FOCUS-calibrated run uses 5 seeds.
