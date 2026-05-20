# FinOps Cost Anomaly Benchmark

Cost anomalies in cloud systems are quiet but expensive: a forgotten GPU
instance, a runaway data-transfer job, an autoscaling misconfiguration, or a
billing correction can change spend before anyone notices. FinOps teams need
detectors that catch these events early, but public cloud-cost anomaly research
has a structural problem: real billing data is sensitive and public FOCUS
billing samples do not include ground-truth anomaly labels.

This repository builds a reproducible benchmark for that gap. It uses real
FOCUS billing data to calibrate cloud-cost baselines, injects controlled labeled
anomalies, and evaluates anomaly detectors with FinOps-aware metrics such as
dollar recall, mean cost-to-detect, and alert cost efficiency.

For the full research narrative, see [report.md](report.md).

## Core Claim

This is not a raw real-billing labeled benchmark. FOCUS data is used as the
real-world source for baseline statistics, while anomaly labels are created by
controlled injection.

```text
Primary quantitative result:
  FOCUS-calibrated synthetic benchmark

External realism source:
  Official FOCUS public billing sample data

Raw real-data use:
  Unsupervised sanity check only

Not claimed:
  Raw FOCUS labeled anomaly benchmark
```

That distinction matters because raw FOCUS data has no anomaly ground truth.
Without labels, precision, recall, F1, detection delay, and MCTD cannot be
computed honestly. The calibrated benchmark keeps labels available while making
the normal spend pattern more realistic than a purely hand-designed synthetic
series.

## Current Research Artifacts

The current report is based mainly on the full FOCUS strict benchmark:

- FOCUS full sample: `5,488,359` rows, `44` source columns
- Full strict service groups: `4`
- Seeds: `0, 1, 2`
- Main output directory: `outputs/results_full_strict/`
- Main figure directory: `outputs/figures_full_strict/`
- Inventory artifact: `outputs/focus_data_inventory.json`

The full strict model ranking at the primary Year-1 false-alarm target of 1% is:

| Model | F1 | Cost-Weighted Recall | Alert Cost Efficiency | Mean MCTD |
|---|---:|---:|---:|---:|
| Prophet | 0.5132 | 0.9553 | 101.33 | 15.08 |
| IsolationForest | 0.4461 | 0.8822 | 157.63 | 26.31 |
| LSTM_AE | 0.3172 | 0.9191 | 37.72 | 12.14 |
| EWMA | 0.2226 | 0.5543 | 484.66 | 174.12 |

Interpretation:

- Prophet is strongest on F1 and dollar recall.
- LSTM autoencoder has the lowest mean MCTD in the full strict run.
- EWMA looks efficient per alert but misses too much dollar impact and detects
  too late for FinOps loss reduction.
- Model choice changes depending on whether the operational objective is
  classification quality, dollars caught, cost-to-detect, or alert efficiency.

## Data Strategy

### FOCUS Data

This project uses the official FOCUS Sample Data repository:

- FOCUS site: <https://focus.finops.org/>
- FOCUS Sample Data: <https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data>
- FOCUS 1.0 sample folder: <https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/tree/main/FOCUS-1.0>

Cached files used locally:

| File | Role |
|---|---|
| `.focus_cache/focus_sample_10000.csv` | Initial small sample |
| `.focus_cache/focus_sample_100000.csv.gz` | 100k relaxed robustness run |
| `.focus_cache/focus_data_table.csv.gz` | Full strict benchmark and raw sanity check |

The raw data cache is ignored by Git because the full file is large and can be
downloaded again.

### Inventory

`scripts/build_focus_inventory.py` creates a reproducible inventory of cached
FOCUS files:

```bash
python scripts/build_focus_inventory.py
```

Output:

```text
outputs/focus_data_inventory.json
```

The inventory records byte size, rows, source column count, date range, unique
billing days, and provider row counts. This backs the report's data claims with
a regenerable artifact.

## What It Runs

### 1. FOCUS-Calibrated Synthetic Benchmark

`scripts/run_focus_benchmark.py` downloads or reuses a FOCUS CSV/CSV.GZ,
aggregates it into daily service-level cost series, extracts calibration
statistics, generates 730-day labeled benchmark series, and runs four detectors.

Calibration parameters extracted from real FOCUS daily series:

- `base_level`: mean daily cost level
- `monthly_growth`: clipped monthly trend estimate
- `noise_pct`: clipped residual variation
- `weekly_factor`: day-of-week multiplier

Anomalies are injected only in Year 2:

- `spike`
- `contextual`
- `gradual`

Each anomaly has `low`, `mid`, or `high` intensity and event-level cost impact.

### 2. Raw FOCUS Sanity Check

`scripts/run_focus_unsupervised.py` applies a look-ahead-free rolling z-score
detector directly to raw aggregated FOCUS daily cost series. This produces
flagged dates but no precision/recall/F1, because raw FOCUS has no anomaly
ground truth.

### 3. Pure Synthetic Benchmark

`scripts/run_benchmark.py` keeps the original controlled synthetic benchmark.
It is useful as a baseline/ablation, but the current report emphasizes the
FOCUS-calibrated path.

## Compared Models

The benchmark compares four representative detection paradigms:

| Model | Paradigm | Score |
|---|---|---|
| EWMA | Statistical residual baseline | EWMA residual z-score |
| IsolationForest | Feature-based unsupervised learning | Calendar/lag/rolling-feature anomaly score |
| LSTM_AE | Sequence reconstruction | Reconstruction error |
| Prophet | Forecast residual | Forecast residual z-score |

All models emit a long-format score table:

```text
model_name, date, day, score
```

## Evaluation Protocol

The benchmark is leakage-free:

- Year 1 contains no injected anomalies.
- Models fit only on Year 1.
- Thresholds are calibrated only on Year-1 scores.
- Anomalies occur only in Year 2.
- All metrics are computed on Year 2.

The primary threshold is a Year-1 false-alarm target of 1%, implemented as the
99th percentile of each model's Year-1 scores. Additional calibration points use
0.5% and 2.0%.

Primary metrics:

- precision, recall, F1
- AUPRC from raw anomaly scores
- false alarm rate
- event recall
- mean detection delay
- cost-weighted recall
- mean cost-to-detect (`mean_mctd`)
- alert cost efficiency

## Repository Structure

```text
finops_benchmark/
  config.py              shared constants, paths, benchmark settings
  data.py                synthetic generation and anomaly injection
  models.py              EWMA, IsolationForest, Prophet, LSTM-AE scoring
  evaluation.py          thresholding and metrics
  experiment.py          seed loops, summaries, rank tables
  focus_loader.py        FOCUS download, parsing, daily aggregation
  focus_calibration.py   calibration stats from FOCUS daily series
  visualization.py       benchmark figures
  sanity.py              validation helpers

scripts/
  run_benchmark.py           pure synthetic benchmark
  run_focus_benchmark.py     FOCUS-calibrated benchmark
  run_focus_unsupervised.py  raw FOCUS rolling z-score sanity check
  build_focus_inventory.py   cached FOCUS file inventory

outputs/
  focus_data_inventory.json
  results_full_strict/
  figures_full_strict/
  results/

report.md                research report and presentation notes
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

On Windows, Prophet/CmdStan temporary files are routed through `.tmp/` to avoid
failures caused by non-ASCII user-profile paths.

## Reproduce The Current Main Results

### 1. Download/verify FOCUS inventory

If the cache already exists:

```bash
python scripts/build_focus_inventory.py
```

The current full sample was downloaded from GitHub's media endpoint because the
normal raw URL returns a Git LFS pointer for the full gzip file:

```text
https://media.githubusercontent.com/media/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_data_table.csv.gz
```

### 2. Full strict FOCUS-calibrated benchmark

```bash
python scripts/run_focus_benchmark.py \
  --url https://media.githubusercontent.com/media/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_data_table.csv.gz \
  --group-by ProviderName,ServiceCategory \
  --min-days 21 \
  --min-nonzero-days 14 \
  --min-mean-cost 1.0 \
  --n-seeds 3 \
  --output-dir outputs/results_full_strict \
  --figure-dir outputs/figures_full_strict
```

Main outputs:

```text
outputs/results_full_strict/focus_run_metadata.json
outputs/results_full_strict/focus_calibration_stats.csv
outputs/results_full_strict/focus_service_summary.csv
outputs/results_full_strict/focus_core_metrics_by_service.csv
outputs/results_full_strict/focus_overall_model_ranking.csv
outputs/results_full_strict/focus_anomaly_type_results.csv
outputs/results_full_strict/focus_anomaly_intensity_results.csv
outputs/results_full_strict/focus_rank_reversal_by_service.csv
outputs/results_full_strict/focus_rank_reversal_summary.csv
outputs/results_full_strict/focus_top_model_disagreement_summary.csv
outputs/figures_full_strict/*.png
```

### 3. 100k relaxed robustness run

```bash
python scripts/run_focus_benchmark.py \
  --url https://raw.githubusercontent.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_sample_100000.csv.gz \
  --group-by ProviderName,ServiceCategory \
  --min-days 14 \
  --min-nonzero-days 7 \
  --min-mean-cost 0.1 \
  --n-seeds 5
```

This writes to `outputs/results/` and `outputs/figures/` by default. The report
uses the summary CSV/metadata from this run, not every per-service raw event
file.

### 4. Raw full FOCUS sanity check

```bash
python scripts/run_focus_unsupervised.py \
  --url https://media.githubusercontent.com/media/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_data_table.csv.gz \
  --group-by ProviderName,ServiceCategory \
  --min-days 14 \
  --min-nonzero-days 7 \
  --min-mean-cost 0.1 \
  --window 7 \
  --sigma 2.5 \
  --output-prefix focus_unsupervised_full_relaxed
```

Outputs:

```text
outputs/results/focus_unsupervised_full_relaxed_alerts.csv
outputs/results/focus_unsupervised_full_relaxed_summary.csv
```

## Git And Outputs

Large raw FOCUS data and most generated intermediate files stay ignored:

```text
.focus_cache/
.tmp/
data/external/
data/processed/
```

Only the research-facing output artifacts are unignored:

- `outputs/focus_data_inventory.json`
- selected CSV/JSON files under `outputs/results_full_strict/`
- full strict figures under `outputs/figures_full_strict/`
- selected 100k relaxed and raw sanity-check CSV/JSON files under `outputs/results/`

The detailed per-service `focus_metrics_*` and `focus_events_*` files remain
ignored to keep the repository focused and lightweight.

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

## Notes

- FOCUS histories in the public samples are short, so direct raw-data anomaly
  detection is a sanity check, not a benchmark.
- FOCUS-calibrated synthetic evaluation is the recommended quantitative path.
- Always inspect `focus_run_metadata.json` before interpreting a result table;
  it records the data URL, service groups, fallback count, seeds, and inventory.
