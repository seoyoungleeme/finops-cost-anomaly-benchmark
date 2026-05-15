# FinOps Cost Anomaly Benchmark

This repository contains a refactored version of the original Colab notebook. The experiment logic is split into importable Python modules while preserving the notebook's behavior.

## Structure

- `finops_benchmark/config.py`: shared constants and output directory helpers
- `finops_benchmark/data.py`: synthetic baseline generation and anomaly injection
- `finops_benchmark/models.py`: EWMA, Isolation Forest, Prophet, and LSTM autoencoder scoring
- `finops_benchmark/evaluation.py`: thresholds, predictions, point/event/cost-weighted metrics
- `finops_benchmark/experiment.py`: seed loops, summaries, rank and paper-table builders
- `finops_benchmark/visualization.py`: exploratory and paper-ready plots
- `finops_benchmark/sanity.py`: validation checks for generated outputs
- `scripts/run_benchmark.py`: command-line full benchmark runner
- `FinOps.ipynb`: thin Colab-friendly driver notebook

## Run

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py
```
