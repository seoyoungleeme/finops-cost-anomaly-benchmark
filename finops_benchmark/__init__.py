"""FinOps cost anomaly benchmark package."""

from .config import RANDOM_SEED, N_DAYS, YEAR2_START
from .data import build_dataset, generate_baseline_series, inject_anomalies
from .evaluation import run_evaluation
from .experiment import run_one_seed
from .models import run_all_models
