"""Load and preprocess FOCUS sample data from the FinOps Open Cost and Usage Specification."""

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from .config import FOCUS_DATA_URL, FOCUS_CACHE_DIR, FOCUS_COST_COL, FOCUS_GROUP_BY
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import FOCUS_DATA_URL, FOCUS_CACHE_DIR, FOCUS_COST_COL, FOCUS_GROUP_BY

_DATE_COL = "ChargePeriodStart"


def download_focus_data(
    url: str = FOCUS_DATA_URL,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download a FOCUS CSV to a local cache file and return the path.

    Skips the download if the file is already cached.
    """
    cache_path = Path(cache_dir or FOCUS_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1]
    dest = cache_path / filename
    if not dest.exists():
        print(f"  Downloading {filename} from GitHub...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_focus_data(path: str) -> pd.DataFrame:
    """Parse a FOCUS CSV file into a typed DataFrame.

    Adds a ``_date`` column (tz-naive date, UTC-normalized) derived from
    ``ChargePeriodStart``.  The cost column is coerced to float.
    """
    df = pd.read_csv(path, low_memory=False)
    df[_DATE_COL] = pd.to_datetime(df[_DATE_COL], utc=True, errors="coerce")
    df["_date"] = df[_DATE_COL].dt.normalize().dt.tz_localize(None)
    df[FOCUS_COST_COL] = pd.to_numeric(df[FOCUS_COST_COL], errors="coerce").fillna(0.0)
    return df


def aggregate_daily(
    df: pd.DataFrame,
    cost_col: str = FOCUS_COST_COL,
    group_by: List[str] = FOCUS_GROUP_BY,
    min_days: int = 7,
    min_mean_cost: float = 0.0,
) -> Dict[str, pd.Series]:
    """Aggregate hourly FOCUS rows to a daily cost series per group.

    Returns
    -------
    dict mapping ``"Provider / Category"`` label → daily pd.Series indexed by
    date (chronological).  Groups with fewer than ``min_days`` distinct daily
    observations or a mean cost of exactly zero are excluded.
    """
    agg = (
        df.groupby(["_date"] + group_by)[cost_col]
        .sum()
        .reset_index()
    )
    result: Dict[str, pd.Series] = {}
    for keys, sub in agg.groupby(group_by):
        label = (
            " / ".join(str(k) for k in keys)
            if isinstance(keys, tuple)
            else str(keys)
        )
        series = sub.set_index("_date")[cost_col].sort_index()
        if len(series) >= min_days and series.mean() > min_mean_cost:
            result[label] = series
    return result
