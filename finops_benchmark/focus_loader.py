"""Load and preprocess FOCUS sample data from the FinOps Open Cost and Usage Specification."""

from collections.abc import Sequence
from urllib.parse import urlparse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from .config import FOCUS_DATA_URL, FOCUS_CACHE_DIR, FOCUS_GROUP_BY
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import FOCUS_DATA_URL, FOCUS_CACHE_DIR, FOCUS_GROUP_BY

_DATE_COL = "ChargePeriodStart"
_PREFERRED_COST_COLS = ("EffectiveCost", "BilledCost")  # priority: EffectiveCost first
_EFF_COST_COL = "_eff_cost"   # internal normalized column added by load_focus_data


def _pick_cost_col(df: pd.DataFrame) -> str:
    """Return the first available FOCUS cost column, in priority order."""
    for col in _PREFERRED_COST_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        f"FOCUS CSV must contain at least one of {list(_PREFERRED_COST_COLS)}. "
        f"Found columns: {sorted(df.columns.tolist())}"
    )


def download_focus_data(
    url: str = FOCUS_DATA_URL,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download a FOCUS CSV or CSV.GZ to a local cache file and return the path.

    Skips the download if the file is already cached.
    """
    cache_path = Path(cache_dir or FOCUS_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or "focus_sample.csv"
    dest = cache_path / filename
    if not dest.exists():
        print(f"  Downloading {filename} from GitHub...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_focus_data(path: str) -> pd.DataFrame:
    """Parse a FOCUS CSV or CSV.GZ file into a typed DataFrame.

    Auto-detects the cost column (``EffectiveCost`` preferred over ``BilledCost``)
    and normalizes it to an internal ``_eff_cost`` column.  Adds ``_date``
    (tz-naive date from ``ChargePeriodStart``).

    Raises ``ValueError`` for missing required columns.
    """
    df = pd.read_csv(path, low_memory=False)   # pandas decompresses .gz natively

    if _DATE_COL not in df.columns:
        raise ValueError(f"FOCUS CSV missing required column: '{_DATE_COL}'")

    cost_src = _pick_cost_col(df)

    df[_DATE_COL] = pd.to_datetime(df[_DATE_COL], utc=True, errors="coerce")
    df["_date"] = df[_DATE_COL].dt.normalize().dt.tz_localize(None)
    df[_EFF_COST_COL] = pd.to_numeric(df[cost_src], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["_date"]).copy()
    return df


def aggregate_total_daily(
    df: pd.DataFrame,
    cost_col: str = _EFF_COST_COL,
    min_days: int = 30,
) -> pd.Series:
    """Aggregate ALL services into a single total daily cost series.

    Sums every billing line item across all providers/services per calendar day.
    Returns a pd.Series indexed by date, sorted chronologically.

    Raises ``ValueError`` if the result has fewer than ``min_days`` observations.
    """
    series = df.groupby("_date")[cost_col].sum().sort_index().clip(lower=0.0)
    series.name = "total_daily_cost"
    if len(series) < min_days:
        raise ValueError(
            f"FOCUS 데이터가 {len(series)}일치밖에 없습니다 (최소 {min_days}일 필요). "
            "더 긴 데이터를 사용하세요."
        )
    return series


def aggregate_daily(
    df: pd.DataFrame,
    cost_col: str = _EFF_COST_COL,
    group_by: Optional[Sequence[str]] = None,
    min_days: int = 21,
    min_nonzero_days: int = 14,
    min_mean_cost: float = 1.0,
) -> Dict[str, pd.Series]:
    """Aggregate hourly FOCUS rows to a daily cost series per group.

    Returns
    -------
    dict mapping ``"Provider / Category"`` label to daily pd.Series indexed by
    date (chronological).  Groups are excluded when they have:
    - fewer than ``min_days`` calendar days of observations,
    - fewer than ``min_nonzero_days`` days with cost > 0, or
    - a mean daily cost below ``min_mean_cost``.
    """
    group_cols = list(group_by or FOCUS_GROUP_BY)
    required = {"_date", cost_col, *group_cols}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"FOCUS data missing required column(s): {sorted(missing)}")

    agg = (
        df.groupby(["_date"] + group_cols, dropna=False)[cost_col]
        .sum()
        .reset_index()
    )
    result: Dict[str, pd.Series] = {}
    for keys, sub in agg.groupby(group_cols, dropna=False):
        label = (
            " / ".join(str(k) for k in keys)
            if isinstance(keys, tuple)
            else str(keys)
        )
        series = sub.set_index("_date")[cost_col].sort_index()
        nonzero_days = int((series > 0).sum())
        if (
            len(series) >= min_days
            and nonzero_days >= min_nonzero_days
            and series.mean() > min_mean_cost
        ):
            result[label] = series
    return result
