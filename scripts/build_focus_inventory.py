"""Summarize raw FOCUS cache files into a reproducible inventory JSON.

For each cached FOCUS sample file under ``.focus_cache/`` this script records:
  - byte size on disk
  - row count
  - column count
  - date range of ``ChargePeriodStart``
  - unique billing days
  - per-provider row counts (``ProviderName``)

This produces ``outputs/focus_data_inventory.json`` so claims like
"full FOCUS sample contains 5,488,359 rows" are backed by a regenerable artifact
instead of only appearing in the report narrative.

Usage
-----
    python scripts/build_focus_inventory.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finops_benchmark.config import FOCUS_CACHE_DIR  # noqa: E402

_DATE_COL = "ChargePeriodStart"
_PROVIDER_COL = "ProviderName"
_USECOLS = [_DATE_COL, _PROVIDER_COL]
_CHUNK = 500_000


def _column_count(path: Path) -> int:
    """Read only the header row to count columns without materializing the file."""
    head = pd.read_csv(path, nrows=0)
    return int(head.shape[1])


def summarize_focus_file(path: Path) -> dict:
    """Compute inventory stats for a single FOCUS CSV/CSV.GZ file."""
    n_cols = _column_count(path)

    n_rows = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    unique_dates: set[pd.Timestamp] = set()
    provider_counts: dict[str, int] = {}

    reader = pd.read_csv(
        path,
        usecols=_USECOLS,
        chunksize=_CHUNK,
        low_memory=False,
    )
    for chunk in reader:
        n_rows += len(chunk)
        ts = pd.to_datetime(chunk[_DATE_COL], utc=True, errors="coerce")
        days = ts.dt.normalize().dt.tz_localize(None).dropna()
        if not days.empty:
            unique_dates.update(days.unique().tolist())
            cur_min = days.min()
            cur_max = days.max()
            date_min = cur_min if date_min is None else min(date_min, cur_min)
            date_max = cur_max if date_max is None else max(date_max, cur_max)
        prov = chunk[_PROVIDER_COL].fillna("(missing)").value_counts()
        for name, count in prov.items():
            provider_counts[name] = provider_counts.get(name, 0) + int(count)

    return {
        "file": path.name,
        "path": str(path).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "rows": int(n_rows),
        "columns": n_cols,
        "date_min": date_min.strftime("%Y-%m-%d") if date_min is not None else None,
        "date_max": date_max.strftime("%Y-%m-%d") if date_max is not None else None,
        "unique_billing_days": len(unique_dates),
        "provider_row_counts": dict(
            sorted(provider_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def main(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"cache dir not found: {cache_dir}")

    candidates = sorted(
        p for p in cache_dir.iterdir()
        if p.suffix in (".csv", ".gz") and p.is_file()
    )
    if not candidates:
        raise SystemExit(f"no FOCUS sample files in {cache_dir}")

    entries = []
    for path in candidates:
        print(f"  Summarizing {path.name} ({path.stat().st_size:,} bytes)...")
        entries.append(summarize_focus_file(path))

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir).replace("\\", "/"),
        "files": entries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate outputs/focus_data_inventory.json from cached FOCUS files."
    )
    parser.add_argument("--cache-dir", default=FOCUS_CACHE_DIR)
    parser.add_argument("--output", default="outputs/focus_data_inventory.json")
    main(parser.parse_args())
