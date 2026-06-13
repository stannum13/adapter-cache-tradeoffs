from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_summaries(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(runs_dir).glob("*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)
