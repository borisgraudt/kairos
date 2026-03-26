import pandas as pd
from pathlib import Path
from .schema import REQUIRED_COLUMNS

def load_ticks(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    df.columns = df.columns.str.lower().str.strip()
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[REQUIRED_COLUMNS]