import pandas as pd
from pathlib import Path
from .schema import REQUIRED_COLUMNS
import yfinance as yf

METALS = {
    "gold":      "GC=F",
    "copper":    "HG=F",
    "aluminium": "ALI=F",
}

def load_futures(ticker: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                             "close": "price", "volume": "volume"})
    df["bid"] = df["low"]   # approximation для hourly data
    df["ask"] = df["high"]
    df["side"] = "B"        # placeholder
    return df.dropna().reset_index()

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
