import yfinance as yf
import pandas as pd

METALS = {
    "gold":      "GC=F",
    "copper":    "HG=F",
    "aluminium": "ALI=F",
}

def load_futures(ticker: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = raw.columns.str.lower()
    df = raw.reset_index().rename(columns={"Datetime": "timestamp"})
    df = df.rename(columns={"close": "price", "high": "ask", "low": "bid"})
    df["side"] = "B"

    required = ["timestamp", "bid", "ask", "price", "volume", "side"]
    return df[required].dropna().reset_index(drop=True)