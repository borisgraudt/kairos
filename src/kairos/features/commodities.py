import pandas as pd
import numpy as np

def roll_yield(near: pd.Series, far: pd.Series, days_to_expiry: int = 30) -> pd.Series:
    return ((near / far - 1) * (365 / days_to_expiry)).rename("roll_yield")

def seasonality_encode(df: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(df["timestamp"]).dt.month
    df["season_sin"] = np.sin(2 * np.pi * month / 12)
    df["season_cos"] = np.cos(2 * np.pi * month / 12)
    return df

def safe_haven_ratio(gold: pd.Series, copper: pd.Series) -> pd.Series:
    return (gold / copper).rename("gc_ratio")