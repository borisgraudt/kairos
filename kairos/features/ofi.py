import pandas as pd

def order_flow_imbalance(df: pd.DataFrame, window: int = 50) -> pd.Series:
    signed = df["volume"] * df["side"].map({"B": 1, "S": -1})
    return signed.rolling(window).sum().rename("ofi")