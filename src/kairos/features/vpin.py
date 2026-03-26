import pandas as pd
import numpy as np

def vpin(df: pd.DataFrame, bucket_size: float = 1000.0) -> pd.Series:
    df = df.copy()
    df["signed_vol"] = df["volume"] * df["side"].map({"B": 1.0, "S": -1.0})
    df["cum_vol"] = df["volume"].cumsum()
    df["bucket"] = (df["cum_vol"] // bucket_size).astype(int)

    buckets = df.groupby("bucket").agg(
        buy_vol=("volume", lambda x: x[df.loc[x.index, "side"] == "B"].sum()),
        total_vol=("volume", "sum"),
    )
    buckets["vpin"] = np.abs(
        buckets["buy_vol"] - (buckets["total_vol"] - buckets["buy_vol"])
    ) / buckets["total_vol"]

    return buckets["vpin"].rolling(50).mean()