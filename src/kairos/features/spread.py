import pandas as pd

def bid_ask_spread(df: pd.DataFrame) -> pd.Series:
    return (df["ask"] - df["bid"]).rename("spread")

def mid_price(df: pd.DataFrame) -> pd.Series:
    return ((df["bid"] + df["ask"]) / 2).rename("mid")

def relative_spread(df: pd.DataFrame) -> pd.Series:
    mid = mid_price(df)
    return ((df["ask"] - df["bid"]) / mid).rename("rel_spread")