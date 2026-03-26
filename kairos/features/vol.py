import pandas as pd
import numpy as np

def realized_vol(df: pd.DataFrame, window: int = 100) -> pd.Series:
    mid = (df["bid"] + df["ask"]) / 2
    log_ret = np.log(mid / mid.shift(1))
    return log_ret.rolling(window).std().rename("realized_vol")