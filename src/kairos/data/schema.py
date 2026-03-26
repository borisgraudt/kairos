from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "bid", "ask", "price", "volume", "side"]

@dataclass
class TickSchema:
    timestamp: pd.Series   # datetime64[ns]
    bid: pd.Series         # float64
    ask: pd.Series         # float64
    price: pd.Series       # float64
    volume: pd.Series      # float64
    side: pd.Series        # "B" | "S"