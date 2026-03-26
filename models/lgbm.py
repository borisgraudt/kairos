import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    from kairos.features.spread import bid_ask_spread, relative_spread
    from kairos.features.ofi import order_flow_imbalance
    from kairos.features.vol import realized_vol

    feats = pd.DataFrame(index=df.index)
    feats["spread"]       = bid_ask_spread(df)
    feats["rel_spread"]   = relative_spread(df)
    feats["ofi"]          = order_flow_imbalance(df)
    feats["realized_vol"] = realized_vol(df)
    return feats

def make_target(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    mid = (df["bid"] + df["ask"]) / 2
    future_ret = mid.shift(-horizon) / mid - 1
    return (future_ret > 0).astype(int).rename("target")

def walk_forward_train(df: pd.DataFrame, n_splits: int = 5):
    X = make_features(df).dropna()
    y = make_target(df).loc[X.index].dropna()
    X = X.loc[y.index]

    split_size = len(X) // (n_splits + 1)
    results = []

    for i in range(1, n_splits + 1):
        train_end = i * split_size
        test_end  = train_end + split_size

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test,  y_test  = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results.append({"fold": i, "accuracy": acc, "model": model})

    return results