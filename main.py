from kairos.data.loader import load_futures, METALS
from kairos.features.spread import bid_ask_spread, mid_price
from kairos.features.ofi import order_flow_imbalance
from kairos.features.vol import realized_vol
from kairos.features.commodities import seasonality_encode, safe_haven_ratio
from kairos.models.lgbm import make_features, make_target, walk_forward_train

print("Загружаем gold futures...")
df = load_futures(METALS["gold"], period="1y", interval="1h")
print(f"Загружено строк: {len(df)}")
print(df.head())

print("\nСчитаем фичи...")
feats = make_features(df)
print(feats.dropna().head())

print("\nЗапускаем walk-forward...")
results = walk_forward_train(df, n_splits=3)
for r in results:
    print(f"Fold {r['fold']}: accuracy = {r['accuracy']:.3f}")