# Kairos

ML signal engine for detecting short-horizon alpha in metals futures markets.
Trained on hourly Gold, Copper, and Aluminium futures with microstructure-aware features.

## Overview

Kairos extracts order book microstructure signals from commodity futures tick data
and trains a walk-forward validated LightGBM classifier to predict short-horizon
price direction. Features include bid-ask spread dynamics, order flow imbalance (OFI),
VPIN (Volume-synchronized Probability of Informed Trading), realized volatility,
and seasonality encoding.

## Stack

- Python 3.11+
- LightGBM · scikit-learn · pandas · yfinance
- Walk-forward cross-validation (no data leakage)
- Data: Gold (GC=F), Copper (HG=F), Aluminium (ALI=F) via Yahoo Finance

## Project structure
```
kairos/
├── data/
│   ├── loader.py       # yfinance futures loader
│   └── schema.py       # column contracts
├── features/
│   ├── spread.py       # bid-ask spread, mid-price, momentum
│   ├── ofi.py          # order flow imbalance
│   ├── vpin.py         # VPIN / informed trading probability
│   ├── vol.py          # realized volatility
│   └── commodities.py  # roll yield, seasonality, Gold/Copper ratio
├── models/
│   └── lgbm.py         # LightGBM baseline + walk-forward train loop
notebooks/
├── 01_baseline.ipynb   # feature analysis + baseline results
└── 02_macro_regimes.ipynb  # Gold/Copper ratio regime analysis
main.py
pyproject.toml
```

## Quickstart
```bash
# install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# run
python main.py
```

## References

- Easley, Lopez de Prado, O'Hara — *Flow Toxicity and Liquidity in a High Frequency World* (2011)
- Glosten, Milgrom — *Bid, Ask and Transaction Prices in a Specialist Market* (1985)

## Status

Active development. Expected completion: Q3 2026.