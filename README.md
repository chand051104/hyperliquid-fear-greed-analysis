# Hyperliquid Sentiment vs Trader Behavior Analysis

This repository analyzes how market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid using:

- `/Users/saichandra/Desktop/assignment/historical_data.csv`
- `/Users/saichandra/Desktop/assignment/fear_greed_index.csv`

## What is included

- Notebook: `/Users/saichandra/Desktop/assignment/analysis.ipynb`
- Script: `/Users/saichandra/Desktop/assignment/analyze_hyperliquid_sentiment.py`
- Write-up: `/Users/saichandra/Desktop/assignment/outputs/summary.md`
- Charts: `/Users/saichandra/Desktop/assignment/outputs/charts/`
- Tables: `/Users/saichandra/Desktop/assignment/outputs/tables/`
- Cleaned/intermediate outputs: `/Users/saichandra/Desktop/assignment/outputs/cleaned/`

## Setup

```bash
cd /Users/saichandra/Desktop/assignment
python3 -m venv venv
venv/bin/pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run

Use writable matplotlib/font caches in restricted environments:

```bash
cd /Users/saichandra/Desktop/assignment
mkdir -p /tmp/mpl /tmp/fontcache
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp venv/bin/python analyze_hyperliquid_sentiment.py
```

## Output artifacts

- Data quality and prep:
  - `/Users/saichandra/Desktop/assignment/outputs/tables/data_quality_summary.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/cleaned/trades_enriched.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/cleaned/daily_account_metrics.csv`
- Fear vs Greed analysis:
  - `/Users/saichandra/Desktop/assignment/outputs/tables/performance_fear_vs_greed.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/tables/behavior_fear_vs_greed.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/tables/bootstrap_differences.csv`
- Segmentation:
  - `/Users/saichandra/Desktop/assignment/outputs/tables/trader_profiles.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/tables/segment_thresholds.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/tables/segment_performance.csv`
- Bonus model:
  - `/Users/saichandra/Desktop/assignment/outputs/tables/predictive_model_metrics.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/tables/predictive_feature_importance.csv`
  - `/Users/saichandra/Desktop/assignment/outputs/charts/predictive_model_roc.png`
- Charts for insights:
  - `/Users/saichandra/Desktop/assignment/outputs/charts/performance_fear_vs_greed.png`
  - `/Users/saichandra/Desktop/assignment/outputs/charts/behavior_fear_vs_greed.png`
  - `/Users/saichandra/Desktop/assignment/outputs/charts/segment_performance_heatmap.png`

## Notes on methodology

- Daily alignment is done on parsed `Timestamp IST` (not `Timestamp`, which is heavily rounded).
- Sentiment buckets used:
  - Fear: `Fear`, `Extreme Fear`
  - Greed: `Greed`, `Extreme Greed`
  - Neutral kept for optional model only
- Because direct account equity/margin is not present, a leverage proxy is used:
  - `leverage_proxy = post_trade_exposure / pre_trade_exposure`
