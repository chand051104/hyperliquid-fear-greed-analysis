# Hyperliquid Fear/Greed Analysis

## Data Quality and Alignment
- `historical_data.csv`: 211,224 rows, 16 columns, 0 duplicates, 0 missing cells.
- `fear_greed_index.csv`: 2,644 rows, 4 columns, 0 duplicates, 0 missing cells.
- Timestamp integrity note: `Timestamp` has only 7 unique values, while `Timestamp IST` has 27,977; analysis uses `Timestamp IST`.
- Sentiment alignment: 6 trade rows had no same-day sentiment match and were excluded from Fear/Greed comparisons.

## Methodology
- Parsed and standardized both datasets; aligned on daily date.
- Used `Timestamp IST` as canonical event time because `Timestamp` is heavily rounded.
- Built account-day metrics: daily PnL, win rate, drawdown, trade count, trade size, long/short bias, leverage proxy.
- Defined leverage proxy as post-trade exposure / pre-trade exposure (capped at 50) due missing explicit margin leverage.
- Segmented traders via 70th percentile thresholds for leverage-proxy, activity, and consistency.
- Trained a bonus random-forest baseline to predict next-day profitable bucket from sentiment + lagged behavior features.

## Key Insights
- Typical performance is stronger on Greed days: median account-day PnL 265.25 USD vs 122.74 USD on Fear, and positive-day rate 64.3% vs 60.4%.
- Loss tails are materially worse in Fear: ES10 daily PnL is -11,003.82 USD vs -2,595.02 USD in Greed, indicating deeper downside episodes.
- Behavior shifts with sentiment: Fear has higher activity and size (105.4 vs 76.9 trades/day; 8,530 vs 5,955 USD average trade size), while net bias flips from long (0.041) to short (-0.042).
- High leverage-proxy traders underperform low leverage-proxy peers in both regimes: Fear 471 vs 6,647 mean daily PnL; Greed 2,117 vs 5,062.
- Frequent traders outperform infrequent traders, especially in Fear: Fear 10,926 vs 2,754; Greed 6,186 vs 3,641 mean daily PnL.

## Strategy Ideas
1. Rule 1: On Fear days, de-risk leverage-proxy accounts first. If a trader belongs to the high leverage-proxy segment, cap position expansion and reduce trade size; Fear downside tails are much larger, so preserve risk budget for only highest-conviction setups.
2. Rule 2: Increase trade count only for proven frequent/consistent operators. Infrequent traders should avoid reacting to sentiment with higher turnover; their average PnL lags frequent peers in both sentiment regimes.

## Model
-  Model: next-day profitability classifier AUC = 0.632 (accuracy 0.577) on time-based holdout, indicating modest predictive signal.

## Output Artifacts
- Tables: `outputs/tables/`
- Charts: `outputs/charts/`
- Cleaned data: `outputs/cleaned/`