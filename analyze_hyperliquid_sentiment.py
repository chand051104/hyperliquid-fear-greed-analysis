#!/usr/bin/env python3
"""Analyze the relationship between Fear/Greed sentiment and Hyperliquid trader behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TRADES_PATH = ROOT / "historical_data.csv"
SENTIMENT_PATH = ROOT / "fear_greed_index.csv"
OUTPUT_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
CHARTS_DIR = OUTPUT_DIR / "charts"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
REPORT_PATH = OUTPUT_DIR / "summary.md"


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, TABLES_DIR, CHARTS_DIR, CLEANED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def profile_dataframe(name: str, df: pd.DataFrame) -> dict[str, float]:
    total_missing = int(df.isna().sum().sum())
    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": total_missing,
        "missing_cell_pct": float(total_missing / (df.shape[0] * df.shape[1])),
    }


def bootstrap_diff(
    fear: pd.Series,
    greed: pd.Series,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    fear_values = fear.dropna().to_numpy()
    greed_values = greed.dropna().to_numpy()
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for idx in range(n_boot):
        fear_sample = rng.choice(fear_values, size=len(fear_values), replace=True)
        greed_sample = rng.choice(greed_values, size=len(greed_values), replace=True)
        diffs[idx] = stat_fn(fear_sample) - stat_fn(greed_sample)

    return {
        "fear_minus_greed": float(np.mean(diffs)),
        "ci_low_95": float(np.quantile(diffs, 0.025)),
        "ci_high_95": float(np.quantile(diffs, 0.975)),
    }


def expected_shortfall_10(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    p10 = values.quantile(0.10)
    return float(values[values <= p10].mean())


def prepare_data(
    trades_raw: pd.DataFrame, sentiment_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = trades_raw.copy()
    sentiment = sentiment_raw.copy()

    trades["timestamp_ist"] = pd.to_datetime(
        trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    trades["timestamp_ms"] = pd.to_datetime(
        trades["Timestamp"], unit="ms", errors="coerce", utc=True
    )
    trades["date"] = trades["timestamp_ist"].dt.date

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date
    sentiment["sentiment_bucket"] = sentiment["classification"].replace(
        {"Extreme Fear": "Fear", "Extreme Greed": "Greed"}
    )

    merged = trades.merge(
        sentiment[["date", "value", "classification", "sentiment_bucket"]],
        on="date",
        how="left",
    )

    side = merged["Side"].str.upper()
    merged["side_sign"] = np.where(side.eq("BUY"), 1, -1)
    merged["signed_size_usd"] = merged["Size USD"] * merged["side_sign"]
    merged["signed_size_tokens"] = merged["Size Tokens"] * merged["side_sign"]

    merged["pos_before_usd"] = merged["Start Position"].abs() * merged["Execution Price"]
    merged["pos_after_tokens"] = merged["Start Position"] + merged["signed_size_tokens"]
    merged["pos_after_usd"] = merged["pos_after_tokens"].abs() * merged["Execution Price"]

    # No explicit wallet equity exists in this dataset; use exposure amplification as leverage proxy.
    merged["leverage_proxy"] = (
        merged["pos_after_usd"] / (merged["pos_before_usd"] + 1.0)
    ).clip(lower=0, upper=50)

    merged["is_realized"] = (merged["Closed PnL"] != 0).astype(int)
    merged["is_win"] = (merged["Closed PnL"] > 0).astype(int)
    merged["is_loss"] = (merged["Closed PnL"] < 0).astype(int)

    daily = (
        merged.groupby(["Account", "date", "sentiment_bucket"], dropna=False, as_index=False)
        .agg(
            trades=("Trade ID", "count"),
            daily_pnl_usd=("Closed PnL", "sum"),
            avg_trade_size_usd=("Size USD", "mean"),
            total_notional_usd=("Size USD", "sum"),
            realized_trades=("is_realized", "sum"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            buy_notional_usd=("signed_size_usd", lambda x: x[x > 0].sum()),
            sell_notional_usd=("signed_size_usd", lambda x: -x[x < 0].sum()),
            avg_leverage_proxy=("leverage_proxy", "mean"),
            p90_leverage_proxy=("leverage_proxy", lambda x: x.quantile(0.90)),
            total_fees_usd=("Fee", "sum"),
        )
        .sort_values(["Account", "date"])
    )

    daily["win_rate"] = np.where(
        daily["realized_trades"] > 0, daily["wins"] / daily["realized_trades"], np.nan
    )
    daily["long_short_ratio"] = daily["buy_notional_usd"] / daily["sell_notional_usd"].replace(
        0, np.nan
    )
    daily["net_long_bias"] = (
        (daily["buy_notional_usd"] - daily["sell_notional_usd"])
        / (daily["buy_notional_usd"] + daily["sell_notional_usd"] + 1e-9)
    )

    daily["cum_pnl_usd"] = daily.groupby("Account")["daily_pnl_usd"].cumsum()
    daily["running_peak_usd"] = daily.groupby("Account")["cum_pnl_usd"].cummax()
    daily["drawdown_usd"] = daily["cum_pnl_usd"] - daily["running_peak_usd"]

    return merged, daily, sentiment


def fear_greed_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fg_daily = daily[daily["sentiment_bucket"].isin(["Fear", "Greed"])].copy()

    performance = (
        fg_daily.groupby("sentiment_bucket", as_index=False)
        .agg(
            account_days=("Account", "count"),
            traders=("Account", "nunique"),
            mean_daily_pnl_usd=("daily_pnl_usd", "mean"),
            median_daily_pnl_usd=("daily_pnl_usd", "median"),
            positive_day_rate=("daily_pnl_usd", lambda s: (s > 0).mean()),
            mean_win_rate=("win_rate", "mean"),
            mean_drawdown_usd=("drawdown_usd", "mean"),
            p10_drawdown_usd=("drawdown_usd", lambda s: s.quantile(0.10)),
            p05_daily_pnl_usd=("daily_pnl_usd", lambda s: s.quantile(0.05)),
            es10_daily_pnl_usd=("daily_pnl_usd", expected_shortfall_10),
        )
        .sort_values("sentiment_bucket")
    )

    behavior = (
        fg_daily.groupby("sentiment_bucket", as_index=False)
        .agg(
            mean_trades_per_account_day=("trades", "mean"),
            median_trades_per_account_day=("trades", "median"),
            mean_trade_size_usd=("avg_trade_size_usd", "mean"),
            median_trade_size_usd=("avg_trade_size_usd", "median"),
            mean_total_notional_usd=("total_notional_usd", "mean"),
            mean_leverage_proxy=("avg_leverage_proxy", "mean"),
            median_leverage_proxy=("avg_leverage_proxy", "median"),
            median_long_short_ratio=(
                "long_short_ratio",
                lambda s: s.replace([np.inf, -np.inf], np.nan).median(),
            ),
            mean_long_short_ratio_capped=(
                "long_short_ratio",
                lambda s: s.replace([np.inf, -np.inf], np.nan).clip(0, 5).mean(),
            ),
            mean_net_long_bias=("net_long_bias", "mean"),
            share_net_long_days=("net_long_bias", lambda s: (s > 0).mean()),
        )
        .sort_values("sentiment_bucket")
    )

    fear = fg_daily[fg_daily["sentiment_bucket"] == "Fear"]
    greed = fg_daily[fg_daily["sentiment_bucket"] == "Greed"]

    tests = []
    metric_specs: list[tuple[str, str, Callable[[np.ndarray], float]]] = [
        ("daily_pnl_usd", "median_daily_pnl_usd", np.median),
        ("win_rate", "mean_win_rate", np.mean),
        ("drawdown_usd", "p10_drawdown_usd", lambda x: float(np.quantile(x, 0.10))),
        ("trades", "mean_trades_per_account_day", np.mean),
        ("avg_trade_size_usd", "mean_trade_size_usd", np.mean),
        ("avg_leverage_proxy", "mean_leverage_proxy", np.mean),
        ("net_long_bias", "mean_net_long_bias", np.mean),
    ]

    for column, label, fn in metric_specs:
        stats = bootstrap_diff(fear[column], greed[column], fn)
        tests.append(
            {
                "metric": label,
                "fear_minus_greed": stats["fear_minus_greed"],
                "ci_low_95": stats["ci_low_95"],
                "ci_high_95": stats["ci_high_95"],
            }
        )
    tests_df = pd.DataFrame(tests)

    return performance, behavior, tests_df


def segment_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fg_daily = daily[daily["sentiment_bucket"].isin(["Fear", "Greed"])].copy()

    trader_profile = (
        fg_daily.groupby("Account", as_index=False)
        .agg(
            active_days=("date", "nunique"),
            total_trades=("trades", "sum"),
            avg_trades_per_day=("trades", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_leverage_proxy=("avg_leverage_proxy", "mean"),
            positive_day_rate=("daily_pnl_usd", lambda s: (s > 0).mean()),
            total_pnl_usd=("daily_pnl_usd", "sum"),
            mean_daily_pnl_usd=("daily_pnl_usd", "mean"),
            pnl_volatility_usd=("daily_pnl_usd", "std"),
            mean_win_rate=("win_rate", "mean"),
        )
        .sort_values("Account")
    )

    trader_profile["pnl_volatility_usd"] = trader_profile["pnl_volatility_usd"].fillna(0)
    trader_profile["mean_win_rate"] = trader_profile["mean_win_rate"].fillna(0)

    lev_q70 = float(trader_profile["avg_leverage_proxy"].quantile(0.70))
    freq_q70 = float(trader_profile["avg_trades_per_day"].quantile(0.70))
    cons_q70 = float(trader_profile["positive_day_rate"].quantile(0.70))

    trader_profile["leverage_segment"] = np.where(
        trader_profile["avg_leverage_proxy"] >= lev_q70, "High leverage-proxy", "Low leverage-proxy"
    )
    trader_profile["activity_segment"] = np.where(
        trader_profile["avg_trades_per_day"] >= freq_q70, "Frequent", "Infrequent"
    )
    trader_profile["consistency_segment"] = np.where(
        trader_profile["positive_day_rate"] >= cons_q70,
        "Consistent winners",
        "Inconsistent",
    )

    segmented = fg_daily.merge(
        trader_profile[
            ["Account", "leverage_segment", "activity_segment", "consistency_segment"]
        ],
        on="Account",
        how="left",
    )

    segment_frames = []
    for segment_col in ("leverage_segment", "activity_segment", "consistency_segment"):
        grouped = (
            segmented.groupby(["sentiment_bucket", segment_col], as_index=False)
            .agg(
                account_days=("Account", "count"),
                traders=("Account", "nunique"),
                mean_daily_pnl_usd=("daily_pnl_usd", "mean"),
                median_daily_pnl_usd=("daily_pnl_usd", "median"),
                positive_day_rate=("daily_pnl_usd", lambda s: (s > 0).mean()),
                mean_win_rate=("win_rate", "mean"),
                mean_trades=("trades", "mean"),
                mean_trade_size_usd=("avg_trade_size_usd", "mean"),
                mean_leverage_proxy=("avg_leverage_proxy", "mean"),
                mean_net_long_bias=("net_long_bias", "mean"),
                es10_daily_pnl_usd=("daily_pnl_usd", expected_shortfall_10),
            )
            .rename(columns={segment_col: "segment"})
        )
        grouped["segment_type"] = segment_col
        segment_frames.append(grouped)

    segment_summary = pd.concat(segment_frames, ignore_index=True)

    threshold_table = pd.DataFrame(
        [
            {"threshold_name": "leverage_q70", "value": lev_q70},
            {"threshold_name": "activity_q70", "value": freq_q70},
            {"threshold_name": "consistency_q70", "value": cons_q70},
        ]
    )

    return trader_profile, segment_summary, threshold_table


def train_predictive_model(
    daily: pd.DataFrame, sentiment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = daily.merge(sentiment[["date", "value"]], on="date", how="left")
    model_df = model_df[
        model_df["sentiment_bucket"].isin(["Fear", "Greed", "Neutral"])
    ].copy()
    model_df = model_df.sort_values(["Account", "date"])

    model_df["next_day_pnl_usd"] = model_df.groupby("Account")["daily_pnl_usd"].shift(-1)
    model_df["next_day_profitable"] = (model_df["next_day_pnl_usd"] > 0).astype(int)

    lag_cols = [
        "daily_pnl_usd",
        "trades",
        "avg_trade_size_usd",
        "total_notional_usd",
        "avg_leverage_proxy",
        "net_long_bias",
    ]
    for col in lag_cols:
        model_df[f"lag1_{col}"] = model_df.groupby("Account")[col].shift(1)

    model_df = model_df.dropna(subset=["next_day_pnl_usd"]).copy()
    features = [
        "value",
        "sentiment_bucket",
        "lag1_daily_pnl_usd",
        "lag1_trades",
        "lag1_avg_trade_size_usd",
        "lag1_total_notional_usd",
        "lag1_avg_leverage_proxy",
        "lag1_net_long_bias",
    ]

    use_df = model_df.dropna(subset=["date"]).copy()
    unique_dates = sorted(use_df["date"].unique())
    split_idx = int(len(unique_dates) * 0.80)
    split_date = unique_dates[split_idx]
    train_mask = use_df["date"] < split_date
    test_mask = use_df["date"] >= split_date

    X_train = use_df.loc[train_mask, features]
    y_train = use_df.loc[train_mask, "next_day_profitable"]
    X_test = use_df.loc[test_mask, features]
    y_test = use_df.loc[test_mask, "next_day_profitable"]

    num_cols = [col for col in features if col != "sentiment_bucket"]
    cat_cols = ["sentiment_bucket"]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.50).astype(int)

    metrics_df = pd.DataFrame(
        [
            {
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "split_date": split_date,
                "test_auc": roc_auc_score(y_test, proba),
                "test_accuracy": accuracy_score(y_test, preds),
                "test_precision": precision_score(y_test, preds, zero_division=0),
                "test_recall": recall_score(y_test, preds, zero_division=0),
                "test_f1": f1_score(y_test, preds, zero_division=0),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            }
        ]
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_classifier = model.named_steps["classifier"]

    cat_feature_names = (
        fitted_preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + cat_feature_names
    importances = fitted_classifier.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fpr, tpr, thresholds = roc_curve(y_test, proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    return metrics_df, importance_df, roc_df


def build_charts(
    daily: pd.DataFrame,
    performance: pd.DataFrame,
    behavior: pd.DataFrame,
    segment_summary: pd.DataFrame,
    roc_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    order = ["Fear", "Greed"]

    fg_daily = daily[daily["sentiment_bucket"].isin(order)].copy()
    plot_data = fg_daily[["sentiment_bucket", "daily_pnl_usd"]].copy()
    plot_data["winsorized_pnl"] = plot_data.groupby("sentiment_bucket")["daily_pnl_usd"].transform(
        lambda s: s.clip(s.quantile(0.05), s.quantile(0.95))
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(
        data=plot_data,
        x="sentiment_bucket",
        y="winsorized_pnl",
        hue="sentiment_bucket",
        order=order,
        ax=axes[0],
        dodge=False,
        palette=["#d66d6d", "#73a3d6"],
        legend=False,
    )
    axes[0].set_title("Account-Day PnL by Sentiment (Winsorized 5%-95%)")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Daily PnL (USD)")

    perf_plot = performance.set_index("sentiment_bucket").reindex(order)
    x = np.arange(len(order))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        perf_plot["positive_day_rate"],
        width=width,
        color="#4c78a8",
        label="Positive day rate",
    )
    axes[1].set_ylabel("Positive Day Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_title("Hit Rate vs Tail-Loss Proxy")

    ax2 = axes[1].twinx()
    ax2.bar(
        x + width / 2,
        perf_plot["es10_daily_pnl_usd"],
        width=width,
        color="#e15759",
        alpha=0.8,
        label="ES10 daily PnL",
    )
    ax2.set_ylabel("Expected Shortfall 10% (USD)")

    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "performance_fear_vs_greed.png", dpi=180)
    plt.close(fig)

    behavior_plot = behavior.set_index("sentiment_bucket").reindex(order).reset_index()
    metric_specs = [
        ("mean_trades_per_account_day", "Trades / Account-Day"),
        ("mean_trade_size_usd", "Avg Trade Size (USD)"),
        ("mean_leverage_proxy", "Leverage Proxy"),
        ("mean_net_long_bias", "Net Long Bias"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (metric, title) in zip(axes.flatten(), metric_specs):
        sns.barplot(
            data=behavior_plot,
            x="sentiment_bucket",
            y=metric,
            hue="sentiment_bucket",
            order=order,
            ax=axis,
            dodge=False,
            palette=["#d66d6d", "#73a3d6"],
            legend=False,
        )
        axis.set_title(title)
        axis.set_xlabel("Sentiment")
        axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "behavior_fear_vs_greed.png", dpi=180)
    plt.close(fig)

    heatmap_data = segment_summary[
        segment_summary["segment_type"].isin(
            ["leverage_segment", "activity_segment", "consistency_segment"]
        )
    ].copy()
    heatmap_data["row_name"] = (
        heatmap_data["segment_type"]
        .str.replace("_segment", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
        + " | "
        + heatmap_data["segment"]
    )
    heatmap_pivot = heatmap_data.pivot(
        index="row_name", columns="sentiment_bucket", values="mean_daily_pnl_usd"
    ).reindex(columns=order)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Mean Daily PnL (USD)"},
        ax=ax,
    )
    ax.set_title("Segment Performance by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Segment")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "segment_performance_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#4c78a8", linewidth=2, label="Model ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Next-Day Profitability Model ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "predictive_model_roc.png", dpi=180)
    plt.close(fig)


def write_summary(
    quality: pd.DataFrame,
    merged: pd.DataFrame,
    daily: pd.DataFrame,
    performance: pd.DataFrame,
    behavior: pd.DataFrame,
    segment_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> None:
    perf = performance.set_index("sentiment_bucket")
    beh = behavior.set_index("sentiment_bucket")

    fear = perf.loc["Fear"]
    greed = perf.loc["Greed"]

    segment_focus = segment_summary[
        segment_summary["segment_type"].isin(["leverage_segment", "activity_segment"])
    ].copy()
    seg_pivot = segment_focus.pivot_table(
        index=["segment_type", "segment"],
        columns="sentiment_bucket",
        values="mean_daily_pnl_usd",
    )

    high_lev_fear = seg_pivot.loc[("leverage_segment", "High leverage-proxy"), "Fear"]
    low_lev_fear = seg_pivot.loc[("leverage_segment", "Low leverage-proxy"), "Fear"]
    high_lev_greed = seg_pivot.loc[("leverage_segment", "High leverage-proxy"), "Greed"]
    low_lev_greed = seg_pivot.loc[("leverage_segment", "Low leverage-proxy"), "Greed"]

    frequent_fear = seg_pivot.loc[("activity_segment", "Frequent"), "Fear"]
    infrequent_fear = seg_pivot.loc[("activity_segment", "Infrequent"), "Fear"]
    frequent_greed = seg_pivot.loc[("activity_segment", "Frequent"), "Greed"]
    infrequent_greed = seg_pivot.loc[("activity_segment", "Infrequent"), "Greed"]

    model_row = model_metrics.iloc[0]

    timestamp_unique = int(merged["Timestamp"].nunique())
    timestamp_ist_unique = int(merged["Timestamp IST"].nunique())
    missing_sentiment_rows = int(merged["sentiment_bucket"].isna().sum())

    methodology = [
        "Parsed and standardized both datasets; aligned on daily date.",
        "Used `Timestamp IST` as canonical event time because `Timestamp` is heavily rounded.",
        "Built account-day metrics: daily PnL, win rate, drawdown, trade count, trade size, long/short bias, leverage proxy.",
        "Defined leverage proxy as post-trade exposure / pre-trade exposure (capped at 50) due missing explicit margin leverage.",
        "Segmented traders via 70th percentile thresholds for leverage-proxy, activity, and consistency.",
        "Trained a bonus random-forest baseline to predict next-day profitable bucket from sentiment + lagged behavior features.",
    ]

    insights = [
        (
            f"Typical performance is stronger on Greed days: median account-day PnL "
            f"{greed['median_daily_pnl_usd']:,.2f} USD vs {fear['median_daily_pnl_usd']:,.2f} USD on Fear, "
            f"and positive-day rate {greed['positive_day_rate']:.1%} vs {fear['positive_day_rate']:.1%}."
        ),
        (
            f"Loss tails are materially worse in Fear: ES10 daily PnL is {fear['es10_daily_pnl_usd']:,.2f} USD "
            f"vs {greed['es10_daily_pnl_usd']:,.2f} USD in Greed, indicating deeper downside episodes."
        ),
        (
            f"Behavior shifts with sentiment: Fear has higher activity and size "
            f"({beh.loc['Fear', 'mean_trades_per_account_day']:.1f} vs {beh.loc['Greed', 'mean_trades_per_account_day']:.1f} trades/day; "
            f"{beh.loc['Fear', 'mean_trade_size_usd']:,.0f} vs {beh.loc['Greed', 'mean_trade_size_usd']:,.0f} USD average trade size), "
            f"while net bias flips from long ({beh.loc['Fear', 'mean_net_long_bias']:.3f}) to short ({beh.loc['Greed', 'mean_net_long_bias']:.3f})."
        ),
        (
            f"High leverage-proxy traders underperform low leverage-proxy peers in both regimes: "
            f"Fear {high_lev_fear:,.0f} vs {low_lev_fear:,.0f} mean daily PnL; "
            f"Greed {high_lev_greed:,.0f} vs {low_lev_greed:,.0f}."
        ),
        (
            f"Frequent traders outperform infrequent traders, especially in Fear: "
            f"Fear {frequent_fear:,.0f} vs {infrequent_fear:,.0f}; "
            f"Greed {frequent_greed:,.0f} vs {infrequent_greed:,.0f} mean daily PnL."
        ),
    ]

    strategy_rules = [
        (
            "Rule 1: On Fear days, de-risk leverage-proxy accounts first. "
            "If a trader belongs to the high leverage-proxy segment, cap position expansion and reduce trade size; "
            "Fear downside tails are much larger, so preserve risk budget for only highest-conviction setups."
        ),
        (
            "Rule 2: Increase trade count only for proven frequent/consistent operators. "
            "Infrequent traders should avoid reacting to sentiment with higher turnover; "
            "their average PnL lags frequent peers in both sentiment regimes."
        ),
    ]

    bonus = (
        f"Bonus model: next-day profitability classifier AUC = {model_row['test_auc']:.3f} "
        f"(accuracy {model_row['test_accuracy']:.3f}) on time-based holdout, indicating modest predictive signal."
    )

    lines: list[str] = []
    lines.append("# Hyperliquid Fear/Greed Analysis")
    lines.append("")
    lines.append("## Data Quality and Alignment")
    lines.append(
        f"- `historical_data.csv`: {int(quality.loc[quality['dataset']=='historical_data','rows'].iloc[0]):,} rows, "
        f"{int(quality.loc[quality['dataset']=='historical_data','columns'].iloc[0])} columns, "
        f"{int(quality.loc[quality['dataset']=='historical_data','duplicate_rows'].iloc[0])} duplicates, "
        f"{int(quality.loc[quality['dataset']=='historical_data','total_missing_cells'].iloc[0])} missing cells."
    )
    lines.append(
        f"- `fear_greed_index.csv`: {int(quality.loc[quality['dataset']=='fear_greed_index','rows'].iloc[0]):,} rows, "
        f"{int(quality.loc[quality['dataset']=='fear_greed_index','columns'].iloc[0])} columns, "
        f"{int(quality.loc[quality['dataset']=='fear_greed_index','duplicate_rows'].iloc[0])} duplicates, "
        f"{int(quality.loc[quality['dataset']=='fear_greed_index','total_missing_cells'].iloc[0])} missing cells."
    )
    lines.append(
        f"- Timestamp integrity note: `Timestamp` has only {timestamp_unique} unique values, "
        f"while `Timestamp IST` has {timestamp_ist_unique:,}; analysis uses `Timestamp IST`."
    )
    lines.append(
        f"- Sentiment alignment: {missing_sentiment_rows} trade rows had no same-day sentiment match and were excluded from Fear/Greed comparisons."
    )
    lines.append("")
    lines.append("## Methodology")
    for item in methodology:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Key Insights")
    for item in insights:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Strategy Ideas")
    for idx, item in enumerate(strategy_rules, start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")
    lines.append("## Bonus")
    lines.append(f"- {bonus}")
    lines.append("")
    lines.append("## Output Artifacts")
    lines.append("- Tables: `outputs/tables/`")
    lines.append("- Charts: `outputs/charts/`")
    lines.append("- Cleaned data: `outputs/cleaned/`")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    trades_raw = pd.read_csv(TRADES_PATH)
    sentiment_raw = pd.read_csv(SENTIMENT_PATH)

    quality_df = pd.DataFrame(
        [
            profile_dataframe("historical_data", trades_raw),
            profile_dataframe("fear_greed_index", sentiment_raw),
        ]
    )
    quality_df.to_csv(TABLES_DIR / "data_quality_summary.csv", index=False)

    merged, daily, sentiment_clean = prepare_data(trades_raw, sentiment_raw)
    merged.to_csv(CLEANED_DIR / "trades_enriched.csv", index=False)
    daily.to_csv(CLEANED_DIR / "daily_account_metrics.csv", index=False)

    performance_df, behavior_df, tests_df = fear_greed_tables(daily)
    performance_df.to_csv(TABLES_DIR / "performance_fear_vs_greed.csv", index=False)
    behavior_df.to_csv(TABLES_DIR / "behavior_fear_vs_greed.csv", index=False)
    tests_df.to_csv(TABLES_DIR / "bootstrap_differences.csv", index=False)

    trader_profile_df, segment_summary_df, segment_thresholds_df = segment_tables(daily)
    trader_profile_df.to_csv(TABLES_DIR / "trader_profiles.csv", index=False)
    segment_summary_df.to_csv(TABLES_DIR / "segment_performance.csv", index=False)
    segment_thresholds_df.to_csv(TABLES_DIR / "segment_thresholds.csv", index=False)

    model_metrics_df, feature_importance_df, roc_df = train_predictive_model(
        daily, sentiment_clean
    )
    model_metrics_df.to_csv(TABLES_DIR / "predictive_model_metrics.csv", index=False)
    feature_importance_df.to_csv(TABLES_DIR / "predictive_feature_importance.csv", index=False)
    roc_df.to_csv(TABLES_DIR / "predictive_model_roc_points.csv", index=False)

    build_charts(daily, performance_df, behavior_df, segment_summary_df, roc_df)
    write_summary(
        quality_df,
        merged,
        daily,
        performance_df,
        behavior_df,
        segment_summary_df,
        model_metrics_df,
    )

    print("Analysis complete. Artifacts written to:")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
