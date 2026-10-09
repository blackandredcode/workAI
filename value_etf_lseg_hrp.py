"""
Value ETF portfolio: LSEG + value score + HRP + risk constraints
Capital: $250,000 | Scheduled rebalance: every 2 months

Important:
- Requires an entitled LSEG Workspace/Desktop session or a configured platform session.
- Verify the LSEG field mnemonics available under your organization's subscription in Data Item Browser.
- This script deliberately fails closed if the configured total-return field is unavailable.
- Historical value fields must be point-in-time/as-of data for a valid historical factor backtest.
- ETF-level valuation fields may be unavailable or unsuitable for some ETFs; a holdings look-through
  valuation model is a more robust production extension.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

try:
    import lseg.data as ld
except ImportError:
    ld = None


# ------------------------------ Configuration ------------------------------

@dataclass
class Config:
    capital: float = 250_000.0
    start_date: str = "2016-01-01"
    end_date: Optional[str] = None  # None means latest available
    lookback_days: int = 756        # approx. 3 years of daily returns
    min_history_days: int = 252
    rebalance_months: int = 2
    max_weight: float = 0.30
    min_weight: float = 0.00
    max_turnover: Optional[float] = 0.50  # one-way turnover cap per rebalance; None disables
    transaction_cost_bps: float = 8.0    # illustrative all-in cost per dollar traded
    slippage_bps: float = 3.0             # additional illustrative slippage
    value_blend: float = 0.40             # blend HRP with value-score portfolio
    max_drawdown_gate: Optional[float] = None  # e.g. -0.25; disabled by default
    output_dir: str = "value_etf_output"

    # LSEG field mnemonics are subscription- and instrument-dependent. Confirm in Data Item Browser.
    # Total-return series is preferred to price-only history.
    total_return_field: str = "TR.TotalReturn1D"
    # LSEG total-return fields commonly report percentage points (e.g. 0.25 means 0.25%).
    # Verify the units in Workspace/Data Item Browser before running the backtest.
    return_values_in_percent: bool = True
    # Candidate valuation fields for historical as-of series. Change these to fields confirmed in your
    # Workspace/Data Item Browser. Positive P/E and P/B are cheap when lower; dividend yield is cheap/high
    # when higher; ROE is a quality measure, higher is better.
    valuation_fields: Dict[str, str] = None

    def __post_init__(self):
        if self.valuation_fields is None:
            self.valuation_fields = {
                "pe": "TR.PE",
                "pb": "TR.PriceToBookPerShare",
                "dividend_yield": "TR.DividendYield",
                "roe": "TR.ReturnOnEquity",
            }


CFG = Config()

# Deliberately diversified starting universe. RICs can vary by LSEG symbology/venue.
# Verify each RIC resolves to the intended US-listed ETF in your Workspace.
ETF_UNIVERSE = {
    "VTV":  {"ric": "VTV",  "role": "US large value"},
    "AVUV": {"ric": "AVUV", "role": "US small value / profitability"},
    "VBR":  {"ric": "VBR",  "role": "US small value"},
    "EFV":  {"ric": "EFV",  "role": "Developed ex-US value"},
    "FNDE": {"ric": "FNDE", "role": "Emerging markets fundamental"},
    "SCHD": {"ric": "SCHD", "role": "US dividend / quality"},
    "SCHY": {"ric": "SCHY", "role": "International dividend / quality"},
}

# Illustrative strategic weights, used as a fallback benchmark—not asserted to be optimized.
STARTING_WEIGHTS = pd.Series({
    "VTV": 0.30, "AVUV": 0.15, "VBR": 0.10, "EFV": 0.15,
    "FNDE": 0.10, "SCHD": 0.10, "SCHY": 0.10,
})


# ------------------------------ Utilities ----------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("value_etf")


def ensure_output_dir() -> Path:
    path = Path(CFG.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_lseg_frame(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Normalize common LSEG get_history layouts to Date x instrument values."""
    if raw is None or raw.empty:
        raise ValueError("LSEG returned an empty data frame.")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Common layouts vary by library version; flatten conservatively.
        df.columns = ["|".join(map(str, c)) for c in df.columns]
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    if df.empty:
        raise ValueError("Could not identify dates in LSEG history response.")

    # get_history often returns one column per field for one instrument, and a MultiIndex
    # or instrument/field-labelled columns for multiple instruments. The caller's adapter
    # below handles both the simple and flattened cases.
    return df


def _extract_single_series(raw: pd.DataFrame, ric: str, field: str) -> pd.Series:
    df = normalize_lseg_frame(raw, field)
    candidates = [
        field, field.upper(), ric, f"{ric}|{field}", f"{field}|{ric}",
        f"{ric}|{field.upper()}", f"{field.upper()}|{ric}",
    ]
    for col in candidates:
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = ric
            return pd.to_numeric(s, errors="coerce")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        s = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
        s.name = ric
        return s
    raise ValueError(
        f"Could not uniquely identify {field} for {ric}. Returned columns: {list(df.columns)}"
    )


def lseg_history_one(ric: str, field: str, start: str, end: Optional[str]) -> pd.Series:
    if ld is None:
        raise ImportError("Install the official LSEG Data Library: pip install lseg-data")
    kwargs = dict(universe=ric, fields=[field], interval="daily", start=start)
    if end:
        kwargs["end"] = end
    raw = ld.get_history(**kwargs)
    return _extract_single_series(raw, ric, field)


def fetch_total_return_panel() -> pd.DataFrame:
    """Download daily total-return series; stops rather than silently switching to price-only returns."""
    series = {}
    failures = []
    for ticker, meta in ETF_UNIVERSE.items():
        try:
            LOG.info("Downloading %s (%s)", ticker, meta["ric"])
            series[ticker] = lseg_history_one(
                meta["ric"], CFG.total_return_field, CFG.start_date, CFG.end_date
            )
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            LOG.exception("Failed to retrieve total return for %s", ticker)

    if failures:
        raise RuntimeError(
            "Could not retrieve a complete total-return panel. Verify RICs, field mnemonics, "
            "entitlements and returned column layout in LSEG Workspace. Errors:\n- "
            + "\n- ".join(failures)
        )

    panel = pd.concat(series.values(), axis=1).sort_index()
    panel.columns = list(series.keys())
    panel = panel.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    # This pipeline expects a one-day total-return field, not a total-return index/level.
    # Explicitly convert percent points to decimal returns when configured; do not infer units
    # from the magnitude because that can silently corrupt small daily returns.
    if CFG.return_values_in_percent:
        panel = panel / 100.0
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    panel = panel.dropna(axis=1, thresh=CFG.min_history_days)
    if panel.shape[1] < 4:
        raise RuntimeError(f"Only {panel.shape[1]} ETFs have sufficient history; need at least 4.")
    return panel


def fetch_historical_value_fields() -> Dict[str, pd.DataFrame]:
    """
    Download historical valuation/quality fields. Requires fields with genuine historical
    observations, not merely today's values repeated backwards. If unsupported, replace with
    a licensed point-in-time ETF holdings/valuation dataset.
    """
    out = {}
    for metric, field in CFG.valuation_fields.items():
        by_ticker = {}
        for ticker, meta in ETF_UNIVERSE.items():
            try:
                by_ticker[ticker] = lseg_history_one(
                    meta["ric"], field, CFG.start_date, CFG.end_date
                )
            except Exception as exc:
                LOG.warning("Historical factor unavailable: %s / %s: %s", ticker, field, exc)
        if by_ticker:
            frame = pd.concat(by_ticker.values(), axis=1)
            frame.columns = list(by_ticker.keys())
            out[metric] = frame.sort_index()
    return out


def robust_zscore(row: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(row, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-12:
        z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    else:
        z = (x - med) / (1.4826 * mad)
    z = z.clip(-3, 3)
    return z if higher_is_better else -z


def compute_value_scores(
    asof: pd.Timestamp,
    factor_panel: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> pd.Series:
    """
    Score at date using latest available observation <= asof.
    Valuation metrics: low P/E, low P/B, high dividend yield; quality: high ROE.
    Cross-sectional robust z-scores avoid domination by outliers.
    """
    score_parts = []
    metric_directions = {
        "pe": False, "pb": False, "dividend_yield": True, "roe": True
    }
    metric_weights = {
        "pe": 0.35, "pb": 0.25, "dividend_yield": 0.20, "roe": 0.20
    }
    for metric, panel in factor_panel.items():
        if metric not in metric_weights:
            continue
        eligible_dates = panel.index[panel.index <= asof]
        if len(eligible_dates) == 0:
            continue
        row = panel.loc[eligible_dates[-1]].reindex(tickers)
        # Exclude nonpositive P/E and P/B; negative earnings are not "cheap".
        if metric in ("pe", "pb"):
            row = row.where(row > 0)
        z = robust_zscore(row, metric_directions[metric])
        score_parts.append((metric, z, metric_weights[metric]))

    if not score_parts:
        # Explicit fallback: equal value scores. Never pretend missing valuation fields are real data.
        warnings.warn(
            f"No historical valuation fields available as of {asof.date()}; using equal scores. "
            "This is not a value-factor backtest."
        )
        return pd.Series(1.0, index=tickers)

    weighted = pd.Series(0.0, index=tickers)
    total_weight = pd.Series(0.0, index=tickers)
    for metric, z, weight in score_parts:
        valid = z.notna()
        weighted.loc[valid] += weight * z.loc[valid]
        total_weight.loc[valid] += weight
    score = weighted.div(total_weight.replace(0, np.nan))
    score = score.fillna(score.median() if score.notna().any() else 0.0)
    # Softmax converts scores to positive long-only allocations.
    exp_score = np.exp(score.clip(-4, 4))
    return exp_score / exp_score.sum()


# --------------------------- HRP and constraints ---------------------------

def correlation_distance(returns: pd.DataFrame) -> np.ndarray:
    corr = returns.corr().clip(-1, 1).fillna(0)
    np.fill_diagonal(corr.values, 1.0)
    dist = np.sqrt(np.maximum(0.0, (1.0 - corr.values) / 2.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def get_quasi_diag_order(link: np.ndarray, tickers: List[str]) -> List[str]:
    return [tickers[i] for i in leaves_list(link)]


def cluster_variance(cov: pd.DataFrame, items: List[str]) -> float:
    sub = cov.loc[items, items].values
    inv_diag = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    returns = returns.dropna(axis=1, how="any")
    tickers = list(returns.columns)
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers)
    cov = pd.DataFrame(
        LedoitWolf().fit(returns.values).covariance_,
        index=tickers, columns=tickers
    )
    dist = correlation_distance(returns)
    link = linkage(squareform(dist, checks=False), method="single")
    ordered = get_quasi_diag_order(link, tickers)
    clusters = [ordered]

    while True:
        new_clusters = []
        split_any = False
        for cluster in clusters:
            if len(cluster) <= 1:
                new_clusters.append(cluster)
                continue
            split_any = True
            mid = len(cluster) // 2
            new_clusters.extend([cluster[:mid], cluster[mid:]])
        clusters = new_clusters
        if not split_any:
            break

    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while any(len(c) > 1 for c in clusters):
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                new_clusters.append(cluster)
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            v_left = cluster_variance(cov, left)
            v_right = cluster_variance(cov, right)
            alpha_left = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha_left
            weights[right] *= (1.0 - alpha_left)
            new_clusters.extend([left, right])
        clusters = new_clusters
    weights = weights.reindex(tickers).clip(lower=0)
    return weights / weights.sum()


def project_to_box_simplex(
    weights: pd.Series, lower: float, upper: float, total: float = 1.0
) -> pd.Series:
    """Project arbitrary positive weights to bounds and sum=total by iterative water filling."""
    names = list(weights.index)
    if len(names) * lower > total + 1e-10 or len(names) * upper < total - 1e-10:
        raise ValueError("Infeasible min/max weight constraints for number of ETFs.")
    x = weights.clip(lower=1e-12).values.astype(float)
    x /= x.sum()
    lo, hi = -2.0, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        y = np.clip(x + mid, lower, upper)
        if y.sum() > total:
            hi = mid
        else:
            lo = mid
    y = np.clip(x + (lo + hi) / 2, lower, upper)
    y *= total / y.sum()
    # numerical cleanup while retaining bounds to practical tolerance
    return pd.Series(y, index=names)


def apply_turnover_cap(target: pd.Series, current: pd.Series, max_turnover: Optional[float]) -> pd.Series:
    target = target.reindex(current.index).fillna(0.0)
    target = target / target.sum()
    if max_turnover is None:
        return target
    turnover = 0.5 * (target - current).abs().sum()
    if turnover <= max_turnover or turnover < 1e-12:
        return target
    alpha = max_turnover / turnover
    limited = current + alpha * (target - current)
    return limited / limited.sum()


def make_target_weights(
    returns: pd.DataFrame,
    value_scores: pd.Series,
    current_weights: Optional[pd.Series] = None,
) -> pd.Series:
    hrp = hrp_weights(returns)
    value_scores = value_scores.reindex(hrp.index).fillna(0.0).clip(lower=0)
    if value_scores.sum() <= 0:
        value_scores = pd.Series(1.0, index=hrp.index)
    value_alloc = value_scores / value_scores.sum()
    blend = (1.0 - CFG.value_blend) * hrp + CFG.value_blend * value_alloc
    constrained = project_to_box_simplex(blend, CFG.min_weight, CFG.max_weight)
    if current_weights is not None:
        constrained = apply_turnover_cap(constrained, current_weights, CFG.max_turnover)
    return constrained.sort_index()


# ------------------------------ Backtest -----------------------------------

def rebalance_dates(index: pd.DatetimeIndex, months: int = 2) -> List[pd.Timestamp]:
    monthly = pd.Series(index=index, data=index)
    month_ends = monthly.groupby([index.year, index.month]).last().tolist()
    dates = pd.DatetimeIndex(month_ends)
    dates = dates[(dates >= index.min()) & (dates <= index.max())]
    # Rebalance every Nth month, aligned to first observed month in the backtest.
    return list(dates[::months])


def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def performance_summary(portfolio_returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    r = portfolio_returns.dropna()
    if len(r) < 2:
        return pd.Series(dtype=float)
    wealth = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / (r.std(ddof=1) * np.sqrt(periods_per_year) + 1e-12)
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    sortino = (r.mean() * periods_per_year) / (downside + 1e-12)
    return pd.Series({
        "CAGR": cagr,
        "Annualized volatility": vol,
        "Sharpe (zero RF approximation)": sharpe,
        "Sortino (zero MAR approximation)": sortino,
        "Maximum drawdown": max_drawdown(wealth),
        "Total return": wealth.iloc[-1] - 1.0,
        "Daily observations": len(r),
    })


def walk_forward_backtest(
    returns: pd.DataFrame,
    factor_panel: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Monthly-end rebalance schedule, trading every 2 months.
    Signal uses trailing returns and factor observations <= rebalance date.
    Holdings are applied from the next trading day to avoid same-close look-ahead.
    """
    returns = returns.sort_index().replace([np.inf, -np.inf], np.nan)
    common = [t for t in ETF_UNIVERSE if t in returns.columns]
    returns = returns[common].dropna(how="all")
    dates = rebalance_dates(returns.index, CFG.rebalance_months)
    if len(dates) < 3:
        raise ValueError("Not enough history for a useful walk-forward backtest.")

    holdings = pd.Series(1.0 / len(common), index=common)
    daily_port = pd.Series(index=returns.index, dtype=float)
    weights_log = []
    trades_log = []
    prev_rebalance_idx = 0

    for j, date in enumerate(dates):
        loc = returns.index.get_indexer([date], method="pad")[0]
        if loc < 0:
            continue
        # Establish the period over which these weights are held: next session after rebalance.
        next_date = dates[j + 1] if j + 1 < len(dates) else returns.index[-1]
        next_loc = returns.index.get_indexer([next_date], method="pad")[0]
        start_loc = loc + 1
        if start_loc > next_loc:
            continue

        hist = returns.iloc[max(0, loc - CFG.lookback_days + 1):loc + 1]
        eligible = [t for t in common if hist[t].notna().sum() >= CFG.min_history_days]
        if len(eligible) < 4:
            continue
        hist = hist[eligible].dropna()
        if len(hist) < CFG.min_history_days:
            continue

        scores = compute_value_scores(date, factor_panel, eligible)
        current = holdings.reindex(eligible).fillna(0)
        if current.sum() <= 0:
            current[:] = 1.0 / len(current)
        else:
            current = current / current.sum()
        target = make_target_weights(hist, scores, current_weights=current)
        target = target.reindex(eligible).fillna(0)
        target = target / target.sum()

        # Trading costs charged at rebalance, based on one-way traded notional.
        turnover = 0.5 * (target - current).abs().sum()
        cost_rate = (CFG.transaction_cost_bps + CFG.slippage_bps) / 10_000.0
        cost = turnover * cost_rate
        segment = returns.iloc[start_loc:next_loc + 1][eligible].fillna(0.0)
        daily_segment = segment.dot(target)
        if len(daily_segment):
            daily_segment.iloc[0] -= cost
            daily_port.loc[daily_segment.index] = daily_segment

        weights_log.append(pd.DataFrame({
            "date": date, "ticker": target.index, "weight": target.values
        }))
        trades_log.append({
            "date": date, "one_way_turnover": turnover,
            "estimated_cost_bps_of_portfolio": cost * 10_000,
            "estimated_cost_usd": cost * CFG.capital,
        })
        holdings = pd.Series(0.0, index=common)
        holdings.loc[target.index] = target
        prev_rebalance_idx = loc

    daily_port = daily_port.dropna()
    weights_df = pd.concat(weights_log, ignore_index=True) if weights_log else pd.DataFrame()
    trades_df = pd.DataFrame(trades_log)
    return daily_port, weights_df, trades_df


# ------------------------------- Live target --------------------------------

def live_target(returns: pd.DataFrame, factor_panel: Dict[str, pd.DataFrame]) -> pd.Series:
    asof = returns.index.max()
    hist = returns.loc[returns.index <= asof].tail(CFG.lookback_days)
    eligible = [t for t in hist.columns if hist[t].notna().sum() >= CFG.min_history_days]
    hist = hist[eligible].dropna()
    if len(eligible) < 4 or len(hist) < CFG.min_history_days:
        raise ValueError("Insufficient aligned history for live allocation.")
    scores = compute_value_scores(asof, factor_panel, eligible)
    return make_target_weights(hist, scores)


def main():
    if ld is None:
        raise SystemExit("Missing dependency. Install with: pip install lseg-data")

    out = ensure_output_dir()
    ld.open_session()
    try:
        returns = fetch_total_return_panel()
        factors = fetch_historical_value_fields()

        returns.to_csv(out / "daily_total_returns.csv", index_label="Date")
        for metric, frame in factors.items():
            frame.to_csv(out / f"factor_{metric}.csv", index_label="Date")

        # Backtest results
        port_returns, weights_history, trades = walk_forward_backtest(returns, factors)
        port_returns.to_csv(out / "backtest_daily_returns.csv", header=["portfolio_return"], index_label="Date")
        weights_history.to_csv(out / "backtest_weights.csv", index=False)
        trades.to_csv(out / "backtest_trades_and_costs.csv", index=False)
        summary = performance_summary(port_returns)
        summary.to_csv(out / "backtest_performance_summary.csv", header=["value"])

        # Current model target and dollar allocations
        target = live_target(returns, factors)
        allocation = pd.DataFrame({
            "ticker": target.index,
            "weight": target.values,
            "target_usd": target.values * CFG.capital,
        }).sort_values("weight", ascending=False)
        allocation.to_csv(out / "current_target_allocation.csv", index=False)

        print("\n=== Walk-forward backtest summary ===")
        print(summary.to_string())
        print("\n=== Current target allocation ===")
        print(allocation.to_string(index=False, formatters={
            "weight": "{:.2%}".format, "target_usd": "${:,.0f}".format
        }))
        print(f"\nOutputs written to: {out.resolve()}")
        print("\nReview field units and factor timestamps before relying on results.")
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
