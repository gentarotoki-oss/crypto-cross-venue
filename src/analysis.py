"""
Cross-venue analysis of equity perpetuals: price gaps between venues,
how fast they close, which venue leads, and the perp's premium over the stock.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "data" / "trades"
STOCK = ROOT / "data" / "stock"

VENUES = ("binance", "bybit", "okx")
PAIRS = list(combinations(VENUES, 2))

# OKX daily files are cut on UTC+8 days, so this is the window all venues cover.
START = pd.Timestamp("2026-10-13 00:00", tz="UTC")
END = pd.Timestamp("2026-10-26 16:00", tz="UTC")

SESSIONS = ("regular", "extended", "overnight", "weekend")


def load_trades(venue: str, symbol: str) -> pd.DataFrame:
    files = sorted((TRADES / venue / symbol).glob("*.parquet"))
    df = pd.concat(pd.read_parquet(f) for f in files)
    return df[(df.ts >= START) & (df.ts < END)].reset_index(drop=True)


def last_price_grid(symbol: str, freq: str = "1s", max_age: str = "10s") -> pd.DataFrame:
    """
    Last trade price of each venue on a regular time grid.

    A price older than `max_age` is treated as missing, so quiet periods on one
    venue do not show up as fake gaps.
    """
    grid = pd.date_range(START, END, freq=freq, inclusive="left")
    cols = {}
    for v in VENUES:
        t = load_trades(v, symbol)
        last = t.groupby(t.ts.dt.floor(freq)).agg(price=("price", "last"), ts=("ts", "last"))
        last = last.reindex(grid).ffill()
        age = grid.to_series(index=grid) - last["ts"]
        cols[v] = last["price"].where(age <= pd.Timedelta(max_age))
    return pd.DataFrame(cols, index=grid)


def session_of(index: pd.DatetimeIndex) -> pd.Series:
    """US equity session for each timestamp (New York time)."""
    et = index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    weekday = et.dayofweek < 5
    regular = weekday & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    extended = weekday & ~regular & (minutes >= 4 * 60) & (minutes < 20 * 60)
    # Friday 20:00 to Monday 04:00 ET counts as weekend
    weekend = (et.dayofweek >= 5) | ((et.dayofweek == 4) & (minutes >= 20 * 60)) | (
        (et.dayofweek == 0) & (minutes < 4 * 60)
    )
    s = np.select([regular, extended, weekend], ["regular", "extended", "weekend"], "overnight")
    return pd.Series(s, index=index)


def gaps_bps(prices: pd.DataFrame) -> pd.DataFrame:
    """Log price difference between each pair of venues, in basis points."""
    logp = np.log(prices)
    return pd.DataFrame({f"{a}-{b}": 1e4 * (logp[a] - logp[b]) for a, b in PAIRS})


def gap_stats(gaps: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    rows = []
    for col in gaps:
        for name in SESSIONS:
            g = gaps.loc[sessions == name, col].dropna()
            if len(g) < 100:
                continue
            a = g.abs()
            rows.append({
                "pair": col,
                "session": name,
                "hours": len(g) / 3600,
                "mean": g.mean(),
                "std": g.std(),
                "median abs gap": a.median(),
                "p99 abs gap": a.quantile(0.99),
                "share abs gap > 10": (a > 10).mean(),
                "half-life (s)": half_life(g),
            })
    return pd.DataFrame(rows)


def half_life(gap: pd.Series) -> float:
    """
    Half-life of the gap from an AR(1) fit on consecutive 1s observations:
    g_{t+1} - μ = φ (g_t - μ) + ε,  half-life = -ln 2 / ln φ.
    """
    x = gap - gap.mean()
    prev, curr = x.shift(1), x
    ok = prev.notna() & curr.notna()
    phi = (prev[ok] * curr[ok]).sum() / (prev[ok] ** 2).sum()
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("nan")


def lead_lag(symbol: str, freq: str = "200ms", max_lag: int = 10, session: str | None = "regular") -> pd.DataFrame:
    """
    Cross-correlation of returns between venues at lags of -max_lag..max_lag steps.

    corr(r_a(t), r_b(t + k)) peaking at k > 0 means a's moves show up on b
    k steps later, i.e. a leads b.
    """
    prices = last_price_grid(symbol, freq=freq, max_age="5s")
    if session is not None:
        prices = prices[session_of(prices.index) == session]
    rets = np.log(prices).diff()
    rows = []
    for a, b in PAIRS:
        for k in range(-max_lag, max_lag + 1):
            rows.append({"pair": f"{a}->{b}", "lag": k, "corr": rets[a].corr(rets[b].shift(-k))})
    return pd.DataFrame(rows)


def load_stock(symbol: str) -> pd.Series:
    s = pd.read_parquet(STOCK / f"{symbol}.parquet")["close"]
    return s[(s.index >= START) & (s.index < END)]


def load_usdt_usd() -> pd.Series:
    return pd.read_parquet(ROOT / "data" / "fx" / "usdt_usd.parquet")["usdt_usd"]


def perp_premium(symbol: str, venue: str = "binance") -> pd.DataFrame:
    """
    Perp price vs. the stock's last traded price, on 1-minute bars.

    The perp is quoted in USDT and the stock in USD, so the perp is converted
    to USD with the hourly USDT/USD rate before comparing.

    Outside trading hours (including Yahoo's pre/post-market) the stock price
    is the last print, so the premium measures how far the perp has moved
    since the stock last traded.
    """
    t = load_trades(venue, symbol)
    perp = t.groupby(t.ts.dt.floor("1min"))["price"].last()
    grid = pd.date_range(START, END, freq="1min", inclusive="left")
    perp = perp.reindex(grid).ffill()
    stock = load_stock(symbol).reindex(grid).ffill()
    usdt_usd = load_usdt_usd().reindex(grid, method="ffill")
    df = pd.DataFrame({"perp": perp, "stock": stock, "usdt_usd": usdt_usd})
    df["premium_usdt_bps"] = 1e4 * np.log(df.perp / df.stock)
    df["premium_bps"] = 1e4 * np.log(df.perp * df.usdt_usd / df.stock)
    df["session"] = session_of(df.index)
    return df


def overnight_gaps(symbol: str, venue: str = "binance") -> pd.DataFrame:
    """
    For each break between extended-hours sessions: how much the perp moved
    while the stock was closed, and how much the stock gapped when it reopened.
    """
    df = perp_premium(symbol, venue)
    stock_bars = load_stock(symbol)
    # a break is a pause of more than 30 minutes in the stock's 1-minute bars
    gaps = stock_bars.index.to_series().diff() > pd.Timedelta("30min")
    rows = []
    for reopen in stock_bars.index[gaps]:
        close_t = stock_bars.index[stock_bars.index < reopen][-1]
        perp_close = df.perp.asof(close_t)
        perp_reopen = df.perp.asof(reopen - pd.Timedelta("1min"))
        rows.append({
            "close": close_t,
            "reopen": reopen,
            "perp move (bps)": 1e4 * np.log(perp_reopen / perp_close),
            "stock gap (bps)": 1e4 * np.log(stock_bars.loc[reopen] / stock_bars.loc[close_t]),
        })
    return pd.DataFrame(rows)


def backtest(
    prices: pd.DataFrame,
    a: str,
    b: str,
    entry_bps: float,
    exit_bps: float = 0.0,
    fee_bps: float = 5.0,
    window: str = "30min",
    delay: int = 1,
) -> pd.DataFrame:
    """
    Trade the gap between venues a and b around its rolling median.

    - deviation d = gap - rolling median of the gap over `window` (removes
      slow, persistent offsets such as different funding rates)
    - enter when |d| > entry_bps: sell the rich venue, buy the cheap one
    - exit when d crosses back within exit_bps of zero
    - fills happen `delay` seconds after the signal, at the last trade price
    - each fill pays `fee_bps` (taker), so a round trip costs 4 × fee_bps

    Returns one row per trade.
    """
    gap = 1e4 * np.log(prices[a] / prices[b])
    dev = gap - gap.rolling(window, min_periods=600).median()
    fill_gap = gap.shift(-delay)

    trades = []
    pos = 0  # +1: long a / short b (entered when a was cheap)
    entry_gap = entry_t = None
    for t, d, g in zip(dev.index, dev.to_numpy(), fill_gap.to_numpy()):
        if np.isnan(d) or np.isnan(g):
            continue
        if pos == 0:
            if d > entry_bps:
                pos, entry_gap, entry_t = -1, g, t
            elif d < -entry_bps:
                pos, entry_gap, entry_t = 1, g, t
        elif (pos == -1 and d <= exit_bps) or (pos == 1 and d >= -exit_bps):
            gross = pos * (g - entry_gap)
            trades.append({
                "entry": entry_t,
                "exit": t,
                "side": "short a" if pos == -1 else "long a",
                "gross_bps": gross,
                "net_bps": gross - 4 * fee_bps,
            })
            pos = 0
    return pd.DataFrame(trades)
