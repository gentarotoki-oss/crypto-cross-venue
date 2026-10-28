"""Unit tests for the analysis functions, on synthetic data (no downloads needed)."""

import numpy as np
import pandas as pd
import pytest

from analysis import backtest, gaps_bps, half_life, session_of


def ny(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps).tz_localize("America/New_York").tz_convert("UTC")


def test_sessions_follow_new_york_hours():
    idx = ny(
        "2026-10-20 10:00",  # Tue, regular
        "2026-10-20 09:29",  # Tue, pre-market
        "2026-10-20 16:00",  # Tue, after-market
        "2026-10-20 22:00",  # Tue, overnight
        "2026-10-23 21:00",  # Fri night, weekend
        "2026-10-24 12:00",  # Sat
        "2026-10-26 03:00",  # Mon before pre-market, still weekend
        "2026-10-26 04:00",  # Mon pre-market
    )
    assert session_of(idx).tolist() == [
        "regular", "extended", "extended", "overnight",
        "weekend", "weekend", "weekend", "extended",
    ]


def test_gap_is_log_difference_in_bps():
    prices = pd.DataFrame({"binance": [100.0], "bybit": [100.1], "okx": [100.0]})
    gaps = gaps_bps(prices)
    assert gaps["binance-bybit"].iloc[0] == pytest.approx(1e4 * np.log(100 / 100.1))
    assert gaps["binance-okx"].iloc[0] == 0.0


def test_half_life_of_ar1():
    # φ = 0.9 per step -> half-life = ln 2 / -ln 0.9 ≈ 6.58 steps
    rng = np.random.default_rng(0)
    x = np.zeros(200_000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    assert half_life(pd.Series(x)) == pytest.approx(np.log(2) / -np.log(0.9), rel=0.05)


def flat_prices(n: int = 2000) -> pd.DataFrame:
    idx = pd.date_range("2026-10-20", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"a": 100.0, "b": 100.0}, index=idx)


def test_backtest_captures_a_persistent_gap_net_of_fees():
    prices = flat_prices()
    prices.iloc[1000:1010, 0] = 100.1  # a is ~10 bps rich for 10 seconds

    trades = backtest(prices, "a", "b", entry_bps=8, fee_bps=5, delay=1)

    assert len(trades) == 1
    assert trades.side.iloc[0] == "short a"
    assert trades.gross_bps.iloc[0] == pytest.approx(1e4 * np.log(100.1 / 100))
    assert trades.net_bps.iloc[0] == pytest.approx(trades.gross_bps.iloc[0] - 20)


def test_backtest_misses_a_one_second_gap_with_one_second_delay():
    prices = flat_prices()
    prices.iloc[1000, 0] = 100.1  # gone after one second

    instant = backtest(prices, "a", "b", entry_bps=8, delay=0)
    delayed = backtest(prices, "a", "b", entry_bps=8, delay=1)

    assert instant.gross_bps.iloc[0] > 9
    assert delayed.gross_bps.iloc[0] == pytest.approx(0)
