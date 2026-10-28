"""
Download daily trade files for equity perpetuals from Binance, Bybit and OKX,
and normalize them to one parquet file per (venue, symbol, day).

Columns: ts (UTC, ns), price, size, side ("buy" / "sell", the taker side).

Sources (all public, no API key):
  Binance  https://data.binance.vision  USD-M futures aggTrades
  Bybit    https://public.bybit.com     linear trades
  OKX      https://static.okx.com       swap trades

Usage:
  python src/fetch.py --symbols TSLA MSTR QQQ --start 2026-10-13 --end 2026-10-26
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TRADES = ROOT / "data" / "trades"

VENUES = ("binance", "bybit", "okx")


def url(venue: str, symbol: str, day: date) -> str:
    d = day.isoformat()
    if venue == "binance":
        return f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}USDT/{symbol}USDT-aggTrades-{d}.zip"
    if venue == "bybit":
        return f"https://public.bybit.com/trading/{symbol}USDT/{symbol}USDT{d}.csv.gz"
    if venue == "okx":
        return f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{day:%Y%m%d}/{symbol}-USDT-SWAP-trades-{d}.zip"
    raise ValueError(venue)


def read_zip_csv(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f)


def normalize(venue: str, blob: bytes) -> pd.DataFrame:
    if venue == "binance":
        df = read_zip_csv(blob)
        out = pd.DataFrame({
            "ts": pd.to_datetime(df["transact_time"], unit="ms", utc=True),
            "price": df["price"],
            "size": df["quantity"],
            # is_buyer_maker = True means the taker sold
            "side": df["is_buyer_maker"].map({True: "sell", False: "buy"}),
        })
    elif venue == "bybit":
        df = pd.read_csv(io.BytesIO(blob), compression="gzip")
        out = pd.DataFrame({
            "ts": pd.to_datetime(df["timestamp"] * 1e6, unit="us", utc=True),
            "price": df["price"],
            "size": df["size"],
            "side": df["side"].str.lower(),
        })
    elif venue == "okx":
        df = read_zip_csv(blob)
        out = pd.DataFrame({
            "ts": pd.to_datetime(df["created_time"], unit="ms", utc=True),
            "price": df["price"],
            "size": df["size"],
            "side": df["side"].str.lower(),
        })
    else:
        raise ValueError(venue)
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def fetch(venue: str, symbol: str, day: date) -> Path | None:
    dest = TRADES / venue / symbol / f"{day.isoformat()}.parquet"
    if dest.exists():
        return dest
    try:
        with urllib.request.urlopen(url(venue, symbol, day), timeout=120) as r:
            blob = r.read()
    except Exception as e:  # missing day or network error: report and move on
        print(f"  skip {venue} {symbol} {day}: {e}")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    normalize(venue, blob).to_parquet(dest, index=False)
    return dest


def fetch_usdt_usd(start: date, end: date) -> Path:
    """Hourly USDT/USD closes from Kraken. Perps are quoted in USDT, stocks in USD."""
    import json

    since = int(pd.Timestamp(start, tz="UTC").timestamp())
    with urllib.request.urlopen(
        f"https://api.kraken.com/0/public/OHLC?pair=USDTZUSD&interval=60&since={since}", timeout=60
    ) as r:
        result = json.load(r)["result"]
    rows = next(v for k, v in result.items() if k != "last")
    s = pd.Series(
        [float(row[4]) for row in rows],
        index=pd.to_datetime([row[0] for row in rows], unit="s", utc=True),
        name="usdt_usd",
    )
    s = s[s.index < pd.Timestamp(end + timedelta(days=1), tz="UTC")]
    dest = ROOT / "data" / "fx" / "usdt_usd.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(dest)
    return dest


def fetch_stock(symbol: str, start: date, end: date) -> Path:
    """1-minute stock bars from Yahoo Finance, including pre- and after-market."""
    import yfinance as yf

    parts = []
    a = start
    while a <= end:  # Yahoo serves at most 7 days of 1-minute bars per request
        b = min(a + timedelta(days=7), end + timedelta(days=1))
        parts.append(yf.download(symbol, start=a, end=b, interval="1m", prepost=True,
                                 progress=False, auto_adjust=False))
        a = b
    df = pd.concat(parts)
    df.columns = [c[0].lower() for c in df.columns]
    df.index = df.index.tz_convert("UTC")
    dest = ROOT / "data" / "stock" / f"{symbol}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    args = p.parse_args()

    print(f"usdt/usd  {fetch_usdt_usd(args.start, args.end)}")
    for symbol in args.symbols:
        print(f"stock {symbol}  {fetch_stock(symbol, args.start, args.end)}")

    day = args.start
    while day <= args.end:
        for symbol in args.symbols:
            for venue in VENUES:
                path = fetch(venue, symbol, day)
                if path:
                    print(f"{venue:8} {symbol:5} {day}  {path.stat().st_size / 1e6:6.1f} MB")
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
