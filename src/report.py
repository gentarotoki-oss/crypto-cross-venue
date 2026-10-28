"""
Builds the figures and tables in reports/ from the downloaded data.

  python src/report.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    PAIRS,
    ROOT,
    backtest,
    gap_stats,
    gaps_bps,
    last_price_grid,
    lead_lag,
    overnight_gaps,
    perp_premium,
    session_of,
)

SYMBOLS = ("TSLA", "MSTR", "QQQ")
OUT = ROOT / "reports"
FIG = OUT / "figures"


def fmt(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tables: list[str] = []

    # 1. gaps between venues
    prices = {s: last_price_grid(s) for s in SYMBOLS}
    stats = pd.concat(
        gap_stats(gaps_bps(p), session_of(p.index)).assign(symbol=s) for s, p in prices.items()
    )
    cols = ["symbol", "pair", "session", "hours", "mean", "std", "median abs gap", "p99 abs gap", "half-life (s)"]
    tables.append("## Gaps between venues (bps, 1s last trade)\n\n" + fmt(stats[cols], 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    g = gaps_bps(prices["TSLA"])["binance-okx"]
    sess = session_of(g.index)
    for name in ("regular", "weekend"):
        ax.hist(g[sess == name].dropna(), bins=np.arange(-10, 10.25, 0.25), alpha=0.6, density=True, label=name)
    ax.set(xlabel="Binance − OKX (bps)", ylabel="density", title="TSLA perp: gap between Binance and OKX")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "gap_hist_tsla.png", dpi=120)
    plt.close(fig)

    # 2. lead-lag
    rows = []
    fig, axes = plt.subplots(1, len(SYMBOLS), figsize=(12, 3.5), sharey=True)
    for ax, s in zip(axes, SYMBOLS):
        ll = lead_lag(s, freq="200ms", max_lag=10, session="regular")
        for pair, d in ll.groupby("pair"):
            d = d.set_index("lag")["corr"]
            ax.plot(d.index * 0.2, d.values, marker=".", label=pair)
            rows.append({
                "symbol": s,
                "pair": pair,
                "corr at 0": d.loc[0],
                "sum corr, a leads (+0.2..1s)": d.loc[1:5].sum(),
                "sum corr, b leads (−1..−0.2s)": d.loc[-5:-1].sum(),
            })
        ax.set(title=s, xlabel="lag (s)")
        ax.axvline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("corr(r_a(t), r_b(t + lag))")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-correlation of 200ms returns, US regular hours")
    fig.tight_layout()
    fig.savefig(FIG / "lead_lag.png", dpi=120)
    plt.close(fig)
    tables.append("## Lead-lag (200ms returns, regular hours)\n\n" + fmt(pd.DataFrame(rows), 3))

    # 3. perp vs stock
    rows = []
    for s in SYMBOLS:
        df = perp_premium(s).dropna()
        for name, d in df.groupby("session"):
            rows.append({
                "symbol": s,
                "session": name,
                "premium in USDT (bps)": d.premium_usdt_bps.mean(),
                "premium in USD (bps)": d.premium_bps.mean(),
                "std (bps)": d.premium_bps.std(),
            })
    tables.append("## Binance perp vs. stock (1-minute)\n\n" + fmt(pd.DataFrame(rows), 1))

    df = perp_premium("TSLA")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df.index, df.premium_bps, lw=0.6)
    closed = df.session.isin(["overnight", "weekend"])
    ax.fill_between(df.index, *ax.get_ylim(), where=closed, color="grey", alpha=0.15, label="stock closed")
    ax.set(ylabel="bps", title="TSLA: Binance perp (in USD) vs. last stock price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "premium_tsla.png", dpi=120)
    plt.close(fig)

    gaps = pd.concat(overnight_gaps(s).assign(symbol=s) for s in SYMBOLS)
    fig, ax = plt.subplots(figsize=(5, 5))
    for s, d in gaps.groupby("symbol"):
        ax.scatter(d["perp move (bps)"], d["stock gap (bps)"], label=s)
    lim = gaps[["perp move (bps)", "stock gap (bps)"]].abs().to_numpy().max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.5)
    ax.set(xlabel="perp move while stock closed (bps)", ylabel="stock gap at reopen (bps)",
           title="Overnight and weekend breaks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "overnight_gap.png", dpi=120)
    plt.close(fig)
    summary = gaps.groupby("symbol").apply(
        lambda d: pd.Series({
            "breaks": len(d),
            "corr": d["perp move (bps)"].corr(d["stock gap (bps)"]),
            "slope": np.polyfit(d["perp move (bps)"], d["stock gap (bps)"], 1)[0],
        }),
        include_groups=False,
    ).reset_index()
    tables.append("## Perp move while closed vs. stock gap at reopen\n\n" + fmt(summary, 3))

    # 4. backtest
    rows = []
    for s, p in prices.items():
        for a, b in PAIRS:
            for entry in (3, 5, 8):
                for delay in (0, 1):
                    t = backtest(p, a, b, entry_bps=entry, delay=delay)
                    if t.empty:
                        continue
                    rows.append({
                        "symbol": s,
                        "pair": f"{a}-{b}",
                        "entry (bps)": entry,
                        "delay (s)": delay,
                        "trades": len(t),
                        "gross (bps/trade)": t.gross_bps.mean(),
                        "net (bps/trade)": t.net_bps.mean(),
                        "median hold (s)": (t.exit - t.entry).dt.total_seconds().median(),
                    })
    bt = pd.DataFrame(rows)
    tables.append("## Backtest: trading the gap (taker 5 bps per fill)\n\n" + fmt(bt, 2))

    fig, ax = plt.subplots(figsize=(6, 4))
    agg = bt.groupby(["entry (bps)", "delay (s)"])["gross (bps/trade)"].mean().unstack()
    for delay in agg.columns:
        ax.plot(agg.index, agg[delay], marker="o", label=f"fill after {delay}s")
    ax.axhline(20, color="red", ls="--", lw=1, label="round-trip taker fees (4 × 5 bps)")
    ax.set(xlabel="entry threshold (bps)", ylabel="gross profit per trade (bps)",
           title="Gross edge vs. cost (mean over symbols and pairs)", ylim=(0, 22))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "backtest_edge.png", dpi=120)
    plt.close(fig)

    (OUT / "tables.md").write_text("# Tables\n\nGenerated by `src/report.py`.\n\n" + "\n\n".join(tables) + "\n",
                                   encoding="utf-8")
    print(f"wrote {OUT / 'tables.md'} and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
