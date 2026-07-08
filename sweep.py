"""
sweep.py
Stage 3b of the Nixie's SMC backtester: detect when an inducement is SWEPT.

Sweep rule (Tony, via you): price trades BEYOND the marked inducement.
  - bullish inducement (a swing LOW) is swept when a later candle's LOW < level
  - bearish inducement (a swing HIGH) is swept when a later candle's HIGH > level
Wick OR body counts, so we test the candle's Low/High (which include wicks).

The sweep is the green light: only AFTER it are we allowed to look for an entry
at the POI. Inducements that are never swept (within max_wait) produce no trade.

Run it with:  uv run sweep.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inducement import find_inducements


def find_sweeps(df: pd.DataFrame, inducements: pd.DataFrame, max_wait: int = 96) -> pd.DataFrame:
    """
    For each inducement, look forward up to `max_wait` candles for the first
    candle that pierces it. Adds columns: swept (bool), sweep_time.
    max_wait = 96 M15 candles = 24 hours (a generous but bounded window).
    """
    lows = df["Low"].values
    highs = df["High"].values
    n = len(df)

    swept, sweep_time = [], []
    for _, r in inducements.iterrows():
        pos = df.index.get_loc(r["ind_time"])   # position of the inducement candle
        level = r["ind_price"]
        bull = r["direction"] == "bull"

        found, when = False, pd.NaT
        for k in range(pos + 1, min(pos + 1 + max_wait, n)):
            pierced = (lows[k] < level) if bull else (highs[k] > level)
            if pierced:
                found, when = True, df.index[k]
                break
        swept.append(found)
        sweep_time.append(when)

    out = inducements.copy()
    out["swept"] = swept
    out["sweep_time"] = sweep_time
    return out


if __name__ == "__main__":
    N = 5
    df = pd.read_parquet("xauusd_m15_clean.parquet")
    df, ind = find_inducements(df, n=N)

    valids = ind[ind["valid"]].reset_index(drop=True)
    valids = find_sweeps(df, valids, max_wait=96)

    n_swept = valids["swept"].sum()
    print(f"Valid inducements        : {len(valids):,}")
    print(f"  swept within 24h       : {n_swept:,}  ({100*n_swept/len(valids):.0f}%)")
    print(f"  never swept (no trade) : {(~valids['swept']).sum():,}")

    print("\nLast 8 valid inducements and their sweep status:")
    for _, r in valids.tail(8).iterrows():
        if r["swept"]:
            print(f"   {r['ind_time']}  {r['direction']:4s}  IDM @ {r['ind_price']:.2f}  "
                  f"-> SWEPT at {r['sweep_time']}")
        else:
            print(f"   {r['ind_time']}  {r['direction']:4s}  IDM @ {r['ind_price']:.2f}  -> not swept")

    # --- Plot: last 400 candles, inducements + where they got swept --------
    window = df.iloc[-400:]
    t0 = window.index[0]
    vv = valids[valids["ind_time"] >= t0]

    plt.figure(figsize=(15, 7))
    plt.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    for _, r in vv.iterrows():
        plt.scatter([r["ind_time"]], [r["ind_price"]], color="purple", s=45, zorder=5)
        plt.text(r["ind_time"], r["ind_price"], "  IDM", color="purple", fontsize=7, va="center")
        if r["swept"] and pd.notna(r["sweep_time"]) and r["sweep_time"] >= t0:
            # dashed line at the inducement level from IDM to the sweep, + an X at the sweep
            plt.hlines(r["ind_price"], xmin=r["ind_time"], xmax=r["sweep_time"],
                       color="orange", linestyle="--", linewidth=1)
            plt.scatter([r["sweep_time"]], [r["ind_price"]], color="orange", marker="x", s=60, zorder=6)
    plt.title(f"XAUUSD M15 - inducements & sweeps (n={N}), last 400 candles")
    plt.ylabel("Price (USD)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("sweep_check.png", dpi=120)
    print("\nSaved sweep_check.png - purple dot = inducement, orange X = where it got swept.")