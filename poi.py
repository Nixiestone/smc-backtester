"""
poi.py
Stage 4a of the Nixie's SMC backtester: find & mark the ORDER BLOCK.

Order block (Tony) = the last OPPOSING candle before the move that broke
structure. For a bullish setup that's the last DOWN candle before the up-move
(sits at the impulse origin); for bearish, the last UP candle before the drop.

Marking rule (yours):
  - wick shorter than body (or none)  -> mark FULL candle (High..Low)
  - a wick longer than the body       -> mark BODY ONLY (Open..Close)
  (we compare the LARGER wick to the body -- confirm this matches Tony.)

Run it with:  uv run poi.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from inducement import find_inducements
from sweep import find_sweeps


def find_order_blocks(df: pd.DataFrame, inducements: pd.DataFrame, lookback: int = 15) -> pd.DataFrame:
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    ob_time, ob_top, ob_bottom, ob_mode = [], [], [], []
    for _, r in inducements.iterrows():
        o_pos = df.index.get_loc(r["origin_time"])
        bull = r["direction"] == "bull"

        # walk BACK from the origin to the last opposing candle
        found = None
        for k in range(o_pos, max(o_pos - lookback, -1), -1):
            if bull and closes[k] < opens[k]:      # last DOWN candle (bullish OB)
                found = k; break
            if (not bull) and closes[k] > opens[k]:  # last UP candle (bearish OB)
                found = k; break

        if found is None:
            ob_time.append(pd.NaT); ob_top.append(np.nan)
            ob_bottom.append(np.nan); ob_mode.append("")
            continue

        k = found
        body_top = max(opens[k], closes[k])
        body_bot = min(opens[k], closes[k])
        body = body_top - body_bot
        upper_wick = highs[k] - body_top
        lower_wick = body_bot - lows[k]

        if max(upper_wick, lower_wick) > body:      # a long wick -> body only
            top, bot, mode = body_top, body_bot, "body"
        else:                                       # tidy candle -> full range
            top, bot, mode = highs[k], lows[k], "full"

        ob_time.append(df.index[k]); ob_top.append(top)
        ob_bottom.append(bot); ob_mode.append(mode)

    out = inducements.copy()
    out["ob_time"] = ob_time
    out["ob_top"] = ob_top
    out["ob_bottom"] = ob_bottom
    out["ob_mode"] = ob_mode
    return out


if __name__ == "__main__":
    N = 5
    df = pd.read_parquet("xauusd_m15_clean.parquet")
    df, ind = find_inducements(df, n=N)
    valids = ind[ind["valid"]].reset_index(drop=True)
    valids = find_sweeps(df, valids, max_wait=96)
    valids = find_order_blocks(df, valids)

    setups = valids[valids["swept"] & valids["ob_time"].notna()]
    print(f"Swept inducements with an order block behind them: {len(setups):,}")
    print(f"  marked FULL candle : {(setups['ob_mode']=='full').sum():,}")
    print(f"  marked BODY only   : {(setups['ob_mode']=='body').sum():,}")

    print("\nLast 6 setups (bullish OB should sit BELOW its inducement):")
    for _, r in setups.tail(6).iterrows():
        print(f"   {r['direction']:4s}  OB {r['ob_bottom']:.2f}-{r['ob_top']:.2f} ({r['ob_mode']})"
              f"   IDM @ {r['ind_price']:.2f}")

    # --- Plot: last 400 candles with OB boxes drawn -----------------------
    window = df.iloc[-400:]
    t0 = window.index[0]
    sc = setups[setups["ind_time"] >= t0]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    for _, r in sc.iterrows():
        color = "green" if r["direction"] == "bull" else "red"
        x0 = mdates.date2num(r["ob_time"])
        x1 = mdates.date2num(r["ind_time"]) + 0.15   # extend box right, past the inducement
        ax.add_patch(Rectangle((x0, r["ob_bottom"]), x1 - x0, r["ob_top"] - r["ob_bottom"],
                               color=color, alpha=0.25, zorder=2))
        ax.scatter([r["ind_time"]], [r["ind_price"]], color="purple", s=30, zorder=5)
    ax.set_title(f"XAUUSD M15 - order blocks (green=bull, red=bear), last 400 candles")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("poi_check.png", dpi=120)
    print("\nSaved poi_check.png - shaded boxes are the order blocks; purple dot = its inducement.")