"""
breaks.py
Stage 2b of the Nixie's SMC backtester: label each structure break as
BOS (continuation) or MSS (reversal), and record the LEVEL that was broken
so we can draw it as a proper horizontal line.

Run it with:  uv run breaks.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from structure import find_swings


def label_structure(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Adds columns:
        structure_event -> "", "BOS_up", "MSS_up", "BOS_down", "MSS_down"
        trend           -> "", "up", "down"   (trend AFTER this candle)
        break_level     -> the PRICE that was broken (NaN if no event)
        break_from      -> timestamp of the swing that created that level (NaT if none)
    """
    df = find_swings(df, n=n)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sh = df["swing_high"].values
    sl = df["swing_low"].values
    times = df.index
    total = len(df)

    event = [""] * total
    trend_col = [""] * total
    break_level = [np.nan] * total
    break_from = [pd.NaT] * total

    trend = None
    ref_high = None; ref_high_time = None   # last confirmed, unbroken swing HIGH (+ when it formed)
    ref_low = None;  ref_low_time = None    # last confirmed, unbroken swing LOW

    for i in range(total):
        # LOOKAHEAD GUARD: only "learn" the swing that happened n candles ago.
        j = i - n
        if j >= 0:
            if sh[j]:
                ref_high = highs[j]; ref_high_time = times[j]
            if sl[j]:
                ref_low = lows[j];  ref_low_time = times[j]

        # Break check on this candle's CLOSE.
        if ref_high is not None and closes[i] > ref_high:
            event[i] = "BOS_up" if trend == "up" else "MSS_up"
            break_level[i] = ref_high
            break_from[i] = ref_high_time
            trend = "up"
            ref_high = None; ref_high_time = None
        elif ref_low is not None and closes[i] < ref_low:
            event[i] = "BOS_down" if trend == "down" else "MSS_down"
            break_level[i] = ref_low
            break_from[i] = ref_low_time
            trend = "down"
            ref_low = None; ref_low_time = None

        trend_col[i] = trend if trend else ""

    df["structure_event"] = event
    df["trend"] = trend_col
    df["break_level"] = break_level
    df["break_from"] = break_from
    return df


if __name__ == "__main__":
    N = 5   # <-- structure sensitivity. Try 5, then 8, then 12, and watch the trend settle down.

    df = pd.read_parquet("xauusd_m15_clean.parquet")
    df = label_structure(df, n=N)

    counts = df["structure_event"].value_counts()
    print(f"Structure events across the whole dataset (n={N}):")
    for name in ["BOS_up", "MSS_up", "BOS_down", "MSS_down"]:
        print(f"   {name:9s}: {counts.get(name, 0):,}")
    total_events = sum(counts.get(x, 0) for x in ["BOS_up", "MSS_up", "BOS_down", "MSS_down"])
    print(f"   {'TOTAL':9s}: {total_events:,}  (~1 every {len(df)/max(total_events,1):.0f} candles)")

    # How often does the trend actually FLIP? (MSS events = flips). Fewer flips = cleaner structure.
    n_flips = counts.get("MSS_up", 0) + counts.get("MSS_down", 0)
    print(f"   Trend flips (MSS): {n_flips:,}  -> lower is calmer/more tradeable")

    print("\nLast 12 structure events:")
    recent = df[df["structure_event"] != ""].tail(12)
    for ts, row in recent.iterrows():
        print(f"   {ts}   {row['structure_event']:9s} @ close {row['Close']:.2f}")

    # --- Validation plot: HORIZONTAL level-lines from swing to break --------
    window = df.iloc[-400:]
    events = window[window["structure_event"] != ""]
    colors = {"BOS_up": "green", "MSS_up": "blue", "BOS_down": "red", "MSS_down": "orange"}

    plt.figure(figsize=(15, 7))
    plt.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    for ts, row in events.iterrows():
        ev = row["structure_event"]
        level = row["break_level"]
        origin = row["break_from"]
        # draw the broken level as a horizontal line from where it formed to where it broke
        start = origin if pd.notna(origin) and origin >= window.index[0] else window.index[0]
        plt.hlines(level, xmin=start, xmax=ts, color=colors[ev], linewidth=1.4)
        plt.text(ts, level, "  " + ev.replace("_", " "), fontsize=6,
                 color=colors[ev], va="center", ha="left")
    plt.title(f"XAUUSD M15 - BOS/MSS broken levels (n={N}), last 400 candles")
    plt.ylabel("Price (USD)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("structure_check.png", dpi=120)
    print("\nSaved structure_check.png - horizontal lines now sit AT the broken price level.")