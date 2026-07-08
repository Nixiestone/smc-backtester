"""
structure.py
Stage 2a of the Nixie's SMC backtester: find swing highs and swing lows.

A swing high = a candle whose HIGH is strictly greater than the highs of the
`n` candles on its left AND the `n` candles on its right (a local peak).
A swing low  = a candle whose LOW is strictly less than the lows of the
`n` candles on each side (a local trough).

`n` is the one big knob: bigger n = fewer, more significant swings.

Run it with:  uv run structure.py
"""

import pandas as pd
import matplotlib.pyplot as plt


def find_swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Adds two boolean columns to df:
        swing_high  -> True on candles that are a local peak
        swing_low   -> True on candles that are a local trough
    A candle needs `n` candles on EACH side to be judged, so the first `n`
    and last `n` candles can never be swings (not enough neighbours).
    """
    highs = df["High"].values          # pull the raw numbers out as fast numpy arrays
    lows = df["Low"].values
    total = len(df)

    # Start every candle as "not a swing", then flip the ones that qualify.
    is_swing_high = [False] * total
    is_swing_low = [False] * total

    # We can only judge candles that have n neighbours on both sides,
    # so we walk from index n up to (total - n).
    for i in range(n, total - n):
        left_highs = highs[i - n:i]        # the n candles immediately to the left
        right_highs = highs[i + 1:i + n + 1]  # the n candles immediately to the right
        # Strict > on both sides: the peak must be taller than everything around it.
        # (Strict avoids counting flat plateaus, which are ambiguous.)
        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            is_swing_high[i] = True

        left_lows = lows[i - n:i]
        right_lows = lows[i + 1:i + n + 1]
        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            is_swing_low[i] = True

    df = df.copy()                          # don't mutate the caller's dataframe by surprise
    df["swing_high"] = is_swing_high
    df["swing_low"] = is_swing_low
    return df


# ---------------------------------------------------------------------------
# When run directly, load the cleaned data, find swings, report, and plot.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N = 3   # the swing sensitivity knob — change this and re-run to see the effect

    df = pd.read_parquet("xauusd_m15_clean.parquet")   # the clean data from Stage 1
    df = find_swings(df, n=N)

    n_highs = df["swing_high"].sum()
    n_lows = df["swing_low"].sum()
    print(f"Swing highs found : {n_highs:,}")
    print(f"Swing lows found  : {n_lows:,}")
    print(f"That's a swing every ~{len(df) / (n_highs + n_lows):.0f} candles on average.")

    # --- Validation plot: last 500 candles, with swings marked -------------
    window = df.iloc[-500:]                      # a readable slice
    sh = window[window["swing_high"]]            # just the swing-high rows in that slice
    sl = window[window["swing_low"]]             # just the swing-low rows

    plt.figure(figsize=(14, 7))
    plt.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    # Mark swing highs on their HIGH price with a red down-triangle,
    # swing lows on their LOW price with a green up-triangle.
    plt.scatter(sh.index, sh["High"], color="red", marker="v", s=40, label="Swing High", zorder=5)
    plt.scatter(sl.index, sl["Low"], color="green", marker="^", s=40, label="Swing Low", zorder=5)
    plt.title(f"XAUUSD M15 - swing highs & lows (n={N}), last 500 candles")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("swings_check.png", dpi=120)
    print("Saved swings_check.png - open it and eyeball whether the peaks/troughs look right.")