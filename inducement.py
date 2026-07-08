"""
inducement.py
Stage 3a of the Nixie's SMC backtester: find inducements.

Inducement (Tony) = the first pullback after a BOS/MSS that retraces >= 50%
(but < 100%) of the impulse, measured ORIGIN swing low -> PEAK after the break.
It's the trap that gets swept before the POI.

KEY FIX vs first draft:
  - PEAK is now the TRUE highest high between the break and the pullback,
    not just the first swing high (which was often barely above the break and
    made the impulse look tiny -> inflated retrace %).
  - Valid now requires 50% <= retrace < 100% (a pullback that breaks the origin
    is a reversal, not an inducement).

Run it with:  uv run inducement.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from breaks import label_structure


def find_inducements(df: pd.DataFrame, n: int = 5):
    df = label_structure(df, n=n)

    highs = df["High"].values
    lows = df["Low"].values
    sh_idx = np.where(df["swing_high"].values)[0]
    sl_idx = np.where(df["swing_low"].values)[0]
    times = df.index
    events = df["structure_event"].values

    records = []
    for i in np.where(events != "")[0]:
        ev = events[i]
        bullish = ev.endswith("_up")

        if bullish:
            prior_lows = sl_idx[sl_idx < i]                 # origin candidates
            later_lows = sl_idx[sl_idx > i]                 # the pullback ends on a swing low
            if len(prior_lows) == 0 or len(later_lows) == 0:
                continue
            o_idx = prior_lows[-1]                           # ORIGIN = last swing low before break
            ind_idx = later_lows[0]                          # INDUCEMENT = first swing low after break
            peak_idx = i + int(np.argmax(highs[i:ind_idx + 1]))  # PEAK = true max in between
            origin, peak, ind = lows[o_idx], highs[peak_idx], lows[ind_idx]
            if peak <= origin:
                continue
            retrace = (peak - ind) / (peak - origin) * 100
            end_idx = peak_idx
        else:
            prior_highs = sh_idx[sh_idx < i]
            later_highs = sh_idx[sh_idx > i]
            if len(prior_highs) == 0 or len(later_highs) == 0:
                continue
            o_idx = prior_highs[-1]                          # ORIGIN = last swing high before break
            ind_idx = later_highs[0]                         # INDUCEMENT = first swing high after break
            trough_idx = i + int(np.argmin(lows[i:ind_idx + 1]))  # TROUGH = true min in between
            origin, trough, ind = highs[o_idx], lows[trough_idx], highs[ind_idx]
            if origin <= trough:
                continue
            retrace = (ind - trough) / (origin - trough) * 100
            peak = trough
            end_idx = trough_idx

        records.append({
            "event_time": times[i],
            "direction": "bull" if bullish else "bear",
            "origin_time": times[o_idx],   "origin_price": origin,
            "impulse_end_time": times[end_idx], "impulse_end_price": peak,
            "ind_time": times[ind_idx],    "ind_price": ind,
            "retrace_pct": retrace,
            "valid": 50 <= retrace < 100,          # <-- upper bound added
        })

    inducements = pd.DataFrame(records)
    if len(inducements):
        # DEDUPE: if two events produced an inducement on the same candle, keep one.
        inducements = inducements.drop_duplicates(subset=["ind_time"], keep="first")
        inducements = inducements.reset_index(drop=True)
    return df, inducements


if __name__ == "__main__":
    N = 5
    df = pd.read_parquet("xauusd_m15_clean.parquet")
    df, ind = find_inducements(df, n=N)

    valids = ind[ind["valid"]]
    print(f"Inducement candidates (deduped): {len(ind):,}")
    print(f"  valid (50%-100% pullback)    : {len(valids):,}")
    print(f"  invalid                      : {(~ind['valid']).sum():,}")
    print(f"  median retrace of valids     : {valids['retrace_pct'].median():.0f}%")
    print(f"  retrace range of valids      : {valids['retrace_pct'].min():.0f}% - {valids['retrace_pct'].max():.0f}%")

    print("\nLast 8 VALID inducements:")
    for _, r in valids.tail(8).iterrows():
        print(f"   {r['ind_time']}  {r['direction']:4s}  IDM @ {r['ind_price']:.2f}  "
              f"(pulled back {r['retrace_pct']:.0f}%)")

    window = df.iloc[-400:]
    t0 = window.index[0]
    vind = valids[valids["ind_time"] >= t0]

    plt.figure(figsize=(15, 7))
    plt.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    for _, r in vind.iterrows():
        xs = [r["origin_time"], r["impulse_end_time"], r["ind_time"]]
        ys = [r["origin_price"], r["impulse_end_price"], r["ind_price"]]
        plt.plot(xs, ys, color="purple", linewidth=1, alpha=0.7)
        plt.scatter([r["ind_time"]], [r["ind_price"]], color="purple", s=45, zorder=5)
        plt.text(r["ind_time"], r["ind_price"], "  IDM", color="purple", fontsize=7, va="center")
    plt.title(f"XAUUSD M15 - valid inducements (n={N}), last 400 candles")
    plt.ylabel("Price (USD)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("inducement_check.png", dpi=120)
    print("\nSaved inducement_check.png")