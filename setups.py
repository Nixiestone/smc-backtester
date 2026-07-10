"""
setups.py
Stage 4b of the Nixie's SMC backtester: turn detections into FINAL setups.

Applies, in order:
  1. Geometry filter  - OB must sit entirely BEYOND its inducement
                        (bull: ob_top < IDM ; bear: ob_bottom > IDM).
  2. Mitigation check - was the OB tapped AFTER the structure break but BEFORE
                        the sweep? (Tony: mitigation only counts after BOS/MSS.)
  3. The fork         - unmitigated -> OB trade (our v1 path);
                        mitigated   -> breaker trade (marked, DEFERRED to v2).

Output: the tidy list of tradeable OB setups that Stage 5 will simulate.

Run it with:  uv run setups.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from inducement import find_inducements
from sweep import find_sweeps
from poi import find_order_blocks


def finalize_setups(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    lows = df["Low"].values
    highs = df["High"].values

    geom_ok, mitigated, poi_type = [], [], []
    for _, r in s.iterrows():
        bull = r["direction"] == "bull"

        # 1) GEOMETRY: OB fully beyond the inducement?
        if bull:
            g = r["ob_top"] < r["ind_price"]
        else:
            g = r["ob_bottom"] > r["ind_price"]
        geom_ok.append(bool(g))

        # 2) MITIGATION: any tap of the OB between the break and the sweep?
        ev_pos = df.index.get_loc(r["event_time"])
        sweep_pos = df.index.get_loc(r["sweep_time"]) if pd.notna(r["sweep_time"]) else ev_pos
        mit = False
        for k in range(ev_pos + 1, sweep_pos + 1):
            if bull and lows[k] <= r["ob_top"]:       # price dipped into the OB
                mit = True; break
            if (not bull) and highs[k] >= r["ob_bottom"]:
                mit = True; break
        mitigated.append(mit)

        # 3) FORK
        poi_type.append("breaker" if mit else "OB")

    out = s.copy()
    out["geom_ok"] = geom_ok
    out["mitigated"] = mitigated
    out["poi_type"] = poi_type
    # a v1-tradeable setup: swept, has an OB, geometry ok, and unmitigated
    out["tradeable"] = out["swept"] & out["ob_time"].notna() & out["geom_ok"] & (~out["mitigated"])
    return out


if __name__ == "__main__":
    N = 5
    df = pd.read_parquet("xauusd_m15_clean.parquet")

    df, ind = find_inducements(df, n=N)
    valids = ind[ind["valid"]].reset_index(drop=True)
    valids = find_sweeps(df, valids, max_wait=96)
    valids = find_order_blocks(df, valids)
    s = finalize_setups(df, valids)

    swept = s[s["swept"]]
    with_ob = swept[swept["ob_time"].notna()]
    geom = with_ob[with_ob["geom_ok"]]
    print("FUNNEL (how many survive each rule):")
    print(f"  valid inducements        : {len(s):,}")
    print(f"  ...swept                 : {len(swept):,}")
    print(f"  ...with an order block   : {len(with_ob):,}")
    print(f"  ...OB beyond inducement  : {len(geom):,}   (dropped {len(with_ob)-len(geom)} overlaps)")
    print(f"  ...unmitigated -> OB trade : {s['tradeable'].sum():,}   <-- Stage 5 will simulate these")
    print(f"     mitigated  -> breaker (v2, deferred): {(geom['mitigated']).sum():,}")

    trades = s[s["tradeable"]]
    print("\nLast 6 tradeable OB setups:")
    for _, r in trades.tail(6).iterrows():
        print(f"   {r['direction']:4s}  OB {r['ob_bottom']:.2f}-{r['ob_top']:.2f}  "
              f"IDM {r['ind_price']:.2f}  swept {r['sweep_time']}")

    # --- Plot last 400 candles: only the final tradeable OB setups ---------
    window = df.iloc[-400:]
    t0 = window.index[0]
    tc = trades[trades["ind_time"] >= t0]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(window.index, window["Close"], color="lightgray", linewidth=1, label="Close")
    for _, r in tc.iterrows():
        color = "green" if r["direction"] == "bull" else "red"
        x0 = mdates.date2num(r["ob_time"])
        x1 = mdates.date2num(r["sweep_time"]) if pd.notna(r["sweep_time"]) else x0 + 0.2
        ax.add_patch(Rectangle((x0, r["ob_bottom"]), x1 - x0, r["ob_top"] - r["ob_bottom"],
                               color=color, alpha=0.3, zorder=2))
        ax.scatter([r["ind_time"]], [r["ind_price"]], color="purple", s=25, zorder=5)
        if pd.notna(r["sweep_time"]):
            ax.scatter([r["sweep_time"]], [r["ind_price"]], color="orange", marker="x", s=45, zorder=6)
    ax.set_title(f"XAUUSD M15 - final tradeable OB setups (n={N}), last 400 candles")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("setups_check.png", dpi=120)
    print("\nSaved setups_check.png - box=OB, purple=inducement, orange x=sweep. These become trades.")