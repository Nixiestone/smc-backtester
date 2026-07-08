"""
load_data.py
Stage 1 of the Nixie's SMC backtester: get the price data into Python, cleanly.

What this script does, in plain English:
  1. Peeks at the raw CSV so we SEE its exact format before trusting it.
  2. Loads it into a tidy pandas table with a real datetime index and the
     columns we need: Open, High, Low, Close (+ Volume if present).
  3. Runs sanity checks, because building on unvalidated data is how you get
     a "profitable" backtest that's actually just broken data.
  4. Saves a quick chart so your eyes can confirm it looks like gold.

Run it with:  python load_data.py   (or: uv run load_data.py)
"""

import pandas as pd               # pandas: holds the candles as a table and slices them by time
import matplotlib.pyplot as plt   # matplotlib: draws the sanity-check chart

# ---------------------------------------------------------------------------
# 1. POINT THIS AT YOUR FILE
# ---------------------------------------------------------------------------
CSV_PATH = "XAUUSD15.csv"       # <-- change to the EXACT filename you downloaded

# ---------------------------------------------------------------------------
# 2. PEEK AT THE RAW FILE FIRST  (never assume the format)
# ---------------------------------------------------------------------------
# Different sources differ: comma vs semicolon separators, header row or not,
# date+time in one column or two. So we print the first 3 lines and LOOK.
print("First 3 raw lines of your file:")
with open(CSV_PATH) as f:
    for _ in range(3):
        print("   ", f.readline().strip())
print("-" * 60)

# ---------------------------------------------------------------------------
# 3. LOAD IT
# ---------------------------------------------------------------------------
# This assumes the common MT-style layout (no header, comma-separated):
#   2023.01.02,00:00,1824.50,1825.00,1824.00,1824.80,123
#   = date , time , open , high , low , close , volume
#
# If your peek in section 2 showed something different, adjust:
#   - semicolons instead of commas?  add  sep=";"
#   - a header row of names already?  remove header=None and the names=[...]
#   - date & time already in ONE column? see the note under section 4.
df = pd.read_csv(
    CSV_PATH,
    sep=r"\s+",                                             # columns are split by WHITESPACE (tabs/spaces), not commas
    header=None,                                            # file has no column-name row
    names=["date", "time", "Open", "High", "Low", "Close", "Volume"],  # so we name them
)

# Glue the separate date + time text columns into one, then convert to real
# datetime objects. The format string must match your file EXACTLY:
#   %Y = 4-digit year, %m = month, %d = day, %H = 24h hour, %M = minute
df["Datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),  # .astype(str) dodges pandas 3.0 Arrow-string quirks
    format="%Y-%m-%d %H:%M",                                # DASHES, matching your file (2022-04-12 19:00)
)
# NOTE: if your file has ONE datetime column instead of two, delete the two
# lines above and instead do:
#   df["Datetime"] = pd.to_datetime(df["datetime_col"])

# ---------------------------------------------------------------------------
# 4. TIDY IT UP
# ---------------------------------------------------------------------------
df = df.set_index("Datetime")                          # timestamp becomes the row index
df = df[["Open", "High", "Low", "Close", "Volume"]]    # keep only what we need, in order
df = df.astype(float)                                  # force real numbers (not Arrow strings) so later math is predictable
df = df.sort_index()                                   # oldest candle first, guaranteed

# ---------------------------------------------------------------------------
# 5. SANITY CHECKS  (prove the data is clean before trusting it)
# ---------------------------------------------------------------------------
print(f"Candles loaded : {len(df):,}")
print(f"Date range     : {df.index[0]}  ->  {df.index[-1]}")
print("Missing values per column:")
print(df.isna().sum().to_string())

# Logic check: a real candle ALWAYS has High >= Low and High >= Open/Close etc.
# If any row fails this, the data (or our parsing) is broken.
broken = df[(df["High"] < df["Low"]) |
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"])]
print(f"Broken candles : {len(broken)}   (should be 0)")
print("-" * 60)
print("First 3 clean rows:")
print(df.head(3).to_string())

# ---------------------------------------------------------------------------
# 6. EYEBALL IT
# ---------------------------------------------------------------------------
# Plot only the last 2000 candles so the chart is readable, not a smear.
df["Close"].iloc[-2000:].plot(title="XAUUSD M15 Close - last 2000 candles")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("data_check.png", dpi=120)
print("Saved chart to data_check.png - open it and confirm it looks like gold.")

# We also save the cleaned data so the NEXT stage can just load this, fast.
df.to_parquet("xauusd_m15_clean.parquet")
print("Saved cleaned data to xauusd_m15_clean.parquet")