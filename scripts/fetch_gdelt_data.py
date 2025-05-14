"""
scripts/fetch_gdelt_data.py

Fetch full GDELT events (2015–2025) via the Python gdelt wrapper,
filter for macro themes, process, and write a single CSV.
"""

import logging
from datetime import datetime
import pandas as pd
from gdelt import gdelt
import os

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def fetch_all_events(start: str, end: str) -> pd.DataFrame:
    """
    Fetch GDELT events between start and end dates (YYYY-MM-DD)
    using the gdelt Python wrapper.
    """
    client = gdelt(version=2)               # use GDELT 2.0 events table
    logging.info(f"Requesting GDELT events from {start} to {end}...")
    df = client.Search([start, end], table="events", normcols=True)
    logging.info(f"Fetched {len(df):,} total events.")
    return df

def filter_and_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only macro-related eventcode prefixes (100-199) and
    select+rename columns for sentiment analysis.
    """
    # Log unique eventcode values before filtering
    logging.info(f"Unique eventcode values before filtering: {df.eventcode.unique()}")

    # Macro event codes: 100-199 (economic/financial/monetary)
    macro_code_prefixes = tuple(str(i).zfill(3) for i in range(100, 200))
    macro_df = df[df.eventcode.str.startswith(macro_code_prefixes, na=False)]
    logging.info(f"Events remaining after macro filter: {len(macro_df):,}")

    # Select & rename (all lowercase, correct typos)
    macro_df = macro_df.rename(columns={
        "sqldate":       "date",
        "actor1name":   "actor1",
        "actor2name":   "actor2",
        "eventcode":    "event_type",
        "goldsteinscale":"goldstein_scale",
        "nummentions": "num_mentions",
        "numsources":   "num_sources",
        "numarticles":  "num_articles",
        "avgtone":      "tone",
        "sourceurl":    "url",
    })[
        ["date","actor1","actor2","event_type","goldstein_scale",
         "num_mentions","num_sources","num_articles","tone","url"]
    ]

    # Convert date column
    macro_df["date"] = pd.to_datetime(macro_df["date"], format="%Y%m%d", errors="coerce")
    macro_df = macro_df.dropna(subset=["date"])
    return macro_df

def main():
    start_date = "2015-02-18"  # Earliest date supported by GDELT 2.0
    end_date   = "2025-04-30"
    raw = fetch_all_events(start_date, end_date)
    proc = filter_and_process(raw)
    os.makedirs("data/raw/gdelt", exist_ok=True)
    proc.to_csv("data/raw/gdelt/gdelt_macro_events_2015_2025.csv", index=False)
    logging.info("Saved processed events to data/raw/gdelt/gdelt_macro_events_2015_2025.csv")

if __name__ == "__main__":
    main() 