#!/usr/bin/env python
# scripts/fetch_gdelt_data.py

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from gdelt import gdelt
import os
import warnings
from bs4 import GuessedAtParserWarning, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
from utils.headline_utils import fetch_headline
from concurrent.futures import ThreadPoolExecutor, as_completed

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def fetch_all_events(start: str, end: str) -> pd.DataFrame:
    client = gdelt(version=2)
    logger.info(f"Requesting GDELT events from {start} to {end}...")
    df = client.Search([start, end], table="events", normcols=True)
    logger.info(f"Fetched {len(df):,} total events.")
    return df

def filter_and_process(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Unique eventcode values before filtering: {df.eventcode.unique()}")
    macro_code_prefixes = tuple(str(i).zfill(3) for i in range(100, 200))
    macro_df = df[df.eventcode.str.startswith(macro_code_prefixes, na=False)]
    logger.info(f"Events remaining after macro filter: {len(macro_df):,}")

    macro_df = (
        macro_df.rename(columns={
            "sqldate": "date",
            "actor1name": "actor1",
            "actor2name": "actor2",
            "eventcode": "event_type",
            "goldsteinscale": "goldstein_scale",
            "nummentions": "num_mentions",
            "numsources": "num_sources",
            "numarticles": "num_articles",
            "avgtone": "tone",
            "sourceurl": "url",
        })
        .loc[:, ["date","actor1","actor2","event_type","goldstein_scale",
                 "num_mentions","num_sources","num_articles","tone","url"]]
    )
    macro_df["date"] = pd.to_datetime(macro_df["date"], format="%Y%m%d", errors="coerce")
    macro_df = macro_df.dropna(subset=["date"])
    return macro_df

def scope_top_events(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Keep only the top_n events per day by num_articles.
    """
    df = (
        df.sort_values(["date","num_articles"], ascending=[True, False])
          .groupby("date")
          .head(top_n)
          .reset_index(drop=True)
    )
    logger.info(f"Scoped to top {top_n} events/day → {len(df):,} rows")
    return df

def scrape_headlines(urls: pd.Series, max_workers: int = 20) -> pd.Series:
    """
    Fetch headlines in parallel for a series of URLs.
    Returns a Series of the same index with headlines or empty strings.
    """
    headlines = pd.Series([""] * len(urls), index=urls.index)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_idx = {exe.submit(fetch_headline, url): idx 
                         for idx, url in enumerate(urls)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                headlines.iat[idx] = fut.result()
            except Exception:
                headlines.iat[idx] = ""
    return headlines

def main():
    start_date = "2015-02-18"
    end_date   = "2025-04-30"
    raw = fetch_all_events(start_date, end_date)
    proc = filter_and_process(raw)

    # 1) Scope to top 100 events per day
    proc = scope_top_events(proc, top_n=100)

    # 2) Scrape headlines in monthly batches
    proc = proc.sort_values("date").reset_index(drop=True)
    headlines = pd.Series("", index=proc.index)

    # group by each year-month period
    periods = proc["date"].dt.to_period("M").unique()
    for period in periods:
        try:
            # mask for this year-month
            mask = proc["date"].dt.to_period("M") == period
            urls = proc.loc[mask, "url"]
            logger.info(f"Starting headline scrape for {period} ({urls.size} URLs)…")
            sub = scrape_headlines(urls, max_workers=50)
            headlines.loc[sub.index] = sub
            logger.info(f"Finished headline scrape for {period}")
        except Exception as e:
            logger.error(f"Month {period} failed: {e}")
            continue

    proc["headline"] = headlines

    # 3) Drop failures
    before = len(proc)
    proc = proc[proc["headline"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Dropped {before - len(proc):,} rows without headlines → {len(proc):,} remain")

    # 4) Save
    out_path = Path("data/raw/gdelt/gdelt_macro_events_top100_with_headlines.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    # 5) Intermediate checkpointing every quarter
    if period.month % 3 == 0:
        checkpoint_path = Path(f"data/raw/gdelt/gdelt_macro_events_top100_with_headlines_checkpoint_{period}.csv")
        proc.loc[:headlines.notna().sum(), :].to_csv(checkpoint_path, index=False)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
