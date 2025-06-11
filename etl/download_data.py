#!/usr/bin/env python3
"""
etl/download_data.py
====================
Simple script to download Statcast data for a given year.
"""

import pandas as pd
import sys
import pathlib
from pybaseball import statcast

# --- Config ---
RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_year(year):
    """Downloads and saves Statcast data for a full year."""
    print(f"📥 Downloading Statcast data for {year}...")
    
    # Define date ranges for the download
    start_date = f"{year}-03-01"
    end_date = f"{year}-12-01"
    
    # Fetch data
    try:
        data = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"❌ Failed to download data for {year}: {e}")
        return
        
    if data.empty:
        print(f"⚠️ No data found for {year}.")
        return

    # Save to parquet
    output_path = RAW_DIR / f"statcast_{year}.parquet"
    print(f"💾 Saving {len(data)} rows to {output_path}")
    data.to_parquet(output_path, index=False)
    print("✅ Download complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            year_to_process = int(sys.argv[1])
            download_year(year_to_process)
        except ValueError:
            print("Usage: python etl/download_data.py <year>")
    else:
        print("Usage: python etl/download_data.py <year>") 