#!/usr/bin/env python3
"""
etl/fetch_statcast.py
====================
Fetch raw Statcast data from Baseball Savant and save as parquet files.

This script downloads raw pitch-by-pitch data for specified seasons
and saves it in a format that can be consumed by the historical
features ETL pipeline.

Usage:
    python etl/fetch_statcast.py 2023
    python etl/fetch_statcast.py 2024
"""

import sys
import pathlib
import pandas as pd
from pybaseball import statcast
from pybaseball.cache import enable
from datetime import datetime

# Enable pybaseball caching
enable()

# Directories
RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_statcast_season(year):
    """Fetch Statcast data for a full season."""
    print(f"🔄 Fetching Statcast data for {year}...")

    start_date = f"{year}-03-01"  # Spring training
    end_date = f"{year}-11-30"  # World Series

    # Check if file already exists
    output_path = RAW_DIR / f"statcast_{year}.parquet"
    if output_path.exists():
        print(f"✅ {output_path} already exists - skipping download")
        return

    try:
        # Fetch data from Baseball Savant
        print(f"📡 Downloading data from {start_date} to {end_date}...")
        data = statcast(start_dt=start_date, end_dt=end_date)

        if data.empty:
            print(f"❌ No data returned for {year}")
            return

        print(f"📊 Downloaded {len(data):,} pitches")

        # Basic data cleaning
        data["game_date"] = pd.to_datetime(data["game_date"])

        # Select essential columns for features
        essential_cols = [
            # Identifiers & temporal
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "pitcher",
            "batter",
            "home_team",
            # Player characteristics
            "stand",
            "p_throws",
            # Game situation
            "inning",
            "inning_topbot",
            "outs_when_up",
            "on_1b",
            "on_2b",
            "on_3b",
            "balls",
            "strikes",
            "home_score",
            "away_score",
            # Pitch characteristics
            "pitch_type",
            "pitch_name",
            "release_speed",
            "release_spin_rate",
            "effective_speed",
            "release_extension",
            "description",
            "events",
            # Outcome measures
            "estimated_woba_using_speedangle",
            "delta_home_win_exp",
            "delta_run_exp",
            "home_win_exp",
            "bat_win_exp",
            # Physics
            "pfx_x",
            "pfx_z",
            "plate_x",
            "plate_z",
            "vx0",
            "vy0",
            "vz0",
            "ax",
            "ay",
            "az",
            "sz_top",
            "sz_bot",
            "zone",
            "spin_axis",
            "release_pos_x",
            "release_pos_y",
            "release_pos_z",
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "hc_x",
            "hc_y",
        ]

        # Keep only columns that exist in the data
        available_cols = [col for col in essential_cols if col in data.columns]
        data_clean = data[available_cols].copy()

        print(f"📋 Keeping {len(available_cols)} essential columns")

        # Optimize data types for storage
        for col in data_clean.select_dtypes(include=["float64"]).columns:
            data_clean[col] = data_clean[col].astype("float32")

        for col in [
            "pitch_type",
            "pitch_name",
            "stand",
            "p_throws",
            "inning_topbot",
            "description",
            "events",
        ]:
            if col in data_clean.columns:
                data_clean[col] = data_clean[col].astype("category")

        # Save as compressed parquet
        data_clean.to_parquet(output_path, compression="zstd")
        print(f"✅ Saved to {output_path}")
        print(f"📁 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Print summary statistics
        print(f"\n📊 Data Summary for {year}:")
        print(f"  Total pitches: {len(data_clean):,}")
        print(
            f"  Date range: {data_clean['game_date'].min()} to {data_clean['game_date'].max()}"
        )
        print(f"  Unique games: {data_clean['game_pk'].nunique():,}")
        print(f"  Unique pitchers: {data_clean['pitcher'].nunique():,}")
        print(f"  Unique batters: {data_clean['batter'].nunique():,}")

        if "pitch_type" in data_clean.columns:
            pitch_counts = data_clean["pitch_type"].value_counts()
            print(f"  Top pitch types:")
            for pitch, count in pitch_counts.head(5).items():
                print(f"    {pitch}: {count:,} ({count/len(data_clean)*100:.1f}%)")

    except Exception as e:
        print(f"❌ Error fetching data for {year}: {e}")
        print("💡 Try running: pip install pybaseball --upgrade")


def main():
    if len(sys.argv) < 2:
        print("Usage: python etl/fetch_statcast.py <year>")
        print("Example: python etl/fetch_statcast.py 2023")
        sys.exit(1)

    try:
        year = int(sys.argv[1])
        if year < 2008 or year > datetime.now().year:
            print(
                f"❌ Invalid year: {year}. Statcast data available from 2008-{datetime.now().year}"
            )
            sys.exit(1)

        fetch_statcast_season(year)

    except ValueError:
        print(f"❌ Invalid year: {sys.argv[1]}. Please provide a valid year.")
        sys.exit(1)


if __name__ == "__main__":
    main()
