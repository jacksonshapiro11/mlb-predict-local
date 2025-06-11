#!/usr/bin/env python3

import argparse
import polars as pl
from pathlib import Path


def select_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Select required columns from the DataFrame."""
    required_columns = [
        "game_date",
        "pitch_type",
        "events",
        "release_speed",
        "release_spin_rate",
        "hc_x",
        "hc_y",
        "stand",
        "p_throws",
        "home_team",
    ]
    return df.select(required_columns)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Statcast CSV files to Parquet format"
    )
    parser.add_argument(
        "--in_dir", required=True, help="Input directory containing CSV files"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for Parquet files"
    )
    args = parser.parse_args()

    # Read the first CSV file found in the input directory
    in_dir = Path(args.in_dir)
    csv_files = list(in_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in input directory")
        return 1

    # Read and process the first file
    df = pl.read_csv(csv_files[0])
    df = select_columns(df)

    # Extract year from filename (assuming format statcast_YYYY.csv)
    year = csv_files[0].stem.split("_")[1]

    # Create output directory for the year
    out_dir = Path(args.out_dir) / year
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write to Parquet with snappy compression
    output_file = out_dir / "part_0.parquet"
    df.write_parquet(output_file, compression="snappy")

    print("OK")
    return 0


if __name__ == "__main__":
    exit(main())
