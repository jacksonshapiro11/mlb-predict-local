#!/usr/bin/env python3
"""
etl/build_cumulative_features.py
================================
ETL to create truly predictive, point-in-time cumulative features.
"""

import duckdb as db
import pandas as pd
import numpy as np
import sys
import pathlib

# --- Config ---
RAW_DIR = pathlib.Path("data/raw")
FEATURE_DIR = pathlib.Path("data/features_cumulative")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
XWALK_PATH = pathlib.Path("etl/fg_xwalk.csv")


def canon(pt):
    """Canonical pitch types - consistent with historical features"""
    CANONICAL_MAPPING = {
        "FF": "FF",  # Four-seam fastball
        "SI": "SI",  # Sinker
        "SL": "SL",  # Slider
        "CH": "CH",  # Changeup
        "CU": "CU",  # Curveball
        "FC": "FC",  # Cutter
        "KC": "KC",  # Knuckle curve
        "FS": "FS",  # Splitter
        "ST": "ST",  # Sweeper (modern slider variant)
        "SV": "SL",  # Slurve → map to Slider
        "FA": "FF",  # Fastball → map to Four-seam
        "FT": "SI",  # Two-seam → map to Sinker
        "CS": "CU",  # Slow curve → map to Curveball
        "SC": "OTHER",  # Screwball
        "KN": "OTHER",  # Knuckleball
        "EP": "OTHER",  # Eephus
        "UN": "OTHER",  # Unknown
        "PO": "OTHER",  # Pitchout
    }
    return CANONICAL_MAPPING.get(pt, "OTHER")


def build_season_cumulative(year):
    """Builds cumulative features for a given season."""
    print(f"🛠️  Building cumulative features for {year}...")

    # --- Load Data ---
    raw_path = RAW_DIR / f"statcast_{year}.parquet"
    if not raw_path.exists():
        print(f"❌ Raw data for {year} not found at {raw_path}")
        return

    raw = pd.read_parquet(raw_path)
    xwalk = pd.read_csv(XWALK_PATH)

    # Basic cleaning and prep
    keep = [
        "pitch_type",
        "game_date",
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "player_name",
        "batter",
        "pitcher",
        "events",
        "description",
        "zone",
        "des",
        "game_type",
        "stand",
        "p_throws",
        "home_team",
        "away_team",
        "type",
        "hit_location",
        "bb_type",
        "on_3b",
        "on_2b",
        "on_1b",
        "inning",
        "inning_topbot",
        "game_pk",
        "at_bat_number",
        "pitch_number",
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
        "effective_speed",
        "release_spin_rate",
        "release_extension",
        "release_pos_y",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "launch_speed_angle",
        "launch_speed",
        "launch_angle",
        "spin_axis",
        "delta_home_win_exp",
        "delta_run_exp",
        "bat_score",
        "fld_score",
        "post_away_score",
        "post_home_score",
        "post_bat_score",
        "post_fld_score",
        "if_fielding_alignment",
        "of_fielding_alignment",
        "balls",
        "strikes",
        "outs_when_up",
        "hc_x",
        "hc_y",
    ]

    raw = raw[[c for c in keep if c in raw.columns]].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["pitch_type_can"] = raw["pitch_type"].apply(canon)

    # Join with FanGraphs ID crosswalk
    raw = (
        raw.merge(xwalk.rename(columns={"mlbam": "batter"}), on="batter", how="left")
        .rename(columns={"fg": "batter_fg"})
        .merge(xwalk.rename(columns={"mlbam": "pitcher"}), on="pitcher", how="left")
        .rename(columns={"fg": "pitcher_fg"})
    )

    # Sort for window functions
    raw = raw.sort_values(
        ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    # --- DuckDB for Cumulative Features ---
    con = db.connect()
    con.register("p", raw)

    # Define the core SQL for cumulative calculations
    cumulative_sql = """
    SELECT 
        *,
        -- Cumulative count of pitches today before this one
        COUNT(*) OVER w AS cum_game_pitches,
        
        -- Cumulative stats for EACH pitch type today before this one
        SUM(CASE WHEN pitch_type_can = 'FF' THEN 1 ELSE 0 END) OVER w AS cum_ff_count,
        AVG(CASE WHEN pitch_type_can = 'FF' THEN release_speed ELSE NULL END) OVER w AS cum_ff_velo,
        AVG(CASE WHEN pitch_type_can = 'FF' THEN release_spin_rate ELSE NULL END) OVER w AS cum_ff_spin,
        
        SUM(CASE WHEN pitch_type_can = 'SL' THEN 1 ELSE 0 END) OVER w AS cum_sl_count,
        AVG(CASE WHEN pitch_type_can = 'SL' THEN release_speed ELSE NULL END) OVER w AS cum_sl_velo,
        AVG(CASE WHEN pitch_type_can = 'SL' THEN release_spin_rate ELSE NULL END) OVER w AS cum_sl_spin,
        
        SUM(CASE WHEN pitch_type_can = 'CH' THEN 1 ELSE 0 END) OVER w AS cum_ch_count,
        AVG(CASE WHEN pitch_type_can = 'CH' THEN release_speed ELSE NULL END) OVER w AS cum_ch_velo,
        AVG(CASE WHEN pitch_type_can = 'CH' THEN release_spin_rate ELSE NULL END) OVER w AS cum_ch_spin,

        -- Lag features
        LAG(pitch_type_can, 1) OVER w AS prev_pt1,
        LAG(pitch_type_can, 2) OVER w AS prev_pt2,
        release_speed - LAG(release_speed, 1) OVER w AS dvelo1
        
    FROM p
    WINDOW w AS (
        PARTITION BY pitcher, game_pk 
        ORDER BY at_bat_number, pitch_number 
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
    """

    # Execute and fetch results
    print("🧠 Calculating cumulative features with DuckDB...")
    final_df = con.execute(cumulative_sql).df()

    # Close the connection
    con.close()

    # Save the new feature set
    output_path = FEATURE_DIR / f"statcast_cumulative_{year}.parquet"
    print(f"💾 Saving cumulative features to {output_path}")
    final_df.to_parquet(output_path, index=False)

    print(f"✅ Finished cumulative features for {year}. Shape: {final_df.shape}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        year_to_process = int(sys.argv[1])
        build_season_cumulative(year_to_process)
    else:
        print("Usage: python etl/build_cumulative_features.py <year>")
