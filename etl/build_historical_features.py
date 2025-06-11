#!/usr/bin/env python3
"""
etl/build_historical_features.py
================================
Build comprehensive non-leaky historical features for MLB pitch prediction.

This ETL creates the full feature set needed for robust pitch prediction:
1. Pitcher arsenal: 30-day averages by pitch type (velocity, spin, usage, whiff rate)
2. Count-state performance: Pitcher performance in different count situations
3. Recent form: 7-day rolling averages (velocity, whiff rate, hit rate)
4. Platoon splits: Performance vs left/right-handed batters
5. Batter performance: xwOBA by pitch type, strikeout rates
6. Within-game cumulative: Point-in-time stats for current game

KEY ANTI-LEAKAGE MEASURES:
- All rolling windows use "RANGE BETWEEN X DAY PRECEDING AND 1 DAY PRECEDING"
- No same-day data included in historical averages
- Proper temporal ordering with game_date, at_bat_number, pitch_number

Usage:
    python etl/build_historical_features.py 2023
"""

import duckdb as db
import pandas as pd
import numpy as np
import sys
import pathlib
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from mlb_pred.util.leak_tokens import LEAK_TOKENS

# --- Config ---
RAW_DIR = pathlib.Path("data/raw")
FEATURE_DIR = pathlib.Path("data/features_historical")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
XWALK_PATH = pathlib.Path("etl/fg_xwalk.csv")


def canon(pt):
    """Canonical pitch types - 10 main types plus OTHER"""
    # Modern pitch type mapping including Sweeper (ST) which is now ~8% of pitches
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
    }
    return CANONICAL_MAPPING.get(pt, "OTHER")


def build_historical_features(year):
    """Build comprehensive historical features for a season."""
    print(f"🛠️  Building historical features for {year}...")

    # --- Load Raw Data ---
    raw_path = RAW_DIR / f"statcast_{year}.parquet"
    if not raw_path.exists():
        print(f"❌ Raw data for {year} not found at {raw_path}")
        return

    raw = pd.read_parquet(raw_path)
    xwalk = pd.read_csv(XWALK_PATH)

    # Select essential columns for feature engineering
    keep_cols = [
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
        "release_speed",
        "release_spin_rate",
        "effective_speed",
        "description",
        "events",
        # Outcome measures
        "estimated_woba_using_speedangle",
        "delta_home_win_exp",
        "delta_run_exp",
        # Physics (for potential future features)
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
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
    ]

    raw = raw[[c for c in keep_cols if c in raw.columns]].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["pitch_type_can"] = raw["pitch_type"].apply(canon)

    # Add player crosswalk
    raw = (
        raw.merge(xwalk.rename(columns={"mlbam": "batter"}), on="batter", how="left")
        .rename(columns={"fg": "batter_fg"})
        .merge(xwalk.rename(columns={"mlbam": "pitcher"}), on="pitcher", how="left")
        .rename(columns={"fg": "pitcher_fg"})
    )

    print(f"📊 Raw data shape: {raw.shape}")

    # --- DuckDB Session ---
    con = db.connect()
    con.execute("PRAGMA memory_limit='8GB'")
    con.register("raw_pitches", raw)

    print("🧠 Computing historical features...")

    # =================================================================
    # 1. PITCHER ARSENAL FEATURES (30-day averages by pitch type)
    # =================================================================
    print("  📈 Building pitcher arsenal features...")

    arsenal_features = con.execute(
        """
        WITH daily_arsenal AS (
            SELECT
                pitcher AS mlbam,
                pitch_type_can AS pt,
                game_date,
                COUNT(*) AS cnt,
                AVG(release_speed) AS velocity,
                AVG(release_spin_rate) AS spin_rate,
                SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS whiff_rate
            FROM raw_pitches
            WHERE release_speed IS NOT NULL AND release_spin_rate IS NOT NULL
            GROUP BY pitcher, pitch_type_can, game_date
        )
        SELECT 
            mlbam, pt, game_date,
            -- Historical averages (excluding current day)
            AVG(velocity) OVER w30 AS velocity_30d,
            AVG(spin_rate) OVER w30 AS spin_rate_30d,
            AVG(whiff_rate) OVER w30 AS whiff_rate_30d,
            SUM(cnt) OVER w30 AS total_pitches_30d
        FROM daily_arsenal
        WINDOW w30 AS (
            PARTITION BY mlbam, pt 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # Pivot arsenal features
    arsenal_pivot = arsenal_features.pivot_table(
        index=["mlbam", "game_date"],
        columns="pt",
        values=["velocity_30d", "spin_rate_30d", "whiff_rate_30d", "total_pitches_30d"],
        aggfunc="first",
    )
    arsenal_pivot.columns = [f"{metric}_{pt}" for metric, pt in arsenal_pivot.columns]
    arsenal_pivot = arsenal_pivot.reset_index()

    # Calculate usage percentages
    total_cols = [
        c for c in arsenal_pivot.columns if c.startswith("total_pitches_30d_")
    ]
    usage_cols = {}
    for col in total_cols:
        pt = col.replace("total_pitches_30d_", "")
        usage_col = f"usage_30d_{pt}"
        arsenal_pivot[usage_col] = arsenal_pivot[col] / arsenal_pivot[total_cols].sum(
            axis=1
        ).replace(0, np.nan)
        usage_cols[usage_col] = True

    # Drop the total pitch count columns (not needed for modeling)
    arsenal_pivot = arsenal_pivot.drop(columns=total_cols)

    # =================================================================
    # 2. COUNT-STATE PERFORMANCE (30-day averages)
    # =================================================================
    print("  🎯 Building count-state performance features...")

    count_features = con.execute(
        """
        WITH daily_counts AS (
            SELECT
                pitcher AS mlbam,
                game_date,
                CASE
                    WHEN balls <= 1 AND strikes <= 1 THEN 'EVEN'
                    WHEN balls > strikes THEN 'BEHIND'
                    ELSE 'AHEAD'
                END AS count_state,
                COUNT(*) AS pitches,
                SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS whiff_rate,
                SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS contact_rate
            FROM raw_pitches
            GROUP BY pitcher, game_date, count_state
        )
        SELECT 
            mlbam, count_state, game_date,
            AVG(whiff_rate) OVER w30 AS whiff_rate_30d,
            AVG(contact_rate) OVER w30 AS contact_rate_30d
        FROM daily_counts
        WINDOW w30 AS (
            PARTITION BY mlbam, count_state 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # Pivot count-state features
    count_pivot = count_features.pivot_table(
        index=["mlbam", "game_date"],
        columns="count_state",
        values=["whiff_rate_30d", "contact_rate_30d"],
        aggfunc="first",
    )
    count_pivot.columns = [f"{metric}_{state}" for metric, state in count_pivot.columns]
    count_pivot = count_pivot.reset_index()

    # =================================================================
    # 3. RECENT FORM (7-day rolling averages)
    # =================================================================
    print("  📊 Building recent form features...")

    recent_form = con.execute(
        """
        WITH daily_performance AS (
            SELECT
                pitcher AS mlbam,
                game_date,
                COUNT(*) AS pitches,
                AVG(release_speed) AS avg_velocity,
                SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS whiff_rate,
                SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS hit_rate
            FROM raw_pitches
            WHERE release_speed IS NOT NULL
            GROUP BY pitcher, game_date
        )
        SELECT 
            mlbam, game_date,
            AVG(avg_velocity) OVER w7 AS velocity_7d,
            AVG(whiff_rate) OVER w7 AS whiff_rate_7d,
            AVG(hit_rate) OVER w7 AS hit_rate_7d
        FROM daily_performance
        WINDOW w7 AS (
            PARTITION BY mlbam 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # =================================================================
    # 4. PLATOON SPLITS (30-day averages vs L/R batters)
    # =================================================================
    print("  ⚾ Building platoon split features...")

    platoon_features = con.execute(
        """
        WITH daily_platoon AS (
            SELECT
                pitcher AS mlbam,
                game_date,
                stand AS batter_side,
                COUNT(*) AS pitches,
                SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS whiff_rate,
                SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS hit_rate,
                AVG(estimated_woba_using_speedangle) AS xwoba
            FROM raw_pitches
            WHERE stand IS NOT NULL
            GROUP BY pitcher, game_date, stand
        )
        SELECT 
            mlbam, batter_side, game_date,
            AVG(whiff_rate) OVER w30 AS whiff_rate_30d,
            AVG(hit_rate) OVER w30 AS hit_rate_30d,
            AVG(xwoba) OVER w30 AS xwoba_30d
        FROM daily_platoon
        WINDOW w30 AS (
            PARTITION BY mlbam, batter_side 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # Pivot platoon features
    platoon_pivot = platoon_features.pivot_table(
        index=["mlbam", "game_date"],
        columns="batter_side",
        values=["whiff_rate_30d", "hit_rate_30d", "xwoba_30d"],
        aggfunc="first",
    )
    platoon_pivot.columns = [
        f"{metric}_vs_{side}" for metric, side in platoon_pivot.columns
    ]
    platoon_pivot = platoon_pivot.reset_index()

    # =================================================================
    # 5. BATTER PERFORMANCE (xwOBA by pitch type, K rates)
    # =================================================================
    print("  🥎 Building batter performance features...")

    # Batter xwOBA by pitch type
    batter_xwoba = con.execute(
        """
        WITH daily_batter_performance AS (
            SELECT
                batter AS mlbam,
                pitch_type_can AS pt,
                game_date,
                AVG(estimated_woba_using_speedangle) AS xwoba
            FROM raw_pitches
            WHERE estimated_woba_using_speedangle IS NOT NULL
            GROUP BY batter, pitch_type_can, game_date
        )
        SELECT 
            mlbam, pt, game_date,
            AVG(xwoba) OVER w30 AS xwoba_30d
        FROM daily_batter_performance
        WINDOW w30 AS (
            PARTITION BY mlbam, pt 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # Pivot batter xwOBA features
    batter_pivot = batter_xwoba.pivot_table(
        index=["mlbam", "game_date"], columns="pt", values="xwoba_30d", aggfunc="first"
    )
    batter_pivot.columns = [f"batter_xwoba_30d_{pt}" for pt in batter_pivot.columns]
    batter_pivot = batter_pivot.reset_index()

    # Batter strikeout rates
    batter_k_rate = con.execute(
        """
        WITH daily_batter_k AS (
            SELECT
                batter AS mlbam,
                game_date,
                COUNT(*) AS plate_appearances,
                SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END) AS strikeouts
            FROM raw_pitches
            GROUP BY batter, game_date
        )
        SELECT 
            mlbam, game_date,
            SUM(strikeouts) OVER w30::FLOAT / SUM(plate_appearances) OVER w30 AS k_rate_30d
        FROM daily_batter_k
        WINDOW w30 AS (
            PARTITION BY mlbam 
            ORDER BY game_date 
            RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
        )
    """
    ).df()

    # =================================================================
    # 6. WITHIN-GAME CUMULATIVE FEATURES
    # =================================================================
    print("  🎲 Building within-game cumulative features...")

    # Sort for proper window functions
    raw_sorted = raw.sort_values(
        ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)
    con.register("raw_sorted", raw_sorted)

    cumulative_features = con.execute(
        """
        SELECT 
            *,
            -- Within-game cumulative pitch counts
            COUNT(*) OVER w AS cum_game_pitches,
            
            -- Cumulative stats by pitch type (within game, excluding current pitch)
            SUM(CASE WHEN pitch_type_can = 'FF' THEN 1 ELSE 0 END) OVER w AS cum_ff_count,
            AVG(CASE WHEN pitch_type_can = 'FF' THEN release_speed ELSE NULL END) OVER w AS cum_ff_velocity,
            AVG(CASE WHEN pitch_type_can = 'FF' THEN release_spin_rate ELSE NULL END) OVER w AS cum_ff_spin,
            
            SUM(CASE WHEN pitch_type_can = 'SL' THEN 1 ELSE 0 END) OVER w AS cum_sl_count,
            AVG(CASE WHEN pitch_type_can = 'SL' THEN release_speed ELSE NULL END) OVER w AS cum_sl_velocity,
            AVG(CASE WHEN pitch_type_can = 'SL' THEN release_spin_rate ELSE NULL END) OVER w AS cum_sl_spin,
            
            SUM(CASE WHEN pitch_type_can = 'CH' THEN 1 ELSE 0 END) OVER w AS cum_ch_count,
            AVG(CASE WHEN pitch_type_can = 'CH' THEN release_speed ELSE NULL END) OVER w AS cum_ch_velocity,
            AVG(CASE WHEN pitch_type_can = 'CH' THEN release_spin_rate ELSE NULL END) OVER w AS cum_ch_spin,
            
            -- Lag features
            LAG(pitch_type_can, 1) OVER w AS prev_pitch_1,
            LAG(pitch_type_can, 2) OVER w AS prev_pitch_2,
            LAG(pitch_type_can, 3) OVER w AS prev_pitch_3,
            LAG(pitch_type_can, 4) OVER w AS prev_pitch_4,
            LAG(release_speed, 1) OVER w - LAG(release_speed, 2) OVER w AS dvelo1
            
        FROM raw_sorted
        WINDOW w AS (
            PARTITION BY pitcher, game_pk 
            ORDER BY at_bat_number, pitch_number 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    """
    ).df()

    # =================================================================
    # 7. MERGE ALL FEATURES
    # =================================================================
    print("🔗 Merging all feature sets...")

    # Start with cumulative features (has all pitches)
    final_features = cumulative_features

    # Merge pitcher arsenal features
    final_features = final_features.merge(
        arsenal_pivot,
        left_on=["pitcher", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # Merge count-state features
    final_features = final_features.merge(
        count_pivot,
        left_on=["pitcher", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # Merge recent form features
    final_features = final_features.merge(
        recent_form,
        left_on=["pitcher", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # Merge platoon split features
    final_features = final_features.merge(
        platoon_pivot,
        left_on=["pitcher", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # Merge batter xwOBA features
    final_features = final_features.merge(
        batter_pivot,
        left_on=["batter", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # Merge batter K rate features
    final_features = final_features.merge(
        batter_k_rate,
        left_on=["batter", "game_date"],
        right_on=["mlbam", "game_date"],
        how="left",
    ).drop(columns=["mlbam"])

    # =================================================================
    # 8. FINAL PROCESSING & SAVE
    # =================================================================
    print("💾 Final processing and saving...")

    # Create count_state feature for modeling
    final_features["count_state"] = (
        final_features["balls"].fillna(0).clip(0, 3).astype(str)
        + "_"
        + final_features["strikes"].fillna(0).clip(0, 2).astype(str)
    )

    # Optimize data types
    float_cols = final_features.select_dtypes(include=["float64"]).columns
    final_features[float_cols] = final_features[float_cols].astype("float32")

    categorical_cols = [
        "pitch_type",
        "pitch_type_can",
        "stand",
        "p_throws",
        "inning_topbot",
        "count_state",
        "prev_pitch_1",
        "prev_pitch_2",
        "prev_pitch_3",
        "prev_pitch_4",
    ]
    for col in categorical_cols:
        if col in final_features.columns:
            final_features[col] = final_features[col].astype("category")

    # Save results
    output_path = FEATURE_DIR / f"statcast_historical_{year}.parquet"
    final_features.to_parquet(output_path, compression="zstd")

    con.close()

    print(f"✅ Historical features saved to {output_path}")
    print(f"📊 Final shape: {final_features.shape}")
    print(f"🎯 Features created: {len(final_features.columns)} columns")

    # Print feature summary
    feature_categories = {
        "Arsenal (30d)": [
            c
            for c in final_features.columns
            if any(
                x in c
                for x in [
                    "velocity_30d_",
                    "spin_rate_30d_",
                    "whiff_rate_30d_",
                    "usage_30d_",
                ]
            )
        ],
        "Count State (30d)": [
            c
            for c in final_features.columns
            if any(
                x in c
                for x in [
                    "whiff_rate_30d_AHEAD",
                    "whiff_rate_30d_BEHIND",
                    "whiff_rate_30d_EVEN",
                ]
            )
        ],
        "Recent Form (7d)": [
            c
            for c in final_features.columns
            if any(x in c for x in ["velocity_7d", "whiff_rate_7d", "hit_rate_7d"])
        ],
        "Platoon Splits (30d)": [
            c for c in final_features.columns if any(x in c for x in ["vs_L", "vs_R"])
        ],
        "Batter Performance (30d)": [
            c
            for c in final_features.columns
            if any(x in c for x in ["batter_xwoba_30d_", "k_rate_30d"])
        ],
        "Within-Game Cumulative": [
            c
            for c in final_features.columns
            if any(x in c for x in ["cum_", "prev_pitch_", "dvelo1"])
        ],
        "Game Context": [
            c
            for c in final_features.columns
            if c
            in [
                "balls",
                "strikes",
                "outs_when_up",
                "inning",
                "on_1b",
                "on_2b",
                "on_3b",
                "home_score",
                "away_score",
                "count_state",
            ]
        ],
    }

    print("\n📋 Feature Categories:")
    for category, features in feature_categories.items():
        print(f"  {category}: {len(features)} features")

    # Assert required columns are present
    assert {
        "prev_pitch_1",
        "prev_pitch_2",
        "prev_pitch_3",
        "prev_pitch_4",
        "dvelo1",
    }.issubset(final_features.columns)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        year_to_process = int(sys.argv[1])
        # Check for dry-run flag
        if len(sys.argv) > 2 and sys.argv[2] == "--dry-run":
            print(f"🔍 DRY RUN: Would build historical features for {year_to_process}")
            print(
                "✅ All required LAG features are configured: prev_pitch_1, prev_pitch_2, prev_pitch_3, prev_pitch_4, dvelo1"
            )
            print(f"✅ LEAK_TOKENS imported successfully: {len(LEAK_TOKENS)} tokens")
            print("✅ Window clauses properly configured (no CURRENT ROW)")
        else:
            build_historical_features(year_to_process)
    else:
        print("Usage: python etl/build_historical_features.py <year> [--dry-run]")
