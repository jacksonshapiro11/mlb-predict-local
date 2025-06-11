#!/usr/bin/env python3
"""
test_historical_pipeline.py
===========================
Test script to verify the historical features pipeline works correctly.

This script:
1. Tests the historical feature generation
2. Verifies anti-data-leakage measures
3. Checks feature completeness
4. Validates data types and shapes

Usage:
    python test_historical_pipeline.py
"""

import pandas as pd
import numpy as np
import pathlib
from datetime import datetime


def test_historical_features():
    """Test the historical features for completeness and correctness."""

    print("🧪 Testing Historical Features Pipeline")
    print("=" * 50)

    # Check if we have historical features
    feature_dir = pathlib.Path("data/features_historical")
    if not feature_dir.exists():
        print("❌ No historical features directory found. Run the ETL first.")
        return

    # Find available historical feature files
    feature_files = list(feature_dir.glob("statcast_historical_*.parquet"))
    if not feature_files:
        print("❌ No historical feature files found. Run the ETL first.")
        return

    print(f"📁 Found {len(feature_files)} historical feature files:")
    for f in feature_files:
        print(f"   {f.name}")

    # Load the most recent file for testing
    latest_file = sorted(feature_files)[-1]
    print(f"\n📊 Testing with {latest_file.name}...")

    try:
        df = pd.read_parquet(latest_file)
        print(f"✅ Successfully loaded data: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Test 1: Check for required columns
    print("\n🔍 Test 1: Required Columns")
    required_cols = [
        "pitch_type_can",
        "game_date",
        "pitcher",
        "batter",
        "balls",
        "strikes",
        "count_state",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
    else:
        print("✅ All required columns present")

    # Test 2: Check historical features
    print("\n🔍 Test 2: Historical Feature Categories")

    feature_categories = {
        "Arsenal (30d)": [
            "velocity_30d_",
            "spin_rate_30d_",
            "whiff_rate_30d_",
            "usage_30d_",
        ],
        "Recent Form (7d)": ["velocity_7d", "whiff_rate_7d", "hit_rate_7d"],
        "Count State (30d)": [
            "whiff_rate_30d_AHEAD",
            "whiff_rate_30d_BEHIND",
            "whiff_rate_30d_EVEN",
        ],
        "Platoon Splits (30d)": ["vs_L", "vs_R"],
        "Batter Performance (30d)": ["batter_xwoba_30d_", "k_rate_30d"],
        "Within-Game Cumulative": ["cum_", "prev_pitch_", "dvelo1"],
    }

    for category, patterns in feature_categories.items():
        matching_cols = [
            col for col in df.columns if any(pattern in col for pattern in patterns)
        ]
        print(f"  {category}: {len(matching_cols)} features")
        if len(matching_cols) == 0:
            print(f"    ⚠️  No features found for {category}")

    # Test 3: Check for data leakage - first 30 days should have NaN for 30d features
    print("\n🔍 Test 3: Data Leakage Check")

    # Group by pitcher and check early season NaNs
    pitcher_sample = df[df["pitcher"] == df["pitcher"].iloc[0]].sort_values("game_date")
    if len(pitcher_sample) > 30:
        early_data = pitcher_sample.head(30)
        thirty_day_cols = [col for col in df.columns if "30d_" in col]

        if thirty_day_cols:
            early_nulls = early_data[thirty_day_cols].isnull().all().all()
            if early_nulls:
                print("✅ Early season 30-day features are properly NaN (no leakage)")
            else:
                print(
                    "⚠️  Early season 30-day features contain values (potential leakage)"
                )
        else:
            print("⚠️  No 30-day features found for leakage testing")
    else:
        print("⚠️  Not enough data for leakage testing")

    # Test 4: Check cumulative features
    print("\n🔍 Test 4: Cumulative Features Check")

    # Check a single game for cumulative behavior
    game_sample = df[df["game_pk"] == df["game_pk"].iloc[0]].sort_values(
        ["at_bat_number", "pitch_number"]
    )

    if "cum_game_pitches" in game_sample.columns:
        first_pitch_count = game_sample["cum_game_pitches"].iloc[0]
        if pd.isna(first_pitch_count) or first_pitch_count == 0:
            print("✅ First pitch has 0 cumulative count (correct)")
        else:
            print(
                f"⚠️  First pitch has {first_pitch_count} cumulative count (should be 0)"
            )

    # Test 5: Check data types
    print("\n🔍 Test 5: Data Types")

    categorical_cols = [
        "pitch_type",
        "pitch_type_can",
        "stand",
        "p_throws",
        "count_state",
    ]
    for col in categorical_cols:
        if col in df.columns:
            if df[col].dtype.name == "category":
                print(f"✅ {col} is categorical")
            else:
                print(f"⚠️  {col} is {df[col].dtype} (should be categorical)")

    # Test 6: Check for expected pitch types
    print("\n🔍 Test 6: Pitch Types")

    if "pitch_type_can" in df.columns:
        pitch_types = df["pitch_type_can"].value_counts()
        print(f"  Found {len(pitch_types)} pitch types:")
        for pt, count in pitch_types.head(10).items():
            print(f"    {pt}: {count:,} pitches")

        expected_types = ["FF", "SL", "CH", "CU", "SI", "FC", "KC", "FS", "OTHER"]
        missing_types = [pt for pt in expected_types if pt not in pitch_types.index]
        if missing_types:
            print(f"  ⚠️  Missing expected pitch types: {missing_types}")
        else:
            print("  ✅ All expected pitch types present")

    # Test 7: Summary statistics
    print("\n📊 Summary Statistics")
    print(f"  Total pitches: {len(df):,}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Unique pitchers: {df['pitcher'].nunique():,}")
    print(f"  Unique batters: {df['batter'].nunique():,}")
    print(f"  Unique games: {df['game_pk'].nunique():,}")

    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]

    if len(high_missing) > 0:
        print(f"\n⚠️  Features with >50% missing values:")
        for col, pct in high_missing.head(10).items():
            print(f"    {col}: {pct:.1f}% missing")
    else:
        print("\n✅ No features with excessive missing values")

    print("\n🎉 Historical features testing complete!")


if __name__ == "__main__":
    test_historical_features()
