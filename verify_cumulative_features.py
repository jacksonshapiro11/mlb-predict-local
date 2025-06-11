#!/usr/bin/env python3
"""
verify_cumulative_features.py
=============================
Verify that the new cumulative features are calculated correctly and do not leak information.
"""

import pandas as pd
import numpy as np
import pathlib
import duckdb as db

# Temporarily add etl directory to path to import build_cumulative_features
import sys
sys.path.append('etl')
from build_cumulative_features import build_season_cumulative

def verify_cumulative_calculation():
    """Verify cumulative features are correct and non-leaky."""
    
    print("🕵️  VERIFYING CUMULATIVE FEATURE CALCULATION")
    print("=" * 60)
    
    # --- Step 1: Generate cumulative features for a sample year ---
    YEAR = 2023  # Use a recent, complete year for verification
    build_season_cumulative(YEAR)
    
    # --- Step 2: Load the newly generated features ---
    feature_path = pathlib.Path(f"data/features_cumulative/statcast_cumulative_{YEAR}.parquet")
    if not feature_path.exists():
        print(f"❌ Cumulative feature file not found at {feature_path}")
        return
        
    df = pd.read_parquet(feature_path)
    print(f"\n📄 Loaded cumulative features for {YEAR}. Shape: {df.shape}")
    
    # --- Step 3: Analyze a single game for a single pitcher ---
    # Find a game with a decent number of pitches to analyze
    game_counts = df.groupby(['game_pk', 'pitcher']).size().reset_index(name='counts')
    sample = game_counts[game_counts['counts'] > 30].iloc[0]
    sample_pitcher = sample['pitcher']
    sample_game = sample['game_pk']
    
    pitcher_game = df[
        (df['pitcher'] == sample_pitcher) & 
        (df['game_pk'] == sample_game)
    ].copy().sort_values(['at_bat_number', 'pitch_number']).reset_index(drop=True)
    
    print(f"\n🎯 Analyzing pitcher {sample_pitcher} in game {sample_game} ({len(pitcher_game)} pitches)")
    
    # --- Step 4: Verify a specific cumulative feature (e.g., cum_ff_velo) ---
    ff_pitches = pitcher_game[pitcher_game['pitch_type_can'] == 'FF'].copy()
    
    if len(ff_pitches) > 2:
        print("\n🔬 Verifying 'cum_ff_velo':")
        
        # Manually calculate the rolling average
        ff_pitches['manual_cum_velo'] = ff_pitches['release_speed'].shift(1).expanding().mean()
        
        for i, (idx, pitch) in enumerate(ff_pitches.head(5).iterrows()):
            pitch_num = i + 1
            actual_velo = pitch['release_speed']
            feature_velo = pitch['cum_ff_velo']
            manual_velo = pitch['manual_cum_velo']
            
            print(f"\n  Fastball #{pitch_num}:")
            print(f"    - Actual Velo          : {actual_velo:.2f}")
            print(f"    - Feature 'cum_ff_velo': {feature_velo:.2f}")
            
            # For the first pitch, the cumulative value should be NaN
            if pitch_num == 1:
                if pd.isna(feature_velo):
                    print("    - ✅ Correct: Cumulative velocity is NaN for the first pitch.")
                else:
                    print(f"    - 🚨 INCORRECT: Cumulative velocity should be NaN, but is {feature_velo:.2f}")
            else:
                 # Check if the feature value matches the manual calculation
                if not pd.isna(manual_velo) and abs(feature_velo - manual_velo) < 0.01:
                    print(f"    - ✅ Correct: Feature value matches manual rolling average ({manual_velo:.2f}).")
                else:
                    print(f"    - 🚨 INCORRECT: Feature value {feature_velo:.2f} does not match manual calc {manual_velo:.2f}")
    else:
        print("Skipping FF verification, not enough fastballs in this sample.")

    # --- Step 5: Verify a cumulative count feature (e.g., cum_sl_count) ---
    sl_pitches = pitcher_game[pitcher_game['pitch_type_can'] == 'SL'].copy()

    if len(sl_pitches) > 2:
        print("\n🔬 Verifying 'cum_sl_count':")
        for i, (idx, pitch) in enumerate(sl_pitches.head(5).iterrows()):
            pitch_num = i + 1
            feature_count = pitch['cum_sl_count']
            
            print(f"\n  Slider #{pitch_num}:")
            print(f"    - Feature 'cum_sl_count': {feature_count}")
            
            # The count should be equal to the pitch number minus 1
            if feature_count == pitch_num - 1:
                print(f"    - ✅ Correct: Count is {pitch_num - 1} as expected.")
            else:
                print(f"    - 🚨 INCORRECT: Expected count {pitch_num - 1}, but feature is {feature_count}")

    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    verify_cumulative_calculation() 