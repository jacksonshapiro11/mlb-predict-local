#!/usr/bin/env python3
"""
verify_td_leakage.py
===================
Verify that TD (Today) features include current pitch data - causing leakage.
"""

import pandas as pd
import numpy as np
from run_full_pipeline import load_parquets

def verify_td_leakage():
    """Verify TD features contain current pitch information."""
    
    print("🔍 VERIFYING TD FEATURE LEAKAGE")
    print("=" * 50)
    
    # Load a small sample to examine TD features
    print("📊 Loading sample data...")
    test_years = [2024]
    test_range = "2024-08-01:2024-08-03"  # Just 3 days
    
    df = load_parquets(test_years, test_range)
    print(f"Sample shape: {df.shape}")
    
    # Focus on one pitcher and one game for clear analysis
    sample_pitcher = df['pitcher'].iloc[0]
    sample_game = df['game_pk'].iloc[0]
    
    print(f"\n🎯 Analyzing pitcher {sample_pitcher} in game {sample_game}")
    
    # Get all pitches from this pitcher in this game
    pitcher_game = df[
        (df['pitcher'] == sample_pitcher) & 
        (df['game_pk'] == sample_game)
    ].copy()
    
    if len(pitcher_game) < 2:
        print("⚠️  Not enough pitches to analyze, trying another sample...")
        # Try another combination
        for i in range(10):
            sample_pitcher = df['pitcher'].iloc[i*100]
            sample_game = df['game_pk'].iloc[i*100]
            pitcher_game = df[
                (df['pitcher'] == sample_pitcher) & 
                (df['game_pk'] == sample_game)
            ].copy()
            if len(pitcher_game) >= 3:
                break
    
    if len(pitcher_game) < 2:
        print("❌ Could not find suitable sample")
        return
    
    # Sort by pitch sequence
    pitcher_game = pitcher_game.sort_values(['at_bat_number', 'pitch_number']).reset_index(drop=True)
    
    print(f"Found {len(pitcher_game)} pitches from this pitcher in this game")
    print(f"Game date: {pitcher_game['game_date'].iloc[0]}")
    
    # Focus on fastballs to check SPIN_TD_FF
    ff_pitches = pitcher_game[pitcher_game['pitch_type_can'] == 'FF'].copy()
    
    if len(ff_pitches) >= 2:
        print(f"\n🔬 ANALYZING {len(ff_pitches)} FASTBALLS:")
        
        for i, (idx, pitch) in enumerate(ff_pitches.iterrows()):
            print(f"\nPitch {i+1}:")
            print(f"  Actual release_speed: {pitch['release_speed']:.1f} mph")
            print(f"  Actual release_spin_rate: {pitch['release_spin_rate']:.0f} rpm") 
            print(f"  SPIN_TD_FF feature: {pitch['SPIN_TD_FF']:.1f} rpm")
            print(f"  V_TD_FF feature: {pitch['V_TD_FF']:.1f} mph")
            
            # Check if TD feature matches current pitch
            if abs(pitch['release_spin_rate'] - pitch['SPIN_TD_FF']) < 1:
                print(f"  🚨 LEAKAGE: SPIN_TD_FF exactly matches current pitch spin!")
            elif abs(pitch['release_speed'] - pitch['V_TD_FF']) < 0.1:
                print(f"  🚨 LEAKAGE: V_TD_FF exactly matches current pitch velocity!")
    
    # Check if TD features change within the same game
    print(f"\n📈 TD FEATURE EVOLUTION WITHIN GAME:")
    
    # Get all pitch types thrown
    pitch_types = pitcher_game['pitch_type_can'].unique()
    
    for pt in pitch_types[:3]:  # Check first 3 pitch types
        pt_pitches = pitcher_game[pitcher_game['pitch_type_can'] == pt]
        if len(pt_pitches) >= 2:
            td_col = f'SPIN_TD_{pt}'
            v_td_col = f'V_TD_{pt}'
            
            if td_col in pitcher_game.columns:
                print(f"\n{pt} pitches:")
                for i, (idx, pitch) in enumerate(pt_pitches.iterrows()):
                    actual_spin = pitch.get('release_spin_rate', 'N/A')
                    td_spin = pitch.get(td_col, 'N/A')
                    print(f"  Pitch {i+1}: actual={actual_spin}, TD_feature={td_spin}")
                
                # Check if TD value changes (indicating it includes current pitch)
                td_values = pt_pitches[td_col].dropna()
                if len(td_values) > 1 and td_values.nunique() > 1:
                    print(f"  🚨 {td_col} values change within game: {td_values.tolist()}")
                    print(f"     This suggests TD feature includes current pitch data!")
                elif len(td_values) > 1:
                    print(f"  ✅ {td_col} constant within game: {td_values.iloc[0]:.1f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"If TD features change within a single game, they're including current pitch data.")
    print(f"If TD features are constant within a game, they're calculated up to (but excluding) current pitch.")
    
    return pitcher_game

if __name__ == "__main__":
    sample_data = verify_td_leakage() 