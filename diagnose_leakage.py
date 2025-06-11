#!/usr/bin/env python3
"""
diagnose_leakage.py
==================
Diagnose potential data leakage in the GPU-trained model.
"""

import pickle
import pathlib
import pandas as pd
import numpy as np
from run_full_pipeline import load_parquets, prep_balanced

def diagnose_data_leakage():
    """Check for potential data leakage in model features."""
    
    print("🔍 MLB Model Data Leakage Diagnostic")
    print("=" * 60)
    
    # Load encoders
    checkpoint_dir = pathlib.Path("models/checkpoint_gpu_20250605_112919")
    with open(checkpoint_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(checkpoint_dir / "target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    
    # Load a small sample of test data to inspect features
    print("📊 Loading sample test data...")
    test_years = [2024]
    test_range = "2024-08-01:2024-08-31"  # Just August 2024
    
    test_df = load_parquets(test_years, test_range)
    print(f"Raw test data shape: {test_df.shape}")
    print(f"Raw columns: {len(test_df.columns)}")
    
    # Show all raw columns
    print(f"\n📋 ALL RAW COLUMNS ({len(test_df.columns)}):")
    for i, col in enumerate(sorted(test_df.columns)):
        print(f"  {i+1:3d}. {col}")
    
    # Prepare data (same as model training)
    X_te, y_te, _, _ = prep_balanced(test_df, label_encoders)
    print(f"\nProcessed data shape: X={X_te.shape}, y={y_te.shape}")
    print(f"Features used by model: {len(X_te.columns)}")
    
    # Categorize features used by the model
    print(f"\n🔬 FEATURES USED BY MODEL ({len(X_te.columns)}):")
    
    # Suspicious features (potential leakage)
    suspicious = []
    situational = []
    historical = []
    lag_features = []
    
    for col in X_te.columns:
        col_lower = col.lower()
        
        # Lag features (previous pitch info)
        if any(x in col_lower for x in ["prev_", "lag_", "dvelo"]):
            lag_features.append(col)
        
        # Historical/aggregate features
        elif any(x in col_lower for x in ["avg_", "mean_", "median_", "std_", "count_", "pct_", "rate_", "ratio_"]):
            historical.append(col)
        
        # Situational features
        elif any(x in col_lower for x in ["balls", "strikes", "outs", "inning", "count_", "score", "runner"]):
            situational.append(col)
        
        # Potentially suspicious features
        elif any(x in col_lower for x in ["release", "pfx", "plate", "vx0", "vy0", "vz0", "ax", "ay", "az", "spin", "zone", "launch", "hit"]):
            suspicious.append(col)
        
        # Everything else
        else:
            historical.append(col)
    
    print(f"\n🚨 SUSPICIOUS FEATURES (potential leakage): {len(suspicious)}")
    for feat in suspicious:
        print(f"     ❌ {feat}")
    
    print(f"\n📊 SITUATIONAL FEATURES: {len(situational)}")
    for feat in sorted(situational):
        print(f"     ✅ {feat}")
    
    print(f"\n🔄 LAG FEATURES: {len(lag_features)}")
    for feat in sorted(lag_features):
        print(f"     🔄 {feat}")
    
    print(f"\n📈 HISTORICAL/AGGREGATE FEATURES: {len(historical)}")
    for feat in sorted(historical):
        print(f"     📈 {feat}")
    
    # Check if suspicious features have high importance
    if suspicious:
        print(f"\n🚨 POTENTIAL DATA LEAKAGE DETECTED!")
        print(f"Found {len(suspicious)} suspicious features that may contain current pitch information.")
        print("This could explain the unusually high accuracy.")
        
        # Sample some values to check
        print(f"\n📊 Sample values from suspicious features:")
        sample_df = test_df.head(5)
        for feat in suspicious[:5]:  # Show first 5 suspicious features
            if feat in sample_df.columns:
                print(f"  {feat}: {sample_df[feat].values}")
    
    # Check lag features for reasonableness
    print(f"\n🔍 Checking lag features...")
    sample_size = min(1000, len(test_df))
    sample_df = test_df.head(sample_size)
    
    # Check prev_pt1 and prev_pt2 distribution
    if 'prev_pt1' in test_df.columns:
        print(f"\nprev_pt1 distribution (first {sample_size} rows):")
        print(test_df['prev_pt1'].head(sample_size).value_counts().head(10))
    
    if 'prev_pt2' in test_df.columns:
        print(f"\nprev_pt2 distribution (first {sample_size} rows):")
        print(test_df['prev_pt2'].head(sample_size).value_counts().head(10))
    
    # Check dvelo1 (velocity difference) 
    if 'dvelo1' in test_df.columns:
        dvelo_stats = test_df['dvelo1'].head(sample_size).describe()
        print(f"\ndvelo1 stats (first {sample_size} rows):")
        print(dvelo_stats)
        
        # Check for unrealistic values
        if dvelo_stats['std'] > 20:  # Very high variance might indicate leakage
            print("🚨 WARNING: dvelo1 has unusually high variance - possible leakage!")
    
    return X_te.columns.tolist(), suspicious

if __name__ == "__main__":
    features_used, suspicious_features = diagnose_data_leakage() 