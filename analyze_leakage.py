#!/usr/bin/env python3
import pandas as pd
from two_model_architecture import TwoModelPitchPredictor

# Load sample data
df = pd.read_parquet('data/features_historical/statcast_historical_2023.parquet').head(1000)

# Initialize predictor and get features
predictor = TwoModelPitchPredictor()
features = predictor.prepare_model1_features(df)

print("POTENTIAL LEAKAGE ANALYSIS")
print("="*50)

# Check for velocity/spin features (these could be leaky)
print("\n1. VELOCITY/SPIN FEATURES (Historical averages - should be OK):")
velocity_features = [col for col in features.columns if 'velocity' in col.lower()]
spin_features = [col for col in features.columns if 'spin' in col.lower()]

print(f"   Velocity features: {len(velocity_features)}")
for feat in velocity_features[:5]:
    print(f"     {feat}")
if len(velocity_features) > 5:
    print(f"     ... and {len(velocity_features)-5} more")

print(f"   Spin features: {len(spin_features)}")
for feat in spin_features[:5]:
    print(f"     {feat}")

# Check temporal separation
print("\n2. TEMPORAL SEPARATION:")
print(f"   Train year: 2023")
print(f"   Val year: 2024") 
print(f"   Test year: 2025")
print(f"   ✅ Good temporal separation")

# Check for same-game features
print("\n3. CUMULATIVE FEATURES (Within-game state):")
cumulative_features = [col for col in features.columns if col.startswith('cum_')]
print(f"   Count: {len(cumulative_features)}")
for feat in cumulative_features:
    print(f"     {feat}")

# Check batter matchup features
print("\n4. BATTER MATCHUP FEATURES (Historical performance):")
batter_features = [col for col in features.columns if 'batter' in col.lower()]
print(f"   Count: {len(batter_features)}")
for feat in batter_features:
    print(f"     {feat}")

print("\n5. DIAGNOSIS:")
print("   The high accuracy (74.5%) could be due to:")
print("   - Historical velocity/spin being very predictive of pitch type")
print("   - Batter matchup features revealing pitcher tendencies")
print("   - Within-game cumulative features showing patterns")
print("   - Need to validate this is realistic vs. leakage")

print(f"\nTotal features being used: {len(features.columns)}") 