import pandas as pd
import numpy as np

# Load 2024 data
df_2024 = pd.read_parquet('data/features/statcast_2024.parquet')

print("=== 2024 Data Overview ===")
print(f"Total samples: {len(df_2024)}")
print(f"Date range: {df_2024['game_date'].min()} to {df_2024['game_date'].max()}")

# Analyze by month
df_2024['month'] = pd.to_datetime(df_2024['game_date']).dt.month
monthly_stats = df_2024.groupby('month').agg({
    'pitch_type_can': 'count',
    'pitcher': 'nunique'
}).reset_index()

print("\n=== Monthly Distribution ===")
for _, row in monthly_stats.iterrows():
    print(f"Month {row['month']}: {row['pitch_type_can']} pitches, {row['pitcher']} pitchers")

# Analyze 'OTHER' class
other_pitches = df_2024[df_2024['pitch_type_can'] == 'OTHER']
print("\n=== 'OTHER' Class Analysis ===")
print(f"Total 'OTHER' pitches: {len(other_pitches)}")
print("\n'OTHER' pitches by month:")
other_by_month = other_pitches.groupby('month').size()
for month, count in other_by_month.items():
    print(f"Month {month}: {count} pitches ({count/len(other_pitches)*100:.1f}%)")

# Check the first few rows of 'OTHER' pitches
print("\n=== Sample 'OTHER' Pitches ===")
print(other_pitches[['game_date', 'pitcher', 'pitch_type', 'pitch_type_can']].head()) 