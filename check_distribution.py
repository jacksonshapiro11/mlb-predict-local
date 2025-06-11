import pandas as pd
import numpy as np
from collections import Counter

def analyze_dataset(df, name):
    print(f"\n=== {name} Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    
    # Class distribution
    class_counts = df['pitch_type_can'].value_counts()
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check for any potential data leakage
    print("\nUnique pitchers:", df['pitcher'].nunique())
    print("Unique batters:", df['batter'].nunique())
    print("Date range:", df['game_date'].min(), "to", df['game_date'].max())

# Load the data
train_df = pd.read_parquet('data/features/statcast_2023.parquet')
val_df = pd.read_parquet('data/features/statcast_2024.parquet')
val_df = val_df[(val_df['game_date'] >= '2024-03-15') & (val_df['game_date'] <= '2024-03-31')]
test_df = pd.read_parquet('data/features/statcast_2024.parquet')
test_df = test_df[(test_df['game_date'] >= '2024-04-01') & (test_df['game_date'] <= '2024-04-15')]

# Analyze each dataset
analyze_dataset(train_df, "Training")
analyze_dataset(val_df, "Validation")
analyze_dataset(test_df, "Test")

# Check for overlap between sets
print("\n=== Checking for Data Leakage ===")
train_pitchers = set(train_df['pitcher'].unique())
val_pitchers = set(val_df['pitcher'].unique())
test_pitchers = set(test_df['pitcher'].unique())

print(f"Pitchers in both train and val: {len(train_pitchers.intersection(val_pitchers))}")
print(f"Pitchers in both train and test: {len(train_pitchers.intersection(test_pitchers))}")
print(f"Pitchers in both val and test: {len(val_pitchers.intersection(test_pitchers))}") 