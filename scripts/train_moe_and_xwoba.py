#!/usr/bin/env python
"""
scripts/train_moe_and_xwoba.py
=============================
Train Mixture-of-Experts (MoE) and xwOBA outcome models.

1. MoE: Per-pitcher LightGBM models learning residuals from global ensemble
2. xwOBA: Pitch-type specific regressors for expected outcome prediction

USAGE
-----
python scripts/train_moe_and_xwoba.py --train-years 2019 2020 2021 2022 2023
"""

import argparse
import pathlib
import json
import os
import warnings
import pandas as pd
import lightgbm as lgb
import duckdb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# GPU Configuration
USE_GPU = os.getenv("GPU", "0") == "1"
print(f"🖥️  Training MoE/xwOBA on {'GPU' if USE_GPU else 'CPU'}")

# Configuration
PARQUET_DIR = pathlib.Path("data/features_historical")
MODEL_DIR = pathlib.Path("models")
MOE_DIR = MODEL_DIR / "pitcher_moe"
XWOBA_DIR = MODEL_DIR / "xwoba_by_pitch"

TARGET_PT = "pitch_type_can"
MIN_PITCHER_PITCHES = 400

# Pitch types
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "FS", "KC", "OTHER"]

# MoE features (simple situational context)
MOE_FEATURES = ["count_state", "prev_pt1", "balls", "strikes", "stand", "inning_topbot"]

# Current pitch markers (for leakage detection)
CURRENT_PITCH_MARKERS = [
    "release_",
    "pfx_",
    "plate_",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "sz_top",
    "sz_bot",
    "effective_speed",
    "spin_axis",
    "zone",
    "hc_x",
    "hc_y",
    "launch_speed",
    "launch_angle",
    "hit_distance",
    "delta_run_exp",
    "delta_home_win_exp",
]

LAG_SQL = """
WITH base AS (
  SELECT *
  FROM parquet_scan({paths})
)
SELECT *, 
       LAG(pitch_type_can,1) OVER w AS prev_pt1,
       LAG(pitch_type_can,2) OVER w AS prev_pt2,
       release_speed - LAG(release_speed,1) OVER w AS dvelo1
FROM base
WINDOW w AS (
  PARTITION BY pitcher, game_pk
  ORDER BY at_bat_number, pitch_number
)
"""


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def load_duck(query: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df


def load_parquets(years):
    paths = [str(PARQUET_DIR / f"statcast_historical_{y}.parquet") for y in years]
    path_expr = "[" + ",".join([f"'{p}'" for p in paths]) + "]"
    q = LAG_SQL.format(paths=path_expr)
    print(f"🗄️  Loading training data: {years}")
    return load_duck(q)


def prep_features(df):
    """Prepare features with count_state and handle missing values."""
    df = df.copy()

    # Create count_state
    balls_cap = df["balls"].fillna(0).clip(0, 3)
    strikes_cap = df["strikes"].fillna(0).clip(0, 2)
    df["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)

    # Fill missing prev_pt1 with 'NONE'
    df["prev_pt1"] = df["prev_pt1"].fillna("NONE")

    return df


def filter_leakage(df):
    """Remove columns with current pitch markers."""
    drop_cols = []
    for col in df.columns:
        if any(marker in col.lower() for marker in CURRENT_PITCH_MARKERS):
            drop_cols.append(col)

    # Also drop obvious leakage columns
    drop_cols.extend(
        [
            TARGET_PT,
            "estimated_woba_using_speedangle",
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
            "home_team",
            "pitch_name",
            "events",
            "description",
            "pitch_type",
        ]
    )

    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


# --------------------------------------------------------------------------- #
# MOE TRAINING
# --------------------------------------------------------------------------- #
def train_moe_models(df_tr):
    """Train per-pitcher MoE models."""
    print("\n🎯 Training Mixture-of-Experts models...")

    # Prepare target encoder
    pt_encoder = LabelEncoder()
    pt_encoder.fit(PITCH_TYPES)

    # Count pitchers
    pitcher_counts = df_tr["pitcher"].value_counts()
    eligible_pitchers = pitcher_counts[pitcher_counts >= MIN_PITCHER_PITCHES].index

    print(
        f"📊 Found {len(eligible_pitchers)} pitchers with ≥{MIN_PITCHER_PITCHES} pitches"
    )

    moe_stats = {
        "total_pitchers": len(eligible_pitchers),
        "min_pitches": MIN_PITCHER_PITCHES,
        "pitcher_models": {},
    }

    # Train individual pitcher models
    for pid in tqdm(eligible_pitchers, desc="Training pitcher models"):
        df_p = df_tr[df_tr["pitcher"] == pid].copy()

        if len(df_p) < MIN_PITCHER_PITCHES:
            continue

        # Prepare features
        X_moe = df_p[MOE_FEATURES].copy()

        # Handle categorical encoding
        for col in ["count_state", "prev_pt1", "stand", "inning_topbot"]:
            if col in X_moe.columns:
                X_moe[col] = X_moe[col].astype(str)

        # Target
        y_moe = df_p[TARGET_PT].fillna("OTHER")
        valid_mask = y_moe.isin(PITCH_TYPES)

        if valid_mask.sum() < 50:  # Need minimum samples
            continue

        X_moe = X_moe[valid_mask]
        y_moe = y_moe[valid_mask]
        y_moe_enc = pt_encoder.transform(y_moe)

        # Train LightGBM
        params = {
            "objective": "multiclass",
            "num_class": len(PITCH_TYPES),
            "num_leaves": 64,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "random_state": 42,
        }

        if USE_GPU:
            params["device_type"] = "gpu"

        train_data = lgb.Dataset(X_moe, y_moe_enc)
        model = lgb.train(params, train_data, num_boost_round=200)

        # Save model
        model_path = MOE_DIR / f"{pid}.lgb"
        model.save_model(str(model_path))

        moe_stats["pitcher_models"][str(pid)] = {
            "pitches": len(df_p),
            "valid_pitches": len(X_moe),
            "model_path": str(model_path),
        }

    print(f"✅ Trained {len(moe_stats['pitcher_models'])} MoE models")
    return moe_stats


# --------------------------------------------------------------------------- #
# XWOBA TRAINING
# --------------------------------------------------------------------------- #
def train_xwoba_models(df_tr):
    """Train pitch-type specific xwOBA regressors."""
    print("\n🎯 Training xwOBA outcome models...")
    
    xwoba_stats = {
        'pitch_models': {}
    }
    
    for pt in tqdm(PITCH_TYPES, desc="Training xwOBA models"):
        # Filter to specific pitch type
        df_pt = df_tr[df_tr[TARGET_PT] == pt].copy()
        
        if len(df_pt) < 1000:  # Need minimum samples
            print(f"⚠️  Skipping {pt}: only {len(df_pt)} samples")
            continue
        
        # Target: xwOBA
        y_xwoba = df_pt['estimated_woba_using_speedangle']
        valid_mask = ~y_xwoba.isna()
        
        if valid_mask.sum() < 500:
            print(f"⚠️  Skipping {pt}: only {valid_mask.sum()} valid xwOBA samples")
            continue
        
        # Features (remove leakage)
        X_xwoba = filter_leakage(df_pt[valid_mask])
        y_xwoba = y_xwoba[valid_mask]
        
        # Handle categorical columns
        for col in X_xwoba.columns:
            if X_xwoba[col].dtype == 'object':
                X_xwoba[col] = X_xwoba[col].astype('category')
        
        # Fill missing values
        X_xwoba = X_xwoba.fillna(0)
        
        # Train LightGBM regressor
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 128,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'random_state': 42
        }
        
        if USE_GPU:
            params['device_type'] = 'gpu'
        
        train_data = lgb.Dataset(X_xwoba, y_xwoba)
        model = lgb.train(params, train_data, num_boost_round=400)
        
        # Save model
        model_path = XWOBA_DIR / f"{pt}.lgb"
        model.save_model(str(model_path))
        
        xwoba_stats['pitch_models'][pt] = {
            'samples': len(df_pt),
            'valid_samples': len(X_xwoba),
            'features': list(X_xwoba.columns),
            'mean_xwoba': float(y_xwoba.mean()),
            'model_path': str(model_path)
        }
    
    print(f"✅ Trained {len(xwoba_stats['pitch_models'])} xwOBA models")
    return xwoba_stats


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-years', nargs='+', required=True, type=int)
    args = parser.parse_args()
    
    print("🚀 Training MoE and xwOBA models...")
    print(f"📅 Years: {args.train_years}")
    
    # Create directories
    MOE_DIR.mkdir(parents=True, exist_ok=True)
    XWOBA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_tr = load_parquets(args.train_years)
    df_tr = prep_features(df_tr)
    
    print(f"📊 Training data: {len(df_tr):,} pitches")
    
    # Train models
    moe_stats = train_moe_models(df_tr)
    xwoba_stats = train_xwoba_models(df_tr)
    
    # Combine stats
    manifest = {
        'created': pd.Timestamp.now().isoformat(),
        'train_years': args.train_years,
        'total_pitches': len(df_tr),
        'use_gpu': USE_GPU,
        'moe': moe_stats,
        'xwoba': xwoba_stats
    }
    
    # Save manifest
    manifest_path = MODEL_DIR / "pitcher_moe_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("✅ MoE/xwOBA training complete!")
    print(f"📄 Manifest saved: {manifest_path}")
    print(f"🎯 MoE models: {len(moe_stats['pitcher_models'])}")
    print(f"🎯 xwOBA models: {len(xwoba_stats['pitch_models'])}")


if __name__ == "__main__":
    main()
