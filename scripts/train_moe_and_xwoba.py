#!/usr/bin/env python
"""
scripts/train_moe_and_xwoba.py
=============================
Train per-pitcher residual MoE and nine pitch-type xwOBA regressors.
Called automatically from run_full_pipeline.py if models are absent.
"""

import os
import pathlib
import argparse
import json
import warnings
import pandas as pd
import numpy as np
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

# Create directories
MOE_DIR.mkdir(parents=True, exist_ok=True)
XWOBA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PT = "pitch_type_can"
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "FS", "KC", "OTHER"]

# Fast leakage detection tokens
LEAK_TOKENS = [
    "release_", "pfx_", "plate_", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "launch_speed", "launch_angle", "hit_distance", "zone", "delta_"
]

LAG_SQL = """
WITH base AS (
  SELECT * FROM parquet_scan({paths})
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


def load_data(years):
    """Load training data with lag features."""
    paths = [str(PARQUET_DIR / f"statcast_historical_{y}.parquet") for y in years]
    path_expr = "[" + ",".join([f"'{p}'" for p in paths]) + "]"
    query = LAG_SQL.format(paths=path_expr)
    
    print(f"🗄️  Loading data for years: {years}")
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    
    # Prepare features
    balls_cap = df["balls"].fillna(0).clip(0, 3)
    strikes_cap = df["strikes"].fillna(0).clip(0, 2)
    df["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)
    df["prev_pt1"] = df["prev_pt1"].fillna("NONE")
    
    return df


def train_moe_models(df):
    """Train per-pitcher MoE models."""
    print("\n🎯 Training Mixture-of-Experts models...")
    
    # Label encoder for pitch types
    pt_enc = LabelEncoder().fit(PITCH_TYPES)
    
    moe_manifest = {}
    moe_features = ["count_state", "prev_pt1", "balls", "strikes", "stand", "inning_topbot"]
    
    for pid, grp in tqdm(df.groupby("pitcher"), desc="Training pitcher models"):
        if len(grp) < 400:  # Skip low-sample pitchers
            continue
            
        # Prepare features
        X = grp[moe_features].copy()
        X["prev_pt1"] = X["prev_pt1"].fillna("NONE")
        
        # Target
        y_valid = grp["pitch_type_can"].fillna("OTHER")
        valid_mask = y_valid.isin(PITCH_TYPES)
        
        if valid_mask.sum() < 50:
            continue
            
        X = X[valid_mask]
        y = pt_enc.transform(y_valid[valid_mask])
        
        # Train LightGBM
        params = {
            "objective": "multiclass",
            "num_class": len(PITCH_TYPES),
            "num_leaves": 64,
            "learning_rate": 0.1,
            "device_type": "gpu" if USE_GPU else "cpu",
            "verbosity": -1,
            "random_state": 42
        }
        
        lgb_train = lgb.Dataset(X, y)
        model = lgb.train(params, lgb_train, num_boost_round=200)
        
        # Save model
        model.save_model(str(MOE_DIR / f"{pid}.lgb"))
        moe_manifest[int(pid)] = len(grp)
    
    print(f"✅ Trained {len(moe_manifest)} MoE models")
    return moe_manifest


def train_xwoba_models(df):
    """Train pitch-type specific xwOBA regressors."""
    print("\n🎯 Training xwOBA outcome models...")
    
    xwoba_stats = {}
    
    for pt in tqdm(PITCH_TYPES, desc="Training xwOBA models"):
        # Filter to specific pitch type
        grp = df[df["pitch_type_can"] == pt].copy()
        
        # Target: xwOBA (drop missing values)
        trg = grp["estimated_woba_using_speedangle"].dropna()
        if len(trg) < 500:  # Skip small samples
            print(f"⚠️  Skipping {pt}: only {len(trg)} valid xwOBA samples")
            continue
        
        # Features: Remove leakage columns fast
        X = grp.loc[trg.index].drop(columns=[
            "estimated_woba_using_speedangle", "pitch_type_can"
        ], errors="ignore")
        
        # Fast leakage filter
        X = X[[c for c in X.columns if not any(t in c.lower() for t in LEAK_TOKENS)]]
        
        # Handle categorical columns
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].astype("category")
        
        # Fill missing values
        X = X.fillna(0)
        
        # Train LightGBM regressor
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 128,
            "learning_rate": 0.05,
            "device_type": "gpu" if USE_GPU else "cpu",
            "verbosity": -1,
            "random_state": 42
        }
        
        reg_ds = lgb.Dataset(X, trg)
        model = lgb.train(params, reg_ds, num_boost_round=400)
        
        # Save model
        model.save_model(str(XWOBA_DIR / f"{pt}.lgb"))
        
        xwoba_stats[pt] = {
            "samples": len(grp),
            "valid_samples": len(X),
            "mean_xwoba": float(trg.mean()),
            "features": len(X.columns)
        }
    
    print(f"✅ Trained {len(xwoba_stats)} xwOBA models")
    return xwoba_stats


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", nargs="+", required=True, type=int)
    args = parser.parse_args()
    
    print("🚀 Training MoE and xwOBA models...")
    print(f"📅 Years: {args.train_years}")
    
    # Load data
    df = load_data(args.train_years)
    print(f"📊 Training data: {len(df):,} pitches")
    
    # Train models
    moe_manifest = train_moe_models(df)
    xwoba_stats = train_xwoba_models(df)
    
    # Create combined manifest
    manifest = {
        "created": pd.Timestamp.now().isoformat(),
        "train_years": args.train_years,
        "total_pitches": len(df),
        "use_gpu": USE_GPU,
        "moe": {
            "total_pitchers": len(moe_manifest),
            "pitcher_models": moe_manifest
        },
        "xwoba": {
            "pitch_models": xwoba_stats
        }
    }
    
    # Save manifest
    manifest_path = MODEL_DIR / "pitcher_moe_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ MoE/xwOBA training complete!")
    print(f"📄 Manifest saved: {manifest_path}")
    print(f"🎯 MoE models: {len(moe_manifest)}")
    print(f"🎯 xwOBA models: {len(xwoba_stats)}")


if __name__ == "__main__":
    main()
