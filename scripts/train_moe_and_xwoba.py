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
       LAG(pitch_type_can,1) OVER w AS prev_pitch_1,
       LAG(pitch_type_can,2) OVER w AS prev_pitch_2,
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
    df["prev_pitch_1"] = df["prev_pitch_1"].fillna("NONE")
    
    return df


def train_moe_models(df):
    """Train per-pitcher MoE models."""
    print("💪 Training MoE models...")
    
    # Required columns for MoE
    moe_feats = ["count_state", "prev_pitch_1", "balls", "strikes", "stand", "inning_topbot"]
    
    pitchers = df[df.pitch_count_train >= 400].pitcher.unique()
    
    for pid in tqdm(pitchers, desc="Fitting MoE heads"):
        df_p = df[df.pitcher == pid].copy()
        
        X = df_p[moe_feats]
        enc = LabelEncoder().fit(df_p[TARGET_PT])
        y = enc.transform(df_p[TARGET_PT])
        
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(enc.classes_),
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=64,
            n_jobs=1,
            seed=42,
        )
        model.fit(X, y)
        model.booster_.save_model(str(MOE_DIR / f"{pid}.lgb"))


def train_xwoba_models(df):
    """Train pitch-type specific xwOBA regressors."""
    print("💪 Training xwOBA models...")
    
    # Check for leakage
    cols = list(df.columns)
    leaky_cols = [c for c in cols if any(tok in c for tok in LEAK_TOKENS)]
    if leaky_cols:
        raise ValueError(f"Leakage columns found in xwOBA features: {leaky_cols}")

    # Features for xwOBA models
    xwoba_feats = [c for c in df.columns if c.endswith("_30d") or c.endswith("_7d")]
    xwoba_feats += ["balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b"]
    
    df_bip = df[df.estimated_woba_using_speedangle.notna()].copy()

    for pt in tqdm(PITCH_TYPES, desc="Fitting xwOBA heads"):
        df_pt = df_bip[df_bip.pitch_type_can == pt]
        X = df_pt[xwoba_feats]
        y = df_pt.estimated_woba_using_speedangle

        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=128,
            n_jobs=-1,
            seed=42,
        )
        model.fit(X, y)
        model.booster_.save_model(str(XWOBA_DIR / f"{pt}.lgb"))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MoE and xwOBA models.")
    parser.add_argument("--train-years", nargs="+", type=int, required=True,
                      help="Years to train on")
    args = parser.parse_args()
    
    df = load_data(args.train_years)
    
    # Get pitcher pitch counts for MoE eligibility
    pitch_counts = df.groupby('pitcher')[TARGET_PT].count().rename("pitch_count_train").reset_index()
    df = df.merge(pitch_counts, on="pitcher")
    
    train_moe_models(df)
    train_xwoba_models(df)
    
    print("✅ MoE and xwOBA models trained successfully.")


if __name__ == "__main__":
    main()
