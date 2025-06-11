#!/usr/bin/env python
"""
train_cumulative.py
===================
Train a model using the new, verified cumulative features.
"""

import argparse
import pathlib
import time
import warnings
import pickle
from datetime import date
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# --- CONFIG ---
CUMULATIVE_FEATURE_DIR = pathlib.Path("data/features_cumulative")
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PT = "pitch_type_can"
CAT_COLS = [
    "stand",
    "p_throws",
    "count_state",
    "prev_pt1",
    "prev_pt2",
]
DECAY_DEFAULT = 0.0008

def load_cumulative_parquets(years, date_range: str | None = None):
    paths = [str(CUMULATIVE_FEATURE_DIR / f"statcast_cumulative_{y}.parquet") for y in years]
    path_expr = "[" + ",".join([f"'{p}'" for p in paths]) + "]"
    where_clause = ""
    if date_range:
        start, end = date_range.split(":")
        where_clause = f"WHERE game_date BETWEEN DATE '{start}' AND DATE '{end}'"
    
    # We already have lag features in the cumulative files, so the query is simpler
    query = f"SELECT * FROM parquet_scan({path_expr}) {where_clause}"
    print(f"🗄️  DuckDB query: {query[:120]}…")
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df

def add_temporal_weight(df, latest_date, lam):
    delta = (latest_date - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-lam * delta)
    return df

def prep_cumulative_data(df: pd.DataFrame, label_encoders: dict = None):
    df = df.copy()
    
    # Create count_state feature
    balls_cap = df["balls"].fillna(0).clip(0, 3)
    strikes_cap = df["strikes"].fillna(0).clip(0, 2)
    df["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)

    # Define columns to drop
    drop_cols = {
        TARGET_PT, "game_date", "game_pk", "at_bat_number", "pitch_number",
        "pitcher", "batter", "home_team", "away_team", "events", "description",
        "pitch_type", "player_name", "des", "w", "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "babip_value", "iso_value", "launch_speed", "launch_angle",
        "release_speed", "release_spin_rate", 
    }
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode categoricals
    if label_encoders is None:
        label_encoders = {}
    for c in CAT_COLS:
        if c not in X.columns:
            continue
        if c not in label_encoders:
            le = LabelEncoder()
            le.fit(X[c].fillna("__MISSING__").astype(str))
            label_encoders[c] = le
        X[c] = label_encoders[c].transform(X[c].fillna("__MISSING__").astype(str))

    # Convert all other columns to numeric, coercing errors
    for col in X.columns:
        if col not in CAT_COLS:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fill NaNs
    X = X.fillna(-1) 
    y = df[TARGET_PT]
    w = df.get("w", pd.Series(1, index=df.index))
    
    return X, y, w, label_encoders

def train_catboost_cumulative(X_tr, y_tr, w_tr, X_val, y_val):
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in CAT_COLS]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        learning_rate=0.05,
        depth=8,
        iterations=2000,
        random_state=42,
        od_type="Iter",
        od_wait=150,
        verbose=250,
        task_type="GPU",
        devices="0",
    )
    model.fit(
        Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        use_best_model=True,
    )
    return model

def main(args):
    # --- Data Loading ---
    print("\n📊 Loading cumulative data...")
    train_years = [int(y) for y in args.train_years]
    val_range = args.val
    test_range = args.test

    val_start, val_end = val_range.split(":")
    test_start, test_end = test_range.split(":")
    val_years = {int(val_start[:4])}
    test_years = {int(test_start[:4]), int(test_end[:4])}
    
    # We only have 2023 cumulative data, so for this test run, we'll use it for training,
    # and use the val/test ranges from 2023 as well. This is NOT a real training scenario,
    # but it will validate the pipeline with the new features.
    print("⚠️  WARNING: Using 2023 data for train/val/test to validate the pipeline.")
    train_df = load_cumulative_parquets([2023], "2023-03-01:2023-07-31")
    val_df = load_cumulative_parquets([2023], "2023-08-01:2023-08-31")
    test_df = load_cumulative_parquets([2023], "2023-09-01:2023-10-01")

    train_df = add_temporal_weight(train_df, pd.to_datetime("2023-07-31"), args.decay)

    # --- Preprocessing ---
    print("\n🔄 Preprocessing data...")
    X_tr, y_tr, w_tr, enc = prep_cumulative_data(train_df)
    X_val, y_val, _, _ = prep_cumulative_data(val_df, enc)
    X_te, y_te, _, _ = prep_cumulative_data(test_df, enc)

    # Encode target
    pt_enc = LabelEncoder()
    y_tr_enc = pt_enc.fit_transform(y_tr)
    y_val_enc = pt_enc.transform(y_val)
    y_te_enc = pt_enc.transform(y_te)
    print(f"\nTarget classes: {pt_enc.classes_}")

    # --- Training ---
    print("\n🚂 Training CatBoost on cumulative features...")
    model = train_catboost_cumulative(X_tr, y_tr_enc, w_tr, X_val, y_val_enc)
    
    # --- Evaluation ---
    print("\n🎯 Evaluating on test set...")
    preds_proba = model.predict_proba(X_te)
    preds = preds_proba.argmax(1)
    
    acc_te = accuracy_score(y_te_enc, preds)
    ll_te = log_loss(y_te_enc, preds_proba)
    top3_te = np.mean([y_te_enc[i] in np.argsort(preds_proba[i])[-3:] for i in range(len(y_te_enc))])

    print("\n✅ FINAL TEST RESULTS (CUMULATIVE FEATURES)")
    print(f"  Accuracy : {acc_te:.4f}")
    print(f"  Top-3 Acc: {top3_te:.4f}")
    print(f"  Log Loss : {ll_te:.4f}")
    
    # --- Save Model ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = MODEL_DIR / f"checkpoint_cumulative_{ts}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / "catboost_cumulative.cbm"
    model.save_model(model_path)
    
    with open(checkpoint_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(checkpoint_dir / "target_encoder.pkl", "wb") as f:
        pickle.dump(pt_enc, f)
        
    print(f"\n💾 Model and encoders saved to {checkpoint_dir.resolve()}")
    print("🚀 Pipeline finished!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    st = p.add_argument_group("train")
    st.add_argument("--train-years", nargs="+", default=["2023"])
    st.add_argument("--val", type=str, default="2023-08-01:2023-08-31")
    st.add_argument("--test", type=str, default="2023-09-01:2023-10-01")
    st.add_argument("--decay", type=float, default=DECAY_DEFAULT)
    args = p.parse_args()
    main(args) 