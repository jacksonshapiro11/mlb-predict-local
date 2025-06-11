#!/usr/bin/env python3
"""
scripts/train_family_head.py
============================
Train a 3-class pitch family model (FB/BR/OS) using LightGBM.

This creates a lightweight family classifier that provides strategic-level 
pitch approach probabilities as features for the main 10-class model.

Family Mapping:
- FB (Fastball family): FF, SI, FC 
- BR (Breaking ball family): SL, CU, KC, OTHER
- OS (Off-speed family): CH, FS, ST

Usage:
    python scripts/train_family_head.py --train-years 2023 2024
    python scripts/train_family_head.py --train-years 2019 2020 2021 2022 2023 --toy
"""

import argparse
import sys
import pathlib
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from run_full_pipeline import load_parquets, prep_balanced, CAT_COLS

# --- Config ---
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def map_pitch_to_family(pitch_type_can):
    """Map canonical pitch types to 3 families."""
    family_mapping = {
        # Fastball family (FB)
        "FF": "FB",  # Four-seam fastball
        "SI": "FB",  # Sinker
        "FC": "FB",  # Cutter
        # Breaking ball family (BR)
        "SL": "BR",  # Slider
        "CU": "BR",  # Curveball
        "KC": "BR",  # Knuckle curve
        "OTHER": "BR",  # Other breaking balls
        # Off-speed family (OS)
        "CH": "OS",  # Changeup
        "FS": "OS",  # Splitter
        "ST": "OS",  # Sweeper (modern off-speed slider)
    }
    return family_mapping.get(pitch_type_can, "BR")  # Default to breaking ball


def train_family_head(train_years, toy_mode=False):
    """Train family head model on specified years."""
    print(f"🏗️  Training family head model on years: {train_years}")
    if toy_mode:
        print("🧪 TOY MODE: Using 400 iterations")

    # Load training data
    print("⏳ Loading training data...")
    df = load_parquets(train_years)

    # Create family target
    df["pitch_family"] = df["pitch_type_can"].apply(map_pitch_to_family)

    # Sample for toy mode
    if toy_mode:
        sample_size = min(50000, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"🎲 Sampled {sample_size} pitches for toy training")

    print(f"📊 Training data shape: {df.shape}")
    print(f"📈 Family distribution:")
    print(df["pitch_family"].value_counts().sort_index())

    # Prepare features using existing prep pipeline
    # Temporarily set target to family for prep_balanced
    original_target = df.get("pitch_type_can", pd.Series())
    df["pitch_type_can"] = df["pitch_family"]  # Temporarily use family as target

    X, y, w, label_encoders = prep_balanced(df)

    # Get family encoder from the prep process
    family_encoder = label_encoders["target"]

    print(f"✅ Preprocessed data: {X.shape}")
    print(f"📋 Family classes: {family_encoder.classes_}")

    # Train LightGBM family model
    print("🚀 Training LightGBM family classifier...")
    max_iters = 400 if toy_mode else 1000

    params = {
        "objective": "multiclass",
        "num_class": len(family_encoder.classes_),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": max_iters,
        "learning_rate": 0.1,
        "num_leaves": 32,
        "max_depth": 6,
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
    }

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    w_train, w_val = w[:split_idx], w[split_idx:]

    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        weight=w_train,
        categorical_feature=[i for i, c in enumerate(X_train.columns) if c in CAT_COLS],
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(X_val, y_val, weight=w_val, reference=lgb_train)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)]

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # Evaluate model
    val_proba = model.predict(X_val)
    val_pred = np.argmax(val_proba, axis=1)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_logloss = log_loss(y_val, val_proba)

    print(f"📊 Validation Results:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Log-Loss: {val_logloss:.4f}")

    # Print per-class accuracy
    for i, family in enumerate(family_encoder.classes_):
        mask = y_val == i
        if mask.sum() > 0:
            class_acc = accuracy_score(y_val[mask], val_pred[mask])
            print(f"  {family} accuracy: {class_acc:.4f} ({mask.sum()} samples)")

    # Save model, encoder, and feature names
    model_path = MODEL_DIR / "fam_head.lgb"
    encoder_path = MODEL_DIR / "fam_encoder.pkl"
    features_path = MODEL_DIR / "fam_features.pkl"

    model.save_model(str(model_path))
    with open(encoder_path, "wb") as f:
        pickle.dump(family_encoder, f)
    with open(features_path, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"💾 Model saved to: {model_path}")
    print(f"💾 Encoder saved to: {encoder_path}")
    print("✅ Family head training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train pitch family head model")
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        required=True,
        help="Years to use for training (e.g., --train-years 2023 2024)",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use toy mode (reduced iterations and sampling)",
    )

    args = parser.parse_args()

    train_family_head(args.train_years, toy_mode=args.toy)


if __name__ == "__main__":
    main()
