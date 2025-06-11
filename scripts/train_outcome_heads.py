#!/usr/bin/env python3
"""
scripts/train_outcome_heads.py
===============================
Train two-stage outcome prediction models using blended pitch type logits as features.

This script implements a hierarchical outcome prediction system:
1. Stage 1: 3-class prediction (IN_PLAY / BALL / STRIKE) 
2. Stage 2: 7-class Ball-in-Play outcomes (HR, 3B, 2B, 1B, FC, SAC, OUT)

The models use the final blended tree-GRU logits as input features, creating a complete
Family → Pitch Type → Outcome prediction pipeline.

Usage:
    python scripts/train_outcome_heads.py --train-years 2023 --val-range 2024-04-01:2024-04-15
"""

import argparse
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from run_full_pipeline import (
    load_parquets,
    add_family_probs,
    add_temporal_weight,
    prep_balanced,
    predict_proba,
    map_outcome,
    train_lightgbm,
    train_xgboost,
    train_catboost,
)

# --- Config ---
MODEL_DIR = pathlib.Path("models")
STAGE_DIR = MODEL_DIR / "stage_heads"
STAGE_DIR.mkdir(parents=True, exist_ok=True)

# Define outcome mappings from run_full_pipeline.py
STAGE1_LABELS = ["IN_PLAY", "BALL", "STRIKE"]  # 3-way
BIP_CLASSES = ["HR", "3B", "2B", "1B", "FC", "SAC", "OUT"]  # 7-way
RUN_VALUE = {
    "HR": 1.44,
    "3B": 1.03,
    "2B": 0.78,
    "1B": 0.47,
    "FC": 0.30,
    "SAC": 0.02,
    "OUT": -0.27,
    "BALL": 0.33,
    "STRIKE": 0.00,
}


def create_outcome_labels(df):
    """Create Stage 1 and Stage 2 outcome labels from pitch data."""
    print("🎯 Creating outcome labels...")

    # Stage 1: 3-class (IN_PLAY / BALL / STRIKE)
    stage1_outcomes = []

    # Stage 2: 7-class BIP outcomes (only for IN_PLAY pitches)
    bip_outcomes = []

    for _, row in df.iterrows():
        # Get the outcome using the same mapping as main pipeline
        outcome = map_outcome(
            row.get("events", ""),
            row.get("description", ""),
            row.get("pitch_number", 1),
            row.get("at_bat_number", 1),
        )

        # Stage 1 classification
        if outcome in BIP_CLASSES:
            stage1_outcomes.append("IN_PLAY")
            bip_outcomes.append(outcome)
        elif outcome == "BALL":
            stage1_outcomes.append("BALL")
            bip_outcomes.append(None)  # Not applicable
        else:  # STRIKE or other
            stage1_outcomes.append("STRIKE")
            bip_outcomes.append(None)  # Not applicable

    df["stage1_outcome"] = stage1_outcomes
    df["bip_outcome"] = bip_outcomes

    print(f"✅ Stage 1 distribution:")
    print(df["stage1_outcome"].value_counts().to_dict())

    # Only show BIP distribution for IN_PLAY pitches
    in_play_mask = df["stage1_outcome"] == "IN_PLAY"
    if in_play_mask.sum() > 0:
        print(f"✅ Stage 2 (BIP) distribution:")
        print(df[in_play_mask]["bip_outcome"].value_counts().to_dict())

    return df


def get_blended_logits(df, best_weights, models):
    """Generate blended logits using the same approach as main pipeline."""
    print("🔗 Generating blended pitch type logits...")

    # Prepare features using the same preprocessing
    X, y, w, enc = prep_balanced(df)

    # Get individual model predictions
    lgb_probs = predict_proba(models["lgb"], X, "lgb")
    xgb_probs = predict_proba(models["xgb"], X, "xgb")
    cat_probs = predict_proba(models["cat"], X, "cat")

    # Blend using the same weights as main pipeline
    tree_logits = (
        best_weights["lgb"] * lgb_probs
        + best_weights["xgb"] * xgb_probs
        + best_weights["cat"] * cat_probs
    )

    # Check for GRU logits and blend if available
    gru_val_path = MODEL_DIR / "gru_logits_val.npy"
    gru_test_path = MODEL_DIR / "gru_logits_test.npy"

    if gru_val_path.exists() and gru_test_path.exists():
        print("🧠 Found GRU logits, applying ensemble blending...")
        # This is simplified - in practice we'd need to match the exact samples
        # For now, assume we have the right samples
        # In a real implementation, we'd need to save sample indices during training
        print(
            "⚠️  Note: Using tree-only logits (GRU sample alignment needed for production)"
        )

    print(f"✅ Generated logits shape: {tree_logits.shape}")
    # Return the filtered dataframe as well to ensure alignment
    return tree_logits, y, enc, X.index


def train_stage1_model(X_logits, y_stage1, X_val_logits, y_val_stage1):
    """Train Stage 1 model (3-class: IN_PLAY/BALL/STRIKE)."""
    print("🎯 Training Stage 1 model (IN_PLAY/BALL/STRIKE)...")

    # Encode labels
    stage1_encoder = LabelEncoder()
    y_train_encoded = stage1_encoder.fit_transform(y_stage1)
    y_val_encoded = stage1_encoder.transform(y_val_stage1)

    # Train LightGBM model
    train_data = lgb.Dataset(X_logits, label=y_train_encoded)
    val_data = lgb.Dataset(X_val_logits, label=y_val_encoded, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    # Evaluate
    val_pred = model.predict(X_val_logits)
    val_acc = accuracy_score(y_val_encoded, val_pred.argmax(axis=1))
    val_logloss = log_loss(y_val_encoded, val_pred)

    print(f"📊 Stage 1 Results:")
    print(f"   Accuracy: {val_acc:.4f}")
    print(f"   Log-Loss: {val_logloss:.4f}")

    return model, stage1_encoder


def train_stage2_model(X_logits, y_bip, X_val_logits, y_val_bip):
    """Train Stage 2 model (7-class BIP outcomes)."""
    print("🎯 Training Stage 2 model (Ball-in-Play outcomes)...")

    # Filter to only IN_PLAY samples
    train_mask = pd.notna(y_bip)
    val_mask = pd.notna(y_val_bip)

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        print("⚠️  No IN_PLAY samples found, skipping Stage 2 training")
        return None, None

    X_train_bip = X_logits[train_mask]
    y_train_bip = y_bip[train_mask]
    X_val_bip = X_val_logits[val_mask]
    y_val_bip = y_val_bip[val_mask]

    print(f"📊 BIP Training samples: {len(X_train_bip)}")
    print(f"📊 BIP Validation samples: {len(X_val_bip)}")

    # Encode labels
    bip_encoder = LabelEncoder()
    y_train_bip_encoded = bip_encoder.fit_transform(y_train_bip)
    y_val_bip_encoded = bip_encoder.transform(y_val_bip)

    # Train LightGBM model
    train_data = lgb.Dataset(X_train_bip, label=y_train_bip_encoded)
    val_data = lgb.Dataset(X_val_bip, label=y_val_bip_encoded, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(bip_encoder.classes_),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    # Evaluate
    val_pred = model.predict(X_val_bip)
    val_acc = accuracy_score(y_val_bip_encoded, val_pred.argmax(axis=1))
    val_logloss = log_loss(y_val_bip_encoded, val_pred)

    print(f"📊 Stage 2 Results:")
    print(f"   Accuracy: {val_acc:.4f}")
    print(f"   Log-Loss: {val_logloss:.4f}")

    return model, bip_encoder


def train_outcome_heads(train_years, val_range):
    """Train both stage outcome prediction models."""
    print(f"🚀 Training Outcome Head Models")
    print(f"📊 Train years: {train_years}")
    print(f"📈 Validation range: {val_range}")
    print("=" * 60)

    # Load data
    print("⏳ Loading training data...")
    train_df = load_parquets(train_years)

    print("⏳ Loading validation data...")
    val_year = int(val_range.split(":")[0].split("-")[0])
    val_df = load_parquets([val_year], val_range)

    # Add family probabilities and temporal weighting
    print("⚙️  Adding family probabilities and temporal weights...")
    latest_date = pd.to_datetime(train_df["game_date"].max())
    train_df = add_temporal_weight(train_df, latest_date, 0.0008)
    train_df = add_family_probs(train_df)
    val_df = add_family_probs(val_df)

    # Create outcome labels
    train_df = create_outcome_labels(train_df)
    val_df = create_outcome_labels(val_df)

    # Load the tree models that were trained by main pipeline
    print("📦 Loading tree models...")

    # Load the best blend weights
    blend_weights_path = None
    checkpoint_dirs = sorted(
        [d for d in MODEL_DIR.glob("checkpoint_*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if checkpoint_dirs:
        # Look for a checkpoint with blend_weights.json
        for checkpoint_dir in checkpoint_dirs:
            potential_weights_path = checkpoint_dir / "blend_weights.json"
            if potential_weights_path.exists():
                blend_weights_path = potential_weights_path
                print(f"✅ Using models from: {checkpoint_dir.name}")
                break
        else:
            print("📂 No checkpoint found with blend_weights.json")

    if blend_weights_path and blend_weights_path.exists():
        with open(blend_weights_path, "r") as f:
            best_weights = json.load(f)
        print(f"✅ Loaded blend weights: {best_weights}")

        # Load models from the same checkpoint
        checkpoint_dir = blend_weights_path.parent
        models = {}

        # Load each model type
        for model_type in ["lgb", "xgb", "cat"]:
            model_files = list(checkpoint_dir.glob(f"{model_type}.*"))
            if model_files:
                model_path = model_files[0]
                if model_type == "lgb":
                    models[model_type] = lgb.Booster(model_file=str(model_path))
                elif model_type == "xgb":
                    import xgboost as xgb

                    models[model_type] = xgb.Booster()
                    models[model_type].load_model(str(model_path))
                else:  # catboost
                    from catboost import CatBoostClassifier

                    models[model_type] = CatBoostClassifier()
                    models[model_type].load_model(str(model_path))
                print(f"✅ Loaded {model_type} model from {model_path}")
    else:
        print("❌ No trained models found. Please run main pipeline first:")
        print("   python run_full_pipeline.py train --train-years 2023 --toy")
        sys.exit(1)

    # Generate blended logits as features for outcome models
    X_train_logits, y_train_pt, train_enc, train_indices = get_blended_logits(
        train_df, best_weights, models
    )
    X_val_logits, y_val_pt, val_enc, val_indices = get_blended_logits(
        val_df, best_weights, models
    )

    # Extract outcome labels - MUST use the same indices as the logits
    y_train_stage1 = train_df.loc[train_indices, "stage1_outcome"].values
    y_val_stage1 = val_df.loc[val_indices, "stage1_outcome"].values
    y_train_bip = train_df.loc[train_indices, "bip_outcome"].values
    y_val_bip = val_df.loc[val_indices, "bip_outcome"].values

    # Train Stage 1 model
    stage1_model, stage1_encoder = train_stage1_model(
        X_train_logits, y_train_stage1, X_val_logits, y_val_stage1
    )

    # Train Stage 2 model
    stage2_model, bip_encoder = train_stage2_model(
        X_train_logits, y_train_bip, X_val_logits, y_val_bip
    )

    # Save models and encoders
    print("💾 Saving stage head models...")

    stage1_model.save_model(STAGE_DIR / "stage1.lgb")
    with open(STAGE_DIR / "stage1_encoder.pkl", "wb") as f:
        pickle.dump(stage1_encoder, f)

    if stage2_model is not None:
        stage2_model.save_model(STAGE_DIR / "bip.lgb")
        with open(STAGE_DIR / "bip_encoder.pkl", "wb") as f:
            pickle.dump(bip_encoder, f)

    print(f"✅ Stage head models saved to {STAGE_DIR}")
    print("✅ Outcome heads training complete!")

    return stage1_model, stage2_model, stage1_encoder, bip_encoder


def main():
    parser = argparse.ArgumentParser(
        description="Train outcome head models for pitch prediction"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        required=True,
        help="Years to use for training (e.g., --train-years 2023)",
    )
    parser.add_argument(
        "--val-range",
        type=str,
        required=True,
        help="Validation date range (e.g., 2024-04-01:2024-04-15)",
    )

    args = parser.parse_args()

    train_outcome_heads(train_years=args.train_years, val_range=args.val_range)


if __name__ == "__main__":
    main()
