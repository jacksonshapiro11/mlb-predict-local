#!/usr/bin/env python3
"""
scripts/optuna_lgb.py
=====================
Hyperparameter optimization for LightGBM using Optuna.

This script performs automated hyperparameter search for the LightGBM model
in the MLB pitch prediction pipeline, optimizing validation log-loss.

Search Space:
- num_leaves: [64, 128, 192, 256]
- min_data_in_leaf: [20, 50, 100]  
- feature_fraction: [0.6, 0.7, 0.8, 0.9]
- lambda_l1: [0, 0.1, 0.5]
- lambda_l2: [0, 0.1, 0.5]

Usage:
    python scripts/optuna_lgb.py --train-years 2023 --val-range 2024-04-01:2024-04-15 --trials 20
    python scripts/optuna_lgb.py --train-years 2023 --val-range 2024-04-01:2024-04-15 --trials 5 --toy
"""

import argparse
import sys
import pathlib
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import log_loss

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from run_full_pipeline import (
    load_parquets,
    prep_balanced,
    add_family_probs,
    add_temporal_weight,
    CAT_COLS,
)

# --- Config ---
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def optimize_lgb_hyperparams(train_years, val_range, n_trials=20, toy_mode=False):
    """Run Optuna hyperparameter optimization for LightGBM."""

    print(f"🔍 Optuna LightGBM Hyperparameter Optimization")
    print(f"📊 Train years: {train_years}")
    print(f"📈 Validation range: {val_range}")
    print(f"🔄 Trials: {n_trials}")
    if toy_mode:
        print(f"🧪 TOY MODE: Reduced iterations")
    print("=" * 60)

    # Load and prepare data
    print("⏳ Loading training data...")
    train_df = load_parquets(train_years)

    print("⏳ Loading validation data...")
    # Extract year from val_range for loading
    val_year = int(val_range.split(":")[0].split("-")[0])
    val_df = load_parquets([val_year], val_range)

    # Add temporal weighting and family probabilities
    print("⚙️  Adding temporal weights and family probabilities...")
    latest_date = pd.to_datetime(train_df["game_date"].max())
    train_df = add_temporal_weight(train_df, latest_date, 0.0008)
    train_df = add_family_probs(train_df)
    val_df = add_family_probs(val_df)

    # Prepare features and targets
    print("🔧 Preprocessing data...")
    X_train, y_train, w_train, encoders = prep_balanced(train_df)
    X_val, y_val, w_val, _ = prep_balanced(val_df, encoders)

    print(f"✅ Training data: {X_train.shape}")
    print(f"✅ Validation data: {X_val.shape}")
    print(f"✅ Target classes: {len(encoders['target'].classes_)}")

    # Sample data for toy mode
    if toy_mode:
        sample_size = min(10000, len(X_train))
        sample_idx = np.random.RandomState(42).choice(
            len(X_train), sample_size, replace=False
        )
        X_train = X_train.iloc[sample_idx]
        y_train = y_train[sample_idx]
        w_train = w_train.iloc[sample_idx]
        print(f"🎲 Sampled {sample_size} training examples for toy mode")

    def objective(trial):
        """Optuna objective function for LightGBM hyperparameter optimization."""

        # Define search space
        params = {
            "objective": "multiclass",
            "num_class": len(encoders["target"].classes_),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            # Hyperparameters to optimize
            "num_leaves": trial.suggest_categorical("num_leaves", [64, 128, 192, 256]),
            "min_data_in_leaf": trial.suggest_categorical(
                "min_data_in_leaf", [20, 50, 100]
            ),
            "feature_fraction": trial.suggest_categorical(
                "feature_fraction", [0.6, 0.7, 0.8, 0.9]
            ),
            "lambda_l1": trial.suggest_categorical("lambda_l1", [0.0, 0.1, 0.5]),
            "lambda_l2": trial.suggest_categorical("lambda_l2", [0.0, 0.1, 0.5]),
            # Fixed parameters
            "learning_rate": 0.1,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }

        # Prepare datasets
        lgb_train = lgb.Dataset(
            X_train,
            y_train,
            weight=w_train,
            categorical_feature=[
                i for i, c in enumerate(X_train.columns) if c in CAT_COLS
            ],
            free_raw_data=False,
        )
        lgb_val = lgb.Dataset(X_val, y_val, weight=w_val, reference=lgb_train)

        # Train model
        max_iters = 100 if toy_mode else 500
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),  # Silent
        ]

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=max_iters,
            valid_sets=[lgb_val],
            callbacks=callbacks,
        )

        # Predict and calculate validation loss
        val_pred = model.predict(X_val)
        val_loss = log_loss(y_val, val_pred)

        return val_loss

    # Create Optuna study
    print(f"🚀 Starting Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n🎯 Optimization Results:")
    print(f"   Best validation log-loss: {best_value:.6f}")
    print(f"   Best parameters:")
    for param, value in best_params.items():
        print(f"     {param}: {value}")

    # Add fixed parameters to best_params for completeness
    complete_params = {
        "objective": "multiclass",
        "num_class": len(encoders["target"].classes_),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
        "learning_rate": 0.1,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        **best_params,
    }

    # Save results
    output_path = MODEL_DIR / "optuna_lgb.json"
    result = {
        "best_params": complete_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "train_years": train_years,
        "val_range": val_range,
        "toy_mode": toy_mode,
        "timestamp": time.time(),
        "study_trials": len(study.trials),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Show trial summary
    print(f"\n📊 Trial Summary:")
    print(f"   Completed trials: {len(study.trials)}")
    print(f"   Best trial: #{study.best_trial.number}")

    # Show top 3 trials
    sorted_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )
    print(f"\n🏆 Top 3 Trials:")
    for i, trial in enumerate(sorted_trials[:3]):
        if trial.value is not None:
            print(f"   #{trial.number}: {trial.value:.6f}")
            for param, value in trial.params.items():
                print(f"     {param}: {value}")
            print()

    return best_params, best_value


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for LightGBM"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        required=True,
        help="Years to use for training (e.g., --train-years 2023 2024)",
    )
    parser.add_argument(
        "--val-range",
        type=str,
        required=True,
        help="Validation date range (e.g., 2024-04-01:2024-04-15)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run (default: 20)",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use toy mode (reduced iterations and data sampling)",
    )

    args = parser.parse_args()

    optimize_lgb_hyperparams(
        train_years=args.train_years,
        val_range=args.val_range,
        n_trials=args.trials,
        toy_mode=args.toy,
    )


if __name__ == "__main__":
    main()
