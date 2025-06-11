#!/usr/bin/env python3
"""
Two-Model Architecture for MLB Pitch Prediction
===============================================
Model 1: Pitch Type Prediction
Model 2: Outcome Prediction Given Pitch Type

Key Anti-Leakage Measures:
1. Temporal separation: Train on 2023, validate on 2024 April, test on 2025 April-June
2. Proper feature separation for each model
3. Model 2 uses Model 1 predictions, not ground truth
4. No outcome-related features in Model 1
5. No future information in historical windows
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import duckdb
from mlb_pred.util.leak_tokens import LEAK_TOKENS
from run_full_pipeline import add_family_probs, load_parquets


class TwoModelPitchPredictor:
    def __init__(self):
        self.model1 = None  # Pitch type predictor
        self.model2 = None  # Outcome predictor
        self.pitch_type_encoder = LabelEncoder()
        self.outcome_encoder = LabelEncoder()
        self.model1_feature_names = None  # Store feature names for consistency

    def prepare_model1_features(self, df):
        """
        Prepare features for Model 1 (Pitch Type Prediction)

        INCLUDES:
        - Game context: count, score, baserunners, inning
        - Pitcher arsenal: historical usage, velocity, spin rates
        - Batter matchup: historical performance vs pitch types
        - Recent form: 7-day trends
        - Sequence: previous pitches

        EXCLUDES:
        - Current pitch characteristics (release_speed, pfx_x, etc.)
        - Outcome information (events, description, woba)
        - Within-pitch physics (plate_x, zone, launch_speed)
        """

        # ---------- explicit whitelist of safe features ----------
        KEEP_FEATURES = {
            # Core situational
            "balls",
            "strikes",
            "outs_when_up",
            "on_1b",
            "on_2b",
            "on_3b",
            "home_score",
            "away_score",
            "stand",
            "p_throws",
            "count_state",
            # Player identifiers
            "batter_fg",
            "pitcher_fg",
            # Sequence/lags
            "prev_pitch_1",
            "prev_pitch_2",
            "prev_pitch_3",
            "prev_pitch_4",
            "dvelo1",
            # Arsenal features - 30d averages by pitch type
            "velocity_30d_CH",
            "velocity_30d_CU",
            "velocity_30d_FC",
            "velocity_30d_FF",
            "velocity_30d_FS",
            "velocity_30d_KC",
            "velocity_30d_OTHER",
            "velocity_30d_SI",
            "velocity_30d_SL",
            "velocity_30d_ST",
            "spin_rate_30d_CH",
            "spin_rate_30d_CU",
            "spin_rate_30d_FC",
            "spin_rate_30d_FF",
            "spin_rate_30d_FS",
            "spin_rate_30d_KC",
            "spin_rate_30d_OTHER",
            "spin_rate_30d_SI",
            "spin_rate_30d_SL",
            "spin_rate_30d_ST",
            "usage_30d_CH",
            "usage_30d_CU",
            "usage_30d_FC",
            "usage_30d_FF",
            "usage_30d_FS",
            "usage_30d_KC",
            "usage_30d_OTHER",
            "usage_30d_SI",
            "usage_30d_SL",
            "usage_30d_ST",
            # Batter matchups - 30d
            "batter_xwoba_30d_CH",
            "batter_xwoba_30d_CU",
            "batter_xwoba_30d_FC",
            "batter_xwoba_30d_FF",
            "batter_xwoba_30d_FS",
            "batter_xwoba_30d_KC",
            "batter_xwoba_30d_OTHER",
            "batter_xwoba_30d_SI",
            "batter_xwoba_30d_SL",
            "batter_xwoba_30d_ST",
            # Count state performance
            "contact_rate_30d_AHEAD",
            "contact_rate_30d_BEHIND",
            "contact_rate_30d_EVEN",
            "whiff_rate_30d_AHEAD",
            "whiff_rate_30d_BEHIND",
            "whiff_rate_30d_EVEN",
            # Whiff rates by pitch type
            "whiff_rate_30d_CH",
            "whiff_rate_30d_CU",
            "whiff_rate_30d_FC",
            "whiff_rate_30d_FF",
            "whiff_rate_30d_FS",
            "whiff_rate_30d_KC",
            "whiff_rate_30d_OTHER",
            "whiff_rate_30d_SI",
            "whiff_rate_30d_SL",
            "whiff_rate_30d_ST",
            # Performance vs handedness
            "hit_rate_30d_vs_L",
            "hit_rate_30d_vs_R",
            "whiff_rate_30d_vs_L",
            "whiff_rate_30d_vs_R",
            "xwoba_30d_vs_L",
            "xwoba_30d_vs_R",
            # Overall rates
            "k_rate_30d",
            # Recent form - 7 day
            "hit_rate_7d",
            "velocity_7d",
            "whiff_rate_7d",
            # Cumulative within-game
            "cum_ch_count",
            "cum_ch_spin",
            "cum_ch_velocity",
            "cum_ff_count",
            "cum_ff_spin",
            "cum_ff_velocity",
            "cum_sl_count",
            "cum_sl_spin",
            "cum_sl_velocity",
            "cum_game_pitches",
            # Family probabilities
            "FAM_PROB_FB",
            "FAM_PROB_BR",
            "FAM_PROB_OS",
        }

        # Only keep features that exist in the data AND are in our safe list
        available_features = [col for col in KEEP_FEATURES if col in df.columns]
        drop_cols = [col for col in df.columns if col not in available_features]

        print(f"Model 1 features: {len(available_features)}")

        # If we're in training phase, store the feature names for consistency
        if self.model1_feature_names is None:
            self.model1_feature_names = available_features
        else:
            # Use consistent features from training
            available_features = [
                f for f in self.model1_feature_names if f in df.columns
            ]

        # Get the feature dataframe
        feature_df = df[available_features].copy()

        # Encode categorical features for LightGBM
        categorical_cols = ["stand", "p_throws", "prev_pitch_1", "prev_pitch_2"]

        for col in categorical_cols:
            if col in feature_df.columns:
                # Convert to string to handle categorical columns
                feature_df[col] = feature_df[col].astype(str).fillna("UNKNOWN")
                # Simple label encoding for categorical features
                unique_vals = sorted(feature_df[col].unique())
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                feature_df[col] = feature_df[col].map(val_to_num)

        # Fill missing features with zeros if needed (for test data)
        if len(feature_df.columns) < len(self.model1_feature_names):
            for missing_feature in self.model1_feature_names:
                if missing_feature not in feature_df.columns:
                    feature_df[missing_feature] = 0.0
            # Reorder columns to match training order
            feature_df = feature_df[self.model1_feature_names]

        return feature_df

    def prepare_model2_features(self, df, predicted_pitch_type=None):
        """
        Prepare features for Model 2 (Outcome Prediction)

        INCLUDES:
        - All Model 1 features
        - Predicted pitch type from Model 1
        - Count state (balls/strikes matter for outcome)

        EXCLUDES:
        - Ground truth pitch type (no cheating!)
        - Current pitch physics (release_speed, pfx_x, etc.)
        - Outcome information (the thing we're predicting)
        """

        # Get Model 1 features
        model1_features = self.prepare_model1_features(df)

        # Add predicted pitch type
        if predicted_pitch_type is not None:
            model1_features = model1_features.copy()
            # Encode predicted pitch type
            unique_pitch_types = sorted(set(predicted_pitch_type))
            pitch_type_map = {pt: i for i, pt in enumerate(unique_pitch_types)}
            model1_features["predicted_pitch_type"] = [
                pitch_type_map[pt] for pt in predicted_pitch_type
            ]

        # Add count state encoding
        model1_features = model1_features.copy()
        count_states = (
            model1_features["balls"].astype(str)
            + "_"
            + model1_features["strikes"].astype(str)
        )
        unique_count_states = sorted(count_states.unique())
        count_state_map = {cs: i for i, cs in enumerate(unique_count_states)}
        model1_features["count_state"] = count_states.map(count_state_map)

        print(f"Model 2 features: {len(model1_features.columns)}")

        return model1_features

    def create_outcome_target(self, df):
        """
        Create simplified outcome target for Model 2

        Categories:
        - BALL: ball, hit_by_pitch
        - STRIKE: called_strike, swinging_strike, foul, foul_tip
        - HIT: single, double, triple, home_run
        - OUT: all types of outs in play
        - WALK: walk (4 balls)
        - STRIKEOUT: strikeout (3 strikes)
        """

        def classify_outcome(row):
            desc = row.get("description", "")
            event = row.get("events", "")

            # Prioritize events over descriptions
            if pd.notna(event) and event != "":
                if event in ["single", "double", "triple", "home_run"]:
                    return "HIT"
                elif event in ["walk", "intent_walk"]:
                    return "WALK"
                elif event in ["strikeout", "strikeout_double_play"]:
                    return "STRIKEOUT"
                elif "out" in event.lower() or event in [
                    "field_error",
                    "fielders_choice",
                ]:
                    return "OUT"

            # Fall back to description
            if "ball" in desc.lower() or "hit_by_pitch" in desc.lower():
                return "BALL"
            elif any(
                x in desc.lower() for x in ["called_strike", "swinging_strike", "foul"]
            ):
                return "STRIKE"

            # Default for unclear cases
            return "BALL"

        outcomes = df.apply(classify_outcome, axis=1)

        print("Outcome distribution:")
        print(outcomes.value_counts(normalize=True))

        return outcomes

    def load_data_with_temporal_split(self):
        """
        Load data with proper temporal separation to avoid leakage

        Automatically detects available years and uses:
        - Train: All available years EXCEPT the most recent two
        - Val: Second most recent year
        - Test: Most recent year (avoiding March to prevent leakage)

        Examples:
        - Available: 2023, 2025 → Train: 2023, Val: 2023 (split), Test: 2025
        - Available: 2018-2025 → Train: 2018-2023, Val: 2024, Test: 2025
        """

        print("🔍 Auto-detecting available historical feature files...")

        import glob
        import os

        # Find all available historical feature files
        pattern = "data/features_historical/statcast_historical_*.parquet"
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                "No historical feature files found! Run: python etl/build_historical_features.py YEAR"
            )

        # Extract years from filenames
        available_years = []
        for file in files:
            year = int(os.path.basename(file).split("_")[-1].split(".")[0])
            available_years.append(year)

        available_years.sort()
        print(f"📊 Available years: {available_years}")

        if len(available_years) < 2:
            raise ValueError("Need at least 2 years of data for temporal separation")

        # Determine train/val/test split
        test_year = available_years[-1]  # Most recent year for testing

        if len(available_years) >= 3:
            # 3+ years: use most recent for test, second most recent for val, rest for train
            train_years = available_years[:-2]
            val_year = available_years[-2]
        elif len(available_years) == 2:
            # Only 2 years: use older for train, newer for test, split train for validation
            train_years = [available_years[0]]
            val_year = None  # Will split training data
        else:
            raise ValueError("Need at least 2 years of data for temporal separation")

        print(f"🎯 Data split:")
        print(f"   Train years: {train_years}")
        print(f"   Val year: {val_year if val_year else f'{train_years[-1]} (split)'}")
        print(f"   Test year: {test_year}")

        con = duckdb.connect()

        # Load training data (all training years, avoiding March)
        train_dfs = []
        for year in train_years:
            query = f"""
                SELECT * FROM parquet_scan('data/features_historical/statcast_historical_{year}.parquet')
                WHERE game_date >= '{year}-04-01' AND game_date <= '{year}-09-30'
            """
            df = con.execute(query).df()
            train_dfs.append(df)
            print(f"   Loaded {len(df):,} pitches from {year}")

        train_df = (
            pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        )

        # Load validation data
        if val_year:
            # Use separate year for validation
            val_query = f"""
                SELECT * FROM parquet_scan('data/features_historical/statcast_historical_{val_year}.parquet')
                WHERE game_date >= '{val_year}-04-01' AND game_date <= '{val_year}-09-30'
            """
            val_df = con.execute(val_query).df()
            print(f"   Loaded {len(val_df):,} pitches from {val_year} for validation")
        else:
            # Split training data for validation (temporal split - later part of training data)
            train_size = int(0.8 * len(train_df))
            val_df = train_df[train_size:].copy()
            train_df = train_df[:train_size].copy()
            print(
                f"   Split training data: {len(train_df):,} train, {len(val_df):,} validation"
            )

        # Load test data (avoiding March to prevent temporal leakage)
        test_query = f"""
            SELECT * FROM parquet_scan('data/features_historical/statcast_historical_{test_year}.parquet')
            WHERE game_date >= '{test_year}-04-01' AND game_date <= '{test_year}-06-30'
        """
        test_df = con.execute(test_query).df()
        print(f"   Loaded {len(test_df):,} pitches from {test_year} for testing")

        con.close()

        # Add family probabilities to all datasets
        print("\n🏗️  Adding pitch family probabilities...")
        train_df = add_family_probs(train_df)
        val_df = add_family_probs(val_df)
        test_df = add_family_probs(test_df)

        print(f"\n📊 Final dataset sizes:")
        print(f"   Train: {len(train_df):,} pitches")
        print(f"   Val:   {len(val_df):,} pitches")
        print(f"   Test:  {len(test_df):,} pitches")

        return train_df, val_df, test_df

    def train_model1(self, train_df, val_df):
        """Train Model 1: Pitch Type Prediction"""

        print("\n" + "=" * 50)
        print("TRAINING MODEL 1: PITCH TYPE PREDICTION")
        print("=" * 50)

        # Prepare features and target
        X_train = self.prepare_model1_features(train_df)
        X_val = self.prepare_model1_features(val_df)

        y_train = train_df["pitch_type_can"].fillna("OTHER")
        y_val = val_df["pitch_type_can"].fillna("OTHER")

        # Encode targets
        y_train_encoded = self.pitch_type_encoder.fit_transform(y_train)
        y_val_encoded = self.pitch_type_encoder.transform(y_val)

        print(f"Training Model 1 on {len(X_train):,} pitches")
        print(f"Pitch types: {list(self.pitch_type_encoder.classes_)}")

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train_encoded)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": len(self.pitch_type_encoder.classes_),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        self.model1 = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        # Validate Model 1
        val_pred = self.model1.predict(X_val)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_accuracy = accuracy_score(y_val_encoded, val_pred_classes)

        print(f"\nModel 1 Validation Accuracy: {val_accuracy:.3f}")

        # This should be 55-65% for realistic performance
        if val_accuracy > 0.75:
            print("⚠️  WARNING: Accuracy > 75% suggests potential leakage!")
        elif val_accuracy < 0.45:
            print("⚠️  WARNING: Accuracy < 45% suggests model issues!")
        else:
            print("✅ Accuracy in realistic range (45-75%)")

        return val_accuracy

    def train_model2(self, train_df, val_df):
        """Train Model 2: Outcome Prediction Given Pitch Type"""

        print("\n" + "=" * 50)
        print("TRAINING MODEL 2: OUTCOME PREDICTION")
        print("=" * 50)

        # Get Model 1 predictions for training data
        X_train_m1 = self.prepare_model1_features(train_df)
        train_pitch_pred = self.model1.predict(X_train_m1)
        train_pitch_classes = np.argmax(train_pitch_pred, axis=1)
        train_pitch_types = self.pitch_type_encoder.inverse_transform(
            train_pitch_classes
        )

        # Get Model 1 predictions for validation data
        X_val_m1 = self.prepare_model1_features(val_df)
        val_pitch_pred = self.model1.predict(X_val_m1)
        val_pitch_classes = np.argmax(val_pitch_pred, axis=1)
        val_pitch_types = self.pitch_type_encoder.inverse_transform(val_pitch_classes)

        # Prepare Model 2 features (using Model 1 predictions)
        X_train = self.prepare_model2_features(train_df, train_pitch_types)
        X_val = self.prepare_model2_features(val_df, val_pitch_types)

        # Prepare Model 2 targets
        y_train = self.create_outcome_target(train_df)
        y_val = self.create_outcome_target(val_df)

        # Encode targets
        y_train_encoded = self.outcome_encoder.fit_transform(y_train)
        y_val_encoded = self.outcome_encoder.transform(y_val)

        print(f"Training Model 2 on {len(X_train):,} pitches")
        print(f"Outcomes: {list(self.outcome_encoder.classes_)}")

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train_encoded)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": len(self.outcome_encoder.classes_),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        self.model2 = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        # Validate Model 2
        val_pred = self.model2.predict(X_val)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_accuracy = accuracy_score(y_val_encoded, val_pred_classes)

        print(f"\nModel 2 Validation Accuracy: {val_accuracy:.3f}")

        # Check for leakage
        if val_accuracy > 0.80:
            print("⚠️  WARNING: Accuracy > 80% suggests potential leakage!")
        else:
            print("✅ Outcome prediction accuracy seems reasonable")

        return val_accuracy

    def evaluate_on_test(self, test_df):
        """Evaluate both models on test set"""

        print("\n" + "=" * 50)
        print("EVALUATING ON TEST SET")
        print("=" * 50)

        # Model 1 evaluation
        X_test_m1 = self.prepare_model1_features(test_df)
        test_pitch_pred = self.model1.predict(X_test_m1)
        test_pitch_classes = np.argmax(test_pitch_pred, axis=1)
        test_pitch_types = self.pitch_type_encoder.inverse_transform(test_pitch_classes)

        y_test_pitch = test_df["pitch_type_can"].fillna("OTHER")
        y_test_pitch_encoded = self.pitch_type_encoder.transform(y_test_pitch)

        model1_accuracy = accuracy_score(y_test_pitch_encoded, test_pitch_classes)
        print(f"Model 1 Test Accuracy: {model1_accuracy:.3f}")

        # Model 2 evaluation
        X_test_m2 = self.prepare_model2_features(test_df, test_pitch_types)
        test_outcome_pred = self.model2.predict(X_test_m2)
        test_outcome_classes = np.argmax(test_outcome_pred, axis=1)

        y_test_outcome = self.create_outcome_target(test_df)
        y_test_outcome_encoded = self.outcome_encoder.transform(y_test_outcome)

        model2_accuracy = accuracy_score(y_test_outcome_encoded, test_outcome_classes)
        print(f"Model 2 Test Accuracy: {model2_accuracy:.3f}")

        # Combined evaluation
        print(f"\nCombined Model Performance:")
        print(f"  Pitch Type Prediction: {model1_accuracy:.1%}")
        print(f"  Outcome Prediction: {model2_accuracy:.1%}")

        return model1_accuracy, model2_accuracy


def main():
    """Main training pipeline"""

    print("🎯 TWO-MODEL MLB PITCH PREDICTION PIPELINE")
    print("=" * 60)

    predictor = TwoModelPitchPredictor()

    # Load data with temporal separation (auto-detects available years)
    train_df, val_df, test_df = predictor.load_data_with_temporal_split()

    # Train Model 1: Pitch Type Prediction
    model1_acc = predictor.train_model1(train_df, val_df)

    # Train Model 2: Outcome Prediction
    model2_acc = predictor.train_model2(train_df, val_df)

    # Final evaluation
    test_m1_acc, test_m2_acc = predictor.evaluate_on_test(test_df)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Model 1 (Pitch Type): Val {model1_acc:.3f} | Test {test_m1_acc:.3f}")
    print(f"Model 2 (Outcome): Val {model2_acc:.3f} | Test {test_m2_acc:.3f}")

    # Leakage checks
    if test_m1_acc > 0.75 or test_m2_acc > 0.80:
        print("\n⚠️  HIGH ACCURACY DETECTED - CHECK FOR LEAKAGE!")
    else:
        print("\n✅ Accuracies in realistic range")


if __name__ == "__main__":
    main()
