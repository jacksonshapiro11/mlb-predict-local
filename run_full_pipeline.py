#!/usr/bin/env python
"""
run_full_pipeline.py
====================
One-shot ETL → train → evaluate → save for the MLB Pitch-Prediction project.

USAGE
-----
# build 2015-2025 features (will skip seasons already built)
python run_full_pipeline.py build --years 2015 2016 … 2025

# train / evaluate balanced-predictive ensemble on 2018-2023 train,
# 2024-04-01→07-31 val, 2024-08-01→2025-12-31 test
python run_full_pipeline.py train \
    --train-years 2018 2019 2020 2021 2022 2023 \
    --val "2024-04-01:2024-07-31" \
    --test "2024-08-01:2025-12-31" \
    --decay 0.0008
"""

import argparse
import subprocess
import pathlib
import json
import time
import warnings
import pickle
import os
import re
from datetime import date
from functools import lru_cache
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import ParameterGrid
from scipy.special import softmax as sp_softmax
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from mlb_pred.util.leak_tokens import LEAK_TOKENS

warnings.filterwarnings("ignore")

# GPU Configuration
USE_GPU = os.getenv("GPU", "0") == "1"
print(f"🖥️  Training on {'GPU' if USE_GPU else 'CPU'}")

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
PARQUET_DIR = pathlib.Path("data/features_historical")
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PT = "pitch_type_can"
# ---- refined outcome taxonomy ----
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
CAT_COLS = [
    "stand",
    "p_throws",
    "count_state",
    "prev_pitch_1",
    "prev_pitch_2",
    "prev_pitch_3",
    "prev_pitch_4",
]  # dvelo1 stays numeric
DECAY_DEFAULT = 0.0008  # ~2.4-season half-life

LAG_SQL = """
WITH base AS (
  SELECT *
  FROM parquet_scan({paths})
  {where_clause}
)
SELECT *, 
       LAG(pitch_type_can,1) OVER w AS prev_pitch_1,
       LAG(pitch_type_can,2) OVER w AS prev_pitch_2,
       LAG(pitch_type_can,3) OVER w AS prev_pitch_3,
       LAG(pitch_type_can,4) OVER w AS prev_pitch_4,
       release_speed - LAG(release_speed,1) OVER w AS dvelo1
FROM base
WINDOW w AS (
  PARTITION BY pitcher, game_pk
  ORDER BY at_bat_number, pitch_number
)
"""


# --------------------------------------------------------------------------- #
#  0. HELPERS
# --------------------------------------------------------------------------- #
def run(cmd):
    """Run a shell command & pipe through."""
    print(f"ℹ️  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def build_season(year: int):
    pq = PARQUET_DIR / f"statcast_historical_{year}.parquet"
    if pq.exists():
        print(f"✅ {pq} already exists – skipping build")
    else:
        run(f"python etl/build_historical_features.py {year}")


def load_duck(query: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df


def load_parquets(years, date_range: str | None = None):
    paths = [str(PARQUET_DIR / f"statcast_historical_{y}.parquet") for y in years]
    path_expr = "[" + ",".join([f"'{p}'" for p in paths]) + "]"
    where_clause = ""
    if date_range:
        start, end = date_range.split(":")
        where_clause = f"WHERE game_date BETWEEN DATE '{start}' AND DATE '{end}'"
    q = LAG_SQL.format(paths=path_expr, where_clause=where_clause)
    print(f"🗄️  DuckDB query: {q[:120]}…")
    df = load_duck(q)
    return df


def map_outcome(ev, des, pitch_num, ab_end):
    """Map pitch events and descriptions to outcome categories."""
    if ev == "home_run":
        return "HR"
    if ev == "triple":
        return "3B"
    if ev == "double":
        return "2B"
    if ev == "single":
        return "1B"
    if ev in ("fielders_choice_out", "fielders_choice"):
        return "FC"
    if ev in ("sac_fly", "sac_bunt"):
        return "SAC"
    # ----- outs in play -----
    if ev in ("groundout", "flyout", "pop_out", "lineout"):
        return "OUT"
    # ----- balls / strikes -----
    if des == "hit_by_pitch":
        return "BALL"
    if des.startswith("ball"):
        return "BALL"
    if des.startswith(("called_strike", "foul", "swinging_strike")):
        return "STRIKE"
    return "STRIKE"  # default safety


def add_temporal_weight(df, latest_date, lam):
    delta = (latest_date - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-lam * delta)
    return df


# Define runtime-safe columns for family model prediction
RUNTIME_SAFE_COLS = [
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
    "batter_fg",
    "pitcher_fg",
    "prev_pitch_1",
    "prev_pitch_2",
    "prev_pitch_3",
    "prev_pitch_4",
    "dvelo1",
    "k_rate_30d",
    "hit_rate_7d",
    "velocity_7d",
    "whiff_rate_7d",
    "cum_game_pitches",
    "cum_ff_count",
    "cum_sl_count",
    "cum_ch_count",
]


def load_optuna_params():
    """Load Optuna optimized parameters if available."""
    optuna_path = MODEL_DIR / "optuna_lgb.json"
    if optuna_path.exists():
        try:
            with open(optuna_path, "r") as f:
                result = json.load(f)
                return result.get("best_params", None)
        except Exception as e:
            print(f"⚠️  Warning: Could not load Optuna parameters: {e}")
            return None
    return None


def add_family_probs(df):
    """Add pitch family probability features (FB/BR/OS) to dataframe."""
    model_path = MODEL_DIR / "fam_head.lgb"
    encoder_path = MODEL_DIR / "fam_encoder.pkl"
    features_path = MODEL_DIR / "fam_features.pkl"

    if not model_path.exists():
        print("🏗️  Family model not found, training it now...")
        run(
            "python scripts/train_family_head.py --train-years 2019 2020 2021 2022 2023"
        )

    # Load family model, encoder, and feature names
    model = lgb.Booster(model_file=str(model_path))
    with open(encoder_path, "rb") as f:
        enc = pickle.load(f)
    with open(features_path, "rb") as f:
        required_features = pickle.load(f)

    # Create prediction data using the exact same features as training
    pred_data = pd.DataFrame(index=df.index)

    # Create count_state first if needed
    if (
        "count_state" in required_features
        and "balls" in df.columns
        and "strikes" in df.columns
    ):
        balls_cap = df["balls"].fillna(0).clip(0, 3)
        strikes_cap = df["strikes"].fillna(0).clip(0, 2)
        pred_data["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)

    # Add all required features
    for feature in required_features:
        if feature in df.columns:
            pred_data[feature] = df[feature]
        else:
            # Fill missing features with zeros
            pred_data[feature] = 0.0

    # Handle categorical columns same as training
    for col in CAT_COLS:
        if col in pred_data.columns:
            # Convert to string and handle missing
            pred_data[col] = pred_data[col].astype(str).fillna("__MISSING__")
            # Simple categorical to numeric conversion for prediction
            unique_vals = sorted(pred_data[col].unique())
            val_to_num = {val: i for i, val in enumerate(unique_vals)}
            pred_data[col] = pred_data[col].map(val_to_num)

    # Convert any remaining object columns to numeric
    for col in pred_data.columns:
        if pred_data[col].dtype == "object":
            pred_data[col] = pd.to_numeric(pred_data[col], errors="coerce")

    # Fill any remaining missing values
    pred_data = pred_data.fillna(0)

    # Ensure columns are in the same order as training
    pred_data = pred_data[required_features]

    # Get family probabilities
    proba = model.predict(pred_data)

    # Add probability columns
    for i, cls in enumerate(enc.classes_):
        df[f"FAM_PROB_{cls}"] = proba[:, i]
        print(f"✅ Added feature: FAM_PROB_{cls}")

    return df


# --------------------------------------------------------------------------- #
#  1.  BALANCED-PREDICTIVE  FEATURE  FILTER
# --------------------------------------------------------------------------- #
DROP_ALWAYS = [
    TARGET_PT,  # drop label
    "estimated_woba_using_speedangle",  # xwOBA label
    "pitch_outcome",  # outcome label
    "stage1_target",  # hierarchical stage1 label
    "bip_target",  # ball-in-play specific label
    "w",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
    "batter",
    "pitcher",
    "home_team",
    "pitch_name",
    "events",
    "description",
    "pitch_type",  # raw MLB pitch_type
]

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


def assert_no_leakage(cols):
    """Assert that no columns contain current pitch markers that could cause data leakage."""
    leak = [c for c in cols if any(tok in c.lower() for tok in CURRENT_PITCH_MARKERS)]
    if leak:
        raise RuntimeError(f"🚨 Leakage columns detected → {leak}")


# LEAKAGE-FOCUSED: All features that use same-day data or future information
LEAKY_FEATURES = [
    # "Today" (TD) features - calculated across entire game
    "SPIN_TD_CH",
    "SPIN_TD_CU",
    "SPIN_TD_FC",
    "SPIN_TD_FF",
    "SPIN_TD_FS",
    "SPIN_TD_KC",
    "SPIN_TD_OTHER",
    "SPIN_TD_SI",
    "SPIN_TD_SL",
    "V_TD_CH",
    "V_TD_CU",
    "V_TD_FC",
    "V_TD_FF",
    "V_TD_FS",
    "V_TD_KC",
    "V_TD_OTHER",
    "V_TD_SI",
    "V_TD_SL",
    "USAGE_TD_CH",
    "USAGE_TD_CU",
    "USAGE_TD_FC",
    "USAGE_TD_FF",
    "USAGE_TD_FS",
    "USAGE_TD_KC",
    "USAGE_TD_OTHER",
    "USAGE_TD_SI",
    "USAGE_TD_SL",
    "XWOBA_TD_CH",
    "XWOBA_TD_CU",
    "XWOBA_TD_FC",
    "XWOBA_TD_FF",
    "XWOBA_TD_FS",
    "XWOBA_TD_KC",
    "XWOBA_TD_OTHER",
    "XWOBA_TD_SI",
    "XWOBA_TD_SL",
    "K_PCT_TD",
    # 7-day features that may include same-day data
    "HIT_7D",
    "VELO_7D",
    "WHIFF_7D",
    # Current pitch physics/outcome (future information)
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "spin_axis",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "zone",
    "hc_x",
    "hc_y",
    "estimated_woba_using_speedangle",
    "delta_run_exp",
    "delta_home_win_exp",
]


def prep_balanced(df: pd.DataFrame, label_encoders: dict = None):
    df = df.copy()

    # Drop rows with missing target variable
    df.dropna(subset=[TARGET_PT], inplace=True)

    # -------- enhanced situational features (count_state etc.) -----------
    balls_cap = df["balls"].fillna(0).clip(0, 3)
    strikes_cap = df["strikes"].fillna(0).clip(0, 2)
    df["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)

    # ------------------- column filtering (explicit allow-list) -------
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

    X = df.drop(columns=drop_cols, errors="ignore")

    # ------------------- categorical encoding ---------------------------
    # Assert new lag features are present
    required_lag_features = {
        "prev_pitch_1",
        "prev_pitch_2",
        "prev_pitch_3",
        "prev_pitch_4",
        "dvelo1",
    }
    assert required_lag_features.issubset(
        X.columns
    ), f"Missing lag features: {required_lag_features - set(X.columns)}"

    if label_encoders is None:
        label_encoders = {}
    for c in CAT_COLS:
        if c not in X.columns:
            continue
        if c not in label_encoders:
            le = LabelEncoder()
            # Convert categorical columns to string first to avoid category issues
            col_values = X[c].astype(str).fillna("__MISSING__")
            le.fit(col_values)
            label_encoders[c] = le
        # Convert categorical columns to string first to avoid category issues
        col_values = X[c].astype(str).fillna("__MISSING__")
        X[c] = label_encoders[c].transform(col_values)

    # Make anything still object → numeric
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    # Encode target variable
    y_series = df[TARGET_PT]
    if "target" not in label_encoders:
        le = LabelEncoder()
        le.fit(y_series)
        label_encoders["target"] = le
    y = label_encoders["target"].transform(y_series)

    w = df.get("w", pd.Series(1, index=df.index))

    return X, y, w, label_encoders


# --------------------------------------------------------------------------- #
#  2.  MODEL  TRAINERS
# --------------------------------------------------------------------------- #
def train_lightgbm(X_tr, y_tr, w_tr, X_val, y_val, max_iters=2000):
    # Load Optuna optimized parameters if available
    optuna_params = load_optuna_params()

    if optuna_params:
        print("🔧 Using Optuna optimized LightGBM parameters")
        params = optuna_params.copy()
        params["n_estimators"] = max_iters  # Override with current max_iters
        # Ensure these are set correctly for current run
        params["num_class"] = len(np.unique(y_tr))
        params["n_jobs"] = -1
        params["verbose"] = -1
    else:
        print("🔧 Using default LightGBM parameters")
        params = {
            "objective": "multiclass",
            "num_class": len(np.unique(y_tr)),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": max_iters,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": 8,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            "colsample_bytree": 0.8,
            "subsample": 0.9,
        }

    if USE_GPU:
        params["device"] = "gpu"

    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        weight=w_tr,
        categorical_feature=[i for i, c in enumerate(X_tr.columns) if c in CAT_COLS],
        free_raw_data=False,
    )

    callbacks = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=250)]

    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr, weight=w_tr),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=callbacks,
    )
    return model


def train_xgboost(X_tr, y_tr, w_tr_final, X_val, y_val, max_iters=2000):
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr_final.astype(np.float32))
    dvl = xgb.DMatrix(X_val, label=y_val)
    params = dict(
        objective="multi:softprob",
        num_class=len(np.unique(y_tr)),
        eta=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        seed=42,
        verbosity=0,
    )

    if USE_GPU:
        params.update(
            {
                "tree_method": "gpu_hist",
                "gpu_id": 0,
                "predictor": "gpu_predictor",
            }
        )

    model = xgb.train(
        params,
        dtr,
        max_iters,
        evals=[(dvl, "val")],
        early_stopping_rounds=150,
        verbose_eval=250,
    )
    return model


def train_catboost(X_tr, y_tr, w_tr, X_val, y_val, pt_encoder, max_iters=2000):
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in CAT_COLS]

    # Build dynamic class weights
    class_factors = {"FS": 2, "OTHER": 2, "KC": 1.5, "FC": 1.3}
    cat_weights = [class_factors.get(cls, 1.0) for cls in pt_encoder.classes_]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        iterations=max_iters,
        random_state=42,
        od_type="Iter",
        od_wait=200,
        verbose=250,
        task_type="GPU" if USE_GPU else "CPU",
        devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        class_weights=cat_weights,
    )
    model.fit(
        Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        use_best_model=True,
        plot=False,
    )
    return model


# helper to get proba arrays in same order
def softmax(z):
    e = np.exp(z)
    return e / e.sum()


@lru_cache(maxsize=2048)
def load_moe(pid: int):
    p = MODEL_DIR / "pitcher_moe" / f"{pid}.lgb"
    return lgb.Booster(model_file=str(p)) if p.exists() else None


def apply_moe(row, base_logits):
    model = load_moe(int(row.pitcher))
    if model is None:
        return base_logits
    feats = pd.DataFrame(
        {
            "count_state": [row.count_state],
            "prev_pitch_1": [getattr(row, "prev_pitch_1", "NONE") or "NONE"],
            "balls": [row.balls],
            "strikes": [row.strikes],
            "stand": [row.stand],
            "inning_topbot": [row.inning_topbot],
        }
    )
    delta = model.predict(feats)[0]
    return 0.85 * base_logits + 0.15 * softmax(delta)


@lru_cache(maxsize=9)
def load_xwoba(pt):
    return lgb.Booster(model_file=str(MODEL_DIR / "xwoba_by_pitch" / f"{pt}.lgb"))


def expected_xwoba(row, logits, pt_classes):
    xs = row._asdict()
    feat = {k: [v] for k, v in xs.items() if k not in CURRENT_PITCH_MARKERS}
    Xrow = pd.DataFrame(feat)
    preds = []
    for j, pt in enumerate(pt_classes):
        try:
            model = load_xwoba(pt)
            preds.append(model.predict(Xrow)[0])
        except:
            preds.append(0.3)  # Default xwOBA if model fails
    return float(np.dot(logits, preds))


def predict_proba(model, X, model_type):
    if model_type == "lgb":
        return model.predict(X, num_iteration=model.best_iteration)
    if model_type == "xgb":
        # Handle both old and new XGBoost versions
        try:
            return model.predict(xgb.DMatrix(X), ntree_limit=model.best_ntree_limit)
        except AttributeError:
            return model.predict(
                xgb.DMatrix(X), iteration_range=(0, model.best_iteration)
            )
    if model_type == "cat":
        return model.predict_proba(X)
    raise ValueError


def load_outcome_heads():
    """Load Stage 1 and Stage 2 outcome prediction models."""
    stage_dir = MODEL_DIR / "stage_heads"

    stage1_path = stage_dir / "stage1.lgb"
    stage1_enc_path = stage_dir / "stage1_encoder.pkl"
    bip_path = stage_dir / "bip.lgb"
    bip_enc_path = stage_dir / "bip_encoder.pkl"

    if not stage1_path.exists():
        print("⚠️  Stage head models not found. Train them with:")
        print(
            "   python scripts/train_outcome_heads.py --train-years 2023 --val-range 2024-04-01:2024-04-15"
        )
        return None, None, None, None

    # Load Stage 1 model
    stage1_model = lgb.Booster(model_file=str(stage1_path))
    with open(stage1_enc_path, "rb") as f:
        stage1_encoder = pickle.load(f)

    # Load Stage 2 model if available
    bip_model = None
    bip_encoder = None
    if bip_path.exists():
        bip_model = lgb.Booster(model_file=str(bip_path))
        with open(bip_enc_path, "rb") as f:
            bip_encoder = pickle.load(f)

    return stage1_model, stage1_encoder, bip_model, bip_encoder


def outcome_probs(row_logits, stage1_model, stage1_encoder, bip_model, bip_encoder):
    """
    Predict outcome probabilities using two-stage approach.

    Returns: np.array with [7 BIP probabilities, BALL prob, STRIKE prob] = 9-element vector
    """
    # Stage 1: Predict IN_PLAY/BALL/STRIKE
    stage1_probs = stage1_model.predict(row_logits.reshape(1, -1))[0]

    # Get class indices
    class_names = stage1_encoder.classes_
    in_play_idx = np.where(class_names == "IN_PLAY")[0]
    ball_idx = np.where(class_names == "BALL")[0]
    strike_idx = np.where(class_names == "STRIKE")[0]

    # Stage 2: If IN_PLAY, predict BIP outcome
    if (
        len(in_play_idx) > 0
        and stage1_probs.argmax() == in_play_idx[0]
        and bip_model is not None
    ):
        bip_probs = bip_model.predict(row_logits.reshape(1, -1))[0]
        # Scale BIP probabilities by IN_PLAY probability
        bip_probs = bip_probs * stage1_probs[in_play_idx[0]]
    else:
        # No IN_PLAY prediction or no BIP model, return zeros for BIP
        bip_probs = np.zeros(7)

    # Extract BALL and STRIKE probabilities
    ball_prob = stage1_probs[ball_idx[0]] if len(ball_idx) > 0 else 0.0
    strike_prob = stage1_probs[strike_idx[0]] if len(strike_idx) > 0 else 0.0

    # Concatenate: [7 BIP, BALL, STRIKE] = 9 elements
    out_vec = np.concatenate([bip_probs, [ball_prob, strike_prob]])

    return out_vec


# --------------------------------------------------------------------------- #
#  3.  MAIN  ENTRY
# --------------------------------------------------------------------------- #
def cmd_build(args):
    for year in args.years:
        build_season(year)


def cmd_optuna(args):
    """Run Optuna hyperparameter optimization for LightGBM."""
    print("🔍 Running Optuna hyperparameter optimization...")

    # Import and run Optuna optimization
    import subprocess
    import sys

    # Build command for Optuna script
    cmd = (
        [sys.executable, "scripts/optuna_lgb.py", "--train-years"]
        + [str(y) for y in args.train_years]
        + ["--val-range", args.val, "--trials", str(args.trials)]
    )

    if args.toy:
        cmd.append("--toy")

    # Run Optuna optimization
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("✅ Optuna optimization completed successfully!")
        print("📊 Optimized parameters saved to models/optuna_lgb.json")
    else:
        print("❌ Optuna optimization failed!")
        sys.exit(1)


def cmd_train(args):
    """Run end-to-end training pipeline for the pitch-type model."""
    t0 = time.time()
    train_years = [int(y) for y in args.train_years]

    if args.toy:
        print("🧪 TOY MODE: capping iters & skipping blend grid")
        MAX_ITERS = 200
    else:
        MAX_ITERS = 2000

    # Determine years for validation and test sets from their date ranges
    val_year = int(args.val.split(":")[0].split("-")[0])
    test_year = int(args.test.split(":")[0].split("-")[0])

    # ---- 1. Load Data ----
    print(f"⏳ Loading training data for years: {train_years} ({args.train_range})")
    df_tr = load_parquets(train_years, args.train_range)
    print(f"⏳ Loading validation data from: {args.val}")
    df_val = load_parquets([val_year], args.val)
    print(f"⏳ Loading test data from: {args.test}")
    df_test = load_parquets([test_year], args.test)

    # ---- Temporal weighting ----
    latest_date = pd.to_datetime(df_tr["game_date"].max())
    df_tr = add_temporal_weight(df_tr, latest_date, args.decay)

    # ---- 1.5. Add family probabilities ----
    print("🏗️  Adding pitch family probabilities...")
    df_tr = add_family_probs(df_tr)
    df_val = add_family_probs(df_val)
    df_test = add_family_probs(df_test)

    # ---- 2. Prep data (feature filter, encode labels) ----
    print("⚙️  Preprocessing and encoding data...")
    X_tr, y_tr, w_tr, enc = prep_balanced(df_tr)
    X_val, y_val, w_val, _ = prep_balanced(df_val, enc)
    X_test, y_test, w_test, _ = prep_balanced(df_test, enc)

    # Apply sampling if requested
    if args.sample_frac and args.sample_frac < 1.0:
        print(f"🎲 Sampling {args.sample_frac:.1%} of training data")
        sample_size = int(len(X_tr) * args.sample_frac)
        sample_idx = np.random.RandomState(42).choice(
            len(X_tr), sample_size, replace=False
        )
        X_tr = X_tr.iloc[sample_idx]
        y_tr = y_tr[sample_idx]
        w_tr = w_tr.iloc[sample_idx]

    # ---- 3. Train base models ----
    print("💪 Training base models...")

    # Create decay-adjusted, class-weighted vector for XGBoost
    pt_enc = enc["target"]  # This is the pitch type encoder
    class_factors = {"FS": 2, "OTHER": 2, "KC": 1.5, "FC": 1.3}
    class_weights = np.array(
        [
            class_factors.get(pt_enc.inverse_transform([i])[0], 1.0)
            for i in range(len(pt_enc.classes_))
        ]
    )
    w_tr_final = w_tr * class_weights[y_tr]

    models = {}
    models["lgb"] = train_lightgbm(X_tr, y_tr, w_tr, X_val, y_val, MAX_ITERS)
    models["xgb"] = train_xgboost(X_tr, y_tr, w_tr_final, X_val, y_val, MAX_ITERS)
    models["cat"] = train_catboost(X_tr, y_tr, w_tr, X_val, y_val, pt_enc, MAX_ITERS)

    # ---- 4. Find optimal blend weights ----
    print("⚖️  Finding optimal blend weights...")
    best_logloss = float("inf")
    best_weights = None

    if args.toy:
        blend_grid = [{"lgb": 0.6, "xgb": 0.3, "cat": 0.1}]
    else:
        blend_grid = list(
            ParameterGrid(
                {
                    "lgb": np.arange(0.1, 1.0, 0.1),
                    "xgb": np.arange(0.1, 1.0, 0.1),
                    "cat": np.arange(0.1, 1.0, 0.1),
                }
            )
        )

    for weights in blend_grid:
        if sum(weights.values()) != 1.0:
            continue

        y_val_prob = (
            weights["lgb"] * predict_proba(models["lgb"], X_val, "lgb")
            + weights["xgb"] * predict_proba(models["xgb"], X_val, "xgb")
            + weights["cat"] * predict_proba(models["cat"], X_val, "cat")
        )

        loss = log_loss(y_val, y_val_prob)
        if loss < best_logloss:
            best_logloss = loss
            best_weights = weights

    print(f"✅ Best tree ensemble weights: {best_weights} (logloss: {best_logloss:.4f})")

    # ---- 4.5. GRU Ensemble Integration ----
    gru_val_path = MODEL_DIR / "gru_logits_val.npy"
    gru_test_path = MODEL_DIR / "gru_logits_test.npy"

    if gru_val_path.exists() and gru_test_path.exists():
        print("🧠 Integrating GRU model into ensemble...")

        # Load GRU logits
        gru_val_logits = np.load(gru_val_path)
        gru_test_logits = np.load(gru_test_path)

        # Convert logits to probabilities
        gru_val_probs = sp_softmax(gru_val_logits, axis=1)
        gru_test_probs = sp_softmax(gru_test_logits, axis=1)

        # Get tree ensemble probabilities
        tree_val_probs = (
            best_weights["lgb"] * predict_proba(models["lgb"], X_val, "lgb")
            + best_weights["xgb"] * predict_proba(models["xgb"], X_val, "xgb")
            + best_weights["cat"] * predict_proba(models["cat"], X_val, "cat")
        )

        # Ensure we have matching number of samples (GRU might have fewer due to filtering)
        min_val_samples = min(len(tree_val_probs), len(gru_val_probs))
        tree_val_probs = tree_val_probs[:min_val_samples]
        gru_val_probs = gru_val_probs[:min_val_samples]
        y_val_gru = y_val[:min_val_samples]

        # Find optimal tree-GRU blend weights
        print("🔍 Searching for optimal tree-GRU blend weights...")
        gru_blend_weights = [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4)]
        best_gru_logloss = float("inf")
        best_gru_weights = None

        for tree_weight, gru_weight in gru_blend_weights:
            # Blend tree and GRU predictions
            # Handle dimension mismatch: tree has 10 classes, GRU has 9 (no OTHER)
            if tree_val_probs.shape[1] > gru_val_probs.shape[1]:
                # Pad GRU probabilities with zeros for missing classes
                padding = np.zeros(
                    (
                        gru_val_probs.shape[0],
                        tree_val_probs.shape[1] - gru_val_probs.shape[1],
                    )
                )
                gru_val_probs_padded = np.concatenate([gru_val_probs, padding], axis=1)
            else:
                gru_val_probs_padded = gru_val_probs

            blended_probs = (
                tree_weight * tree_val_probs + gru_weight * gru_val_probs_padded
            )
            # Use consistent class labels for log_loss evaluation
            labels = list(
                range(tree_val_probs.shape[1])
            )  # Use all classes from tree model
            loss = log_loss(y_val_gru, blended_probs, labels=labels)

            print(
                f"   Tree: {tree_weight:.1f}, GRU: {gru_weight:.1f} -> LogLoss: {loss:.4f}"
            )

            if loss < best_gru_logloss:
                best_gru_logloss = loss
                best_gru_weights = (tree_weight, gru_weight)

        print(
            f"✅ Best tree-GRU weights: Tree={best_gru_weights[0]:.1f}, GRU={best_gru_weights[1]:.1f} (logloss: {best_gru_logloss:.4f})"
        )

        # Store GRU info for test evaluation
        gru_info = {
            "weights": best_gru_weights,
            "test_probs": gru_test_probs,
            "available": True,
        }
    else:
        print("⚠️  GRU logits not found, using tree ensemble only")
        gru_info = {"available": False}

    # ---- 5. Evaluate on Test set ----
    print("🧪 Evaluating on test set...")

    # Get tree ensemble predictions
    tree_test_probs = (
        best_weights["lgb"] * predict_proba(models["lgb"], X_test, "lgb")
        + best_weights["xgb"] * predict_proba(models["xgb"], X_test, "xgb")
        + best_weights["cat"] * predict_proba(models["cat"], X_test, "cat")
    )

    if gru_info["available"]:
        # Blend with GRU
        gru_test_probs = gru_info["test_probs"]
        tree_weight, gru_weight = gru_info["weights"]

        # Handle dimension mismatch for test set
        min_test_samples = min(len(tree_test_probs), len(gru_test_probs))
        tree_test_probs = tree_test_probs[:min_test_samples]
        gru_test_probs = gru_test_probs[:min_test_samples]
        y_test_final = y_test[:min_test_samples]

        if tree_test_probs.shape[1] > gru_test_probs.shape[1]:
            padding = np.zeros(
                (
                    gru_test_probs.shape[0],
                    tree_test_probs.shape[1] - gru_test_probs.shape[1],
                )
            )
            gru_test_probs_padded = np.concatenate([gru_test_probs, padding], axis=1)
        else:
            gru_test_probs_padded = gru_test_probs

        y_test_prob = tree_weight * tree_test_probs + gru_weight * gru_test_probs_padded
        print(
            f"🧠 Using Tree-GRU ensemble: Tree={tree_weight:.1f}, GRU={gru_weight:.1f}"
        )
    else:
        y_test_prob = tree_test_probs
        y_test_final = y_test
        print("🌳 Using tree ensemble only")

    test_acc = accuracy_score(y_test_final, y_test_prob.argmax(axis=1))
    # Use consistent class labels for log_loss evaluation
    labels = list(range(y_test_prob.shape[1]))
    test_logloss = log_loss(y_test_final, y_test_prob, labels=labels)

    print(f"\n🎯 FINAL TEST RESULTS")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Log-Loss: {test_logloss:.4f}")
    if gru_info["available"]:
        print(f"   Ensemble: Tree + GRU")
    else:
        print(f"   Ensemble: Tree only")

    # ---- 5.5. Outcome Prediction Evaluation ----
    stage1_model, stage1_encoder, bip_model, bip_encoder = load_outcome_heads()

    if stage1_model is not None:
        print("\n🎯 OUTCOME PREDICTION RESULTS")

        # Get true outcomes for test set
        test_outcomes = []
        for _, row in df_test.iloc[: len(y_test_final)].iterrows():
            outcome = map_outcome(
                row.get("events", ""),
                row.get("description", ""),
                row.get("pitch_number", 1),
                row.get("at_bat_number", 1),
            )
            test_outcomes.append(outcome)

        # Generate outcome predictions and collect metrics
        stage1_preds = []
        bip_preds = []
        expected_run_values = []

        print("🔄 Generating outcome predictions...")
        for i in range(len(y_test_prob)):
            # Get outcome probabilities for this pitch
            row_logits = y_test_prob[i : i + 1]  # Keep 2D shape
            out_probs = outcome_probs(
                row_logits, stage1_model, stage1_encoder, bip_model, bip_encoder
            )

            # Stage 1 prediction (IN_PLAY/BALL/STRIKE)
            true_outcome = test_outcomes[i]
            if true_outcome in BIP_CLASSES:
                stage1_true = "IN_PLAY"
            elif true_outcome == "BALL":
                stage1_true = "BALL"
            else:
                stage1_true = "STRIKE"
            stage1_preds.append(stage1_true)

            # Ball-in-play prediction
            if true_outcome in BIP_CLASSES:
                bip_preds.append(true_outcome)

            # Expected run value calculation
            # out_probs = [HR, 3B, 2B, 1B, FC, SAC, OUT, BALL, STRIKE]
            bip_labels = BIP_CLASSES
            stage1_labels = ["BALL", "STRIKE"]

            expected_runs = 0.0
            for j, bip_label in enumerate(bip_labels):
                expected_runs += out_probs[j] * RUN_VALUE[bip_label]
            for j, stage1_label in enumerate(stage1_labels):
                expected_runs += out_probs[7 + j] * RUN_VALUE[stage1_label]

            expected_run_values.append(expected_runs)

        # Stage 1 metrics (IN_PLAY vs others)
        stage1_true_labels = [
            test_outcomes[i] if test_outcomes[i] in ["BALL", "STRIKE"] else "IN_PLAY"
            for i in range(len(test_outcomes))
        ]
        stage1_pred_probs = []

        for i in range(len(y_test_prob)):
            row_logits = y_test_prob[i : i + 1]
            stage1_prob_vec = stage1_model.predict(row_logits)[0]
            stage1_pred_probs.append(stage1_prob_vec)

        stage1_pred_probs = np.array(stage1_pred_probs)

        # Calculate Stage 1 AUC for IN_PLAY detection
        stage1_binary_true = [
            1 if label == "IN_PLAY" else 0 for label in stage1_true_labels
        ]
        in_play_class_idx = np.where(stage1_encoder.classes_ == "IN_PLAY")[0]

        if len(in_play_class_idx) > 0:
            in_play_probs = stage1_pred_probs[:, in_play_class_idx[0]]
            stage1_auc = roc_auc_score(stage1_binary_true, in_play_probs)
            print(f"   Stage 1 AUC (IN_PLAY detection): {stage1_auc:.4f}")

        # Ball-in-play top-3 accuracy
        if bip_model is not None and len(bip_preds) > 0:
            bip_correct_count = 0
            bip_total = 0

            for i in range(len(y_test_prob)):
                true_outcome = test_outcomes[i]
                if true_outcome in BIP_CLASSES:
                    row_logits = y_test_prob[i : i + 1]
                    bip_pred_probs = bip_model.predict(row_logits)[0]
                    top3_classes = bip_encoder.classes_[np.argsort(bip_pred_probs)[-3:]]

                    if true_outcome in top3_classes:
                        bip_correct_count += 1
                    bip_total += 1

            if bip_total > 0:
                bip_top3_acc = bip_correct_count / bip_total
                print(
                    f"   BIP Top-3 Accuracy: {bip_top3_acc:.4f} ({bip_correct_count}/{bip_total})"
                )

        # Expected run value
        mean_expected_runs = np.mean(expected_run_values)
        print(f"   Mean Expected Run Value: {mean_expected_runs:.4f}")

    else:
        print("\n⚠️  Outcome head models not available for evaluation")

    # ---- 6. Save models and encoders ----
    print("💾 Saving models and artifacts...")
    ts = int(time.time())
    checkpoint_dir = MODEL_DIR / f"checkpoint_{ts}"
    checkpoint_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        if name == "lgb":
            model.save_model(checkpoint_dir / f"{name}.lgb")
        elif name == "xgb":
            model.save_model(checkpoint_dir / f"{name}.xgb")
        else:  # catboost
            model.save_model(checkpoint_dir / f"{name}.cbm")

    with open(checkpoint_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(enc, f)

    with open(checkpoint_dir / "blend_weights.json", "w") as f:
        json.dump(best_weights, f)

    print(
        f"✅ Pipeline finished in {time.time() - t0:.1f}s. Models saved to {checkpoint_dir}"
    )


def cmd_train_two_models(args):
    """Train the two-model architecture: pitch type + outcome prediction"""
    from two_model_architecture import TwoModelPitchPredictor

    print("🎯 Training Two-Model Architecture")
    print("=" * 50)

    # Run the two-model pipeline
    predictor = TwoModelPitchPredictor()

    # Load data with temporal separation
    train_df, val_df, test_df = predictor.load_data_with_temporal_split()

    # Train both models
    model1_acc = predictor.train_model1(train_df, val_df)
    model2_acc = predictor.train_model2(train_df, val_df)

    # Final evaluation
    test_m1_acc, test_m2_acc = predictor.evaluate_on_test(test_df)

    print(f"\n🎯 FINAL RESULTS:")
    print(f"Model 1 (Pitch Type): {test_m1_acc:.1%} accuracy")
    print(f"Model 2 (Outcome): {test_m2_acc:.1%} accuracy")

    # Save models
    import pickle

    with open("models/two_model_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)
    print("✅ Models saved to models/two_model_predictor.pkl")


def main():
    """Main function to parse arguments and call the appropriate command."""
    parser = argparse.ArgumentParser(description="MLB Pitch Prediction Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Build command ---
    p_build = subparsers.add_parser("build", help="Build historical features")
    p_build.add_argument("years", nargs="+", type=int, help="Years to build")
    p_build.set_defaults(func=cmd_build)

    # --- Optuna command ---
    p_optuna = subparsers.add_parser(
        "optuna", help="Run Optuna hyperparameter optimization"
    )
    p_optuna.add_argument(
        "--train-years",
        nargs="+",
        required=True,
        help="List of years for training data",
    )
    p_optuna.add_argument(
        "--val",
        required=True,
        help="Validation date range (e.g., YYYY-MM-DD:YYYY-MM-DD)",
    )
    p_optuna.add_argument(
        "--trials", type=int, default=20, help="Number of Optuna trials (default: 20)"
    )
    p_optuna.add_argument(
        "--toy", action="store_true", help="Use toy mode for fast optimization"
    )
    p_optuna.set_defaults(func=cmd_optuna)

    # --- Train command ---
    p_train = subparsers.add_parser("train", help="Train the model ensemble")
    p_train.add_argument(
        "--train-years",
        nargs="+",
        required=True,
        help="List of years for training data",
    )
    p_train.add_argument(
        "--train-range",
        required=True,
        help="Training date range (e.g., YYYY-MM-DD:YYYY-MM-DD)",
    )
    p_train.add_argument(
        "--val",
        required=True,
        help="Validation date range (e.g., YYYY-MM-DD:YYYY-MM-DD)",
    )
    p_train.add_argument(
        "--test", required=True, help="Test date range (e.g., YYYY-MM-DD:YYYY-MM-DD)"
    )
    p_train.add_argument(
        "--decay", type=float, default=DECAY_DEFAULT, help="Lambda for temporal decay"
    )
    p_train.add_argument(
        "--sample-frac", type=float, help="Fraction of data to sample for testing"
    )
    p_train.add_argument(
        "--toy",
        action="store_true",
        help="Drastically cut iterations / grid for fast smoke run",
    )
    p_train.set_defaults(func=cmd_train)

    # --- Train two models command ---
    p_train_two_models = subparsers.add_parser(
        "train_two_models", help="Train the two-model architecture"
    )
    p_train_two_models.set_defaults(func=cmd_train_two_models)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
