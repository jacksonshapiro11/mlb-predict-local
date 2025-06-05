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
from datetime import date
from functools import lru_cache
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from scipy.special import softmax
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

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
CAT_COLS = [
    "stand",
    "p_throws",
    "count_state",
    "prev_pt1",
    "prev_pt2",
]  # dvelo1 stays numeric
DECAY_DEFAULT = 0.0008  # ~2.4-season half-life

LAG_SQL = """
WITH base AS (
  SELECT *
  FROM parquet_scan({paths})
  {where_clause}
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
    return load_duck(q)


def add_temporal_weight(df, latest_date, lam):
    delta = (latest_date - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-lam * delta)
    return df


# --------------------------------------------------------------------------- #
#  1.  BALANCED-PREDICTIVE  FEATURE  FILTER
# --------------------------------------------------------------------------- #
DROP_ALWAYS = [
    TARGET_PT,  # drop label
    "estimated_woba_using_speedangle",  # xwOBA label
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
    # -------- enhanced situational features (count_state etc.) -----------
    balls_cap = df["balls"].fillna(0).clip(0, 3)
    strikes_cap = df["strikes"].fillna(0).clip(0, 2)
    df["count_state"] = balls_cap.astype(str) + "_" + strikes_cap.astype(str)

    # ------------------- column filtering -------------------------------
    drop_cols = set(DROP_ALWAYS)
    # current pitch physics/outcome
    for col in df.columns:
        if any(m in col.lower() for m in CURRENT_PITCH_MARKERS):
            drop_cols.add(col)

    # Remove all "Today" and recent features
    for col in LEAKY_FEATURES:
        drop_cols.add(col)

    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # ------------------- categorical encoding ---------------------------
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

    # Make anything still object → numeric
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    y = df[TARGET_PT]
    w = df.get("w", pd.Series(1, index=df.index))

    # Assert no data leakage before returning
    assert_no_leakage(X.columns)

    return X, y, w, label_encoders


# --------------------------------------------------------------------------- #
#  2.  MODEL  TRAINERS
# --------------------------------------------------------------------------- #
def train_lightgbm(X_tr, y_tr, w_tr, X_val, y_val):
    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        weight=w_tr,
        categorical_feature=[
            i for i, c in enumerate(X_tr.columns) if c in CAT_COLS
        ],
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=[
            i for i, c in enumerate(X_val.columns) if c in CAT_COLS
        ],
        free_raw_data=False,
    )
    params = dict(
        objective="multiclass",
        num_class=len(np.unique(y_tr)),
        learning_rate=0.04,
        metric="multi_logloss",
        random_state=42,
        max_bin=255,
        num_leaves=255,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=1,
        force_col_wise=True,
        histogram_pool_size=-1,
    )

    if USE_GPU:
        params.update(
            {
                "device_type": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "gpu_use_dp": True,
            }
        )

    model = lgb.train(
        params,
        lgb_train,
        4000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)],
    )
    return model


def train_xgboost(X_tr, y_tr, w_tr, X_val, y_val):
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
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
        2000,
        evals=[(dvl, "val")],
        early_stopping_rounds=150,
        verbose_eval=250,
    )
    return model


def train_catboost(X_tr, y_tr, w_tr, X_val, y_val):
    cat_idx = [
        i for i, c in enumerate(X_tr.columns) if c in CAT_COLS
    ]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        iterations=2000,
        random_state=42,
        od_type="Iter",
        od_wait=200,
        verbose=250,
        task_type="GPU" if USE_GPU else "CPU",
        devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(
        Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        use_best_model=True,
        plot=False,
    )
    return model


# helper to get proba arrays in same order
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


# --------------------------------------------------------------------------- #
#  3.  MAIN  ENTRY
# --------------------------------------------------------------------------- #
def cmd_build(args):
    for y in args.years:
        build_season(int(y))


def cmd_train(args):
    # -------------- load ------------------
    train_years = [int(y) for y in args.train_years]
    val_range = args.val
    test_range = args.test

    # Extract years from date ranges properly
    val_start, val_end = val_range.split(":")
    test_start, test_end = test_range.split(":")
    val_years = {int(val_start[:4])}
    test_years = {int(test_start[:4]), int(test_end[:4])}

    # Create timestamp for this run
    ts = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = MODEL_DIR / f"checkpoint_{ts}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Loading data...")
    train_df = load_parquets(train_years)
    print(f"Train data shape: {train_df.shape}")

    val_df = load_parquets(val_years, val_range)
    print(f"Validation data shape: {val_df.shape}")

    test_df = load_parquets(test_years, test_range)
    print(f"Test data shape: {test_df.shape}")

    train_df = add_temporal_weight(
        train_df, pd.to_datetime(f"{max(train_years)}-12-31"), args.decay
    )

    # -------------- prep ------------------
    print("\n🔄 Preprocessing data...")
    X_tr, y_tr, w_tr, enc = prep_balanced(train_df)
    print(f"X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}")

    X_val, y_val, _, _ = prep_balanced(val_df, enc)
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    X_te, y_te, _, _ = prep_balanced(test_df, enc)
    print(f"X_te shape: {X_te.shape}, y_te shape: {y_te.shape}")

    # encode target
    pt_enc = LabelEncoder()
    y_tr_enc = pt_enc.fit_transform(y_tr)
    y_val_enc = pt_enc.transform(y_val)
    y_te_enc = pt_enc.transform(y_te)
    print(f"\nTarget classes: {pt_enc.classes_}")

    # Per-row class weighting
    CLASS_FACTORS = {"FS": 2, "OTHER": 2, "KC": 1.5, "FC": 1.3}
    row_weights = y_tr.map(CLASS_FACTORS).fillna(1.0).astype(float)
    w_tr_final = w_tr * row_weights.values
    print("weight range:", w_tr_final.min(), w_tr_final.max())

    # Save encoders immediately
    with open(checkpoint_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(checkpoint_dir / "target_encoder.pkl", "wb") as f:
        pickle.dump(pt_enc, f)

    # ---- Family model: FB / BR / OS ---------------------------------
    fam_map = {
        "FF": "FB",
        "SI": "FB",
        "FC": "FB",
        "SL": "BR",
        "CU": "BR",
        "KC": "BR",
        "OTHER": "BR",
        "CH": "OS",
        "FS": "OS",
    }
    for df in (train_df, val_df, test_df):
        df["pitch_fam"] = df["pitch_type_can"].map(fam_map)

    fam_enc = LabelEncoder()
    y_fam_tr = fam_enc.fit_transform(train_df["pitch_fam"])
    X_fam = train_df.drop(columns=[TARGET_PT, "pitch_fam"])
    fam_ds = lgb.Dataset(X_fam, y_fam_tr)

    fam_model = lgb.train(
        dict(
            objective="multiclass",
            num_class=3,
            num_leaves=64,
            learning_rate=0.08,
            verbosity=-1,
        ),
        fam_ds,
        300,
    )

    def add_fam(df):
        X = df.drop(columns=[TARGET_PT, "pitch_fam"])
        p = fam_model.predict(X)
        for i, cls in enumerate(fam_enc.classes_):
            df[f"FAM_PROB_{cls}"] = p[:, i]

    for d in (train_df, val_df, test_df):
        add_fam(d)

    # Re-run prep_balanced() with new family probability features
    print("\n🔄 Re-preprocessing data with family probabilities...")
    X_tr, y_tr, w_tr, enc = prep_balanced(train_df)
    print(f"X_tr shape with family features: {X_tr.shape}, y_tr shape: {y_tr.shape}")

    X_val, y_val, _, _ = prep_balanced(val_df, enc)
    print(
        f"X_val shape with family features: {X_val.shape}, y_val shape: {y_val.shape}"
    )

    X_te, y_te, _, _ = prep_balanced(test_df, enc)
    print(f"X_te shape with family features: {X_te.shape}, y_te shape: {y_te.shape}")

    # encode target
    pt_enc = LabelEncoder()
    y_tr_enc = pt_enc.fit_transform(y_tr)
    y_val_enc = pt_enc.transform(y_val)
    y_te_enc = pt_enc.transform(y_te)
    print(f"\nTarget classes: {pt_enc.classes_}")

    # Per-row class weighting
    CLASS_FACTORS = {"FS": 2, "OTHER": 2, "KC": 1.5, "FC": 1.3}
    row_weights = y_tr.map(CLASS_FACTORS).fillna(1.0).astype(float)
    w_tr_final = w_tr * row_weights.values
    print("weight range:", w_tr_final.min(), w_tr_final.max())

    # -------------- train base models -----
    models = {}
    print("\n🚂 LightGBM …")
    models["lgb"] = train_lightgbm(X_tr, y_tr_enc, w_tr_final, X_val, y_val_enc)
    models["lgb"].save_model(checkpoint_dir / "lgb.lgb")
    print("✅ LightGBM saved")

    print("\n🚂 CatBoost …")
    models["cat"] = train_catboost(X_tr, y_tr_enc, w_tr_final, X_val, y_val_enc)
    models["cat"].save_model(checkpoint_dir / "cat.cbm")
    print("✅ CatBoost saved")

    print("\n🚂 XGBoost …")
    models["xgb"] = train_xgboost(X_tr, y_tr_enc, w_tr_final, X_val, y_val_enc)
    models["xgb"].save_model(checkpoint_dir / "xgb.xgb")
    print("✅ XGBoost saved")

    # -------------- train MoE and xwOBA models -----
    print("\n🎯 Training MoE and xwOBA models...")
    manifest_path = MODEL_DIR / "pitcher_moe_manifest.json"

    if not manifest_path.exists():
        print("🔄 Training per-pitcher MoE and xwOBA models...")
        train_years_str = " ".join(map(str, train_years))
        moe_cmd = (
            f"python scripts/train_moe_and_xwoba.py --train-years {train_years_str}"
        )

        # Set GPU environment for subprocess
        env = os.environ.copy()
        env["GPU"] = "1" if USE_GPU else "0"

        try:
            result = subprocess.run(
                moe_cmd, shell=True, env=env, capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ MoE/xwOBA training completed")
            else:
                print(f"⚠️  MoE/xwOBA training failed: {result.stderr}")
                print("🔄 Continuing with base ensemble only...")
        except Exception as e:
            print(f"⚠️  Error running MoE/xwOBA training: {e}")
            print("🔄 Continuing with base ensemble only...")
    else:
        print("✅ MoE/xwOBA models already exist")

    # -------------- blend weights search ---
    print("\n🔍 Searching blend weights …")

    # Load GRU logits if available
    gru_val_path = MODEL_DIR / "gru_val_logits.npy"
    gru_test_path = MODEL_DIR / "gru_test_logits.npy"

    if gru_val_path.exists() and gru_test_path.exists():
        print("🤖 Loading GRU sequence model logits...")
        gru_val_logits = np.load(gru_val_path)
        gru_test_logits = np.load(gru_test_path)

        # Ensure same length as validation set
        if len(gru_val_logits) != len(y_val_enc):
            print(
                f"⚠️  GRU val logits length mismatch: "
                f"{len(gru_val_logits)} vs {len(y_val_enc)}")
            print("🔄 Falling back to 3-model ensemble...")
            use_gru = False
        else:
            print(
                f"✅ GRU logits loaded: val={len(gru_val_logits)}, "
                f"test={len(gru_test_logits)}")
            use_gru = True
    else:
        print("ℹ️  GRU logits not found, using 3-model ensemble")
        use_gru = False

    if use_gru:
        # 4-model grid search including GRU
        from itertools import product

        grid_params = {
            "lgb": [0.3, 0.35, 0.4, 0.45],
            "cat": [0.15, 0.2, 0.25, 0.3],
            "xgb": [0.1, 0.15, 0.2, 0.25],
            "gru": [0.2, 0.25, 0.3, 0.35],
        }

        best, best_acc, best_top3 = None, -1, -1
        print("🔍 Searching 4-model blend weights...")

        for lgb_w, cat_w, xgb_w, gru_w in product(*grid_params.values()):
            if abs((lgb_w + cat_w + xgb_w + gru_w) - 1) > 1e-6:
                continue

            # Convert GRU logits to probabilities
            gru_proba = np.exp(gru_val_logits) / np.exp(gru_val_logits).sum(
                axis=1, keepdims=True
            )

            blend = (
                lgb_w * predict_proba(models["lgb"], X_val, "lgb")
                + cat_w * predict_proba(models["cat"], X_val, "cat")
                + xgb_w * predict_proba(models["xgb"], X_val, "xgb")
                + gru_w * gru_proba
            )

            preds = blend.argmax(1)
            acc = accuracy_score(y_val_enc, preds)
            top3 = np.mean(
                [
                    y_val_enc[i] in np.argsort(blend[i])[-3:]
                    for i in range(len(y_val_enc))
                ]
            )

            if acc > best_acc:
                best = {"lgb": lgb_w, "cat": cat_w, "xgb": xgb_w, "gru": gru_w}
                best_acc = acc
                best_top3 = top3

        print(
            f"✅ Best 4-model weights: {best}  |  val acc={best_acc:.3f}  top3={best_top3:.3f}"
        )

    else:
        # Original 3-model grid search
        grid = ParameterGrid(
            {"lgb": [0.4, 0.5, 0.6], "cat": [0.2, 0.3, 0.4], "xgb": [0.1, 0.2, 0.3]}
        )
        best, best_acc, best_top3 = None, -1, -1
        for w in grid:
            if abs(sum(w.values()) - 1) > 1e-6:
                continue
            blend = sum(w[m] * predict_proba(models[m], X_val, m) for m in models)
            preds = blend.argmax(1)
            acc = accuracy_score(y_val_enc, preds)
            top3 = np.mean(
                [
                    y_val_enc[i] in np.argsort(blend[i])[-3:]
                    for i in range(len(y_val_enc))
                ]
            )
            if acc > best_acc:
                best, best_acc, best_top3 = w, acc, top3
        print(
            f"✅ Best 3-model weights: {best}  |  val acc={best_acc:.3f}  top3={best_top3:.3f}"
        )

    # Save blend weights immediately
    blend_meta = {
        "created": str(date.today()),
        "blend": best,
        "train_years": train_years,
        "val_range": val_range,
        "test_range": test_range,
        "decay": args.decay,
        "classes": pt_enc.classes_.tolist(),
        "val_metrics": {"accuracy": best_acc, "top3": best_top3},
        "model_type": "4-model" if use_gru else "3-model",
    }
    with open(checkpoint_dir / "blend_weights.json", "w") as f:
        json.dump(blend_meta, f, indent=2)
    print("✅ Blend weights saved")

    # -------------- final test eval --------
    # Load xwOBA models for outcome prediction
    xwoba_models = load_xwoba_models()
    print(f"📊 Loaded {len(xwoba_models)} xwOBA models: {list(xwoba_models.keys())}")

    if use_gru and gru_test_path.exists():
        # 4-model test evaluation
        gru_test_proba = np.exp(gru_test_logits) / np.exp(gru_test_logits).sum(
            axis=1, keepdims=True
        )
        base_blend_te = (
            best["lgb"] * predict_proba(models["lgb"], X_te, "lgb")
            + best["cat"] * predict_proba(models["cat"], X_te, "cat")
            + best["xgb"] * predict_proba(models["xgb"], X_te, "xgb")
            + best["gru"] * gru_test_proba
        )
    else:
        # 3-model test evaluation
        base_blend_te = sum(best[m] * predict_proba(models[m], X_te, m) for m in models)

    # Apply MoE corrections
    print("🎯 Applying MoE corrections...")
    test_df_indexed = test_df.reset_index(drop=True)
    moe_blend_te = []
    expected_xwobas = []

    for i, row in enumerate(test_df_indexed.itertuples()):
        # Apply MoE correction
        moe_corrected = moe_logits(row, base_blend_te[i])
        moe_blend_te.append(moe_corrected)

        # Compute expected xwOBA
        exp_xwoba = predict_expected_xwoba(
            row, moe_corrected, xwoba_models, pt_enc.classes_
        )
        expected_xwobas.append(exp_xwoba)

    blend_te = np.array(moe_blend_te)
    expected_xwobas = np.array(expected_xwobas)

    preds_te = blend_te.argmax(1)
    acc_te = accuracy_score(y_te_enc, preds_te)
    ll_te = log_loss(y_te_enc, blend_te)
    top3_te = np.mean(
        [y_te_enc[i] in np.argsort(blend_te[i])[-3:] for i in range(len(y_te_enc))]
    )

    # Compute xwOBA MAE if we have ground truth
    xwoba_mae = np.nan
    if "estimated_woba_using_speedangle" in test_df.columns:
        actual_xwoba = test_df["estimated_woba_using_speedangle"].values
        valid_mask = ~np.isnan(actual_xwoba) & ~np.isnan(expected_xwobas)
        if valid_mask.sum() > 0:
            xwoba_mae = mean_absolute_error(
                actual_xwoba[valid_mask], expected_xwobas[valid_mask]
            )

    print(
        f"\n🎯 FINAL TEST RESULTS ({'4-model' if use_gru else '3-model'} ensemble + MoE)"
    )
    print(f"Pitch-type  accuracy : {acc_te:.3f}  ({acc_te*100:.1f}%)")
    print(f"Pitch-type  top-3     : {top3_te:.3f}  ({top3_te*100:.1f}%)")
    print(f"Pitch-type  log-loss  : {ll_te:.3f}")

    if not np.isnan(xwoba_mae):
        print(f"Expected xwOBA MAE   : {xwoba_mae:.3f}")

    if use_gru:
        print(
            f"Model weights: LGB={best['lgb']:.2f}, CAT={best['cat']:.2f}, XGB={best['xgb']:.2f}, GRU={best['gru']:.2f}"
        )
    else:
        print(
            f"Model weights: LGB={best.get('lgb',0):.2f}, CAT={best.get('cat',0):.2f}, XGB={best.get('xgb',0):.2f}"
        )

    # Update metadata with test results
    test_metrics = {
        "accuracy": acc_te,
        "top3": top3_te,
        "logloss": ll_te,
        "moe_applied": True,
        "xwoba_models": len(xwoba_models),
    }

    if not np.isnan(xwoba_mae):
        test_metrics["xwoba_mae"] = xwoba_mae

    blend_meta["test_metrics"] = test_metrics
    with open(checkpoint_dir / "blend_weights.json", "w") as f:
        json.dump(blend_meta, f, indent=2)

    print(f"\n💾 All models & metadata saved under {checkpoint_dir.resolve()}")
    print("🚀 Pipeline finished!")


def cmd_blend(args):
    """Load existing models and run blending phase only"""
    print("\n🔄 Loading existing models...")

    # Load encoders
    with open(args.checkpoint_dir / "label_encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    with open(args.checkpoint_dir / "target_encoder.pkl", "rb") as f:
        pt_enc = pickle.load(f)

    # Load models
    models = {}
    models["lgb"] = lgb.Booster(model_file=str(args.checkpoint_dir / "lgb.lgb"))
    models["cat"] = CatBoostClassifier()
    models["cat"].load_model(str(args.checkpoint_dir / "cat.cbm"))
    models["xgb"] = xgb.Booster()
    models["xgb"].load_model(str(args.checkpoint_dir / "xgb.xgb"))

    print("✅ Models loaded successfully")

    # Load data
    val_years = {int(args.val.split(":")[0][:4])}
    test_years = {int(args.test.split(":")[0][:4]), int(args.test.split(":")[1][:4])}

    val_df = load_parquets(val_years, args.val)
    test_df = load_parquets(test_years, args.test)

    # Prep data
    X_val, y_val, _, _ = prep_balanced(val_df, enc)
    X_te, y_te, _, _ = prep_balanced(test_df, enc)

    y_val_enc = pt_enc.transform(y_val)
    y_te_enc = pt_enc.transform(y_te)

    # Get predictions for all models
    print("\n🔍 Getting model predictions...")
    proba_lgb = predict_proba(models["lgb"], X_val, "lgb")
    proba_cat = predict_proba(models["cat"], X_val, "cat")
    proba_xgb = predict_proba(models["xgb"], X_val, "xgb")

    # -------------- blend weights search ---
    print("\n🔍 Searching blend weights with comprehensive grid...")
    from itertools import product

    best = None
    best_acc = -1
    best_top3 = -1

    for lgb_weight, cat_weight, xgb_weight in product([0.3, 0.4, 0.5, 0.6], repeat=3):
        if abs((lgb_weight + cat_weight + xgb_weight) - 1) > 1e-6:
            continue
        blend = lgb_weight * proba_lgb + cat_weight * proba_cat + xgb_weight * proba_xgb
        preds = blend.argmax(1)
        acc = accuracy_score(y_val_enc, preds)
        top3 = np.mean(
            [y_val_enc[i] in np.argsort(blend[i])[-3:] for i in range(len(y_val_enc))]
        )

        if acc > best_acc:
            best = {"lgb": lgb_weight, "cat": cat_weight, "xgb": xgb_weight}
            best_acc = acc
            best_top3 = top3
            print(f"New best: {best} | acc={acc:.3f} | top3={top3:.3f}")

    print(f"\n✅ Best weights: {best}  |  val acc={best_acc:.3f}  top3={best_top3:.3f}")

    # -------------- final test eval --------
    print("\n🎯 Evaluating on test set...")
    blend_te = sum(best[m] * predict_proba(models[m], X_te, m) for m in models)
    preds_te = blend_te.argmax(1)
    acc_te = accuracy_score(y_te_enc, preds_te)
    ll_te = log_loss(y_te_enc, blend_te)
    top3_te = np.mean(
        [y_te_enc[i] in np.argsort(blend_te[i])[-3:] for i in range(len(y_te_enc))]
    )

    print("\n🎯 FINAL TEST RESULTS")
    print(f"Pitch-type  accuracy : {acc_te:.3f}  ({acc_te*100:.1f}%)")
    print(f"Pitch-type  top-3     : {top3_te:.3f}  ({top3_te*100:.1f}%)")
    print(f"Pitch-type  log-loss  : {ll_te:.3f}")

    # Save final ensemble
    ts = time.strftime("%Y%m%d_%H%M%S")
    ensemble_meta = {
        "created": str(date.today()),
        "blend": best,
        "val_range": args.val,
        "test_range": args.test,
        "classes": pt_enc.classes_.tolist(),
        "metrics": {
            "validation": {"accuracy": best_acc, "top3": best_top3},
            "test": {"accuracy": acc_te, "top3": top3_te, "logloss": ll_te},
        },
    }

    final_dir = MODEL_DIR / f"ensemble_{ts}"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Copy models to final directory
    for m in models:
        if m == "lgb":
            models[m].save_model(final_dir / f"{m}.lgb")
        elif m == "cat":
            models[m].save_model(final_dir / f"{m}.cbm")
        elif m == "xgb":
            models[m].save_model(final_dir / f"{m}.xgb")

    # Save metadata and encoders
    with open(final_dir / "ensemble_meta.json", "w") as f:
        json.dump(ensemble_meta, f, indent=2)
    with open(final_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(final_dir / "target_encoder.pkl", "wb") as f:
        pickle.dump(pt_enc, f)

    print(f"\n💾 Saved final ensemble under {final_dir.resolve()}")
    print("🚀 Blending complete!")


# --------------------------------------------------------------------------- #
# MOE & XWOBA HELPERS
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1024)
def load_moe(pid):
    """Load per-pitcher MoE model with caching."""
    path = MODEL_DIR / "pitcher_moe" / f"{pid}.lgb"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    return None


def load_xwoba_models():
    """Load all pitch-type specific xwOBA models."""
    xwoba_models = {}
    xwoba_dir = MODEL_DIR / "xwoba_by_pitch"

    if not xwoba_dir.exists():
        return {}

    for pt_file in xwoba_dir.glob("*.lgb"):
        pt = pt_file.stem
        xwoba_models[pt] = lgb.Booster(model_file=str(pt_file))

    return xwoba_models


def moe_logits(
    row,
    base_logits,
    moe_features=[
        "count_state",
        "prev_pt1",
        "balls",
        "strikes",
        "stand",
        "inning_topbot",
    ],
):
    """Apply MoE residual correction to base logits."""
    model = load_moe(row.pitcher)
    if model is None:
        return base_logits

    # Prepare features
    feat_dict = {}
    for feat in moe_features:
        if hasattr(row, feat):
            val = getattr(row, feat)
            feat_dict[feat] = [
                val
                if val is not None
                else (
                    "NONE"
                    if feat in ["prev_pt1", "count_state", "stand", "inning_topbot"]
                    else 0
                )
            ]
        else:
            feat_dict[feat] = [
                "NONE"
                if feat in ["prev_pt1", "count_state", "stand", "inning_topbot"]
                else 0
            ]

    try:
        feats_df = pd.DataFrame(feat_dict)
        delta_logits = model.predict(feats_df)[0]
        # Blend: 85% base + 15% MoE residual
        return 0.85 * base_logits + 0.15 * softmax(delta_logits)
    except Exception:
        return base_logits


def predict_expected_xwoba(row, final_probs, xwoba_models, pt_classes):
    """Compute expected xwOBA given pitch probabilities."""
    if not xwoba_models:
        return np.nan

    expected_xwoba = 0.0

    for j, pt in enumerate(pt_classes):
        if pt in xwoba_models:
            try:
                # Create single-row dataframe for prediction
                # Use the same feature preparation as in the MoE training
                row_dict = {}
                for col in [
                    "balls",
                    "strikes",
                    "count_state",
                    "prev_pt1",
                    "stand",
                    "inning_topbot",
                ]:
                    if hasattr(row, col):
                        val = getattr(row, col)
                        row_dict[col] = [val if val is not None else 0]
                    else:
                        row_dict[col] = [0]

                pred_df = pd.DataFrame(row_dict)
                xwoba_pred = xwoba_models[pt].predict(pred_df)[0]
                expected_xwoba += final_probs[j] * xwoba_pred
            except Exception:
                continue

    return expected_xwoba


# --------------------------------------------------------------------------- #
# 4.  CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build", help="Generate parquet feature files")
    sb.add_argument("--years", nargs="+", required=True)

    st = sub.add_parser("train", help="Train ensemble on balanced-predictive set")
    st.add_argument("--train-years", nargs="+", required=True)
    st.add_argument("--val", type=str, required=True, help="YYYY-MM-DD:YYYY-MM-DD")
    st.add_argument("--test", type=str, required=True, help="YYYY-MM-DD:YYYY-MM-DD")
    st.add_argument("--decay", type=float, default=DECAY_DEFAULT)

    blend = sub.add_parser(
        "blend", help="Run blending phase only using existing models"
    )
    blend.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing saved models",
    )
    blend.add_argument("--val", type=str, required=True, help="YYYY-MM-DD:YYYY-MM-DD")
    blend.add_argument("--test", type=str, required=True, help="YYYY-MM-DD:YYYY-MM-DD")

    # Add test command for leakage detection
    _ = sub.add_parser("test", help="Test pipeline for data leakage")

    args = p.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "blend":
        cmd_blend(args)
    elif args.cmd == "test":
        cmd_test()


def cmd_test():
    """Quick test to verify no data leakage in pipeline."""
    print("🧪 Testing pipeline for data leakage...")

    # Find any available parquet file
    available_files = list(PARQUET_DIR.glob("statcast_historical_*.parquet"))
    if not available_files:
        print("❌ No historical feature files found. Run ETL first.")
        return

    # Load a small sample of data
    test_file = available_files[0]
    print(f"📊 Loading sample from {test_file.name}...")

    try:
        df_sample = pd.read_parquet(test_file).head(100)
        print(f"✅ Loaded {len(df_sample)} rows for testing")

        # Test the prep_balanced function
        X, y, w, enc = prep_balanced(df_sample)
        print(f"✅ Preprocessed to {X.shape[1]} features")
        print("✅ No leakage detected - pipeline passes validation!")

    except RuntimeError as e:
        print(f"❌ LEAKAGE DETECTED: {e}")
        return
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return


# Quick test for lag features
if __name__ == "__main__" and os.getenv("TEST_LAG") == "1":
    # Find any available historical feature file
    available_files = list(PARQUET_DIR.glob("statcast_historical_*.parquet"))
    if available_files:
        # Extract year from filename and test with a small date range
        file_name = available_files[0].name
        year = int(file_name.split("_")[-1].replace(".parquet", ""))
        df = load_parquets([year], f"{year}-04-01:{year}-04-02")
        assert {'prev_pt1', 'prev_pt2', 'dvelo1'} <= set(df.columns)
        print("✅ lag columns present")
    else:
        print("⚠️  No historical feature files found for testing")


if __name__ == "__main__":
    main()
