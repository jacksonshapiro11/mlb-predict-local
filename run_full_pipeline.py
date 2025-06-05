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
from datetime import date
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
PARQUET_DIR = pathlib.Path("data/features")
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
       LAG(pitch_type_can,1) OVER w  AS prev_pt1,
       LAG(pitch_type_can,2) OVER w  AS prev_pt2,
       release_speed - LAG(release_speed,1) OVER w  AS dvelo1
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
    pq = PARQUET_DIR / f"statcast_{year}.parquet"
    if pq.exists():
        print(f"✅ {pq} already exists – skipping build")
    else:
        run(f"python etl/build_offline_features.py {year}")


def load_duck(query: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df


def load_parquets(years, date_range: str | None = None):
    paths = [str(PARQUET_DIR / f"statcast_{y}.parquet") for y in years]
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
    return X, y, w, label_encoders


# --------------------------------------------------------------------------- #
#  2.  MODEL  TRAINERS
# --------------------------------------------------------------------------- #
def train_lightgbm(X_tr, y_tr, w_tr, X_val, y_val):
    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        weight=w_tr,
        categorical_feature=[i for i, c in enumerate(X_tr.columns) if c in CAT_COLS],
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=[i for i, c in enumerate(X_val.columns) if c in CAT_COLS],
        free_raw_data=False,
    )
    params = dict(
        objective="multiclass",
        num_class=len(np.unique(y_tr)),
        learning_rate=0.04,
        metric="multi_logloss",
        random_state=42,
        # GPU-optimized parameters
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        gpu_use_dp=True,
        max_bin=255,
        num_leaves=255,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=1,
        force_col_wise=True,  # GPU optimization
        histogram_pool_size=-1,  # Use all available memory
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
        # GPU-optimized parameters
        tree_method="gpu_hist",
        gpu_id=0,
        predictor="gpu_predictor",  # GPU prediction optimization
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
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in CAT_COLS]
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
        # GPU-optimized parameters
        task_type="GPU",
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

    # Build a Series of multiplicative factors for each row
    class_factors = {"FS": 2, "OTHER": 2, "KC": 1.5, "FC": 1.3}
    row_weights = y_tr.map(class_factors).fillna(1).astype(float)
    # combine with the exponential time-decay weights
    w_tr_final = w_tr * row_weights.values
    print(f"\nWeight statistics:")
    print(f"w_tr range: [{w_tr.min():.3f}, {w_tr.max():.3f}]")
    print(f"row_weights range: [{row_weights.min():.3f}, {row_weights.max():.3f}]")
    print(f"w_tr_final range: [{w_tr_final.min():.3f}, {w_tr_final.max():.3f}]")

    # Save encoders immediately
    with open(checkpoint_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(checkpoint_dir / "target_encoder.pkl", "wb") as f:
        pickle.dump(pt_enc, f)

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

    # -------------- blend weights search ---
    print("\n🔍 Searching blend weights …")
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
            [y_val_enc[i] in np.argsort(blend[i])[-3:] for i in range(len(y_val_enc))]
        )
        if acc > best_acc:
            best, best_acc, best_top3 = w, acc, top3
    print(f"✅ Best weights: {best}  |  val acc={best_acc:.3f}  top3={best_top3:.3f}")

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
    }
    with open(checkpoint_dir / "blend_weights.json", "w") as f:
        json.dump(blend_meta, f, indent=2)
    print("✅ Blend weights saved")

    # -------------- final test eval --------
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

    # Update metadata with test results
    blend_meta["test_metrics"] = {"accuracy": acc_te, "top3": top3_te, "logloss": ll_te}
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

    args = p.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    if args.cmd == "train":
        cmd_train(args)
    if args.cmd == "blend":
        cmd_blend(args)


if __name__ == "__main__":
    main()
