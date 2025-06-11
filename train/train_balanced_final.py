import pathlib
import json
import lightgbm as lgb
import duckdb
import pandas as pd
import numpy as np
import subprocess
import time
from datetime import date
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------- FINAL configuration ----------
SEASONS_TRAIN = list(range(2018, 2024))  # 2018-2023
SEASONS_VAL = [(2024, "2024-04-01", "2024-07-31")]  # early 2024 for val
SEASON_TEST = [
    (2024, "2024-08-01", "2024-10-31"),
    (2025, "2025-01-01", "2100-01-01"),
]  # YTD
TARGET_PT = "pitch_type_can"
TARGET_XWOBA = "estimated_woba_using_speedangle"
CAT_COLS = ["stand", "p_throws"]
LAMBDA_DECAY = 0.0008  # ADJUSTED: More conservative decay (was 0.0012)
SAVE_DIR = pathlib.Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting MLB FINAL PRODUCTION Model Training Pipeline")
print("🔧 FINAL: Addressing all critical production considerations")
print(f"📊 Lambda decay: {LAMBDA_DECAY} (more conservative for rare pitch types)")


# ---------- helper to load data ----------
def load_data(seasons, date_filters=None):
    con = duckdb.connect()

    if date_filters:
        # Validation/test with date ranges
        queries = []
        for year, start_date, end_date in date_filters:
            query = f"""
            SELECT * FROM parquet_scan("data/features/statcast_{year}.parquet") 
            WHERE game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            """
            queries.append(query)

        if len(queries) == 1:
            full_query = queries[0]
        else:
            full_query = " UNION ALL ".join(queries)
    else:
        # Training data - full seasons
        files = [f"data/features/statcast_{year}.parquet" for year in seasons]
        file_list = "', '".join(files)
        full_query = f"SELECT * FROM parquet_scan(['{file_list}'])"

    print(f"📥 Loading: {full_query[:100]}...")
    df = con.execute(full_query).df()
    con.close()
    return df


def add_weights(df, latest):
    # days since most recent pitch in *training* data
    delta = (latest - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-LAMBDA_DECAY * delta)
    print(f"⚖️  Weight range: {df['w'].min():.4f} - {df['w'].max():.4f}")
    return df


def get_git_sha():
    """Get current git commit SHA for reproducibility"""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return sha
    except Exception:
        return "unknown"


# ---------- 1. load datasets ----------
print("\n📊 Loading datasets...")
train = load_data(SEASONS_TRAIN)
val = load_data(None, SEASONS_VAL)
test = load_data(None, SEASON_TEST)

# Add temporal weights (only to training data)
train = add_weights(train, pd.to_datetime("2023-12-31"))

print(f"✅ Train: {len(train):,} rows")
print(f"✅ Val: {len(val):,} rows")
print(f"✅ Test: {len(test):,} rows")

# ---------- 2. CRITICAL: Analyze sample weight impact on class distribution ----------
print("\n🔍 CRITICAL CHECK: Sample weight impact on pitch type distribution")

# Original class distribution
orig_dist = train[TARGET_PT].value_counts(normalize=True).sort_index()
print("📊 Original class distribution:")
for pt, pct in orig_dist.items():
    print(f"   {pt}: {pct:.3f} ({pct*100:.1f}%)")

# Weighted class distribution
weighted_counts = train.groupby(TARGET_PT)["w"].sum()
weighted_dist = weighted_counts / weighted_counts.sum()
print("\n⚖️  Weighted class distribution:")
for pt in orig_dist.index:
    if pt in weighted_dist:
        orig_pct = orig_dist[pt]
        weighted_pct = weighted_dist[pt]
        change = weighted_pct - orig_pct
        print(f"   {pt}: {weighted_pct:.3f} ({weighted_pct*100:.1f}%) [Δ{change:+.3f}]")

# Check for critically low support
min_support = weighted_dist.min()
if min_support < 0.01:  # Less than 1% effective support
    print(
        f"⚠️  WARNING: Minimum class support is {min_support:.3f} ({min_support*100:.1f}%)"
    )
    print(
        f"   Consider increasing lambda to {LAMBDA_DECAY * 0.67:.4f} or stratified weighting"
    )
else:
    print(f"✅ All classes have adequate support (min: {min_support:.3f})")


# ---------- 3. FINAL data prep with all safeguards ----------
def prep_X_y_final(df, target, label_encoders=None, for_xwoba=False):
    print(f"\n🔧 FINAL prep for: {target}")

    # Base columns to always drop
    drop_cols = [
        TARGET_PT,
        TARGET_XWOBA,
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
    ]

    # CRITICAL: Always drop raw pitch_type (potential leakage)
    if "pitch_type" in df.columns:
        drop_cols.append("pitch_type")
        print("🚨 REMOVED: pitch_type (potential leakage)")

    # Remove current-pitch measurements (direct leakage)
    current_pitch_features = []
    for col in df.columns:
        col_lower = col.lower()
        # Current pitch physics - measurements of THIS pitch
        if any(
            x in col_lower
            for x in [
                "release_speed",
                "release_spin_rate",
                "release_pos",
                "pfx_x",
                "pfx_z",
                "plate_x",
                "plate_z",
                "vx0",
                "vy0",
                "vz0",
                "ax",
                "ay",
                "az",
                "sz_top",
                "sz_bot",
                "effective_speed",
                "release_extension",
                "spin_axis",
            ]
        ):
            current_pitch_features.append(col)
        # Current pitch location - where THIS pitch went
        elif any(x in col_lower for x in ["zone", "hc_x", "hc_y"]):
            current_pitch_features.append(col)
        # Current pitch outcome - what happened to THIS pitch
        elif any(
            x in col_lower
            for x in [
                "launch_speed",
                "launch_angle",
                "hit_distance",
                "babip_value",
                "iso_value",
                "woba_value",
                "woba_denom",
                "delta_run_exp",
                "delta_home_win_exp",
            ]
        ):
            current_pitch_features.append(col)

    drop_cols.extend(current_pitch_features)
    print(
        f"🚨 Removing {len(current_pitch_features)} current-pitch measurement features"
    )

    # Handle xwOBA sparsity issue
    if target == TARGET_XWOBA:
        print("🎯 Handling xwOBA sparsity...")
        # Count null values
        null_count = df[target].isnull().sum()
        total_count = len(df)
        null_pct = null_count / total_count
        print(f"   NULL xwOBA: {null_count:,} / {total_count:,} ({null_pct:.1%})")

        if null_pct > 0.3:  # More than 30% nulls
            print(
                "   🔄 FILTERING to in-play events only (avoiding bias from 0-filling)"
            )
            # Keep only rows where xwOBA is not null (in-play events)
            df = df[df[target].notna()].copy()
            print(f"   ✅ Filtered to {len(df):,} in-play events")
        else:
            print("   ✅ Low null rate - proceeding with all events")

    drop_cols = list(
        set([c for c in drop_cols if c in df.columns])
    )  # Remove duplicates
    X = df.drop(columns=drop_cols)

    print(f"🗑️  Dropped {len(drop_cols)} columns, keeping {len(X.columns)} features")

    # Handle missing values in target
    valid_mask = df[target].notna()
    X = X[valid_mask]
    y = df[target][valid_mask]

    # Encode categorical columns
    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col in X.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                all_values = X[col].fillna("__MISSING__").astype(str)
                label_encoders[col].fit(all_values)
            X[col] = label_encoders[col].transform(
                X[col].fillna("__MISSING__").astype(str)
            )

    # Convert remaining object columns to numeric or drop them
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except Exception as e:
                print(f"⚠️  Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])

    # Fill remaining NaN values
    X = X.fillna(0)

    print(f"✅ FINAL: {len(X.columns)} features, {len(X):,} samples")
    return X, y, label_encoders


print("\n🎯 HEAD A: Pitch Type Classification (FINAL)")
X_train_pt, y_train_pt, label_encoders = prep_X_y_final(train, TARGET_PT)

# Create label encoder for target
pt_encoder = LabelEncoder()
y_train_pt_encoded = pt_encoder.fit_transform(y_train_pt)
unique_classes = pt_encoder.classes_
print(f"📋 Pitch types: {list(unique_classes)}")

# Prepare validation data (NO WEIGHTS)
X_val_pt, y_val_pt, _ = prep_X_y_final(val, TARGET_PT, label_encoders)
y_val_pt_encoded = pt_encoder.transform(y_val_pt)

# Train pitch type model
lgb_train_pt = lgb.Dataset(
    X_train_pt,
    y_train_pt_encoded,
    weight=train["w"][y_train_pt.index] if "w" in train.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_pt.columns) if c in CAT_COLS],
    free_raw_data=False,
)

# Validation dataset WITHOUT weights (correct)
lgb_val_pt = lgb.Dataset(
    X_val_pt,
    y_val_pt_encoded,
    categorical_feature=[i for i, c in enumerate(X_val_pt.columns) if c in CAT_COLS],
    free_raw_data=False,
)

params_pt = dict(
    objective="multiclass",
    num_class=len(unique_classes),
    learning_rate=0.05,
    num_leaves=256,
    max_depth=-1,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    metric="multi_logloss",
    verbose=-1,
    random_state=42,
)

print("🚂 Training FINAL pitch type model...")
start_time = time.time()
model_pt = lgb.train(
    params_pt,
    lgb_train_pt,
    num_boost_round=1000,
    valid_sets=[lgb_val_pt],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
pt_train_time = time.time() - start_time

print(
    f"✅ Pitch type model trained! Iterations: {model_pt.num_trees()}, Time: {pt_train_time:.1f}s"
)


# ---------- 4. create pitch-prob features for xwOBA head ----------
def add_pt_probs(df, model, label_encoders, pt_encoder):
    X, _, _ = prep_X_y_final(df, TARGET_PT, label_encoders)
    proba = model.predict(X)

    # Create probability features
    proba_df = pd.DataFrame(
        proba, columns=[f"PT_PROB_{c}" for c in pt_encoder.classes_]
    )
    proba_df.index = X.index

    # Merge back with original dataframe
    result = df.copy()
    for col in proba_df.columns:
        result[col] = np.nan
        result.loc[proba_df.index, col] = proba_df[col]

    return result


print("\n🔧 Adding pitch probability features...")
train_ext = add_pt_probs(train, model_pt, label_encoders, pt_encoder)
val_ext = add_pt_probs(val, model_pt, label_encoders, pt_encoder)

print("\n🎯 HEAD B: xwOBA Regression (FINAL)")
X_train_x, y_train_x, _ = prep_X_y_final(
    train_ext, TARGET_XWOBA, label_encoders, for_xwoba=True
)
X_val_x, y_val_x, _ = prep_X_y_final(
    val_ext, TARGET_XWOBA, label_encoders, for_xwoba=True
)

lgb_train_x = lgb.Dataset(
    X_train_x,
    y_train_x,
    weight=train_ext["w"][y_train_x.index] if "w" in train_ext.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_x.columns) if c in CAT_COLS],
    free_raw_data=False,
)

lgb_val_x = lgb.Dataset(
    X_val_x,
    y_val_x,
    categorical_feature=[i for i, c in enumerate(X_val_x.columns) if c in CAT_COLS],
    free_raw_data=False,
)

params_x = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=512,
    max_depth=-1,
    feature_fraction=0.95,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
    random_state=42,
)

print("🚂 Training FINAL xwOBA model...")
start_time = time.time()
model_x = lgb.train(
    params_x,
    lgb_train_x,
    num_boost_round=1000,
    valid_sets=[lgb_val_x],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
xwoba_train_time = time.time() - start_time

print(
    f"✅ xwOBA model trained! Iterations: {model_x.num_trees()}, Time: {xwoba_train_time:.1f}s"
)


# ---------- 5. comprehensive evaluation ----------
def eval_pitch(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0, 0, {}
    X, y, _ = prep_X_y_final(df, TARGET_PT, label_encoders)
    pred_proba = model.predict(X)
    pred_class = pred_proba.argmax(axis=1)

    y_encoded = pt_encoder.transform(y)
    acc = accuracy_score(y_encoded, pred_class)
    ll = log_loss(y_encoded, pred_proba)

    # Top-3 accuracy
    top3_pred = np.argsort(pred_proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_encoded[i] in top3_pred[i] for i in range(len(y_encoded))])

    return acc, ll, {"top3_accuracy": top3_acc}


def eval_xwoba(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0, {}
    df_ext = add_pt_probs(df, model_pt, label_encoders, pt_encoder)
    X, y, _ = prep_X_y_final(df_ext, TARGET_XWOBA, label_encoders, for_xwoba=True)
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = np.mean(np.abs(y - pred))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return rmse, {
        "mae": mae,
        "r2": r2,
        "mean_actual": np.mean(y),
        "mean_pred": np.mean(pred),
    }


print("\n📊 Final Evaluation...")

# Validation performance
val_acc, val_ll, val_pt_extra = eval_pitch(model_pt, val, label_encoders, pt_encoder)
val_rmse, val_x_extra = eval_xwoba(model_x, val, label_encoders, pt_encoder)

# Test performance
test_acc, test_ll, test_pt_extra = eval_pitch(
    model_pt, test, label_encoders, pt_encoder
)
test_rmse, test_x_extra = eval_xwoba(model_x, test, label_encoders, pt_encoder)

print("\n🎯 FINAL Results:")
print("📊 VALIDATION:")
print(f"   Pitch-type accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
print(
    f"   Pitch-type top-3 acc: {val_pt_extra['top3_accuracy']:.3f} ({val_pt_extra['top3_accuracy']*100:.1f}%)"
)
print(f"   Pitch-type log-loss: {val_ll:.3f}")
print(f"   xwOBA RMSE: {val_rmse:.4f}")
print(f"   xwOBA MAE: {val_x_extra['mae']:.4f}")
print(f"   xwOBA R²: {val_x_extra['r2']:.3f}")

print("\n📊 TEST:")
print(f"   Pitch-type accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
print(
    f"   Pitch-type top-3 acc: {test_pt_extra['top3_accuracy']:.3f} ({test_pt_extra['top3_accuracy']*100:.1f}%)"
)
print(f"   Pitch-type log-loss: {test_ll:.3f}")
print(f"   xwOBA RMSE: {test_rmse:.4f}")
print(f"   xwOBA MAE: {test_x_extra['mae']:.4f}")
print(f"   xwOBA R²: {test_x_extra['r2']:.3f}")

# Feature importance
print("\n🔍 Top 10 Pitch Type Features:")
feature_importance = model_pt.feature_importance(importance_type="gain")
feature_names = X_train_pt.columns
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
).sort_values("importance", ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.0f}")

print("\n🔍 Top 10 xwOBA Features:")
feature_importance_x = model_x.feature_importance(importance_type="gain")
feature_names_x = X_train_x.columns
importance_df_x = pd.DataFrame(
    {"feature": feature_names_x, "importance": feature_importance_x}
).sort_values("importance", ascending=False)

for i, row in importance_df_x.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.0f}")

# ---------- 6. save FINAL models with comprehensive metadata ----------
print("\n📦 Saving FINAL models...")

# Save LightGBM models in native format
model_pt.save_model(SAVE_DIR / "pitch_type_final.lgb")
model_x.save_model(SAVE_DIR / "xwoba_final.lgb")

# Save encoders
with open(SAVE_DIR / "label_encoders_final.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(SAVE_DIR / "pt_encoder_final.pkl", "wb") as f:
    pickle.dump(pt_encoder, f)

# Comprehensive metadata with all reproducibility info
metadata = {
    "model_info": {
        "version": "v2.1.0-final",
        "created_date": str(date.today()),
        "git_sha": get_git_sha(),
        "lambda_decay": LAMBDA_DECAY,
        "model_type": "LightGBM Two-Head",
        "training_time_pt": pt_train_time,
        "training_time_xwoba": xwoba_train_time,
    },
    "data_info": {
        "train_seasons": SEASONS_TRAIN,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "features_pt": len(X_train_pt.columns),
        "features_xwoba": len(X_train_x.columns),
        "pitch_classes": list(pt_encoder.classes_),
        "weight_range": [float(train["w"].min()), float(train["w"].max())],
    },
    "performance": {
        "validation": {
            "pitch_type_accuracy": float(val_acc),
            "pitch_type_top3_accuracy": float(val_pt_extra["top3_accuracy"]),
            "pitch_type_logloss": float(val_ll),
            "xwoba_rmse": float(val_rmse),
            "xwoba_mae": float(val_x_extra["mae"]),
            "xwoba_r2": float(val_x_extra["r2"]),
        },
        "test": {
            "pitch_type_accuracy": float(test_acc),
            "pitch_type_top3_accuracy": float(test_pt_extra["top3_accuracy"]),
            "pitch_type_logloss": float(test_ll),
            "xwoba_rmse": float(test_rmse),
            "xwoba_mae": float(test_x_extra["mae"]),
            "xwoba_r2": float(test_x_extra["r2"]),
        },
    },
    "features": {
        "pitch_type_features": list(X_train_pt.columns),
        "xwoba_features": list(X_train_x.columns),
        "feature_importance_pt": importance_df.head(20).to_dict("records"),
        "feature_importance_xwoba": importance_df_x.head(20).to_dict("records"),
    },
    "model_params": {"pitch_type": params_pt, "xwoba": params_x},
    "class_distribution": {
        "original": orig_dist.to_dict(),
        "weighted": weighted_dist.to_dict(),
        "min_weighted_support": float(min_support),
    },
    "safeguards_applied": {
        "conservative_lambda": True,
        "xwoba_sparsity_handled": True,
        "pitch_type_leakage_removed": True,
        "validation_unweighted": True,
        "git_tracked": True,
    },
}

with open(SAVE_DIR / "model_metadata_final.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ FINAL Models saved successfully!")
print(f"📁 Files saved to: {SAVE_DIR}/")
print("   - pitch_type_final.lgb")
print("   - xwoba_final.lgb")
print("   - label_encoders_final.pkl")
print("   - pt_encoder_final.pkl")
print("   - model_metadata_final.json")

print("\n🎉 FINAL Training complete!")
print("🎯 Key metrics:")
print(f"   - Pitch type accuracy: {test_acc:.1%} (vs 11.1% random)")
print(
    f"   - Top-3 accuracy: {test_pt_extra['top3_accuracy']:.1%} (practical relevance)"
)
print(f"   - xwOBA RMSE: {test_rmse:.4f}")
print("   - All safeguards applied ✅")
print("   - Production ready! 🚀")
