import glob
import pathlib
import json
import lightgbm as lgb
import duckdb
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------- configuration ----------
SEASONS_TRAIN = list(range(2018, 2024))  # 2018-2023
SEASONS_VAL = [(2024, "2024-04-01", "2024-07-31")]  # early 2024 for val
SEASON_TEST = [
    (2024, "2024-08-01", "2024-10-31"),
    (2025, "2025-01-01", "2100-01-01"),
]  # YTD
TARGET_PT = "pitch_type_can"
TARGET_XWOBA = "estimated_woba_using_speedangle"
CAT_COLS = ["stand", "p_throws"]  # Removed inning_topbot as it's dropped anyway
LAMBDA_DECAY = 0.0012  # ≈ 2-season half-life
PARQUET_GLOB = "data/features/statcast_*.parquet"
SAVE_DIR = pathlib.Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting MLB Prediction Model Training Pipeline (LightGBM - ULTRA CLEAN)")
print(f"📊 Training seasons: {SEASONS_TRAIN}")
print(f"🎯 Targets: {TARGET_PT} → {TARGET_XWOBA}")
print(
    "🔧 ULTRA CLEAN: Removing ALL pitch-type-specific features for pitch type prediction"
)


# ---------- helper to load parquet subset ----------
def load_parquets(files, start=None, end=None):
    con = duckdb.connect()
    files_str = str(files).replace("'", '"')  # DuckDB needs double quotes
    if start and end:
        query = (
            f"SELECT * FROM parquet_scan({files_str}) "
            f"WHERE game_date BETWEEN DATE '{start}' AND DATE '{end}'"
        )
    else:
        query = f"SELECT * FROM parquet_scan({files_str})"
    print(f"📥 Loading: {query[:80]}...")
    df = con.execute(query).df()
    con.close()
    return df


def add_weights(df, latest):
    # days since most recent pitch in *training* data
    delta = (latest - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-LAMBDA_DECAY * delta)
    print(f"⚖️  Weight range: {df['w'].min():.4f} - {df['w'].max():.4f}")
    return df


# ---------- 1. collect files ----------
files_all = sorted(glob.glob(PARQUET_GLOB))
print(f"📁 Found {len(files_all)} parquet files")

files_by_year = {}
for p in files_all:
    year = int(p.split("_")[1][:4])
    files_by_year[year] = p
    print(f"   {year}: {p}")

files_train = [files_by_year[y] for y in SEASONS_TRAIN if y in files_by_year]
latest_train_date = pd.to_datetime("2023-12-31")

print(f"\n🏋️  Training files: {len(files_train)}")

# ---------- 2. load sets ----------
print("\n📊 Loading training data...")
train = load_parquets(files_train)
train = add_weights(train, latest_train_date)
print(f"✅ Training: {len(train):,} rows")

print("\n📊 Loading validation data...")
val_frames = []
for y, s, e in SEASONS_VAL:
    if y in files_by_year:
        val_frames.append(load_parquets([files_by_year[y]], s, e))
val = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
print(f"✅ Validation: {len(val):,} rows")

print("\n📊 Loading test data...")
test_frames = []
for y, s, e in SEASON_TEST:
    if y in files_by_year:
        test_frames.append(load_parquets([files_by_year[y]], s, e))
test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
print(f"✅ Test: {len(test):,} rows")


# ---------- 3. data prep & feature engineering ----------
def prep_X_y(df, target, label_encoders=None, for_pitch_type=False):
    print(f"\n🔧 Preparing features for: {target}")

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
        "pitch_type",
    ]

    # For pitch type prediction: Remove ALL pitch-type-specific features
    if for_pitch_type:
        print("🚨 ULTRA CLEAN MODE: Removing pitch-type-specific features")
        pitch_type_suffixes = [
            "_CH",
            "_CU",
            "_FC",
            "_FF",
            "_FS",
            "_KC",
            "_OTHER",
            "_SI",
            "_SL",
        ]
        pitch_type_features = []
        for col in df.columns:
            if any(suffix in col for suffix in pitch_type_suffixes):
                pitch_type_features.append(col)
        drop_cols.extend(pitch_type_features)
        print(f"   Removing {len(pitch_type_features)} pitch-type-specific features")

    # For xwOBA prediction: Remove xwOBA-derived features
    if target == TARGET_XWOBA:
        xwoba_features = [col for col in df.columns if "xwoba" in col.lower()]
        drop_cols.extend(xwoba_features)
        print(
            f"🚨 Removing {len(xwoba_features)} xwOBA-derived features for xwOBA prediction"
        )

    # Remove any features that contain the target name
    target_related = [
        col
        for col in df.columns
        if target.lower().replace("_", "") in col.lower().replace("_", "")
        and col != target
    ]
    drop_cols.extend(target_related)

    drop_cols = list(
        set([c for c in drop_cols if c in df.columns])
    )  # Remove duplicates
    X = df.drop(columns=drop_cols)

    print(f"🗑️  Dropped {len(drop_cols)} columns, keeping {len(X.columns)} features")

    # Handle missing values in target
    valid_mask = df[target].notna()
    X = X[valid_mask]
    y = df[target][valid_mask]

    # Encode categorical columns for LightGBM
    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col in X.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                # Fit on all unique values including NaN
                all_values = X[col].fillna("__MISSING__").astype(str)
                label_encoders[col].fit(all_values)

            # Transform
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

    print(f"✅ Final: {len(X.columns)} features, {len(X):,} samples")
    return X, y, label_encoders


print("\n🎯 HEAD A: Pitch Type Classification (ULTRA CLEAN - NO PITCH TYPE FEATURES)")
X_train_pt, y_train_pt, label_encoders = prep_X_y(train, TARGET_PT, for_pitch_type=True)
X_val_pt, y_val_pt, _ = (
    prep_X_y(val, TARGET_PT, label_encoders, for_pitch_type=True)
    if len(val) > 0
    else (None, None, None)
)

# Create label encoder for target
pt_encoder = LabelEncoder()
y_train_pt_encoded = pt_encoder.fit_transform(y_train_pt)
unique_classes = pt_encoder.classes_
print(f"📋 Pitch types: {list(unique_classes)}")

if X_val_pt is not None:
    y_val_pt_encoded = pt_encoder.transform(y_val_pt)

# Train pitch type model
lgb_train_pt = lgb.Dataset(
    X_train_pt,
    y_train_pt_encoded,
    weight=train["w"][y_train_pt.index] if "w" in train.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_pt.columns) if c in CAT_COLS],
    free_raw_data=False,
)

lgb_val_pt = None
if X_val_pt is not None:
    lgb_val_pt = lgb.Dataset(
        X_val_pt,
        y_val_pt_encoded,
        categorical_feature=[
            i for i, c in enumerate(X_val_pt.columns) if c in CAT_COLS
        ],
        reference=lgb_train_pt,
        free_raw_data=False,
    )

params_pt = dict(
    objective="multiclass",
    num_class=len(unique_classes),
    learning_rate=0.04,
    num_leaves=256,
    max_depth=-1,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    metric="multi_logloss",
    verbose=-1,
)

print("🚂 Training pitch type model...")
callbacks = []
if lgb_val_pt is not None:
    callbacks.append(lgb.early_stopping(150))
callbacks.append(lgb.log_evaluation(200))

model_pt = lgb.train(
    params_pt,
    lgb_train_pt,
    num_boost_round=4000,
    valid_sets=[lgb_val_pt] if lgb_val_pt else None,
    callbacks=callbacks,
)

print(f"✅ Pitch type model trained! Best iteration: {model_pt.best_iteration}")


# ---------- 4. create pitch-prob features for Outcome head ----------
def add_pt_probs(df, model, label_encoders, pt_encoder):
    X, _, _ = prep_X_y(
        df, TARGET_PT, label_encoders, for_pitch_type=True
    )  # Use same clean features
    proba = model.predict(X, num_iteration=model.best_iteration)

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
val_ext = (
    add_pt_probs(val, model_pt, label_encoders, pt_encoder)
    if len(val) > 0
    else pd.DataFrame()
)

print("\n🎯 HEAD B: xwOBA Regression")
X_train_x, y_train_x, _ = prep_X_y(train_ext, TARGET_XWOBA, label_encoders)
X_val_x, y_val_x, _ = (
    prep_X_y(val_ext, TARGET_XWOBA, label_encoders)
    if len(val_ext) > 0
    else (None, None, None)
)

lgb_train_x = lgb.Dataset(
    X_train_x,
    y_train_x,
    weight=train_ext["w"][y_train_x.index] if "w" in train_ext.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_x.columns) if c in CAT_COLS],
    free_raw_data=False,
)

lgb_val_x = None
if X_val_x is not None:
    lgb_val_x = lgb.Dataset(
        X_val_x,
        y_val_x,
        categorical_feature=[i for i, c in enumerate(X_val_x.columns) if c in CAT_COLS],
        reference=lgb_train_x,
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
)

print("🚂 Training xwOBA model...")
callbacks_x = []
if lgb_val_x is not None:
    callbacks_x.append(lgb.early_stopping(200))
callbacks_x.append(lgb.log_evaluation(200))

model_x = lgb.train(
    params_x,
    lgb_train_x,
    6000,
    valid_sets=[lgb_val_x] if lgb_val_x else None,
    callbacks=callbacks_x,
)

print(f"✅ xwOBA model trained! Best iteration: {model_x.best_iteration}")


# ---------- 5. evaluation on hold-out ----------
def eval_pitch(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0, 0
    X, y, _ = prep_X_y(df, TARGET_PT, label_encoders, for_pitch_type=True)
    pred_proba = model.predict(X, num_iteration=model.best_iteration)
    pred_class = pred_proba.argmax(axis=1)

    y_encoded = pt_encoder.transform(y)
    acc = accuracy_score(y_encoded, pred_class)
    ll = log_loss(y_encoded, pred_proba)
    return acc, ll


def eval_xwoba(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0
    df_ext = add_pt_probs(df, model_pt, label_encoders, pt_encoder)
    X, y, _ = prep_X_y(df_ext, TARGET_XWOBA, label_encoders)
    pred = model.predict(X, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return rmse


print("\n📊 Evaluating on test set...")
acc, ll = eval_pitch(model_pt, test, label_encoders, pt_encoder)
rmse = eval_xwoba(model_x, test, label_encoders, pt_encoder)
print("🎯 Test Results (ULTRA CLEAN - NO DATA LEAKAGE):")
print(f"   Pitch-type accuracy: {acc:.3f}")
print(f"   Pitch-type log-loss: {ll:.3f}")
print(f"   xwOBA RMSE: {rmse:.4f}")

# ---------- 6. save models ----------
print("\n📦 Saving models...")

# Save LightGBM models in native format
model_pt.save_model(SAVE_DIR / "pitch_type_clean.lgb")
model_x.save_model(SAVE_DIR / "xwoba_clean.lgb")

# Save encoders
with open(SAVE_DIR / "label_encoders_clean.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(SAVE_DIR / "pt_encoder_clean.pkl", "wb") as f:
    pickle.dump(pt_encoder, f)

# Save feature metadata
metadata = {
    "pitch_type_features": list(X_train_pt.columns),
    "xwoba_features": list(X_train_x.columns),
    "pitch_classes": list(pt_encoder.classes_),
    "training_info": {
        "train_seasons": SEASONS_TRAIN,
        "lambda_decay": LAMBDA_DECAY,
        "test_accuracy": float(acc),
        "test_logloss": float(ll),
        "test_rmse": float(rmse),
        "model_type": "LightGBM",
        "ultra_clean": True,
        "removed_pitch_type_features": True,
        "removed_xwoba_features": True,
    },
}

with open(SAVE_DIR / "model_metadata_clean.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ ULTRA CLEAN Models saved successfully!")
print(f"📁 Files saved to: {SAVE_DIR}/")
print("   - pitch_type_clean.lgb")
print("   - xwoba_clean.lgb")
print("   - label_encoders_clean.pkl")
print("   - pt_encoder_clean.pkl")
print("   - model_metadata_clean.json")

print("\n🎉 ULTRA CLEAN Training pipeline complete (NO DATA LEAKAGE)!")
print("🎯 Expected realistic accuracy: ~30-60% (vs previous 95%)")
print("🎯 This should give us proper model performance metrics")
