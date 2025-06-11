import glob
import pathlib
import json
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
CAT_COLS = ["pitch_type", "pitch_type_can", "stand", "p_throws", "inning_topbot"]
LAMBDA_DECAY = 0.0012  # ≈ 2-season half-life
PARQUET_GLOB = "data/features/statcast_*.parquet"
SAVE_DIR = pathlib.Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting MLB Prediction Model Training Pipeline (Random Forest)")
print(f"📊 Training seasons: {SEASONS_TRAIN}")
print(f"🎯 Targets: {TARGET_PT} → {TARGET_XWOBA}")


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
def prep_X_y(df, target, label_encoders=None):
    # Remove target columns and any problematic columns
    drop_cols = [TARGET_PT, TARGET_XWOBA, "w"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

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
                # Fit on all unique values including NaN
                all_values = X[col].fillna("__MISSING__").astype(str)
                label_encoders[col].fit(all_values)

            # Transform
            X[col] = label_encoders[col].transform(
                X[col].fillna("__MISSING__").astype(str)
            )

    # Fill remaining NaN values
    X = X.fillna(0)

    print(f"🔧 Features: {len(X.columns)}, Samples: {len(X):,}")
    return X, y, label_encoders


print("\n🎯 HEAD A: Pitch Type Classification")
X_train_pt, y_train_pt, label_encoders = prep_X_y(train, TARGET_PT)

# Create label encoder for target
pt_encoder = LabelEncoder()
y_train_pt_encoded = pt_encoder.fit_transform(y_train_pt)
unique_classes = pt_encoder.classes_
print(f"📋 Pitch types: {list(unique_classes)}")

# Train pitch type model
print("🚂 Training pitch type model...")
model_pt = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

# Use sample weights for time decay
sample_weights = train["w"][y_train_pt.index] if "w" in train.columns else None
model_pt.fit(X_train_pt, y_train_pt_encoded, sample_weight=sample_weights)

print("✅ Pitch type model trained!")


# ---------- 4. create pitch-prob features for Outcome head ----------
def add_pt_probs(df, model, label_encoders, pt_encoder):
    X, _, _ = prep_X_y(
        df, TARGET_PT, label_encoders
    )  # Just to get consistent feature set
    proba = model.predict_proba(X)

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

print("\n🎯 HEAD B: xwOBA Regression")
X_train_x, y_train_x, _ = prep_X_y(train_ext, TARGET_XWOBA, label_encoders)

# Train xwOBA model
print("🚂 Training xwOBA model...")
model_x = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

# Use sample weights for time decay
sample_weights_x = train_ext["w"][y_train_x.index] if "w" in train_ext.columns else None
model_x.fit(X_train_x, y_train_x, sample_weight=sample_weights_x)

print("✅ xwOBA model trained!")


# ---------- 5. evaluation on hold-out ----------
def eval_pitch(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0, 0
    X, y, _ = prep_X_y(df, TARGET_PT, label_encoders)
    pred_proba = model.predict_proba(X)
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
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return rmse


print("\n📊 Evaluating on test set...")
acc, ll = eval_pitch(model_pt, test, label_encoders, pt_encoder)
rmse = eval_xwoba(model_x, test, label_encoders, pt_encoder)
print("🎯 Test Results:")
print(f"   Pitch-type accuracy: {acc:.3f}")
print(f"   Pitch-type log-loss: {ll:.3f}")
print(f"   xwOBA RMSE: {rmse:.4f}")

# ---------- 6. save models ----------
print("\n📦 Saving models...")

# Save models
with open(SAVE_DIR / "pitch_type_model.pkl", "wb") as f:
    pickle.dump(model_pt, f)

with open(SAVE_DIR / "xwoba_model.pkl", "wb") as f:
    pickle.dump(model_x, f)

# Save encoders
with open(SAVE_DIR / "label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(SAVE_DIR / "pt_encoder.pkl", "wb") as f:
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
        "model_type": "RandomForest",
    },
}

with open(SAVE_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Models saved successfully!")
print(f"📁 Files saved to: {SAVE_DIR}/")
print("   - pitch_type_model.pkl")
print("   - xwoba_model.pkl")
print("   - label_encoders.pkl")
print("   - pt_encoder.pkl")
print("   - model_metadata.json")

# ---------- 7. feature importance ----------
print("\n🔍 Top 10 Features for Pitch Type:")
pt_importance = pd.DataFrame(
    {"feature": X_train_pt.columns, "importance": model_pt.feature_importances_}
).sort_values("importance", ascending=False)
print(pt_importance.head(10).to_string(index=False))

print("\n🔍 Top 10 Features for xwOBA:")
x_importance = pd.DataFrame(
    {"feature": X_train_x.columns, "importance": model_x.feature_importances_}
).sort_values("importance", ascending=False)
print(x_importance.head(10).to_string(index=False))

print("\n🎉 Training pipeline complete!")
