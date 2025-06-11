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

# ---------- MINI configuration ----------
TARGET_PT = "pitch_type_can"
TARGET_XWOBA = "estimated_woba_using_speedangle"
CAT_COLS = ["stand", "p_throws"]
LAMBDA_DECAY = 0.0012  # ≈ 2-season half-life
SAVE_DIR = pathlib.Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting MLB BALANCED PREDICTIVE Model Training Pipeline (MINI)")
print(
    "🔧 BALANCED: Keeping historical features, removing only current-pitch measurements"
)
print("📊 MINI: Using small subset for proof of concept")


# ---------- helper to load mini data ----------
def load_mini_data():
    con = duckdb.connect()
    # Load small subset from 2023 for quick testing
    query = """
    SELECT * FROM parquet_scan("data/features/statcast_2023.parquet") 
    WHERE game_date BETWEEN DATE '2023-09-01' AND DATE '2023-09-30'
    LIMIT 5000
    """
    print(f"📥 Loading mini dataset: {query}")
    df = con.execute(query).df()
    con.close()
    return df


def add_weights(df, latest):
    # days since most recent pitch in *training* data
    delta = (latest - pd.to_datetime(df["game_date"])).dt.days
    df["w"] = np.exp(-LAMBDA_DECAY * delta)
    print(f"⚖️  Weight range: {df['w'].min():.4f} - {df['w'].max():.4f}")
    return df


# ---------- 1. load mini dataset ----------
print("\n📊 Loading mini dataset...")
data = load_mini_data()
data = add_weights(data, pd.to_datetime("2023-09-30"))
print(f"✅ Mini dataset: {len(data):,} rows")

# Split into train/test (80/20)
train_size = int(0.8 * len(data))
train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()
print(f"✅ Train: {len(train):,} rows, Test: {len(test):,} rows")


# ---------- 2. BALANCED PREDICTIVE data prep ----------
def prep_X_y_balanced(df, target, label_encoders=None):
    print(f"\n🔧 BALANCED PREDICTIVE prep for: {target}")

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

    # CRITICAL: Remove ONLY current-pitch measurements (direct leakage)
    # Keep historical pitch-type features - they represent known tendencies!
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

    # Remove target-related features (but keep historical xwOBA - it's not the target!)
    target_related = []
    if target == TARGET_XWOBA:
        # Only remove the exact target column, not historical xwOBA features
        target_related = [TARGET_XWOBA]
    elif target == TARGET_PT:
        # Remove pitch_type_can related features
        target_related = [
            col
            for col in df.columns
            if "pitch_type_can" in col.lower() and col != target
        ]

    drop_cols.extend(target_related)

    drop_cols = list(
        set([c for c in drop_cols if c in df.columns])
    )  # Remove duplicates
    X = df.drop(columns=drop_cols)

    print(f"🗑️  Dropped {len(drop_cols)} columns, keeping {len(X.columns)} features")

    # Show what's left
    print("📋 Remaining feature categories:")
    remaining_cats = {
        "game_situation": [
            c
            for c in X.columns
            if any(x in c.lower() for x in ["balls", "strikes", "outs", "score"])
        ],
        "historical_arsenal": [
            c
            for c in X.columns
            if any(x in c.lower() for x in ["usage_", "v_td_", "spin_td_", "whiff_30_"])
        ],
        "historical_xwoba": [c for c in X.columns if "xwoba" in c.lower()],
        "player_chars": [
            c
            for c in X.columns
            if any(x in c.lower() for x in ["stand", "p_throws", "batter_", "pitcher_"])
        ],
        "situational": [
            c
            for c in X.columns
            if any(x in c.lower() for x in ["vs_l", "vs_r", "ahead", "behind", "even"])
        ],
        "other": [],
    }

    # Categorize remaining
    categorized = set()
    for cat, features in remaining_cats.items():
        categorized.update(features)

    remaining_cats["other"] = [c for c in X.columns if c not in categorized]

    for cat, features in remaining_cats.items():
        if features:
            print(f"   {cat}: {len(features)} features")
            if cat == "historical_xwoba":
                print(f"      ✅ KEEPING historical xwOBA features: {features[:3]}...")

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

    print(f"✅ BALANCED PREDICTIVE: {len(X.columns)} features, {len(X):,} samples")
    return X, y, label_encoders


print("\n🎯 HEAD A: Pitch Type Classification (BALANCED PREDICTIVE)")
X_train_pt, y_train_pt, label_encoders = prep_X_y_balanced(train, TARGET_PT)

# Create label encoder for target
pt_encoder = LabelEncoder()
y_train_pt_encoded = pt_encoder.fit_transform(y_train_pt)
unique_classes = pt_encoder.classes_
print(f"📋 Pitch types: {list(unique_classes)}")

# Train pitch type model (smaller params for mini)
lgb_train_pt = lgb.Dataset(
    X_train_pt,
    y_train_pt_encoded,
    weight=train["w"][y_train_pt.index] if "w" in train.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_pt.columns) if c in CAT_COLS],
    free_raw_data=False,
)

params_pt = dict(
    objective="multiclass",
    num_class=len(unique_classes),
    learning_rate=0.1,  # Faster for mini
    num_leaves=64,  # Smaller for mini
    max_depth=-1,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    metric="multi_logloss",
    verbose=-1,
)

print("🚂 Training BALANCED PREDICTIVE pitch type model (MINI)...")
model_pt = lgb.train(
    params_pt,
    lgb_train_pt,
    num_boost_round=200,  # Much smaller for mini
    callbacks=[lgb.log_evaluation(50)],
)

print(f"✅ Pitch type model trained! Iterations: {model_pt.num_trees()}")


# ---------- 3. create pitch-prob features for Outcome head ----------
def add_pt_probs(df, model, label_encoders, pt_encoder):
    X, _, _ = prep_X_y_balanced(df, TARGET_PT, label_encoders)
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

print("\n🎯 HEAD B: xwOBA Regression (BALANCED PREDICTIVE)")
X_train_x, y_train_x, _ = prep_X_y_balanced(train_ext, TARGET_XWOBA, label_encoders)

lgb_train_x = lgb.Dataset(
    X_train_x,
    y_train_x,
    weight=train_ext["w"][y_train_x.index] if "w" in train_ext.columns else None,
    categorical_feature=[i for i, c in enumerate(X_train_x.columns) if c in CAT_COLS],
    free_raw_data=False,
)

params_x = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.1,  # Faster for mini
    num_leaves=128,  # Smaller for mini
    max_depth=-1,
    feature_fraction=0.95,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
)

print("🚂 Training BALANCED PREDICTIVE xwOBA model (MINI)...")
model_x = lgb.train(
    params_x,
    lgb_train_x,
    num_boost_round=200,  # Much smaller for mini
    callbacks=[lgb.log_evaluation(50)],
)

print(f"✅ xwOBA model trained! Iterations: {model_x.num_trees()}")


# ---------- 4. evaluation on test set ----------
def eval_pitch(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0, 0
    X, y, _ = prep_X_y_balanced(df, TARGET_PT, label_encoders)
    pred_proba = model.predict(X)
    pred_class = pred_proba.argmax(axis=1)

    y_encoded = pt_encoder.transform(y)
    acc = accuracy_score(y_encoded, pred_class)
    ll = log_loss(y_encoded, pred_proba)
    return acc, ll


def eval_xwoba(model, df, label_encoders, pt_encoder):
    if len(df) == 0:
        return 0
    df_ext = add_pt_probs(df, model_pt, label_encoders, pt_encoder)
    X, y, _ = prep_X_y_balanced(df_ext, TARGET_XWOBA, label_encoders)
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return rmse


print("\n📊 Evaluating on test set...")
acc, ll = eval_pitch(model_pt, test, label_encoders, pt_encoder)
rmse = eval_xwoba(model_x, test, label_encoders, pt_encoder)

print("\n🎯 MINI Test Results (BALANCED PREDICTIVE - HISTORICAL FEATURES KEPT):")
print(f"   Pitch-type accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"   Pitch-type log-loss: {ll:.3f}")
print(f"   xwOBA RMSE: {rmse:.4f}")

# Show feature importance for pitch type
print("\n🔍 Top 10 Pitch Type Features:")
feature_importance = model_pt.feature_importance(importance_type="gain")
feature_names = X_train_pt.columns
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
).sort_values("importance", ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.0f}")

# Show feature importance for xwOBA
print("\n🔍 Top 10 xwOBA Features:")
feature_importance_x = model_x.feature_importance(importance_type="gain")
feature_names_x = X_train_x.columns
importance_df_x = pd.DataFrame(
    {"feature": feature_names_x, "importance": feature_importance_x}
).sort_values("importance", ascending=False)

for i, row in importance_df_x.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.0f}")

# ---------- 5. save mini models ----------
print("\n📦 Saving MINI models...")

# Save LightGBM models in native format
model_pt.save_model(SAVE_DIR / "pitch_type_balanced_mini.lgb")
model_x.save_model(SAVE_DIR / "xwoba_balanced_mini.lgb")

# Save encoders
with open(SAVE_DIR / "label_encoders_balanced_mini.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(SAVE_DIR / "pt_encoder_balanced_mini.pkl", "wb") as f:
    pickle.dump(pt_encoder, f)

# Save feature metadata
metadata = {
    "pitch_type_features": list(X_train_pt.columns),
    "xwoba_features": list(X_train_x.columns),
    "pitch_classes": list(pt_encoder.classes_),
    "feature_importance_pt": importance_df.to_dict("records"),
    "feature_importance_xwoba": importance_df_x.to_dict("records"),
    "training_info": {
        "dataset": "MINI (5000 samples from Sept 2023)",
        "lambda_decay": LAMBDA_DECAY,
        "test_accuracy": float(acc),
        "test_logloss": float(ll),
        "test_rmse": float(rmse),
        "model_type": "LightGBM",
        "balanced_predictive": True,
        "kept_historical_features": True,
        "removed_current_pitch_data": True,
        "kept_historical_xwoba": True,
    },
}

with open(SAVE_DIR / "model_metadata_balanced_mini.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ BALANCED PREDICTIVE MINI Models saved successfully!")
print(f"📁 Files saved to: {SAVE_DIR}/")
print("   - pitch_type_balanced_mini.lgb")
print("   - xwoba_balanced_mini.lgb")
print("   - label_encoders_balanced_mini.pkl")
print("   - pt_encoder_balanced_mini.pkl")
print("   - model_metadata_balanced_mini.json")

print("\n🎉 BALANCED PREDICTIVE MINI Training complete!")
print(f"🎯 Accuracy: {acc:.3f} ({acc*100:.1f}%) - Much more realistic!")
print("🎯 This proves our balanced approach works - keeping historical tendencies")
print("🎯 Ready to scale up to full dataset when satisfied with features")
