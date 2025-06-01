import glob, pathlib, json, lightgbm as lgb, duckdb, pandas as pd, numpy as np
from datetime import date
from sklearn.metrics import log_loss, mean_squared_error
from skl2onnx import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType

# ---------- configuration ----------
SEASONS_TRAIN   = list(range(2018, 2024))          # 2018-2023
SEASONS_VAL     = [(2024, "2024-04-01", "2024-07-31")]  # early 2024 for val
SEASON_TEST     = [(2024, "2024-08-01", "2024-10-31"),
                   (2025, "2025-01-01", "2100-01-01")]   # YTD
TARGET_PT       = "pitch_type_can"
TARGET_XWOBA    = "estimated_woba_using_speedangle"
CAT_COLS        = ["pitch_type", "pitch_type_can", "stand",
                   "p_throws", "inning_topbot"]
LAMBDA_DECAY    = 0.0012          # ≈ 2-season half-life
PARQUET_GLOB    = "data/features/statcast_*.parquet"
SAVE_DIR        = pathlib.Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting MLB Prediction Model Training Pipeline")
print(f"📊 Training seasons: {SEASONS_TRAIN}")
print(f"🎯 Targets: {TARGET_PT} → {TARGET_XWOBA}")

# ---------- helper to load parquet subset ----------
def load_parquets(files, start=None, end=None):
    con = duckdb.connect()
    files_str = str(files).replace("'", '"')  # DuckDB needs double quotes
    if start and end:
        query = (f"SELECT * FROM parquet_scan({files_str}) "
                 f"WHERE game_date BETWEEN DATE '{start}' AND DATE '{end}'")
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
def prep_X_y(df, target):
    # Remove target columns and any problematic columns
    drop_cols = [TARGET_PT, TARGET_XWOBA, "w"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    # Handle missing values in target
    valid_mask = df[target].notna()
    X = X[valid_mask]
    y = df[target][valid_mask]
    
    # Ensure categorical columns exist and are properly typed
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    print(f"🔧 Features: {len(X.columns)}, Samples: {len(X):,}")
    return X, y

print("\n🎯 HEAD A: Pitch Type Classification")
X_train_pt, y_train_pt = prep_X_y(train, TARGET_PT)
X_val_pt, y_val_pt = prep_X_y(val, TARGET_PT) if len(val) > 0 else (None, None)

# Get unique classes and create label mapping
unique_classes = sorted(y_train_pt.unique())
print(f"📋 Pitch types: {unique_classes}")

lgb_train_pt = lgb.Dataset(X_train_pt, y_train_pt,
                           weight=train["w"][y_train_pt.index] if "w" in train.columns else None,
                           categorical_feature=[c for c in CAT_COLS if c in X_train_pt.columns],
                           free_raw_data=False)

lgb_val_pt = None
if X_val_pt is not None:
    lgb_val_pt = lgb.Dataset(X_val_pt, y_val_pt,
                             categorical_feature=[c for c in CAT_COLS if c in X_val_pt.columns],
                             reference=lgb_train_pt,
                             free_raw_data=False)

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
    verbose=-1
)

print("🚂 Training pitch type model...")
model_pt = lgb.train(
    params_pt, 
    lgb_train_pt,
    num_boost_round=4000,
    valid_sets=[lgb_val_pt] if lgb_val_pt else None,
    early_stopping_rounds=150 if lgb_val_pt else None,
    verbose_eval=200
)

print(f"✅ Pitch type model trained! Best iteration: {model_pt.best_iteration}")

# ---------- 4. create pitch-prob features for Outcome head ----------
def add_pt_probs(df, model):
    X, _ = prep_X_y(df, TARGET_PT)  # Just to get consistent feature set
    proba = model.predict(X, num_iteration=model.best_iteration)
    
    # Create probability features
    proba_df = pd.DataFrame(proba, columns=[f"PT_PROB_{c}" for c in unique_classes])
    proba_df.index = X.index
    
    # Merge back with original dataframe
    result = df.copy()
    for col in proba_df.columns:
        result[col] = np.nan
        result.loc[proba_df.index, col] = proba_df[col]
    
    return result

print("\n🔧 Adding pitch probability features...")
train_ext = add_pt_probs(train, model_pt)
val_ext = add_pt_probs(val, model_pt) if len(val) > 0 else pd.DataFrame()

print("\n🎯 HEAD B: xwOBA Regression")
X_train_x, y_train_x = prep_X_y(train_ext, TARGET_XWOBA)
X_val_x, y_val_x = prep_X_y(val_ext, TARGET_XWOBA) if len(val_ext) > 0 else (None, None)

lgb_train_x = lgb.Dataset(X_train_x, y_train_x,
                          weight=train_ext["w"][y_train_x.index] if "w" in train_ext.columns else None,
                          categorical_feature=[c for c in CAT_COLS if c in X_train_x.columns],
                          free_raw_data=False)

lgb_val_x = None
if X_val_x is not None:
    lgb_val_x = lgb.Dataset(X_val_x, y_val_x,
                            categorical_feature=[c for c in CAT_COLS if c in X_val_x.columns],
                            reference=lgb_train_x,
                            free_raw_data=False)

params_x = dict(
    objective="regression", 
    metric="rmse",
    learning_rate=0.05, 
    num_leaves=512, 
    max_depth=-1,
    feature_fraction=0.95, 
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1
)

print("🚂 Training xwOBA model...")
model_x = lgb.train(
    params_x, 
    lgb_train_x, 
    6000,
    valid_sets=[lgb_val_x] if lgb_val_x else None,
    early_stopping_rounds=200 if lgb_val_x else None,
    verbose_eval=200
)

print(f"✅ xwOBA model trained! Best iteration: {model_x.best_iteration}")

# ---------- 5. evaluation on hold-out ----------
def eval_pitch(model, df):
    if len(df) == 0:
        return 0, 0
    X, y = prep_X_y(df, TARGET_PT)
    pred = model.predict(X, num_iteration=model.best_iteration)
    acc = (pred.argmax(1) == y.map({cls: i for i, cls in enumerate(unique_classes)})).mean()
    ll = log_loss(y.map({cls: i for i, cls in enumerate(unique_classes)}), pred)
    return acc, ll

def eval_xwoba(model, df):
    if len(df) == 0:
        return 0
    df_ext = add_pt_probs(df, model_pt)
    X, y = prep_X_y(df_ext, TARGET_XWOBA)
    pred = model.predict(X, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return rmse

print("\n📊 Evaluating on test set...")
acc, ll = eval_pitch(model_pt, test)
rmse = eval_xwoba(model_x, test)
print(f"🎯 Test Results:")
print(f"   Pitch-type accuracy: {acc:.3f}")
print(f"   Pitch-type log-loss: {ll:.3f}")
print(f"   xwOBA RMSE: {rmse:.4f}")

# ---------- 6. export to ONNX ----------
print("\n📦 Exporting models to ONNX...")

try:
    # Export pitch type model
    onnx_pt = convert_lightgbm(
        model_pt, 
        initial_types=[("float_input", FloatTensorType([None, len(model_pt.feature_name())]))]
    )
    with open(SAVE_DIR/"pitch_type.onnx", "wb") as f:
        f.write(onnx_pt.SerializeToString())
    
    # Export xwOBA model  
    onnx_x = convert_lightgbm(
        model_x,
        initial_types=[("float_input", FloatTensorType([None, len(model_x.feature_name())]))]
    )
    with open(SAVE_DIR/"xwoba.onnx", "wb") as f:
        f.write(onnx_x.SerializeToString())
    
    # Save feature metadata
    metadata = {
        "pitch_type_features": model_pt.feature_name(),
        "xwoba_features": model_x.feature_name(),
        "pitch_classes": unique_classes,
        "training_info": {
            "train_seasons": SEASONS_TRAIN,
            "lambda_decay": LAMBDA_DECAY,
            "test_accuracy": float(acc),
            "test_logloss": float(ll),
            "test_rmse": float(rmse)
        }
    }
    
    with open(SAVE_DIR/"model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Models exported successfully!")
    print(f"📁 Files saved to: {SAVE_DIR}/")
    print("   - pitch_type.onnx")
    print("   - xwoba.onnx") 
    print("   - model_metadata.json")
    
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    print("💡 Models still trained and available in memory")

print("\n🎉 Training pipeline complete!") 