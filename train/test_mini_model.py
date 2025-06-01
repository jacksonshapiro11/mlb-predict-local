import glob, pathlib, json, lightgbm as lgb, duckdb, pandas as pd, numpy as np
from datetime import date
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🧪 Mini Model Test - Data Validation")
print("📊 Testing with small subset to catch issues quickly")

# ---------- configuration ----------
TARGET_PT = "pitch_type_can"
TARGET_XWOBA = "estimated_woba_using_speedangle"
CAT_COLS = ["stand", "p_throws", "inning_topbot"]
PARQUET_GLOB = "data/features/statcast_*.parquet"

# ---------- load small test sample ----------
def load_sample_data():
    con = duckdb.connect()
    # Just get 2023 data, limited sample
    query = """
    SELECT * FROM parquet_scan("data/features/statcast_2023.parquet") 
    WHERE game_date BETWEEN DATE '2023-09-01' AND DATE '2023-09-30'
    LIMIT 10000
    """
    print(f"📥 Loading sample: {query}")
    df = con.execute(query).df()
    con.close()
    return df

# ---------- data prep & feature engineering ----------
def prep_X_y(df, target, label_encoders=None):
    print(f"\n🔍 Analyzing features for target: {target}")
    
    # Remove target columns and any problematic columns
    drop_cols = [TARGET_PT, TARGET_XWOBA, "game_date", "game_pk", "at_bat_number", 
                 "pitch_number", "inning", "inning_topbot", "batter", "pitcher", 
                 "home_team", "pitch_name", "events", "description", "pitch_type"]
    
    # Additional leakage prevention based on target
    if target == TARGET_XWOBA:
        # Remove ALL xwOBA-derived features when predicting xwOBA
        xwoba_features = [col for col in df.columns if 'xwoba' in col.lower()]
        drop_cols.extend(xwoba_features)
        print(f"🚨 Removing {len(xwoba_features)} xwOBA-derived features for xwOBA prediction")
    
    # Remove any features that contain the target name
    target_related = [col for col in df.columns if target.lower().replace('_', '') in col.lower().replace('_', '') and col != target]
    drop_cols.extend(target_related)
    
    drop_cols = [c for c in drop_cols if c in df.columns]
    print(f"🗑️  Dropping {len(drop_cols)} columns total")
    X = df.drop(columns=drop_cols)
    
    # Handle missing values in target
    valid_mask = df[target].notna()
    X = X[valid_mask]
    y = df[target][valid_mask]
    
    print(f"📊 Target distribution:")
    if target == TARGET_PT:
        print(y.value_counts().head())
    else:
        print(f"   Mean: {y.mean():.3f}, Std: {y.std():.3f}, Range: {y.min():.3f}-{y.max():.3f}")
    
    # Check feature types
    print(f"\n🔧 Feature analysis:")
    print(f"   Total features: {len(X.columns)}")
    print(f"   Numeric: {len(X.select_dtypes(include=[np.number]).columns)}")
    print(f"   Object: {len(X.select_dtypes(include=['object']).columns)}")
    
    # Encode categorical columns
    if label_encoders is None:
        label_encoders = {}
        
    for col in CAT_COLS:
        if col in X.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                all_values = X[col].fillna('__MISSING__').astype(str)
                label_encoders[col].fit(all_values)
            X[col] = label_encoders[col].transform(X[col].fillna('__MISSING__').astype(str))
    
    # Convert remaining object columns to numeric or drop them
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"⚠️  Dropping non-numeric column: {col}")
            X = X.drop(columns=[col])
    
    # Fill remaining NaN values
    nan_counts = X.isnull().sum()
    high_nan_cols = nan_counts[nan_counts > len(X) * 0.5].index
    if len(high_nan_cols) > 0:
        print(f"⚠️  High NaN columns (>50%): {len(high_nan_cols)} columns")
    
    X = X.fillna(0)
    
    print(f"✅ Final: {len(X.columns)} features, {len(X):,} samples")
    return X, y, label_encoders

# ---------- run mini test ----------
print("\n📊 Loading sample data...")
sample_data = load_sample_data()
print(f"✅ Loaded: {len(sample_data):,} rows, {len(sample_data.columns)} columns")

# Test pitch type prediction
print("\n🎯 Testing Pitch Type Classification...")
X_pt, y_pt, encoders = prep_X_y(sample_data, TARGET_PT)

# Quick model test
pt_encoder = LabelEncoder()
y_pt_encoded = pt_encoder.fit_transform(y_pt)
unique_classes = pt_encoder.classes_
print(f"📋 Pitch types: {list(unique_classes)}")

# Split data
split_idx = int(len(X_pt) * 0.8)
X_train, X_test = X_pt[:split_idx], X_pt[split_idx:]
y_train, y_test = y_pt_encoded[:split_idx], y_pt_encoded[split_idx:]

# Train simple model
lgb_train = lgb.Dataset(X_train, y_train, 
                        categorical_feature=[i for i, c in enumerate(X_train.columns) if c in CAT_COLS],
                        free_raw_data=False)

params = dict(
    objective="multiclass", 
    num_class=len(unique_classes),
    learning_rate=0.1, 
    num_leaves=32, 
    max_depth=6,
    metric="multi_logloss",
    verbose=-1
)

print("🚂 Training mini model...")
model = lgb.train(params, lgb_train, num_boost_round=100)

# Test predictions
pred_proba = model.predict(X_test)
pred_class = pred_proba.argmax(axis=1)
acc = accuracy_score(y_test, pred_class)
ll = log_loss(y_test, pred_proba)

print(f"🎯 Mini Model Results:")
print(f"   Accuracy: {acc:.3f}")
print(f"   Log-loss: {ll:.3f}")

# Check if results are reasonable
if acc > 0.95:
    print("🚨 WARNING: Accuracy too high - possible data leakage!")
elif acc < 0.2:
    print("🚨 WARNING: Accuracy too low - possible feature issues!")
else:
    print("✅ Accuracy looks reasonable")

# Test xwOBA prediction
print("\n🎯 Testing xwOBA Regression...")
X_xwoba, y_xwoba, _ = prep_X_y(sample_data, TARGET_XWOBA, encoders)

if len(X_xwoba) > 100:  # Only test if we have enough samples
    split_idx = int(len(X_xwoba) * 0.8)
    X_train_x, X_test_x = X_xwoba[:split_idx], X_xwoba[split_idx:]
    y_train_x, y_test_x = y_xwoba[:split_idx], y_xwoba[split_idx:]
    
    lgb_train_x = lgb.Dataset(X_train_x, y_train_x, 
                              categorical_feature=[i for i, c in enumerate(X_train_x.columns) if c in CAT_COLS],
                              free_raw_data=False)
    
    params_x = dict(objective="regression", metric="rmse", learning_rate=0.1, 
                    num_leaves=32, max_depth=6, verbose=-1)
    
    model_x = lgb.train(params_x, lgb_train_x, num_boost_round=100)
    
    pred_x = model_x.predict(X_test_x)
    rmse = np.sqrt(mean_squared_error(y_test_x, pred_x))
    
    print(f"🎯 xwOBA Results:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Target range: {y_xwoba.min():.3f} - {y_xwoba.max():.3f}")
    
    if rmse < 0.01:
        print("🚨 WARNING: RMSE too low - possible data leakage!")
    elif rmse > 0.5:
        print("🚨 WARNING: RMSE too high - possible feature issues!")
    else:
        print("✅ RMSE looks reasonable")

print("\n🎉 Mini model test complete!")
print("\n📋 Summary:")
print(f"   Pitch type accuracy: {acc:.3f} (should be ~0.3-0.8)")
print(f"   Pitch type log-loss: {ll:.3f} (should be ~0.5-2.0)")
if 'rmse' in locals():
    print(f"   xwOBA RMSE: {rmse:.4f} (should be ~0.05-0.15)")

if acc <= 0.95 and ll >= 0.3 and (not 'rmse' in locals() or rmse >= 0.01):
    print("✅ No obvious data leakage detected - safe to run full training!")
else:
    print("⚠️  Potential issues detected - review features before full training") 