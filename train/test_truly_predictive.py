import duckdb
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

TARGET_PT = "pitch_type_can"
CAT_COLS = ["stand", "p_throws"]


def load_sample_data():
    con = duckdb.connect()
    query = """
    SELECT * FROM parquet_scan("data/features/statcast_2023.parquet") 
    WHERE game_date BETWEEN DATE '2023-09-01' AND DATE '2023-09-30'
    LIMIT 5000
    """
    df = con.execute(query).df()
    con.close()
    return df


def prep_X_y_truly_predictive(df, target):
    print(f"\n🔧 TRULY PREDICTIVE prep for: {target}")

    # Base drops
    drop_cols = [
        TARGET_PT,
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
        "estimated_woba_using_speedangle",
    ]

    # Remove pitch-type-specific features (indirect leakage)
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
    print(f"🚨 Removing {len(pitch_type_features)} pitch-type-specific features")

    # Remove ALL current-pitch measurements (direct leakage)
    current_pitch_features = []
    for col in df.columns:
        col_lower = col.lower()
        # Current pitch physics
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
        # Current pitch location
        elif any(x in col_lower for x in ["zone", "hc_x", "hc_y"]):
            current_pitch_features.append(col)
        # Current pitch outcome
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
            ]
        ):
            current_pitch_features.append(col)

    drop_cols.extend(current_pitch_features)
    print(
        f"🚨 Removing {len(current_pitch_features)} current-pitch measurement features"
    )

    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Handle target
    valid_mask = df[target].notna()
    X = X[valid_mask]
    y = df[target][valid_mask]

    # Encode categoricals
    for col in CAT_COLS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("__MISSING__").astype(str))

    # Convert objects to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except:
                X = X.drop(columns=[col])

    X = X.fillna(0)

    print(f"✅ TRULY PREDICTIVE: {len(X.columns)} features, {len(X):,} samples")

    # Show what's left
    print("📋 Remaining feature categories:")
    remaining_cats = {
        "game_situation": [
            c
            for c in X.columns
            if any(x in c.lower() for x in ["balls", "strikes", "outs", "score"])
        ],
        "historical_stats": [
            c
            for c in X.columns
            if any(
                x in c.lower()
                for x in ["_30d", "_td", "_7d", "k_pct", "contact", "whiff", "hit_"]
            )
        ],
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
    }

    for cat, features in remaining_cats.items():
        if features:
            print(f"   {cat}: {len(features)} features")

    return X, y


# Test truly predictive approach
print("🧪 TRULY PREDICTIVE Mini Test")
sample_data = load_sample_data()
print(f"✅ Loaded: {len(sample_data):,} rows")

X, y = prep_X_y_truly_predictive(sample_data, TARGET_PT)

# Encode target
pt_encoder = LabelEncoder()
y_encoded = pt_encoder.fit_transform(y)
print(f"📋 Pitch types: {list(pt_encoder.classes_)}")

# Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

# Train
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
params = dict(
    objective="multiclass",
    num_class=len(pt_encoder.classes_),
    learning_rate=0.1,
    num_leaves=32,
    max_depth=6,
    metric="multi_logloss",
    verbose=-1,
)

print("🚂 Training TRULY PREDICTIVE model...")
model = lgb.train(params, lgb_train, num_boost_round=100)

# Test
pred_proba = model.predict(X_test)
pred_class = pred_proba.argmax(axis=1)
acc = accuracy_score(y_test, pred_class)
ll = log_loss(y_test, pred_proba)

print("\n🎯 TRULY PREDICTIVE Results:")
print(f"   Accuracy: {acc:.3f}")
print(f"   Log-loss: {ll:.3f}")

print("\n📊 Baseline comparison:")
print(f"   Random guess: {1/len(pt_encoder.classes_):.3f}")
print(
    f"   Most frequent class: {max(pd.Series(y_train).value_counts()) / len(y_train):.3f}"
)

print("\n💡 Performance Analysis:")
if acc > 0.6:
    print("🚨 Still suspiciously high - check for remaining leakage!")
elif acc < 0.2:
    print("⚠️  Very low - may be too restrictive, but realistic for true prediction")
else:
    print("✅ Realistic performance for true pre-pitch prediction!")

print("\n🎯 This represents what we can ACTUALLY predict before a pitch is thrown:")
print("   - Using only game situation, historical stats, and player characteristics")
print("   - NO information about the current pitch's physics, location, or outcome")
print("   - This is the true test of predictive power!")
