import glob, duckdb, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

TARGET_PT = "pitch_type_can"
TARGET_XWOBA = "estimated_woba_using_speedangle"
CAT_COLS = ["stand", "p_throws", "inning_topbot"]

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

def analyze_features_for_leakage(df, target):
    print(f"\n🔍 DETAILED FEATURE ANALYSIS FOR: {target}")
    print("="*60)
    
    # Standard drops
    drop_cols = [TARGET_PT, TARGET_XWOBA, "game_date", "game_pk", "at_bat_number", 
                 "pitch_number", "inning", "inning_topbot", "batter", "pitcher", 
                 "home_team", "pitch_name", "events", "description", "pitch_type"]
    
    # Additional leakage prevention
    if target == TARGET_XWOBA:
        xwoba_features = [col for col in df.columns if 'xwoba' in col.lower()]
        drop_cols.extend(xwoba_features)
    
    target_related = [col for col in df.columns if target.lower().replace('_', '') in col.lower().replace('_', '') and col != target]
    drop_cols.extend(target_related)
    
    drop_cols = [c for c in drop_cols if c in df.columns]
    remaining_features = [c for c in df.columns if c not in drop_cols]
    
    print(f"📊 REMAINING FEATURES ({len(remaining_features)}):")
    print("-" * 40)
    
    # Group features by category
    feature_groups = {
        'count_situation': [],
        'velocity': [],
        'spin': [],
        'usage': [],
        'whiff': [],
        'platoon': [],
        'recent_form': [],
        'ballpark': [],
        'other': []
    }
    
    for feat in remaining_features:
        feat_lower = feat.lower()
        if any(x in feat_lower for x in ['ball', 'strike', 'count']):
            feature_groups['count_situation'].append(feat)
        elif any(x in feat_lower for x in ['velocity', '_v_', 'velo']):
            feature_groups['velocity'].append(feat)
        elif any(x in feat_lower for x in ['spin', 'rpm']):
            feature_groups['spin'].append(feat)
        elif any(x in feat_lower for x in ['usage', 'pct']):
            feature_groups['usage'].append(feat)
        elif any(x in feat_lower for x in ['whiff', 'swing']):
            feature_groups['whiff'].append(feat)
        elif any(x in feat_lower for x in ['vs_l', 'vs_r', 'stand']):
            feature_groups['platoon'].append(feat)
        elif any(x in feat_lower for x in ['_7_', '_30_', 'recent']):
            feature_groups['recent_form'].append(feat)
        elif any(x in feat_lower for x in ['park', 'stadium']):
            feature_groups['ballpark'].append(feat)
        else:
            feature_groups['other'].append(feat)
    
    for group, features in feature_groups.items():
        if features:
            print(f"\n{group.upper()} ({len(features)}):")
            for feat in sorted(features)[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")
    
    # Check for highly correlated features with target
    print(f"\n🔍 CHECKING CORRELATIONS WITH TARGET...")
    valid_mask = df[target].notna()
    target_values = df[target][valid_mask]
    
    high_corr_features = []
    for feat in remaining_features:
        if feat in df.columns and df[feat].dtype in ['int64', 'float64']:
            try:
                feat_values = df[feat][valid_mask]
                if feat_values.std() > 0:  # Avoid constant features
                    corr = np.corrcoef(target_values, feat_values)[0, 1]
                    if abs(corr) > 0.8:  # High correlation threshold
                        high_corr_features.append((feat, corr))
            except:
                pass
    
    if high_corr_features:
        print("🚨 HIGH CORRELATION FEATURES (>0.8):")
        for feat, corr in sorted(high_corr_features, key=lambda x: abs(x[1]), reverse=True):
            print(f"  - {feat}: {corr:.3f}")
    else:
        print("✅ No extremely high correlations found")
    
    return remaining_features

# Load and analyze
print("🧪 FEATURE LEAKAGE ANALYSIS")
sample_data = load_sample_data()
print(f"✅ Loaded: {len(sample_data):,} rows, {len(sample_data.columns)} columns")

# Analyze for pitch type
pt_features = analyze_features_for_leakage(sample_data, TARGET_PT)

# Analyze for xwOBA  
xwoba_features = analyze_features_for_leakage(sample_data, TARGET_XWOBA)

print(f"\n📋 SUMMARY:")
print(f"   Pitch type features: {len(pt_features)}")
print(f"   xwOBA features: {len(xwoba_features)}") 