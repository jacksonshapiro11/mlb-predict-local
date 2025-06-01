import duckdb, pandas as pd, numpy as np

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

def analyze_pitch_specific_features(df):
    print("🔍 CHECKING FOR PITCH-SPECIFIC DATA LEAKAGE")
    print("="*60)
    
    # What we're keeping in ultra-clean mode
    TARGET_PT = "pitch_type_can"
    drop_cols = [TARGET_PT, "game_date", "game_pk", "at_bat_number", 
                 "pitch_number", "inning", "inning_topbot", "batter", "pitcher", 
                 "home_team", "pitch_name", "events", "description", "pitch_type",
                 "estimated_woba_using_speedangle"]
    
    # Remove pitch-type-specific features
    pitch_type_suffixes = ['_CH', '_CU', '_FC', '_FF', '_FS', '_KC', '_OTHER', '_SI', '_SL']
    pitch_type_features = []
    for col in df.columns:
        if any(suffix in col for suffix in pitch_type_suffixes):
            pitch_type_features.append(col)
    drop_cols.extend(pitch_type_features)
    
    drop_cols = [c for c in drop_cols if c in df.columns]
    remaining_features = [c for c in df.columns if c not in drop_cols]
    
    print(f"📊 REMAINING FEATURES: {len(remaining_features)}")
    print("-" * 40)
    
    # Categorize potential leakage sources
    leakage_categories = {
        'CURRENT_PITCH_PHYSICS': [],  # Direct measurements of THIS pitch
        'CURRENT_PITCH_LOCATION': [], # Where THIS pitch was thrown
        'CURRENT_PITCH_OUTCOME': [],  # What happened to THIS pitch
        'HISTORICAL_AGGREGATES': [],  # Historical stats (OK)
        'GAME_SITUATION': [],         # Count, inning, etc (OK)
        'PLAYER_CHARACTERISTICS': [], # Handedness, etc (OK)
        'UNCLEAR': []
    }
    
    for feat in remaining_features:
        feat_lower = feat.lower()
        
        # Current pitch physics - MAJOR LEAKAGE
        if any(x in feat_lower for x in ['release_speed', 'release_spin_rate', 'release_pos', 
                                        'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0',
                                        'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed',
                                        'release_extension', 'spin_axis']):
            leakage_categories['CURRENT_PITCH_PHYSICS'].append(feat)
            
        # Current pitch location - MAJOR LEAKAGE  
        elif any(x in feat_lower for x in ['zone', 'plate_x', 'plate_z', 'hc_x', 'hc_y']):
            leakage_categories['CURRENT_PITCH_LOCATION'].append(feat)
            
        # Current pitch outcome - MAJOR LEAKAGE
        elif any(x in feat_lower for x in ['launch_speed', 'launch_angle', 'hit_distance',
                                          'babip_value', 'iso_value', 'woba_value', 'woba_denom']):
            leakage_categories['CURRENT_PITCH_OUTCOME'].append(feat)
            
        # Historical aggregates - OK
        elif any(x in feat_lower for x in ['_30d', '_td', '_7d', '_30_', 'k_pct', 'bb_pct',
                                          'contact_', 'whiff_', 'hit_', 'avg_']):
            leakage_categories['HISTORICAL_AGGREGATES'].append(feat)
            
        # Game situation - OK
        elif any(x in feat_lower for x in ['balls', 'strikes', 'outs', 'inning', 'score']):
            leakage_categories['GAME_SITUATION'].append(feat)
            
        # Player characteristics - OK
        elif any(x in feat_lower for x in ['stand', 'p_throws', 'batter_', 'pitcher_']):
            leakage_categories['PLAYER_CHARACTERISTICS'].append(feat)
            
        else:
            leakage_categories['UNCLEAR'].append(feat)
    
    # Report findings
    total_leakage = 0
    for category, features in leakage_categories.items():
        if features:
            is_leakage = category in ['CURRENT_PITCH_PHYSICS', 'CURRENT_PITCH_LOCATION', 'CURRENT_PITCH_OUTCOME']
            status = "🚨 LEAKAGE" if is_leakage else "✅ OK"
            print(f"\n{status} {category} ({len(features)}):")
            
            if is_leakage:
                total_leakage += len(features)
                
            for feat in sorted(features)[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")
    
    print(f"\n📋 SUMMARY:")
    print(f"   🚨 TOTAL LEAKAGE FEATURES: {total_leakage}")
    print(f"   ✅ CLEAN FEATURES: {len(remaining_features) - total_leakage}")
    
    if total_leakage > 0:
        print(f"\n🚨 CRITICAL: We still have {total_leakage} features that contain information about the CURRENT pitch!")
        print("   These features would not be available when making real predictions.")
        print("   We need to remove these for a truly realistic model.")
    else:
        print("\n✅ No current-pitch leakage detected!")
    
    return leakage_categories

# Run analysis
print("🧪 PITCH-SPECIFIC LEAKAGE ANALYSIS")
sample_data = load_sample_data()
print(f"✅ Loaded: {len(sample_data):,} rows, {len(sample_data.columns)} columns")

leakage_analysis = analyze_pitch_specific_features(sample_data)

# Show specific examples of potential leakage
print(f"\n🔍 DETAILED LEAKAGE EXAMPLES:")
print("-" * 40)

sample_pitch = sample_data.iloc[0]
print(f"Example pitch: {sample_pitch['pitch_type']} thrown by pitcher {sample_pitch['pitcher']}")

leakage_features = []
leakage_features.extend(leakage_analysis['CURRENT_PITCH_PHYSICS'])
leakage_features.extend(leakage_analysis['CURRENT_PITCH_LOCATION']) 
leakage_features.extend(leakage_analysis['CURRENT_PITCH_OUTCOME'])

if leakage_features:
    print(f"\nLeakage feature values for this pitch:")
    for feat in leakage_features[:5]:  # Show first 5
        if feat in sample_pitch.index:
            print(f"  {feat}: {sample_pitch[feat]}")
    
    print(f"\n💡 These values are measurements OF the pitch we're trying to predict!")
    print("   In real prediction, we wouldn't know these values beforehand.")
else:
    print("✅ No leakage features found!") 