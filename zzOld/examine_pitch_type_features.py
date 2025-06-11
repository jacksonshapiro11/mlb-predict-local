import duckdb


def load_sample_data():
    con = duckdb.connect()
    query = """
    SELECT * FROM parquet_scan("data/features/statcast_2023.parquet") 
    WHERE game_date BETWEEN DATE '2023-09-01' AND DATE '2023-09-30'
    LIMIT 1000
    """
    df = con.execute(query).df()
    con.close()
    return df


def analyze_pitch_type_features():
    print("🔍 ANALYZING PITCH-TYPE-SPECIFIC FEATURES")
    print("=" * 60)

    sample_data = load_sample_data()

    # Find all pitch-type-specific features
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
    for col in sample_data.columns:
        if any(suffix in col for suffix in pitch_type_suffixes):
            pitch_type_features.append(col)

    print(f"📊 Found {len(pitch_type_features)} pitch-type-specific features:")
    print("-" * 40)

    # Categorize these features
    feature_categories = {
        "HISTORICAL_USAGE": [],  # ✅ Available before pitch - pitcher's tendencies
        "HISTORICAL_VELOCITY": [],  # ✅ Available before pitch - pitcher's typical speeds
        "HISTORICAL_SPIN": [],  # ✅ Available before pitch - pitcher's typical spin
        "HISTORICAL_WHIFF": [],  # ✅ Available before pitch - how batters perform vs this pitch type
        "HISTORICAL_XWOBA": [],  # ❌ Target-related - should remove for xwOBA prediction
        "UNCLEAR": [],
    }

    for feat in sorted(pitch_type_features):
        feat_lower = feat.lower()

        if "usage" in feat_lower:
            feature_categories["HISTORICAL_USAGE"].append(feat)
        elif any(x in feat_lower for x in ["v_td", "velo", "velocity"]):
            feature_categories["HISTORICAL_VELOCITY"].append(feat)
        elif "spin" in feat_lower:
            feature_categories["HISTORICAL_SPIN"].append(feat)
        elif "whiff" in feat_lower:
            feature_categories["HISTORICAL_WHIFF"].append(feat)
        elif "xwoba" in feat_lower:
            feature_categories["HISTORICAL_XWOBA"].append(feat)
        else:
            feature_categories["UNCLEAR"].append(feat)

    # Report findings
    for category, features in feature_categories.items():
        if features:
            is_ok = category != "HISTORICAL_XWOBA"
            status = "✅ SHOULD KEEP" if is_ok else "❌ SHOULD REMOVE"
            print(f"\n{status} {category} ({len(features)}):")

            for feat in features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")

    print("\n💡 ANALYSIS:")
    print("✅ USAGE features: Pitcher's historical pitch mix - AVAILABLE before pitch")
    print("   Example: 'This pitcher throws 60% fastballs, 25% sliders'")
    print(
        "✅ VELOCITY features: Pitcher's typical velocity by pitch type - AVAILABLE before pitch"
    )
    print("   Example: 'His fastball averages 95 mph, slider averages 87 mph'")
    print(
        "✅ SPIN features: Pitcher's typical spin by pitch type - AVAILABLE before pitch"
    )
    print("   Example: 'His curveball averages 2800 RPM'")
    print(
        "✅ WHIFF features: How batters perform vs this pitch type - AVAILABLE before pitch"
    )
    print("   Example: 'Batters whiff 30% against his slider'")
    print(
        "❌ XWOBA features: Target-related for xwOBA prediction - SHOULD REMOVE for xwOBA only"
    )

    # Show example values
    print("\n📋 EXAMPLE VALUES (first pitch):")
    sample_pitch = sample_data.iloc[0]
    print(f"Pitch type: {sample_pitch['pitch_type']}")

    for category, features in feature_categories.items():
        if features and category != "HISTORICAL_XWOBA":
            print(f"\n{category}:")
            for feat in features[:3]:  # Show first 3
                if feat in sample_pitch.index:
                    print(f"  {feat}: {sample_pitch[feat]}")

    return feature_categories


# Run analysis
result = analyze_pitch_type_features()

print("\n🎯 RECOMMENDATION:")
print("We should KEEP most pitch-type-specific features because they represent")
print("historical tendencies that ARE available before the pitch is thrown!")
print("\nOnly remove:")
print("- XWOBA features when predicting xwOBA (target leakage)")
print("- Current pitch measurements (velocity, spin, location of THIS pitch)")
print(
    "\n28.4% accuracy was too low because we removed valuable predictive information!"
)
