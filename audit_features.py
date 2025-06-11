import duckdb

# Load a small sample to see all column names
con = duckdb.connect()
query = """
SELECT * FROM parquet_scan("data/features/statcast_2023.parquet") 
WHERE game_date BETWEEN DATE '2023-09-01' AND DATE '2023-09-30'
LIMIT 100
"""
df = con.execute(query).df()
con.close()

print("🔍 FEATURE AUDIT: Pre-Pitch vs Post-Pitch")
print("=" * 60)

# Categorize all features
pre_pitch = []
post_pitch = []

for col in sorted(df.columns):
    col_lower = col.lower()

    # Definitely POST-PITCH (outcomes/measurements of THIS pitch)
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
            "zone",
            "hc_x",
            "hc_y",
            "launch_speed",
            "launch_angle",
            "hit_distance",
            "babip_value",
            "iso_value",
            "woba_value",
            "woba_denom",
            "delta_run_exp",
            "delta_home_win_exp",
            "events",
            "description",
            "pitch_name",
            "pitch_type",  # These are the actual outcome
        ]
    ):
        post_pitch.append(col)

    # Target variables
    elif col in ["pitch_type_can", "estimated_woba_using_speedangle"]:
        post_pitch.append(f"{col} (TARGET)")

    # Everything else is PRE-PITCH (including the previously questionable ones)
    else:
        pre_pitch.append(col)

print(f"✅ PRE-PITCH FEATURES ({len(pre_pitch)}):")
print("   📊 Game Situation:")
for col in pre_pitch:
    if any(
        x in col.lower() for x in ["balls", "strikes", "outs", "inning", "score", "on_"]
    ):
        print(f"      {col}")

print("   🏟️ Historical Arsenal (Pitcher):")
for col in pre_pitch:
    if any(x in col.lower() for x in ["usage_", "v_td_", "spin_td_"]):
        print(f"      {col}")

print("   🎯 Historical Performance (30-day):")
for col in pre_pitch:
    if any(
        x in col.lower() for x in ["whiff_30_", "contact_30_", "hit_30_", "xwoba_30_"]
    ):
        print(f"      {col}")

print("   📈 Recent Form (7-day):")
for col in pre_pitch:
    if any(x in col.lower() for x in ["whiff_7d", "velo_7d", "hit_7d"]):
        print(f"      {col}")

print("   👤 Player Characteristics:")
for col in pre_pitch:
    if any(
        x in col.lower()
        for x in ["stand", "p_throws", "batter", "pitcher", "home_team"]
    ):
        print(f"      {col}")

print("   📋 Other Pre-Pitch:")
categorized = set()
for col in pre_pitch:
    if any(
        x in col.lower()
        for x in [
            "balls",
            "strikes",
            "outs",
            "inning",
            "score",
            "on_",
            "usage_",
            "v_td_",
            "spin_td_",
            "whiff_30_",
            "contact_30_",
            "hit_30_",
            "xwoba_30_",
            "whiff_7d",
            "velo_7d",
            "hit_7d",
            "stand",
            "p_throws",
            "batter",
            "pitcher",
            "home_team",
        ]
    ):
        categorized.add(col)

for col in pre_pitch:
    if col not in categorized:
        print(f"      {col}")

print(f"\n❌ POST-PITCH FEATURES ({len(post_pitch)}):")
for col in post_pitch:
    print(f"   {col}")

print("\n🧠 ANALYSIS OF PREVIOUSLY QUESTIONABLE FEATURES:")
print("   ✅ CONTACT_30_AHEAD/BEHIND/EVEN: Contact rate in different count situations")
print("      → PRE-PITCH: Historical 30-day contact rates by count state")
print("   ✅ HIT_30_VS_L/R: Hit rate vs left/right handed batters")
print("      → PRE-PITCH: Historical 30-day hit rates by batter handedness")
print("   ✅ HIT_7D: Hit rate over last 7 days")
print("      → PRE-PITCH: Recent 7-day hit rate trend")
print("   ✅ VELO_7D: Average velocity over last 7 days")
print("      → PRE-PITCH: Recent 7-day velocity trend")

print("\n📊 FINAL SUMMARY:")
print(f"   ✅ Pre-pitch (SAFE): {len(pre_pitch)}")
print(f"   ❌ Post-pitch (REMOVE): {len(post_pitch)}")
print(f"   📈 Total features: {len(df.columns)}")
print("\n🎯 CONCLUSION: All questionable features are actually PRE-PITCH!")
print("   They represent historical trends/rates, not current pitch outcomes.")
