import pathlib
import lightgbm as lgb
import pickle
import warnings
import os

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

print("🔄 Resume Training Script")

# Check what models already exist
pt_model_exists = os.path.exists(SAVE_DIR / "pitch_type.lgb")
xwoba_model_exists = os.path.exists(SAVE_DIR / "xwoba.lgb")
encoders_exist = os.path.exists(SAVE_DIR / "label_encoders.pkl")

print("📋 Status Check:")
print(f"   Pitch Type Model: {'✅' if pt_model_exists else '❌'}")
print(f"   xwOBA Model: {'✅' if xwoba_model_exists else '❌'}")
print(f"   Encoders: {'✅' if encoders_exist else '❌'}")

if pt_model_exists and xwoba_model_exists and encoders_exist:
    print("🎉 All models already trained! Loading for evaluation...")

    # Load models
    model_pt = lgb.Booster(model_file=str(SAVE_DIR / "pitch_type.lgb"))
    model_x = lgb.Booster(model_file=str(SAVE_DIR / "xwoba.lgb"))

    with open(SAVE_DIR / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(SAVE_DIR / "pt_encoder.pkl", "rb") as f:
        pt_encoder = pickle.load(f)

    print("✅ Models loaded successfully!")

elif pt_model_exists and encoders_exist and not xwoba_model_exists:
    print("🔄 Pitch type model exists, training xwOBA model...")
    # Resume from xwOBA training
    # [Implementation would go here]

else:
    print("❌ No complete models found. Run full training script.")
    print("   Use: python3 train/train_heads_lgb_simple.py")

print("\n📊 Training Status Summary:")
if pt_model_exists and xwoba_model_exists:
    print("🎯 Ready for inference!")
else:
    print("⏳ Training in progress or needed.")
