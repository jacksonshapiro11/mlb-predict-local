#!/usr/bin/env python
"""
example_gru_workflow.py
======================
Example workflow demonstrating the complete pipeline with GRU sequence model.

This script shows the step-by-step process of:
1. Training tree models (LGB, XGB, CatBoost)
2. Training GRU sequence model 
3. Ensemble blending with 4 models

USAGE
-----
python example_gru_workflow.py
"""

import subprocess
import pathlib


def run_cmd(cmd, description):
    """Run a command with description."""
    print(f"\n🔧 {description}")
    print(f"$ {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                print(result.stdout.strip()[-200:])  # Show last 200 chars
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    return True


def main():
    print("🚀 MLB Pitch Prediction: GRU + Tree Ensemble Workflow")
    print("=" * 60)

    # Check if we have data
    data_dir = pathlib.Path("data/features_historical")
    if not data_dir.exists() or not list(data_dir.glob("*.parquet")):
        print("❌ No historical feature files found.")
        print("📝 Please run the ETL pipeline first:")
        print("   python etl/fetch_statcast.py 2023")
        print("   python etl/build_historical_features.py 2023")
        return

    # Step 1: Train GRU sequence model (optional)
    print("\n📊 Step 1: Train GRU Sequence Model")
    try:
        import torch

        print("✅ PyTorch available - can train GRU model")

        gru_success = run_cmd("python train_seq_head.py", "Training GRU sequence model")

        if gru_success:
            print("✅ GRU model training complete")
        else:
            print("⚠️  GRU training failed - continuing with 3-model ensemble")

    except ImportError:
        print("ℹ️  PyTorch not available - skipping GRU model")
        print("💡 To enable GRU: pip install torch tqdm")

    # Step 2: Train tree ensemble (automatically includes GRU if available)
    print("\n📊 Step 2: Train Tree Ensemble + Blend")

    # Use a smaller dataset for demo
    tree_success = run_cmd(
        "python run_full_pipeline.py train "
        "--train-years 2023 "
        "--val '2024-04-01:2024-05-31' "
        "--test '2024-06-01:2024-06-30' "
        "--decay 0.001",
        "Training tree ensemble with automatic GRU integration",
    )

    if tree_success:
        print("✅ Complete pipeline finished successfully!")

        # Check what models were used
        gru_logits = pathlib.Path("models/gru_val_logits.npy")
        if gru_logits.exists():
            print("🤖 Used 4-model ensemble: LGB + XGB + CatBoost + GRU")
        else:
            print("🌳 Used 3-model ensemble: LGB + XGB + CatBoost")

    else:
        print("❌ Pipeline failed")

    print("\n" + "=" * 60)
    print("🎯 Workflow Complete!")
    print("\n💡 Key Points:")
    print("• GRU model captures pitch sequence patterns")
    print("• Tree models capture feature interactions")
    print("• Ensemble combines both approaches")
    print("• Pipeline automatically detects available models")


if __name__ == "__main__":
    main()
