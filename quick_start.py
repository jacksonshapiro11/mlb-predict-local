#!/usr/bin/env python3
"""
quick_start.py
==============
Quick start guide for the MLB Pitch Prediction pipeline.

This script demonstrates the complete workflow:
1. Download raw Statcast data
2. Build historical features
3. Test the features
4. Train models (optional)

Usage:
    python quick_start.py --demo     # Demo with 2023 data only
    python quick_start.py --full     # Full pipeline with multiple years
"""

import argparse
import subprocess
import pathlib
import time


def run_command(cmd, description):
    """Run a command with nice output formatting."""
    print(f"\n🔄 {description}")
    print(f"💻 Running: {cmd}")
    print("-" * 60)

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s (exit code {e.returncode})")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("🔍 Checking dependencies...")

    required_packages = [
        "pandas",
        "numpy",
        "duckdb",
        "pybaseball",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False

    print("✅ All dependencies satisfied")
    return True


def demo_pipeline():
    """Run a demo of the pipeline with 2023 data."""
    print("🚀 Running MLB Pitch Prediction Demo Pipeline")
    print("=" * 60)

    year = 2023

    # Step 1: Download raw data
    if not run_command(
        f"python etl/fetch_statcast.py {year}", f"Downloading Statcast data for {year}"
    ):
        return False

    # Step 2: Build historical features
    if not run_command(
        f"python etl/build_historical_features.py {year}",
        f"Building historical features for {year}",
    ):
        return False

    # Step 3: Test the features
    if not run_command(
        "python test_historical_pipeline.py", "Testing historical features"
    ):
        return False

    # Step 4: Test for data leakage
    if not run_command("python run_full_pipeline.py test", "Testing for data leakage"):
        return False

    print("\n🎉 Demo pipeline completed successfully!")
    print(f"📁 Check data/features_historical/ for output files")

    return True


def full_pipeline():
    """Run the full pipeline with multiple years."""
    print("🚀 Running Full MLB Pitch Prediction Pipeline")
    print("=" * 60)

    years = [2022, 2023]  # Can be extended

    # Step 1: Download raw data for all years
    for year in years:
        if not run_command(
            f"python etl/fetch_statcast.py {year}",
            f"Downloading Statcast data for {year}",
        ):
            return False

    # Step 2: Build historical features for all years
    for year in years:
        if not run_command(
            f"python etl/build_historical_features.py {year}",
            f"Building historical features for {year}",
        ):
            return False

    # Step 3: Test the features
    if not run_command(
        "python test_historical_pipeline.py", "Testing historical features"
    ):
        return False

    # Step 4: Test for data leakage
    if not run_command("python run_full_pipeline.py test", "Testing for data leakage"):
        return False

    # Step 5: Train models (optional - can be resource intensive)
    print("\n🤖 Model Training Available")
    print("To train models, run:")
    print(f"python run_full_pipeline.py train \\")
    print(f"    --train-years {' '.join(map(str, years[:-1]))} \\")
    print(f'    --val "2024-04-01:2024-07-31" \\')
    print(f'    --test "2024-08-01:2024-10-31"')

    print("\n🎉 Full pipeline setup completed successfully!")

    return True


def main():
    parser = argparse.ArgumentParser(description="MLB Pitch Prediction Quick Start")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo with 2023 data only"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full pipeline with multiple years"
    )
    parser.add_argument(
        "--check-deps", action="store_true", help="Only check dependencies"
    )

    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        return

    if not args.demo and not args.full:
        print("Usage: python quick_start.py [--demo|--full|--check-deps]")
        print("  --demo: Quick demo with 2023 data")
        print("  --full: Full pipeline with multiple years")
        print("  --check-deps: Check if dependencies are installed")
        return

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing")
        return

    # Create necessary directories
    pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/features_historical").mkdir(parents=True, exist_ok=True)
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    if args.demo:
        success = demo_pipeline()
    elif args.full:
        success = full_pipeline()

    if success:
        print("\n📚 Next Steps:")
        print("1. Review the README.md for detailed documentation")
        print("2. Examine the generated features in data/features_historical/")
        print("3. Run test_historical_pipeline.py to validate features")
        print("4. Run 'python run_full_pipeline.py test' to verify no data leakage")
        print("5. Train models with run_full_pipeline.py when ready")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")


if __name__ == "__main__":
    main()
