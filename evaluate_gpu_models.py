#!/usr/bin/env python3
"""
evaluate_gpu_models.py
======================
Evaluate GPU-trained LightGBM and CatBoost models.
"""

import pickle
import pathlib
import json
from datetime import date
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

# Import functions from run_full_pipeline
import sys
sys.path.append('.')
from run_full_pipeline import load_parquets, prep_balanced, predict_proba

def evaluate_gpu_models():
    """Evaluate GPU-trained models on test data."""
    
    # Set up paths
    checkpoint_dir = pathlib.Path("models/checkpoint_gpu_20250605_112919")
    
    print("🚀 MLB GPU-Trained Model Evaluation")
    print("=" * 50)
    
    # Load encoders
    print("📁 Loading encoders...")
    with open(checkpoint_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(checkpoint_dir / "target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    
    # Load models
    print("📁 Loading GPU-trained models...")
    models = {}
    
    # LightGBM
    lgb_path = checkpoint_dir / "lgb.lgb"
    models["lgb"] = lgb.Booster(model_file=str(lgb_path))
    print(f"✅ LightGBM: {lgb_path.stat().st_size / (1024*1024):.1f} MB")
    
    # CatBoost
    cat_path = checkpoint_dir / "cat.cbm"
    models["cat"] = CatBoostClassifier()
    models["cat"].load_model(str(cat_path))
    print(f"✅ CatBoost: {cat_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Load test data (same as training command)
    print("\n📊 Loading test data...")
    test_years = [2024, 2025]
    test_range = "2024-08-01:2025-12-31"
    
    test_df = load_parquets(test_years, test_range)
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare test data
    X_te, y_te, _, _ = prep_balanced(test_df, label_encoders)
    y_te_enc = target_encoder.transform(y_te)
    
    print(f"Prepared test data: X={X_te.shape}, y={y_te.shape}")
    print(f"Target classes: {target_encoder.classes_}")
    
    # Class distribution
    unique, counts = np.unique(y_te_enc, return_counts=True)
    print("\nTest set class distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = target_encoder.classes_[cls_idx]
        pct = count / len(y_te_enc) * 100
        print(f"  {cls_name}: {count:,} ({pct:.1f}%)")
    
    # Evaluate individual models
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name.upper()}...")
        
        # Get predictions
        proba = predict_proba(model, X_te, model_name)
        preds = proba.argmax(1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_te_enc, preds)
        logloss = log_loss(y_te_enc, proba)
        
        # Top-3 accuracy
        top3_acc = np.mean([
            y_te_enc[i] in np.argsort(proba[i])[-3:] 
            for i in range(len(y_te_enc))
        ])
        
        results[model_name] = {
            "accuracy": accuracy,
            "top3_accuracy": top3_acc,
            "logloss": logloss
        }
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Top-3 Accuracy: {top3_acc:.3f} ({top3_acc*100:.1f}%)")
        print(f"   Log Loss: {logloss:.3f}")
        
        # Per-class accuracy
        print(f"   Per-class accuracy:")
        for cls_idx in unique:
            cls_name = target_encoder.classes_[cls_idx]
            cls_mask = y_te_enc == cls_idx
            if cls_mask.sum() > 0:
                cls_acc = accuracy_score(y_te_enc[cls_mask], preds[cls_mask])
                print(f"     {cls_name}: {cls_acc:.3f} ({cls_acc*100:.1f}%)")
    
    # Create ensemble
    print(f"\n🎯 Creating ensemble of LightGBM + CatBoost...")
    
    # Equal-weight ensemble
    weights = {"lgb": 0.5, "cat": 0.5}
    ensemble_proba = (
        weights["lgb"] * predict_proba(models["lgb"], X_te, "lgb") +
        weights["cat"] * predict_proba(models["cat"], X_te, "cat")
    )
    ensemble_preds = ensemble_proba.argmax(1)
    
    # Calculate ensemble metrics
    ens_accuracy = accuracy_score(y_te_enc, ensemble_preds)
    ens_logloss = log_loss(y_te_enc, ensemble_proba)
    ens_top3_acc = np.mean([
        y_te_enc[i] in np.argsort(ensemble_proba[i])[-3:] 
        for i in range(len(y_te_enc))
    ])
    
    results["ensemble"] = {
        "accuracy": ens_accuracy,
        "top3_accuracy": ens_top3_acc,
        "logloss": ens_logloss,
        "weights": weights
    }
    
    print(f"   Ensemble Accuracy: {ens_accuracy:.3f} ({ens_accuracy*100:.1f}%)")
    print(f"   Ensemble Top-3 Accuracy: {ens_top3_acc:.3f} ({ens_top3_acc*100:.1f}%)")
    print(f"   Ensemble Log Loss: {ens_logloss:.3f}")
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🏆 FINAL GPU MODEL RESULTS")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.3f} ({metrics['top3_accuracy']*100:.1f}%)")
        print(f"  Log Loss: {metrics['logloss']:.3f}")
    
    # Save results
    results_file = f"gpu_model_results_{date.today().strftime('%Y%m%d')}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    evaluate_gpu_models() 