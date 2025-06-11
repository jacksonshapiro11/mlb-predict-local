#!/usr/bin/env python3
"""
evaluate_catboost_only.py
========================
Evaluate only the CatBoost model (LightGBM model is corrupted).
"""

import pickle
import pathlib
import json
from datetime import date
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# Import functions from run_full_pipeline
import sys
sys.path.append('.')
from run_full_pipeline import load_parquets, prep_balanced

def evaluate_catboost_model():
    """Evaluate CatBoost model on test data."""
    
    # Set up paths
    checkpoint_dir = pathlib.Path("models/checkpoint_gpu_20250605_112919")
    
    print("🚀 MLB CatBoost Model Evaluation")
    print("=" * 50)
    
    # Load encoders
    print("📁 Loading encoders...")
    with open(checkpoint_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(checkpoint_dir / "target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    
    # Load CatBoost model
    print("📁 Loading CatBoost model...")
    cat_path = checkpoint_dir / "cat.cbm"
    model = CatBoostClassifier()
    model.load_model(str(cat_path))
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
    print("\n📊 Test set class distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = target_encoder.classes_[cls_idx]
        pct = count / len(y_te_enc) * 100
        print(f"  {cls_name}: {count:,} ({pct:.1f}%)")
    
    # Evaluate CatBoost model
    print(f"\n🔍 Evaluating CatBoost...")
    
    # Get predictions
    proba = model.predict_proba(X_te)
    preds = proba.argmax(1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_te_enc, preds)
    logloss = log_loss(y_te_enc, proba)
    
    # Top-3 accuracy
    top3_acc = np.mean([
        y_te_enc[i] in np.argsort(proba[i])[-3:] 
        for i in range(len(y_te_enc))
    ])
    
    print(f"\n🎯 CATBOOST RESULTS")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Top-3 Accuracy: {top3_acc:.3f} ({top3_acc*100:.1f}%)")
    print(f"   Log Loss: {logloss:.3f}")
    
    # Per-class accuracy
    print(f"\n📋 Per-class accuracy:")
    for cls_idx in unique:
        cls_name = target_encoder.classes_[cls_idx]
        cls_mask = y_te_enc == cls_idx
        if cls_mask.sum() > 0:
            cls_acc = accuracy_score(y_te_enc[cls_mask], preds[cls_mask])
            print(f"     {cls_name}: {cls_acc:.3f} ({cls_acc*100:.1f}%) [{cls_mask.sum():,} samples]")
    
    # Confusion matrix
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_te_enc, preds)
    class_names = target_encoder.classes_
    
    # Print header
    print("      ", end="")
    for name in class_names:
        print(f"{name:>6}", end="")
    print()
    
    # Print matrix
    for i, actual_class in enumerate(class_names):
        print(f"{actual_class:>6}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>6}", end="")
        print()
    
    # Save results
    results = {
        "catboost": {
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_acc),
            "logloss": float(logloss)
        },
        "test_samples": len(y_te_enc),
        "classes": target_encoder.classes_.tolist(),
        "per_class_accuracy": {}
    }
    
    for cls_idx in unique:
        cls_name = target_encoder.classes_[cls_idx]
        cls_mask = y_te_enc == cls_idx
        if cls_mask.sum() > 0:
            cls_acc = accuracy_score(y_te_enc[cls_mask], preds[cls_mask])
            results["per_class_accuracy"][cls_name] = {
                "accuracy": float(cls_acc),
                "samples": int(cls_mask.sum())
            }
    
    results_file = f"catboost_results_{date.today().strftime('%Y%m%d')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    evaluate_catboost_model() 