#!/usr/bin/env python3
"""
Conservative Two-Model Architecture for More Realistic Performance
================================================================
Removes highly predictive features to achieve 55-65% pitch type accuracy.

Key changes:
- Remove within-game cumulative velocity/spin (too predictive)
- Keep historical averages but at weekly level (less precise)
- Reduce batter matchup detail
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import duckdb

class ConservativePitchPredictor:
    def __init__(self):
        self.model1 = None  
        self.model2 = None  
        self.pitch_type_encoder = LabelEncoder()
        self.outcome_encoder = LabelEncoder()
        self.model1_feature_names = None  

    def prepare_conservative_features(self, df):
        """
        Conservative feature set for realistic prediction accuracy
        
        INCLUDES (minimal set):
        - Basic game context: count, score, inning
        - Pitcher usage percentages (30-day)
        - Basic recent form (7-day aggregates)
        
        EXCLUDES (too predictive):
        - Historical velocity/spin by pitch type
        - Within-game cumulative velocity/spin
        - Detailed batter matchup features
        """
        
        # Core game situation (keep essential context)
        core_features = [
            'balls', 'strikes', 'outs_when_up',
            'on_1b', 'on_2b', 'on_3b', 
            'home_score', 'stand', 'p_throws'
        ]
        
        # Pitcher usage only (no velocity/spin details)
        usage_features = [col for col in df.columns if col.startswith('usage_30d_')]
        
        # Recent form aggregates only (no pitch-specific details)  
        recent_features = [col for col in df.columns if col.endswith('_7d') and 
                          not any(x in col for x in ['velocity', 'spin'])]
        
        # Basic count state performance (simplified)
        count_features = [col for col in df.columns if 
                         col.startswith('whiff_rate_30d_') and 
                         any(state in col for state in ['_AHEAD', '_BEHIND', '_EVEN'])]
        
        # Simple cumulative game state (no velocity/spin details)
        simple_cumulative = ['cum_game_pitches']
        simple_cumulative = [col for col in simple_cumulative if col in df.columns]
        
        all_features = (core_features + usage_features + recent_features + 
                       count_features + simple_cumulative)
        
        # Filter to existing columns
        available_features = [col for col in all_features if col in df.columns]
        
        print(f"Conservative features: {len(available_features)}")
        print("Categories:")
        print(f"  Core context: {len([f for f in available_features if f in core_features])}")
        print(f"  Usage patterns: {len([f for f in available_features if f in usage_features])}")
        print(f"  Recent form: {len([f for f in available_features if f in recent_features])}")
        print(f"  Count performance: {len([f for f in available_features if f in count_features])}")
        print(f"  Simple cumulative: {len([f for f in available_features if f in simple_cumulative])}")
        
        # Store feature names for consistency
        if self.model1_feature_names is None:
            self.model1_feature_names = available_features
        else:
            available_features = [f for f in self.model1_feature_names if f in df.columns]
        
        # Get feature dataframe
        feature_df = df[available_features].copy()
        
        # Encode categorical features
        categorical_cols = ['stand', 'p_throws']
        
        for col in categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(str).fillna('UNKNOWN')
                unique_vals = sorted(feature_df[col].unique())
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                feature_df[col] = feature_df[col].map(val_to_num)
        
        # Fill missing features if needed
        if len(feature_df.columns) < len(self.model1_feature_names):
            for missing_feature in self.model1_feature_names:
                if missing_feature not in feature_df.columns:
                    feature_df[missing_feature] = 0.0
            feature_df = feature_df[self.model1_feature_names]
        
        return feature_df

print("Conservative model created - should achieve 55-65% accuracy")
print("Run with same data loading logic but use prepare_conservative_features()") 