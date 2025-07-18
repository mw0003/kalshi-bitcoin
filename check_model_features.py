#!/usr/bin/env python3
"""
Check what features the trained model expects
"""

import joblib
import os

def check_model_features():
    model_path = 'trained_models/simple_demo_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    model_data = joblib.load(model_path)
    print(f"Model data keys: {model_data.keys()}")
    
    if 'feature_names' in model_data:
        features = model_data['feature_names']
        print(f"\nModel expects {len(features)} features:")
        for i, feature in enumerate(features, 1):
            print(f"{i:2d}. {feature}")
    
    if 'scaler' in model_data:
        scaler = model_data['scaler']
        print(f"\nScaler feature names: {getattr(scaler, 'feature_names_in_', 'Not available')}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Scaler expects {len(scaler.feature_names_in_)} features")

if __name__ == "__main__":
    check_model_features()
