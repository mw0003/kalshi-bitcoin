#!/usr/bin/env python3
"""
Quick training script to create a model with real feature names for demo
"""

import pandas as pd
import numpy as np
from models.random_forest_model import RandomForestForecaster
from enhanced_feature_engineering import EnhancedFeatureEngineer
from utils import setup_logging

def train_quick_model():
    """Train a model quickly using a subset of real data"""
    print("Loading real Bitcoin data...")
    data = pd.read_csv('bitcoin_2year_data.csv', index_col=0, parse_dates=True)
    
    data_subset = data.tail(10000).copy()
    print(f"Using subset of {len(data_subset)} records for quick training")
    
    print("Engineering features...")
    engineer = EnhancedFeatureEngineer()
    features_df = engineer.engineer_features(data_subset)
    
    feature_cols = [col for col in features_df.columns 
                   if not col.startswith('target_') and col not in ['open', 'high', 'low', 'close', 'volume']]
    target_cols = [col for col in features_df.columns if col.startswith('target_')]
    
    clean_data = features_df.dropna()
    X = clean_data[feature_cols]
    y = clean_data[target_cols]
    
    print(f"Clean data shape: X{X.shape}, y{y.shape}")
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training with {len(X_train)} samples...")
    
    forecaster = RandomForestForecaster()
    forecaster.fit(X_train, y_train)
    
    predictions = forecaster.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    print("Testing probabilistic features...")
    quantile_preds = forecaster.predict_quantiles(X_test.head(10))
    print(f"Quantile predictions: {list(quantile_preds.keys())}")
    
    model_path = forecaster.save_model('trained_models/quick_real_model.joblib')
    print(f"Model saved to: {model_path}")
    
    return model_path

if __name__ == "__main__":
    setup_logging('quick_training.log')
    model_path = train_quick_model()
    print(f"\nQuick model training complete! Model saved to: {model_path}")
