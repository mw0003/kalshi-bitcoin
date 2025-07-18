#!/usr/bin/env python3
"""
Simple inference script that works with the simple demo model
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

from models.random_forest_model import RandomForestForecaster
from utils import setup_logging, ProgressTimer

def create_simple_features(data):
    """Create the same simple features as used in simple demo training"""
    features_df = data.copy()
    
    for lag in [1, 2, 5, 10]:
        features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
    
    for window in [5, 20]:
        features_df[f'close_ma_{window}'] = features_df['close'].rolling(window).mean()
        features_df[f'close_std_{window}'] = features_df['close'].rolling(window).std()
    
    features_df['price_change_1'] = features_df['close'].pct_change()
    features_df['price_change_5'] = features_df['close'].pct_change(5)
    
    features_df['volume_ma_5'] = features_df['volume'].rolling(5).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_5']
    
    return features_df

def run_simple_inference():
    """Run inference with simple demo model on fresh data"""
    print("=" * 80)
    print("SIMPLE INFERENCE DEMONSTRATION")
    print("=" * 80)
    
    print("STEP 1: LOADING TRAINED MODEL")
    print("-" * 40)
    
    model_path = 'trained_models/simple_demo_model.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    forecaster = RandomForestForecaster()
    forecaster.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Model features: {forecaster.feature_names}")
    
    print("\nSTEP 2: LOADING FRESH DATA")
    print("-" * 40)
    
    data = pd.read_csv('bitcoin_2year_data.csv', index_col=0, parse_dates=True)
    
    fresh_data = data.head(5000).copy()  # First 5000 records instead of last 10000
    print(f"Fresh data: {len(fresh_data)} records")
    print(f"Date range: {fresh_data.index[0]} to {fresh_data.index[-1]}")
    print(f"Price range: ${fresh_data['close'].min():.2f} - ${fresh_data['close'].max():.2f}")
    
    print("\nSTEP 3: FEATURE ENGINEERING")
    print("-" * 40)
    
    with ProgressTimer("Simple feature engineering"):
        features_df = create_simple_features(fresh_data)
    
    feature_cols = ['close_lag_1', 'close_lag_2', 'close_lag_5', 'close_lag_10', 
                   'close_ma_5', 'close_std_5', 'close_ma_20', 'close_std_20', 
                   'price_change_1', 'price_change_5', 'volume_ma_5', 'volume_ratio']
    
    X_new = features_df[feature_cols].dropna()
    print(f"✓ Features prepared: {X_new.shape}")
    print(f"✓ Feature columns: {list(X_new.columns)}")
    
    X_demo = X_new.head(1000)
    print(f"✓ Demo subset: {X_demo.shape}")
    
    print("\nSTEP 4: GENERATING PREDICTIONS")
    print("-" * 40)
    
    predictions = forecaster.predict(X_demo)
    print(f"✓ Point predictions: {predictions.shape}")
    
    os.makedirs('inference_demo_results', exist_ok=True)
    
    pred_df = pd.DataFrame(predictions, 
                          columns=[f'horizon_{i+1}' for i in range(predictions.shape[1])],
                          index=X_demo.index)
    pred_file = 'inference_demo_results/fresh_predictions.csv'
    pred_df.to_csv(pred_file)
    print(f"✓ Saved to: {pred_file}")
    
    print("\nSTEP 5: PROBABILISTIC FORECASTING")
    print("-" * 40)
    
    print("Generating quantile predictions...")
    quantile_preds = forecaster.predict_quantiles(X_demo)
    print(f"✓ Quantiles: {list(quantile_preds.keys())}")
    
    quantile_files = {}
    for q, q_preds in quantile_preds.items():
        q_df = pd.DataFrame(q_preds,
                           columns=[f'horizon_{i+1}' for i in range(q_preds.shape[1])],
                           index=X_demo.index)
        q_file = f'inference_demo_results/fresh_quantile_{q}.csv'
        q_df.to_csv(q_file)
        quantile_files[q] = q_file
        print(f"✓ Quantile {q}: {q_file}")
    
    current_price = fresh_data['close'].iloc[-1]
    print(f"\nCurrent price in fresh data: ${current_price:.2f}")
    
    thresholds = {
        f'above_{int(current_price * 1.005)}': current_price * 1.005,
        f'above_{int(current_price * 1.01)}': current_price * 1.01,
        f'below_{int(current_price * 0.995)}': current_price * 0.995,
        f'below_{int(current_price * 0.99)}': current_price * 0.99,
        f'between_{int(current_price * 0.998)}_{int(current_price * 1.002)}': (current_price * 0.998, current_price * 1.002)
    }
    
    print("Calculating probability queries...")
    probabilities = forecaster.calculate_probabilities(X_demo, thresholds)
    print(f"✓ Probabilities calculated for {len(probabilities)} queries")
    
    prob_results = []
    for query_name, prob_array in probabilities.items():
        for horizon in range(prob_array.shape[1]):
            prob_results.append({
                'query': query_name,
                'horizon': horizon + 1,
                'mean_probability': np.mean(prob_array[:, horizon]),
                'std_probability': np.std(prob_array[:, horizon]),
                'min_probability': np.min(prob_array[:, horizon]),
                'max_probability': np.max(prob_array[:, horizon])
            })
    
    prob_file = 'inference_demo_results/fresh_probabilities.csv'
    pd.DataFrame(prob_results).to_csv(prob_file, index=False)
    print(f"✓ Saved to: {prob_file}")
    
    print("\nSAMPLE PROBABILITY RESULTS ON FRESH DATA:")
    for query_name, prob_array in probabilities.items():
        print(f"  {query_name}:")
        for i, horizon in enumerate([1, 2, 5]):
            if i < prob_array.shape[1]:
                mean_prob = np.mean(prob_array[:, i])
                std_prob = np.std(prob_array[:, i])
                print(f"    Horizon {i+1:2d}: {mean_prob:.3f} ± {std_prob:.3f}")
    
    print("\n" + "=" * 80)
    print("INFERENCE DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print("Model Reuse Demonstration:")
    print(f"✓ Loaded trained model: {model_path}")
    print(f"✓ Applied to fresh data: {fresh_data.index[0]} to {fresh_data.index[-1]}")
    print(f"✓ Generated predictions: {pred_file}")
    print(f"✓ Generated quantiles: {len(quantile_files)} files")
    print(f"✓ Generated probabilities: {prob_file}")
    print(f"✓ Fresh data price: ${current_price:.2f}")
    
    return {
        'model_path': model_path,
        'fresh_data_range': (fresh_data.index[0], fresh_data.index[-1]),
        'fresh_price': current_price,
        'predictions_file': pred_file,
        'quantile_files': quantile_files,
        'probabilities_file': prob_file
    }

if __name__ == "__main__":
    setup_logging('simple_inference.log')
    results = run_simple_inference()
    if results:
        print(f"\n🎉 Inference demonstration completed successfully!")
        print(f"📁 All results saved in inference_demo_results/")
    else:
        print(f"\n❌ Inference demonstration failed!")
