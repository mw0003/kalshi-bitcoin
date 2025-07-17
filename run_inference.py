#!/usr/bin/env python3
"""
Inference Script for Bitcoin Forecasting Models

This script loads trained models and runs probabilistic forecasting on new data.

Usage:
    python run_inference.py --model_path trained_models/random_forest_model.joblib --data_path new_data.csv
"""

import argparse
import os
import pandas as pd
from datetime import datetime

from models.random_forest_model import RandomForestForecaster
from models.lightgbm_model import LightGBMForecaster
from models.lstm_model import LSTMForecaster
from enhanced_feature_engineering import EnhancedFeatureEngineer
from utils import setup_logging, ProgressTimer


def load_model(model_path):
    """Load a trained model from file"""
    print(f"Loading model from: {model_path}")
    
    if 'random_forest' in model_path:
        model = RandomForestForecaster()
    elif 'lightgbm' in model_path:
        model = LightGBMForecaster()
    elif 'lstm' in model_path:
        model = LSTMForecaster()
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")
    
    model.load_model(model_path)
    return model

def prepare_new_data(data_path, model):
    """Prepare new data using the same feature engineering as training"""
    print(f"Loading new data from: {data_path}")
    
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError("Only CSV files supported currently")
    
    engineer = EnhancedFeatureEngineer()
    with ProgressTimer("Feature engineering for new data"):
        features_df = engineer.engineer_features(data)
    
    if model.feature_names:
        available_features = [f for f in model.feature_names if f in features_df.columns]
        missing_features = [f for f in model.feature_names if f not in features_df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features[:10]}...")
        
        X_new = features_df[available_features].dropna()
    else:
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith('target_') and col not in ['open', 'high', 'low', 'close', 'volume']]
        X_new = features_df[feature_cols].dropna()
    
    print(f"Prepared data shape: {X_new.shape}")
    return X_new

def run_inference(model, X_new, output_dir):
    """Run probabilistic inference on new data"""
    print("Running probabilistic inference...")
    
    predictions = model.predict(X_new)
    
    quantile_preds = model.predict_quantiles(X_new)
    prediction_intervals = model.get_prediction_intervals(X_new)
    
    if len(X_new) > 0:
        last_price = 50000
        thresholds = {
            f'above_{int(last_price * 1.01)}': last_price * 1.01,
            f'above_{int(last_price * 1.02)}': last_price * 1.02,
            f'below_{int(last_price * 0.99)}': last_price * 0.99,
            f'below_{int(last_price * 0.98)}': last_price * 0.98,
            f'between_{int(last_price * 0.995)}_{int(last_price * 1.005)}': (last_price * 0.995, last_price * 1.005)
        }
        
        probabilities = model.calculate_probabilities(X_new, thresholds)
    else:
        probabilities = {}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    pred_df = pd.DataFrame(predictions, 
                          columns=[f'horizon_{i+1}' for i in range(predictions.shape[1])],
                          index=X_new.index)
    pred_file = os.path.join(output_dir, f'predictions_{timestamp}.csv')
    pred_df.to_csv(pred_file)
    
    for q, q_preds in quantile_preds.items():
        q_df = pd.DataFrame(q_preds,
                           columns=[f'horizon_{i+1}' for i in range(q_preds.shape[1])],
                           index=X_new.index)
        q_file = os.path.join(output_dir, f'quantile_{q}_{timestamp}.csv')
        q_df.to_csv(q_file)
    
    if probabilities:
        prob_results = []
        for query_name, prob_array in probabilities.items():
            for i, idx in enumerate(X_new.index):
                for horizon in range(prob_array.shape[1]):
                    prob_results.append({
                        'timestamp': idx,
                        'query': query_name,
                        'horizon': horizon + 1,
                        'probability': prob_array[i, horizon]
                    })
        
        prob_df = pd.DataFrame(prob_results)
        prob_file = os.path.join(output_dir, f'probabilities_{timestamp}.csv')
        prob_df.to_csv(prob_file, index=False)
    
    return {
        'predictions_file': pred_file,
        'quantile_files': {q: os.path.join(output_dir, f'quantile_{q}_{timestamp}.csv') for q in quantile_preds.keys()},
        'probabilities_file': prob_file if probabilities else None
    }

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained Bitcoin forecasting model')
    parser.add_argument('--model_path', required=True, help='Path to trained model file')
    parser.add_argument('--data_path', required=True, help='Path to new data CSV file')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging('inference.log')
    
    try:
        model = load_model(args.model_path)
        
        X_new = prepare_new_data(args.data_path, model)
        
        results = run_inference(model, X_new, args.output_dir)
        
        print("\nInference completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Point predictions: {results['predictions_file']}")
        if results['probabilities_file']:
            print(f"Probabilities: {results['probabilities_file']}")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
