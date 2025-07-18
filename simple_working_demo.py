#!/usr/bin/env python3
"""
Simple working demo of probabilistic forecasting features
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from models.random_forest_model import RandomForestForecaster
from enhanced_feature_engineering import EnhancedFeatureEngineer
from probabilistic_forecasting import CalibrationAnalyzer
from utils import setup_logging, ProgressTimer
from config import OUTPUT_CONFIG

def create_simple_demo():
    """Create a simple working demo with minimal data"""
    print("=" * 80)
    print("SIMPLE PROBABILISTIC FORECASTING DEMO")
    print("=" * 80)
    
    print("STEP 1: LOADING DATA")
    print("-" * 40)
    data = pd.read_csv('bitcoin_2year_data.csv', index_col=0, parse_dates=True)
    
    data_subset = data.tail(10000).copy()
    print(f"Using subset: {len(data_subset)} records")
    print(f"Date range: {data_subset.index[0]} to {data_subset.index[-1]}")
    print(f"Price range: ${data_subset['close'].min():.2f} - ${data_subset['close'].max():.2f}")
    
    print("\nSTEP 2: SIMPLE FEATURE ENGINEERING")
    print("-" * 40)
    
    features_df = data_subset.copy()
    
    for lag in [1, 2, 5, 10]:
        features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
    
    for window in [5, 20]:
        features_df[f'close_ma_{window}'] = features_df['close'].rolling(window).mean()
        features_df[f'close_std_{window}'] = features_df['close'].rolling(window).std()
    
    features_df['price_change_1'] = features_df['close'].pct_change()
    features_df['price_change_5'] = features_df['close'].pct_change(5)
    
    features_df['volume_ma_5'] = features_df['volume'].rolling(5).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_5']
    
    target_horizons = [1, 2, 5, 10, 20]
    for h in target_horizons:
        features_df[f'target_{h}'] = features_df['close'].shift(-h)
    
    print(f"Features created: {features_df.shape[1]} columns")
    
    print("\nSTEP 3: DATA PREPARATION")
    print("-" * 40)
    
    feature_cols = [col for col in features_df.columns 
                   if not col.startswith('target_') and col not in ['open', 'high', 'low', 'close', 'volume']]
    target_cols = [f'target_{h}' for h in target_horizons]
    
    clean_data = features_df.dropna()
    X = clean_data[feature_cols]
    y = clean_data[target_cols]
    
    print(f"Clean data shape: X{X.shape}, y{y.shape}")
    print(f"Features: {feature_cols}")
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nSTEP 4: QUICK MODEL TRAINING")
    print("-" * 40)
    
    quick_config = {
        'n_estimators': 10,  # Very few trees for speed
        'max_depth': 5,
        'n_jobs': -1
    }
    
    forecaster = RandomForestForecaster(config=quick_config)
    
    with ProgressTimer("Quick model training"):
        forecaster.fit(X_train, y_train)
    
    print("✓ Model trained successfully")
    
    print("\nSTEP 5: POINT PREDICTIONS")
    print("-" * 40)
    
    predictions = forecaster.predict(X_test)
    print(f"✓ Point predictions: {predictions.shape}")
    
    pred_df = pd.DataFrame(predictions, 
                          columns=[f'horizon_{h}' for h in target_horizons],
                          index=X_test.index)
    pred_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'simple_demo_predictions.csv')
    pred_df.to_csv(pred_file)
    print(f"✓ Saved to: {pred_file}")
    
    print("\nSTEP 6: PROBABILISTIC FORECASTING")
    print("-" * 40)
    
    print("Generating quantile predictions...")
    quantile_preds = forecaster.predict_quantiles(X_test)
    print(f"✓ Quantiles: {list(quantile_preds.keys())}")
    
    quantile_files = {}
    for q, q_preds in quantile_preds.items():
        q_df = pd.DataFrame(q_preds,
                           columns=[f'horizon_{h}' for h in target_horizons],
                           index=X_test.index)
        q_file = os.path.join(OUTPUT_CONFIG['results_dir'], f'simple_demo_quantile_{q}.csv')
        q_df.to_csv(q_file)
        quantile_files[q] = q_file
        print(f"✓ Quantile {q}: {q_file}")
    
    current_price = data_subset['close'].iloc[-1]
    print(f"\nCurrent Bitcoin price: ${current_price:.2f}")
    
    thresholds = {
        f'above_{int(current_price * 1.005)}': current_price * 1.005,  # 0.5% above
        f'above_{int(current_price * 1.01)}': current_price * 1.01,    # 1% above
        f'below_{int(current_price * 0.995)}': current_price * 0.995,  # 0.5% below
        f'below_{int(current_price * 0.99)}': current_price * 0.99,    # 1% below
        f'between_{int(current_price * 0.998)}_{int(current_price * 1.002)}': (current_price * 0.998, current_price * 1.002)
    }
    
    print("Calculating probability queries...")
    probabilities = forecaster.calculate_probabilities(X_test, thresholds)
    print(f"✓ Probabilities calculated for {len(probabilities)} queries")
    
    prob_results = []
    for query_name, prob_array in probabilities.items():
        for i, horizon in enumerate(target_horizons):
            prob_results.append({
                'query': query_name,
                'horizon': horizon,
                'mean_probability': np.mean(prob_array[:, i]),
                'std_probability': np.std(prob_array[:, i]),
                'min_probability': np.min(prob_array[:, i]),
                'max_probability': np.max(prob_array[:, i])
            })
    
    prob_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'simple_demo_probabilities.csv')
    pd.DataFrame(prob_results).to_csv(prob_file, index=False)
    print(f"✓ Saved to: {prob_file}")
    
    print("\nSAMPLE PROBABILITY RESULTS:")
    for query_name, prob_array in probabilities.items():
        print(f"  {query_name}:")
        for i, horizon in enumerate([1, 5, 20]):
            if i < prob_array.shape[1]:
                mean_prob = np.mean(prob_array[:, i])
                std_prob = np.std(prob_array[:, i])
                print(f"    Horizon {target_horizons[i]:2d}: {mean_prob:.3f} ± {std_prob:.3f}")
    
    print("\nSTEP 7: CALIBRATION ANALYSIS")
    print("-" * 40)
    
    try:
        calibrator = CalibrationAnalyzer()
        calibration_results = calibrator.evaluate_calibration(
            y_test.values, probabilities, horizons=[1, 5, 10, 20]
        )
        
        cal_metrics_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'simple_demo_calibration.csv')
        calibrator.save_calibration_metrics(calibration_results, cal_metrics_file)
        print(f"✓ Calibration metrics: {cal_metrics_file}")
        
        cal_plot_file = os.path.join(OUTPUT_CONFIG['plots_dir'], 'simple_demo_calibration.png')
        calibrator.plot_calibration_curve(calibration_results, cal_plot_file)
        print(f"✓ Calibration plots: {cal_plot_file}")
        
        print("\nSAMPLE CALIBRATION METRICS:")
        for query, results in list(calibration_results.items())[:2]:
            print(f"  {query}:")
            for horizon, metrics in list(results.items())[:3]:
                if 'brier_score' in metrics:
                    brier = metrics['brier_score']
                    ece = metrics.get('ece', 'N/A')
                    print(f"    {horizon}: Brier={brier:.4f}, ECE={ece}")
        
    except Exception as cal_error:
        print(f"⚠ Calibration analysis failed: {cal_error}")
        cal_metrics_file = None
        cal_plot_file = None
    
    print("\nSTEP 8: MODEL PERSISTENCE")
    print("-" * 40)
    
    model_path = forecaster.save_model('trained_models/simple_demo_model.joblib')
    print(f"✓ Model saved: {model_path}")
    
    print("\n" + "=" * 80)
    print("SIMPLE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Generated Artifacts:")
    print(f"✓ Point predictions: {pred_file}")
    print(f"✓ Quantile files: {len(quantile_files)} files")
    print(f"✓ Probability queries: {prob_file}")
    if cal_metrics_file:
        print(f"✓ Calibration metrics: {cal_metrics_file}")
        print(f"✓ Calibration plots: {cal_plot_file}")
    print(f"✓ Trained model: {model_path}")
    
    return {
        'model_path': model_path,
        'predictions_file': pred_file,
        'quantile_files': quantile_files,
        'probabilities_file': prob_file,
        'calibration_metrics_file': cal_metrics_file,
        'calibration_plot_file': cal_plot_file,
        'current_price': current_price,
        'data_range': (data_subset.index[0], data_subset.index[-1])
    }

if __name__ == "__main__":
    setup_logging('simple_demo.log')
    results = create_simple_demo()
    if results:
        print(f"\n🎉 Simple demo completed successfully!")
        print(f"📊 Data: {results['data_range'][0]} to {results['data_range'][1]}")
        print(f"💰 Current price: ${results['current_price']:.2f}")
        print(f"📁 All artifacts saved in results/ and trained_models/")
    else:
        print(f"\n❌ Demo failed!")
