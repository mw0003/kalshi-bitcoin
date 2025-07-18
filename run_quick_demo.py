#!/usr/bin/env python3
"""
Quick demo script for probabilistic forecasting with manageable dataset size
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from models.random_forest_model import RandomForestForecaster
from enhanced_feature_engineering import EnhancedFeatureEngineer
from probabilistic_forecasting import CalibrationAnalyzer
from utils import setup_logging, ProgressTimer
from config import OUTPUT_CONFIG, VALIDATION_CONFIG

def run_quick_demo():
    """Run a complete demo with manageable dataset size"""
    print("=" * 80)
    print("QUICK PROBABILISTIC FORECASTING DEMO")
    print("=" * 80)
    
    print("STEP 1: LOADING DATA")
    print("-" * 40)
    data = pd.read_csv('bitcoin_2year_data.csv', index_col=0, parse_dates=True)
    
    data_subset = data.tail(50000).copy()
    print(f"Using subset: {len(data_subset)} records")
    print(f"Date range: {data_subset.index[0]} to {data_subset.index[-1]}")
    
    print("\nSTEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    engineer = EnhancedFeatureEngineer()
    with ProgressTimer("Feature engineering"):
        features_df = engineer.engineer_features(data_subset)
    
    print("\nSTEP 3: DATA PREPARATION")
    print("-" * 40)
    feature_cols = [col for col in features_df.columns 
                   if not col.startswith('target_') and col not in ['open', 'high', 'low', 'close', 'volume']]
    target_cols = [col for col in features_df.columns if col.startswith('target_')]
    
    clean_data = features_df.dropna()
    X = clean_data[feature_cols]
    y = clean_data[target_cols]
    
    print(f"Clean data shape: X{X.shape}, y{y.shape}")
    
    print("\nSTEP 4: SKIPPING VALIDATION FOR QUICK DEMO")
    print("-" * 40)
    print("Skipping rolling validation to speed up demo...")
    
    forecaster = RandomForestForecaster()
    
    print("\nSTEP 5: FINAL MODEL TRAINING")
    print("-" * 40)
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    with ProgressTimer("Final model training"):
        forecaster.fit(X_train, y_train)
    
    print("\nSTEP 6: GENERATING PREDICTIONS")
    print("-" * 40)
    
    predictions = forecaster.predict(X_test)
    print(f"Point predictions shape: {predictions.shape}")
    
    pred_df = pd.DataFrame(predictions, 
                          columns=[f'horizon_{i+1}' for i in range(predictions.shape[1])],
                          index=X_test.index)
    pred_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_predictions.csv')
    pred_df.to_csv(pred_file)
    print(f"Predictions saved to: {pred_file}")
    
    print("\nSTEP 7: PROBABILISTIC FORECASTING")
    print("-" * 40)
    
    print("Generating quantile predictions...")
    quantile_preds = forecaster.predict_quantiles(X_test)
    
    for q, q_preds in quantile_preds.items():
        q_df = pd.DataFrame(q_preds,
                           columns=[f'horizon_{i+1}' for i in range(q_preds.shape[1])],
                           index=X_test.index)
        q_file = os.path.join(OUTPUT_CONFIG['results_dir'], f'demo_quantile_{q}.csv')
        q_df.to_csv(q_file)
        print(f"Quantile {q} saved to: {q_file}")
    
    current_price = y_test.iloc[-1, 0] if len(y_test) > 0 else data_subset['close'].iloc[-1]
    print(f"Current price: ${current_price:.2f}")
    
    thresholds = {
        f'above_{int(current_price * 1.01)}': current_price * 1.01,  # 1% above
        f'above_{int(current_price * 1.02)}': current_price * 1.02,  # 2% above
        f'below_{int(current_price * 0.99)}': current_price * 0.99,  # 1% below
        f'below_{int(current_price * 0.98)}': current_price * 0.98,  # 2% below
        f'between_{int(current_price * 0.995)}_{int(current_price * 1.005)}': (current_price * 0.995, current_price * 1.005)
    }
    
    print("Calculating probability queries...")
    probabilities = forecaster.calculate_probabilities(X_test, thresholds)
    
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
    
    prob_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_probabilities.csv')
    pd.DataFrame(prob_results).to_csv(prob_file, index=False)
    print(f"Probability results saved to: {prob_file}")
    
    print("\nSample Probability Results:")
    for query_name, prob_array in probabilities.items():
        print(f"  {query_name}:")
        print(f"    Horizon 1: {np.mean(prob_array[:, 0]):.3f} ± {np.std(prob_array[:, 0]):.3f}")
        print(f"    Horizon 5: {np.mean(prob_array[:, 4]):.3f} ± {np.std(prob_array[:, 4]):.3f}")
        print(f"    Horizon 20: {np.mean(prob_array[:, -1]):.3f} ± {np.std(prob_array[:, -1]):.3f}")
    
    print("\nSTEP 8: CALIBRATION ANALYSIS")
    print("-" * 40)
    
    try:
        calibrator = CalibrationAnalyzer()
        calibration_results = calibrator.evaluate_calibration(
            y_test.values, probabilities, horizons=[1, 5, 10, 20]
        )
        
        cal_metrics_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_calibration_metrics.csv')
        calibrator.save_calibration_metrics(calibration_results, cal_metrics_file)
        
        cal_plot_file = os.path.join(OUTPUT_CONFIG['plots_dir'], 'demo_calibration_curves.png')
        calibrator.plot_calibration_curve(calibration_results, cal_plot_file)
        
        print(f"Calibration metrics saved to: {cal_metrics_file}")
        print(f"Calibration plots saved to: {cal_plot_file}")
        
        print("\nSample Calibration Metrics:")
        for query, results in list(calibration_results.items())[:2]:  # Show first 2 queries
            print(f"  {query}:")
            for horizon, metrics in results.items():
                if 'brier_score' in metrics:
                    print(f"    {horizon}: Brier={metrics['brier_score']:.4f}, ECE={metrics.get('ece', 'N/A'):.4f}")
        
    except Exception as cal_error:
        print(f"Warning: Calibration analysis failed: {cal_error}")
    
    print("\nSTEP 9: MODEL PERSISTENCE")
    print("-" * 40)
    
    model_path = forecaster.save_model('trained_models/demo_model.joblib')
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Generated Artifacts:")
    print(f"- Point predictions: {pred_file}")
    print(f"- Quantile predictions: results/demo_quantile_*.csv")
    print(f"- Probability queries: {prob_file}")
    print(f"- Calibration metrics: {cal_metrics_file}")
    print(f"- Calibration plots: {cal_plot_file}")
    print(f"- Trained model: {model_path}")
    
    return {
        'model_path': model_path,
        'predictions_file': pred_file,
        'probabilities_file': prob_file,
        'calibration_metrics_file': cal_metrics_file,
        'calibration_plot_file': cal_plot_file
    }

if __name__ == "__main__":
    setup_logging('demo.log')
    results = run_quick_demo()
    print(f"\nDemo artifacts ready for inspection!")
