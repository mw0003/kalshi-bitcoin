#!/usr/bin/env python3
"""
Demo script using existing trained model to show probabilistic forecasting features
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

def demo_with_existing_model():
    """Demo probabilistic features using existing trained model"""
    print("=" * 80)
    print("PROBABILISTIC FORECASTING DEMO - USING EXISTING MODEL")
    print("=" * 80)
    
    print("STEP 1: LOADING EXISTING TRAINED MODEL")
    print("-" * 40)
    
    model_path = 'trained_models/random_forest_model.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    forecaster = RandomForestForecaster()
    forecaster.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Model type: {forecaster.model_name}")
    print(f"✓ Feature count: {len(forecaster.feature_names) if forecaster.feature_names else 'Unknown'}")
    
    print("\nSTEP 2: PREPARING TEST DATA")
    print("-" * 40)
    
    data = pd.read_csv('bitcoin_2year_data.csv', index_col=0, parse_dates=True)
    test_data = data.tail(5000).copy()  # Use last 5000 records for testing
    print(f"Using test data: {len(test_data)} records")
    print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    engineer = EnhancedFeatureEngineer()
    with ProgressTimer("Feature engineering"):
        features_df = engineer.engineer_features(test_data)
    
    if forecaster.feature_names:
        available_features = [f for f in forecaster.feature_names if f in features_df.columns]
        missing_features = [f for f in forecaster.feature_names if f not in features_df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing from test data")
        
        X_test = features_df[available_features].dropna()
    else:
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith('target_') and col not in ['open', 'high', 'low', 'close', 'volume']]
        X_test = features_df[feature_cols].dropna()
    
    print(f"Test features shape: {X_test.shape}")
    
    X_demo = X_test.head(1000)
    print(f"Demo subset: {X_demo.shape}")
    
    print("\nSTEP 3: POINT PREDICTIONS")
    print("-" * 40)
    
    predictions = forecaster.predict(X_demo)
    print(f"✓ Point predictions generated: {predictions.shape}")
    
    pred_df = pd.DataFrame(predictions, 
                          columns=[f'horizon_{i+1}' for i in range(predictions.shape[1])],
                          index=X_demo.index)
    pred_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_point_predictions.csv')
    pred_df.to_csv(pred_file)
    print(f"✓ Predictions saved to: {pred_file}")
    
    print("\nSTEP 4: PROBABILISTIC FORECASTING")
    print("-" * 40)
    
    print("Generating quantile predictions...")
    quantile_preds = forecaster.predict_quantiles(X_demo)
    print(f"✓ Quantiles generated: {list(quantile_preds.keys())}")
    
    quantile_files = {}
    for q, q_preds in quantile_preds.items():
        q_df = pd.DataFrame(q_preds,
                           columns=[f'horizon_{i+1}' for i in range(q_preds.shape[1])],
                           index=X_demo.index)
        q_file = os.path.join(OUTPUT_CONFIG['results_dir'], f'demo_quantile_{q}.csv')
        q_df.to_csv(q_file)
        quantile_files[q] = q_file
        print(f"✓ Quantile {q} saved to: {q_file}")
    
    current_price = test_data['close'].iloc[-1]
    print(f"\nCurrent Bitcoin price: ${current_price:.2f}")
    
    thresholds = {
        f'above_{int(current_price * 1.01)}': current_price * 1.01,  # 1% above
        f'above_{int(current_price * 1.02)}': current_price * 1.02,  # 2% above  
        f'below_{int(current_price * 0.99)}': current_price * 0.99,  # 1% below
        f'below_{int(current_price * 0.98)}': current_price * 0.98,  # 2% below
        f'between_{int(current_price * 0.995)}_{int(current_price * 1.005)}': (current_price * 0.995, current_price * 1.005)
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
    
    prob_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_probability_queries.csv')
    pd.DataFrame(prob_results).to_csv(prob_file, index=False)
    print(f"✓ Probability results saved to: {prob_file}")
    
    print("\nSAMPLE PROBABILITY RESULTS:")
    for query_name, prob_array in probabilities.items():
        print(f"  {query_name}:")
        print(f"    Horizon 1:  {np.mean(prob_array[:, 0]):.3f} ± {np.std(prob_array[:, 0]):.3f}")
        print(f"    Horizon 5:  {np.mean(prob_array[:, 4]):.3f} ± {np.std(prob_array[:, 4]):.3f}")
        print(f"    Horizon 20: {np.mean(prob_array[:, -1]):.3f} ± {np.std(prob_array[:, -1]):.3f}")
    
    print("\nSTEP 5: CALIBRATION ANALYSIS")
    print("-" * 40)
    
    try:
        target_cols = [col for col in features_df.columns if col.startswith('target_')]
        if target_cols:
            y_demo = features_df.loc[X_demo.index, target_cols].dropna()
            if len(y_demo) > 0:
                common_idx = X_demo.index.intersection(y_demo.index)
                X_demo_aligned = X_demo.loc[common_idx]
                y_demo_aligned = y_demo.loc[common_idx]
                
                if len(X_demo_aligned) > 100:  # Need sufficient data for calibration
                    prob_aligned = forecaster.calculate_probabilities(X_demo_aligned, thresholds)
                    
                    calibrator = CalibrationAnalyzer()
                    calibration_results = calibrator.evaluate_calibration(
                        y_demo_aligned.values, prob_aligned, horizons=[1, 5, 10, 20]
                    )
                    
                    cal_metrics_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'demo_calibration_metrics.csv')
                    calibrator.save_calibration_metrics(calibration_results, cal_metrics_file)
                    
                    cal_plot_file = os.path.join(OUTPUT_CONFIG['plots_dir'], 'demo_calibration_curves.png')
                    calibrator.plot_calibration_curve(calibration_results, cal_plot_file)
                    
                    print(f"✓ Calibration metrics saved to: {cal_metrics_file}")
                    print(f"✓ Calibration plots saved to: {cal_plot_file}")
                    
                    print("\nSAMPLE CALIBRATION METRICS:")
                    for query, results in list(calibration_results.items())[:2]:
                        print(f"  {query}:")
                        for horizon, metrics in results.items():
                            if 'brier_score' in metrics:
                                brier = metrics['brier_score']
                                ece = metrics.get('ece', 'N/A')
                                print(f"    {horizon}: Brier={brier:.4f}, ECE={ece}")
                else:
                    print("⚠ Insufficient aligned data for calibration analysis")
            else:
                print("⚠ No target data available for calibration analysis")
        else:
            print("⚠ No target columns found for calibration analysis")
            
    except Exception as cal_error:
        print(f"⚠ Calibration analysis failed: {cal_error}")
    
    print("\nSTEP 6: MODEL PERSISTENCE DEMONSTRATION")
    print("-" * 40)
    print(f"✓ Model artifacts location: {model_path}")
    print(f"✓ Model can be reloaded using: forecaster.load_model('{model_path}')")
    print(f"✓ Model includes: weights, scaler, feature names, config")
    
    print("\n" + "=" * 80)
    print("PROBABILISTIC FORECASTING DEMO COMPLETED!")
    print("=" * 80)
    print("Generated Artifacts:")
    print(f"✓ Point predictions: {pred_file}")
    print(f"✓ Quantile predictions: {len(quantile_files)} files in results/")
    print(f"✓ Probability queries: {prob_file}")
    if 'cal_metrics_file' in locals():
        print(f"✓ Calibration metrics: {cal_metrics_file}")
        print(f"✓ Calibration plots: {cal_plot_file}")
    print(f"✓ Trained model: {model_path}")
    
    return {
        'model_path': model_path,
        'predictions_file': pred_file,
        'quantile_files': quantile_files,
        'probabilities_file': prob_file,
        'calibration_metrics_file': locals().get('cal_metrics_file'),
        'calibration_plot_file': locals().get('cal_plot_file')
    }

if __name__ == "__main__":
    setup_logging('demo_probabilistic.log')
    results = demo_with_existing_model()
    if results:
        print(f"\n🎉 Demo completed! All artifacts ready for inspection.")
    else:
        print(f"\n❌ Demo failed!")
