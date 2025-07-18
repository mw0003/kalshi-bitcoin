#!/usr/bin/env python3
"""
Test script for probabilistic forecasting pipeline
"""

import numpy as np
import pandas as pd
from models.random_forest_model import RandomForestForecaster
from probabilistic_forecasting import CalibrationAnalyzer
from config import PROBABILISTIC_CONFIG

def test_probabilistic_features():
    """Test probabilistic forecasting features with synthetic data"""
    print("Testing probabilistic forecasting features...")
    
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    n_horizons = 20
    
    X_test = pd.DataFrame(np.random.randn(n_samples, n_features), 
                         columns=[f'feature_{i}' for i in range(n_features)])
    y_test = pd.DataFrame(np.random.randn(n_samples, n_horizons) * 1000 + 50000,
                         columns=[f'target_{i+1}' for i in range(n_horizons)])
    
    forecaster = RandomForestForecaster()
    
    forecaster.is_fitted = True
    forecaster.feature_names = X_test.columns.tolist()
    
    from sklearn.preprocessing import RobustScaler
    forecaster.scaler = RobustScaler()
    forecaster.scaler.fit(X_test)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    forecaster.model = MultiOutputRegressor(base_model)
    forecaster.model.fit(X_test, y_test)
    
    print("✓ Model setup complete")
    
    try:
        quantiles = PROBABILISTIC_CONFIG['quantiles']
        quantile_preds = forecaster.predict_quantiles(X_test, quantiles)
        print(f"✓ Quantile predictions: {list(quantile_preds.keys())}")
        print(f"  Shape for q=0.5: {quantile_preds[0.5].shape}")
    except Exception as e:
        print(f"✗ Quantile prediction failed: {e}")
        return False
    
    try:
        thresholds = {
            'above_51000': 51000,
            'below_49000': 49000,
            'between_49500_50500': (49500, 50500)
        }
        probabilities = forecaster.calculate_probabilities(X_test, thresholds)
        print(f"✓ Probability calculations: {list(probabilities.keys())}")
        for name, probs in probabilities.items():
            print(f"  {name}: shape {probs.shape}, mean {np.mean(probs):.3f}")
    except Exception as e:
        print(f"✗ Probability calculation failed: {e}")
        return False
    
    try:
        intervals = forecaster.get_prediction_intervals(X_test)
        print(f"✓ Prediction intervals: {list(intervals.keys())}")
        for conf, interval in intervals.items():
            print(f"  {conf}: lower shape {interval['lower'].shape}, upper shape {interval['upper'].shape}")
    except Exception as e:
        print(f"✗ Prediction intervals failed: {e}")
        return False
    
    try:
        calibrator = CalibrationAnalyzer()
        calibration_results = calibrator.evaluate_calibration(
            y_test.values, probabilities, horizons=[1, 5, 10]
        )
        print(f"✓ Calibration analysis: {len(calibration_results)} queries analyzed")
        
        for query, results in calibration_results.items():
            print(f"  {query}: {len(results)} horizons")
            for horizon, metrics in results.items():
                if 'brier_score' in metrics:
                    print(f"    {horizon}: Brier={metrics['brier_score']:.4f}")
    except Exception as e:
        print(f"✗ Calibration analysis failed: {e}")
        return False
    
    print("✓ All probabilistic features working correctly!")
    return True

if __name__ == "__main__":
    success = test_probabilistic_features()
    if success:
        print("\n🎉 Probabilistic forecasting test PASSED!")
    else:
        print("\n❌ Probabilistic forecasting test FAILED!")
        exit(1)
