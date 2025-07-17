#!/usr/bin/env python3
"""
Test script to verify the enhanced training pipeline works
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from models.random_forest_model import RandomForestForecaster
    from probabilistic_forecasting import CalibrationAnalyzer
    from config import PROBABILISTIC_CONFIG, OUTPUT_CONFIG
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_enhanced_pipeline():
    """Test the enhanced training pipeline with minimal data"""
    print("Testing enhanced training pipeline...")
    
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    n_horizons = 20
    
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    X_data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_cols)
    
    base_price = 50000
    price_changes = np.random.randn(n_samples, n_horizons) * 100
    y_data = pd.DataFrame(
        base_price + np.cumsum(price_changes, axis=1),
        columns=[f'target_{i+1}' for i in range(n_horizons)]
    )
    
    print(f"Created synthetic data: X{X_data.shape}, y{y_data.shape}")
    
    train_size = int(0.8 * n_samples)
    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    print("Training RandomForest model...")
    forecaster = RandomForestForecaster()
    forecaster.fit(X_train, y_train)
    print("✓ Model training completed")
    
    predictions = forecaster.predict(X_test)
    print(f"✓ Point predictions: {predictions.shape}")
    
    print("Testing probabilistic features...")
    
    quantile_preds = forecaster.predict_quantiles(X_test)
    print(f"✓ Quantile predictions: {list(quantile_preds.keys())}")
    
    current_price = y_test.iloc[-1, 0]
    thresholds = {
        f'above_{int(current_price * 1.02)}': current_price * 1.02,
        f'below_{int(current_price * 0.98)}': current_price * 0.98,
        f'between_{int(current_price * 0.99)}_{int(current_price * 1.01)}': (current_price * 0.99, current_price * 1.01)
    }
    
    probabilities = forecaster.calculate_probabilities(X_test, thresholds)
    print(f"✓ Probability calculations: {list(probabilities.keys())}")
    
    print("Testing calibration analysis...")
    calibrator = CalibrationAnalyzer()
    calibration_results = calibrator.evaluate_calibration(
        y_test.values, probabilities, horizons=[1, 5, 10]
    )
    print(f"✓ Calibration analysis: {len(calibration_results)} queries")
    
    print("Testing model persistence...")
    model_path = forecaster.save_model()
    print(f"✓ Model saved to: {model_path}")
    
    new_forecaster = RandomForestForecaster()
    new_forecaster.load_model(model_path)
    print("✓ Model loaded successfully")
    
    loaded_predictions = new_forecaster.predict(X_test)
    print(f"✓ Loaded model predictions: {loaded_predictions.shape}")
    
    if np.allclose(predictions, loaded_predictions):
        print("✓ Loaded model predictions match original")
    else:
        print("✗ Loaded model predictions don't match!")
        return False
    
    print("✓ All enhanced pipeline tests passed!")
    return True

if __name__ == "__main__":
    success = test_enhanced_pipeline()
    if success:
        print("\n🎉 Enhanced training pipeline test PASSED!")
    else:
        print("\n❌ Enhanced training pipeline test FAILED!")
        sys.exit(1)
