#!/usr/bin/env python3
"""
Test script for live trading components
"""

import sys
import os
from datetime import datetime, timedelta

def test_live_data_ingestion():
    """Test live Bitcoin data fetching"""
    print("=" * 50)
    print("TESTING LIVE DATA INGESTION")
    print("=" * 50)
    
    try:
        from data_ingestion import BitcoinDataIngester
        
        ingester = BitcoinDataIngester()
        
        print("1. Testing current price fetch...")
        current_price = ingester.fetch_current_price()
        if current_price:
            print(f"✓ Current price: ${current_price['price']:.2f}")
            print(f"✓ Timestamp: {current_price['timestamp']}")
        else:
            print("✗ Failed to fetch current price")
            return False
        
        print("\n2. Testing live data fetch (last 60 minutes)...")
        live_data = ingester.fetch_live_data(minutes_back=60)
        if not live_data.empty:
            print(f"✓ Fetched {len(live_data)} minutes of data")
            print(f"✓ Latest price: ${live_data['close'].iloc[-1]:.2f}")
            print(f"✓ Time range: {live_data.index.min()} to {live_data.index.max()}")
        else:
            print("✗ Failed to fetch live data")
            return False
        
        print("\n3. Testing feature-ready data fetch...")
        feature_data = ingester.get_live_features_data(minutes_back=120)
        if not feature_data.empty:
            print(f"✓ Fetched {len(feature_data)} minutes for features")
        else:
            print("✗ Failed to fetch feature data")
            return False
        
        print("\n✓ Live data ingestion tests PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Live data ingestion test FAILED: {e}")
        return False

def test_live_forecasting():
    """Test live forecasting script"""
    print("\n" + "=" * 50)
    print("TESTING LIVE FORECASTING")
    print("=" * 50)
    
    try:
        model_path = 'trained_models/simple_demo_model.joblib'
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            print("Run simple_working_demo.py first to create a model")
            return False
        
        print(f"✓ Model found: {model_path}")
        
        print("\n1. Testing with historical timestamp...")
        test_timestamp = "2025-01-18 15:30:00"
        
        from run_live_forecast import LiveForecaster
        
        forecaster = LiveForecaster(model_path, output_dir='test_live_forecasts')
        
        forecaster.load_model()
        print("✓ Model loaded successfully")
        
        print(f"\n2. Testing forecast with timestamp: {test_timestamp}")
        results = forecaster.run_live_forecast(timestamp_override=test_timestamp)
        
        if results:
            print("✓ Live forecast completed successfully")
            print(f"✓ Current price: ${results['current_price']:.2f}")
            print(f"✓ 1-min forecast: ${results['point_forecasts']['horizon_1min']:.2f}")
            print(f"✓ Generated {len(results['probabilities'])} probability queries")
            return True
        else:
            print("✗ Live forecast failed")
            return False
        
    except Exception as e:
        print(f"✗ Live forecasting test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all live component tests"""
    print("TESTING LIVE BITCOIN TRADING COMPONENTS")
    print("=" * 60)
    
    data_test_passed = test_live_data_ingestion()
    
    forecast_test_passed = test_live_forecasting()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Live Data Ingestion: {'✓ PASSED' if data_test_passed else '✗ FAILED'}")
    print(f"Live Forecasting: {'✓ PASSED' if forecast_test_passed else '✗ FAILED'}")
    
    overall_success = data_test_passed and forecast_test_passed
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
