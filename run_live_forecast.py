#!/usr/bin/env python3
"""
Live Bitcoin Forecasting Script

This script fetches live Bitcoin data and generates real-time probabilistic forecasts
for trading applications. Designed to run as a cron job every minute.

Usage:
    python run_live_forecast.py --model_path trained_models/random_forest_model.joblib
    python run_live_forecast.py --model_path trained_models/random_forest_model.joblib --timestamp "2025-01-18 18:45:00"
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import logging

from data_ingestion import BitcoinDataIngester
from enhanced_feature_engineering import EnhancedFeatureEngineer
from models.random_forest_model import RandomForestForecaster
from models.lightgbm_model import LightGBMForecaster
from models.lstm_model import LSTMForecaster
from utils import setup_logging, ProgressTimer
from config import LIVE_CONFIG, PROBABILISTIC_CONFIG


class LiveForecaster:
    """Live Bitcoin forecasting system for real-time trading"""
    
    def __init__(self, model_path, output_dir=None):
        """
        Initialize live forecaster
        
        Args:
            model_path: Path to trained model file
            output_dir: Directory to save live forecasts (optional)
        """
        self.model_path = model_path
        self.output_dir = output_dir or LIVE_CONFIG['forecast_output_dir']
        self.logger = setup_logging('live_forecast.log')
        
        self.data_ingester = BitcoinDataIngester()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        if LIVE_CONFIG['save_live_data']:
            os.makedirs(LIVE_CONFIG['live_data_dir'], exist_ok=True)
    
    def load_model(self):
        """Load the trained forecasting model"""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if 'random_forest' in self.model_path or 'demo' in self.model_path:
            self.model = RandomForestForecaster()
        elif 'lightgbm' in self.model_path:
            self.model = LightGBMForecaster()
        elif 'lstm' in self.model_path:
            self.model = LSTMForecaster()
        else:
            self.logger.warning("Cannot determine model type, defaulting to RandomForest")
            self.model = RandomForestForecaster()
        
        self.model.load_model(self.model_path)
        self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")
        
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            self.logger.info(f"Model expects {len(self.model.feature_names)} features")
        
        return self.model
    
    def fetch_live_data(self, timestamp_override=None):
        """
        Fetch recent Bitcoin data for forecasting
        
        Args:
            timestamp_override: Optional timestamp to simulate historical live data
            
        Returns:
            pandas.DataFrame: Recent Bitcoin data
        """
        if timestamp_override:
            self.logger.info(f"Using timestamp override: {timestamp_override}")
            end_time = pd.to_datetime(timestamp_override)
            start_time = end_time - timedelta(minutes=LIVE_CONFIG['data_fetch_minutes'])
            
            live_data = self.data_ingester._fetch_coinbase_chunk(
                start_time.isoformat() + 'Z',
                end_time.isoformat() + 'Z',
                60  # 1 minute granularity
            )
        else:
            live_data = self.data_ingester.get_live_features_data(
                minutes_back=LIVE_CONFIG['data_fetch_minutes']
            )
        
        if live_data.empty:
            raise ValueError("No live data available")
        
        self.logger.info(f"Fetched {len(live_data)} minutes of data")
        self.logger.info(f"Latest price: ${live_data['close'].iloc[-1]:.2f}")
        
        if LIVE_CONFIG['save_live_data']:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            live_data_file = os.path.join(
                LIVE_CONFIG['live_data_dir'], 
                f'live_data_{timestamp_str}.csv'
            )
            live_data.to_csv(live_data_file)
            self.logger.info(f"Live data saved to: {live_data_file}")
        
        return live_data
    
    def prepare_features(self, live_data):
        """
        Engineer features from live data for model input
        
        Args:
            live_data: Raw Bitcoin OHLCV data
            
        Returns:
            pandas.DataFrame: Features ready for model inference
        """
        self.logger.info("Engineering features from live data...")
        
        with ProgressTimer("Feature engineering"):
            features_df = self._create_simple_features(live_data)
        
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            available_features = [f for f in self.model.feature_names if f in features_df.columns]
            missing_features = [f for f in self.model.feature_names if f not in features_df.columns]
            
            if missing_features:
                self.logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                raise ValueError(f"Model requires features that are not available: {missing_features}")
            
            X_live = features_df[available_features].dropna()
        else:
            feature_cols = [col for col in features_df.columns 
                           if not col.startswith('target_') and 
                           col not in ['open', 'high', 'low', 'close', 'volume']]
            X_live = features_df[feature_cols].dropna()
        
        if X_live.empty:
            raise ValueError("No valid features available after processing")
        
        self.logger.info(f"Features prepared: {X_live.shape}")
        return X_live
    
    def _create_simple_features(self, df):
        """
        Create simple features that match the trained model expectations
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with simple features
        """
        features_df = df.copy()
        
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
    
    def generate_forecasts(self, X_live, current_price):
        """
        Generate probabilistic forecasts from live features
        
        Args:
            X_live: Feature matrix for inference
            current_price: Current Bitcoin price for probability calculations
            
        Returns:
            dict: Complete forecast results
        """
        self.logger.info("Generating live forecasts...")
        
        X_latest = X_live.tail(1)
        
        point_predictions = self.model.predict(X_latest)
        
        quantile_predictions = self.model.predict_quantiles(X_latest)
        
        try:
            prediction_intervals = self.model.get_prediction_intervals(X_latest)
        except Exception as e:
            self.logger.warning(f"Failed to generate prediction intervals: {e}")
            prediction_intervals = {}
        
        thresholds = self._generate_price_thresholds(current_price)
        probabilities = self.model.calculate_probabilities(X_latest, thresholds)
        
        forecast_results = {
            'timestamp': X_latest.index[0].isoformat(),
            'current_price': current_price,
            'point_forecasts': {
                f'horizon_{i+1}min': float(point_predictions[0, i])
                for i in range(point_predictions.shape[1])
            },
            'quantile_forecasts': {
                f'q{int(q*100)}': {
                    f'horizon_{i+1}min': float(q_preds[0, i])
                    for i in range(q_preds.shape[1])
                }
                for q, q_preds in quantile_predictions.items()
            },
            'prediction_intervals': {
                f'{int(level*100)}pct': {
                    f'horizon_{i+1}min': {
                        'lower': float(intervals['lower'][0, i]),
                        'upper': float(intervals['upper'][0, i])
                    }
                    for i in range(intervals['lower'].shape[1])
                }
                for level, intervals in prediction_intervals.items()
            } if prediction_intervals else {},
            'probabilities': {
                query: {
                    f'horizon_{i+1}min': float(prob_array[0, i])
                    for i in range(prob_array.shape[1])
                }
                for query, prob_array in probabilities.items()
            }
        }
        
        self.logger.info("Live forecasts generated successfully")
        return forecast_results
    
    def _generate_price_thresholds(self, current_price):
        """Generate relevant price thresholds for probability queries"""
        thresholds = {}
        
        for pct in [0.5, 1.0, 2.0, 5.0]:
            thresholds[f'above_{pct}pct'] = current_price * (1 + pct/100)
            thresholds[f'below_{pct}pct'] = current_price * (1 - pct/100)
        
        base_price = round(current_price / 100) * 100
        for delta in [500, 1000, 2000]:
            thresholds[f'above_{base_price + delta}'] = base_price + delta
            thresholds[f'below_{base_price - delta}'] = base_price - delta
        
        thresholds[f'between_{int(current_price * 0.995)}_{int(current_price * 1.005)}'] = (
            current_price * 0.995, current_price * 1.005
        )
        
        return thresholds
    
    def save_forecasts(self, forecast_results):
        """Save forecast results to file"""
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if LIVE_CONFIG['output_format'] == 'json':
            output_file = os.path.join(self.output_dir, f'live_forecast_{timestamp_str}.json')
            with open(output_file, 'w') as f:
                json.dump(forecast_results, f, indent=2)
        else:
            output_file = os.path.join(self.output_dir, f'live_forecast_{timestamp_str}.csv')
            flat_data = self._flatten_forecast_results(forecast_results)
            pd.DataFrame([flat_data]).to_csv(output_file, index=False)
        
        self.logger.info(f"Forecast results saved to: {output_file}")
        return output_file
    
    def _flatten_forecast_results(self, results):
        """Flatten nested forecast results for CSV output"""
        flat = {
            'timestamp': results['timestamp'],
            'current_price': results['current_price']
        }
        
        for horizon, value in results['point_forecasts'].items():
            flat[f'point_{horizon}'] = value
        
        for query, horizons in results['probabilities'].items():
            for horizon, prob in horizons.items():
                flat[f'prob_{query}_{horizon}'] = prob
        
        return flat
    
    def run_live_forecast(self, timestamp_override=None):
        """
        Run complete live forecasting pipeline
        
        Args:
            timestamp_override: Optional timestamp for historical simulation
            
        Returns:
            dict: Forecast results
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING LIVE BITCOIN FORECAST")
            self.logger.info("=" * 60)
            
            self.load_model()
            
            live_data = self.fetch_live_data(timestamp_override)
            current_price = live_data['close'].iloc[-1]
            
            X_live = self.prepare_features(live_data)
            
            forecast_results = self.generate_forecasts(X_live, current_price)
            
            output_file = self.save_forecasts(forecast_results)
            
            self.logger.info("=" * 60)
            self.logger.info("LIVE FORECAST COMPLETED SUCCESSFULLY")
            self.logger.info(f"Current Price: ${current_price:.2f}")
            self.logger.info(f"Results saved to: {output_file}")
            self.logger.info("=" * 60)
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Live forecast failed: {str(e)}")
            raise


def main():
    """Main entry point for live forecasting script"""
    parser = argparse.ArgumentParser(description='Run live Bitcoin forecasting')
    parser.add_argument('--model_path', required=True, 
                       help='Path to trained model file')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for forecasts')
    parser.add_argument('--timestamp', default=None,
                       help='Override timestamp for historical simulation (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    logger = setup_logging('live_forecast.log')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        forecaster = LiveForecaster(args.model_path, args.output_dir)
        
        results = forecaster.run_live_forecast(args.timestamp)
        
        print(f"SUCCESS: Live forecast completed at {datetime.now()}")
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"1-min forecast: ${results['point_forecasts']['horizon_1min']:.2f}")
        print(f"5-min forecast: ${results['point_forecasts']['horizon_5min']:.2f}")
        
        for query, horizons in results['probabilities'].items():
            if 'horizon_1min' in horizons:
                print(f"P({query}|1min): {horizons['horizon_1min']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Live forecasting failed: {str(e)}")
        print(f"ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
