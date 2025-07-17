"""
Configuration file for Bitcoin forecasting pipeline
"""

import os
from datetime import datetime, timedelta

DATA_CONFIG = {
    'exchange': 'coinbase',
    'symbol': 'BTC/USD',
    'timeframe': '1m',
    'years_back': 2,
    'data_file': 'bitcoin_2year_data.csv'
}

MODEL_CONFIG = {
    'forecast_horizons': list(range(1, 21)),  # 1-20 minutes
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 15,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    },
    'lstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    }
}

VALIDATION_CONFIG = {
    'train_days': 7,
    'test_days': 1,
    'min_train_samples': 1000,
    'step_days': 1  # How many days to step forward each split
}

PROBABILISTIC_CONFIG = {
    'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
    'confidence_levels': [0.68, 0.95],
    'n_bootstrap': 100,
    'n_distribution_samples': 1000,
    'calibration_bins': 20,
    'default_thresholds': {
        'above_2pct': 1.02,
        'below_2pct': 0.98,
        'between_1pct': (0.99, 1.01)
    }
}

FEATURE_CONFIG = {
    'lag_periods': [1, 2, 3, 5, 10, 15, 30, 60],
    'rolling_windows': [5, 10, 15, 30, 60, 120, 240],
    'technical_indicators': True,
    'time_features': True
}

OUTPUT_CONFIG = {
    'results_dir': 'results',
    'models_dir': 'trained_models',
    'plots_dir': 'plots',
    'metrics_file_template': '{model_name}_metrics.csv',
    'model_file_template': '{model_name}_model.joblib'
}

for dir_name in OUTPUT_CONFIG.values():
    if dir_name.endswith('_dir'):
        os.makedirs(dir_name, exist_ok=True)
