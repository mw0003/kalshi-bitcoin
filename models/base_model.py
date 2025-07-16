"""
Base Model Class for Bitcoin Forecasting

This module defines the abstract base class that all forecasting models
must implement to ensure consistent interface across different model types.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from utils import calculate_metrics, create_time_based_splits, setup_logging
from config import VALIDATION_CONFIG, OUTPUT_CONFIG

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models
    """
    
    def __init__(self, model_name, config=None):
        """
        Initialize the base forecaster
        
        Args:
            model_name: Name of the model (e.g., 'random_forest', 'lstm', 'lightgbm')
            config: Model-specific configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.logger = setup_logging(f'{model_name}.log')
        
    @abstractmethod
    def _create_model(self):
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets (multiple horizons)
        """
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test: Test features
            
        Returns:
            numpy.ndarray: Predictions for all horizons
        """
        pass
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            pandas.DataFrame: Evaluation metrics per horizon
        """
        predictions = self.predict(X_test)
        
        if predictions is None:
            return pd.DataFrame()
        
        horizons = range(1, predictions.shape[1] + 1)
        metrics_list = []
        
        for i, horizon in enumerate(horizons):
            y_true = y_test.iloc[:, i].values
            y_pred = predictions[:, i]
            
            metrics = calculate_metrics(y_true, y_pred)
            metrics['horizon'] = horizon
            metrics['model'] = self.model_name
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def rolling_validation(self, df, feature_cols, target_cols):
        """
        Perform rolling time-based validation
        
        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            target_cols: List of target column names
            
        Returns:
            list: List of evaluation results for each split
        """
        self.logger.info(f"Starting rolling validation for {self.model_name}")
        
        splits = create_time_based_splits(
            df,
            train_days=VALIDATION_CONFIG['train_days'],
            test_days=VALIDATION_CONFIG['test_days'],
            step_days=VALIDATION_CONFIG['step_days'],
            min_train_samples=VALIDATION_CONFIG['min_train_samples']
        )
        
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            self.logger.info(f"Validation split {i+1}/{len(splits)}")
            self.logger.info(f"Train: {train_start} to {train_end}")
            self.logger.info(f"Test: {test_start} to {test_end}")
            
            train_mask = (df.index >= train_start) & (df.index < train_end)
            test_mask = (df.index >= test_start) & (df.index < test_end)
            
            X_train = df.loc[train_mask, feature_cols]
            X_test = df.loc[test_mask, feature_cols]
            y_train = df.loc[train_mask, target_cols]
            y_test = df.loc[test_mask, target_cols]
            
            if len(X_train) < VALIDATION_CONFIG['min_train_samples']:
                self.logger.warning(f"Insufficient training samples: {len(X_train)}")
                continue
            
            model_instance = self.__class__(self.model_name, self.config)
            model_instance.fit(X_train, y_train)
            
            metrics = model_instance.evaluate(X_test, y_test)
            metrics['split'] = i + 1
            metrics['train_start'] = train_start
            metrics['train_end'] = train_end
            metrics['test_start'] = test_start
            metrics['test_end'] = test_end
            metrics['train_samples'] = len(X_train)
            metrics['test_samples'] = len(X_test)
            
            results.append(metrics)
        
        return results
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        if filepath is None:
            filepath = os.path.join(
                OUTPUT_CONFIG['models_dir'],
                OUTPUT_CONFIG['model_file_template'].format(model_name=self.model_name)
            )
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance (if supported by the model)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pandas.DataFrame: Feature importance scores or None
        """
        return None
