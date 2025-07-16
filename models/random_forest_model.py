"""
Random Forest Model for Bitcoin Forecasting

This module implements a Random Forest-based multi-horizon forecasting model
using scikit-learn's RandomForestRegressor with MultiOutputRegressor.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from .base_model import BaseForecaster
from config import MODEL_CONFIG

class RandomForestForecaster(BaseForecaster):
    """
    Random Forest implementation for multi-horizon Bitcoin price forecasting
    """
    
    def __init__(self, model_name='random_forest', config=None):
        """
        Initialize Random Forest forecaster
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
        """
        if config is None:
            config = MODEL_CONFIG['random_forest']
        
        super().__init__(model_name, config)
        self.scaler = RobustScaler()
    
    def _create_model(self):
        """Create Random Forest model with MultiOutputRegressor"""
        base_model = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 200),
            max_depth=self.config.get('max_depth', 15),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        return MultiOutputRegressor(base_model)
    
    def fit(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets (multiple horizons)
        """
        self.logger.info(f"Training Random Forest model with {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        self.logger.info("Random Forest training completed")
    
    def predict(self, X_test):
        """
        Make predictions using the trained Random Forest model
        
        Args:
            X_test: Test features
            
        Returns:
            numpy.ndarray: Predictions for all horizons
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from Random Forest
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pandas.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if not importances:
                return None
            
            avg_importance = np.mean(importances, axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
