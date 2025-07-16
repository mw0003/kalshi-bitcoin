"""
LSTM Model for Bitcoin Forecasting using Darts

This module implements an LSTM-based multi-horizon forecasting model
using the Darts library for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .base_model import BaseForecaster
from config import MODEL_CONFIG

try:
    from darts import TimeSeries
    from darts.models import RNNModel
    from darts.dataprocessing.transformers import Scaler
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

class LSTMForecaster(BaseForecaster):
    """
    LSTM implementation for multi-horizon Bitcoin price forecasting using Darts
    """
    
    def __init__(self, model_name='lstm', config=None):
        """
        Initialize LSTM forecaster
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
        """
        if not DARTS_AVAILABLE:
            raise ImportError("Darts library is required for LSTM model. Install with: pip install darts")
        
        if config is None:
            config = MODEL_CONFIG['lstm']
        
        super().__init__(model_name, config)
        self.scaler = Scaler()
        self.target_scaler = RobustScaler()
        self.models = {}  # Store separate models for each horizon
    
    def _create_model(self):
        """Create LSTM model using Darts RNNModel"""
        model = RNNModel(
            model='LSTM',
            hidden_dim=self.config.get('hidden_size', 64),
            n_rnn_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2),
            batch_size=self.config.get('batch_size', 32),
            n_epochs=self.config.get('epochs', 50),
            optimizer_kwargs={'lr': self.config.get('learning_rate', 0.001)},
            model_name=f"lstm_{self.model_name}",
            log_tensorboard=False,
            random_state=self.config.get('random_state', 42),
            force_reset=True,
            save_checkpoints=False
        )
        
        return model
    
    def _prepare_time_series(self, df, target_col):
        """
        Convert DataFrame to Darts TimeSeries format
        
        Args:
            df: Input DataFrame with datetime index
            target_col: Target column name
            
        Returns:
            TimeSeries: Darts TimeSeries object
        """
        return TimeSeries.from_dataframe(
            df[[target_col]], 
            time_col=None,
            value_cols=[target_col],
            freq='T'  # 1-minute frequency
        )
    
    def fit(self, X_train, y_train):
        """
        Train the LSTM model for each horizon
        
        Args:
            X_train: Training features
            y_train: Training targets (multiple horizons)
        """
        self.logger.info(f"Training LSTM model with {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        
        feature_series = TimeSeries.from_dataframe(
            X_train.reset_index(),
            time_col=X_train.index.name or 'timestamp',
            value_cols=self.feature_names,
            freq='T'
        )
        
        feature_series_scaled = self.scaler.fit_transform(feature_series)
        
        horizons = range(1, y_train.shape[1] + 1)
        
        for i, horizon in enumerate(horizons):
            self.logger.info(f"Training LSTM for horizon {horizon}")
            
            target_data = pd.DataFrame({
                'target': y_train.iloc[:, i]
            }, index=X_train.index)
            
            target_series = self._prepare_time_series(target_data, 'target')
            
            model = self._create_model()
            
            try:
                model.fit(
                    series=target_series,
                    past_covariates=feature_series_scaled,
                    verbose=False
                )
                self.models[horizon] = model
                
            except Exception as e:
                self.logger.error(f"Failed to train LSTM for horizon {horizon}: {e}")
                continue
        
        self.is_fitted = len(self.models) > 0
        
        if self.is_fitted:
            self.logger.info(f"LSTM training completed for {len(self.models)} horizons")
        else:
            raise RuntimeError("Failed to train LSTM for any horizon")
    
    def predict(self, X_test):
        """
        Make predictions using the trained LSTM models
        
        Args:
            X_test: Test features
            
        Returns:
            numpy.ndarray: Predictions for all horizons
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        feature_series = TimeSeries.from_dataframe(
            X_test.reset_index(),
            time_col=X_test.index.name or 'timestamp',
            value_cols=self.feature_names,
            freq='T'
        )
        
        feature_series_scaled = self.scaler.transform(feature_series)
        
        predictions = []
        max_horizon = max(self.models.keys())
        
        for horizon in range(1, max_horizon + 1):
            if horizon in self.models:
                try:
                    pred_series = self.models[horizon].predict(
                        n=1,
                        past_covariates=feature_series_scaled,
                        num_samples=1
                    )
                    
                    pred_values = pred_series.values().flatten()
                    
                    horizon_preds = np.full(len(X_test), pred_values[0] if len(pred_values) > 0 else 0.0)
                    
                except Exception as e:
                    self.logger.error(f"Prediction failed for horizon {horizon}: {e}")
                    horizon_preds = np.zeros(len(X_test))
            else:
                horizon_preds = np.zeros(len(X_test))
            
            predictions.append(horizon_preds)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self, top_n=20):
        """
        LSTM models don't provide direct feature importance
        
        Returns:
            None: LSTM doesn't support feature importance
        """
        self.logger.info("Feature importance not available for LSTM models")
        return None
