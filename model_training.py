import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHorizonForecaster:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the multi-horizon forecasting model
        
        Args:
            model_type: Type of model to use ('random_forest', 'linear', 'ridge')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _create_model(self):
        """Create the base model based on model_type"""
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'linear':
            base_model = LinearRegression()
        elif self.model_type == 'ridge':
            base_model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return MultiOutputRegressor(base_model)
    
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets (multiple horizons)
        """
        print(f"Training {self.model_type} model...")
        
        self.feature_names = X_train.columns.tolist()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        print("Model training completed!")
        
    def predict(self, X_test):
        """
        Make predictions
        
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
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        horizons = range(1, predictions.shape[1] + 1)
        metrics = {
            'horizon': [],
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        for i, horizon in enumerate(horizons):
            y_true = y_test.iloc[:, i]
            y_pred = predictions[:, i]
            
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
                
                metrics['horizon'].append(horizon)
                metrics['rmse'].append(rmse)
                metrics['mae'].append(mae)
                metrics['mape'].append(mape)
        
        return pd.DataFrame(metrics)
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance (for tree-based models)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pandas.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.model_type != 'random_forest':
            print("Feature importance only available for random forest models")
            return None
        
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")

class WalkForwardValidator:
    def __init__(self, model_class, model_params=None):
        """
        Initialize walk-forward validator
        
        Args:
            model_class: Model class to use
            model_params: Parameters for model initialization
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        
    def validate(self, X, y, n_splits=5, test_size=0.2):
        """
        Perform walk-forward validation
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of validation splits
            test_size: Size of each test set
            
        Returns:
            list: List of evaluation results for each split
        """
        print(f"Performing walk-forward validation with {n_splits} splits...")
        
        results = []
        total_samples = len(X)
        test_samples = int(total_samples * test_size)
        
        for i in range(n_splits):
            print(f"\nValidation split {i+1}/{n_splits}")
            
            test_end = total_samples - i * (test_samples // n_splits)
            test_start = test_end - test_samples
            train_end = test_start
            
            if train_end <= test_samples:  # Not enough training data
                break
            
            X_train = X.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_train = y.iloc[:train_end]
            y_test = y.iloc[test_start:test_end]
            
            print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            
            metrics = model.evaluate(X_test, y_test)
            metrics['split'] = i + 1
            results.append(metrics)
        
        return results

if __name__ == "__main__":
    from data_ingestion import BitcoinDataIngester
    from feature_engineering import FeatureEngineer
    
    ingester = BitcoinDataIngester()
    try:
        data = ingester.load_data()
    except:
        data = ingester.fetch_historical_data(days_back=7)
        ingester.save_data(data)
    
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(data)
    X_train, X_test, y_train, y_test = engineer.prepare_for_modeling(features_df)
    
    forecaster = MultiHorizonForecaster(model_type='random_forest')
    forecaster.fit(X_train, y_train)
    
    metrics = forecaster.evaluate(X_test, y_test)
    print("\nEvaluation Results:")
    print(metrics)
    
    importance = forecaster.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))
