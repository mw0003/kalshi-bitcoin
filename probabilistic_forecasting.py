"""
Probabilistic Forecasting Module for Bitcoin Price Prediction

This module provides probabilistic forecasting capabilities including:
- Distribution-based predictions with uncertainty quantification
- Probability queries for price thresholds and ranges
- Calibration analysis for reliability evaluation
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt


class ProbabilisticForecaster(ABC):
    """
    Mixin class for adding probabilistic forecasting capabilities to existing models
    """
    
    def __init__(self):
        self.quantile_models = {}
        self.bootstrap_predictions = None
        self.prediction_intervals = {}
    @abstractmethod
    def predict_quantiles(self, X_test, quantiles: List[float] = None) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for uncertainty estimation

        Args:
            X_test: Test features
            quantiles: List of quantiles to predict (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])

        Returns:
            Dictionary mapping quantiles to predictions
        """
        pass
    def predict_distribution(self, X_test, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the predictive distribution

        Args:
            X_test: Test features
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_test_samples, n_horizons, n_samples)
        """
        quantiles = np.linspace(0.01, 0.99, 99)
        quantile_preds = self.predict_quantiles(X_test, quantiles)

        n_test = X_test.shape[0]
        sample_quantiles = self.predict_quantiles(X_test.iloc[:1] if hasattr(X_test, 'iloc') else X_test[:1], [0.5])
        n_horizons = sample_quantiles[0.5].shape[1]
        samples = np.zeros((n_test, n_horizons, n_samples))

        for i in range(n_test):
            for h in range(n_horizons):
                q_values = np.array([quantile_preds[q][i, h] for q in quantiles])

                uniform_samples = np.random.uniform(0, 1, n_samples)
                samples[i, h, :] = np.interp(uniform_samples, quantiles, q_values)

        return samples
    def calculate_probabilities(self, X_test,
                                thresholds: Dict[str, Union[float, Tuple[float, float]]] = None,
                                n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate probabilities for various price events

        Args:
            X_test: Test features
            thresholds: Dictionary of threshold queries:
                - 'above_X': probability price >= X
                - 'below_X': probability price < X
                - 'between_X_Y': probability X <= price < Y
            n_samples: Number of samples for probability estimation

        Returns:
            Dictionary mapping query names to probability arrays (n_test_samples, n_horizons)
        """
        if thresholds is None:
            thresholds = {
                'above_65000': 65000,
                'below_60000': 60000,
                'between_63000_66000': (63000, 66000)
            }

        distribution_samples = self.predict_distribution(X_test, n_samples)
        n_test, n_horizons, _ = distribution_samples.shape

        probabilities = {}

        for query_name, threshold in thresholds.items():
            probs = np.zeros((n_test, n_horizons))

            if isinstance(threshold, tuple):
                low, high = threshold
                for i in range(n_test):
                    for h in range(n_horizons):
                        samples = distribution_samples[i, h, :]
                        probs[i, h] = np.mean((samples >= low) & (samples < high))
            else:
                if 'above' in query_name:
                    for i in range(n_test):
                        for h in range(n_horizons):
                            samples = distribution_samples[i, h, :]
                            probs[i, h] = np.mean(samples >= threshold)
                elif 'below' in query_name:
                    for i in range(n_test):
                        for h in range(n_horizons):
                            samples = distribution_samples[i, h, :]
                            probs[i, h] = np.mean(samples < threshold)

            probabilities[query_name] = probs

        return probabilities
    
    def get_prediction_intervals(self, X_test,
                                 confidence_levels: List[float] = None) -> Dict[float, Dict[str, np.ndarray]]:
        """
        Calculate prediction intervals at various confidence levels

        Args:
            X_test: Test features
            confidence_levels: List of confidence levels (e.g., [0.68, 0.95])

        Returns:
            Dictionary mapping confidence levels to {'lower': array, 'upper': array}
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95]

        intervals = {}

        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            quantile_preds = self.predict_quantiles(X_test, [lower_q, upper_q])

            intervals[conf_level] = {
                'lower': quantile_preds[lower_q],
                'upper': quantile_preds[upper_q]
            }

        return intervals


class BootstrapProbabilisticForecaster(ProbabilisticForecaster):
    """
    Bootstrap-based probabilistic forecasting for tree-based models
    """
    
    def __init__(self, base_forecaster, n_bootstrap: int = 100):
        super().__init__()
        self.base_forecaster = base_forecaster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = []
        
    def fit_bootstrap_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fit multiple bootstrap models for uncertainty estimation
        """
        self.bootstrap_models = []
        n_samples = len(X_train)
        
        for i in range(self.n_bootstrap):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train.iloc[bootstrap_indices]
            y_boot = y_train.iloc[bootstrap_indices]
            
            model = self.base_forecaster.__class__(
                self.base_forecaster.model_name, 
                self.base_forecaster.config
            )
            model.fit(X_boot, y_boot)
            self.bootstrap_models.append(model)
    
    def predict_quantiles(self, X_test: pd.DataFrame, quantiles: List[float] = None) -> Dict[float, np.ndarray]:
        """
        Predict quantiles using bootstrap ensemble
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        if not self.bootstrap_models:
            raise ValueError("Bootstrap models not fitted. Call fit_bootstrap_models first.")
        
        all_predictions = []
        for model in self.bootstrap_models:
            pred = model.predict(X_test)
            all_predictions.append(pred)
        
        all_predictions = np.stack(all_predictions, axis=0)
        
        quantile_predictions = {}
        for q in quantiles:
            quantile_predictions[q] = np.quantile(all_predictions, q, axis=0)
        
        return quantile_predictions


class CalibrationAnalyzer:
    """
    Analyze and visualize probability calibration for forecasting models
    """
    
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.calibration_data = {}
        
    def evaluate_calibration(self, y_true: np.ndarray, probabilities: Dict[str, np.ndarray], 
                           horizons: List[int] = None) -> Dict[str, Dict]:
        """
        Evaluate calibration for probability predictions
        
        Args:
            y_true: True values (n_samples, n_horizons)
            probabilities: Dictionary of probability predictions for different queries
            horizons: List of forecast horizons to analyze
            
        Returns:
            Dictionary with calibration metrics and data for each query and horizon
        """
        if horizons is None:
            horizons = list(range(y_true.shape[1]))
        
        calibration_results = {}
        
        for query_name, prob_preds in probabilities.items():
            calibration_results[query_name] = {}
            
            threshold = self._extract_threshold_from_query(query_name)
            
            for h_idx, horizon in enumerate(horizons):
                if h_idx >= y_true.shape[1] or h_idx >= prob_preds.shape[1]:
                    continue
                    
                y_h = y_true[:, h_idx]
                p_h = prob_preds[:, h_idx]
                
                if isinstance(threshold, tuple):
                    low, high = threshold
                    events = ((y_h >= low) & (y_h < high)).astype(int)
                elif 'above' in query_name:
                    events = (y_h >= threshold).astype(int)
                elif 'below' in query_name:
                    events = (y_h < threshold).astype(int)
                else:
                    continue
                
                cal_data = self._calculate_calibration_metrics(events, p_h)
                calibration_results[query_name][f'horizon_{horizon}'] = cal_data
        
        return calibration_results
    
    def _extract_threshold_from_query(self, query_name: str) -> Union[float, Tuple[float, float]]:
        """Extract threshold values from query name"""
        if 'between' in query_name:
            import re
            numbers = re.findall(r'\d+', query_name)
            if len(numbers) >= 2:
                return (float(numbers[0]), float(numbers[1]))
        else:
            import re
            numbers = re.findall(r'\d+', query_name)
            if numbers:
                return float(numbers[0])
        
        return 65000.0
    
    def _calculate_calibration_metrics(self, events: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate calibration metrics for binary events"""
        valid_mask = ~(np.isnan(events) | np.isnan(probabilities))
        events = events[valid_mask]
        probabilities = probabilities[valid_mask]
        
        if len(events) == 0:
            return {'brier_score': np.nan, 'log_loss': np.nan, 'calibration_error': np.nan}
        
        brier_score = np.mean((probabilities - events) ** 2)
        
        epsilon = 1e-15
        p_clipped = np.clip(probabilities, epsilon, 1 - epsilon)
        log_loss = -np.mean(events * np.log(p_clipped) + (1 - events) * np.log(1 - p_clipped))
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = events[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        calibration_curve = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            if in_bin.sum() > 0:
                bin_accuracy = events[in_bin].mean()
                bin_confidence = probabilities[in_bin].mean()
                bin_count = in_bin.sum()
                calibration_curve.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'confidence': bin_confidence,
                    'accuracy': bin_accuracy,
                    'count': bin_count
                })
        
        return {
            'brier_score': brier_score,
            'log_loss': log_loss,
            'calibration_error': ece,
            'calibration_curve': calibration_curve,
            'n_samples': len(events)
        }
    
    def plot_calibration_curve(self, calibration_data: Dict, save_path: str = None):
        """
        Plot calibration curves for different queries and horizons
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for query_name, query_data in calibration_data.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            for horizon_key, horizon_data in query_data.items():
                if 'calibration_curve' not in horizon_data:
                    continue
                    
                curve_data = horizon_data['calibration_curve']
                if not curve_data:
                    continue
                
                confidences = [d['confidence'] for d in curve_data]
                accuracies = [d['accuracy'] for d in curve_data]
                
                horizon_num = horizon_key.replace('horizon_', '')
                ax.plot(confidences, accuracies, 'o-', label=f'Horizon {horizon_num}', alpha=0.7)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve: {query_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_calibration_metrics(self, calibration_data: Dict, filepath: str):
        """Save calibration metrics to CSV"""
        rows = []
        
        for query_name, query_data in calibration_data.items():
            for horizon_key, horizon_data in query_data.items():
                row = {
                    'query': query_name,
                    'horizon': horizon_key.replace('horizon_', ''),
                    'brier_score': horizon_data.get('brier_score', np.nan),
                    'log_loss': horizon_data.get('log_loss', np.nan),
                    'calibration_error': horizon_data.get('calibration_error', np.nan),
                    'n_samples': horizon_data.get('n_samples', 0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return df
