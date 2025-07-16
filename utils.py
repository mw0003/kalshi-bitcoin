"""
Utility functions for Bitcoin forecasting pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import time

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

def create_time_based_splits(df, train_days=7, test_days=1, step_days=1, min_train_samples=1000):
    """
    Create rolling time-based train/test splits
    
    Args:
        df: DataFrame with datetime index
        train_days: Number of days for training
        test_days: Number of days for testing
        step_days: Number of days to step forward each split
        min_train_samples: Minimum number of training samples required
        
    Returns:
        list: List of (train_start, train_end, test_start, test_end) tuples
    """
    splits = []
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_date = start_date + timedelta(days=train_days)
    
    while current_date + timedelta(days=test_days) <= end_date:
        train_start = current_date - timedelta(days=train_days)
        train_end = current_date
        test_start = current_date
        test_end = current_date + timedelta(days=test_days)
        
        train_mask = (df.index >= train_start) & (df.index < train_end)
        if train_mask.sum() >= min_train_samples:
            splits.append((train_start, train_end, test_start, test_end))
        
        current_date += timedelta(days=step_days)
    
    return splits

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
    
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    mape_mask = y_true_clean != 0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
    else:
        mape = np.nan
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape}

def save_metrics_to_csv(metrics_list, filepath):
    """
    Save metrics to CSV file
    
    Args:
        metrics_list: List of metric dictionaries
        filepath: Path to save CSV file
    """
    df = pd.DataFrame(metrics_list)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")

def load_metrics_from_csv(filepath):
    """
    Load metrics from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pandas.DataFrame: Loaded metrics
    """
    return pd.read_csv(filepath)

class ProgressTimer:
    """Context manager for timing operations with progress display"""
    
    def __init__(self, description):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        print(f"Starting: {self.description}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"Completed: {self.description} (took {elapsed:.2f} seconds)")

def estimate_time_remaining(current_step, total_steps, elapsed_time):
    """
    Estimate remaining time for a process
    
    Args:
        current_step: Current step number
        total_steps: Total number of steps
        elapsed_time: Time elapsed so far
        
    Returns:
        str: Formatted time remaining estimate
    """
    if current_step == 0:
        return "Calculating..."
    
    avg_time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_time = avg_time_per_step * remaining_steps
    
    if remaining_time < 60:
        return f"{remaining_time:.0f}s"
    elif remaining_time < 3600:
        return f"{remaining_time/60:.1f}m"
    else:
        return f"{remaining_time/3600:.1f}h"
