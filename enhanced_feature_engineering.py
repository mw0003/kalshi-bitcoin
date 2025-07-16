"""
Enhanced Feature Engineering for Bitcoin Forecasting

This module provides advanced feature engineering capabilities for the Bitcoin
forecasting pipeline, including technical indicators, time-based features,
and custom transformations.
"""

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import logging
from utils import setup_logging, ProgressTimer
from config import FEATURE_CONFIG

class EnhancedFeatureEngineer:
    def __init__(self, config=None):
        """
        Initialize the enhanced feature engineering pipeline
        
        Args:
            config: Feature configuration dictionary
        """
        self.config = config or FEATURE_CONFIG
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_columns = []
        self.logger = setup_logging()
        
    def create_lagged_features(self, df, target_col='close'):
        """
        Create lagged price features with progress tracking
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            
        Returns:
            pandas.DataFrame: DataFrame with lagged features
        """
        result_df = df.copy()
        lags = self.config['lag_periods']
        
        self.logger.info(f"Creating {len(lags)} lagged features for {target_col}")
        
        for lag in tqdm(lags, desc="Creating lag features"):
            result_df[f'{target_col}_lag_{lag}'] = result_df[target_col].shift(lag)
            
        return result_df
    
    def create_rolling_features(self, df, target_col='close'):
        """
        Create comprehensive rolling statistical features
        
        Args:
            df: Input DataFrame
            target_col: Column to calculate rolling stats for
            
        Returns:
            pandas.DataFrame: DataFrame with rolling features
        """
        result_df = df.copy()
        windows = self.config['rolling_windows']
        
        self.logger.info(f"Creating rolling features for {len(windows)} windows")
        
        for window in tqdm(windows, desc="Creating rolling features"):
            result_df[f'{target_col}_ma_{window}'] = result_df[target_col].rolling(window=window).mean()
            result_df[f'{target_col}_std_{window}'] = result_df[target_col].rolling(window=window).std()
            result_df[f'{target_col}_min_{window}'] = result_df[target_col].rolling(window=window).min()
            result_df[f'{target_col}_max_{window}'] = result_df[target_col].rolling(window=window).max()
            
            result_df[f'{target_col}_median_{window}'] = result_df[target_col].rolling(window=window).median()
            result_df[f'{target_col}_skew_{window}'] = result_df[target_col].rolling(window=window).skew()
            result_df[f'{target_col}_kurt_{window}'] = result_df[target_col].rolling(window=window).kurt()
            
            result_df[f'{target_col}_rel_ma_{window}'] = result_df[target_col] / result_df[f'{target_col}_ma_{window}']
            result_df[f'{target_col}_zscore_{window}'] = (
                (result_df[target_col] - result_df[f'{target_col}_ma_{window}']) / 
                result_df[f'{target_col}_std_{window}']
            )
            
            result_df[f'{target_col}_range_{window}'] = (
                result_df[f'{target_col}_max_{window}'] - result_df[f'{target_col}_min_{window}']
            )
            
        return result_df
    
    def create_technical_indicators(self, df):
        """
        Create comprehensive technical indicator features
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with technical indicators
        """
        if not self.config['technical_indicators']:
            return df
            
        result_df = df.copy()
        
        self.logger.info("Creating technical indicators")
        
        with ProgressTimer("Technical indicators creation"):
            result_df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
            result_df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            result_df['rsi_21'] = ta.momentum.RSIIndicator(close=df['close'], window=21).rsi()
            
            macd = ta.trend.MACD(close=df['close'])
            result_df['macd'] = macd.macd()
            result_df['macd_signal'] = macd.macd_signal()
            result_df['macd_diff'] = macd.macd_diff()
            
            bollinger = ta.volatility.BollingerBands(close=df['close'])
            result_df['bb_high'] = bollinger.bollinger_hband()
            result_df['bb_low'] = bollinger.bollinger_lband()
            result_df['bb_mid'] = bollinger.bollinger_mavg()
            result_df['bb_width'] = result_df['bb_high'] - result_df['bb_low']
            result_df['bb_position'] = (df['close'] - result_df['bb_low']) / result_df['bb_width']
            
            stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            result_df['stoch_k'] = stoch.stoch()
            result_df['stoch_d'] = stoch.stoch_signal()
            
            result_df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close']
            ).average_true_range()
            
            result_df['williams_r'] = ta.momentum.WilliamsRIndicator(
                high=df['high'], low=df['low'], close=df['close']
            ).williams_r()
            
            result_df['cci'] = ta.trend.CCIIndicator(
                high=df['high'], low=df['low'], close=df['close']
            ).cci()
            
            result_df['volume_sma'] = df['volume'].rolling(window=20).mean()
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']
            
            result_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'], volume=df['volume']
            ).on_balance_volume()
            
            result_df['price_change'] = df['close'].pct_change()
            result_df['price_change_abs'] = np.abs(result_df['price_change'])
            result_df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            result_df['hl_spread'] = (df['high'] - df['low']) / df['close']
            result_df['oc_spread'] = (df['close'] - df['open']) / df['open']
            result_df['ho_spread'] = (df['high'] - df['open']) / df['open']
            result_df['lo_spread'] = (df['low'] - df['open']) / df['open']
            
        return result_df
    
    def create_time_features(self, df):
        """
        Create comprehensive time-based features
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            pandas.DataFrame: DataFrame with time features
        """
        if not self.config['time_features']:
            return df
            
        result_df = df.copy()
        
        self.logger.info("Creating time-based features")
        
        result_df['hour'] = df.index.hour
        result_df['day_of_week'] = df.index.dayofweek
        result_df['day_of_month'] = df.index.day
        result_df['month'] = df.index.month
        result_df['quarter'] = df.index.quarter
        result_df['minute'] = df.index.minute
        
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['dow_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['dow_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df['minute'] / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df['minute'] / 60)
        
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
        result_df['is_market_hours'] = ((result_df['hour'] >= 9) & (result_df['hour'] <= 16)).astype(int)
        result_df['is_overnight'] = ((result_df['hour'] >= 22) | (result_df['hour'] <= 6)).astype(int)
        
        return result_df
    
    def create_target_variables(self, df, target_col='close', horizons=None):
        """
        Create target variables for multi-horizon forecasting
        
        Args:
            df: Input DataFrame
            target_col: Column to forecast
            horizons: List of forecast horizons (minutes ahead)
            
        Returns:
            pandas.DataFrame: DataFrame with target variables
        """
        if horizons is None:
            from config import MODEL_CONFIG
            horizons = MODEL_CONFIG['forecast_horizons']
            
        result_df = df.copy()
        
        self.logger.info(f"Creating {len(horizons)} target variables")
        
        for horizon in tqdm(horizons, desc="Creating targets"):
            result_df[f'target_{horizon}'] = result_df[target_col].shift(-horizon)
            
        return result_df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering steps with progress tracking
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with all engineered features
        """
        self.logger.info("Starting comprehensive feature engineering...")
        
        with ProgressTimer("Complete feature engineering process"):
            df_features = self.create_lagged_features(df)
            df_features = self.create_rolling_features(df_features)
            df_features = self.create_technical_indicators(df_features)
            df_features = self.create_time_features(df_features)
            df_features = self.create_target_variables(df_features)
        
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        target_cols = [col for col in df_features.columns if col.startswith('target_')]
        self.feature_columns = [col for col in df_features.columns 
                               if col not in original_cols and col not in target_cols]
        
        self.logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        self.logger.info(f"Number of features created: {len(self.feature_columns)}")
        
        return df_features
    
    def prepare_for_modeling(self, df, split_config):
        """
        Prepare data for time-based modeling splits
        
        Args:
            df: DataFrame with engineered features
            split_config: Configuration for time-based splits
            
        Returns:
            pandas.DataFrame: Clean DataFrame ready for modeling
        """
        self.logger.info("Preparing data for modeling...")
        
        df_clean = df.dropna()
        self.logger.info(f"Data shape after removing NaN: {df_clean.shape}")
        
        self.logger.info(f"Number of features: {len(self.feature_columns)}")
        self.logger.info(f"Sample features: {self.feature_columns[:10]}")
        
        return df_clean
