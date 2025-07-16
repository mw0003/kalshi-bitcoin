import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        """
        Initialize the feature engineering pipeline
        """
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_lagged_features(self, df, target_col='close', lags=[1, 2, 3, 5, 10, 15, 30, 60]):
        """
        Create lagged price features
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lags: List of lag periods
            
        Returns:
            pandas.DataFrame: DataFrame with lagged features
        """
        result_df = df.copy()
        
        for lag in lags:
            result_df[f'{target_col}_lag_{lag}'] = result_df[target_col].shift(lag)
            
        return result_df
    
    def create_rolling_features(self, df, target_col='close', windows=[5, 10, 15, 30, 60]):
        """
        Create rolling statistical features
        
        Args:
            df: Input DataFrame
            target_col: Column to calculate rolling stats for
            windows: List of window sizes
            
        Returns:
            pandas.DataFrame: DataFrame with rolling features
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f'{target_col}_ma_{window}'] = result_df[target_col].rolling(window=window).mean()
            
            result_df[f'{target_col}_std_{window}'] = result_df[target_col].rolling(window=window).std()
            
            result_df[f'{target_col}_min_{window}'] = result_df[target_col].rolling(window=window).min()
            result_df[f'{target_col}_max_{window}'] = result_df[target_col].rolling(window=window).max()
            
            result_df[f'{target_col}_rel_ma_{window}'] = result_df[target_col] / result_df[f'{target_col}_ma_{window}']
            
        return result_df
    
    def create_technical_indicators(self, df):
        """
        Create technical indicator features
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with technical indicators
        """
        result_df = df.copy()
        
        result_df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
        
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
        
        result_df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        result_df['volume_sma'] = df['volume'].rolling(window=20).mean()
        result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']
        
        result_df['price_change'] = df['close'].pct_change()
        result_df['price_change_abs'] = np.abs(result_df['price_change'])
        
        result_df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        result_df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        return result_df
    
    def create_time_features(self, df):
        """
        Create time-based features
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            pandas.DataFrame: DataFrame with time features
        """
        result_df = df.copy()
        
        result_df['hour'] = df.index.hour
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        
        result_df['day_of_week'] = df.index.dayofweek
        result_df['dow_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['dow_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        result_df['minute'] = df.index.minute
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df['minute'] / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df['minute'] / 60)
        
        return result_df
    
    def create_target_variables(self, df, target_col='close', horizons=range(1, 21)):
        """
        Create target variables for multi-horizon forecasting
        
        Args:
            df: Input DataFrame
            target_col: Column to forecast
            horizons: List of forecast horizons (minutes ahead)
            
        Returns:
            pandas.DataFrame: DataFrame with target variables
        """
        result_df = df.copy()
        
        for horizon in horizons:
            result_df[f'target_{horizon}'] = result_df[target_col].shift(-horizon)
            
        return result_df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering steps
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with all engineered features
        """
        print("Starting feature engineering...")
        
        df_features = self.create_lagged_features(df)
        df_features = self.create_rolling_features(df_features)
        df_features = self.create_technical_indicators(df_features)
        df_features = self.create_time_features(df_features)
        df_features = self.create_target_variables(df_features)
        
        print(f"Feature engineering complete. Shape: {df_features.shape}")
        
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        target_cols = [col for col in df_features.columns if col.startswith('target_')]
        self.feature_columns = [col for col in df_features.columns 
                               if col not in original_cols and col not in target_cols]
        
        print(f"Number of features created: {len(self.feature_columns)}")
        
        return df_features
    
    def prepare_for_modeling(self, df, test_size=0.2):
        """
        Prepare data for modeling by handling missing values and splitting
        
        Args:
            df: DataFrame with engineered features
            test_size: Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Preparing data for modeling...")
        
        df_clean = df.dropna()
        print(f"Data shape after removing NaN: {df_clean.shape}")
        
        X = df_clean[self.feature_columns]
        target_cols = [col for col in df_clean.columns if col.startswith('target_')]
        y = df_clean[target_cols]
        
        split_idx = int(len(df_clean) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_ingestion import BitcoinDataIngester
    
    ingester = BitcoinDataIngester()
    try:
        data = ingester.load_data()
    except:
        data = ingester.fetch_historical_data(days_back=7)
        ingester.save_data(data)
    
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(data)
    
    X_train, X_test, y_train, y_test = engineer.prepare_for_modeling(features_df)
    
    print("\nFeature columns:")
    for i, col in enumerate(engineer.feature_columns[:10]):  # Show first 10
        print(f"{i+1}. {col}")
    if len(engineer.feature_columns) > 10:
        print(f"... and {len(engineer.feature_columns) - 10} more features")
