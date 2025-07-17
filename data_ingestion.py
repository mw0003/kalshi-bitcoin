import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
import os
import requests
from utils import setup_logging, ProgressTimer

class BitcoinDataIngester:
    def __init__(self, exchange_name='coinbase', symbol='BTC/USD'):
        """
        Initialize the Bitcoin data ingester
        
        Args:
            exchange_name: Name of the exchange (default: coinbase)
            symbol: Trading pair symbol (default: BTC/USD)
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.logger = setup_logging()
        
    def fetch_historical_data(self, timeframe='1m', limit=1000, years_back=2):
        """
        Fetch historical OHLCV data using Coinbase REST API with pagination
        
        Args:
            timeframe: Timeframe for the data (default: 1m for 1-minute)
            limit: Not used for REST API (kept for compatibility)
            years_back: Number of years to go back from current time
            
        Returns:
            pandas.DataFrame: Historical OHLCV data
        """
        self.logger.info(f"Fetching {years_back} years of historical data from Coinbase REST API")
        return self.fetch_coinbase_historical_data(timeframe, years_back)
    
    def fetch_historical_data_chunked(self, timeframe='1m', years_back=2, chunk_days=30):
        """
        Fetch historical data in chunks to avoid API timeouts and rate limits
        
        Args:
            timeframe: Timeframe for the data (default: 1m)
            years_back: Number of years to go back
            chunk_days: Size of each chunk in days (default: 30 days)
            
        Returns:
            pandas.DataFrame: Combined historical OHLCV data
        """
        self.logger.info(f"Fetching {years_back} years of data in {chunk_days}-day chunks")
        
        all_chunks = []
        total_days = int(years_back * 365)
        total_chunks = (total_days + chunk_days - 1) // chunk_days  # Ceiling division
        
        with ProgressTimer(f"Fetching {years_back} years of {timeframe} data for {self.symbol}"):
            for i in range(total_chunks):
                start_days_back = i * chunk_days
                end_days_back = min((i + 1) * chunk_days, total_days)
                
                self.logger.info(f"Fetching chunk {i+1}/{total_chunks}: {end_days_back} to {start_days_back} days ago")
                
                end_time = datetime.now() - timedelta(days=start_days_back)
                start_time = datetime.now() - timedelta(days=end_days_back)
                
                self.logger.info(f"Time range: {start_time} to {end_time}")
                
                chunk_data = self._fetch_chunk(start_time, end_time, timeframe)
                if not chunk_data.empty:
                    all_chunks.append(chunk_data)
                    self.logger.info(f"Chunk {i+1} completed: {len(chunk_data)} rows, date range: {chunk_data.index.min()} to {chunk_data.index.max()}")
                else:
                    self.logger.warning(f"Chunk {i+1} returned no data")
                
                # Wait between chunks to respect rate limits
                time.sleep(2)
        
        if not all_chunks:
            raise ValueError("No data was fetched from any chunk")
        
        self.logger.info("Combining all chunks...")
        combined_df = pd.concat(all_chunks).sort_index()
        
        initial_rows = len(combined_df)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        duplicates_removed = initial_rows - len(combined_df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        self.logger.info(f"Combined dataset shape: {combined_df.shape}")
        self.logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        
        expected_minutes = int(years_back * 365 * 24 * 60)
        actual_minutes = len(combined_df)
        completeness_pct = (actual_minutes / expected_minutes) * 100
        
        self.logger.info(f"Data completeness: {actual_minutes:,} / {expected_minutes:,} minutes ({completeness_pct:.1f}%)")
        
        return combined_df
    
    def fetch_coinbase_historical_data(self, timeframe='1m', years_back=2):
        """
        Fetch historical data from Coinbase REST API with proper pagination
        
        Args:
            timeframe: Timeframe for the data (default: 1m)
            years_back: Number of years to go back
            
        Returns:
            pandas.DataFrame: Combined historical OHLCV data
        """
        self.logger.info(f"Fetching {years_back} years of data from Coinbase REST API")
        
        granularity = self._timeframe_to_seconds(timeframe)
        
        all_chunks = []
        total_minutes = int(years_back * 365 * 24 * 60)
        chunk_size = 300
        total_requests = (total_minutes + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Will make {total_requests} API requests to fetch {total_minutes:,} minutes of data")
        
        current_time = datetime.now() - timedelta(days=years_back * 365)
        end_time = datetime.now()
        
        request_count = 0
        
        with tqdm(total=total_requests, desc="Fetching historical data", unit="req") as pbar:
            while current_time < end_time:
                chunk_end = min(current_time + timedelta(minutes=chunk_size), end_time)
                
                chunk_data = self._fetch_coinbase_chunk(
                    current_time.isoformat() + 'Z',
                    chunk_end.isoformat() + 'Z',
                    granularity
                )
                
                if not chunk_data.empty:
                    all_chunks.append(chunk_data)
                    pbar.set_postfix({
                        'Total candles': sum(len(chunk) for chunk in all_chunks),
                        'Latest': chunk_data.index.max().strftime('%Y-%m-%d %H:%M')
                    })
                
                current_time = chunk_end
                request_count += 1
                pbar.update(1)
                
                time.sleep(0.1)
                
                if request_count % 100 == 0:
                    self.logger.info(f"Completed {request_count}/{total_requests} requests")
        
        if not all_chunks:
            raise ValueError("No data was fetched from Coinbase API")
        
        self.logger.info("Combining all chunks...")
        combined_df = pd.concat(all_chunks).sort_index()
        
        initial_rows = len(combined_df)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        duplicates_removed = initial_rows - len(combined_df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        self.logger.info(f"Final dataset shape: {combined_df.shape}")
        self.logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        
        expected_minutes = int(years_back * 365 * 24 * 60)
        actual_minutes = len(combined_df)
        completeness_pct = (actual_minutes / expected_minutes) * 100
        
        self.logger.info(f"Data completeness: {actual_minutes:,} / {expected_minutes:,} minutes ({completeness_pct:.1f}%)")
        
        return combined_df
    
    def _fetch_coinbase_chunk(self, start_iso, end_iso, granularity):
        """
        Fetch a chunk of data from Coinbase REST API
        
        Args:
            start_iso: Start time in ISO format with Z suffix
            end_iso: End time in ISO format with Z suffix
            granularity: Granularity in seconds (60 for 1 minute)
            
        Returns:
            pandas.DataFrame: Chunk of historical data
        """
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {
            'start': start_iso,
            'end': end_iso,
            'granularity': granularity
        }
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    self.logger.warning(f"No data returned for time range {start_iso} to {end_iso}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                df = df.sort_index()
                
                return df
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)
                self.logger.warning(f"Request failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to fetch data after {max_retries} retries")
                    return pd.DataFrame()
    
    def _timeframe_to_seconds(self, timeframe):
        """Convert timeframe string to seconds for Coinbase API"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 60 * 60 * 24
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def _fetch_chunk(self, start_time, end_time, timeframe='1m', limit=1000):
        """Legacy method - no longer used with Coinbase REST API"""
        self.logger.warning("_fetch_chunk method is deprecated - use Coinbase REST API methods instead")
        return pd.DataFrame()
    
    def save_data(self, df, filename='bitcoin_data.csv'):
        """
        Save the data to a CSV file
        
        Args:
            df: DataFrame to save
            filename: Name of the file to save
        """
        df.to_csv(filename)
        print(f"Data saved to {filename}")
    
    def load_data(self, filename='bitcoin_data.csv'):
        """
        Load data from a CSV file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        print(f"Data loaded from {filename}, shape: {df.shape}")
        return df

if __name__ == "__main__":
    ingester = BitcoinDataIngester()
    
    data = ingester.fetch_historical_data(years_back=2)
    
    ingester.save_data(data)
    
    print("\nData sample:")
    print(data.head())
    print("\nData info:")
    print(data.info())
