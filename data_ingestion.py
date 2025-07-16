import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
from utils import setup_logging, ProgressTimer

class BitcoinDataIngester:
    def __init__(self, exchange_name='kraken', symbol='BTC/USD'):
        """
        Initialize the Bitcoin data ingester
        
        Args:
            exchange_name: Name of the exchange (default: kraken)
            symbol: Trading pair symbol (default: BTC/USD)
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.exchange = getattr(ccxt, exchange_name)()
        self.logger = setup_logging()
        
    def fetch_historical_data(self, timeframe='1m', limit=1000, years_back=2):
        """
        Fetch historical OHLCV data from the exchange with progress tracking
        
        Args:
            timeframe: Timeframe for the data (default: 1m for 1-minute)
            limit: Number of candles to fetch per request
            years_back: Number of years to go back from current time
            
        Returns:
            pandas.DataFrame: Historical OHLCV data
        """
        with ProgressTimer(f"Fetching {years_back} years of {timeframe} data for {self.symbol}"):
            if hasattr(self.exchange, 'sandbox'):
                self.exchange.sandbox = False
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=years_back * 365)
            since = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            end_timestamp = int(end_time.timestamp() * 1000)
            
            total_minutes = int((end_timestamp - since) / 60000)
            estimated_requests = max(1, total_minutes // limit)
            
            self.logger.info(f"Estimated {estimated_requests} requests needed for {total_minutes:,} minutes of data")
            
            all_ohlcv = []
            current_since = since
            request_count = 0
            
            pbar = tqdm(total=estimated_requests, desc="Fetching data", unit="req")
            
            while current_since < end_timestamp:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=limit
                    )
                    
                    if not ohlcv:
                        self.logger.warning("No more data available")
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 60000  # Add 1 minute in milliseconds
                    request_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Total candles': len(all_ohlcv),
                        'Latest': pd.to_datetime(ohlcv[-1][0], unit='ms').strftime('%Y-%m-%d')
                    })
                    
                    time.sleep(0.1)
                    
                    if request_count % 100 == 0:
                        self.logger.info(f"Completed {request_count} requests, {len(all_ohlcv):,} candles fetched")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching data: {e}")
                    time.sleep(1)  # Wait longer on error
                    continue
            
            pbar.close()
            
            if not all_ohlcv:
                raise ValueError("No data was fetched")
            
            self.logger.info("Converting data to DataFrame...")
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            initial_shape = df.shape[0]
            df = df[~df.index.duplicated(keep='first')]
            duplicates_removed = initial_shape - df.shape[0]
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate entries")
            
            self.logger.info(f"Final dataset shape: {df.shape}")
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Data completeness: {len(df):,} minutes over {years_back} years")
            
            return df
    
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
