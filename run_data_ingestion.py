#!/usr/bin/env python3
"""
Bitcoin Data Ingestion Script

This script fetches 2 years of minute-level Bitcoin price data from Kraken exchange.
Run this script first before training any models.

Usage:
    python run_data_ingestion.py
"""

import os
import sys
from datetime import datetime
from data_ingestion import BitcoinDataIngester
from config import DATA_CONFIG, OUTPUT_CONFIG
from utils import setup_logging, ProgressTimer

def main():
    """Main data ingestion process"""
    print("=" * 80)
    print("BITCOIN DATA INGESTION - 2 YEARS OF MINUTE-LEVEL DATA")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    logger = setup_logging('data_ingestion.log')
    
    try:
        logger.info(f"Initializing data ingester for {DATA_CONFIG['exchange']} - {DATA_CONFIG['symbol']}")
        ingester = BitcoinDataIngester(
            exchange_name=DATA_CONFIG['exchange'],
            symbol=DATA_CONFIG['symbol']
        )
        
        data_file = DATA_CONFIG['data_file']
        if os.path.exists(data_file):
            print(f"Data file {data_file} already exists.")
            response = input("Do you want to re-fetch the data? (y/N): ").strip().lower()
            if response != 'y':
                print("Using existing data file.")
                try:
                    existing_data = ingester.load_data(data_file)
                    print(f"Existing data shape: {existing_data.shape}")
                    print(f"Date range: {existing_data.index.min()} to {existing_data.index.max()}")
                    return
                except Exception as e:
                    logger.error(f"Error loading existing data: {e}")
                    print("Proceeding with fresh data fetch...")
        
        print(f"Fetching {DATA_CONFIG['years_back']} years of {DATA_CONFIG['timeframe']} data...")
        print("This may take 10-30 minutes depending on your connection and rate limits.")
        print()
        
        with ProgressTimer("Complete data ingestion process"):
            data = ingester.fetch_historical_data(
                timeframe=DATA_CONFIG['timeframe'],
                years_back=DATA_CONFIG['years_back']
            )
            
            logger.info(f"Saving data to {data_file}")
            ingester.save_data(data, data_file)
        
        print("\n" + "=" * 50)
        print("DATA INGESTION SUMMARY")
        print("=" * 50)
        print(f"Total records: {len(data):,}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Duration: {(data.index.max() - data.index.min()).days} days")
        print(f"File size: {os.path.getsize(data_file) / (1024*1024):.1f} MB")
        print()
        
        print("DATA QUALITY CHECKS:")
        print(f"Missing values: {data.isnull().sum().sum()}")
        print(f"Duplicate timestamps: {data.index.duplicated().sum()}")
        print(f"Zero volume records: {(data['volume'] == 0).sum()}")
        
        print(f"\nPRICE STATISTICS:")
        print(f"Min price: ${data['close'].min():,.2f}")
        print(f"Max price: ${data['close'].max():,.2f}")
        print(f"Mean price: ${data['close'].mean():,.2f}")
        print(f"Price volatility (std): ${data['close'].std():,.2f}")
        
        print(f"\nData saved to: {data_file}")
        print("You can now run the model training scripts.")
        
    except KeyboardInterrupt:
        logger.info("Data ingestion interrupted by user")
        print("\nData ingestion interrupted. Partial data may have been saved.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        print(f"ERROR: Data ingestion failed - {str(e)}")
        print("Check the log file 'data_ingestion.log' for detailed error information.")
        sys.exit(1)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
