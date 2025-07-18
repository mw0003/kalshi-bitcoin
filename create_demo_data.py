#!/usr/bin/env python3
"""
Create a small demo dataset for testing inference
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_demo_data():
    """Create a small demo dataset with Bitcoin-like price data"""
    
    n_points = 1000
    start_time = datetime(2025, 7, 17, 12, 0, 0)
    
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
    
    np.random.seed(42)
    base_price = 65000
    
    price_changes = np.random.normal(0, 50, n_points)  # $50 std dev per minute
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        if new_price > base_price * 1.05:
            new_price -= abs(change) * 0.5
        elif new_price < base_price * 0.95:
            new_price += abs(change) * 0.5
        prices.append(max(new_price, 1000))  # Minimum price of $1000
    
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        volatility = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% intrabar volatility
        
        high = close * (1 + volatility * np.random.uniform(0.5, 1.0))
        low = close * (1 - volatility * np.random.uniform(0.5, 1.0))
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.normal(0, 10)
        
        volume = np.random.uniform(100, 1000)  # Random volume
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == "__main__":
    demo_data = create_demo_data()
    demo_data.to_csv('demo_bitcoin_data.csv')
    print(f"Created demo data with {len(demo_data)} records")
    print(f"Price range: ${demo_data['close'].min():.2f} - ${demo_data['close'].max():.2f}")
    print(f"Time range: {demo_data.index[0]} to {demo_data.index[-1]}")
