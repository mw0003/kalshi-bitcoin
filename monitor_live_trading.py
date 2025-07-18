#!/usr/bin/env python3
"""
Live Trading System Monitor

This script monitors the health of the live Bitcoin forecasting system.
Run this periodically to check system status.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from glob import glob

def check_recent_forecasts(hours_back=1):
    """Check if forecasts have been generated recently"""
    forecast_dir = 'live_forecasts'
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    
    if not os.path.exists(forecast_dir):
        return False, "Forecast directory does not exist"
    
    forecast_files = glob(os.path.join(forecast_dir, 'live_forecast_*.json'))
    
    if not forecast_files:
        return False, "No forecast files found"
    
    recent_files = []
    for file_path in forecast_files:
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_time > cutoff_time:
            recent_files.append(file_path)
    
    if recent_files:
        return True, f"Found {len(recent_files)} recent forecasts"
    else:
        return False, f"No forecasts in the last {hours_back} hours"

def check_data_freshness():
    """Check if live data is being fetched successfully"""
    data_dir = 'live_data'
    
    if not os.path.exists(data_dir):
        return False, "Live data directory does not exist"
    
    data_files = glob(os.path.join(data_dir, 'live_data_*.csv'))
    
    if not data_files:
        return False, "No live data files found"
    
    latest_file = max(data_files, key=os.path.getmtime)
    file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
    age_minutes = (datetime.now() - file_time).total_seconds() / 60
    
    if age_minutes < 10:  # Data should be less than 10 minutes old
        return True, f"Latest data is {age_minutes:.1f} minutes old"
    else:
        return False, f"Latest data is {age_minutes:.1f} minutes old (too old)"

def main():
    """Run system health checks"""
    print("=" * 50)
    print("LIVE TRADING SYSTEM HEALTH CHECK")
    print(f"Time: {datetime.now()}")
    print("=" * 50)
    
    forecast_ok, forecast_msg = check_recent_forecasts()
    print(f"Recent Forecasts: {'✓' if forecast_ok else '✗'} {forecast_msg}")
    
    data_ok, data_msg = check_data_freshness()
    print(f"Data Freshness: {'✓' if data_ok else '✗'} {data_msg}")
    
    overall_ok = forecast_ok and data_ok
    print(f"\nOverall Status: {'✓ HEALTHY' if overall_ok else '✗ ISSUES DETECTED'}")
    
    return 0 if overall_ok else 1

if __name__ == "__main__":
    exit(main())
