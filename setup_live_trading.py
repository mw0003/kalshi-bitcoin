#!/usr/bin/env python3
"""
Setup script for live Bitcoin trading infrastructure

This script helps configure the live trading environment and provides
instructions for setting up cron jobs and monitoring.
"""

import os
import json
from datetime import datetime
from config import LIVE_CONFIG, OUTPUT_CONFIG


def create_directories():
    """Create necessary directories for live trading"""
    directories = [
        LIVE_CONFIG['live_data_dir'],
        LIVE_CONFIG['forecast_output_dir'],
        'logs',
        'monitoring'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_config_file():
    """Create live trading configuration file"""
    config = {
        'live_trading': {
            'enabled': True,
            'model_path': 'trained_models/random_forest_model.joblib',
            'forecast_frequency': '1min',
            'active_hours': {
                'start': '09:00',
                'end': '17:00',
                'timezone': 'UTC'
            },
            'api_settings': {
                'timeout': LIVE_CONFIG['api_timeout'],
                'retry_attempts': LIVE_CONFIG['retry_attempts']
            },
            'output_settings': {
                'format': LIVE_CONFIG['output_format'],
                'save_live_data': LIVE_CONFIG['save_live_data']
            }
        }
    }
    
    config_file = 'live_trading_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created configuration file: {config_file}")
    return config_file


def generate_cron_examples():
    """Generate example cron job configurations"""
    cron_examples = """

40-59 * * * * cd /path/to/kalshi-bitcoin && python run_live_forecast.py --model_path trained_models/random_forest_model.joblib >> logs/live_forecast.log 2>&1

*/5 9-17 * * * cd /path/to/kalshi-bitcoin && python run_live_forecast.py --model_path trained_models/random_forest_model.joblib >> logs/live_forecast.log 2>&1

* 9-17 * * 1-5 cd /path/to/kalshi-bitcoin && python run_live_forecast.py --model_path trained_models/random_forest_model.joblib >> logs/live_forecast.log 2>&1

0 * * * * cd /path/to/kalshi-bitcoin && python -c "from data_ingestion import BitcoinDataIngester; print('Health check:', BitcoinDataIngester().fetch_current_price())" >> logs/health_check.log 2>&1
"""
    
    cron_file = 'cron_examples.txt'
    with open(cron_file, 'w') as f:
        f.write(cron_examples)
    
    print(f"✓ Created cron examples: {cron_file}")
    return cron_file


def create_monitoring_script():
    """Create monitoring script for live trading system"""
    monitoring_script = '''#!/usr/bin/env python3
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
    print(f"\\nOverall Status: {'✓ HEALTHY' if overall_ok else '✗ ISSUES DETECTED'}")
    
    return 0 if overall_ok else 1

if __name__ == "__main__":
    exit(main())
'''
    
    monitor_file = 'monitor_live_trading.py'
    with open(monitor_file, 'w') as f:
        f.write(monitoring_script)
    
    os.chmod(monitor_file, 0o755)  # Make executable
    print(f"✓ Created monitoring script: {monitor_file}")
    return monitor_file


def print_setup_instructions():
    """Print setup instructions for live trading"""
    instructions = f"""
🚀 LIVE BITCOIN TRADING SETUP COMPLETE!

📁 Directory Structure:
   ├── {LIVE_CONFIG['live_data_dir']}/          # Live Bitcoin data storage
   ├── {LIVE_CONFIG['forecast_output_dir']}/    # Live forecast outputs
   ├── logs/                    # System logs
   └── monitoring/              # Monitoring scripts

🔧 Configuration:
   • API: Coinbase REST API (no authentication required)
   • Data fetch: {LIVE_CONFIG['data_fetch_minutes']} minutes of recent data
   • Output format: {LIVE_CONFIG['output_format'].upper()}
   • Timeout: {LIVE_CONFIG['api_timeout']} seconds

📋 Next Steps:

1. Train a model (if not already done):
   python simple_working_demo.py

2. Test live forecasting:
   python run_live_forecast.py --model_path trained_models/simple_demo_model.joblib

3. Test with historical timestamp:
   python run_live_forecast.py --model_path trained_models/simple_demo_model.joblib --timestamp "2025-01-18 15:30:00"

4. Set up cron job (see cron_examples.txt):
   crontab -e

5. Monitor system health:
   python monitor_live_trading.py

🔍 Monitoring:
   • Check logs/live_forecast.log for detailed logs
   • Monitor {LIVE_CONFIG['forecast_output_dir']}/ for forecast outputs
   • Run monitor_live_trading.py for health checks

⚠️  Production Notes:
   • Ensure stable internet connection for API calls
   • Monitor disk space for log and data files
   • Consider log rotation for long-running systems
   • Test thoroughly before deploying to production

🎯 Trading Integration:
   The forecast outputs include:
   • Point predictions for 1-20 minute horizons
   • Quantile predictions (5%, 25%, 50%, 75%, 95%)
   • Probability queries for price thresholds
   • Prediction intervals for uncertainty quantification

   Use these outputs in your trading logic to make informed decisions!
"""
    
    print(instructions)


def main():
    """Main setup function"""
    print("Setting up live Bitcoin trading infrastructure...")
    print()
    
    create_directories()
    create_config_file()
    generate_cron_examples()
    create_monitoring_script()
    
    print()
    print_setup_instructions()


if __name__ == "__main__":
    main()
