# Bitcoin Multi-Horizon Price Forecasting Pipeline

This project implements a comprehensive machine learning pipeline for predicting Bitcoin prices at multiple horizons (1-20 minutes ahead) using 2 years of historical minute-level data.

## Features

- **2 Years of Data**: Uses ~1M data points of 1-minute Bitcoin price data from Kraken
- **3 Forecasting Models**: Random Forest, LSTM (using Darts), and LightGBM
- **Rolling Time-Based Validation**: 7-day training windows, 1-day test windows, sliding forward
- **Comprehensive Evaluation**: RMSE, MAE, MAPE metrics per horizon
- **Progress Tracking**: Progress bars and time estimation for long operations
- **Modular Design**: Independent scripts with proper error handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Ingestion (Run First)

```bash
python run_data_ingestion.py
```

This script:
- Fetches 2 years of 1-minute Bitcoin data from Kraken
- Includes progress tracking and quality checks
- Saves data to `bitcoin_2year_data.csv`
- Takes 10-30 minutes depending on connection

### 2. Model Training (Run Independently)

#### Random Forest Model
```bash
python run_model_random_forest.py
```

#### LightGBM Model
```bash
python run_model_lightgbm.py
```

#### LSTM Model
```bash
python run_model_lstm.py
```

Each training script:
- Performs rolling time-based validation
- Saves trained model and metrics
- Generates visualization plots
- Takes 30-120 minutes depending on model complexity

### 3. Model Comparison

```bash
python compare_model_results.py
```

This script:
- Loads results from all trained models
- Creates performance leaderboard
- Generates comparison visualizations
- Highlights best-performing model

## Project Structure

```
bitcoin_forecasting/
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── base_model.py          # Abstract base class
│   ├── random_forest_model.py # Random Forest implementation
│   ├── lightgbm_model.py      # LightGBM implementation
│   └── lstm_model.py          # LSTM implementation
├── results/                   # Model results and metrics
├── trained_models/           # Saved model files
├── plots/                    # Generated visualizations
├── run_data_ingestion.py     # Data fetching script
├── run_model_random_forest.py # Random Forest training
├── run_model_lightgbm.py     # LightGBM training
├── run_model_lstm.py         # LSTM training
├── compare_model_results.py  # Model comparison
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── enhanced_feature_engineering.py # Feature engineering
├── data_ingestion.py         # Data ingestion class
└── visualization.py          # Plotting functions
```

## Configuration

Edit `config.py` to modify:
- Model hyperparameters
- Validation settings (train/test window sizes)
- Feature engineering options
- Output directories

## Key Features

### Rolling Time-Based Validation
- 7-day training windows (10,080 minutes)
- 1-day test windows (1,440 minutes)
- Sliding forward by 1 day between splits
- Minimum 1,000 training samples per split

### Feature Engineering
- Lagged price features (1, 2, 3, 5, 10, 15, 30, 60 minutes)
- Rolling statistics (mean, std, min, max, median, skew, kurtosis)
- Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ATR, etc.)
- Time-based features (hour, day of week, cyclical encodings)

### Model Implementations
- **Random Forest**: Ensemble method with feature importance
- **LightGBM**: Gradient boosting with fast training
- **LSTM**: Deep learning for time series using Darts library

## Output Files

### Data Files
- `bitcoin_2year_data.csv`: Raw Bitcoin price data
- `{model_name}_metrics.csv`: Validation metrics per model
- `model_leaderboard.csv`: Performance comparison

### Model Files
- `{model_name}_model.joblib`: Trained model files

### Visualizations
- Model-specific prediction plots
- Error metrics across horizons
- Feature importance plots (for tree-based models)
- Model comparison charts
- Performance heatmaps

## Requirements

- Python 3.8+
- See `requirements.txt` for complete dependency list
- Key dependencies: pandas, scikit-learn, lightgbm, darts, torch, tqdm

## Notes

- All scripts include comprehensive error handling and logging
- Progress bars show estimated completion times
- Scripts are designed to run independently without auto-execution
- LSTM model requires PyTorch and may benefit from GPU acceleration
- Data fetching respects API rate limits with built-in delays

## Troubleshooting

1. **Data fetching fails**: Check internet connection and API availability
2. **LSTM training fails**: Ensure PyTorch is properly installed
3. **Memory issues**: Reduce feature set or use smaller validation windows
4. **Long training times**: Consider using smaller datasets for testing

For detailed logs, check the generated `.log` files in the project directory.
