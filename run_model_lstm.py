#!/usr/bin/env python3
"""
LSTM Model Training Script

This script trains an LSTM model for multi-horizon Bitcoin price forecasting
using rolling time-based validation and saves the results.

Usage:
    python run_model_lstm.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from data_ingestion import BitcoinDataIngester
from enhanced_feature_engineering import EnhancedFeatureEngineer
from models.lstm_model import LSTMForecaster
from visualization import ForecastVisualizer
from utils import setup_logging, ProgressTimer, save_metrics_to_csv
from config import DATA_CONFIG, MODEL_CONFIG, VALIDATION_CONFIG, OUTPUT_CONFIG

def main():
    """Main LSTM training process"""
    print("=" * 80)
    print("LSTM MODEL TRAINING - BITCOIN FORECASTING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    logger = setup_logging('lstm_training.log')
    
    try:
        print("STEP 1: DEPENDENCY CHECK")
        print("-" * 40)
        
        try:
            import darts
            import torch
            print("✓ Darts and PyTorch are available")
        except ImportError as e:
            print(f"ERROR: Required dependencies not available: {e}")
            print("Please install with: pip install darts torch")
            sys.exit(1)
        
        print("STEP 2: LOADING DATA")
        print("-" * 40)
        
        data_file = DATA_CONFIG['data_file']
        if not os.path.exists(data_file):
            print(f"ERROR: Data file {data_file} not found!")
            print("Please run 'python run_data_ingestion.py' first to fetch the data.")
            sys.exit(1)
        
        ingester = BitcoinDataIngester()
        with ProgressTimer("Loading Bitcoin data"):
            data = ingester.load_data(data_file)
        
        print(f"Data loaded: {data.shape[0]:,} records")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Duration: {(data.index.max() - data.index.min()).days} days")
        print()
        
        print("STEP 3: FEATURE ENGINEERING")
        print("-" * 40)
        
        engineer = EnhancedFeatureEngineer()
        with ProgressTimer("Feature engineering"):
            features_df = engineer.engineer_features(data)
        
        print(f"Features created: {len(engineer.feature_columns)}")
        print(f"Data shape after feature engineering: {features_df.shape}")
        print()
        
        print("STEP 4: DATA PREPARATION")
        print("-" * 40)
        
        df_clean = engineer.prepare_for_modeling(features_df, VALIDATION_CONFIG)
        
        feature_cols = engineer.feature_columns[:20]  # Limit features for LSTM
        target_cols = [col for col in df_clean.columns if col.startswith('target_')]
        
        print(f"Clean data shape: {df_clean.shape}")
        print(f"Features (limited for LSTM): {len(feature_cols)}")
        print(f"Targets: {len(target_cols)}")
        print()
        
        print("STEP 5: ROLLING TIME-BASED VALIDATION")
        print("-" * 40)
        print("Note: LSTM validation may take significantly longer than other models")
        
        forecaster = LSTMForecaster()
        
        with ProgressTimer("Rolling validation"):
            validation_results = forecaster.rolling_validation(
                df_clean, feature_cols, target_cols
            )
        
        if not validation_results:
            print("ERROR: No validation results obtained!")
            sys.exit(1)
        
        print(f"Completed {len(validation_results)} validation splits")
        
        all_metrics = pd.concat(validation_results, ignore_index=True)
        
        metrics_file = os.path.join(
            OUTPUT_CONFIG['results_dir'],
            OUTPUT_CONFIG['metrics_file_template'].format(model_name='lstm')
        )
        save_metrics_to_csv(all_metrics.to_dict('records'), metrics_file)
        
        print(f"Validation metrics saved to: {metrics_file}")
        print()
        
        print("STEP 6: FINAL MODEL TRAINING")
        print("-" * 40)
        
        train_size = int(len(df_clean) * 0.8)
        X_train = df_clean[feature_cols].iloc[:train_size]
        y_train = df_clean[target_cols].iloc[:train_size]
        X_test = df_clean[feature_cols].iloc[train_size:]
        y_test = df_clean[target_cols].iloc[train_size:]
        
        final_forecaster = LSTMForecaster()
        with ProgressTimer("Final LSTM model training"):
            final_forecaster.fit(X_train, y_train)
        
        model_file = final_forecaster.save_model()
        print(f"Final model saved to: {model_file}")
        
        print("STEP 7: FINAL EVALUATION")
        print("-" * 40)
        
        final_metrics = final_forecaster.evaluate(X_test, y_test)
        print("Final Model Performance:")
        print(final_metrics.groupby('horizon')[['rmse', 'mae', 'mape']].mean())
        
        avg_rmse = final_metrics['rmse'].mean()
        avg_mae = final_metrics['mae'].mean()
        avg_mape = final_metrics['mape'].mean()
        
        print(f"\nOverall Performance:")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Average MAPE: {avg_mape:.2f}%")
        
        print("STEP 8: VISUALIZATION")
        print("-" * 40)
        
        visualizer = ForecastVisualizer()
        
        plots_dir = OUTPUT_CONFIG['plots_dir']
        
        predictions = final_forecaster.predict(X_test)
        visualizer.plot_predictions_vs_actual(
            y_test, predictions,
            horizons_to_plot=[1, 5, 10, 20],
            save_path=os.path.join(plots_dir, 'lstm_predictions.png')
        )
        
        visualizer.plot_error_metrics(
            final_metrics,
            save_path=os.path.join(plots_dir, 'lstm_error_metrics.png')
        )
        
        print("Visualizations saved to plots/ directory")
        
        print("\n" + "=" * 80)
        print("LSTM TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Generated files:")
        print(f"- {metrics_file}")
        print(f"- {model_file}")
        print("- Visualization plots in plots/ directory")
        
    except KeyboardInterrupt:
        logger.info("LSTM training interrupted by user")
        print("\nTraining interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}")
        print(f"ERROR: Training failed - {str(e)}")
        print("Check the log file 'lstm_training.log' for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
