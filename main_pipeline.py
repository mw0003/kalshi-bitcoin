#!/usr/bin/env python3
"""
Multi-Horizon Bitcoin Price Forecasting Pipeline

This script implements a complete machine learning pipeline for predicting
Bitcoin prices at multiple horizons (1-20 minutes ahead).
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

from data_ingestion import BitcoinDataIngester
from feature_engineering import FeatureEngineer
from model_training import MultiHorizonForecaster, WalkForwardValidator
from visualization import ForecastVisualizer

def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("MULTI-HORIZON BITCOIN PRICE FORECASTING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    DAYS_BACK = 7  # Number of days of historical data to fetch
    TEST_SIZE = 0.2  # Proportion of data for testing
    MODEL_TYPE = 'random_forest'  # Model type to use
    
    try:
        print("STEP 1: DATA INGESTION")
        print("-" * 30)
        
        ingester = BitcoinDataIngester()
        
        try:
            data = ingester.load_data('bitcoin_data.csv')
            print("Loaded existing data from file.")
        except FileNotFoundError:
            print("No existing data found. Fetching new data...")
            data = ingester.fetch_historical_data(days_back=DAYS_BACK)
            ingester.save_data(data, 'bitcoin_data.csv')
        
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print()
        
        print("STEP 2: FEATURE ENGINEERING")
        print("-" * 30)
        
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(data)
        
        print(f"Features created: {len(engineer.feature_columns)}")
        print("Sample features:", engineer.feature_columns[:5])
        print()
        
        print("STEP 3: DATA PREPARATION")
        print("-" * 30)
        
        X_train, X_test, y_train, y_test = engineer.prepare_for_modeling(
            features_df, test_size=TEST_SIZE
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of forecast horizons: {y_train.shape[1]}")
        print()
        
        print("STEP 4: MODEL TRAINING")
        print("-" * 30)
        
        forecaster = MultiHorizonForecaster(model_type=MODEL_TYPE)
        forecaster.fit(X_train, y_train)
        
        forecaster.save_model('bitcoin_forecaster.joblib')
        print()
        
        print("STEP 5: MODEL EVALUATION")
        print("-" * 30)
        
        predictions = forecaster.predict(X_test)
        
        metrics = forecaster.evaluate(X_test, y_test)
        
        print("Evaluation Results:")
        print(metrics.to_string(index=False))
        
        metrics.to_csv('evaluation_metrics.csv', index=False)
        print("\nMetrics saved to evaluation_metrics.csv")
        print()
        
        print("STEP 6: FEATURE IMPORTANCE")
        print("-" * 30)
        
        importance = forecaster.get_feature_importance(top_n=15)
        if importance is not None:
            print("Top 15 Most Important Features:")
            print(importance.to_string(index=False))
            importance.to_csv('feature_importance.csv', index=False)
            print("\nFeature importance saved to feature_importance.csv")
        print()
        
        print("STEP 7: VISUALIZATION")
        print("-" * 30)
        
        visualizer = ForecastVisualizer()
        
        print("Creating data overview plot...")
        visualizer.plot_data_overview(data, save_path='plots/data_overview.png')
        
        if importance is not None:
            print("Creating feature importance plot...")
            visualizer.plot_feature_importance(importance, save_path='plots/feature_importance.png')
        
        print("Creating predictions vs actual plots...")
        visualizer.plot_predictions_vs_actual(
            y_test, predictions, 
            horizons_to_plot=[1, 5, 10, 20],
            save_path='plots/predictions_vs_actual.png'
        )
        
        print("Creating error metrics plot...")
        visualizer.plot_error_metrics(metrics, save_path='plots/error_metrics.png')
        
        print("Creating residuals plot...")
        visualizer.plot_residuals(
            y_test, predictions,
            horizons_to_plot=[1, 5, 10, 20],
            save_path='plots/residuals.png'
        )
        
        print("All plots saved to 'plots/' directory")
        print()
        
        print("STEP 8: WALK-FORWARD VALIDATION")
        print("-" * 30)
        
        print("Performing walk-forward validation...")
        validator = WalkForwardValidator(
            MultiHorizonForecaster, 
            {'model_type': MODEL_TYPE}
        )
        
        subset_size = min(2000, len(X_train))
        X_subset = X_train.iloc[-subset_size:]
        y_subset = y_train.iloc[-subset_size:]
        
        wf_results = validator.validate(X_subset, y_subset, n_splits=3, test_size=0.3)
        
        visualizer.plot_walk_forward_results(wf_results, save_path='plots/walk_forward_results.png')
        
        all_wf_metrics = pd.concat(wf_results, ignore_index=True)
        all_wf_metrics.to_csv('walk_forward_metrics.csv', index=False)
        print("Walk-forward validation results saved to walk_forward_metrics.csv")
        print()
        
        print("STEP 9: SUMMARY REPORT")
        print("-" * 30)
        
        avg_rmse = metrics['rmse'].mean()
        avg_mae = metrics['mae'].mean()
        avg_mape = metrics['mape'].mean()
        
        best_horizon_rmse = metrics.loc[metrics['rmse'].idxmin(), 'horizon']
        worst_horizon_rmse = metrics.loc[metrics['rmse'].idxmax(), 'horizon']
        
        print(f"Model Type: {MODEL_TYPE}")
        print(f"Training Samples: {len(X_train):,}")
        print(f"Test Samples: {len(X_test):,}")
        print(f"Number of Features: {len(engineer.feature_columns)}")
        print(f"Forecast Horizons: 1-20 minutes")
        print()
        print("PERFORMANCE SUMMARY:")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Average MAPE: {avg_mape:.2f}%")
        print()
        print(f"Best performing horizon (lowest RMSE): {best_horizon_rmse} minutes")
        print(f"Worst performing horizon (highest RMSE): {worst_horizon_rmse} minutes")
        print()
        
        with open('summary_report.txt', 'w') as f:
            f.write("MULTI-HORIZON BITCOIN PRICE FORECASTING - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model Type: {MODEL_TYPE}\n")
            f.write(f"Training Samples: {len(X_train):,}\n")
            f.write(f"Test Samples: {len(X_test):,}\n")
            f.write(f"Number of Features: {len(engineer.feature_columns)}\n")
            f.write(f"Forecast Horizons: 1-20 minutes\n\n")
            f.write("PERFORMANCE SUMMARY:\n")
            f.write(f"Average RMSE: {avg_rmse:.4f}\n")
            f.write(f"Average MAE: {avg_mae:.4f}\n")
            f.write(f"Average MAPE: {avg_mape:.2f}%\n\n")
            f.write(f"Best performing horizon (lowest RMSE): {best_horizon_rmse} minutes\n")
            f.write(f"Worst performing horizon (highest RMSE): {worst_horizon_rmse} minutes\n\n")
            f.write("FILES GENERATED:\n")
            f.write("- bitcoin_data.csv: Raw Bitcoin price data\n")
            f.write("- bitcoin_forecaster.joblib: Trained model\n")
            f.write("- evaluation_metrics.csv: Detailed metrics per horizon\n")
            f.write("- feature_importance.csv: Feature importance scores\n")
            f.write("- walk_forward_metrics.csv: Walk-forward validation results\n")
            f.write("- plots/: Directory containing all visualization plots\n")
        
        print("Summary report saved to summary_report.txt")
        print()
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Generated files:")
        print("- bitcoin_data.csv")
        print("- bitcoin_forecaster.joblib")
        print("- evaluation_metrics.csv")
        print("- feature_importance.csv")
        print("- walk_forward_metrics.csv")
        print("- summary_report.txt")
        print("- plots/ directory with visualization plots")
        
    except Exception as e:
        print(f"ERROR: Pipeline failed with exception: {str(e)}")
        print("Please check the error details above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    
    main()
