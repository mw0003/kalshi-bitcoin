#!/usr/bin/env python3
"""
Random Forest Model Training Script

This script trains a Random Forest model for multi-horizon Bitcoin price forecasting
using rolling time-based validation and saves the results.

Usage:
    python run_model_random_forest.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from data_ingestion import BitcoinDataIngester
from enhanced_feature_engineering import EnhancedFeatureEngineer
from models.random_forest_model import RandomForestForecaster
from visualization import ForecastVisualizer
from utils import setup_logging, ProgressTimer, save_metrics_to_csv
from config import DATA_CONFIG, MODEL_CONFIG, VALIDATION_CONFIG, OUTPUT_CONFIG

def main():
    """Main Random Forest training process"""
    print("=" * 80)
    print("RANDOM FOREST MODEL TRAINING - BITCOIN FORECASTING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    logger = setup_logging('random_forest_training.log')
    
    try:
        print("STEP 1: LOADING DATA")
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
        
        print("STEP 2: FEATURE ENGINEERING")
        print("-" * 40)
        
        engineer = EnhancedFeatureEngineer()
        with ProgressTimer("Feature engineering"):
            features_df = engineer.engineer_features(data)
        
        print(f"Features created: {len(engineer.feature_columns)}")
        print(f"Data shape after feature engineering: {features_df.shape}")
        print()
        
        print("STEP 3: DATA PREPARATION")
        print("-" * 40)
        
        df_clean = engineer.prepare_for_modeling(features_df, VALIDATION_CONFIG)
        
        feature_cols = engineer.feature_columns
        target_cols = [col for col in df_clean.columns if col.startswith('target_')]
        
        print(f"Clean data shape: {df_clean.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Targets: {len(target_cols)}")
        print()
        
        print("STEP 4: ROLLING TIME-BASED VALIDATION")
        print("-" * 40)
        
        forecaster = RandomForestForecaster()
        
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
            OUTPUT_CONFIG['metrics_file_template'].format(model_name='random_forest')
        )
        save_metrics_to_csv(all_metrics.to_dict('records'), metrics_file)
        
        print(f"Validation metrics saved to: {metrics_file}")
        print()
        
        print("STEP 5: FINAL MODEL TRAINING")
        print("-" * 40)
        
        train_size = int(len(df_clean) * 0.8)
        X_train = df_clean[feature_cols].iloc[:train_size]
        y_train = df_clean[target_cols].iloc[:train_size]
        X_test = df_clean[feature_cols].iloc[train_size:]
        y_test = df_clean[target_cols].iloc[train_size:]
        
        final_forecaster = RandomForestForecaster()
        with ProgressTimer("Final model training"):
            final_forecaster.fit(X_train, y_train)
        
        model_file = final_forecaster.save_model()
        print(f"Final model saved to: {model_file}")
        
        print("STEP 6: FINAL EVALUATION")
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
        
        print("STEP 7: FEATURE IMPORTANCE")
        print("-" * 40)
        
        importance = final_forecaster.get_feature_importance(top_n=20)
        if importance is not None:
            print("Top 20 Most Important Features:")
            print(importance.to_string(index=False))
            
            importance_file = os.path.join(
                OUTPUT_CONFIG['results_dir'],
                'random_forest_feature_importance.csv'
            )
            importance.to_csv(importance_file, index=False)
            print(f"Feature importance saved to: {importance_file}")
        
        print("STEP 8: VISUALIZATION")
        print("-" * 40)
        
        visualizer = ForecastVisualizer()
        
        plots_dir = OUTPUT_CONFIG['plots_dir']
        
        if importance is not None:
            visualizer.plot_feature_importance(
                importance, 
                save_path=os.path.join(plots_dir, 'random_forest_feature_importance.png')
            )
        
        predictions = final_forecaster.predict(X_test)
        visualizer.plot_predictions_vs_actual(
            y_test, predictions,
            horizons_to_plot=[1, 5, 10, 20],
            save_path=os.path.join(plots_dir, 'random_forest_predictions.png')
        )
        
        visualizer.plot_error_metrics(
            final_metrics,
            save_path=os.path.join(plots_dir, 'random_forest_error_metrics.png')
        )
        
        print("Visualizations saved to plots/ directory")
        
        print("STEP 9: PROBABILISTIC FORECASTING")
        print("-" * 40)
        
        print("Generating probabilistic forecasts...")
        quantile_preds = final_forecaster.predict_quantiles(X_test)
        
        current_price = y_test.iloc[-1, 0] if len(y_test) > 0 else 50000
        thresholds = {
            f'above_{int(current_price * 1.02)}': current_price * 1.02,
            f'below_{int(current_price * 0.98)}': current_price * 0.98,
            f'between_{int(current_price * 0.99)}_{int(current_price * 1.01)}': (current_price * 0.99, current_price * 1.01)
        }
        
        probabilities = final_forecaster.calculate_probabilities(X_test, thresholds)
        
        prob_results = []
        for query_name, prob_array in probabilities.items():
            for horizon in range(prob_array.shape[1]):
                prob_results.append({
                    'query': query_name,
                    'horizon': horizon + 1,
                    'mean_probability': np.mean(prob_array[:, horizon]),
                    'std_probability': np.std(prob_array[:, horizon])
                })
        
        prob_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'random_forest_probabilities.csv')
        pd.DataFrame(prob_results).to_csv(prob_file, index=False)
        print(f"Probability results saved to: {prob_file}")
        
        print("STEP 10: CALIBRATION ANALYSIS")
        print("-" * 40)
        
        try:
            from probabilistic_forecasting import CalibrationAnalyzer
            
            calibrator = CalibrationAnalyzer()
            calibration_results = calibrator.evaluate_calibration(
                y_test.values, probabilities, horizons=[1, 5, 10, 20]
            )
            
            cal_metrics_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'random_forest_calibration.csv')
            calibrator.save_calibration_metrics(calibration_results, cal_metrics_file)
            
            cal_plot_file = os.path.join(OUTPUT_CONFIG['plots_dir'], 'random_forest_calibration.png')
            calibrator.plot_calibration_curve(calibration_results, cal_plot_file)
            
            print(f"Calibration metrics saved to: {cal_metrics_file}")
            print(f"Calibration plots saved to: {cal_plot_file}")
        except Exception as cal_error:
            print(f"Warning: Calibration analysis failed: {cal_error}")
        
        print("\n" + "=" * 80)
        print("RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Generated files:")
        print(f"- {metrics_file}")
        print(f"- {model_file}")
        if importance is not None:
            print(f"- {importance_file}")
        print(f"- {prob_file}")
        print(f"- {cal_metrics_file}")
        print(f"- {cal_plot_file}")
        print("- Visualization plots in plots/ directory")
        
    except KeyboardInterrupt:
        logger.info("Random Forest training interrupted by user")
        print("\nTraining interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Random Forest training failed: {str(e)}")
        print(f"ERROR: Training failed - {str(e)}")
        print("Check the log file 'random_forest_training.log' for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
