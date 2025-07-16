#!/usr/bin/env python3
"""
Model Comparison Script

This script loads results from all trained models, compares their performance,
and generates visualizations and leaderboards.

Usage:
    python compare_model_results.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils import setup_logging, load_metrics_from_csv
from config import OUTPUT_CONFIG

def load_all_model_results():
    """
    Load results from all trained models
    
    Returns:
        dict: Dictionary of model results
    """
    models = ['random_forest', 'lightgbm', 'lstm']
    results = {}
    
    for model in models:
        metrics_file = os.path.join(
            OUTPUT_CONFIG['results_dir'],
            OUTPUT_CONFIG['metrics_file_template'].format(model_name=model)
        )
        
        if os.path.exists(metrics_file):
            try:
                results[model] = load_metrics_from_csv(metrics_file)
                print(f"✓ Loaded results for {model}: {len(results[model])} records")
            except Exception as e:
                print(f"✗ Failed to load results for {model}: {e}")
        else:
            print(f"✗ Results file not found for {model}: {metrics_file}")
    
    return results

def create_leaderboard(results):
    """
    Create performance leaderboard across all models
    
    Args:
        results: Dictionary of model results
        
    Returns:
        pandas.DataFrame: Leaderboard DataFrame
    """
    leaderboard_data = []
    
    for model_name, df in results.items():
        if df is not None and len(df) > 0:
            avg_rmse = df['rmse'].mean()
            avg_mae = df['mae'].mean()
            avg_mape = df['mape'].mean()
            
            best_horizon_rmse = df.loc[df['rmse'].idxmin(), 'horizon']
            worst_horizon_rmse = df.loc[df['rmse'].idxmax(), 'horizon']
            
            leaderboard_data.append({
                'model': model_name.replace('_', ' ').title(),
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'avg_mape': avg_mape,
                'best_horizon': best_horizon_rmse,
                'worst_horizon': worst_horizon_rmse,
                'total_predictions': len(df)
            })
    
    leaderboard = pd.DataFrame(leaderboard_data)
    leaderboard = leaderboard.sort_values('avg_rmse')
    leaderboard['rank'] = range(1, len(leaderboard) + 1)
    
    return leaderboard

def plot_model_comparison(results, save_path=None):
    """
    Create comprehensive model comparison plots
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    all_data = []
    for model_name, df in results.items():
        if df is not None and len(df) > 0:
            df_copy = df.copy()
            df_copy['model'] = model_name.replace('_', ' ').title()
            all_data.append(df_copy)
    
    if not all_data:
        print("No data available for plotting")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    sns.boxplot(data=combined_df, x='model', y='rmse', ax=axes[0, 0])
    axes[0, 0].set_title('RMSE Distribution by Model')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=combined_df, x='model', y='mae', ax=axes[0, 1])
    axes[0, 1].set_title('MAE Distribution by Model')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    horizon_performance = combined_df.groupby(['model', 'horizon'])[['rmse', 'mae']].mean().reset_index()
    
    for model in horizon_performance['model'].unique():
        model_data = horizon_performance[horizon_performance['model'] == model]
        axes[1, 0].plot(model_data['horizon'], model_data['rmse'], 
                       marker='o', label=model, linewidth=2)
    
    axes[1, 0].set_title('RMSE by Forecast Horizon')
    axes[1, 0].set_xlabel('Forecast Horizon (minutes)')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for model in horizon_performance['model'].unique():
        model_data = horizon_performance[horizon_performance['model'] == model]
        axes[1, 1].plot(model_data['horizon'], model_data['mae'], 
                       marker='s', label=model, linewidth=2)
    
    axes[1, 1].set_title('MAE by Forecast Horizon')
    axes[1, 1].set_xlabel('Forecast Horizon (minutes)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.close()

def plot_performance_heatmap(results, save_path=None):
    """
    Create performance heatmap across models and horizons
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the plot
    """
    all_data = []
    for model_name, df in results.items():
        if df is not None and len(df) > 0:
            df_copy = df.copy()
            df_copy['model'] = model_name.replace('_', ' ').title()
            all_data.append(df_copy)
    
    if not all_data:
        print("No data available for heatmap")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    pivot_rmse = combined_df.groupby(['model', 'horizon'])['rmse'].mean().unstack()
    pivot_mae = combined_df.groupby(['model', 'horizon'])['mae'].mean().unstack()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Heatmaps: Models vs Forecast Horizons', fontsize=14, fontweight='bold')
    
    sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('RMSE Heatmap')
    axes[0].set_xlabel('Forecast Horizon (minutes)')
    axes[0].set_ylabel('Model')
    
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('MAE Heatmap')
    axes[1].set_xlabel('Forecast Horizon (minutes)')
    axes[1].set_ylabel('Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to: {save_path}")
    
    plt.close()

def main():
    """Main comparison process"""
    print("=" * 80)
    print("MODEL COMPARISON AND ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    logger = setup_logging('model_comparison.log')
    
    try:
        print("STEP 1: LOADING MODEL RESULTS")
        print("-" * 40)
        
        results = load_all_model_results()
        
        if not results:
            print("ERROR: No model results found!")
            print("Please run the model training scripts first:")
            print("- python run_model_random_forest.py")
            print("- python run_model_lightgbm.py")
            print("- python run_model_lstm.py")
            sys.exit(1)
        
        print(f"Loaded results for {len(results)} models")
        print()
        
        print("STEP 2: CREATING LEADERBOARD")
        print("-" * 40)
        
        leaderboard = create_leaderboard(results)
        
        print("🏆 MODEL PERFORMANCE LEADERBOARD 🏆")
        print("=" * 60)
        print(leaderboard.to_string(index=False))
        print()
        
        best_model = leaderboard.iloc[0]
        print(f"🥇 BEST PERFORMING MODEL: {best_model['model']}")
        print(f"   Average RMSE: {best_model['avg_rmse']:.4f}")
        print(f"   Average MAE: {best_model['avg_mae']:.4f}")
        print(f"   Average MAPE: {best_model['avg_mape']:.2f}%")
        print()
        
        leaderboard_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'model_leaderboard.csv')
        leaderboard.to_csv(leaderboard_file, index=False)
        print(f"Leaderboard saved to: {leaderboard_file}")
        print()
        
        print("STEP 3: GENERATING COMPARISON VISUALIZATIONS")
        print("-" * 40)
        
        plots_dir = OUTPUT_CONFIG['plots_dir']
        
        plot_model_comparison(
            results, 
            save_path=os.path.join(plots_dir, 'model_comparison.png')
        )
        
        plot_performance_heatmap(
            results,
            save_path=os.path.join(plots_dir, 'performance_heatmap.png')
        )
        
        print("STEP 4: DETAILED ANALYSIS")
        print("-" * 40)
        
        for model_name, df in results.items():
            if df is not None and len(df) > 0:
                print(f"\n{model_name.upper()} DETAILED STATS:")
                print(f"  Total predictions: {len(df):,}")
                print(f"  RMSE - Mean: {df['rmse'].mean():.4f}, Std: {df['rmse'].std():.4f}")
                print(f"  MAE  - Mean: {df['mae'].mean():.4f}, Std: {df['mae'].std():.4f}")
                print(f"  MAPE - Mean: {df['mape'].mean():.2f}%, Std: {df['mape'].std():.2f}%")
                
                best_horizon = df.loc[df['rmse'].idxmin()]
                worst_horizon = df.loc[df['rmse'].idxmax()]
                print(f"  Best horizon: {best_horizon['horizon']} (RMSE: {best_horizon['rmse']:.4f})")
                print(f"  Worst horizon: {worst_horizon['horizon']} (RMSE: {worst_horizon['rmse']:.4f})")
        
        print("\nSTEP 5: SUMMARY REPORT")
        print("-" * 40)
        
        summary_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'comparison_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("BITCOIN FORECASTING MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("LEADERBOARD:\n")
            f.write(leaderboard.to_string(index=False))
            f.write(f"\n\nBEST MODEL: {best_model['model']}\n")
            f.write(f"Average RMSE: {best_model['avg_rmse']:.4f}\n")
            f.write(f"Average MAE: {best_model['avg_mae']:.4f}\n")
            f.write(f"Average MAPE: {best_model['avg_mape']:.2f}%\n\n")
            f.write("FILES GENERATED:\n")
            f.write("- model_leaderboard.csv: Performance rankings\n")
            f.write("- model_comparison.png: Comparison visualizations\n")
            f.write("- performance_heatmap.png: Performance heatmaps\n")
        
        print(f"Summary report saved to: {summary_file}")
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Generated files:")
        print(f"- {leaderboard_file}")
        print(f"- {summary_file}")
        print("- Comparison plots in plots/ directory")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        print(f"ERROR: Comparison failed - {str(e)}")
        print("Check the log file 'model_comparison.log' for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
