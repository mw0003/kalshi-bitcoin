import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

class ForecastVisualizer:
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the forecast visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_data_overview(self, df, save_path=None):
        """
        Plot overview of the Bitcoin price data
        
        Args:
            df: DataFrame with OHLCV data
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(df.index, df['close'], linewidth=1)
        axes[0, 0].set_title('Bitcoin Price Over Time')
        axes[0, 0].set_ylabel('Price (USDT)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df.index, df['volume'], color='orange', linewidth=1)
        axes[0, 1].set_title('Trading Volume Over Time')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(df['close'], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Price Distribution')
        axes[1, 0].set_xlabel('Price (USDT)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        price_changes = df['close'].pct_change().dropna()
        axes[1, 1].hist(price_changes, bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Price Changes Distribution')
        axes[1, 1].set_xlabel('Price Change (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data overview plot saved to {save_path}")
        
        plt.close()
        
    def plot_feature_importance(self, importance_df, save_path=None):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
    def plot_predictions_vs_actual(self, y_test, predictions, horizons_to_plot=[1, 5, 10, 20], save_path=None):
        """
        Plot predictions vs actual values for selected horizons
        
        Args:
            y_test: Actual values
            predictions: Predicted values
            horizons_to_plot: List of horizons to plot
            save_path: Path to save the plot
        """
        n_plots = len(horizons_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, horizon in enumerate(horizons_to_plot):
            if i >= len(axes):
                break
                
            horizon_idx = horizon - 1  # Convert to 0-based index
            
            if horizon_idx < predictions.shape[1]:
                y_true = y_test.iloc[:, horizon_idx]
                y_pred = predictions[:, horizon_idx]
                
                time_idx = y_test.index[:len(y_pred)]
                
                axes[i].plot(time_idx, y_true[:len(y_pred)], label='Actual', alpha=0.7, linewidth=1)
                axes[i].plot(time_idx, y_pred, label='Predicted', alpha=0.7, linewidth=1)
                axes[i].set_title(f'Horizon {horizon} Minutes Ahead')
                axes[i].set_ylabel('Price (USDT)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions vs actual plot saved to {save_path}")
        
        plt.close()
        
    def plot_error_metrics(self, metrics_df, save_path=None):
        """
        Plot error metrics across horizons
        
        Args:
            metrics_df: DataFrame with evaluation metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(metrics_df['horizon'], metrics_df['rmse'], marker='o', linewidth=2, markersize=6)
        axes[0].set_title('RMSE Across Horizons')
        axes[0].set_xlabel('Forecast Horizon (minutes)')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(metrics_df['horizon'], metrics_df['mae'], marker='s', color='orange', linewidth=2, markersize=6)
        axes[1].set_title('MAE Across Horizons')
        axes[1].set_xlabel('Forecast Horizon (minutes)')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(metrics_df['horizon'], metrics_df['mape'], marker='^', color='green', linewidth=2, markersize=6)
        axes[2].set_title('MAPE Across Horizons')
        axes[2].set_xlabel('Forecast Horizon (minutes)')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error metrics plot saved to {save_path}")
        
        plt.close()
        
    def plot_residuals(self, y_test, predictions, horizons_to_plot=[1, 5, 10, 20], save_path=None):
        """
        Plot residuals for selected horizons
        
        Args:
            y_test: Actual values
            predictions: Predicted values
            horizons_to_plot: List of horizons to plot
            save_path: Path to save the plot
        """
        n_plots = len(horizons_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, horizon in enumerate(horizons_to_plot):
            if i >= len(axes):
                break
                
            horizon_idx = horizon - 1
            
            if horizon_idx < predictions.shape[1]:
                y_true = y_test.iloc[:, horizon_idx]
                y_pred = predictions[:, horizon_idx]
                residuals = y_true[:len(y_pred)] - y_pred
                
                axes[i].scatter(y_pred, residuals, alpha=0.6, s=20)
                axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
                axes[i].set_title(f'Residuals - Horizon {horizon} Minutes')
                axes[i].set_xlabel('Predicted Values')
                axes[i].set_ylabel('Residuals')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residuals plot saved to {save_path}")
        
        plt.close()
        
    def plot_walk_forward_results(self, wf_results, save_path=None):
        """
        Plot walk-forward validation results
        
        Args:
            wf_results: List of validation results
            save_path: Path to save the plot
        """
        all_metrics = pd.concat(wf_results, ignore_index=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        pivot_rmse = all_metrics.pivot(index='horizon', columns='split', values='rmse')
        sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('RMSE Across Splits and Horizons')
        axes[0].set_xlabel('Validation Split')
        axes[0].set_ylabel('Forecast Horizon (minutes)')
        
        avg_rmse = all_metrics.groupby('horizon')['rmse'].mean()
        axes[1].plot(avg_rmse.index, avg_rmse.values, marker='o', linewidth=2, markersize=6)
        axes[1].set_title('Average RMSE Across Horizons (Walk-Forward)')
        axes[1].set_xlabel('Forecast Horizon (minutes)')
        axes[1].set_ylabel('Average RMSE')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Walk-forward results plot saved to {save_path}")
        
        plt.close()
    
    def plot_model_comparison_summary(self, leaderboard, save_path=None):
        """
        Plot model comparison summary from leaderboard
        
        Args:
            leaderboard: DataFrame with model performance comparison
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison Summary', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(leaderboard)]
        
        axes[0].bar(leaderboard['model'], leaderboard['avg_rmse'], color=colors)
        axes[0].set_title('Average RMSE by Model')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(leaderboard['model'], leaderboard['avg_mae'], color=colors)
        axes[1].set_title('Average MAE by Model')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(leaderboard['model'], leaderboard['avg_mape'], color=colors)
        axes[2].set_title('Average MAPE by Model')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison summary saved to {save_path}")
        
        plt.close()

if __name__ == "__main__":
    from data_ingestion import BitcoinDataIngester
    from feature_engineering import FeatureEngineer
    from model_training import MultiHorizonForecaster
    
    ingester = BitcoinDataIngester()
    try:
        data = ingester.load_data()
    except:
        data = ingester.fetch_historical_data(years_back=2)
        ingester.save_data(data)
    
    visualizer = ForecastVisualizer()
    
    visualizer.plot_data_overview(data, save_path='data_overview.png')
    
    print("Visualization example completed!")
