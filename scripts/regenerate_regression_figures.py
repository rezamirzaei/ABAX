#!/usr/bin/env python
"""
Regenerate regression figures for the ABAX Technical Report.

This script regenerates figures 16, 18, 19, 20 which appear corrupted:
- regressor_comparison.png
- feature_importance_regression.png
- residuals.png
- prediction_intervals.png
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Ensure we're in the right directory
os.chdir('/Users/rezami/PycharmProjects/ABAX')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set style for nice figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

FIGURES_DIR = 'results/figures'

def load_data():
    """Load the EPA fuel economy dataset."""
    df = pd.read_csv('data/processed/epa_fuel_economy.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)[:10]}...")  # First 10 columns
    return df

def prepare_data(df):
    """Prepare features and target for regression with proper encoding."""
    # Target column
    target_col = 'comb08'
    
    print(f"Target column: {target_col}")
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Select useful features
    numeric_features = ['year', 'cylinders', 'displ']
    categorical_features = ['VClass', 'drive', 'fuelType', 'make']
    
    # Filter to available columns
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Prepare X and y
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown')
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    print(f"Total features after encoding: {X_encoded.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Scale numeric features only
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Return feature names for importance plot
    feature_names = list(X_encoded.columns)
    
    return X_train_scaled.values, X_test_scaled.values, y_train.values, y_test.values, feature_names

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple regressors and collect results."""
    models = {
        'Ridge (L2)': Ridge(alpha=1.0),
        'OLS (baseline)': LinearRegression(),
        'Huber (robust)': HuberRegressor(epsilon=1.35, max_iter=500),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Lasso (L1)': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        })
        trained_models[name] = model
        predictions[name] = y_pred
        print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}")

    return pd.DataFrame(results), trained_models, predictions

def plot_regressor_comparison(results_df):
    """Figure 16: Regressor comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort by R²
    sorted_df = results_df.sort_values('R²', ascending=True)

    # R² plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df)))
    bars = axes[0].barh(sorted_df['Model'], sorted_df['R²'], color=colors)
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('R² Score (higher is better)', fontsize=11, fontweight='bold')
    axes[0].set_xlim(0.9, 1.0)
    for bar, val in zip(bars, sorted_df['R²']):
        axes[0].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

    # RMSE plot
    sorted_df_rmse = results_df.sort_values('RMSE', ascending=False)
    bars = axes[1].barh(sorted_df_rmse['Model'], sorted_df_rmse['RMSE'], color='#e74c3c', alpha=0.8)
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('RMSE (lower is better)', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, sorted_df_rmse['RMSE']):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    # MAPE plot
    sorted_df_mape = results_df.sort_values('MAPE', ascending=False)
    bars = axes[2].barh(sorted_df_mape['Model'], sorted_df_mape['MAPE'], color='#3498db', alpha=0.8)
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('MAPE (lower is better)', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, sorted_df_mape['MAPE']):
        axes[2].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=9)

    plt.suptitle('Regression Model Comparison (EPA Fuel Economy)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/regressor_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: regressor_comparison.png")

def plot_feature_importance(model, feature_names):
    """Figure 18: Feature importance from Random Forest (top 15)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Show only top 15 features
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(top_n), importances[top_indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Feature Importance (Random Forest Regression)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, importances[top_indices]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance_regression.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: feature_importance_regression.png")

def plot_residuals(y_test, y_pred):
    """Figure 19: Residual plot."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, c='#3498db', edgecolors='white', linewidth=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted MPG')
    axes[0].set_ylabel('Residuals (Actual - Predicted)')
    axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Add bounds
    std_res = np.std(residuals)
    axes[0].axhline(y=2*std_res, color='orange', linestyle=':', alpha=0.7, label=f'±2σ ({2*std_res:.2f})')
    axes[0].axhline(y=-2*std_res, color='orange', linestyle=':', alpha=0.7)
    axes[0].legend()

    # Residual distribution
    axes[1].hist(residuals, bins=30, color='#2ecc71', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].axvline(x=np.mean(residuals), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals):.3f}')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.suptitle('Residual Analysis (Ridge Regression)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residuals.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: residuals.png")

def plot_prediction_intervals(y_test, y_pred):
    """Figure 20: Prediction intervals."""
    # Sort by predicted value for better visualization
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    y_test_sorted = y_test[sorted_idx]

    # Calculate residual std for prediction intervals
    residuals = y_test - y_pred
    std_res = np.std(residuals)

    # Sample for cleaner visualization
    n_samples = min(200, len(y_pred))
    sample_idx = np.linspace(0, len(y_pred_sorted)-1, n_samples, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_range = np.arange(n_samples)

    # Prediction intervals (95%)
    lower = y_pred_sorted[sample_idx] - 1.96 * std_res
    upper = y_pred_sorted[sample_idx] + 1.96 * std_res

    ax.fill_between(x_range, lower, upper, alpha=0.3, color='#3498db', label='95% Prediction Interval')
    ax.plot(x_range, y_pred_sorted[sample_idx], 'b-', linewidth=2, label='Predicted MPG')
    ax.scatter(x_range, y_test_sorted[sample_idx], c='#e74c3c', s=20, alpha=0.6, label='Actual MPG', zorder=5)

    ax.set_xlabel('Sample Index (sorted by predicted MPG)')
    ax.set_ylabel('MPG')
    ax.set_title('Prediction Intervals with Actual Values', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add coverage statistic
    coverage = np.mean((y_test >= y_pred - 1.96*std_res) & (y_test <= y_pred + 1.96*std_res))
    ax.text(0.98, 0.02, f'Coverage: {coverage:.1%}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/prediction_intervals.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: prediction_intervals.png")

def plot_actual_vs_predicted(y_test, y_pred):
    """Bonus: Actual vs Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.5, c='#3498db', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title('Actual vs Predicted MPG (Ridge Regression)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/actual_vs_predicted.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: actual_vs_predicted.png")

def main():
    print("=" * 60)
    print("Regenerating Regression Figures (16, 18, 19, 20)")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # Train models
    results_df, trained_models, predictions = train_models(X_train, y_train, X_test, y_test)

    # Get predictions from Ridge (best model)
    ridge_pred = predictions['Ridge (L2)']
    rf_model = trained_models['Random Forest']

    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)

    # Generate all figures
    plot_regressor_comparison(results_df)
    plot_feature_importance(rf_model, feature_names)
    plot_residuals(y_test, ridge_pred)
    plot_prediction_intervals(y_test, ridge_pred)
    plot_actual_vs_predicted(y_test, ridge_pred)

    print("\n" + "=" * 60)
    print("All regression figures regenerated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

