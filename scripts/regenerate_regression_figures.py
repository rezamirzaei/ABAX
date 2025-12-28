#!/usr/bin/env python3
"""
Regenerate regression figures matching the notebook style.
Uses the same src modules as the notebook.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Setup style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'epa_fuel_economy.csv'

print("=" * 60)
print("Regenerating Regression Figures (Notebook Style)")
print("=" * 60)

# Load data
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded: {len(df)} samples from EPA dataset")
else:
    print(f"❌ Data file not found: {DATA_PATH}")
    sys.exit(1)

# Prepare features
target_col = 'comb08'
exclude_cols = [target_col]

# Get numeric columns only for simplicity
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

# Use more key features for better performance
key_features = ['year', 'cylinders', 'displ', 'city08', 'highway08', 'co2TailpipeGpm',
                'fuelCost08', 'youSaveSpend', 'charge120', 'charge240']
available_features = [c for c in key_features if c in df.columns]

# If not enough features, add more numeric columns
if len(available_features) < 5:
    for col in numeric_cols:
        if col not in available_features:
            available_features.append(col)
        if len(available_features) >= 10:
            break

print(f"   Features used: {len(available_features)} features")

X = df[available_features].fillna(0)
y = df[target_col].fillna(df[target_col].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# Train All Models
# ============================================================================
print("\n🎯 Training regression models...")

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=2000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
    'Huber': HuberRegressor(epsilon=1.35, max_iter=1000),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale'),
    'SVR (Linear)': SVR(kernel='linear', C=1.0),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'OLS': LinearRegression(),
}

results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        trained_models[name] = model

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({'Model': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae})
        print(f"   ✅ {name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    except Exception as e:
        print(f"   ❌ {name}: {e}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R²']
print(f"\n🏆 Best Model: {best_model_name} (R²={best_r2:.4f})")

# ============================================================================
# Figure 1: Model Comparison (Detailed)
# ============================================================================
print("\n🎨 Figure 1: Regressor Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R² Score
ax = axes[0]
sorted_r2 = results_df.sort_values('R²', ascending=True)
colors = plt.cm.RdYlGn(sorted_r2['R²'].values / sorted_r2['R²'].max())
bars = ax.barh(range(len(sorted_r2)), sorted_r2['R²'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_r2)))
ax.set_yticklabels(sorted_r2['Model'].values, fontsize=10)
ax.set_xlabel('R² Score', fontsize=12)
ax.set_title('R² Score (Higher = Better)', fontweight='bold', fontsize=13)
ax.set_xlim(0, 1.05)
for i, r2 in enumerate(sorted_r2['R²'].values):
    ax.text(r2 + 0.01, i, f'{r2:.3f}', va='center', fontsize=9)

# RMSE
ax = axes[1]
sorted_rmse = results_df.sort_values('RMSE', ascending=False)
colors = plt.cm.RdYlGn_r(sorted_rmse['RMSE'].values / sorted_rmse['RMSE'].max())
bars = ax.barh(range(len(sorted_rmse)), sorted_rmse['RMSE'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_rmse)))
ax.set_yticklabels(sorted_rmse['Model'].values, fontsize=10)
ax.set_xlabel('RMSE (MPG)', fontsize=12)
ax.set_title('RMSE (Lower = Better)', fontweight='bold', fontsize=13)
for i, rmse in enumerate(sorted_rmse['RMSE'].values):
    ax.text(rmse + 0.1, i, f'{rmse:.2f}', va='center', fontsize=9)

# MAE
ax = axes[2]
sorted_mae = results_df.sort_values('MAE', ascending=False)
colors = plt.cm.RdYlGn_r(sorted_mae['MAE'].values / sorted_mae['MAE'].max())
bars = ax.barh(range(len(sorted_mae)), sorted_mae['MAE'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_mae)))
ax.set_yticklabels(sorted_mae['Model'].values, fontsize=10)
ax.set_xlabel('MAE (MPG)', fontsize=12)
ax.set_title('MAE (Lower = Better)', fontweight='bold', fontsize=13)
for i, mae in enumerate(sorted_mae['MAE'].values):
    ax.text(mae + 0.1, i, f'{mae:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'regressor_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ regressor_comparison.png")

# ============================================================================
# Figure 2: Actual vs Predicted
# ============================================================================
print("\n🎨 Figure 2: Actual vs Predicted...")

y_pred_best = predictions[best_model_name]

fig, ax = plt.subplots(figsize=(9, 9))

# Scatter plot
scatter = ax.scatter(y_test, y_pred_best, alpha=0.5, c=y_test, cmap='viridis',
                     edgecolors='white', linewidth=0.3, s=40)

# Perfect prediction line
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')

# Add ±10% lines
ax.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g:', linewidth=1.5, alpha=0.7, label='±10% Error')
ax.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g:', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Actual MPG', fontsize=13)
ax.set_ylabel('Predicted MPG', fontsize=13)
ax.set_title(f'Actual vs Predicted Fuel Economy\n{best_model_name} (R² = {best_r2:.4f})',
             fontweight='bold', fontsize=14)
ax.legend(fontsize=11, loc='upper left')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Actual MPG', fontsize=11)

# Equal aspect ratio
ax.set_aspect('equal')
ax.set_xlim(min_val - 5, max_val + 5)
ax.set_ylim(min_val - 5, max_val + 5)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ actual_vs_predicted.png")

# ============================================================================
# Figure 3: Residual Analysis
# ============================================================================
print("\n🎨 Figure 3: Residual Analysis...")

residuals = y_test - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Predicted
ax = axes[0, 0]
ax.scatter(y_pred_best, residuals, alpha=0.5, c='#3498db', edgecolors='white', linewidth=0.3, s=40)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.axhline(y=np.std(residuals)*2, color='orange', linestyle=':', alpha=0.7, label='±2σ')
ax.axhline(y=-np.std(residuals)*2, color='orange', linestyle=':', alpha=0.7)
ax.set_xlabel('Predicted MPG', fontsize=12)
ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
ax.set_title('Residuals vs Predicted Values', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)

# 2. Residual Histogram
ax = axes[0, 1]
n, bins, patches = ax.hist(residuals, bins=40, color='#3498db', edgecolor='white',
                            linewidth=0.5, alpha=0.8, density=True)
# Fit normal distribution
from scipy import stats
mu, std = stats.norm.fit(residuals)
x = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal Fit\nμ={mu:.2f}, σ={std:.2f}')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual Value (MPG)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Residual Distribution', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)

# 3. Q-Q Plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold', fontsize=13)
ax.get_lines()[0].set_markerfacecolor('#3498db')
ax.get_lines()[0].set_markersize(5)
ax.get_lines()[1].set_color('red')
ax.get_lines()[1].set_linewidth(2)

# 4. Residuals vs Actual
ax = axes[1, 1]
ax.scatter(y_test, residuals, alpha=0.5, c='#9b59b6', edgecolors='white', linewidth=0.3, s=40)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Actual MPG', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Residuals vs Actual Values', fontweight='bold', fontsize=13)

# Add statistics box
stats_text = f'Mean: {np.mean(residuals):.3f}\nStd: {np.std(residuals):.3f}\nMax: {np.max(residuals):.2f}\nMin: {np.min(residuals):.2f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Residual Analysis - {best_model_name}', fontweight='bold', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'residuals.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ residuals.png")

# ============================================================================
# Figure 4: Feature Importance
# ============================================================================
print("\n🎨 Figure 4: Feature Importance...")

rf_model = trained_models.get('Random Forest')
if rf_model is not None:
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))

    bars = ax.barh(range(len(importance_df)), importance_df['Importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'].values, fontsize=11)
    ax.set_xlabel('Feature Importance (MDI)', fontsize=12)
    ax.set_title('Feature Importance for Fuel Economy Prediction\n(Random Forest)',
                 fontweight='bold', fontsize=14)

    # Add value labels
    for i, v in enumerate(importance_df['Importance'].values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance_regression.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("   ✅ feature_importance_regression.png")

# ============================================================================
# Figure 5: Prediction Intervals (Quantile-like visualization)
# ============================================================================
print("\n🎨 Figure 5: Prediction Intervals...")

# Sort by actual values for better visualization
sort_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sort_idx]
y_pred_sorted = y_pred_best[sort_idx]

# Estimate prediction intervals using residual std
residual_std = np.std(residuals)
lower_bound = y_pred_sorted - 1.645 * residual_std  # ~90% CI
upper_bound = y_pred_sorted + 1.645 * residual_std

# Sample for cleaner visualization
sample_size = min(200, len(y_test_sorted))
sample_idx = np.linspace(0, len(y_test_sorted)-1, sample_size, dtype=int)

fig, ax = plt.subplots(figsize=(14, 7))

# Plot prediction interval
ax.fill_between(range(sample_size), lower_bound[sample_idx], upper_bound[sample_idx],
                alpha=0.3, color='#3498db', label='90% Prediction Interval')
ax.plot(range(sample_size), y_pred_sorted[sample_idx], 'b-', linewidth=1.5, label='Predicted')
ax.scatter(range(sample_size), y_test_sorted[sample_idx], c='red', s=20, alpha=0.6,
           label='Actual', zorder=5)

# Calculate coverage
coverage = np.mean((y_test_sorted >= lower_bound) & (y_test_sorted <= upper_bound))

ax.set_xlabel('Samples (sorted by actual MPG)', fontsize=12)
ax.set_ylabel('MPG', fontsize=12)
ax.set_title(f'Prediction Intervals (Coverage: {coverage:.1%})', fontweight='bold', fontsize=14)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'prediction_intervals.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ prediction_intervals.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("✅ All regression figures regenerated successfully!")
print("=" * 60)
print(f"\nFigures saved to: {FIGURES_DIR}")
print(f"\nModel Results:")
print(results_df.to_string(index=False))

