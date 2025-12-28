"""
Script to regenerate corrupted figures for the LaTeX report.
Figures 6, 7, 8, 16, 18, 19, 20 need to be regenerated.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error

from src.visualization import setup_style
from src.data.splitter import split_by_driver

# Setup
setup_style()
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("Regenerating Corrupted Figures")
print("=" * 60)

# Load classification data
print("\n📊 Loading classification data...")
df_class = pd.read_csv("data/processed/uah_classification.csv")
feature_cols = [c for c in df_class.columns if c not in ['driver', 'behavior']]
X = df_class[feature_cols + ['driver']].copy()
y = df_class['behavior'].copy()

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = split_by_driver(X, y_enc, test_drivers=['D6'])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# Figure 6: classifier_comparison.png
# ============================================================================
print("\n🎨 Generating Figure 6: classifier_comparison.png...")

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic (L1)': LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42, class_weight='balanced'),
    'Logistic (L2)': LogisticRegression(penalty='l2', max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
}

results = []
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax.barh(y_pos - width/2, results_df['Accuracy'], width, label='Accuracy', color='#3498db', edgecolor='black')
bars2 = ax.barh(y_pos + width/2, results_df['F1 Score'], width, label='F1 Score', color='#2ecc71', edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('Score')
ax.set_title('Classifier Comparison (D6 Held-Out Test Set)', fontweight='bold', fontsize=14)
ax.legend(loc='lower right')
ax.set_xlim(0, 1.1)

# Add value labels
for bar in bars1:
    width_val = bar.get_width()
    ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2, f'{width_val:.3f}',
            ha='left', va='center', fontsize=9)
for bar in bars2:
    width_val = bar.get_width()
    ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2, f'{width_val:.3f}',
            ha='left', va='center', fontsize=9)

ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/classifier_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ classifier_comparison.png saved")

# ============================================================================
# Figure 7: confusion_matrix_classification.png
# ============================================================================
print("\n🎨 Generating Figure 7: confusion_matrix_classification.png...")

# Use Random Forest (best model)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Random Forest\n(D6 Held-Out Test Set)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/confusion_matrix_classification.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ confusion_matrix_classification.png saved")

# ============================================================================
# Figure 8: feature_importance_classification.png
# ============================================================================
print("\n🎨 Generating Figure 8: feature_importance_classification.png...")

importances = rf.feature_importances_
feature_names = feature_cols
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Feature Importance - Random Forest Classification', fontweight='bold', fontsize=14)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
            ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/feature_importance_classification.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ feature_importance_classification.png saved")

# ============================================================================
# Now load regression data
# ============================================================================
print("\n📊 Loading regression data...")
df_reg = pd.read_csv("data/processed/epa_fuel_economy.csv")

# Sample for efficiency
if len(df_reg) > 3000:
    df_reg = df_reg.sample(n=3000, random_state=42)

# Define target and features
target_col = 'comb08'
numeric_cols = ['year', 'cylinders', 'displ']
cat_cols = ['drive', 'VClass', 'fuelType', 'trany']

# Prepare features
X_reg = df_reg[numeric_cols + cat_cols].copy()
y_reg = df_reg[target_col].copy()

# Handle missing values
X_reg = X_reg.fillna(X_reg.mode().iloc[0])

# One-hot encode categorical features
X_reg_encoded = pd.get_dummies(X_reg, columns=cat_cols, drop_first=True)

# Train-test split
from sklearn.model_selection import train_test_split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_encoded, y_reg, test_size=0.2, random_state=42
)

# Scale
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# ============================================================================
# Figure 16: regressor_comparison.png
# ============================================================================
print("\n🎨 Generating Figure 16: regressor_comparison.png...")

regressors = {
    'Ridge (L2)': Ridge(alpha=1.0, random_state=42),
    'Lasso (L1)': Lasso(alpha=0.1, max_iter=2000, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

reg_results = []
for name, reg in regressors.items():
    reg.fit(X_train_reg_scaled, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg_scaled)
    r2 = r2_score(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    reg_results.append({'Model': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae})

reg_results_df = pd.DataFrame(reg_results).sort_values('R²', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² plot
y_pos = np.arange(len(reg_results_df))
bars = axes[0].barh(y_pos, reg_results_df['R²'], color='#3498db', edgecolor='black')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(reg_results_df['Model'])
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison - R² Score', fontweight='bold', fontsize=12)
axes[0].set_xlim(0.99, 1.001)

for bar in bars:
    width = bar.get_width()
    axes[0].text(width - 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                 ha='right', va='center', fontsize=10, color='white', fontweight='bold')

# RMSE plot
bars2 = axes[1].barh(y_pos, reg_results_df['RMSE'], color='#e74c3c', edgecolor='black')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(reg_results_df['Model'])
axes[1].set_xlabel('RMSE (MPG)')
axes[1].set_title('Model Comparison - RMSE', fontweight='bold', fontsize=12)

for bar in bars2:
    width = bar.get_width()
    axes[1].text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                 ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/regressor_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ regressor_comparison.png saved")

# ============================================================================
# Figure 18: feature_importance_regression.png
# ============================================================================
print("\n🎨 Generating Figure 18: feature_importance_regression.png...")

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg_scaled, y_train_reg)

importances_reg = rf_reg.feature_importances_
feature_names_reg = list(X_train_reg.columns)

# Get top 15 features
importance_df_reg = pd.DataFrame({
    'Feature': feature_names_reg,
    'Importance': importances_reg
}).sort_values('Importance', ascending=False).head(15).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df_reg)))
bars = ax.barh(importance_df_reg['Feature'], importance_df_reg['Importance'], color=colors, edgecolor='black')
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Feature Importance - Random Forest Regression (Top 15)', fontweight='bold', fontsize=14)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
            ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/feature_importance_regression.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ feature_importance_regression.png saved")

# ============================================================================
# Figure 19: residuals.png
# ============================================================================
print("\n🎨 Generating Figure 19: residuals.png...")

# Use Ridge model for predictions
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_reg_scaled, y_train_reg)
y_pred_ridge = ridge.predict(X_test_reg_scaled)
residuals = y_test_reg.values - y_pred_ridge

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted
axes[0].scatter(y_pred_ridge, residuals, alpha=0.5, c='#3498db', edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted MPG', fontsize=12)
axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[0].set_title('Residuals vs Predicted Values', fontweight='bold', fontsize=12)

# Add confidence bands
std_res = np.std(residuals)
axes[0].axhline(y=2*std_res, color='orange', linestyle=':', alpha=0.7, label=f'±2σ ({2*std_res:.2f})')
axes[0].axhline(y=-2*std_res, color='orange', linestyle=':', alpha=0.7)
axes[0].legend()

# Histogram of residuals
axes[1].hist(residuals, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[1].axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(residuals):.3f})')
axes[1].set_xlabel('Residual', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontweight='bold', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/residuals.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ residuals.png saved")

# ============================================================================
# Figure 20: prediction_intervals.png
# ============================================================================
print("\n🎨 Generating Figure 20: prediction_intervals.png...")

# Sort by actual values for better visualization
sort_idx = np.argsort(y_test_reg.values)
y_actual_sorted = y_test_reg.values[sort_idx]
y_pred_sorted = y_pred_ridge[sort_idx]

# Calculate prediction intervals (using residual std)
std_res = np.std(residuals)
lower_bound = y_pred_sorted - 1.96 * std_res
upper_bound = y_pred_sorted + 1.96 * std_res

fig, ax = plt.subplots(figsize=(12, 6))

# Sample for cleaner visualization
sample_size = min(100, len(y_actual_sorted))
sample_idx = np.linspace(0, len(y_actual_sorted)-1, sample_size, dtype=int)

x_range = np.arange(sample_size)
ax.fill_between(x_range, lower_bound[sample_idx], upper_bound[sample_idx],
                alpha=0.3, color='#3498db', label='95% Prediction Interval')
ax.plot(x_range, y_pred_sorted[sample_idx], 'b-', linewidth=2, label='Predicted')
ax.scatter(x_range, y_actual_sorted[sample_idx], c='#e74c3c', s=20, zorder=5,
           label='Actual', edgecolors='black', linewidth=0.5)

ax.set_xlabel('Sample Index (sorted by actual MPG)', fontsize=12)
ax.set_ylabel('MPG', fontsize=12)
ax.set_title('Prediction Intervals - Ridge Regression (95% CI)', fontweight='bold', fontsize=14)
ax.legend(loc='upper left')

# Add metrics
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))
r2 = r2_score(y_test_reg, y_pred_ridge)
ax.text(0.98, 0.02, f'R² = {r2:.4f}\nRMSE = {rmse:.3f} MPG',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/prediction_intervals.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ prediction_intervals.png saved")

print("\n" + "=" * 60)
print("✅ All corrupted figures regenerated successfully!")
print("=" * 60)
print("\nRegenerated figures:")
print("  - Figure 6: classifier_comparison.png")
print("  - Figure 7: confusion_matrix_classification.png")
print("  - Figure 8: feature_importance_classification.png")
print("  - Figure 16: regressor_comparison.png")
print("  - Figure 18: feature_importance_regression.png")
print("  - Figure 19: residuals.png")
print("  - Figure 20: prediction_intervals.png")

