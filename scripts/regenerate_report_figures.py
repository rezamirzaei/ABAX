#!/usr/bin/env python3
"""
Regenerate all figures for the ABAX Technical Report.
This script creates clean, high-quality figures for the LaTeX document.
"""
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Regenerating All Figures for ABAX Technical Report")
print("=" * 60)

# Import project modules
from src.classification import (
    get_all_trips, load_raw_gps, load_raw_accelerometer,
    compute_acceleration_magnitude, load_or_build_dataset,
    prepare_classification_data, get_all_classifiers,
    train_all_classifiers, get_best_model, get_feature_importance,
    results_to_dataframe,
)
from sklearn.metrics import confusion_matrix

# ============================================================================
# Load Data
# ============================================================================
print("\n📂 Loading data...")
DATA_DIR = PROJECT_ROOT / 'data' / 'UAH-DRIVESET-v1'
CACHE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'uah_raw_features.csv'

df = load_or_build_dataset(data_dir=DATA_DIR, cache_path=CACHE_PATH)
feature_cols = [c for c in df.columns if c not in ['driver', 'behavior', 'road_type']]
print(f"   Dataset: {df.shape[0]} trips, {len(feature_cols)} features")

# Prepare data
data = prepare_classification_data(df, test_drivers=['D6'])
print(f"   Train: {data.X_train.shape[0]}, Test: {data.X_test.shape[0]}")

# ============================================================================
# Figure 1: Raw Accelerometer Data
# ============================================================================
print("\n🎨 Figure 1: Raw Accelerometer Data...")
trips = get_all_trips(DATA_DIR)
sample_trip = next((t for t in trips if t.behavior.upper() == 'AGGRESSIVE'), trips[0])
acc = load_raw_accelerometer(sample_trip.path)
gps = load_raw_gps(sample_trip.path)

if acc is not None:
    acc['acc_magnitude'] = compute_acceleration_magnitude(
        acc['acc_x_kf'].values, acc['acc_y_kf'].values, acc['acc_z_kf'].values
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    n_samples = 500

    # X-axis (Braking/Acceleration)
    ax = axes[0, 0]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_x'][:n_samples],
            alpha=0.4, label='Raw', color='lightblue', linewidth=0.8)
    ax.plot(acc['timestamp'][:n_samples], acc['acc_x_kf'][:n_samples],
            label='Kalman Filtered', color='#2980b9', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.3, color='#e74c3c', linestyle=':', alpha=0.7, label='Brake threshold')
    ax.set_title('X-Axis: Longitudinal (Braking/Acceleration)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(acc['timestamp'].iloc[0], acc['timestamp'].iloc[min(n_samples, len(acc)-1)])

    # Y-axis (Turning)
    ax = axes[0, 1]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_y'][:n_samples],
            alpha=0.4, label='Raw', color='lightgreen', linewidth=0.8)
    ax.plot(acc['timestamp'][:n_samples], acc['acc_y_kf'][:n_samples],
            label='Kalman Filtered', color='#27ae60', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='#f39c12', linestyle=':', alpha=0.7, label='Turn threshold')
    ax.axhline(y=-0.3, color='#f39c12', linestyle=':', alpha=0.7)
    ax.set_title('Y-Axis: Lateral (Turning)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(acc['timestamp'].iloc[0], acc['timestamp'].iloc[min(n_samples, len(acc)-1)])

    # Magnitude
    ax = axes[1, 0]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_magnitude'][:n_samples],
            color='#9b59b6', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Gravity baseline')
    ax.fill_between(acc['timestamp'][:n_samples], 1, acc['acc_magnitude'][:n_samples],
                    alpha=0.3, color='#9b59b6')
    ax.set_title('Acceleration Magnitude', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Acceleration| (g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(acc['timestamp'].iloc[0], acc['timestamp'].iloc[min(n_samples, len(acc)-1)])

    # Speed from GPS
    if gps is not None and len(gps) > 0:
        ax = axes[1, 1]
        ax.plot(gps['timestamp'], gps['speed'], color='#2c3e50', linewidth=1.2)
        ax.fill_between(gps['timestamp'], 0, gps['speed'], alpha=0.3, color='#3498db')
        ax.set_title('Speed (from GPS)', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        mean_speed = gps['speed'].mean()
        ax.axhline(y=mean_speed, color='#e74c3c', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_speed:.1f} km/h')
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'raw_accelerometer_data.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("   ✅ raw_accelerometer_data.png")

# ============================================================================
# Figure 2: Class Distribution
# ============================================================================
print("\n🎨 Figure 2: Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

behavior_counts = df['behavior'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']

# Bar plot
ax = axes[0]
bars = ax.bar(behavior_counts.index, behavior_counts.values, color=colors, edgecolor='black', linewidth=1.2)
ax.set_title('Class Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Behavior', fontsize=12)
ax.set_ylabel('Number of Trips', fontsize=12)
for bar, count in zip(bars, behavior_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{count}\n({100*count/len(df):.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(behavior_counts.values) * 1.2)

# Pie chart
ax = axes[1]
wedges, texts, autotexts = ax.pie(behavior_counts.values, labels=behavior_counts.index,
                                   autopct='%1.1f%%', colors=colors, startangle=90,
                                   explode=(0.02, 0.02, 0.02), shadow=True)
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax.set_title('Class Proportions', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ class_distribution.png")

# ============================================================================
# Figure 3: Feature Distributions
# ============================================================================
print("\n🎨 Figure 3: Feature Distributions...")
key_features = ['speed_mean', 'speed_std', 'acc_magnitude_std',
                'jerk_x_std', 'hard_brake_count', 'sharp_turn_count']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
behavior_colors = {'NORMAL': '#2ecc71', 'DROWSY': '#f39c12', 'AGGRESSIVE': '#e74c3c'}

for ax, feat in zip(axes.flatten(), key_features):
    for behavior in ['NORMAL', 'DROWSY', 'AGGRESSIVE']:
        data_subset = df[df['behavior'] == behavior][feat].dropna()
        if len(data_subset) > 0:
            ax.hist(data_subset, alpha=0.6, label=behavior, bins=8,
                    color=behavior_colors[behavior], edgecolor='white', linewidth=0.5)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=12)
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_distributions_classification.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ feature_distributions_classification.png")

# ============================================================================
# Figure 4: Driver Behavior Distribution
# ============================================================================
print("\n🎨 Figure 4: Driver Behavior Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

crosstab = pd.crosstab(df['driver'], df['behavior'])
crosstab = crosstab[['NORMAL', 'DROWSY', 'AGGRESSIVE']]  # Consistent order
crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'],
              edgecolor='black', linewidth=1)
ax.set_title('Trips per Driver by Behavior', fontweight='bold', fontsize=14)
ax.set_xlabel('Driver', fontsize=12)
ax.set_ylabel('Number of Trips', fontsize=12)
ax.legend(title='Behavior', fontsize=10)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'driver_behavior_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ driver_behavior_distribution.png")

# ============================================================================
# Figure 5: Correlation Matrix
# ============================================================================
print("\n🎨 Figure 5: Correlation Matrix...")
fig, ax = plt.subplots(figsize=(14, 12))

corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_matrix_classification.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ correlation_matrix_classification.png")

# ============================================================================
# Figure 6: Train Models and Comparison
# ============================================================================
print("\n🎨 Figure 6: Training models and comparison...")
classifiers = get_all_classifiers()
results = train_all_classifiers(data, classifiers, verbose=False)
comparison_df = results_to_dataframe(results)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Sort by test accuracy
sorted_results = sorted(results, key=lambda x: x.test_accuracy, reverse=True)
names = [r.model_name for r in sorted_results]
test_accs = [r.test_accuracy for r in sorted_results]
train_accs = [r.train_accuracy for r in sorted_results]

# Test Accuracy
ax = axes[0]
colors = plt.cm.RdYlGn(np.array(test_accs) / max(test_accs))
bars = ax.barh(range(len(names)), test_accs, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Test Accuracy', fontsize=12)
ax.set_title('Model Comparison - Test Accuracy\n(D6 Held Out)', fontweight='bold', fontsize=13)
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random guess')
ax.set_xlim(0, 1.0)
for i, acc in enumerate(test_accs):
    ax.text(acc + 0.02, i, f'{acc:.3f}', va='center', fontsize=9)

# Train vs Test
ax = axes[1]
x = np.arange(len(names))
width = 0.35
bars1 = ax.barh(x - width/2, train_accs, width, label='Train', color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax.barh(x + width/2, test_accs, width, label='Test', color='#e74c3c', edgecolor='black', linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Train vs Test Accuracy\n(Overfitting Check)', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 1.1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'classifier_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ classifier_comparison.png")

# ============================================================================
# Figure 7: Confusion Matrix
# ============================================================================
print("\n🎨 Figure 7: Confusion Matrix...")
best = get_best_model(results)
cm = confusion_matrix(data.y_test, best.predictions)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.class_names, yticklabels=data.class_names,
            ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix - {best.model_name}\n(Test Accuracy: {best.test_accuracy:.1%})',
             fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrix_classification.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ confusion_matrix_classification.png")

# ============================================================================
# Figure 8: Feature Importance
# ============================================================================
print("\n🎨 Figure 8: Feature Importance...")
importance_df = get_feature_importance(best.model, data.feature_names)

fig, ax = plt.subplots(figsize=(10, 8))
df_sorted = importance_df.sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))

ax.barh(range(len(df_sorted)), df_sorted['Importance'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['Feature'].values, fontsize=10)
ax.set_xlabel('Importance (|Coefficient|)', fontsize=12)
ax.set_title(f'Top {len(df_sorted)} Features ({best.model_name})', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance_classification.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ feature_importance_classification.png")

# ============================================================================
# Figure 9: Behavior Comparison
# ============================================================================
print("\n🎨 Figure 9: Behavior Comparison...")
comparison_features = ['speed_mean', 'speed_std', 'acc_magnitude_std',
                       'hard_brake_count', 'sharp_turn_count', 'jerk_x_std']

behavior_stats = df.groupby('behavior')[comparison_features].mean()
behavior_stats = behavior_stats.reindex(['NORMAL', 'DROWSY', 'AGGRESSIVE'])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
colors = ['#2ecc71', '#f39c12', '#e74c3c']

for ax, feat in zip(axes.flatten(), comparison_features):
    if feat in behavior_stats.columns:
        bars = ax.bar(behavior_stats.index, behavior_stats[feat], color=colors,
                      edgecolor='black', linewidth=1.2)
        ax.set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'behavior_comparison_raw.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✅ behavior_comparison_raw.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("✅ All figures regenerated successfully!")
print("=" * 60)
print(f"\nFigures saved to: {FIGURES_DIR}")
print(f"\nBest Model: {best.model_name}")
print(f"Test Accuracy: {best.test_accuracy:.1%}")
print(f"Train Accuracy: {best.train_accuracy:.1%}")

