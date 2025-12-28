"""Generate all figures for the LaTeX report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data import get_all_trips, build_raw_dataset, split_by_driver, load_raw_accelerometer, load_raw_gps, compute_acceleration_magnitude
from src.models import CNNClassifier, plot_cnn_training_history
from src.visualization import setup_style

setup_style()
FIGURES_DIR = Path('results/figures')

print("=" * 60)
print("Generating All Figures for LaTeX Report")
print("=" * 60)

# 1. Load data
data_dir = Path('data/UAH-DRIVESET-v1')
trips = get_all_trips(data_dir)
print(f"\n📊 Found {len(trips)} trips")

# Build dataset
raw_df = build_raw_dataset(trips)
print(f"Dataset: {raw_df.shape}")

# Prepare features
feature_cols = [c for c in raw_df.columns if c not in ['driver', 'behavior', 'road_type']]
X = raw_df[feature_cols + ['driver']].copy()
X[feature_cols] = X[feature_cols].fillna(0)
y = raw_df['behavior'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

# Split
X_train, X_test, y_train, y_test = split_by_driver(X, y_enc, test_drivers=['D6'])

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# Figure 1: Raw Accelerometer Data
# ============================================================================
print("\n🎨 Generating raw_accelerometer_data.png...")
sample_trip = [t for t in trips if t['behavior'] == 'AGGRESSIVE'][0]
acc = load_raw_accelerometer(sample_trip['path'])
gps = load_raw_gps(sample_trip['path'])

if acc is not None:
    acc['acc_magnitude'] = compute_acceleration_magnitude(
        acc['acc_x_kf'].values, acc['acc_y_kf'].values, acc['acc_z_kf'].values
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(acc['timestamp'][:500], acc['acc_x'][:500], alpha=0.4, label='Raw', color='lightblue')
    ax.plot(acc['timestamp'][:500], acc['acc_x_kf'][:500], label='Kalman Filtered', color='blue', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('X-Axis: Longitudinal (Braking/Acceleration)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[0, 1]
    ax.plot(acc['timestamp'][:500], acc['acc_y'][:500], alpha=0.4, label='Raw', color='lightgreen')
    ax.plot(acc['timestamp'][:500], acc['acc_y_kf'][:500], label='Kalman Filtered', color='green', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Y-Axis: Lateral (Turning)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 0]
    ax.plot(acc['timestamp'][:500], acc['acc_magnitude'][:500], color='purple', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Gravity baseline')
    ax.set_title('Acceleration Magnitude', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Acceleration| (g)')
    ax.legend(loc='upper right', fontsize=8)

    if gps is not None:
        ax = axes[1, 1]
        ax.plot(gps['timestamp'], gps['speed'], color='navy', linewidth=1)
        ax.set_title('Speed (from GPS)', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'raw_accelerometer_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ raw_accelerometer_data.png saved")

# ============================================================================
# Figure 2: Class Distribution
# ============================================================================
print("\n🎨 Generating class_distribution.png...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
behavior_counts = raw_df['behavior'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']
bars = axes[0].bar(behavior_counts.index, behavior_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_xlabel('Behavior')
axes[0].set_ylabel('Count')
for bar, count in zip(bars, behavior_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(count), ha='center', fontsize=10)
axes[1].pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Class Proportions', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ class_distribution.png saved")

# ============================================================================
# Figure 3: Feature Distributions
# ============================================================================
print("\n🎨 Generating feature_distributions_classification.png...")
key_features = ['speed_mean', 'speed_std', 'acc_magnitude_std', 'jerk_x_std',
                'hard_brake_count', 'sharp_turn_count']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.flatten(), key_features):
    for behavior in ['NORMAL', 'DROWSY', 'AGGRESSIVE']:
        data = raw_df[raw_df['behavior'] == behavior][feat]
        ax.hist(data, alpha=0.5, label=behavior, bins=10)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel(feat)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_distributions_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ feature_distributions_classification.png saved")

# ============================================================================
# Figure 4: Driver Behavior Distribution
# ============================================================================
print("\n🎨 Generating driver_behavior_distribution.png...")
crosstab = pd.crosstab(raw_df['driver'], raw_df['behavior'])
fig, ax = plt.subplots(figsize=(10, 5))
crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
ax.set_title('Trips per Driver by Behavior', fontweight='bold')
ax.set_xlabel('Driver')
ax.set_ylabel('Number of Trips')
ax.legend(title='Behavior')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'driver_behavior_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ driver_behavior_distribution.png saved")

# ============================================================================
# Figure 5: Correlation Matrix
# ============================================================================
print("\n🎨 Generating correlation_matrix_classification.png...")
corr_matrix = raw_df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_matrix_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ correlation_matrix_classification.png saved")

# ============================================================================
# Train classifiers and generate comparison figures
# ============================================================================
print("\n🎯 Training classifiers...")

classifiers = {
    'Logistic (L2)': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
    'Logistic (L1)': LogisticRegression(penalty='l1', solver='saga', max_iter=2000, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
}

results = []
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    results.append({'Model': name, 'Train Acc': train_acc, 'Test Acc': acc, 'F1-Score': f1})
    print(f"   {name}: Train={train_acc:.3f}, Test={acc:.3f}")

# ============================================================================
# Figure 6: Classifier Comparison
# ============================================================================
print("\n🎨 Generating classifier_comparison.png...")
comparison_df = pd.DataFrame(results).sort_values('Test Acc', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
colors = plt.cm.RdYlGn(comparison_df['Test Acc'].values / comparison_df['Test Acc'].max())
bars = ax.barh(range(len(comparison_df)), comparison_df['Test Acc'].values, color=colors)
ax.set_yticks(range(len(comparison_df)))
ax.set_yticklabels(comparison_df['Model'].values)
ax.set_xlabel('Test Accuracy')
ax.set_title('Model Comparison - Test Accuracy (D6 Held Out)', fontweight='bold')
ax.set_xlim(0, 1)
for i, acc in enumerate(comparison_df['Test Acc']):
    ax.text(acc + 0.02, i, f'{acc:.3f}', va='center', fontsize=9)

ax = axes[1]
x = np.arange(len(comparison_df))
width = 0.35
ax.barh(x - width/2, comparison_df['Train Acc'].values, width, label='Train', color='#3498db')
ax.barh(x + width/2, comparison_df['Test Acc'].values, width, label='Test', color='#e74c3c')
ax.set_yticks(x)
ax.set_yticklabels(comparison_df['Model'].values)
ax.set_xlabel('Accuracy')
ax.set_title('Train vs Test Accuracy', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'classifier_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ classifier_comparison.png saved")

# ============================================================================
# Figure 7: Confusion Matrix (Using Logistic Regression - best model)
# ============================================================================
print("\n🎨 Generating confusion_matrix_classification.png...")
lr = classifiers['Logistic (L2)']
y_pred_best = lr.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
            yticklabels=classes, ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Logistic Regression (D6 Held Out)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrix_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ confusion_matrix_classification.png saved")

# ============================================================================
# Figure 8: Feature Importance (Using Logistic Regression coefficients)
# ============================================================================
print("\n🎨 Generating feature_importance_classification.png...")
# Use absolute values of coefficients as importance
# For multiclass, average across classes
coef_importance = np.abs(lr.coef_).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': coef_importance
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top_n = importance_df.head(15).sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_n)))
ax.barh(range(len(top_n)), top_n['Importance'].values, color=colors)
ax.set_yticks(range(len(top_n)))
ax.set_yticklabels(top_n['Feature'].values)
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Features (Logistic Regression Coefficients)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ feature_importance_classification.png saved")

# ============================================================================
# Figure 9: CNN Training
# ============================================================================
print("\n🧠 Training CNN...")
cnn = CNNClassifier(
    n_filters=64, kernel_size=5, hidden_size=128, dropout=0.4,
    epochs=100, batch_size=8, learning_rate=0.0005,
    early_stopping_patience=15, verbose=1, random_state=42
)
cnn.fit(X_train_scaled, y_train)

history = cnn.get_training_history()
plot_cnn_training_history(history, save_path=str(FIGURES_DIR / 'cnn_learning_curves_classification.png'))
print("   ✅ cnn_learning_curves_classification.png saved")

print("\n" + "=" * 60)
print("✅ All figures generated successfully!")
print("=" * 60)

