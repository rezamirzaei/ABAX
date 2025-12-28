"""
Test script for PyTorch CNN Classifier on both processed and raw data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import CNN
from src.models.cnn import CNNClassifier, CNNClassifierRaw, plot_cnn_training_history
from src.data.splitter import split_by_driver

print('=' * 60)
print('Testing PyTorch CNN Classifier')
print('=' * 60)

# ============================================================================
# Test 1: Processed Data Classification
# ============================================================================
print('\n' + '-' * 60)
print('Test 1: CNN on Processed Classification Data')
print('-' * 60)

print('\n📊 Loading processed classification data...')
df = pd.read_csv('data/processed/uah_classification.csv')
feature_cols = [c for c in df.columns if c not in ['driver', 'behavior']]
X = df[feature_cols + ['driver']].copy()
y = df['behavior'].values

# Split
X_train, X_test, y_train, y_test = split_by_driver(X, y, test_drivers=['D6'])

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train CNN
print('\n🧠 Training CNN on processed data...')
cnn = CNNClassifier(
    n_filters=32,
    epochs=100,
    batch_size=8,
    learning_rate=0.001,
    early_stopping_patience=15,
    verbose=1
)
cnn.fit(X_train_scaled, y_train)

# Evaluate
y_pred = cnn.predict(X_test_scaled)
# y_test might be string labels, y_pred is also string labels from inverse_transform
if isinstance(y_test[0], str):
    acc = accuracy_score(y_test, y_pred)
else:
    acc = accuracy_score(y_test, cnn.le_.transform(y_pred))
print(f'\n📈 Test Accuracy (Processed Data): {acc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Save figure
history = cnn.get_training_history()
plot_cnn_training_history(history, save_path='results/figures/cnn_learning_curves_classification.png')
print('\n✅ Saved CNN learning curves to results/figures/')

# ============================================================================
# Test 2: Raw Data Classification
# ============================================================================
print('\n' + '-' * 60)
print('Test 2: CNN on Raw Features Data')
print('-' * 60)

print('\n📊 Loading raw features data...')
df_raw = pd.read_csv('data/processed/uah_raw_features.csv')

# Get feature columns (exclude driver, behavior, road_type)
raw_feature_cols = [c for c in df_raw.columns if c not in ['driver', 'behavior', 'road_type']]
X_raw = df_raw[raw_feature_cols + ['driver']].copy()
y_raw = df_raw['behavior'].values

# Split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_by_driver(X_raw, y_raw, test_drivers=['D6'])

# Scale
scaler_raw = StandardScaler()
X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
X_test_raw_scaled = scaler_raw.transform(X_test_raw)

# Train CNN (use more filters for higher-dimensional raw data)
print('\n🧠 Training CNN on raw features data...')
cnn_raw = CNNClassifier(
    n_filters=64,
    kernel_size=5,
    hidden_size=128,
    epochs=150,
    batch_size=8,
    learning_rate=0.0005,
    dropout=0.4,
    early_stopping_patience=20,
    verbose=1
)
cnn_raw.fit(X_train_raw_scaled, y_train_raw)

# Evaluate
y_pred_raw = cnn_raw.predict(X_test_raw_scaled)
if isinstance(y_test_raw[0], str):
    acc_raw = accuracy_score(y_test_raw, y_pred_raw)
else:
    acc_raw = accuracy_score(y_test_raw, cnn_raw.le_.transform(y_pred_raw))
print(f'\n📈 Test Accuracy (Raw Features): {acc_raw:.4f}')
print('\nClassification Report:')
print(classification_report(y_test_raw, y_pred_raw))

# Save figure
history_raw = cnn_raw.get_training_history()
plot_cnn_training_history(history_raw, save_path='results/figures/cnn_learning_curves_raw.png')
print('\n✅ Saved CNN raw data learning curves to results/figures/')

# ============================================================================
# Summary
# ============================================================================
print('\n' + '=' * 60)
print('Summary')
print('=' * 60)
print(f'\n📊 Results:')
print(f'  - Processed Data CNN Accuracy: {acc:.4f}')
print(f'  - Raw Features CNN Accuracy:   {acc_raw:.4f}')
print(f'\n📁 Saved Figures:')
print(f'  - results/figures/cnn_learning_curves_classification.png')
print(f'  - results/figures/cnn_learning_curves_raw.png')
print('\n✅ All CNN tests completed successfully!')

