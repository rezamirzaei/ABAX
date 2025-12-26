#!/usr/bin/env python
"""Test CNN implementation."""

import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')

# Test CNN
from src.models.cnn import SimpleCNN, train_cnn
from src.data import load_uah_driveset, split_data
from src.features import preprocess_features, encode_target

print('Loading data...')
dataset = load_uah_driveset('data/UAH-DRIVESET-v1')
split = split_data(dataset, test_size=0.2, stratify=True)
train_feat, test_feat = preprocess_features(split, scaler_type='robust')
y_train, y_test, encoder = encode_target(split.y_train, split.y_test)

print(f'Train shape: {train_feat.X.shape}')
print(f'Classes: {encoder.classes_}')

print('\nTraining CNN...')
trained = train_cnn(
    train_feat.X, y_train,
    X_val=test_feat.X, y_val=y_test,
    n_epochs=30,
    learning_rate=0.01,
)

print(f'\nCNN Training History:')
print(f'Iterations: {len(trained.history.iterations)}')
print(f'Train scores (first 5): {trained.history.train_scores[:5]}')
print(f'Val scores (first 5): {trained.history.val_scores[:5]}')
print(f'Train scores (last 3): {trained.history.train_scores[-3:]}')
print(f'Val scores (last 3): {trained.history.val_scores[-3:]}')

# Make predictions
preds = trained.model.predict(test_feat.X)
acc = np.mean(preds == y_test)
print(f'\nTest Accuracy: {acc:.4f}')

print('\n✅ CNN test passed!')
