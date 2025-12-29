#!/usr/bin/env python3
"""Test script for ResNet classifier."""

import sys
sys.path.insert(0, '/Users/rezami/PycharmProjects/ABAX')

from src.models.resnet import ResNetClassifier, plot_resnet_training_history
import numpy as np

# Quick test
print("Testing ResNet classifier...")
X = np.random.randn(30, 20).astype(np.float32)
y = np.array(['NORMAL', 'DROWSY', 'AGGRESSIVE'] * 10)

clf = ResNetClassifier(n_filters=16, n_blocks=2, epochs=10, verbose=1, random_state=42)
clf.fit(X, y)
pred = clf.predict(X[:5])
acc = clf.score(X, y)

print(f'\nPredictions: {list(pred)}')
print(f'Accuracy: {acc:.3f}')
print('\nSUCCESS - ResNet works!')

