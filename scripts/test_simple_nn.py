#!/usr/bin/env python3
"""Test script for SimpleNN classifier."""

import sys
sys.path.insert(0, '/Users/rezami/PycharmProjects/ABAX')

from src.models.simple_nn import SimpleNNClassifier
import numpy as np

print("Testing SimpleNN classifier...")
np.random.seed(42)

# Create test data with some structure
X = np.random.randn(40, 36).astype(np.float32)
X[:14, 0] += 2  # NORMAL
X[14:26, 1] -= 2  # DROWSY
X[26:, 2] += 3  # AGGRESSIVE

y = np.array(['NORMAL']*14 + ['DROWSY']*12 + ['AGGRESSIVE']*14)

print(f"Data shape: {X.shape}, Labels: {np.unique(y)}")

clf = SimpleNNClassifier(
    hidden_sizes=[64, 32],
    epochs=50,
    verbose=1,
    random_state=42
)
clf.fit(X, y)

train_acc = clf.score(X, y)
print(f'\nFinal Train Accuracy: {train_acc:.3f}')
print('SUCCESS - SimpleNN works!')

