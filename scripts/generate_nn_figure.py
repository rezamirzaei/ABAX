#!/usr/bin/env python3
"""Generate Neural Network learning curves figure for the technical report."""

import sys
sys.path.insert(0, '/Users/rezami/PycharmProjects/ABAX')

import numpy as np
import pandas as pd
from pathlib import Path

# Load the actual data
from src.classification import load_or_build_dataset, prepare_classification_data
from src.models.simple_nn import SimpleNNClassifier, plot_nn_training_history

print("Loading data...")
DATA_DIR = Path('/Users/rezami/PycharmProjects/ABAX/data/UAH-DRIVESET-v1')
CACHE_PATH = Path('/Users/rezami/PycharmProjects/ABAX/data/processed/uah_raw_features.csv')
FIGURES_DIR = Path('/Users/rezami/PycharmProjects/ABAX/results/figures')

df = load_or_build_dataset(data_dir=DATA_DIR, cache_path=CACHE_PATH)
data = prepare_classification_data(df, test_drivers=['D6'])

print(f"Training data: {data.X_train.shape[0]} samples")
print(f"Test data: {data.X_test.shape[0]} samples")

# Train Neural Network
print("\nTraining Neural Network with normalization...")
nn_clf = SimpleNNClassifier(
    hidden_sizes=[64, 32],
    dropout=0.3,
    epochs=150,
    batch_size=8,
    learning_rate=0.001,
    weight_decay=1e-4,
    early_stopping_patience=20,
    verbose=1,
    random_state=42
)
nn_clf.fit(data.X_train, data.y_train)

# Evaluate
train_acc = nn_clf.score(data.X_train, data.y_train)
test_acc = nn_clf.score(data.X_test, data.y_test)
print(f"\nResults: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

# Plot and save
history = nn_clf.get_training_history()
save_path = FIGURES_DIR / 'nn_learning_curves_classification.png'
plot_nn_training_history(history, save_path=str(save_path))
print(f"\nFigure saved to: {save_path}")
print("SUCCESS!")

