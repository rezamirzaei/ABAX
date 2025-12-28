#!/usr/bin/env python
"""Verify that all key figures are valid PNG images."""

from PIL import Image
import os

os.chdir('/Users/rezami/PycharmProjects/ABAX')

# Classification figures (6-10)
classification_figures = [
    'classifier_comparison.png',
    'model_comparison_classification.png',
    'feature_importance_classification.png',
    'cnn_learning_curves_classification.png',
    'confusion_matrix_classification.png'
]

# Regression figures (16, 18, 19, 20)
regression_figures = [
    'regressor_comparison.png',
    'actual_vs_predicted.png',
    'feature_importance_regression.png',
    'residuals.png',
    'prediction_intervals.png'
]

all_figures = classification_figures + regression_figures

print("Verifying figures...")
all_ok = True

for fig in all_figures:
    path = f'results/figures/{fig}'
    try:
        img = Image.open(path)
        img.verify()  # Verify it's a valid image
        img = Image.open(path)  # Reopen after verify
        print(f"  OK: {fig} ({img.size[0]}x{img.size[1]}, {img.mode})")
    except Exception as e:
        print(f"  ERROR: {fig} - {e}")
        all_ok = False

if all_ok:
    print("\nAll figures verified successfully!")
else:
    print("\nSome figures have issues!")

