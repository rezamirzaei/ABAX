#!/usr/bin/env python
"""Verify that all classification figures are valid PNG images."""

from PIL import Image
import os

os.chdir('/Users/rezami/PycharmProjects/ABAX')

figures = [
    'classifier_comparison.png',
    'model_comparison_classification.png',
    'feature_importance_classification.png',
    'cnn_learning_curves_classification.png',
    'confusion_matrix_classification.png'
]

print("Verifying figures...")
all_ok = True

for fig in figures:
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

