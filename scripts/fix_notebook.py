#!/usr/bin/env python3
"""Script to fix the classification notebook properly."""

import json

NOTEBOOK_PATH = '/Users/rezami/PycharmProjects/ABAX/notebooks/02_classification.ipynb'

# The correct imports cell content
IMPORTS_CELL = '''import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Project imports - Clean, modular API
from src.classification import (
    ClassificationResult,
    get_all_trips, load_raw_gps, load_raw_accelerometer,
    extract_raw_features, load_or_build_dataset,
    prepare_classification_data, get_all_classifiers,
    train_all_classifiers, get_best_model,
    get_feature_importance, results_to_dataframe
)
from src.classification.visualization import (
    plot_class_distribution, plot_feature_distributions,
    plot_driver_distribution, plot_correlation_matrix,
    plot_model_comparison, plot_confusion_matrix,
    plot_feature_importance, plot_behavior_comparison,
    plot_raw_accelerometer
)
from src.classification.cv import run_logo_cv
from src.models import ResNetClassifier, plot_resnet_training_history
from src.utils import print_header, print_success

# Standard libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuration
DATA_DIR = project_root / 'data' / 'UAH-DRIVESET-v1'
FIGURES_DIR = project_root / 'results' / 'figures'
CACHE_PATH = project_root / 'data' / 'processed' / 'uah_raw_features.csv'

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print_success("Environment configured successfully!")
print(f"📂 Data directory: {DATA_DIR}")
print(f"📊 Figures will be saved to: {FIGURES_DIR}")
'''

# The correct ResNet training cell
RESNET_TRAIN_CELL = '''print_header("RESNET CLASSIFICATION (PyTorch)", "🧠")

resnet = ResNetClassifier(
    n_filters=32, n_blocks=2, kernel_size=3, hidden_size=64, dropout=0.3,
    epochs=100, batch_size=8, learning_rate=0.001, weight_decay=1e-3,
    early_stopping_patience=15, verbose=1, random_state=42
)
resnet.fit(data.X_train, data.y_train)

# Evaluate
y_pred_resnet = resnet.predict(data.X_test)
y_pred_resnet_enc = resnet.le_.transform(y_pred_resnet)
acc_resnet = accuracy_score(data.y_test, y_pred_resnet_enc)
f1_resnet = f1_score(data.y_test, y_pred_resnet_enc, average='weighted')
train_acc_resnet = accuracy_score(data.y_train, resnet.le_.transform(resnet.predict(data.X_train)))

print(f"\\n📊 ResNet Results: Train={train_acc_resnet:.3f}, Test={acc_resnet:.3f}, F1={f1_resnet:.3f}")

# Add to results
results.append(ClassificationResult(
    model_name='ResNet (PyTorch)', train_accuracy=train_acc_resnet,
    test_accuracy=acc_resnet, f1_score=f1_resnet,
    predictions=y_pred_resnet_enc, model=resnet
))
'''

# The correct ResNet plot cell
RESNET_PLOT_CELL = '''# Plot training history
history = resnet.get_training_history()
plot_resnet_training_history(history, save_path=str(FIGURES_DIR / 'resnet_learning_curves_classification.png'))
print_success("ResNet learning curves saved")
'''

def fix_notebook():
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Always fix cell 3 with proper imports
    nb['cells'][3]['source'] = IMPORTS_CELL
    print("Fixed imports in cell 3")

    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Cell 34 should be ResNet training
        if i == 34 and cell['cell_type'] == 'code':
            nb['cells'][i]['source'] = RESNET_TRAIN_CELL
            print(f"Fixed ResNet training in cell {i}")

        # Cell 35 should be ResNet plot
        if i == 35 and cell['cell_type'] == 'code':
            nb['cells'][i]['source'] = RESNET_PLOT_CELL
            print(f"Fixed ResNet plot in cell {i}")

        # Fix comment "Update comparison with CNN" -> "Update comparison with ResNet"
        if 'Update comparison with CNN' in source:
            new_source = source.replace('Update comparison with CNN', 'Update comparison with ResNet')
            nb['cells'][i]['source'] = new_source
            print(f"Fixed comment in cell {i}")

    # Save the notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print("\nNotebook fixed successfully!")

if __name__ == '__main__':
    fix_notebook()

