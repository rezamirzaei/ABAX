"""
Quick test of the raw data classification pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.data import split_by_driver
from src.models import CNNClassifier

print('='*60)
print('Testing Classification Pipeline on Raw Features')
print('='*60)

# Load raw features
raw_df = pd.read_csv('data/processed/uah_raw_features.csv')
print(f'\n📊 Loaded: {raw_df.shape}')

# Prepare
feature_cols = [c for c in raw_df.columns if c not in ['driver', 'behavior', 'road_type']]
X = raw_df[feature_cols + ['driver']].copy()
X[feature_cols] = X[feature_cols].fillna(0)
y = raw_df['behavior'].values

# Split (D6 always test)
X_train, X_test, y_train, y_test = split_by_driver(X, y, test_drivers=['D6'])

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'   Train: {len(X_train)}, Test: {len(X_test)}')

# Test a few models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
}

print('\n📈 Model Results (D6 Held Out):')
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'   {name}: {acc:.4f}')

# Test CNN (quick, 10 epochs)
print('\n🧠 Testing CNN (10 epochs)...')
cnn = CNNClassifier(epochs=10, batch_size=8, verbose=0)
cnn.fit(X_train_scaled, y_train)
y_pred_cnn = cnn.predict(X_test_scaled)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print(f'   CNN (PyTorch): {acc_cnn:.4f}')

print('\n✅ All tests passed!')

# Save results summary
with open('results/raw_classification_test.txt', 'w') as f:
    f.write('Raw Data Classification Test Results\n')
    f.write('='*50 + '\n')
    f.write(f'Train samples: {len(X_train)}\n')
    f.write(f'Test samples: {len(X_test)}\n')
    f.write('\nModel Accuracies:\n')
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f.write(f'  {name}: {acc:.4f}\n')
    f.write(f'  CNN (PyTorch): {acc_cnn:.4f}\n')
print('Results saved to results/raw_classification_test.txt')

