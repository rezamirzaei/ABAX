"""Test notebook 02 and 04 logic for errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score

from src.data import split_by_driver
from src.models import get_classifiers, get_regressors, train_and_evaluate_classifier, train_and_evaluate_regressor
from src.models import CNNClassifier

print("=" * 60)
print("Testing Notebook 02 (Classification) Logic")
print("=" * 60)

# Test notebook 02
df = pd.read_csv('data/processed/uah_raw_features.csv')
print(f'Loaded: {df.shape}')

feature_cols = [c for c in df.columns if c not in ['driver', 'behavior', 'road_type']]
X = df[feature_cols + ['driver']].copy()
X[feature_cols] = X[feature_cols].fillna(0)
y = df['behavior']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = split_by_driver(X, y_enc, test_drivers=['D6'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'X_train_scaled shape: {X_train_scaled.shape}')
print(f'y_train type: {type(y_train)}')

# Test classifiers
classifiers = get_classifiers(class_weight='balanced', random_state=42)
print(f'\nTesting {len(classifiers)} classifiers...')
for name, clf in list(classifiers.items())[:3]:  # Test first 3
    y_pred, acc, f1 = train_and_evaluate_classifier(clf, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f'  {name}: Acc={acc:.3f}')

# Test CNN
print('\nTesting CNN...')
cnn = CNNClassifier(epochs=5, batch_size=8, verbose=0)
cnn.fit(X_train_scaled, y_train)
y_pred_cnn = cnn.predict(X_test_scaled)
print(f'  CNN works: {len(y_pred_cnn)} predictions')

# Test sparse model
print('\nTesting L1 Sparse...')
sparse_lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
sparse_lr.fit(X_train_scaled, y_train)
n_nonzero = np.sum(np.any(sparse_lr.coef_ != 0, axis=0))
print(f'  L1 selected {n_nonzero}/{len(feature_cols)} features')

print('✅ Notebook 02 all tests passed!')

print("\n" + "=" * 60)
print("Testing Notebook 04 (Regression) Logic")
print("=" * 60)

# Test notebook 04
df_reg = pd.read_csv('data/processed/epa_fuel_economy.csv')
print(f'Loaded: {df_reg.shape}')

# Sample for speed
df_reg = df_reg.sample(n=500, random_state=42)

target_col = 'comb08'
numeric_cols = ['year', 'cylinders', 'displ']
cat_cols = ['drive', 'VClass', 'fuelType']

X_reg = df_reg[numeric_cols + cat_cols].copy()
y_reg = df_reg[target_col].copy()

X_reg = X_reg.fillna(X_reg.mode().iloc[0])
X_reg_encoded = pd.get_dummies(X_reg, columns=cat_cols, drop_first=True)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_encoded, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f'X_train_reg_scaled shape: {X_train_reg_scaled.shape}')

# Test regressors
regressors = get_regressors(random_state=42)
print(f'\nTesting {len(regressors)} regressors...')
for name, reg in list(regressors.items())[:3]:  # Test first 3
    y_pred, r2, rmse, mae = train_and_evaluate_regressor(reg, X_train_reg_scaled, y_train_reg, X_test_reg_scaled, y_test_reg)
    print(f'  {name}: R²={r2:.4f}')

# Test Lasso sparse
print('\nTesting Lasso sparse...')
lasso = Lasso(alpha=0.1, max_iter=2000, random_state=42)
lasso.fit(X_train_reg_scaled, y_train_reg)
n_nonzero_reg = np.sum(lasso.coef_ != 0)
print(f'  Lasso selected {n_nonzero_reg}/{X_train_reg_scaled.shape[1]} features')

print('✅ Notebook 04 all tests passed!')

print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)

