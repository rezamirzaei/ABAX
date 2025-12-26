#!/usr/bin/env python3
"""Comprehensive project error check."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("🔍 COMPREHENSIVE ERROR CHECK")
print("=" * 60)

errors = []

# Test 1: Import all core modules
print("\n1️⃣  Testing core imports...")
try:
    from src.data import load_uah_driveset, load_epa_fuel_economy, split_by_driver
    print("   ✅ Data loaders")
except Exception as e:
    errors.append(f"Data loaders: {e}")
    print(f"   ❌ Data loaders: {e}")

try:
    from src.models import ModelTrainer
    print("   ✅ ModelTrainer")
except Exception as e:
    errors.append(f"ModelTrainer: {e}")
    print(f"   ❌ ModelTrainer: {e}")

try:
    from src.models.cnn import CNNClassifier
    print("   ✅ CNNClassifier")
except Exception as e:
    errors.append(f"CNNClassifier: {e}")
    print(f"   ❌ CNNClassifier: {e}")

try:
    from src.visualization import setup_style, plot_learning_curves
    print("   ✅ Visualization")
except Exception as e:
    errors.append(f"Visualization: {e}")
    print(f"   ❌ Visualization: {e}")

# Test 2: Check data files exist
print("\n2️⃣  Testing data files...")
data_dir = Path('data/processed')
required_files = ['uah_classification.csv', 'epa_fuel_economy.csv']
for file in required_files:
    file_path = data_dir / file
    if file_path.exists():
        print(f"   ✅ {file}")
    else:
        errors.append(f"Missing: {file}")
        print(f"   ❌ {file}")

# Test 3: Test data loading
print("\n3️⃣  Testing data loading...")
try:
    import pandas as pd
    df = pd.read_csv('data/processed/uah_classification.csv')
    print(f"   ✅ UAH classification data ({df.shape[0]} samples)")
except Exception as e:
    errors.append(f"Load UAH data: {e}")
    print(f"   ❌ UAH data: {e}")

try:
    df_epa = pd.read_csv('data/processed/epa_fuel_economy.csv')
    print(f"   ✅ EPA fuel economy data ({df_epa.shape[0]} samples)")
except Exception as e:
    errors.append(f"Load EPA data: {e}")
    print(f"   ❌ EPA data: {e}")

# Test 4: Test sklearn imports
print("\n4️⃣  Testing sklearn...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    print("   ✅ Sklearn models and metrics")
except Exception as e:
    errors.append(f"Sklearn: {e}")
    print(f"   ❌ Sklearn: {e}")

# Test 5: Test PyTorch
print("\n5️⃣  Testing PyTorch...")
try:
    import torch
    import numpy as np
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   ✅ NumPy {np.__version__}")
except Exception as e:
    errors.append(f"PyTorch: {e}")
    print(f"   ❌ PyTorch: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"❌ FOUND {len(errors)} ERROR(S):")
    for i, err in enumerate(errors, 1):
        print(f"   {i}. {err}")
    sys.exit(1)
else:
    print("✅ ALL CHECKS PASSED - PROJECT IS HEALTHY!")
    sys.exit(0)

