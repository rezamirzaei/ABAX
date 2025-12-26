#!/usr/bin/env python
"""Run and validate all notebooks."""

import sys
import warnings
import os

warnings.filterwarnings('ignore')
os.chdir('/Users/rezami/PycharmProjects/ABAX')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook_01():
    """Run EDA Classification notebook."""
    print("=" * 60)
    print("NOTEBOOK 01: EDA Classification")
    print("=" * 60)
    
    from src.data import load_uah_driveset
    from src.visualization import setup_style, plot_class_distribution
    
    setup_style()
    print("✓ Imports successful")
    
    # Load data
    dataset = load_uah_driveset('data/UAH-DRIVESET-v1')
    print(f"✓ Loaded {dataset.n_samples} samples, {dataset.n_features} features")
    
    # Create DataFrame
    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    df['behavior'] = dataset.y.values
    print(f"✓ DataFrame shape: {df.shape}")
    
    # Class distribution
    counts = df['behavior'].value_counts()
    print(f"✓ Class counts: {counts.to_dict()}")
    
    imbalance_ratio = counts.min() / counts.max()
    print(f"✓ Imbalance ratio: {imbalance_ratio:.3f}")
    
    print("✅ Notebook 01 PASSED\n")
    return True


def run_notebook_02():
    """Run Classification notebook."""
    print("=" * 60)
    print("NOTEBOOK 02: Classification")
    print("=" * 60)
    
    from src.data import load_uah_driveset, split_data
    from src.features import preprocess_features, encode_target
    from src.models import compare_classifiers
    
    # Load and prepare
    dataset = load_uah_driveset('data/UAH-DRIVESET-v1')
    print(f"✓ Loaded {dataset.n_samples} samples")
    
    split = split_data(dataset, test_size=0.2, stratify=True)
    print(f"✓ Split: Train={split.n_train}, Test={split.n_test}")
    
    train_feat, test_feat = preprocess_features(split, scaler_type='robust')
    print(f"✓ Features: {train_feat.n_features}")
    
    y_train, y_test, encoder = encode_target(split.y_train, split.y_test)
    print(f"✓ Classes: {list(encoder.classes_)}")
    
    # Compare models
    comparison = compare_classifiers(
        X_train=train_feat.X,
        y_train=y_train,
        X_test=test_feat.X,
        y_test=y_test,
        class_names=list(encoder.classes_),
    )
    print(f"✓ Best model: {comparison.best_model_name}")
    
    best = comparison.best_model
    print(f"✓ Accuracy: {best.test_metrics.accuracy:.4f}")
    print(f"✓ F1 Score: {best.test_metrics.f1_score:.4f}")
    
    print("✅ Notebook 02 PASSED\n")
    return comparison


def run_notebook_03():
    """Run EDA Regression notebook."""
    print("=" * 60)
    print("NOTEBOOK 03: EDA Regression")
    print("=" * 60)
    
    from src.data import load_epa_fuel_economy
    
    # Load data
    print("✓ Loading EPA data (this may take a moment)...")
    dataset = load_epa_fuel_economy(year_min=2015, sample_size=3000)
    print(f"✓ Loaded {dataset.n_samples} samples, {dataset.n_features} features")
    
    # Create DataFrame
    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    df['comb08'] = dataset.y.values
    print(f"✓ DataFrame shape: {df.shape}")
    
    # Target stats
    print(f"✓ Target (MPG) range: {df['comb08'].min():.1f} - {df['comb08'].max():.1f}")
    print(f"✓ Target mean: {df['comb08'].mean():.1f}")
    
    print("✅ Notebook 03 PASSED\n")
    return df


def run_notebook_04():
    """Run Regression notebook."""
    print("=" * 60)
    print("NOTEBOOK 04: Regression")
    print("=" * 60)
    
    from src.data import load_epa_fuel_economy, split_data
    from src.features import preprocess_features
    from src.models import compare_regressors
    
    # Load and prepare
    print("✓ Loading EPA data...")
    dataset = load_epa_fuel_economy(year_min=2015, sample_size=3000)
    print(f"✓ Loaded {dataset.n_samples} samples")
    
    split = split_data(dataset, test_size=0.2, stratify=False)
    print(f"✓ Split: Train={split.n_train}, Test={split.n_test}")
    
    train_feat, test_feat = preprocess_features(split, scaler_type='robust')
    print(f"✓ Features: {train_feat.n_features}")
    
    y_train = split.y_train.values if hasattr(split.y_train, 'values') else split.y_train
    y_test = split.y_test.values if hasattr(split.y_test, 'values') else split.y_test
    print(f"✓ Target range: {y_train.min():.1f} - {y_train.max():.1f} MPG")
    
    # Compare models
    comparison = compare_regressors(
        X_train=train_feat.X,
        y_train=y_train,
        X_test=test_feat.X,
        y_test=y_test,
    )
    print(f"✓ Best model: {comparison.best_model_name}")
    
    best = comparison.best_model
    print(f"✓ RMSE: {best.test_metrics.rmse:.4f}")
    print(f"✓ R²: {best.test_metrics.r2:.4f}")
    
    print("✅ Notebook 04 PASSED\n")
    return comparison


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VALIDATING ALL NOTEBOOKS")
    print("=" * 60 + "\n")
    
    try:
        run_notebook_01()
        run_notebook_02()
        run_notebook_03()
        run_notebook_04()
        
        print("=" * 60)
        print("✅ ALL NOTEBOOKS VALIDATED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
