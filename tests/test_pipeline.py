#!/usr/bin/env python
"""Complete pipeline test."""

import sys
import warnings
import os

warnings.filterwarnings('ignore')
os.chdir('/Users/rezami/PycharmProjects/ABAX')
sys.path.insert(0, '.')

def test_classification_pipeline():
    """Test the full classification pipeline."""
    print("=" * 60)
    print("CLASSIFICATION PIPELINE TEST")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading UAH data...")
    from src.data import load_uah_driveset, split_data
    dataset = load_uah_driveset('data/UAH-DRIVESET-v1')
    print(f"   Loaded: {dataset.n_samples} samples, {dataset.n_features} features")
    print(f"   Classes: {dataset.info.class_distribution}")
    
    # 2. Split
    print("\n2. Splitting data...")
    split = split_data(dataset, test_size=0.2, stratify=True)
    print(f"   Train: {split.n_train}, Test: {split.n_test}")
    
    # 3. Preprocess
    print("\n3. Preprocessing...")
    from src.features import preprocess_features, encode_target
    train_feat, test_feat = preprocess_features(split, scaler_type='robust')
    print(f"   Features: {train_feat.n_features}")
    
    # 4. Encode target
    print("\n4. Encoding target...")
    y_train, y_test, encoder = encode_target(split.y_train, split.y_test)
    print(f"   Classes: {encoder.classes_}")
    
    # 5. Train a single model
    print("\n5. Training model...")
    from sklearn.ensemble import RandomForestClassifier
    from src.models import train_model, evaluate_classifier
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    trained = train_model(train_feat.X, y_train, rf, "Random Forest", 
                         X_val=test_feat.X, y_val=y_test)
    print(f"   Trained: {trained.model_name}")
    print(f"   History: {len(trained.history.train_scores)} iterations")
    
    # 6. Evaluate
    print("\n6. Evaluating...")
    metrics = evaluate_classifier(trained.model, test_feat.X, y_test, list(encoder.classes_))
    print(f"   Accuracy: {metrics.accuracy:.4f}")
    print(f"   F1 Score: {metrics.f1_score:.4f}")
    print(f"   Confusion Matrix:\n{metrics.confusion_matrix}")
    
    print("\n✅ Classification pipeline PASSED!")
    return True


def test_regression_pipeline():
    """Test the regression pipeline with sample data."""
    print("\n" + "=" * 60)
    print("REGRESSION PIPELINE TEST")
    print("=" * 60)
    
    # Use synthetic data for quick test (EPA download is slow)
    print("\n1. Creating sample regression data...")
    import numpy as np
    import pandas as pd
    from src.core import Dataset, DatasetInfo
    
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    y = pd.Series(3 * X['feature1'] - 2 * X['feature2'] + np.random.randn(n_samples) * 0.5)
    
    dataset = Dataset(
        X=X, y=y,
        feature_names=list(X.columns),
        target_name='target',
        info=DatasetInfo(
            name="Synthetic", n_samples=n_samples, n_features=4,
            feature_names=list(X.columns), target_name='target',
            task_type="regression"
        )
    )
    print(f"   Created: {dataset.n_samples} samples")
    
    # 2. Split
    print("\n2. Splitting data...")
    from src.data import split_data
    split = split_data(dataset, test_size=0.2, stratify=False)
    print(f"   Train: {split.n_train}, Test: {split.n_test}")
    
    # 3. Preprocess
    print("\n3. Preprocessing...")
    from src.features import preprocess_features
    train_feat, test_feat = preprocess_features(split, scaler_type='robust')
    print(f"   Features: {train_feat.n_features}")
    
    # 4. Get targets
    y_train = split.y_train.values if hasattr(split.y_train, 'values') else split.y_train
    y_test = split.y_test.values if hasattr(split.y_test, 'values') else split.y_test
    
    # 5. Train robust regressors
    print("\n4. Training robust regressors...")
    from sklearn.linear_model import HuberRegressor, LinearRegression
    from src.models import train_model, evaluate_regressor
    
    # OLS
    ols = LinearRegression()
    trained_ols = train_model(train_feat.X, y_train, ols, "OLS")
    metrics_ols = evaluate_regressor(trained_ols.model, test_feat.X, y_test)
    print(f"   OLS - R²: {metrics_ols.r2:.4f}, RMSE: {metrics_ols.rmse:.4f}")
    
    # Huber (robust)
    huber = HuberRegressor(epsilon=1.35, max_iter=200)
    trained_huber = train_model(train_feat.X, y_train, huber, "Huber")
    metrics_huber = evaluate_regressor(trained_huber.model, test_feat.X, y_test)
    print(f"   Huber - R²: {metrics_huber.r2:.4f}, RMSE: {metrics_huber.rmse:.4f}")
    
    print("\n✅ Regression pipeline PASSED!")
    return True


if __name__ == "__main__":
    try:
        test_classification_pipeline()
        test_regression_pipeline()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
