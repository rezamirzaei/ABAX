"""Tests for features module."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import SplitData, FeatureSet
from src.features.preprocessing import (
    FeaturePreprocessor,
    preprocess_features,
    TargetEncoder,
    encode_target,
)


class TestFeaturePreprocessor:
    """Tests for feature preprocessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with numeric and categorical features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
        })
        return df

    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        preprocessor = FeaturePreprocessor(scaler_type="robust")
        X_transformed = preprocessor.fit_transform(sample_data)

        assert X_transformed.shape[0] == 100
        assert preprocessor.numeric_cols == ['num1', 'num2']
        assert preprocessor.categorical_cols == ['cat1']

    def test_transform(self, sample_data):
        """Test transform on new data."""
        preprocessor = FeaturePreprocessor(scaler_type="robust")
        preprocessor.fit_transform(sample_data)

        new_data = sample_data.copy()
        X_transformed = preprocessor.transform(new_data)

        assert X_transformed.shape[0] == 100


class TestPreprocessFeatures:
    """Tests for preprocess_features function."""

    @pytest.fixture
    def split_data(self):
        """Create split data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'num1': np.random.randn(80),
            'num2': np.random.randn(80),
        })
        X_test = pd.DataFrame({
            'num1': np.random.randn(20),
            'num2': np.random.randn(20),
        })
        y_train = pd.Series(np.random.randint(0, 2, 80))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        return SplitData(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=['num1', 'num2']
        )

    def test_preprocess_features(self, split_data):
        """Test preprocessing pipeline."""
        train_feat, test_feat = preprocess_features(split_data)

        assert isinstance(train_feat, FeatureSet)
        assert isinstance(test_feat, FeatureSet)
        assert train_feat.X.shape[0] == 80
        assert test_feat.X.shape[0] == 20


class TestTargetEncoder:
    """Tests for target encoder."""

    def test_encode_target(self):
        """Test target encoding."""
        y_train = pd.Series(['A', 'B', 'C', 'A', 'B'])
        y_test = pd.Series(['A', 'C'])

        y_train_enc, y_test_enc, encoder = encode_target(y_train, y_test)

        assert len(y_train_enc) == 5
        assert len(y_test_enc) == 2
        assert len(encoder.classes_) == 3

    def test_inverse_transform(self):
        """Test inverse transform."""
        y_train = pd.Series(['A', 'B', 'C'])
        y_test = pd.Series(['A'])

        _, _, encoder = encode_target(y_train, y_test)

        original = encoder.inverse_transform([0, 1, 2])
        assert list(original) == ['A', 'B', 'C']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
