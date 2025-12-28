"""
Data splitting utilities with proper generalization strategies.
"""

from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.core.schemas import Dataset, SplitData


def split_data(
    dataset: Dataset,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> SplitData:
    """
    Split dataset into training and test sets.

    Args:
        dataset: Dataset to split.
        test_size: Fraction of data for testing.
        stratify: Whether to stratify (for classification).
        random_state: Random seed.

    Returns:
        SplitData with train and test sets.
    """
    X = dataset.X
    y = dataset.y

    # Determine if stratification is possible
    stratify_col = None
    if stratify and dataset.info and dataset.info.task_type == "classification":
        stratify_col = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state
    )

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name
    )


def split_by_driver(
    X: pd.DataFrame,
    y: pd.Series,
    test_drivers: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data by driver for better generalization testing.
    
    This ensures that trips from the same driver don't appear in both
    train and test sets, which better simulates deployment on new drivers.
    
    **Why this matters:**
    - In real-world deployment, models must work on NEW drivers never seen before
    - Random splits leak information when the same driver appears in train and test
    - Driver-level splits provide realistic generalization estimates
    
    Args:
        X: Features (must contain 'driver' column)
        y: Target
        test_drivers: Specific drivers to hold out for testing (e.g., ['D6'])
        test_size: Approximate fraction for test (used if test_drivers=None)
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test (with 'driver' column removed from features)
    """
    if 'driver' not in X.columns:
        raise ValueError("X must contain 'driver' column for driver-level splitting")
    
    # Get unique drivers
    unique_drivers = X['driver'].unique()
    
    if test_drivers is None:
        # Randomly select drivers for test set
        n_test_drivers = max(1, int(len(unique_drivers) * test_size))
        np.random.seed(random_state)
        test_drivers = np.random.choice(unique_drivers, size=n_test_drivers, replace=False)
    
    # Split by driver
    test_mask = X['driver'].isin(test_drivers)
    train_mask = ~test_mask
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    # Remove driver column from features
    X_train = X_train.drop(columns=['driver'])
    X_test = X_test.drop(columns=['driver'])
    
    print(f"\n📊 Driver-level split:")
    print(f"  Train drivers: {sorted([d for d in unique_drivers if d not in test_drivers])}")
    print(f"  Test drivers: {sorted(test_drivers)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  ✅ This ensures generalization to NEW DRIVERS\n")
    
    return X_train, X_test, y_train, y_test
