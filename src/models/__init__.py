"""Models module exports."""

from src.models.trainer import ModelTrainer, train_model
from src.models.evaluation import (
    ClassificationEvaluator,
    RegressionEvaluator,
    evaluate_classifier,
    evaluate_regressor,
    compute_regression_metrics,
    train_and_evaluate_regressor,
    train_and_evaluate_classifier,
    QuantileRegressor,
    compute_interval_coverage,
    compute_interval_width,
)
from src.models.comparison import (
    get_classifiers,
    get_regressors,
    compare_classifiers,
    compare_regressors,
)

# Optional: CNN models (requires PyTorch)
try:
    from src.models.cnn import CNNClassifier
    _HAS_CNN = True
except ImportError:
    CNNClassifier = None
    _HAS_CNN = False

__all__ = [
    "ModelTrainer",
    "train_model",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "evaluate_classifier",
    "evaluate_regressor",
    "compute_regression_metrics",
    "train_and_evaluate_regressor",
    "train_and_evaluate_classifier",
    "QuantileRegressor",
    "compute_interval_coverage",
    "compute_interval_width",
    "get_classifiers",
    "get_regressors",
    "compare_classifiers",
    "compare_regressors",
]

if _HAS_CNN:
    __all__.extend(["CNNClassifier"])

