"""Test the comprehensive notebook structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print('Testing comprehensive notebook imports...')

try:
    from src.classification import (
        ClassificationResult, DataSplit,
        get_all_trips, load_raw_gps, load_raw_accelerometer,
        compute_acceleration_magnitude, extract_raw_features,
        build_raw_dataset, load_or_build_dataset,
        prepare_classification_data, get_all_classifiers,
        train_all_classifiers, train_single_classifier,
        get_best_model, get_feature_importance, results_to_dataframe,
        run_logo_cv, get_classification_report,
        MCPLogisticRegression, SCADLogisticRegression,
        plot_class_distribution, plot_feature_distributions,
        plot_driver_distribution, plot_correlation_matrix,
        plot_model_comparison, plot_confusion_matrix, plot_feature_importance,
    )
    from src.data import split_by_driver
    from src.models import CNNClassifier, plot_cnn_training_history
    from src.visualization import setup_style
    from src.utils import print_success, print_header

    print('OK: All imports successful!')
except Exception as e:
    print(f'ERROR: Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Quick data test
try:
    df = load_or_build_dataset(
        data_dir=Path('data/UAH-DRIVESET-v1'),
        cache_path=Path('data/processed/uah_raw_features.csv')
    )
    print(f'OK: Dataset loaded: {df.shape}')

    data = prepare_classification_data(df, test_drivers=['D6'])
    print(f'OK: Train: {data.X_train.shape}, Test: {data.X_test.shape}')

    # Train a quick model
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    result = train_single_classifier(lr, 'Logistic', data)
    print(f'OK: Model trained: Test Acc = {result.test_accuracy:.3f}')

    print('\nALL TESTS PASSED!')

except Exception as e:
    print(f'ERROR: Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

