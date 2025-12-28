#!/usr/bin/env python
"""Optional CNN smoke test.

Why this file exists:
- TensorFlow is a heavy, optional dependency.
- The core project (pipelines + sklearn models) should be testable without TensorFlow.

Important:
- This is *not* a pytest unit test.
- It's a standalone script you can run manually when TF is installed.

Run:
  python scripts/test_cnn.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")


def main() -> int:
    try:
        # Import inside try so environments without TF don't error during test collection
        from src.models.cnn import train_cnn
        from src.data import load_uah_driveset, split_data
        from src.features import preprocess_features, encode_target
        import numpy as np
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", str(e))
        print(
            "Skipping CNN smoke test because an optional dependency is missing: "
            f"{missing}.\n"
            "If you want to run this, install TensorFlow and run:\n"
            "  python scripts/test_cnn.py"
        )
        return 0

    print("Loading data...")
    dataset = load_uah_driveset("data/UAH-DRIVESET-v1")
    split = split_data(dataset, test_size=0.2, stratify=True)
    train_feat, test_feat = preprocess_features(split, scaler_type="robust")
    y_train, y_test, encoder = encode_target(split.y_train, split.y_test)

    print(f"Train shape: {train_feat.X.shape}")
    print(f"Classes: {encoder.classes_}")

    print("\nTraining CNN...")
    trained = train_cnn(
        train_feat.X,
        y_train,
        X_val=test_feat.X,
        y_val=y_test,
        n_epochs=30,
        learning_rate=0.01,
    )

    print("\nCNN Training History:")
    print(f"Iterations: {len(trained.history.iterations)}")
    print(f"Train scores (first 5): {trained.history.train_scores[:5]}")
    print(f"Val scores (first 5): {trained.history.val_scores[:5]}")
    print(f"Train scores (last 3): {trained.history.train_scores[-3:]}")
    print(f"Val scores (last 3): {trained.history.val_scores[-3:]}")

    preds = trained.model.predict(test_feat.X)
    acc = float(np.mean(preds == y_test))
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\n✅ CNN smoke test completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
