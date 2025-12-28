"""Generate all figures for the LaTeX report using the clean classification module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.classification import (
    load_or_build_dataset,
    prepare_classification_data,
    get_all_classifiers,
    train_all_classifiers,
    get_best_model,
    get_feature_importance,
    results_to_dataframe,
    # Data loading
    get_all_trips,
    load_raw_accelerometer,
    load_raw_gps,
    compute_acceleration_magnitude,
    # Visualization
    plot_class_distribution,
    plot_feature_distributions,
    plot_driver_distribution,
    plot_correlation_matrix,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.visualization import setup_style

setup_style()
FIGURES_DIR = Path('results/figures')

print("=" * 60)
print("Generating All Figures for LaTeX Report")
print("=" * 60)

# 1. Load data
df = load_or_build_dataset(
    data_dir=Path('data/UAH-DRIVESET-v1'),
    cache_path=Path('data/processed/uah_raw_features.csv')
)
print(f"\n📊 Dataset: {df.shape}")

# 2. Feature columns
feature_cols = [c for c in df.columns if c not in ['driver', 'behavior', 'road_type']]

# 3. Generate EDA figures
print("\n🎨 Generating EDA figures...")
plot_class_distribution(df, save_path=FIGURES_DIR / 'class_distribution.png')
print("   ✅ class_distribution.png")

key_features = ['speed_mean', 'speed_std', 'acc_magnitude_std',
                'jerk_x_std', 'hard_brake_count', 'sharp_turn_count']
plot_feature_distributions(df, key_features, save_path=FIGURES_DIR / 'feature_distributions_classification.png')
print("   ✅ feature_distributions_classification.png")

plot_driver_distribution(df, save_path=FIGURES_DIR / 'driver_behavior_distribution.png')
print("   ✅ driver_behavior_distribution.png")

plot_correlation_matrix(df, feature_cols, save_path=FIGURES_DIR / 'correlation_matrix_classification.png')
print("   ✅ correlation_matrix_classification.png")

# 4. Prepare data and train models
print("\n🎯 Training classifiers...")
data = prepare_classification_data(df, test_drivers=['D6'])
results = train_all_classifiers(data, verbose=True)

# 5. Generate model comparison figures
print("\n🎨 Generating model figures...")
plot_model_comparison(results, save_path=FIGURES_DIR / 'classifier_comparison.png')
print("   ✅ classifier_comparison.png")

best = get_best_model(results)
print(f"\n🥇 Best Model: {best.model_name} (Test Acc: {best.test_accuracy:.3f})")

plot_confusion_matrix(
    data.y_test, best.predictions, data.class_names,
    model_name=best.model_name,
    save_path=FIGURES_DIR / 'confusion_matrix_classification.png'
)
print("   ✅ confusion_matrix_classification.png")

importance_df = get_feature_importance(best.model, data.feature_names)
plot_feature_importance(
    importance_df,
    title=f'Feature Importance ({best.model_name})',
    save_path=FIGURES_DIR / 'feature_importance_classification.png'
)
print("   ✅ feature_importance_classification.png")

# 6. Generate raw accelerometer figure
print("\n🎨 Generating raw sensor figure...")
trips = get_all_trips(Path('data/UAH-DRIVESET-v1'))
# Find an aggressive trip (case insensitive)
sample_trip = None
for t in trips:
    if t.behavior.upper() == 'AGGRESSIVE':
        sample_trip = t
        break
if sample_trip is None:
    sample_trip = trips[0]  # Fallback to first trip

acc = load_raw_accelerometer(sample_trip.path)
gps = load_raw_gps(sample_trip.path)

if acc is not None:
    acc['acc_magnitude'] = compute_acceleration_magnitude(
        acc['acc_x_kf'].values, acc['acc_y_kf'].values, acc['acc_z_kf'].values
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(acc['timestamp'][:500], acc['acc_x'][:500], alpha=0.4, label='Raw', color='lightblue')
    ax.plot(acc['timestamp'][:500], acc['acc_x_kf'][:500], label='Kalman Filtered', color='blue', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('X-Axis: Longitudinal (Braking/Acceleration)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[0, 1]
    ax.plot(acc['timestamp'][:500], acc['acc_y'][:500], alpha=0.4, label='Raw', color='lightgreen')
    ax.plot(acc['timestamp'][:500], acc['acc_y_kf'][:500], label='Kalman Filtered', color='green', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Y-Axis: Lateral (Turning)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 0]
    ax.plot(acc['timestamp'][:500], acc['acc_magnitude'][:500], color='purple', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Gravity baseline')
    ax.set_title('Acceleration Magnitude', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Acceleration| (g)')
    ax.legend(loc='upper right', fontsize=8)

    if gps is not None:
        ax = axes[1, 1]
        ax.plot(gps['timestamp'], gps['speed'], color='navy', linewidth=1)
        ax.set_title('Speed (from GPS)', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'raw_accelerometer_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ raw_accelerometer_data.png")

# 7. Print summary
print("\n" + "=" * 60)
print("✅ All figures generated successfully!")
print("=" * 60)

comparison = results_to_dataframe(results)
print("\nTop Models:")
print(comparison.head(8).to_string(index=False))

