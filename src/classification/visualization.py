"""
Classification Visualization Module.

Clean, reusable plotting functions for classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple

from src.classification.types import ClassificationResult, DataSplit


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """Plot class distribution as bar and pie charts."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution', fontweight='bold')
    axes[0].set_xlabel('Behavior')
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(count), ha='center', fontsize=10)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Class Proportions', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot feature distributions by class."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    classes = df[target_col].unique()
    for ax, feat in zip(axes[:n_features], features):
        for cls in classes:
            data = df[df[target_col] == cls][feat]
            ax.hist(data, alpha=0.5, label=cls, bins=10)
        ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel(feat)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_driver_distribution(
    df: pd.DataFrame,
    driver_col: str = 'driver',
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """Plot trips per driver by behavior."""
    fig, ax = plt.subplots(figsize=figsize)

    crosstab = pd.crosstab(df[driver_col], df[target_col])
    crosstab.plot(kind='bar', ax=ax,
                  color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax.set_title('Trips per Driver by Behavior', fontweight='bold')
    ax.set_xlabel('Driver')
    ax.set_ylabel('Number of Trips')
    ax.legend(title='Behavior')
    plt.xticks(rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Plot feature correlation matrix."""
    fig, ax = plt.subplots(figsize=figsize)

    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_model_comparison(
    results: List[ClassificationResult],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """Plot model comparison with train vs test accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by test accuracy
    results = sorted(results, key=lambda x: x.test_accuracy)
    names = [r.model_name for r in results]
    test_accs = [r.test_accuracy for r in results]
    train_accs = [r.train_accuracy for r in results]

    # Test accuracy
    colors = plt.cm.RdYlGn(np.array(test_accs) / max(test_accs))
    axes[0].barh(range(len(results)), test_accs, color=colors)
    axes[0].set_yticks(range(len(results)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_title('Model Comparison - Test Accuracy', fontweight='bold')
    axes[0].set_xlim(0, 1)
    for i, acc in enumerate(test_accs):
        axes[0].text(acc + 0.02, i, f'{acc:.3f}', va='center', fontsize=9)

    # Train vs Test
    x = np.arange(len(results))
    width = 0.35
    axes[1].barh(x - width/2, train_accs, width, label='Train', color='#3498db')
    axes[1].barh(x + width/2, test_accs, width, label='Test', color='#e74c3c')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('Train vs Test Accuracy', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = 'Model',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = 'Feature Importance',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot feature importance as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    df_sorted = importance_df.sort_values('Importance', ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))

    ax.barh(range(len(df_sorted)), df_sorted['Importance'].values, color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['Feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_raw_accelerometer(
    acc: pd.DataFrame,
    gps: Optional[pd.DataFrame] = None,
    n_samples: int = 500,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot raw accelerometer data with Kalman filtered overlay.

    Args:
        acc: Accelerometer DataFrame with timestamp, acc_x, acc_x_kf, etc.
        gps: Optional GPS DataFrame with timestamp and speed
        n_samples: Number of samples to plot
        save_path: Optional path to save figure
        figsize: Figure size
    """
    from src.classification.data import compute_acceleration_magnitude

    # Compute magnitude if not present
    if 'acc_magnitude' not in acc.columns:
        acc = acc.copy()
        acc['acc_magnitude'] = compute_acceleration_magnitude(
            acc['acc_x_kf'].values, acc['acc_y_kf'].values, acc['acc_z_kf'].values
        )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # X-axis (Braking/Acceleration)
    ax = axes[0, 0]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_x'][:n_samples],
            alpha=0.4, label='Raw', color='lightblue')
    ax.plot(acc['timestamp'][:n_samples], acc['acc_x_kf'][:n_samples],
            label='Kalman Filtered', color='blue', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.3, color='red', linestyle=':', alpha=0.7, label='Brake threshold')
    ax.set_title('X-Axis: Longitudinal (Braking/Acceleration)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    # Y-axis (Turning)
    ax = axes[0, 1]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_y'][:n_samples],
            alpha=0.4, label='Raw', color='lightgreen')
    ax.plot(acc['timestamp'][:n_samples], acc['acc_y_kf'][:n_samples],
            label='Kalman Filtered', color='green', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Turn threshold')
    ax.axhline(y=-0.3, color='orange', linestyle=':', alpha=0.7)
    ax.set_title('Y-Axis: Lateral (Turning)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.legend(loc='upper right', fontsize=8)

    # Magnitude
    ax = axes[1, 0]
    ax.plot(acc['timestamp'][:n_samples], acc['acc_magnitude'][:n_samples],
            color='purple', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Gravity baseline')
    ax.set_title('Acceleration Magnitude', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Acceleration| (g)')
    ax.legend(loc='upper right', fontsize=8)

    # Speed from GPS
    if gps is not None and len(gps) > 0:
        ax = axes[1, 1]
        ax.plot(gps['timestamp'], gps['speed'], color='navy', linewidth=1)
        ax.set_title('Speed (from GPS)', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        mean_speed = gps['speed'].mean()
        ax.axhline(y=mean_speed, color='red', linestyle='--', alpha=0.5,
                   label=f"Mean: {mean_speed:.1f}")
        ax.legend(loc='upper right', fontsize=8)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_behavior_comparison(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot feature comparison across behavior classes.

    Args:
        df: DataFrame with features and target
        feature_cols: List of features to compare
        target_col: Name of behavior column
        save_path: Optional path to save figure
        figsize: Figure size
    """
    behavior_stats = df.groupby(target_col)[feature_cols].mean()

    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    colors = ['#2ecc71', '#e74c3c', '#f1c40f']

    for ax, feat in zip(axes[:n_features], feature_cols):
        if feat in behavior_stats.columns:
            behavior_stats[feat].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
            ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)

    # Hide unused axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

