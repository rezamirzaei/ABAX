#!/usr/bin/env python3
"""Generate Figure 1: Informative Raw Accelerometer Data Comparison."""

import sys
sys.path.insert(0, '/Users/rezami/PycharmProjects/ABAX')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

COLORS = {
    'AGGRESSIVE': '#e74c3c',
    'NORMAL': '#27ae60',
    'DROWSY': '#3498db',
}

DATA_DIR = Path('/Users/rezami/PycharmProjects/ABAX/data/UAH-DRIVESET-v1')
FIGURES_DIR = Path('/Users/rezami/PycharmProjects/ABAX/results/figures')

def load_raw_accelerometer(trip_path):
    """Load raw accelerometer data."""
    acc_file = trip_path / 'RAW_ACCELEROMETERS.txt'
    if not acc_file.exists():
        print(f"  File not found: {acc_file}")
        return None
    try:
        # File format: timestamp flag raw_x raw_y raw_z kf_x kf_y kf_z gravity_x gravity_y gravity_z
        # Values are in g-forces (multiply by 9.81 for m/s²)
        df = pd.read_csv(acc_file, sep=r'\s+', header=None)
        if len(df.columns) >= 8:
            df.columns = ['timestamp', 'flag', 'raw_x', 'raw_y', 'raw_z',
                         'kf_x', 'kf_y', 'kf_z'] + [f'col_{i}' for i in range(8, len(df.columns))]
            # Use Kalman filtered values (columns 5,6,7) which are cleaner
            # Convert from g-forces to m/s² for better visualization
            df['acc_x'] = df['kf_x'] * 9.81
            df['acc_y'] = df['kf_y'] * 9.81
            df['acc_z'] = df['kf_z'] * 9.81
        return df
    except Exception as e:
        print(f"Error loading {acc_file}: {e}")
        return None

def get_all_trips():
    """Get all trips from the dataset."""
    trips = []
    for driver_dir in sorted(DATA_DIR.glob('D*')):
        if not driver_dir.is_dir():
            continue
        for trip_dir in sorted(driver_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            # Format: 20151111125233-24km-D1-AGGRESSIVE-MOTORWAY
            parts = trip_dir.name.split('-')
            if len(parts) >= 4:
                behavior = parts[3].upper()
                trips.append({'path': trip_dir, 'driver': driver_dir.name, 'behavior': behavior})
    return trips

def generate_figure1():
    """Generate informative Figure 1 showing all 3 behaviors."""
    print("Loading trips...")
    trips = get_all_trips()
    print(f"Found {len(trips)} trips")

    behaviors_found = set(t['behavior'] for t in trips)
    print(f"Behaviors in dataset: {behaviors_found}")

    # Get one trip of each behavior
    sample_trips = {}
    for behavior in ['AGGRESSIVE', 'NORMAL', 'DROWSY']:
        for trip in trips:
            trip_behavior = trip['behavior'].upper()
            if trip_behavior.startswith(behavior) and behavior not in sample_trips:
                print(f"  Trying {trip['driver']}/{trip_behavior}...")
                acc = load_raw_accelerometer(trip['path'])
                if acc is not None and len(acc) > 100:
                    sample_trips[behavior] = {'trip': trip, 'acc': acc}
                    print(f"  ✓ Found {behavior}: {trip['driver']}, {len(acc)} samples")
                    break

    if len(sample_trips) < 3:
        print(f"Warning: Only found {len(sample_trips)} behaviors")
        return

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    behaviors = ['AGGRESSIVE', 'NORMAL', 'DROWSY']

    for idx, behavior in enumerate(behaviors):
        if behavior not in sample_trips:
            print(f"Skipping {behavior} - no data")
            continue

        acc = sample_trips[behavior]['acc']
        color = COLORS[behavior]

        # Take 500 samples from middle of trip (~10 seconds at 50Hz)
        start_idx = len(acc) // 4  # Start from 25% into trip
        acc = acc.iloc[start_idx:start_idx+500].copy().reset_index(drop=True)

        # Create time axis (relative seconds)
        time = np.arange(len(acc)) * 0.1  # ~10Hz sampling after processing

        # Left plot: Raw X, Y, Z axes
        ax1 = axes[idx, 0]
        ax1.plot(time, acc['acc_x'].values, label='X (longitudinal)', alpha=0.9, linewidth=1.2, color='#3498db')
        ax1.plot(time, acc['acc_y'].values, label='Y (lateral)', alpha=0.9, linewidth=1.2, color='#e67e22')
        ax1.plot(time, acc['acc_z'].values, label='Z (vertical)', alpha=0.9, linewidth=1.2, color='#9b59b6')
        ax1.set_ylabel('Acceleration (m/s²)', fontsize=11)
        ax1.set_title(f'{behavior} Driving - Accelerometer (Kalman Filtered)', fontweight='bold', color=color, fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Set reasonable y-limits
        y_max = max(abs(acc['acc_x'].max()), abs(acc['acc_y'].max()), abs(acc['acc_z'].max()), 1) * 1.2
        ax1.set_ylim(-y_max, y_max)

        # Right plot: Jerk (rate of change of acceleration)
        ax2 = axes[idx, 1]

        # Calculate jerk
        dt = 0.1  # Approximate time step
        jerk_x = np.abs(np.diff(acc['acc_x'].values)) / dt
        jerk_y = np.abs(np.diff(acc['acc_y'].values)) / dt
        time_jerk = time[1:]

        # Combined jerk magnitude
        jerk_combined = np.sqrt(jerk_x**2 + jerk_y**2)

        # Smooth for visualization
        window = 5
        jerk_smooth = np.convolve(jerk_combined, np.ones(window)/window, mode='valid')
        time_smooth = time_jerk[:len(jerk_smooth)]

        # Plot
        ax2.fill_between(time_smooth, 0, jerk_smooth, alpha=0.4, color=color)
        ax2.plot(time_smooth, jerk_smooth, color=color, alpha=0.9, linewidth=1.5)

        # Threshold line
        threshold = np.percentile(jerk_smooth, 90)
        ax2.axhline(y=threshold, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label=f'90th pctl: {threshold:.1f}')

        # Count harsh events
        n_events = np.sum(jerk_smooth > threshold)

        ax2.set_ylabel('Jerk Magnitude (m/s³)', fontsize=11)
        ax2.set_title(f'{behavior} - Driving Smoothness (Jerk)', fontweight='bold', color=color, fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(0, min(np.max(jerk_smooth) * 1.3, np.percentile(jerk_smooth, 99) * 2))

        # Stats annotation
        stats_text = f'Mean: {np.mean(jerk_smooth):.1f}\nMax: {np.max(jerk_smooth):.1f}\nHarsh events: {n_events}'
        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        if idx == 2:  # Bottom row
            ax1.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle('Raw Sensor Data Comparison: AGGRESSIVE vs NORMAL vs DROWSY Driving\n'
                 '(Left: Kalman-filtered acceleration | Right: Jerk magnitude - higher = harsher driving)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    save_path = FIGURES_DIR / 'raw_accelerometer_data.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n✅ Saved: {save_path}")

if __name__ == '__main__':
    generate_figure1()

