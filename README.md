# ABAX Data Science Project

**Author:** Reza Mirzaeifard  
**Date:** December 2025  
**Purpose:** Technical assessment for ABAX Data Scientist position

---

## 📋 Project Overview

This project demonstrates end-to-end machine learning workflows for:
1. **Driver Behavior Classification** (UAH-DriveSet)
2. **Fuel Economy Prediction** (EPA dataset)

### Key Features
- ✅ Real-world datasets (not synthetic)
- ✅ Proper train/test splitting (driver-level for classification)
- ✅ Robust regression techniques (Huber, Ridge)
- ✅ Deep learning (CNN with learning curves)
- ✅ Production-ready OOP structure
- ✅ Comprehensive notebooks with visualizations

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Run Notebooks

```bash
jupyter lab notebooks/
```

**Important:** Select the **ABAX (.venv)** kernel in JupyterLab.

### 3. Run Full Pipeline

```bash
python main.py
```

---

## 📁 Project Structure

```
ABAX/
├── data/                      # Raw and processed datasets
│   ├── processed/             # Cleaned data (CSV)
│   └── UAH-DRIVESET-v1/      # Raw driving telemetry
├── docs/                      # Documentation
│   ├── data_science_technical_task.md
│   ├── evaluation_report.md
│   └── results_report.md
├── logs/                      # Experiment logs
│   └── cnn_experiments/       # CNN training logs
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda_classification.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_eda_regression.ipynb
│   └── 04_regression.ipynb
├── results/                   # Model outputs
│   ├── results.json
│   └── figures/               # Plots and visualizations
├── scripts/                   # Utility scripts
├── src/                       # Source code
│   ├── core/                  # Pydantic schemas
│   ├── data/                  # Data loaders
│   ├── features/              # Preprocessing
│   ├── models/                # ML models
│   └── visualization/         # Plotting utilities
├── tests/                     # Unit tests
├── main.py                    # Main pipeline
└── pyproject.toml            # Dependencies
```

---

## 🎯 Tasks

### Task 1: Driver Behavior Classification

**Dataset:** UAH-DriveSet (40 trips, 6 drivers)  
**Goal:** Predict NORMAL / DROWSY / AGGRESSIVE

**Key Innovation:** Driver-level splitting (hold out D6) ensures generalization to new drivers.

**Results:**
- Random Forest: **92% accuracy** on held-out driver
- Logistic Regression: 88% accuracy
- 1D CNN: 90% accuracy

### Task 2: Fuel Economy Prediction

**Dataset:** EPA Fuel Economy (5,000 vehicles, 2015-2024)  
**Goal:** Predict combined MPG (continuous regression)

**Key Techniques:**
- Huber Regressor (robust to outliers)
- Ridge Regression (handles multicollinearity)
- Target encoding (high-cardinality categoricals)

**Results:**
- Random Forest: **R²=0.94** (excellent!)
- Huber Regressor: R²=0.89
- Linear Regression: R²=0.87

---

## 🔧 Technical Stack

### Core
- Python 3.9-3.11
- NumPy 1.23.5
- Pandas 2.0.3
- Scikit-learn 1.6.1

### Deep Learning
- TensorFlow 2.13.0 (macOS Intel compatible)

### Visualization
- Matplotlib 3.9.4
- Seaborn 0.13.2

### Data Quality
- Pydantic 1.x (type-safe schemas)

---

## 📊 Notebooks

### 01_eda_classification.ipynb
- UAH-DriveSet exploration
- Class distribution analysis
- Feature engineering philosophy
- Driver-level splitting rationale

### 02_classification.ipynb
- Logistic Regression baseline
- Random Forest (best: 92%)
- 1D CNN with learning curves
- Confusion matrix & feature importance

### 03_eda_regression.ipynb
- EPA dataset exploration
- Outlier detection (~10%)
- High-cardinality categoricals
- Target vs features analysis

### 04_regression.ipynb
- Linear, Huber, Ridge regressors
- Random Forest (R²=0.94)
- Residual analysis
- Feature importance

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 💼 Business Relevance (ABAX Context)

### Fleet Management
- **Driver Safety:** Identify high-risk drivers proactively
- **Fuel Optimization:** Predict consumption for cost forecasting
- **Coaching:** Targeted feedback for drowsy/aggressive drivers

### Sustainability
- **ESG Reporting:** CO2 estimation from fuel economy
- **Route Optimization:** Based on vehicle characteristics

### Insurance
- **Risk Assessment:** Behavior-based premium adjustment

---

## 🐛 Troubleshooting

### TensorFlow Import Error

If you see `ValueError: numpy.dtype size changed`:

1. **Restart Jupyter kernel:** `Kernel → Restart Kernel`
2. **Select correct kernel:** `Kernel → Change Kernel → ABAX (.venv)`
3. **Verify environment:**
   ```bash
   .venv/bin/python -c "import tensorflow; print(tensorflow.__version__)"
   ```
   Should print: `2.13.0`

### NumPy Version Mismatch

```bash
# Force clean sync
rm uv.lock
uv sync
```

---

## 📝 Notes

### Why NumPy 1.23.5?
TensorFlow 2.13 (last version supporting macOS Intel) requires NumPy 1.22-1.24.

### Why Pydantic 1.x?
TensorFlow 2.13 requires `typing-extensions<4.6`, incompatible with Pydantic 2.x.

### Why Driver-Level Splitting?
Random splits leak information (same driver in train/test). Holding out entire drivers ensures model generalizes to NEW drivers in production.

---

## 📧 Contact

**Reza Mirzaeifard**  
Applying for: Data Scientist @ ABAX

---

## ✅ Checklist

- [x] Classification task with real-world data
- [x] Regression task with robust techniques
- [x] Outlier handling (Huber regressor)
- [x] Categorical encoding (target encoding)
- [x] Deep learning (CNN with learning curves)
- [x] Production-ready structure (OOP + Pydantic)
- [x] Comprehensive notebooks with visualizations
- [x] Driver-level splitting for generalization
- [x] Business context (ABAX relevant)

---

**Status:** ✅ Complete and ready for review!

