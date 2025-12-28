# ABAX Data Science Project

**Author:** Reza Mirzaeifard  
**Date:** December 2025  
**Purpose:** Technical assessment for ABAX Data Scientist position

---

## 📄 Main Deliverable

> **📕 [ABAX Technical Report (PDF)](docs/ABAX_Technical_Report.pdf)**
>
> This is the primary deliverable — a comprehensive LaTeX report covering:
> - Exploratory Data Analysis (EDA) with visualizations
> - Preprocessing mindset and leakage awareness
> - Model selection rationale and comparison
> - Failure analysis with concrete misclassification examples
> - Leave-One-Driver-Out cross-validation for realistic evaluation
> - Production considerations for ABAX deployment

---

## 📋 Project Overview

This project demonstrates end-to-end machine learning workflows for two real-world telematics problems:

1. **Driver Behavior Classification** (UAH-DriveSet)
2. **Fuel Economy Prediction** (EPA dataset)

### Key Features
- ✅ Real-world datasets (not synthetic)
- ✅ Driver-level splitting (prevents data leakage)
- ✅ Leave-One-Driver-Out CV with variance analysis
- ✅ Robust regression techniques (Huber, Ridge)
- ✅ Deep learning (CNN with learning curves)
- ✅ Production-ready OOP structure with Pydantic schemas
- ✅ Comprehensive failure analysis with concrete examples
- ✅ Business decision framing (coaching, procurement, insurance)

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

### 4. Run Tests

```bash
pytest tests/
```

### 5. Compile LaTeX Report

```bash
cd docs && ./compile_report.sh
```

---

## 📁 Project Structure

```
ABAX/
├── data/                      # Raw and processed datasets
│   ├── processed/             # Cleaned data (CSV)
│   └── UAH-DRIVESET-v1/       # Raw driving telemetry
├── docs/                      # Documentation
│   ├── ABAX_Technical_Report.pdf   # 📕 MAIN DELIVERABLE
│   ├── ABAX_Technical_Report.tex   # LaTeX source
│   └── compile_report.sh           # Build script
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda_classification.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_eda_regression.ipynb
│   └── 04_regression.ipynb
├── results/                   # Model outputs
│   ├── results.json
│   └── figures/               # 20+ plots and visualizations
├── src/                       # Source code (production-ready)
│   ├── core/                  # Pydantic schemas
│   ├── data/                  # Data loaders
│   ├── features/              # Preprocessing
│   ├── models/                # ML models
│   └── visualization/         # Plotting utilities
├── tests/                     # Unit tests (20 tests, all passing)
├── main.py                    # Main pipeline
└── pyproject.toml             # Dependencies
```

---

## 🎯 Tasks

### Task 1: Driver Behavior Classification

**Dataset:** UAH-DriveSet (40 trips, 6 drivers)  
**Goal:** Predict NORMAL / DROWSY / AGGRESSIVE

**Key Innovation:** Driver-level splitting ensures generalization to new drivers.

| Evaluation | Accuracy | F1 (weighted) |
|------------|----------|---------------|
| Single split (D6 held out) | 0.875 | 0.871 |
| Leave-One-Driver-Out CV | 0.776 ± 0.066 | 0.759 ± 0.070 |

**Insight:** The CV results show realistic performance with variance across drivers, motivating personalization in production.

### Task 2: Fuel Economy Prediction

**Dataset:** EPA Fuel Economy (5,000 vehicles, 2015-2024)  
**Goal:** Predict combined MPG

**Leakage Check:** City/highway MPG columns explicitly excluded (see Appendix A in report).

| Model | RMSE | R² | MAPE |
|-------|------|-----|------|
| Ridge (L2) | 0.385 | 0.9996 | 1.29% |
| Random Forest | 0.441 | 0.9995 | 0.45% |
| Huber (robust) | 0.394 | 0.9996 | 1.27% |

**Why R² is so high:** EPA ratings are deterministic given vehicle specs (standardized testing). Real-world driving would show more variance.

---

## 📊 Key Visualizations

All figures are saved in `results/figures/`:

| Category | Figures |
|----------|---------|
| Classification EDA | `class_distribution.png`, `correlation_matrix_classification.png`, `driver_behavior_distribution.png` |
| Classification Results | `confusion_matrix_classification.png`, `feature_importance_classification.png`, `cnn_learning_curves_classification.png` |
| Regression EDA | `target_distribution_regression.png`, `correlation_matrix_regression.png`, `target_by_categories_regression.png` |
| Regression Results | `actual_vs_predicted.png`, `residuals.png`, `prediction_intervals.png` |

---

## 💼 Business Decisions Enabled

This work directly supports ABAX business objectives:

- **Driver Coaching:** Use drowsy/aggressive predictions for in-cab alerts (coaching, not punishment)
- **Fleet Procurement:** Rank candidate vehicles by predicted fuel cost
- **Insurance Risk Tiers:** Segment drivers by behavior class for usage-based insurance
- **Route Planning:** Combine behavior + efficiency for fuel-sensitive routing

---

## 🔧 Technical Stack

| Category | Technologies |
|----------|--------------|
| Core | Python 3.9-3.11, NumPy 1.23.5, Pandas 2.0.3, Scikit-learn 1.6.1 |
| Deep Learning | TensorFlow 2.13.0 (macOS Intel compatible) |
| Visualization | Matplotlib 3.9.4, Seaborn 0.13.2 |
| Data Quality | Pydantic 1.x (type-safe schemas) |
| Report | LaTeX (tectonic compiler) |

---

## 🐛 Troubleshooting

### TensorFlow Import Error

If you see `ValueError: numpy.dtype size changed`:

1. Restart Jupyter kernel
2. Select correct kernel: `ABAX (.venv)`
3. Verify: `.venv/bin/python -c "import tensorflow; print(tensorflow.__version__)"`

### NumPy Version Mismatch

```bash
rm uv.lock && uv sync
```

---

## ✅ Checklist

- [x] Classification with real-world UAH-DriveSet
- [x] Regression with EPA Fuel Economy data
- [x] Driver-level splitting (no leakage)
- [x] Leave-One-Driver-Out CV with variance
- [x] Leakage check documented (Appendix A)
- [x] Misclassification case study (Appendix C)
- [x] Robust regression (Huber, RANSAC)
- [x] Deep learning (CNN with learning curves)
- [x] Production-ready OOP + Pydantic
- [x] Business decision framing
- [x] Comprehensive LaTeX report

---

## 📧 Contact

**Reza Mirzaeifard**  
Applying for: Data Scientist @ ABAX

---

**Status:** ✅ Complete and ready for review!
