# ABAX Data Science Technical Assessment

<p align="center">
  <strong>Driver Behavior Classification & Fuel Economy Prediction</strong><br>
  <em>Complete Machine Learning Pipeline from Raw Sensors to Production Models</em>
</p>

---

**Author:** Reza Mirzaeifard  
**Email:** [reza.mirzaeifard@gmail.com](mailto:reza.mirzaeifard@gmail.com)  
**Date:** December 2025  

---

## 📄 Main Deliverable

> **📕 [ABAX Technical Report (PDF)](docs/ABAX_Technical_Report.pdf)**
>
> A comprehensive 20+ page LaTeX report covering:
> - Complete exploratory data analysis with visualizations
> - Feature engineering from raw sensor data (24 features)
> - 18 classification models + 13 regression models comparison
> - Advanced regularization (MCP, SCAD penalties)
> - MLP neural network with proper data normalization
> - Driver-level evaluation (D6 held out for testing)
> - Failure analysis with mitigation strategies
> - Production deployment recommendations

---

## 🎯 Project Summary

This project demonstrates end-to-end machine learning workflows for two telematics applications critical to ABAX's business:

### Task 1: Driver Behavior Classification
- **Dataset:** UAH-DriveSet (40 trips, 6 drivers)
- **Goal:** Classify driving as NORMAL, DROWSY, or AGGRESSIVE
- **Best Result:** **100% accuracy** with Gradient Boosting (87.5% with Random Forest, KNN)
- **Key Innovation:** Raw sensor features only (no circular logic from pre-computed scores)
- **Data Preprocessing:** NORMAL1/NORMAL2 labels normalized to single NORMAL class

### Task 2: Fuel Economy Prediction
- **Dataset:** EPA Fuel Economy (~5,000 vehicles)
- **Goal:** Predict combined MPG from vehicle specifications
- **Best Result:** **R² = 0.94, RMSE = 4.5 MPG** with Random Forest

---

## 🏆 Key Achievements

| Achievement | Description |
|-------------|-------------|
| **Raw Sensor Features** | Extracted 24 features from GPS/accelerometer, avoiding circular logic |
| **Driver-Level Splitting** | D6 completely held out—tests generalization to new customers |
| **18 Classification Models** | Including MCP, SCAD, MLP, SVM, Random Forest, KNN |
| **Advanced Regularization** | Implemented MCP and SCAD for nearly unbiased sparse estimates |
| **MLP Neural Network** | Multi-Layer Perceptron with proper StandardScaler normalization |
| **Clean Code Architecture** | Modular `src/` package with testable, reusable functions |
| **Comprehensive Analysis** | Feature importance, failure cases, production recommendations |

---

## 📊 Results Summary

### Classification Results (D6 Held Out)

| Model | Train Acc | Test Acc | F1-Score | Overfit Gap |
|-------|-----------|----------|----------|-------------|
| **Gradient Boosting** | 100% | **100%** | 1.000 | **0.000** |
| KNN (k=7) | 100% | 87.5% | 0.863 | 0.125 |
| Random Forest | 100% | 87.5% | 0.875 | 0.125 |
| Extra Trees | 100% | 87.5% | 0.875 | 0.125 |
| AdaBoost | 100% | 87.5% | 0.863 | 0.125 |
| Logistic (L1) | 84.4% | 75.0% | 0.767 | 0.094 |
| Logistic (SCAD) | 75.0% | 75.0% | 0.767 | **0.000** |
| MLP Neural Network | 87.5% | 62.5% | 0.630 | 0.250 |

**Key Finding:** Gradient Boosting achieves 100% accuracy; ensemble methods outperform on this dataset with good feature engineering.

### Regression Results

| Model | R² | RMSE (MPG) | MAE (MPG) |
|-------|-----|------------|-----------|
| **Random Forest** | **0.938** | 4.52 | 2.31 |
| Gradient Boosting | 0.932 | 4.70 | 2.58 |
| Ridge (L2) | 0.802 | 8.05 | 4.47 |

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

**Notebooks:**
- `01_project_overview.ipynb` - Project introduction and data overview
- `02_classification.ipynb` - Complete classification pipeline (800+ lines)
- `03_eda_regression.ipynb` - Regression EDA
- `04_regression.ipynb` - Complete regression pipeline

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Compile LaTeX Report

```bash
cd docs && bash compile_report.sh
```

---

## 📁 Project Structure

```
ABAX/
├── 📕 docs/                          # Documentation
│   ├── ABAX_Technical_Report.pdf     # ⭐ MAIN DELIVERABLE
│   ├── ABAX_Technical_Report.tex     # LaTeX source (850 lines)
│   └── compile_report.sh             # Build script
│
├── 📓 notebooks/                     # Jupyter notebooks
│   ├── 01_project_overview.ipynb     # Introduction
│   ├── 02_classification.ipynb       # Classification pipeline
│   ├── 03_eda_regression.ipynb       # Regression EDA
│   └── 04_regression.ipynb           # Regression pipeline
│
├── 📊 results/                       # Outputs
│   ├── results.json                  # Model metrics
│   └── figures/                      # 30+ visualizations
│
├── 🔧 src/                           # Production-ready code
│   ├── classification/               # Classification module
│   │   ├── __init__.py               # Clean API (run_logo_cv, etc.)
│   │   ├── data.py                   # Data loading, feature extraction
│   │   ├── sparse_models.py          # MCP, SCAD implementations
│   │   ├── types.py                  # ClassificationResult, DataSplit
│   │   └── visualization.py          # All plotting functions
│   ├── models/                       # Model implementations
│   │   ├── simple_nn.py              # PyTorch Neural Network
│   │   ├── cnn.py                    # CNN classifier
│   │   └── resnet.py                 # ResNet classifier
│   └── utils/                        # Utilities
│
├── 🧪 tests/                         # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── 📦 data/                          # Datasets
│   ├── processed/                    # Cleaned CSVs
│   └── UAH-DRIVESET-v1/              # Raw driving data
│
└── pyproject.toml                    # Dependencies
```

---

## 🔬 Technical Details

### Feature Engineering (36 Raw Sensor Features)

| Category | Features | Physical Meaning |
|----------|----------|------------------|
| Speed | mean, std, max, min | Driving intensity |
| Speed Changes | change_mean, change_std | Acceleration patterns |
| Course/Heading | change_mean, std, max | Lane changes, turns |
| Acceleration | X/Y axis mean, std | Core behavior signal |
| Jerk | x_std, y_std | **Smoothness indicator** |
| Event Counts | brake, turn, hard events | Discrete summaries |

**Why Jerk Matters:** Jerk = d(acceleration)/dt. Aggressive drivers have high jerk variance because they brake suddenly, accelerate abruptly, and make sharp steering corrections.

### Data Splitting Strategy

```
Training: 32 samples (80%) from drivers D1-D5
Testing:  8 samples (20%) = D6 trips + stratified samples

⚠️ D6 is NEVER seen during training (production-realistic evaluation)
```

### Neural Network Architecture

```
Input (36 features)
  → StandardScaler (zero mean, unit variance)  ← CRITICAL
  → BatchNorm1d(36)
  → Linear(36, 64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64, 32) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(32, 3) → Softmax
```

**Why Normalization Matters:** Without it, features with large values (speed in km/h) dominate gradient updates while smaller features (jerk) are ignored.

---

## 📈 Key Visualizations

All figures are in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `raw_accelerometer_data.png` | Sensor comparison: AGGRESSIVE vs NORMAL vs DROWSY |
| `class_distribution.png` | Class balance visualization |
| `classifier_comparison.png` | 18-model comparison (train vs test accuracy) |
| `confusion_matrix_classification.png` | Error analysis |
| `feature_importance_classification.png` | Top features with physical interpretation |
| `nn_learning_curves_classification.png` | Neural network training dynamics |
| `regressor_comparison.png` | Regression model comparison |
| `actual_vs_predicted.png` | Prediction quality |
| `residuals.png` | Residual analysis |

---

## 💼 Business Impact

| Application | How This Work Helps |
|-------------|---------------------|
| **Safety Monitoring** | Real-time alerts for aggressive/drowsy driving |
| **Insurance Pricing** | Usage-based premiums from actual behavior |
| **Driver Coaching** | Personalized feedback based on specific behaviors |
| **Fleet Optimization** | Data-driven vehicle selection for fuel efficiency |
| **Environmental Compliance** | Carbon footprint tracking |

---

## 🔧 Technical Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11, NumPy, Pandas, Scikit-learn |
| **Deep Learning** | PyTorch 2.x |
| **Visualization** | Matplotlib, Seaborn |
| **Report** | LaTeX (tectonic compiler) |
| **Package Management** | uv |

---

## 🐛 Troubleshooting

### Kernel Selection
Select the **ABAX (.venv)** kernel in JupyterLab for correct dependencies.

### Reinstall Dependencies
```bash
rm uv.lock && uv sync
```

### Compile Report
```bash
# Requires tectonic or pdflatex
cd docs && bash compile_report.sh
```

---

## ✅ Deliverables Checklist

- [x] **Technical Report** - Comprehensive PDF (20+ pages)
- [x] **Classification** - 18 models, 87.5% accuracy
- [x] **Regression** - 13 models, R² = 0.94
- [x] **Driver-Level Splitting** - D6 held out
- [x] **Raw Sensor Features** - 36 features, no circular logic
- [x] **Advanced Regularization** - MCP, SCAD implemented
- [x] **Neural Network** - PyTorch with proper normalization
- [x] **Failure Analysis** - Misclassification cases documented
- [x] **Production Recommendations** - Deployment guidance
- [x] **Clean Code** - Modular `src/` architecture
- [x] **Visualizations** - 30+ professional figures
- [x] **Tests** - Unit tests passing

---

## 📧 Contact

**Reza Mirzaeifard**  
📧 [reza.mirzaeifard@gmail.com](mailto:reza.mirzaeifard@gmail.com)

---

<p align="center">
  <strong>✅ Complete and ready for review!</strong>
</p>

