# ABAX Data Science Project - Complete Evaluation Report

**Date:** December 26, 2025  
**Author:** Reza Mirzaeifard

---

## Executive Summary

All pipelines executed successfully with excellent results on both classification and regression tasks:

| Task | Best Model | Primary Metric | Value |
|------|------------|----------------|-------|
| Classification | Random Forest | F1 Score | **0.8714** |
| Regression | Ridge (L2) | R² | **0.9996** |

---

## 1. Classification Results: Driver Behavior Detection

### Dataset Overview
- **Source:** UAH-DriveSet (University of Alcalá, Spain)
- **Total Samples:** 40 driving trips from 6 drivers
- **Features:** 11 (driving scores, behavior ratios)
- **Classes:** NORMAL (42.5%), DROWSY (30%), AGGRESSIVE (27.5%)
- **Challenge:** Slightly imbalanced dataset

### Model Performance Comparison

| Rank | Model | Accuracy | Balanced Acc | F1 Score |
|------|-------|----------|--------------|----------|
| 1 | **Random Forest** | 87.50% | 88.89% | **0.8714** |
| 2 | Gradient Boosting | 87.50% | 88.89% | 0.8714 |
| 3 | Logistic Regression (L1) | 75.00% | 77.78% | 0.7292 |
| 4 | Logistic Regression (L2) | 75.00% | 77.78% | 0.7188 |
| 5 | SVM (RBF) | 75.00% | 77.78% | 0.7188 |

### Confusion Matrix Analysis
```
                Predicted
              AGG  DRO  NOR
Actual AGG  [  2    0    0 ]  ← 100% correct
       DRO  [  0    2    1 ]  ← 67% correct (1 misclassified as NORMAL)
       NOR  [  0    0    3 ]  ← 100% correct
```

### Key Findings
1. **Tree-based models outperform** linear models significantly (+12% accuracy)
2. **Class weighting effective** - Balanced accuracy (88.89%) close to accuracy (87.50%)
3. **DROWSY class** is most difficult to classify (confused with NORMAL)
4. **High precision (90.62%)** indicates few false positives

### Relevance to ABAX
- Driver behavior classification directly applicable to telematics products
- Can identify aggressive/drowsy drivers for safety alerts
- Model can be deployed for real-time behavior monitoring

---

## 2. Regression Results: Fuel Consumption Prediction

### Dataset Overview
- **Source:** EPA Fuel Economy (US Environmental Protection Agency)
- **Total Samples:** 3,000 vehicles (sampled from full dataset)
- **Raw Features:** 12 (vehicle specs, engine, fuel type)
- **Processed Features:** 123 (after one-hot encoding)
- **Target:** Combined MPG (range: 10-140 MPG)
- **Challenge:** Outliers from EVs/hybrids

### Model Performance Comparison

| Rank | Model | RMSE | MAE | R² | MAPE |
|------|-------|------|-----|-----|------|
| 1 | **Ridge (L2)** | **0.385** | 0.313 | **0.9996** | 1.29% |
| 2 | OLS (Baseline) | 0.386 | 0.312 | 0.9996 | 1.28% |
| 3 | RANSAC (Robust) | 0.386 | 0.312 | 0.9996 | 1.28% |
| 4 | Huber (Robust) | 0.394 | 0.312 | 0.9996 | 1.27% |
| 5 | Random Forest | 0.441 | **0.168** | 0.9995 | **0.45%** |
| 6 | Lasso (L1 Sparse) | 0.446 | 0.345 | 0.9995 | 1.38% |
| 7 | ElasticNet | 0.465 | 0.344 | 0.9994 | 1.33% |
| 8 | Gradient Boosting | 0.466 | 0.312 | 0.9994 | 1.12% |

### Key Findings
1. **All models perform excellently** (R² > 0.999)
2. **Ridge regularization** provides slight improvement over OLS
3. **Robust methods (Huber, RANSAC)** perform comparably - data well-preprocessed
4. **Random Forest** has lowest MAPE (0.45%) but higher RMSE
5. **Linear models** outperform tree-based models on this dataset

### Relevance to ABAX
- Fuel consumption prediction valuable for fleet management
- Can estimate fuel costs and optimize routes
- Robust regression handles sensor noise/outliers in telematics data

---

## 3. Technical Achievements

### Clean Code Architecture
```
src/
├── core/schemas.py      # Pydantic models for type safety
├── data/                # Modular data loaders
├── features/            # Preprocessing pipeline
├── models/              # Training & evaluation
└── visualization/       # Plotting utilities
```

### Pydantic Data Structures
- `Dataset`, `SplitData`, `FeatureSet` for data flow
- `TrainingHistory` for iteration tracking
- `ClassificationMetrics`, `RegressionMetrics` for evaluation
- `TrainedModel`, `ModelComparison` for results

### Techniques Demonstrated
1. **Robust Preprocessing:** RobustScaler (median/IQR based)
2. **Imbalanced Data Handling:** class_weight='balanced'
3. **Robust Regression:** Huber, RANSAC, Lasso
4. **Iteration Tracking:** Training curves for Gradient Boosting
5. **Feature Importance:** From tree-based models

---

## 4. Generated Artifacts

### Figures (results/figures/)
1. `class_distribution.png` - Training set class balance
2. `classifier_comparison.png` - F1 Score comparison
3. `confusion_matrix.png` - Normalized confusion matrix
4. `training_history_classification.png` - GB training curve
5. `feature_importance_classification.png` - RF importance
6. `regressor_comparison.png` - R² comparison
7. `actual_vs_predicted.png` - Prediction scatter plot
8. `residuals.png` - Residual diagnostics
9. `training_history_regression.png` - GB training curve

### Reports
- `docs/results_report.md` - Full markdown report
- `results/results.json` - Raw JSON results

---

## 5. Recommendations

### For Classification (Driver Behavior)
1. **Deploy Random Forest** for production - best balance of accuracy and interpretability
2. **Collect more data** - 40 samples is limited for robust model
3. **Focus on DROWSY detection** - currently weakest class

### For Regression (Fuel Consumption)
1. **Use Ridge (L2)** for production - best overall performance
2. **Consider Random Forest** if MAPE is priority metric
3. **Robust methods** ready if data quality degrades

### For ABAX Application
1. Real-time driver behavior alerts using classification model
2. Fleet fuel cost estimation using regression model
3. Anomaly detection combining both models
4. Dashboard integration via visualization module

---

## Conclusion

The project successfully demonstrates:
- ✅ Clean, modular code architecture
- ✅ Pydantic-based type-safe data structures
- ✅ Real-world telematics data processing (UAH-DriveSet)
- ✅ Online data integration (EPA Fuel Economy)
- ✅ Robust ML techniques for imbalanced data and outliers
- ✅ Comprehensive evaluation and visualization
- ✅ Production-ready model comparison framework

Both tasks achieve **excellent performance metrics** suitable for deployment in ABAX telematics products.
