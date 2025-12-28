"""Check generated files."""
import os
from pathlib import Path

print("=" * 60)
print("Checking Generated Files")
print("=" * 60)

# Check PDF
pdf_path = Path("docs/ABAX_Technical_Report.pdf")
if pdf_path.exists():
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"✅ PDF: {pdf_path} ({size_mb:.2f} MB)")
else:
    print(f"❌ PDF not found: {pdf_path}")

# Check key figures
figures = [
    "class_distribution.png",
    "raw_accelerometer_data.png",
    "feature_distributions_classification.png",
    "driver_behavior_distribution.png",
    "correlation_matrix_classification.png",
    "classifier_comparison.png",
    "confusion_matrix_classification.png",
    "feature_importance_classification.png",
    "cnn_learning_curves_classification.png",
]

print("\n📊 Figures:")
for fig in figures:
    path = Path("results/figures") / fig
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"  ✅ {fig} ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ {fig} NOT FOUND")

# Check notebook
nb_path = Path("notebooks/02_classification.ipynb")
if nb_path.exists():
    size_kb = nb_path.stat().st_size / 1024
    print(f"\n✅ Notebook: {nb_path} ({size_kb:.1f} KB)")
else:
    print(f"\n❌ Notebook not found: {nb_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

