#!/bin/bash
# Fix Pandas/NumPy/PyTorch version compatibility

echo "🔧 Fixing environment with compatible package versions..."
echo ""

cd "$(dirname "$0")"

# Clear Python cache
echo "0️⃣  Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove virtual environment and lock file to force clean install
echo "1️⃣  Removing old environment..."
rm -rf .venv
rm -f uv.lock

# Recreate environment with pinned versions
echo "2️⃣  Creating fresh environment..."
uv sync

echo ""
echo "✅ Environment fixed!"
echo ""
echo "📋 Installed versions:"
.venv/bin/python -c "import numpy; import pandas; import torch; import sklearn; print(f'NumPy: {numpy.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'Scikit-learn: {sklearn.__version__}')"
echo ""
echo "⚠️  IMPORTANT: Restart your Jupyter Kernel now!"
echo "   In JupyterLab: Kernel → Restart Kernel"

