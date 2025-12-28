#!/bin/bash
# Restart Jupyter Kernel - Multiple Methods

echo "🔄 Jupyter Kernel Restart Options"
echo "=================================="
echo ""

# Method 1: Kill Python processes associated with Jupyter
echo "Option 1: Kill all Jupyter Python kernels"
echo "   pkill -f 'ipykernel_launcher'"
echo ""

# Method 2: Use Jupyter API (if you have jupyter_client)
echo "Option 2: Using jupyter commands"
echo "   jupyter kernelspec list  # List available kernels"
echo ""

# Method 3: Just restart the specific project kernel
echo "Option 3: Kill kernels using this project's venv"
echo "   pkill -f '.venv/bin/python.*ipykernel'"
echo ""

read -p "Which option? (1/2/3 or 'q' to quit): " choice

case $choice in
    1)
        echo "⚠️  Killing ALL Jupyter kernels..."
        pkill -f 'ipykernel_launcher' 2>/dev/null
        echo "✅ Done! Kernels will auto-restart when you run a cell."
        ;;
    2)
        echo "📋 Available kernels:"
        jupyter kernelspec list 2>/dev/null || echo "jupyter not in PATH"
        ;;
    3)
        echo "🎯 Killing kernels for this project..."
        pkill -f "$(pwd)/.venv/bin/python.*ipykernel" 2>/dev/null
        echo "✅ Done! Kernel will auto-restart when you run a cell."
        ;;
    q|Q)
        echo "Exiting..."
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "💡 TIP: In JupyterLab/PyCharm, just use the menu:"
echo "   - JupyterLab: Kernel → Restart Kernel"
echo "   - PyCharm: Click 'Restart' button in notebook toolbar"
echo "   - VS Code: Click 'Restart' in kernel dropdown"

