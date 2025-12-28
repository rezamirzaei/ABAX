#!/bin/bash
# Script to compile the ABAX Technical Report
# Uses tectonic (installed via brew) for easy compilation

# Check if tectonic is available
if ! command -v tectonic &> /dev/null; then
    echo "Error: tectonic is not installed."
    echo "Please install it using: brew install tectonic"
    exit 1
fi

echo "Compiling ABAX_Technical_Report.tex using tectonic..."
tectonic ABAX_Technical_Report.tex

if [ $? -eq 0 ]; then
    echo "Compilation successful! PDF is located at docs/ABAX_Technical_Report.pdf"
else
    echo "Compilation failed. Check the output for details."
fi

