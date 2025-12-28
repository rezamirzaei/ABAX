#!/bin/bash
# Script to compile the ABAX Technical Report
# Requires pdflatex to be installed (e.g., MacTeX on macOS, TeX Live on Linux)

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex is not installed or not in your PATH."
    echo "Please install a LaTeX distribution (e.g., MacTeX for macOS)."
    exit 1
fi

echo "Compiling ABAX_Technical_Report.tex..."
pdflatex -interaction=nonstopmode ABAX_Technical_Report.tex
pdflatex -interaction=nonstopmode ABAX_Technical_Report.tex # Run twice for TOC

if [ $? -eq 0 ]; then
    echo "Compilation successful! PDF is located at docs/ABAX_Technical_Report.pdf"
else
    echo "Compilation failed. Check the log file for details."
fi

