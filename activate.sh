#!/bin/bash
# Virtual Environment Activation Script
# Usage: source activate.sh

echo "🐍 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
pip list
echo ""
echo "🚀 To run the pipeline: python main.py"
