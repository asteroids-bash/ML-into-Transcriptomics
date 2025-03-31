#!/bin/bash

set -e  # Exit if any command fails

echo "🔧 Setting up your ML project with correct Python & pip..."

# Create project folder if not exists


# Create virtual environment
echo "🐍 Creating virtual environment (.venv)..."
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Show which Python and pip are being used
echo "📍 Python interpreter: $(which python)"
echo "📦 Pip location: $(which pip)"
echo "🧪 Python version: $(python --version)"
echo "🧪 Pip version: $(pip --version)"

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "📄 Creating requirements.txt..."
    cat > requirements.txt <<EOL
pandas
numpy
xgboost
lightgbm
scikit-learn
imbalanced-learn
biomart
EOL
fi

# Install dependencies
source .venv/bin/activate
echo "📦 Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. Virtual environment ready, dependencies installed!"
