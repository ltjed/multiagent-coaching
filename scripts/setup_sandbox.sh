#!/bin/bash
# ============================================================================
# SandboxFusion Setup Script
# ============================================================================
# Prerequisites: Conda must be already installed
# Run this ONCE on a new node to set up SandboxFusion
#
# Usage: bash scripts/setup_sandbox.sh

set -e  # Exit on error

echo "============================================================================"
echo "Setting up SandboxFusion"
echo "============================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo ""
    echo "Please install conda first. Quick setup:"
    echo "  cd ~"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    echo "  \$HOME/miniconda3/bin/conda init bash"
    echo "  source ~/.bashrc"
    exit 1
fi

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Clone SandboxFusion
if [ ! -d ~/SandboxFusion ]; then
    echo "Cloning SandboxFusion repository..."
    cd ~
    git clone https://github.com/bytedance/SandboxFusion.git
    echo "✓ SandboxFusion repository cloned"
else
    echo "✓ SandboxFusion repository already exists"
fi

cd ~/SandboxFusion

# Create sandbox environment
if ! conda env list | grep -q "^sandbox "; then
    echo "Creating 'sandbox' conda environment (Python 3.12)..."
    conda create -n sandbox -y python=3.12
    echo "✓ sandbox environment created"
else
    echo "✓ sandbox environment already exists"
fi

conda activate sandbox

# Install Poetry
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ Poetry installed"
else
    echo "✓ Poetry already installed"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install SandboxFusion dependencies
echo "Installing SandboxFusion dependencies..."
~/.local/bin/poetry install
echo "✓ SandboxFusion dependencies installed"

# Create sandbox-runtime environment
if ! conda env list | grep -q "^sandbox-runtime "; then
    echo "Creating 'sandbox-runtime' conda environment (Python 3.11)..."
    conda create -n sandbox-runtime -y python=3.11
    conda activate sandbox-runtime
    echo "Installing runtime dependencies..."
    pip install -r ~/SandboxFusion/runtime/python/requirements.txt --ignore-requires-python
    echo "Downloading NLTK data..."
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    echo "✓ sandbox-runtime environment created"
else
    echo "✓ sandbox-runtime environment already exists"
fi

# Create required directories
mkdir -p ~/SandboxFusion/docs/build

echo ""
echo "============================================================================"
echo "✓ SandboxFusion setup complete!"
echo "============================================================================"
echo ""
echo "To start SandboxFusion (run in tmux):"
echo "  tmux new -s sandbox"
echo "  cd ~/SandboxFusion"
echo "  conda activate sandbox"
echo "  make run-online"
echo ""
echo "To verify it's working:"
echo "  curl -X POST http://127.0.0.1:8080/run_code \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"language\":\"python\",\"code\":\"print(1+1)\"}'"
echo "============================================================================"
