#!/bin/bash
# MARTI Environment Setup - Single command to get started
# Uses exact working versions from requirements_uv.txt

set -e

echo "🚀 Setting up MARTI environment with UV..."

# Check UV installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Install with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ UV found: $(uv --version)"

# Create Python 3.11 environment
echo "📦 Creating Python 3.11 virtual environment..."
uv venv --python 3.11

# Activate
source .venv/bin/activate || { echo "❌ Failed to activate"; exit 1; }
echo "✓ Environment activated"

# Install build dependencies first (required for flash-attn with no-build-isolation)
echo "🛠️  Installing build dependencies..."
uv pip install setuptools wheel packaging psutil ninja

# Install PyTorch first (required for building flash-attn)
echo "🔥 Installing PyTorch 2.6.0..."
uv pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# Install exact working versions
echo "🔧 Installing remaining dependencies..."
echo "   This takes ~5 minutes (flash-attn build)..."
uv pip install -r requirements_uv.txt --no-build-isolation

# Verify
echo "🧪 Verifying installation..."
if python -c "from flash_attn.utils.distributed import all_gather; import weave; print('✓ All imports successful')"; then
    echo ""
    echo "🎉 SUCCESS! Environment ready."
    echo ""
    echo "Next steps:"
    echo "  source .venv/bin/activate"
    echo "  bash scripts/run_train_mathchat_coach_6gpu_direct.sh MODEL_PATH"
    echo ""
    echo "📖 See UV_SETUP.md for details"
else
    echo "⚠️  Verification failed - check errors above"
    exit 1
fi