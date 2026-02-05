#!/bin/bash
# =========================================
# RunPod AI Environment Bootstrap Script
# =========================================

echo "=== Updating system packages ==="
apt update && apt upgrade -y

echo "=== Installing essential system packages ==="
apt install -y git build-essential cmake curl wget unzip zip tmux python3-pip python3-venv python3-dev

echo "=== Upgrading pip ==="
python3 -m pip install --upgrade pip

echo "=== Creating Python virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Installing PyTorch with CUDA support ==="
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Python requirements ==="
# Make sure requirements.txt uses --extra-index-url instead of --index-url
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Installation complete ==="
echo "Activate venv with: source .venv/bin/activate"
echo "Check GPU: nvidia-smi"
