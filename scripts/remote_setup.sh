#!/usr/bin/env bash
# Run on the Vast.ai instance (via ssh) to prepare the environment for training.
set -euo pipefail

cd ~/finetune-experiment

apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

echo "Setup complete. Verify GPU visibility with: python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
