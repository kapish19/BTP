#!/usr/bin/env bash
set -e

echo "Setting up Python virtual environment for DG-ReID..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Setup complete. Activate with: source .venv/bin/activate"
