#!/usr/bin/env bash
# Setup script for HK-LeadLag - Linux / macOS
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "=== Creating virtual environment ==="
python3 -m venv .venv

echo "=== Activating venv and installing dependencies ==="
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "=== Running tests ==="
pytest -q --tb=short || true

echo ""
echo "=== Setup complete. ==="
echo "To activate later: source .venv/bin/activate"
echo "To run the notebook: jupyter lab notebooks/01_walkthrough.ipynb"
echo "To run the main pipeline: python main.py --config configs/simulation_default.yaml"
