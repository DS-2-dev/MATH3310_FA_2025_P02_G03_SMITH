#!/usr/bin/env bash
set -euo pipefail

# Reproduce all experiments and saved artifacts from a clean clone.
# Usage: ./scripts/reproduce.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
"$PIP" install --upgrade pip

if [ -f requirements-locked.txt ]; then
  echo "Installing pinned requirements from requirements-locked.txt (recommended) ..."
  "$PIP" install -r requirements-locked.txt
else
  echo "Installing loose requirements from requirements.txt ..."
  "$PIP" install -r requirements.txt
fi

export PYTHONHASHSEED=42
export MPLBACKEND=Agg

echo "Running recompute script (this may take several minutes)..."
"$PYTHON" scripts/recompute_resampling_shap.py

echo "Generating plots..."
"$PYTHON" scripts/generate_plots.py
"$PYTHON" scripts/produce_additional_artifacts.py

echo "Done. Results are in the 'results/' folder. Open notebooks/mp2_artifacts.ipynb to view embedded artifacts or the notebook 'notebooks/mp2.ipynb' for analysis." 
