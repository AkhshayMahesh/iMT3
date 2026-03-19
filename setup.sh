#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mrmt3}"
ENV_PREFIX="${ENV_PREFIX:-"$(pwd)/.conda/envs/${ENV_NAME}"}"

# Keep everything self-contained and writable.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-"$(pwd)/.conda/pkgs"}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-"$(pwd)/.conda/envs"}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export CONDA_SOLVER="${CONDA_SOLVER:-classic}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda, then run:"
  echo "  conda env create -f environment.yml -p \"${ENV_PREFIX}\""
  echo "  conda activate \"${ENV_PREFIX}\""
  echo "  python -m pip install -U pip"
  echo "  python -m pip install -r requirements.txt"
  exit 1
fi

if conda env list | awk '{print $NF}' | grep -qx "${ENV_PREFIX}"; then
  echo "Conda env already exists at '${ENV_PREFIX}'."
else
  conda --no-plugins env create --solver classic -f environment.yml -p "${ENV_PREFIX}"
fi

echo "Activating '${ENV_NAME}' and installing Python deps..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
python -m pip install -U pip
python -m pip install -U "setuptools<81" wheel
python -m pip install -r requirements.txt -c constraints.txt || \
  python -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c constraints.txt

echo "Done. Next:"
echo "  conda activate \"${ENV_PREFIX}\""
