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

# Install ddsp without its dependencies — the 'crepe' dep uses an old setup.py
# that requires pkg_resources and fails to build on modern pip/setuptools.
# We only need ddsp.spectral_ops.compute_logmel; crepe is not required.
echo "Installing ddsp (no-deps) and patching crepe imports..."
python -m pip install ddsp --no-deps

# Patch ddsp/__init__.py: make 'losses' (which imports crepe) optional
DDSP_INIT="$(python -c "import ddsp; import os; print(os.path.join(os.path.dirname(ddsp.__file__), '__init__.py'))")"
"${ENV_PREFIX}/bin/python" - "${DDSP_INIT}" <<'PYEOF'
import re, sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from ddsp import losses"
new = ("try:\n"
       "    from ddsp import losses\n"
       "except ModuleNotFoundError:\n"
       "    pass  # 'crepe' optional dependency not installed; losses module unavailable")
if old in src:
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"Patched {path}")
else:
    print(f"Already patched or line not found in {path}")
PYEOF

# Patch ddsp/spectral_ops.py: make top-level 'import crepe' optional
DDSP_SPECTRAL="$(python -c "import ddsp; import os; print(os.path.join(os.path.dirname(ddsp.__file__), 'spectral_ops.py'))")"
"${ENV_PREFIX}/bin/python" - "${DDSP_SPECTRAL}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "import crepe\n"
new = ("try:\n"
       "    import crepe\n"
       "except ModuleNotFoundError:\n"
       "    crepe = None  # optional; only needed for compute_f0 and PretrainedCREPE\n")
if old in src and "except ModuleNotFoundError" not in src:
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print(f"Patched {path}")
else:
    print(f"Already patched or line not found in {path}")
PYEOF

echo "Done. Next:"
echo "  conda activate \"${ENV_PREFIX}\""
