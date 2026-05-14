#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_ROOT}/docs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VERIFY_BUILD=0

usage() {
  cat <<'EOF'
Usage:
  ./enviroments.sh [options]

Options:
  --prefix PATH       Conda environment prefix. Default: ./docs
  --python VERSION    Python version. Default: 3.11
  --verify            Run mkdocs build --strict after installation
  -h, --help          Show this help

Examples:
  ./enviroments.sh
  ./enviroments.sh --verify
  ./enviroments.sh --prefix ./.conda-docs

After creation:
  conda activate ./docs
  make serve
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --verify)
      VERIFY_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH." >&2
  echo "Install Miniconda/Anaconda first, then run this script again." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ -d "${ENV_PREFIX}/conda-meta" ]]; then
  echo "Updating existing conda environment: ${ENV_PREFIX}"
  conda install -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
else
  echo "Creating conda environment: ${ENV_PREFIX}"
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
fi

echo "Installing Python packages from requirements.txt"
conda run -p "${ENV_PREFIX}" python -m pip install --upgrade pip
conda run -p "${ENV_PREFIX}" python -m pip install -r requirements.txt

echo
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_PREFIX}"
echo
echo "Then run:"
echo "  make serve"

if [[ "${VERIFY_BUILD}" -eq 1 ]]; then
  echo
  echo "Running verification build..."
  conda run -p "${ENV_PREFIX}" mkdocs build --strict
fi
