#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN=("$PYTHON_BIN" -m pip)

INSTALL_TORCH_SCATTER="${INSTALL_TORCH_SCATTER:-0}"
INSTALL_POINTNET2_BATCH="${INSTALL_POINTNET2_BATCH:-0}"
INSTALL_POINTOPS="${INSTALL_POINTOPS:-0}"
INSTALL_CHAMFER_DIST="${INSTALL_CHAMFER_DIST:-0}"
INSTALL_EMD="${INSTALL_EMD:-0}"
INSTALL_SUBSAMPLING="${INSTALL_SUBSAMPLING:-0}"

python_module_exists() {
  local module_name="$1"
  "$PYTHON_BIN" -c "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)" "$module_name"
}

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the repository root or keep the original project layout." >&2
  exit 1
fi

echo "Using Python: $("$PYTHON_BIN" -V 2>&1)"

mapfile -t MISSING_PURE_PYTHON_PACKAGES < <(
  "$PYTHON_BIN" - <<'PY'
import importlib.util

packages = {
    "Cython": "Cython",
    "easydict": "easydict",
    "fast-pytorch-kmeans": "fast_pytorch_kmeans",
    "gdown": "gdown",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "multimethod": "multimethod",
    "ninja": "ninja",
    "pandas": "pandas",
    "pickleshare": "pickleshare",
    "protobuf": "google.protobuf",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "shortuuid": "shortuuid",
    "tensorboard": "tensorboard",
    "termcolor": "termcolor",
    "tqdm": "tqdm",
}

for pip_name, module_name in packages.items():
    if importlib.util.find_spec(module_name) is None:
        print(pip_name)
PY
)

if [[ "${#MISSING_PURE_PYTHON_PACKAGES[@]}" -gt 0 ]]; then
  echo "Installing missing Python packages: ${MISSING_PURE_PYTHON_PACKAGES[*]}"
  "${PIP_BIN[@]}" install "${MISSING_PURE_PYTHON_PACKAGES[@]}"
else
  echo "Pure Python dependencies already available in the current runtime."
fi

if [[ "$INSTALL_TORCH_SCATTER" == "1" ]]; then
  if "$PYTHON_BIN" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch_scatter') else 1)"; then
    echo "torch-scatter is already installed."
  else
    readarray -t TORCH_INFO < <(
      "$PYTHON_BIN" - <<'PY'
import torch

torch_version = torch.__version__.split("+", 1)[0]
cuda_version = torch.version.cuda
cuda_flavor = f"cu{cuda_version.replace('.', '')}" if cuda_version else "cpu"
print(torch_version)
print(cuda_flavor)
PY
    )
    TORCH_VERSION="${TORCH_INFO[0]}"
    CUDA_FLAVOR="${TORCH_INFO[1]}"
    WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_FLAVOR}.html"
    echo "Installing torch-scatter for torch ${TORCH_VERSION} (${CUDA_FLAVOR})"
    "${PIP_BIN[@]}" install torch-scatter -f "$WHEEL_URL"
  fi
else
  echo "Skipping torch-scatter installation. Set INSTALL_TORCH_SCATTER=1 if your config needs it."
fi

install_local_extension() {
  local flag="$1"
  local rel_path="$2"
  local label="$3"
  local module_name="${4:-}"

  if [[ "$flag" != "1" ]]; then
    echo "Skipping ${label}. Set ${label}=1 to build it in the current Colab runtime."
    return 0
  fi

  if [[ -n "$module_name" ]] && python_module_exists "$module_name"; then
    echo "${label} is already available (${module_name})."
    return 0
  fi

  echo "Building local extension: ${rel_path}"
  "${PIP_BIN[@]}" install -v "./${rel_path}"
}

install_local_extension "$INSTALL_POINTNET2_BATCH" "openpoints/cpp/pointnet2_batch" "INSTALL_POINTNET2_BATCH" "pointnet2_batch_cuda"
install_local_extension "$INSTALL_POINTOPS" "openpoints/cpp/pointops" "INSTALL_POINTOPS" "pointops_cuda"
install_local_extension "$INSTALL_CHAMFER_DIST" "openpoints/cpp/chamfer_dist" "INSTALL_CHAMFER_DIST" "chamfer"
install_local_extension "$INSTALL_EMD" "openpoints/cpp/emd" "INSTALL_EMD" "emd_cuda"
install_local_extension "$INSTALL_SUBSAMPLING" "openpoints/cpp/subsampling" "INSTALL_SUBSAMPLING" "grid_subsampling"

echo
echo "Colab runtime setup is ready."
echo "If you plan to train PointNeXt or PointNet++ style models, rerun with INSTALL_POINTNET2_BATCH=1."
echo "No virtualenv was created; everything uses the active Colab runtime."
