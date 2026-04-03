#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DATA_DIR="${DATA_DIR:-${1:-$REPO_ROOT/data}}"
TARGET_DIR="${DATA_DIR%/}/S3DIS"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

if [[ ! -f s3disfull.tar ]]; then
    if command -v gdown >/dev/null 2>&1; then
        gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y -O s3disfull.tar
    elif python -c "import gdown" >/dev/null 2>&1; then
        python -m gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y -O s3disfull.tar
    else
        echo "gdown is not available. Run bash script/install_colab_requirements.sh or install gdown, then rerun this script." >&2
        exit 1
    fi
fi

tar -xvf s3disfull.tar
