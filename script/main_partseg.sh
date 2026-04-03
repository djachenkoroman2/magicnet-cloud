#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J partseg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
##SBATCH --gres=gpu:1
##SBATCH --constraint=[v100]
##SBATCH --mem=30G
##SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

detect_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    echo "python3"
}

[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "nvidia-smi failed, continuing"
else
    echo "nvidia-smi command not found, skipping GPU summary"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc command not found, skipping nvcc --version"
fi

hostname
NUM_GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo $NUM_GPU_AVAILABLE


PYTHON_BIN=$(detect_python_bin)
echo "Using Python: $PYTHON_BIN"

cfg=$1
PY_ARGS=${@:2}
"$PYTHON_BIN" examples/shapenetpart/main.py --cfg $cfg ${PY_ARGS}
