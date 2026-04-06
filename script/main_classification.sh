#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J cls
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
##SBATCH --gres=gpu:v100:1
##SBATCH --mem=30G

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CALLER_PWD=$PWD
cd "$REPO_ROOT"

resolve_cfg_path() {
    local input_path=$1
    local candidate
    local compat_candidate
    local task_dir
    local cfg_name
    local nested_candidate

    if [[ "$input_path" == /* ]]; then
        candidate=$input_path
    else
        candidate=$CALLER_PWD/$input_path
    fi
    if [[ -f "$candidate" ]]; then
        readlink -f "$candidate"
        return 0
    fi

    if [[ "$input_path" != /* && -f "$REPO_ROOT/$input_path" ]]; then
        readlink -f "$REPO_ROOT/$input_path"
        return 0
    fi

    if [[ "$input_path" == cfgs/* ]]; then
        task_dir=${input_path#cfgs/}
        task_dir=${task_dir%%/*}
        cfg_name=$(basename "$input_path")
        if [[ -d "$REPO_ROOT/cfgs/$task_dir" ]]; then
            nested_candidate=$(find "$REPO_ROOT/cfgs/$task_dir" -type f -name "$cfg_name" | head -n 1)
            if [[ -n "$nested_candidate" ]]; then
                echo "Resolved nested config fallback: $input_path -> ${nested_candidate#$REPO_ROOT/}" >&2
                readlink -f "$nested_candidate"
                return 0
            fi
        fi
    fi

    if [[ "$input_path" == cfgs/*/* ]]; then
        compat_candidate="$REPO_ROOT/cfgs/${input_path#cfgs/*/}"
        if [[ -f "$compat_candidate" ]]; then
            echo "Resolved config path fallback: $input_path -> ${compat_candidate#$REPO_ROOT/}" >&2
            readlink -f "$compat_candidate"
            return 0
        fi
    fi

    echo "Config not found: $input_path" >&2
    return 1
}

resolve_user_path() {
    local input_path=$1
    local candidate

    if [[ "$input_path" == /* ]]; then
        candidate=$input_path
    else
        candidate=$CALLER_PWD/$input_path
    fi

    if readlink -f "$candidate" >/dev/null 2>&1; then
        readlink -f "$candidate"
    else
        printf '%s\n' "$candidate"
    fi
}

config_chain_declares_nested_key() {
    local cfg_path=$1
    local key_name=$2
    local extension=${cfg_path##*.}
    local default_path
    local dir

    extension=.$extension
    if grep -Eq "^[[:space:]]+${key_name}:" "$cfg_path"; then
        return 0
    fi

    dir=$(dirname "$cfg_path")
    while true; do
        default_path="$dir/default$extension"
        if [[ -f "$default_path" ]] && grep -Eq "^[[:space:]]+${key_name}:" "$default_path"; then
            return 0
        fi

        if [[ "$dir" == "/" ]]; then
            break
        fi

        dir=$(dirname "$dir")
    done

    return 1
}

config_chain_declares_top_level_key() {
    local cfg_path=$1
    local key_name=$2
    local extension=${cfg_path##*.}
    local default_path
    local dir

    extension=.$extension
    if grep -Eq "^${key_name}:" "$cfg_path"; then
        return 0
    fi

    dir=$(dirname "$cfg_path")
    while true; do
        default_path="$dir/default$extension"
        if [[ -f "$default_path" ]] && grep -Eq "^${key_name}:" "$default_path"; then
            return 0
        fi

        if [[ "$dir" == "/" ]]; then
            break
        fi

        dir=$(dirname "$dir")
    done

    return 1
}

detect_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi

    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        echo "$REPO_ROOT/.venv/bin/python"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    echo "python"
}

maybe_load_modules() {
    if [[ "${USE_ENV_MODULES:-0}" != "1" ]]; then
        return 0
    fi

    if command -v module >/dev/null 2>&1; then
        module load cuda/11.1.1
        module load gcc
        echo "Loaded environment modules (cuda/11.1.1, gcc)"
    else
        echo "USE_ENV_MODULES=1 was requested, but \`module\` is unavailable; continuing with current environment"
    fi
}

maybe_load_modules

if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/main_classification.sh <config_path> [--data <dataset_path>] [--log <log_root>] [--resume <checkpoint_path>] [extra args...]" >&2
    exit 1
fi

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
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_QUERY=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_QUERY" ]]; then
        NUM_GPU_AVAILABLE=$(printf '%s\n' "$GPU_QUERY" | wc -l)
    else
        NUM_GPU_AVAILABLE=0
    fi
else
    NUM_GPU_AVAILABLE=0
fi
echo $NUM_GPU_AVAILABLE


PYTHON_BIN=$(detect_python_bin)
echo "Using Python: $PYTHON_BIN"

cfg=$(resolve_cfg_path "$1")
shift

PY_ARGS=()
DATA_OVERRIDE=
LOG_OVERRIDE=
RESUME_OVERRIDE=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data|--data-root|--data-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value after $1" >&2
                exit 1
            fi
            DATA_OVERRIDE=$(resolve_user_path "$2")
            shift 2
            ;;
        --log|--log-root|--root-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value after $1" >&2
                exit 1
            fi
            LOG_OVERRIDE=$(resolve_user_path "$2")
            shift 2
            ;;
        --resume)
            if [[ $# -lt 2 ]]; then
                echo "Missing value after $1" >&2
                exit 1
            fi
            RESUME_OVERRIDE=$(resolve_user_path "$2")
            shift 2
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$DATA_OVERRIDE" ]]; then
    if config_chain_declares_top_level_key "$cfg" "data_dir"; then
        PY_ARGS+=("data_dir=$DATA_OVERRIDE")
    elif config_chain_declares_nested_key "$cfg" "data_dir"; then
        PY_ARGS+=("dataset.common.data_dir=$DATA_OVERRIDE")
    elif config_chain_declares_nested_key "$cfg" "data_root"; then
        PY_ARGS+=("dataset.common.data_root=$DATA_OVERRIDE")
    else
        PY_ARGS+=("data_dir=$DATA_OVERRIDE")
    fi
fi

if [[ -n "$LOG_OVERRIDE" ]]; then
    if config_chain_declares_top_level_key "$cfg" "log_root"; then
        PY_ARGS+=("log_root=$LOG_OVERRIDE")
    else
        PY_ARGS+=("root_dir=$LOG_OVERRIDE")
    fi
fi

if [[ -n "$RESUME_OVERRIDE" ]]; then
    PY_ARGS+=("mode=resume" "pretrained_path=$RESUME_OVERRIDE")
fi

"$PYTHON_BIN" examples/classification/main.py --cfg "$cfg" "${PY_ARGS[@]}"

# how to run
# this script supports training using 1 GPU or multi-gpu,
# simply run jobs on multiple GPUs will launch distributed training by default.
# load different cfgs for training on different benchmarks (modelnet40 classification, or scanobjectnn classification) and using different models.

# For examples,
# if using a cluster with slurm, train PointNeXt-S on scanobjectnn classification using only 1 GPU, by 3 times:
# sbatch --array=0-2 --gres=gpu:1 --time=10:00:00 main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml

# if using local machine with GPUs, train PointNeXt-S on scanobjectnn classification using all GPUs
# bash script/main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml

# if using local machine with GPUs, train PointNeXt-S on scanobjectnn classification using only 1 GPU
# CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml

# force CPU execution
# bash script/main_classification.sh cfgs/birds/pointnet.yaml runtime.device=cpu

# resume from checkpoint
# bash script/main_classification.sh cfgs/birds/pointnet.yaml --resume /abs/path/to/run/checkpoint/<run_name>_ckpt_latest.pth

# override dataset and log locations
# bash script/main_classification.sh cfgs/scanobjectnn/pointnext-s.yaml --data /abs/path/to/ScanObjectNN/h5_files/main_split --log /abs/path/to/logs
# bash script/main_classification.sh cfgs/birds/pointnet.yaml --data /abs/path/to/data --log /abs/path/to/logs
