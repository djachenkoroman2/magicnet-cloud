#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J seg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1

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

    # Compatibility fallback for stale paths like cfgs/k3d_cfgs/k3d_xyz/... .
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

is_google_colab_runtime() {
    [[ -n "${COLAB_GPU:-}" ]] && return 0
    [[ "$REPO_ROOT" == /content/* ]] && return 0
    [[ "$CALLER_PWD" == /content/* ]] && return 0
    return 1
}

infer_colab_install_flags() {
    local cfg_path=$1
    local install_torch_scatter=0
    local install_pointnet2_batch=0
    local install_pointops=0

    if grep -Eq 'NAME:[[:space:]]*PointNet2Encoder|NAME:[[:space:]]*PointNextEncoder' "$cfg_path"; then
        install_pointnet2_batch=1
    fi

    if grep -Eq 'NAME:[[:space:]]*PTSeg|block:[[:space:]]*PointTransformerBlock|Stratified' "$cfg_path"; then
        install_pointops=1
    fi

    if grep -Eq 'Stratified|PointNextPyG' "$cfg_path"; then
        install_torch_scatter=1
    fi

    echo "$install_torch_scatter $install_pointnet2_batch $install_pointops"
}

bootstrap_colab_requirements() {
    local cfg_path=$1
    local auto_torch_scatter
    local auto_pointnet2_batch
    local auto_pointops
    local install_torch_scatter
    local install_pointnet2_batch
    local install_pointops

    if [[ "${SKIP_COLAB_REQUIREMENTS:-0}" == "1" ]]; then
        echo "Skipping Colab dependency bootstrap because SKIP_COLAB_REQUIREMENTS=1"
        return 0
    fi

    if ! is_google_colab_runtime; then
        return 0
    fi

    if [[ ! -f "$SCRIPT_DIR/install_colab_requirements.sh" ]]; then
        echo "Colab dependency installer not found: $SCRIPT_DIR/install_colab_requirements.sh" >&2
        return 1
    fi

    read -r auto_torch_scatter auto_pointnet2_batch auto_pointops < <(infer_colab_install_flags "$cfg_path")

    install_torch_scatter="${INSTALL_TORCH_SCATTER:-$auto_torch_scatter}"
    install_pointnet2_batch="${INSTALL_POINTNET2_BATCH:-$auto_pointnet2_batch}"
    install_pointops="${INSTALL_POINTOPS:-$auto_pointops}"

    echo "Detected Google Colab runtime; preparing optional dependencies."
    echo "INSTALL_TORCH_SCATTER=$install_torch_scatter INSTALL_POINTNET2_BATCH=$install_pointnet2_batch INSTALL_POINTOPS=$install_pointops"

    env \
        "PYTHON_BIN=$PYTHON_BIN" \
        "INSTALL_TORCH_SCATTER=$install_torch_scatter" \
        "INSTALL_POINTNET2_BATCH=$install_pointnet2_batch" \
        "INSTALL_POINTOPS=$install_pointops" \
        "INSTALL_CHAMFER_DIST=${INSTALL_CHAMFER_DIST:-0}" \
        "INSTALL_EMD=${INSTALL_EMD:-0}" \
        "INSTALL_SUBSAMPLING=${INSTALL_SUBSAMPLING:-0}" \
        bash "$SCRIPT_DIR/install_colab_requirements.sh"
}

if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/main_segmentation.sh <config_path> [extra args...]" >&2
    exit 1
fi

[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

PYTHON_BIN=$(detect_python_bin)
echo "Using Python: $PYTHON_BIN"

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


cfg=$(resolve_cfg_path "$1")
PY_ARGS=("${@:2}")

bootstrap_colab_requirements "$cfg"

"$PYTHON_BIN" examples/segmentation/main.py --cfg "$cfg" "${PY_ARGS[@]}"


# how to run
# using slurm, run with 1 GPU, by 3 times (array=0-2):
# sbatch --array=0-2 --gres=gpu:1 --time=12:00:00 script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# if using local machine with GPUs, run with ALL GPUs:
# bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# local machine, run with 1 GPU:
# CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml
