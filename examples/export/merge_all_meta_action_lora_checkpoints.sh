#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-9B/CoT-Ablation/CoT-to-Traj-ALL/v0-20260429-170821}"


if [[ ! -d "$ROOT" ]]; then
    echo "[merge_all_lora] checkpoint root not found: $ROOT" >&2
    exit 1
fi



export CUDA_HOME="${CUDA_HOME:-/root/cuda}"

LOG_DIR="$ROOT/merge_logs"
mkdir -p "$LOG_DIR"

mapfile -t CHECKPOINTS < <(
    find "$ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
        | awk '/^checkpoint-[0-9]+$/' \
        | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "[merge_all_lora] no checkpoint-* directories found under: $ROOT"
    exit 0
fi

echo "[merge_all_lora] root: $ROOT"
echo "[merge_all_lora] found ${#CHECKPOINTS[@]} checkpoint(s)"

for name in "${CHECKPOINTS[@]}"; do
    ckpt="$ROOT/$name"
    merged="$ROOT/${name}-merged"
    log_file="$LOG_DIR/${name}.log"

    if [[ ! -f "$ckpt/adapter_config.json" ]]; then
        echo "[merge_all_lora] skip $name: adapter_config.json not found"
        continue
    fi
    if [[ ! -f "$ckpt/adapter_model.safetensors" && ! -f "$ckpt/adapter_model.bin" ]]; then
        echo "[merge_all_lora] skip $name: adapter_model.safetensors/bin not found"
        continue
    fi
    if [[ -d "$merged" && -f "$merged/config.json" ]]; then
        echo "[merge_all_lora] skip $name: merged model already exists: $merged"
        continue
    fi

    echo "[merge_all_lora] merging $ckpt -> $merged"
    swift export \
        --adapters "$ckpt" \
        --output_dir "$merged" \
        --merge_lora true \
        2>&1 | tee "$log_file"

    if [[ ! -f "$merged/config.json" ]]; then
        echo "[merge_all_lora] merge finished but config.json not found: $merged" >&2
        exit 1
    fi

    echo "[merge_all_lora] removing original checkpoint: $ckpt"
    rm -rf "$ckpt"
done

echo "[merge_all_lora] done"
