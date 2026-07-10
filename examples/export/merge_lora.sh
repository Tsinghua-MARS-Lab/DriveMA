# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
export CUDA_HOME=/root/cuda
swift export \
    --adapters /cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/5w_vision_set/reproduce/v1-20260407-012238/checkpoint-782 \
    --merge_lora true
