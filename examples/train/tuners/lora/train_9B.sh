export CUDA_HOME=/root/cuda

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model /cephfs/shared/zhengwc/LLM_ckpts/Qwen3.5-9B \
    --tuner_type lora \
    --dataset '/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/CoT-Ablation/Compare-CoT-For-Meta-Action/two-stage/No-CoT-to-Two-Stage-MetaAction.jsonl' \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --torch_dtype bfloat16 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --freeze_vit false \
    --freeze_aligner false \
    --max_pixels 262144 \
    --gradient_accumulation_steps 1 \
    --save_steps 100 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --deepspeed zero3 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author zwc \
    --model_name VLA \
    --output_dir /cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-4B/Meta-Action/wo-cot-8ep
