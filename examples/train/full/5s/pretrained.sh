# 76GiB
export CUDA_HOME=/root/cuda

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model /cephfs/shared/zhengwc/LLM_ckpts/Qwen3.5-2B \
    --tuner_type full \
    --dataset ''\
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --freeze_vit false \
    --freeze_aligner false \
    --max_pixels 262144 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 50 \
    --save_total_limit 200 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir outputs \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author zwc \
    --loss_scale ignore_empty_think \
    --add_non_thinking_prefix true \
    --model_name qwen3.5-2B \
    --deepspeed zero3 \
