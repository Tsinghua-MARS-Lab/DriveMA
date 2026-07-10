# 76GiB
export CUDA_HOME=/root/cuda

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model /cephfs/shared/zhengwc/LLM_ckpts/Qwen3.5-2B \
    --tuner_type full \
    --dataset '/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/sft_data/action-conditioned-traj'\
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --freeze_vit false \
    --freeze_aligner false \
    --max_pixels 262144 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 5120 \
    --output_dir outputs \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author zwc \
    --loss_scale ignore_empty_think \
    --add_non_thinking_prefix true \
    --model_name qwen3.5-2B \
    --deepspeed zero3 \
    # --resume_from_checkpoint /cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn/First-Turn-warmup-w-VQA/v2-20260411-223408/checkpoint-3400
    # --val_dataset '/cephfs/zhengwc/FluidDrive/ms-swift/data/val_samples_479_only_meta_action_wo_cot.jsonl' \
    # --eval_strategy steps \
    # --eval_steps 100 \
    # --predict_with_generate true \
    # --eval_metric meta_action \
    # --metric_for_best_model joint_acc \
    # --greater_is_better true \
    # --load_best_model_at_end true \
