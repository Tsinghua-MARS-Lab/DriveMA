# 8*80G GPU
# GSPO https://arxiv.org/pdf/2507.18071
# hyperparameter
# - epsilon = 3e-4 from paper serction 5.1
# - epsilon_high = 4e-4 from paper serction 5.1
# - paper uses steps_per_generation = 4; this example currently uses 1 for simpler rollout/update scheduling
# - paper uses beta = 0; this example currently uses beta = 0.001, so KL is added in the loss path
#   --multi_turn_scheduler fixed_dialogue --max_turns 2
#   --per_turn_grpo_credit true   


export CUDA_HOME=/root/cuda
export RFS_PREFERENCE_FILE='path/to/val_samples_434.jsonl'


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model stage-2-ckpt  \
    --template qwen3_5 \
    --enable_thinking false \
    --loss_scale ignore_empty_think \
    --add_non_thinking_prefix true \
    --dataset path/to/total_434_RFS_RL.jsonl \
    --save_strategy steps \
    --output_dir outputs \
    --torch_dtype bfloat16 \
    --beta 0.04 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 1 \
    --importance_sampling_level sequence \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --external_plugins path/to/examples/train/grpo/plugin/trajectory_reward.py \
    --multi_turn_scheduler fixed_dialogue \
    --max_turns 2 \
    --per_turn_grpo_credit true \
    --reward_funcs meta_action_acc rfs traj_consistency \
    --reward_weights 1.0 1.0 0.5 \
    --max_pixels 262144 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_max_model_len 2048 \
    --max_completion_length 128 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --save_steps 20 \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --save_total_limit 12 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --tuner_type full \
    --freeze_vit false \
    --freeze_aligner false \
    --vllm_mm_processor_cache_gb 0 \
    # --per_turn_loss_weight_turn1 0.2 \ under v2
    # --per_turn_loss_weight_turn2 0.8 \ under v2
    
